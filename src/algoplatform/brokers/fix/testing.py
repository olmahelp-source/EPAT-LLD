"""In-memory FIX acceptor (a tiny exchange simulator) and loopback channels for tests and demos."""
from __future__ import annotations

import itertools
import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .message import (CANCEL_REJECT, CANCEL_REQUEST, EXEC_REPORT, FixMessage, NEW_ORDER_SINGLE, T_AVGPX, T_CLORDID, T_CUMQTY,
                      T_EXECID, T_EXECTYPE, T_LASTPX, T_LASTQTY, T_LEAVES, T_ORDERID, T_ORDSTATUS, T_ORIGCLORDID, T_QTY,
                      T_SIDE, T_SYMBOL, T_TEXT)
from .session import FixSession, MemorySeqStore


class LoopbackChannel:
    """Delivers written bytes straight to the peer session. ``drop`` swallows the next N writes to simulate loss."""

    def __init__(self) -> None:
        self.peer: Optional[FixSession] = None
        self.drop = 0
        self.closed = False
        self.written: List[bytes] = []

    def write(self, data: bytes) -> None:
        self.written.append(data)
        if self.closed:
            return
        if self.drop > 0:
            self.drop -= 1
            return
        if self.peer is not None:
            self.peer.on_bytes(data)

    def close(self) -> None:
        self.closed = True


class FixAcceptor:
    """Accepts logons and orders; the test drives fills explicitly."""

    def __init__(self, sender: str = "EXCH", target: str = "CLIENT", store: Optional[MemorySeqStore] = None,
                 heartbeat: int = 30, clock: Callable[[], float] = time.time) -> None:
        self.session = FixSession(sender, target, None, store, heartbeat, clock, on_app=self._on_app)
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.reject_all: Optional[str] = None
        self.channel = LoopbackChannel()
        self._oid, self._xid = itertools.count(1), itertools.count(1)

    def link(self, client_session: FixSession, client_channel: LoopbackChannel) -> None:
        client_channel.peer = self.session
        self.channel.peer = client_session
        self.session.attach(self.channel)
        client_session.attach(client_channel)

    # ---- application messages ------------------------------------------------------------------------
    def _on_app(self, m: FixMessage) -> None:
        if m.type == NEW_ORDER_SINGLE:
            cid = m.req(T_CLORDID)
            o = {"cid": cid, "oid": f"EX{next(self._oid):06d}", "symbol": m.req(T_SYMBOL), "side": m.req(T_SIDE),
                 "qty": float(m.req(T_QTY)), "cum": 0.0, "notional": 0.0, "open": True}
            self.orders[cid] = o
            if self.reject_all:
                o["open"] = False
                self._er(o, "8", "8", text=self.reject_all)
            else:
                self._er(o, "0", "0")
        elif m.type == CANCEL_REQUEST:
            orig = m.req(T_ORIGCLORDID)
            o = self.orders.get(orig)
            if o is None or not o["open"]:
                self.session.send_app(CANCEL_REJECT, [(T_ORDERID, o["oid"] if o else "NONE"), (T_CLORDID, m.req(T_CLORDID)),
                                                      (T_ORIGCLORDID, orig), (T_ORDSTATUS, "2" if o else "8"),
                                                      (434, "1"), (T_TEXT, "too late to cancel")])
            else:
                o["open"] = False
                self._er(o, "4", "4", clid=m.req(T_CLORDID), orig=orig)

    def _er(self, o: Dict[str, Any], exec_type: str, status: str, last_qty: float = 0.0, last_px: float = 0.0,
            text: Optional[str] = None, clid: Optional[str] = None, orig: Optional[str] = None) -> None:
        avg = o["notional"] / o["cum"] if o["cum"] else 0.0
        body = [(T_ORDERID, o["oid"]), (T_EXECID, f"X{next(self._xid):08d}"), (T_CLORDID, clid or o["cid"]),
                (T_SYMBOL, o["symbol"]), (T_SIDE, o["side"]), (T_EXECTYPE, exec_type), (T_ORDSTATUS, status),
                (T_CUMQTY, f"{o['cum']:g}"), (T_LEAVES, f"{max(o['qty'] - o['cum'], 0):g}"), (T_AVGPX, f"{avg:.6f}")]
        if orig:
            body.append((T_ORIGCLORDID, orig))
        if exec_type == "F":
            body += [(T_LASTQTY, f"{last_qty:g}"), (T_LASTPX, f"{last_px:.6f}")]
        if text:
            body.append((T_TEXT, text))
        self.session.send_app(EXEC_REPORT, body)

    # ---- test drivers --------------------------------------------------------------------------------
    def fill(self, cid: str, qty: Optional[float] = None, px: float = 100.0) -> None:
        o = self.orders[cid]
        q = o["qty"] - o["cum"] if qty is None else qty
        o["cum"] += q
        o["notional"] += q * px
        done = o["cum"] >= o["qty"] - 1e-9
        if done:
            o["open"] = False
        self._er(o, "F", "2" if done else "1", q, px)

    def expire(self, cid: str) -> None:
        o = self.orders[cid]
        o["open"] = False
        self._er(o, "C", "C", text="day order expired")


def make_loopback(client_session: FixSession, acceptor: Optional[FixAcceptor] = None) -> FixAcceptor:
    acceptor = acceptor or FixAcceptor(sender=client_session.target, target=client_session.sender)
    acceptor.link(client_session, LoopbackChannel())
    return acceptor


class TcpAcceptorServer:
    """Real-socket acceptor on 127.0.0.1 for exercising TcpChannel. One connection at a time."""

    def __init__(self, sender: str = "EXCH", target: str = "CLIENT") -> None:
        self.sender, self.target = sender, target
        self._srv = socket.socket()
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(1)
        self._srv.settimeout(0.2)
        self.port = self._srv.getsockname()[1]
        self.acceptor: Optional[FixAcceptor] = None
        self._stop = False
        self._conn: Optional[socket.socket] = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "TcpAcceptorServer":
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop:
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn.settimeout(0.2)
            self._conn = conn
            acc = FixAcceptor(self.sender, self.target)

            class _W:
                def write(_s, data: bytes) -> None:
                    conn.sendall(data)

                def close(_s) -> None:
                    conn.close()
            acc.session.attach(_W())
            self.acceptor = acc
            while not self._stop:
                try:
                    data = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                acc.session.on_bytes(data)

    def stop(self) -> None:
        self._stop = True
        for s in (self._conn, self._srv):
            try:
                if s:
                    s.close()
            except OSError:
                pass
        self._thread.join(timeout=2)
