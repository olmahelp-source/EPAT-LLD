"""FixGateway: a BrokerGateway speaking FIX 4.4 order flow (NewOrderSingle / OrderCancelRequest / ExecutionReport).

Scope: a working subset suitable for an order-entry session with a broker or exchange simulator. It is NOT a
certified implementation; venue-specific tags, repeating groups, order replace and market-data messages are out
of scope. Positions and cash are tracked from fills (FIX has no standard account query).
"""
from __future__ import annotations

import socket
import threading
import time
from typing import Any, Callable, Dict, Optional

from ...core.enums import OrderType, Side, TimeInForce
from ...core.exceptions import BrokerConnectionError, ConfigError, OrderRejectedError
from ...core.logging import get_logger
from ...core.models import Fill, Order
from ..base import BROKERS, BrokerEvent, BrokerGateway
from .message import (CANCEL_REJECT, CANCEL_REQUEST, EXEC_REPORT, FixMessage, FixProtocolError, NEW_ORDER_SINGLE,
                      T_CLORDID, T_EXECID, T_EXECTYPE, T_LASTPX, T_LASTQTY, T_ORDERID, T_ORDTYPE, T_ORIGCLORDID, T_PRICE,
                      T_QTY, T_SIDE, T_STOPPX, T_SYMBOL, T_TEXT, T_TIF, T_TRANSACT, utc_timestamp)
from .session import FixSession, MemorySeqStore

log = get_logger(__name__)
_SIDE = {Side.BUY: "1", Side.SELL: "2"}
_TYPE = {OrderType.MARKET: "1", OrderType.LIMIT: "2", OrderType.STOP: "3", OrderType.STOP_LIMIT: "4"}
_TIF = {TimeInForce.DAY: "0", TimeInForce.GTC: "1", TimeInForce.IOC: "3", TimeInForce.FOK: "4"}


def _num(x: float) -> str:
    return f"{x:.8f}".rstrip("0").rstrip(".") or "0"


class TcpChannel:
    """Blocking TCP client channel with a reader thread feeding the session."""

    def __init__(self, host: str, port: int, on_data: Callable[[bytes], None], timeout: float = 5.0) -> None:
        self.host, self.port, self.on_data, self.timeout = host, port, on_data, timeout
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._closed = False

    def open(self) -> None:
        try:
            self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        except OSError as exc:
            raise BrokerConnectionError(f"cannot connect to FIX endpoint: {type(exc).__name__}") from None
        self._sock.settimeout(0.5)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        while not self._closed and self._sock is not None:
            try:
                data = self._sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            try:
                self.on_data(data)
            except Exception:  # noqa: BLE001 - a bad message must not kill the reader
                log.exception("error handling FIX data")

    def write(self, data: bytes) -> None:
        if self._sock is None:
            raise BrokerConnectionError("FIX channel not open")
        try:
            self._sock.sendall(data)
        except OSError as exc:
            raise BrokerConnectionError(f"FIX send failed: {type(exc).__name__}") from None

    def close(self) -> None:
        self._closed = True
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass


@BROKERS.register("fix")
class FixGateway(BrokerGateway):
    name = "fix"

    def __init__(self, credentials: Optional[Dict[str, str]] = None, paper: bool = True, sender: Optional[str] = None,
                 target: Optional[str] = None, channel: Any = None, store: Optional[MemorySeqStore] = None,
                 heartbeat: int = 30, initial_cash: float = 0.0, logon_timeout: float = 5.0, reset_on_connect: bool = False,
                 clock: Callable[[], float] = time.time) -> None:
        super().__init__()
        c = credentials or {}
        sender, target = sender or c.get("sender_comp_id"), target or c.get("target_comp_id")
        if not sender or not target:
            raise ConfigError("FIX needs sender_comp_id and target_comp_id")
        self._creds = c
        self.paper = paper
        self.session = FixSession(sender, target, channel, store, heartbeat, clock, on_app=self._on_app)
        self._channel = channel
        self.logon_timeout, self.reset_on_connect = logon_timeout, reset_on_connect
        self._tracked: Dict[str, Dict[str, Any]] = {}
        self._cancel_map: Dict[str, str] = {}
        self._pos: Dict[str, float] = {}
        self._cash = float(initial_cash)
        self._clock = clock

    # ---- lifecycle -----------------------------------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self.session.is_active

    def _require_connected(self) -> None:
        if not self.session.is_active:
            raise BrokerConnectionError("fix session not active")

    def connect(self) -> None:
        if self._channel is None:
            host, port = self._creds.get("host"), self._creds.get("port")
            if not host or not port:
                raise ConfigError("FIX needs a channel or host/port credentials")
            self._channel = TcpChannel(host, int(port), self.session.on_bytes)
            self._channel.open()
            self.session.attach(self._channel)
        self.session.logon(self.reset_on_connect)
        deadline = time.time() + self.logon_timeout
        while not self.session.is_active and time.time() < deadline:
            time.sleep(0.005)
        if not self.session.is_active:
            self.session.disconnect()
            raise BrokerConnectionError("FIX logon was not acknowledged")

    def disconnect(self) -> None:
        self.session.logout("client disconnect")

    # ---- order entry ---------------------------------------------------------------------------------
    def submit_order(self, order: Order) -> str:
        self._require_connected()
        if order.client_order_id in self._tracked:
            raise OrderRejectedError("duplicate client order id", code="DUPLICATE_ID")
        if order.order_type not in _TYPE:
            raise OrderRejectedError(f"order type {order.order_type.value} not supported over FIX by this gateway")
        body = [(T_CLORDID, order.client_order_id), (T_SYMBOL, order.symbol), (T_SIDE, _SIDE[order.side]),
                (T_TRANSACT, utc_timestamp(self._clock())), (T_QTY, _num(order.quantity)),
                (T_ORDTYPE, _TYPE[order.order_type]), (T_TIF, _TIF[order.tif])]
        if order.limit_price is not None:
            body.append((T_PRICE, _num(order.limit_price)))
        if order.stop_price is not None:
            body.append((T_STOPPX, _num(order.stop_price)))
        self._tracked[order.client_order_id] = {"order": order, "open": True, "cum": 0.0}   # before sending: replies are fast
        try:
            self.session.send_app(NEW_ORDER_SINGLE, body)
        except FixProtocolError as exc:
            self._tracked.pop(order.client_order_id, None)
            raise BrokerConnectionError(f"FIX send failed: {exc.message}") from None
        return order.client_order_id                                  # broker order id arrives in the first ExecutionReport

    def cancel_order(self, order: Order) -> None:
        self._require_connected()
        t = self._tracked.get(order.client_order_id)
        if t is None or not t["open"]:
            raise OrderRejectedError("order not open at broker", code="UNKNOWN_ORDER")
        cancel_id = f"{order.client_order_id}-C"
        self._cancel_map[cancel_id] = order.client_order_id
        self.session.send_app(CANCEL_REQUEST, [(T_ORIGCLORDID, order.client_order_id), (T_CLORDID, cancel_id),
                                               (T_SYMBOL, order.symbol), (T_SIDE, _SIDE[order.side]),
                                               (T_TRANSACT, utc_timestamp(self._clock())), (T_QTY, _num(order.quantity))])

    def get_positions(self) -> Dict[str, float]:
        return {s: q for s, q in self._pos.items() if abs(q) > 1e-12}

    def get_cash(self) -> float:
        return self._cash

    def open_order_ids(self) -> set:
        return {cid for cid, t in self._tracked.items() if t["open"]}

    # ---- inbound -------------------------------------------------------------------------------------
    def _on_app(self, m: FixMessage) -> None:
        if m.type == EXEC_REPORT:
            self._exec_report(m)
        elif m.type == CANCEL_REJECT:
            cid = m.get(T_ORIGCLORDID, "")
            if cid in self._tracked:
                self._emit(BrokerEvent("cancel_rejected", cid, m.get(T_ORDERID, ""), reason=m.get(T_TEXT, "cancel rejected")))
        else:
            log.debug("ignoring FIX message type %s", m.type)

    def _exec_report(self, m: FixMessage) -> None:
        clid = m.get(T_CLORDID, "")
        cid = self._cancel_map.get(clid) or m.get(T_ORIGCLORDID) or clid
        t = self._tracked.get(cid)
        if t is None:
            log.warning("ExecutionReport for unknown order %s", cid)
            return
        et, bid = m.get(T_EXECTYPE, ""), m.get(T_ORDERID, "")
        if et == "0":
            self._emit(BrokerEvent("ack", cid, bid))
        elif et in ("F", "1", "2"):                                   # trade (4.4 uses F; older dialects 1/2)
            qty, px = float(m.req(T_LASTQTY)), float(m.req(T_LASTPX))
            o: Order = t["order"]
            self._pos[o.symbol] = self._pos.get(o.symbol, 0.0) + o.side.sign * qty
            self._cash -= o.side.sign * qty * px
            t["cum"] += qty
            fill = Fill(m.get(T_EXECID) or f"{bid}:{t['cum']}", cid, o.symbol, o.side, qty, px, None, 0.0, o.strategy_id,
                        o.account_id, bid)
            self._emit(BrokerEvent("fill", cid, bid, fill=fill))
            if t["cum"] >= o.quantity - 1e-9:
                t["open"] = False
        elif et == "4":
            t["open"] = False
            self._emit(BrokerEvent("cancelled", cid, bid))
        elif et == "8":
            t["open"] = False
            self._emit(BrokerEvent("rejected", cid, bid, reason=m.get(T_TEXT, "rejected")))
        elif et == "C":
            t["open"] = False
            self._emit(BrokerEvent("expired", cid, bid, reason=m.get(T_TEXT, "expired")))
