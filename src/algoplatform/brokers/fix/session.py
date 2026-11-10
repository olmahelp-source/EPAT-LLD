"""FIX session layer: logon, sequence numbers, heartbeats, gap detection, resend and gap fill."""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Callable, Dict, List, Optional, Protocol, Tuple

from ...core.logging import get_logger
from .message import (ADMIN_TYPES, FixMessage, FixParser, FixProtocolError, HEARTBEAT, LOGON, LOGOUT, RESEND_REQUEST, REJECT,
                      SEQUENCE_RESET, TEST_REQUEST, T_BEGINSEQ, T_ENCRYPT, T_ENDSEQ, T_GAPFILL, T_HEARTBT, T_NEWSEQ,
                      T_ORIGTIME, T_POSSDUP, T_RESET, T_TESTREQID, T_TEXT, build, utc_timestamp)

log = get_logger(__name__)
DISCONNECTED, LOGON_SENT, ACTIVE = "DISCONNECTED", "LOGON_SENT", "ACTIVE"


class Channel(Protocol):
    def write(self, data: bytes) -> None: ...
    def close(self) -> None: ...


class MemorySeqStore:
    """Sequence numbers plus the sent-message log needed to answer ResendRequests."""

    def __init__(self) -> None:
        self.next_send = 1
        self.next_expected = 1
        self.sent: Dict[int, Tuple[str, List[Tuple[int, str]], str]] = {}

    def save(self) -> None:
        pass

    def reset(self) -> None:
        self.next_send, self.next_expected, self.sent = 1, 1, {}
        self.save()


class FileSeqStore(MemorySeqStore):
    """Same, persisted as JSON (atomic replace) so a restart resumes the session instead of resetting it."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            self.next_send, self.next_expected = d["next_send"], d["next_expected"]
            self.sent = {int(k): (v[0], [tuple(x) for x in v[1]], v[2]) for k, v in d["sent"].items()}

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"next_send": self.next_send, "next_expected": self.next_expected,
                       "sent": {str(k): [v[0], v[1], v[2]] for k, v in self.sent.items()}}, fh)
        os.replace(tmp, self.path)


class FixSession:
    def __init__(self, sender: str, target: str, channel: Optional[Channel] = None, store: Optional[MemorySeqStore] = None,
                 heartbeat: int = 30, clock: Callable[[], float] = time.time,
                 on_app: Optional[Callable[[FixMessage], None]] = None,
                 on_state: Optional[Callable[[str], None]] = None) -> None:
        self.sender, self.target = sender, target
        self.channel = channel
        self.store = store or MemorySeqStore()
        self.heartbeat = heartbeat
        self._clock = clock
        self.on_app, self.on_state = on_app, on_state
        self.state = DISCONNECTED
        self._parser = FixParser()
        self._lock = threading.RLock()
        self._ooo: Dict[int, FixMessage] = {}              # out-of-order messages waiting for a gap to fill
        self._resend_requested = False
        self._last_sent = self._last_recv = clock()
        self._test_pending: Optional[str] = None
        self.protocol_errors = 0

    # ---- state ------------------------------------------------------------------------------------
    def _set_state(self, s: str) -> None:
        if s != self.state:
            self.state = s
            if self.on_state:
                self.on_state(s)

    @property
    def is_active(self) -> bool:
        return self.state == ACTIVE

    def attach(self, channel: Channel) -> None:
        self.channel = channel

    def disconnect(self) -> None:
        with self._lock:
            self._set_state(DISCONNECTED)
            self._ooo.clear()
            self._resend_requested = False
            if self.channel is not None:
                try:
                    self.channel.close()
                except OSError:
                    pass

    # ---- sending ------------------------------------------------------------------------------------
    def _write(self, raw: bytes) -> None:
        if self.channel is None:
            raise FixProtocolError("no channel attached")
        self._last_sent = self._clock()
        self.channel.write(raw)

    def _send(self, msg_type: str, body: List[Tuple[int, str]], seq: Optional[int] = None, poss_dup: bool = False,
              orig_time: Optional[str] = None, store: bool = True) -> int:
        with self._lock:
            now = utc_timestamp(self._clock())
            if seq is None:
                seq = self.store.next_send
                self.store.next_send += 1
            extra = [(T_POSSDUP, "Y"), (T_ORIGTIME, orig_time or now)] if poss_dup else []
            if store:
                self.store.sent[seq] = (msg_type, list(body), now)
                self.store.save()
            self._write(build(msg_type, self.sender, self.target, seq, now, body, extra))
            return seq

    def send_app(self, msg_type: str, body: List[Tuple[int, str]]) -> int:
        if not self.is_active:
            raise FixProtocolError("session not active")
        return self._send(msg_type, body)

    def logon(self, reset: bool = False) -> None:
        with self._lock:
            if reset:
                self.store.reset()
            body = [(T_ENCRYPT, "0"), (T_HEARTBT, str(self.heartbeat))] + ([(T_RESET, "Y")] if reset else [])
            self._set_state(LOGON_SENT)
            self._send(LOGON, body)

    def logout(self, text: str = "") -> None:
        with self._lock:
            if self.state != DISCONNECTED:
                self._send(LOGOUT, [(T_TEXT, text)] if text else [])
            self.disconnect()

    # ---- receiving ----------------------------------------------------------------------------------
    def on_bytes(self, data: bytes) -> None:
        try:
            msgs = self._parser.feed(data)
        except FixProtocolError as exc:
            self.protocol_errors += 1
            log.error("dropping bad FIX data: %s", exc.message)
            return
        for m in msgs:
            with self._lock:
                self._last_recv = self._clock()
                self._test_pending = None
                try:
                    self._handle(m)
                except FixProtocolError as exc:
                    self.protocol_errors += 1
                    log.error("bad FIX message %s: %s", m.type, exc.message)

    def _handle(self, m: FixMessage) -> None:
        seq = m.seq
        poss = m.get(T_POSSDUP) == "Y"
        if m.type == LOGON and m.get(T_RESET) == "Y":
            self.store.next_expected = 1
            if self.state != LOGON_SENT:                                        # we are the passive side: mirror the reset
                self.store.next_send, self.store.sent = 1, {}
        if m.type == SEQUENCE_RESET and m.get(T_GAPFILL) != "Y":
            self.store.next_expected = int(m.req(T_NEWSEQ))                    # hard reset: trust the counterparty
            self.store.save()
            self._drain()
            return
        exp = self.store.next_expected
        if seq == exp:
            self._consume(m)
            self.store.save()
            self._drain()
        elif seq > exp:
            self._ooo[seq] = m
            if not self._resend_requested:
                self._resend_requested = True
                self._send(RESEND_REQUEST, [(T_BEGINSEQ, str(exp)), (T_ENDSEQ, "0")])
        elif poss:
            return                                                              # duplicate we already processed
        else:
            self.logout(f"MsgSeqNum too low, expecting {exp} but received {seq}")

    def _drain(self) -> None:
        while self.store.next_expected in self._ooo:
            self._consume(self._ooo.pop(self.store.next_expected))
        self.store.save()
        if not self._ooo:
            self._resend_requested = False

    def _consume(self, m: FixMessage) -> None:
        """Process one in-sequence message, then advance the expected number (a gap fill jumps ahead)."""
        self._process(m)
        nxt = m.seq + 1
        if m.type == SEQUENCE_RESET:
            nxt = max(nxt, int(m.req(T_NEWSEQ)))
        self.store.next_expected = max(self.store.next_expected, nxt)

    def _process(self, m: FixMessage) -> None:
        t = m.type
        if t == LOGON:
            if self.state == LOGON_SENT:                                        # initiator: this is the reply
                self._set_state(ACTIVE)
            else:                                                               # acceptor: answer the logon
                self.heartbeat = int(m.get(T_HEARTBT, str(self.heartbeat)))
                self._set_state(ACTIVE)
                reply = [(T_ENCRYPT, "0"), (T_HEARTBT, str(self.heartbeat))]
                self._send(LOGON, reply + ([(T_RESET, "Y")] if m.get(T_RESET) == "Y" else []))
        elif t == HEARTBEAT:
            pass
        elif t == TEST_REQUEST:
            self._send(HEARTBEAT, [(T_TESTREQID, m.get(T_TESTREQID, ""))])
        elif t == RESEND_REQUEST:
            self._resend(int(m.req(T_BEGINSEQ)), int(m.req(T_ENDSEQ)))
        elif t == SEQUENCE_RESET:
            pass                                                                # gap fill: handled in _consume
        elif t == REJECT:
            log.warning("session-level reject from counterparty: %s", m.get(T_TEXT))
        elif t == LOGOUT:
            if self.state == ACTIVE:
                self._send(LOGOUT, [])
            self.disconnect()
        elif self.on_app:
            self.on_app(m)

    def _resend(self, begin: int, end: int) -> None:
        last = self.store.next_send - 1
        if end == 0 or end > last:
            end = last
        gap_start: Optional[int] = None
        for seq in range(begin, end + 1):
            rec = self.store.sent.get(seq)
            if rec is None or rec[0] in ADMIN_TYPES:
                gap_start = seq if gap_start is None else gap_start
                continue
            if gap_start is not None:                                           # skip admin messages with a gap fill
                self._send(SEQUENCE_RESET, [(T_GAPFILL, "Y"), (T_NEWSEQ, str(seq))], seq=gap_start, poss_dup=True, store=False)
                gap_start = None
            self._send(rec[0], rec[1], seq=seq, poss_dup=True, orig_time=rec[2], store=False)
        if gap_start is not None:
            self._send(SEQUENCE_RESET, [(T_GAPFILL, "Y"), (T_NEWSEQ, str(end + 1))], seq=gap_start, poss_dup=True, store=False)

    # ---- timers -------------------------------------------------------------------------------------
    def tick(self, now: Optional[float] = None) -> None:
        """Call periodically (e.g. once a second): sends heartbeats, probes silence, drops dead links."""
        now = self._clock() if now is None else now
        with self._lock:
            if self.state != ACTIVE:
                return
            if now - self._last_sent >= self.heartbeat:
                self._send(HEARTBEAT, [])
            silent = now - self._last_recv
            if silent > 2.5 * self.heartbeat:
                log.error("no data for %.0fs: dropping FIX session", silent)
                self.disconnect()
            elif silent > 1.5 * self.heartbeat and self._test_pending is None:
                self._test_pending = f"TEST{int(now)}"
                self._send(TEST_REQUEST, [(T_TESTREQID, self._test_pending)])
