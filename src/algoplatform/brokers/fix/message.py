"""FIX 4.4 tag=value wire format: encoding, decoding, checksum and body-length validation, stream parsing."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from ...core.exceptions import BrokerError

SOH = "\x01"
BEGIN_STRING = "FIX.4.4"

# message types
HEARTBEAT, TEST_REQUEST, RESEND_REQUEST, REJECT, SEQUENCE_RESET, LOGOUT, LOGON = "0", "1", "2", "3", "4", "5", "A"
NEW_ORDER_SINGLE, CANCEL_REQUEST, EXEC_REPORT, CANCEL_REJECT = "D", "F", "8", "9"
ADMIN_TYPES = {HEARTBEAT, TEST_REQUEST, RESEND_REQUEST, REJECT, SEQUENCE_RESET, LOGOUT, LOGON}

# tags
T_BEGIN, T_BODYLEN, T_MSGTYPE, T_SENDER, T_TARGET, T_SEQ, T_TIME, T_CHECKSUM = 8, 9, 35, 49, 56, 34, 52, 10
T_POSSDUP, T_ORIGTIME = 43, 122
T_CLORDID, T_ORIGCLORDID, T_ORDERID, T_EXECID = 11, 41, 37, 17
T_SYMBOL, T_SIDE, T_QTY, T_ORDTYPE, T_PRICE, T_STOPPX, T_TIF, T_TRANSACT = 55, 54, 38, 40, 44, 99, 59, 60
T_EXECTYPE, T_ORDSTATUS, T_CUMQTY, T_AVGPX, T_LASTPX, T_LASTQTY, T_LEAVES, T_TEXT = 150, 39, 14, 6, 31, 32, 151, 58
T_ENCRYPT, T_HEARTBT, T_TESTREQID, T_BEGINSEQ, T_ENDSEQ, T_NEWSEQ, T_GAPFILL, T_RESET = 98, 108, 112, 7, 16, 36, 123, 141


class FixProtocolError(BrokerError):
    """Malformed or inconsistent FIX data (bad checksum, wrong body length, ...). Not retryable."""


def checksum(raw: bytes) -> str:
    return f"{sum(raw) % 256:03d}"


def utc_timestamp(now: Optional[float] = None) -> str:
    dt = datetime.fromtimestamp(now, timezone.utc) if now is not None else datetime.now(timezone.utc)
    return dt.strftime("%Y%m%d-%H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


class FixMessage:
    def __init__(self, fields: List[Tuple[int, str]]) -> None:
        self.fields = fields
        self._first: Dict[int, str] = {}
        for t, v in fields:
            self._first.setdefault(t, v)

    def get(self, tag: int, default: Optional[str] = None) -> Optional[str]:
        return self._first.get(tag, default)

    def req(self, tag: int) -> str:
        v = self._first.get(tag)
        if v is None:
            raise FixProtocolError(f"required tag {tag} missing", details={"msg_type": self.type})
        return v

    @property
    def type(self) -> str:
        return self._first.get(T_MSGTYPE, "")

    @property
    def seq(self) -> int:
        try:
            return int(self.req(T_SEQ))
        except ValueError:
            raise FixProtocolError("MsgSeqNum not an integer") from None

    def __repr__(self) -> str:
        return "|".join(f"{t}={v}" for t, v in self.fields)


def encode(fields: Iterable[Tuple[int, str]]) -> bytes:
    """Wrap header-and-body fields (tag 35 first; no 8/9/10) with BeginString, BodyLength and CheckSum."""
    body = "".join(f"{t}={v}{SOH}" for t, v in fields)
    if SOH in "".join(str(v) for _, v in fields):
        raise FixProtocolError("field value contains SOH")
    head = f"{T_BEGIN}={BEGIN_STRING}{SOH}{T_BODYLEN}={len(body)}{SOH}"
    raw = (head + body).encode("ascii")
    return raw + f"{T_CHECKSUM}={checksum(raw)}{SOH}".encode("ascii")


def build(msg_type: str, sender: str, target: str, seq: int, sending_time: str, body: Iterable[Tuple[int, str]],
          extra_header: Iterable[Tuple[int, str]] = ()) -> bytes:
    header = [(T_MSGTYPE, msg_type), (T_SENDER, sender), (T_TARGET, target), (T_SEQ, str(seq)), (T_TIME, sending_time),
              *extra_header]
    return encode([*header, *body])


def decode(raw: bytes) -> FixMessage:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        raise FixProtocolError("non-ASCII FIX message") from None
    if not text.startswith(f"{T_BEGIN}={BEGIN_STRING}{SOH}9="):
        raise FixProtocolError("bad BeginString/BodyLength header")
    if len(raw) < 7 or raw[-7:-4] != b"10=" or raw[-1:] != SOH.encode():
        raise FixProtocolError("missing CheckSum trailer")
    if raw[-4:-1].decode() != checksum(raw[:-7]):
        raise FixProtocolError("checksum mismatch")
    parts = text[:-1].split(SOH)
    fields: List[Tuple[int, str]] = []
    for p in parts:
        tag, sep, val = p.partition("=")
        if not sep or not tag.isdigit():
            raise FixProtocolError(f"malformed field '{p[:20]}'")
        fields.append((int(tag), val))
    declared = int(fields[1][1])
    body_start = len(f"{T_BEGIN}={BEGIN_STRING}{SOH}{T_BODYLEN}={fields[1][1]}{SOH}")
    if declared != len(text) - 7 - body_start:
        raise FixProtocolError("body length mismatch")
    return FixMessage([f for f in fields if f[0] not in (T_BEGIN, T_BODYLEN, T_CHECKSUM)])


class FixParser:
    """Splits a TCP byte stream into complete messages (handles partial reads and coalesced messages)."""

    def __init__(self, max_buffer: int = 1_000_000) -> None:
        self._buf = b""
        self._max = max_buffer

    def feed(self, data: bytes) -> List[FixMessage]:
        self._buf += data
        if len(self._buf) > self._max:
            self._buf = b""
            raise FixProtocolError("FIX receive buffer overflow")
        out: List[FixMessage] = []
        while True:
            start = self._buf.find(b"8=FIX")
            if start < 0:
                self._buf = self._buf[-4:] if len(self._buf) > 4 else self._buf
                break
            self._buf = self._buf[start:]
            m = self._buf.find(SOH.encode() + b"9=")
            if m < 0:
                break
            end9 = self._buf.find(SOH.encode(), m + 1)
            if end9 < 0:
                break
            try:
                n = int(self._buf[m + 3:end9])
            except ValueError:
                self._buf = b""
                raise FixProtocolError("bad BodyLength value") from None
            total = end9 + 1 + n + 7
            if len(self._buf) < total:
                break
            chunk, self._buf = self._buf[:total], self._buf[total:]
            out.append(decode(chunk))
        return out
