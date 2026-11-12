"""Contract specifications: everything static the platform needs to know about a tradable thing.

``core.models.Instrument`` is the lean record the engine uses on the hot path (tick, lot, multiplier). ``ContractSpec``
is the full reference record kept in the :class:`~algoplatform.instruments.master.InstrumentMaster`: identity, venue,
classification, derivative terms and each broker's own name for the contract.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

from ..core.exceptions import InstrumentError
from ..core.models import Instrument


class AssetClass(str, Enum):
    EQUITY = "equity"
    ETF = "etf"
    INDEX = "index"
    FUTURE = "future"
    OPTION = "option"
    FX = "fx"
    CRYPTO = "crypto"
    BOND = "bond"

    @property
    def is_derivative(self) -> bool:
        return self in (AssetClass.FUTURE, AssetClass.OPTION)


def _as_date(value: Any) -> Optional[date]:
    if value is None or value == "" or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        raise InstrumentError(f"bad date '{value}' (expected YYYY-MM-DD)") from None


@dataclass(frozen=True)
class ContractSpec:
    symbol: str                                   # the platform's own, broker-independent symbol
    asset_class: AssetClass = AssetClass.EQUITY
    name: str = ""
    exchange: str = ""
    currency: str = "USD"
    tick_size: float = 0.01
    lot_size: float = 1.0
    multiplier: float = 1.0
    isin: str = ""
    sector: str = ""
    industry: str = ""
    underlying: Optional[str] = None              # derivatives only
    expiry: Optional[date] = None
    strike: Optional[float] = None
    right: Optional[str] = None                   # "call" | "put"
    shortable: bool = True
    active: bool = True
    margin_pct: float = 1.0                       # initial margin as a fraction of notional (1.0 = fully funded)
    broker_symbols: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)   # ((broker, broker's symbol), ...)

    def __post_init__(self) -> None:
        if not self.symbol or self.symbol != self.symbol.strip():
            raise InstrumentError("symbol required (no surrounding spaces)")
        if not isinstance(self.asset_class, AssetClass):
            try:
                object.__setattr__(self, "asset_class", AssetClass(str(self.asset_class).lower()))
            except ValueError:
                raise InstrumentError(f"{self.symbol}: unknown asset class '{self.asset_class}'",
                                      details={"valid": [a.value for a in AssetClass]}) from None
        if min(self.tick_size, self.lot_size, self.multiplier) <= 0:
            raise InstrumentError(f"{self.symbol}: tick_size, lot_size and multiplier must be positive")
        if not 0 < self.margin_pct <= 1:
            raise InstrumentError(f"{self.symbol}: margin_pct must be in (0, 1]")
        object.__setattr__(self, "expiry", _as_date(self.expiry))
        if self.asset_class.is_derivative and (not self.underlying or self.expiry is None):
            raise InstrumentError(f"{self.symbol}: a {self.asset_class.value} needs an underlying and an expiry")
        if self.asset_class is AssetClass.OPTION:
            if self.right not in ("call", "put") or not self.strike or self.strike <= 0:
                raise InstrumentError(f"{self.symbol}: an option needs right 'call'/'put' and a positive strike")
        if isinstance(self.broker_symbols, Mapping):
            object.__setattr__(self, "broker_symbols", tuple(sorted((str(k).lower(), str(v)) for k, v in self.broker_symbols.items())))

    # ---- views ------------------------------------------------------------------------------------------------------
    def to_instrument(self) -> Instrument:
        """The lean record used by the engine, sizing and the portfolio ledger."""
        return Instrument(self.symbol, self.tick_size, self.lot_size, self.multiplier, self.currency, self.asset_class.value, self.shortable)

    def broker_symbol(self, broker: str) -> str:
        """What ``broker`` calls this contract (the platform symbol when no mapping is recorded)."""
        return dict(self.broker_symbols).get(broker.lower(), self.symbol)

    def days_to_expiry(self, asof: Any) -> Optional[int]:
        return None if self.expiry is None else (self.expiry - _as_date(asof)).days    # type: ignore[operator]

    def is_expired(self, asof: Any) -> bool:
        d = self.days_to_expiry(asof)
        return d is not None and d < 0

    def notional(self, price: float, quantity: float = 1.0) -> float:
        return abs(quantity) * price * self.multiplier

    def to_dict(self) -> Dict[str, Any]:
        return {"symbol": self.symbol, "asset_class": self.asset_class.value, "name": self.name, "exchange": self.exchange,
                "currency": self.currency, "tick_size": self.tick_size, "lot_size": self.lot_size, "multiplier": self.multiplier,
                "isin": self.isin, "sector": self.sector, "industry": self.industry, "underlying": self.underlying,
                "expiry": self.expiry.isoformat() if self.expiry else None, "strike": self.strike, "right": self.right,
                "shortable": self.shortable, "active": self.active, "margin_pct": self.margin_pct,
                "broker_symbols": dict(self.broker_symbols)}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "ContractSpec":
        """Build from a loosely typed row (CSV / JSON): blank cells mean "use the default"; ``broker:<name>`` columns
        become broker symbol mappings."""
        def blank(v: Any) -> bool:
            return v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, float) and v != v)
        kwargs: Dict[str, Any] = {}
        brokers: Dict[str, str] = dict(row.get("broker_symbols") or {}) if isinstance(row.get("broker_symbols"), Mapping) else {}
        fields = cls.__dataclass_fields__
        for key, value in row.items():
            if isinstance(key, str) and key.startswith("broker:"):
                if not blank(value):
                    brokers[key.split(":", 1)[1].lower()] = str(value)
            elif key in fields and key != "broker_symbols" and not blank(value):
                kwargs[key] = value
        try:
            for k in ("tick_size", "lot_size", "multiplier", "strike", "margin_pct"):
                if k in kwargs:
                    kwargs[k] = float(kwargs[k])
        except (TypeError, ValueError):
            raise InstrumentError(f"{row.get('symbol', '?')}: '{k}' must be a number, got {kwargs[k]!r}") from None
        for k in ("shortable", "active"):
            if k in kwargs and not isinstance(kwargs[k], bool):
                kwargs[k] = str(kwargs[k]).strip().lower() in ("1", "true", "yes", "y")
        if "right" in kwargs:
            kwargs["right"] = {"c": "call", "p": "put", "ce": "call", "pe": "put"}.get(str(kwargs["right"]).lower(), str(kwargs["right"]).lower())
        if "symbol" not in kwargs:
            raise InstrumentError("row has no symbol")
        return cls(broker_symbols=tuple(sorted(brokers.items())), **{k: (str(v).strip() if isinstance(v, str) else v) for k, v in kwargs.items()})


_OCC = re.compile(r"^(?P<und>[A-Z][A-Z0-9.]*?)(?P<ymd>\d{6})(?P<right>[CP])(?P<strike>\d{8})$")
_MONTH_CODES = "FGHJKMNQUVXZ"                    # futures month codes, January .. December


def parse_option_symbol(symbol: str) -> Dict[str, Any]:
    """Inverse of ``derivatives.option_symbol``: ``AAPL240119C00150000`` -> underlying, expiry, right, strike."""
    m = _OCC.match(symbol)
    if not m:
        raise InstrumentError(f"'{symbol}' is not an OCC-style option symbol")
    try:
        expiry = datetime.strptime(m["ymd"], "%y%m%d").date()
    except ValueError:
        raise InstrumentError(f"'{symbol}' has an invalid expiry date") from None
    return {"underlying": m["und"], "expiry": expiry, "right": "call" if m["right"] == "C" else "put", "strike": int(m["strike"]) / 1000.0}


def option_spec(underlying: str, expiry: Any, right: str, strike: float, multiplier: float = 100.0, tick_size: float = 0.01,
                **extra: Any) -> ContractSpec:
    exp = _as_date(expiry)
    if exp is None:
        raise InstrumentError("an option needs an expiry")
    symbol = f"{underlying}{exp.strftime('%y%m%d')}{right[0].upper()}{int(round(strike * 1000)):08d}"
    return ContractSpec(symbol, AssetClass.OPTION, underlying=underlying, expiry=exp, strike=float(strike), right=right,
                        multiplier=multiplier, tick_size=tick_size, **extra)


def future_symbol(root: str, expiry: Any) -> str:
    """``ES`` + 2025-03-21 -> ``ESH25`` (root, month code, two-digit year)."""
    exp = _as_date(expiry)
    if exp is None:
        raise InstrumentError("a future needs an expiry")
    return f"{root}{_MONTH_CODES[exp.month - 1]}{exp.year % 100:02d}"


def future_spec(root: str, expiry: Any, multiplier: float, tick_size: float, margin_pct: float = 0.1, **extra: Any) -> ContractSpec:
    return ContractSpec(future_symbol(root, expiry), AssetClass.FUTURE, underlying=root, expiry=_as_date(expiry), multiplier=multiplier,
                        tick_size=tick_size, margin_pct=margin_pct, **extra)
