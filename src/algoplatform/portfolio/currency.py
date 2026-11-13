"""Multi-currency accounting: exchange rates, cash per currency, and what a mixed book is worth in one currency.

The position ledger (``portfolio.Portfolio``) keeps each position in its instrument's own currency. This module adds
what is needed around it when instruments trade in several currencies:

* :class:`FxRates`           point-in-time exchange rates with inverse and cross rates through a pivot currency;
* :class:`CashLedger`        cash balances per currency, conversions between them, and trade settlement;
* :func:`exposure_by_currency` / :func:`portfolio_currency_exposure`   the book valued in a base currency, split by
                             currency, so foreign-exchange risk is visible;
* :func:`fx_pnl_attribution` how much of a result came from the asset and how much from the currency;
* :func:`hedge_notionals`    the forward amounts that would hedge a chosen share of each exposure.

Rate convention: the pair ``"EURUSD"`` at 1.10 means one EUR costs 1.10 USD (first currency is the one being priced).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from ..core.exceptions import MissingDataError, ValidationError


def _ccy(code: Any) -> str:
    c = str(code).strip().upper()
    if len(c) != 3 or not c.isalpha():
        raise ValidationError(f"'{code}' is not a three-letter currency code")
    return c


def _pair(pair: Any) -> Tuple[str, str]:
    if isinstance(pair, (tuple, list)) and len(pair) == 2:
        return _ccy(pair[0]), _ccy(pair[1])
    text = str(pair).replace("/", "").replace("-", "").strip()
    if len(text) != 6:
        raise ValidationError(f"'{pair}' is not a currency pair (expected e.g. 'EURUSD')")
    base, quote = _ccy(text[:3]), _ccy(text[3:])
    if base == quote:
        raise ValidationError("a currency pair needs two different currencies")
    return base, quote


class FxRates:
    """Exchange rates by pair. A rate set with a timestamp is used only from that time on (point-in-time lookups for
    backtests); a rate set without one is a standing rate used when no dated rate applies."""

    def __init__(self, pivot: str = "USD") -> None:
        self.pivot = _ccy(pivot)
        self._static: Dict[Tuple[str, str], float] = {}
        self._series: Dict[Tuple[str, str], pd.Series] = {}

    def set(self, pair: Any, rate: float, ts: Any = None) -> "FxRates":
        key = _pair(pair)
        if not (isinstance(rate, (int, float)) and rate == rate and 0 < rate < float("inf")):
            raise ValidationError(f"rate for {key[0]}{key[1]} must be a positive number", details={"rate": rate})
        if ts is None:
            self._static[key] = float(rate)
        else:
            s = self._series.get(key, pd.Series(dtype=float))
            s.loc[pd.Timestamp(ts)] = float(rate)
            self._series[key] = s.sort_index()
        return self

    def add_series(self, pair: Any, series: pd.Series) -> "FxRates":
        key = _pair(pair)
        s = pd.Series(series, dtype=float).dropna()
        if s.empty or (s <= 0).any() or not isinstance(s.index, pd.DatetimeIndex):
            raise ValidationError("a rate series needs positive values and a datetime index")
        self._series[key] = s.sort_index()
        return self

    def pairs(self) -> List[str]:
        return sorted({a + b for a, b in set(self._static) | set(self._series)})

    def _direct(self, a: str, b: str, asof: Optional[pd.Timestamp]) -> Optional[float]:
        for key, invert in (((a, b), False), ((b, a), True)):
            rate: Optional[float] = None
            series = self._series.get(key)
            if series is not None and len(series):
                if asof is None:
                    rate = float(series.iloc[-1])
                else:
                    known = series.loc[:asof]
                    rate = float(known.iloc[-1]) if len(known) else None      # nothing published yet at that time
            if rate is None:
                rate = self._static.get(key)
            if rate is not None:
                return 1.0 / rate if invert else rate
        return None

    def rate(self, from_ccy: str, to_ccy: str, asof: Any = None) -> float:
        """Units of ``to_ccy`` per one unit of ``from_ccy``, as known at ``asof`` (latest if None). Tries the pair, its
        inverse, then a cross through the pivot currency."""
        a, b = _ccy(from_ccy), _ccy(to_ccy)
        if a == b:
            return 1.0
        ts = None if asof is None else pd.Timestamp(asof)
        direct = self._direct(a, b, ts)
        if direct is not None:
            return direct
        if self.pivot not in (a, b):
            first, second = self._direct(a, self.pivot, ts), self._direct(self.pivot, b, ts)
            if first is not None and second is not None:
                return first * second
        raise MissingDataError(f"no exchange rate from {a} to {b}" + (f" as of {ts.date()}" if ts is not None else ""),
                               details={"known_pairs": self.pairs()})

    def convert(self, amount: float, from_ccy: str, to_ccy: str, asof: Any = None) -> float:
        return float(amount) * self.rate(from_ccy, to_ccy, asof)


@dataclass(frozen=True)
class CashEntry:
    kind: str                   # deposit | withdraw | exchange | settle
    currency: str
    amount: float               # signed change of that currency's balance
    note: str = ""


class CashLedger:
    """Cash held in each currency. Every change is recorded, and a balance cannot go negative unless borrowing in that
    currency is explicitly allowed (``allow_negative``), so an unfunded foreign purchase is an error, not a silent loan."""

    def __init__(self, balances: Optional[Mapping[str, float]] = None, allow_negative: bool = False) -> None:
        self.allow_negative = allow_negative
        self._balances: Dict[str, float] = {}
        self.entries: List[CashEntry] = []
        self._lock = threading.RLock()
        for ccy, amount in (balances or {}).items():
            self.deposit(ccy, amount)

    def balance(self, currency: str) -> float:
        return self._balances.get(_ccy(currency), 0.0)

    def balances(self) -> Dict[str, float]:
        return {c: v for c, v in sorted(self._balances.items()) if abs(v) > 1e-12}

    def _apply(self, kind: str, currency: str, amount: float, note: str = "") -> float:
        c = _ccy(currency)
        with self._lock:
            new = self._balances.get(c, 0.0) + amount
            if new < -1e-9 and not self.allow_negative:
                raise ValidationError(f"insufficient {c}: balance {self._balances.get(c, 0.0):,.2f}, needed {-amount:,.2f}",
                                      code="INSUFFICIENT_CASH", details={"currency": c})
            self._balances[c] = new
            self.entries.append(CashEntry(kind, c, amount, note))
            return new

    def deposit(self, currency: str, amount: float) -> float:
        if not amount > 0:
            raise ValidationError("deposit amount must be positive")
        return self._apply("deposit", currency, float(amount))

    def withdraw(self, currency: str, amount: float) -> float:
        if not amount > 0:
            raise ValidationError("withdrawal amount must be positive")
        return self._apply("withdraw", currency, -float(amount))

    def settle(self, currency: str, signed_cash: float, note: str = "") -> float:
        """Apply a trade's cash flow in its own currency (negative for a purchase or a fee)."""
        return self._apply("settle", currency, float(signed_cash), note)

    def exchange(self, from_ccy: str, to_ccy: str, amount: float, rate: float, fee_bps: float = 0.0) -> float:
        """Sell ``amount`` of ``from_ccy`` at ``rate`` (units of ``to_ccy`` per unit sold); the fee is taken from the
        proceeds. Returns the amount of ``to_ccy`` received. Both legs are applied together or not at all."""
        a, b = _ccy(from_ccy), _ccy(to_ccy)
        if a == b or not amount > 0 or not rate > 0 or fee_bps < 0:
            raise ValidationError("exchange needs two different currencies, a positive amount and rate, and fee_bps >= 0")
        received = float(amount) * float(rate) * (1.0 - fee_bps / 1e4)
        with self._lock:
            self._apply("exchange", a, -float(amount), f"to {b} at {rate:g}")
            self._apply("exchange", b, received, f"from {a} at {rate:g}")
        return received

    def total(self, base: str, fx: FxRates, asof: Any = None) -> float:
        return sum(fx.convert(v, c, base, asof) for c, v in self._balances.items())


def exposure_by_currency(values: Mapping[str, float], base: str, fx: FxRates, asof: Any = None) -> Dict[str, Any]:
    """Value a set of local-currency amounts (positions plus cash, netted per currency) in ``base``.

    ``fx_exposure`` is the share of gross value held in currencies other than the base: the part of the book whose
    base-currency value moves when exchange rates do, even if no asset price changes."""
    b = _ccy(base)
    rows: Dict[str, Dict[str, float]] = {}
    for ccy, local in sorted(values.items()):
        c = _ccy(ccy)
        rate = fx.rate(c, b, asof)
        rows[c] = {"local": float(local), "rate": rate, "base": float(local) * rate}
    total = sum(r["base"] for r in rows.values())
    gross = sum(abs(r["base"]) for r in rows.values())
    for r in rows.values():
        r["share"] = abs(r["base"]) / gross if gross > 0 else 0.0
    foreign = sum(abs(r["base"]) for c, r in rows.items() if c != b)
    return {"base": b, "total": total, "gross": gross, "by_currency": rows, "fx_exposure": foreign / gross if gross > 0 else 0.0}


def portfolio_currency_exposure(portfolio: Any, fx: FxRates, base: str, cash: Optional[Mapping[str, float]] = None,
                                asof: Any = None) -> Dict[str, Any]:
    """Currency breakdown of a ``Portfolio`` ledger: open positions grouped by their instrument's currency (instruments
    without a record are assumed to be in ``base``), plus cash. ``cash`` gives balances per currency (for example
    ``CashLedger.balances()``); without it the ledger's single cash figure is taken to be in ``base``."""
    b = _ccy(base)
    values: Dict[str, float] = {}
    for symbol, position in portfolio.open_positions().items():
        inst = portfolio.instruments.get(symbol)
        ccy = _ccy(inst.currency) if inst is not None else b
        values[ccy] = values.get(ccy, 0.0) + position.market_value
    for ccy, amount in (cash if cash is not None else {b: portfolio.cash}).items():
        values[_ccy(ccy)] = values.get(_ccy(ccy), 0.0) + float(amount)
    return exposure_by_currency(values, b, fx, asof)


def fx_pnl_attribution(value_start: float, value_end: float, rate_start: float, rate_end: float) -> Dict[str, float]:
    """Split a foreign holding's base-currency result into its sources. Values are in the local currency; rates are
    base per local.

    ``asset``: the local price move at the starting rate. ``currency``: the starting value carried by the rate move.
    ``interaction``: the price move times the rate move. The three add up to ``total`` exactly."""
    if rate_start <= 0 or rate_end <= 0:
        raise ValidationError("rates must be positive")
    asset = (value_end - value_start) * rate_start
    currency = value_start * (rate_end - rate_start)
    interaction = (value_end - value_start) * (rate_end - rate_start)
    return {"asset": asset, "currency": currency, "interaction": interaction, "total": value_end * rate_end - value_start * rate_start}


def hedge_notionals(exposure: Mapping[str, Any], hedge_ratio: float = 1.0) -> Dict[str, float]:
    """Amount of each foreign currency to sell forward (negative: buy) to hedge ``hedge_ratio`` of the exposure reported
    by :func:`exposure_by_currency`. The base currency needs no hedge."""
    if not 0 <= hedge_ratio <= 1:
        raise ValidationError("hedge_ratio must be in [0, 1]")
    base = exposure["base"]
    return {ccy: row["local"] * hedge_ratio for ccy, row in exposure["by_currency"].items() if ccy != base and row["local"] != 0}
