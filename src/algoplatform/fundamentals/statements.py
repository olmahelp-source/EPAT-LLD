"""Typed financial statements with point-in-time availability.

The classic backtest bug is using a number on the day a period ENDED instead of the day it was ANNOUNCED. Every
``Statement`` therefore carries both dates, and ``FundamentalsStore.asof`` only returns what was public on the date
asked for, including the right *version* when a period was later restated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

from ..core.exceptions import ValidationError

INCOME = ("revenue", "cogs", "sga", "depreciation", "operating_income", "interest_expense", "pretax_income",
          "tax_expense", "net_income")
BALANCE = ("cash", "receivables", "inventory", "payables", "current_assets", "ppe", "total_assets", "current_liabilities",
           "long_term_debt", "short_term_debt", "total_liabilities", "equity", "retained_earnings", "shares")
CASHFLOW = ("cfo", "capex", "dividends")
FIELDS = INCOME + BALANCE + CASHFLOW


@dataclass(frozen=True)
class Statement:
    symbol: str
    period_end: date
    announced: date                     # first date the market could see these numbers
    data: Dict[str, float] = field(default_factory=dict)
    currency: str = "USD"
    version: int = 1                    # restatements of the same period get higher versions

    def __post_init__(self) -> None:
        if self.announced < self.period_end:
            raise ValidationError("a filing cannot be announced before its period ends",
                                  details={"symbol": self.symbol, "period_end": str(self.period_end),
                                           "announced": str(self.announced)})
        bad = set(self.data) - set(FIELDS)
        if bad:
            raise ValidationError(f"unknown statement fields {sorted(bad)}", details={"valid": FIELDS})

    def __getitem__(self, key: str) -> float:
        try:
            return float(self.data[key])
        except KeyError:
            raise ValidationError(f"{self.symbol} {self.period_end}: field '{key}' not reported") from None

    def get(self, key: str, default: Optional[float] = None) -> Optional[float]:
        v = self.data.get(key)
        return default if v is None else float(v)

    # ---- derived lines ---------------------------------------------------------------------------------
    @property
    def gross_profit(self) -> float:
        return self["revenue"] - self["cogs"]

    @property
    def ebitda(self) -> float:
        return self["operating_income"] + (self.get("depreciation") or 0.0)

    @property
    def total_debt(self) -> float:
        return (self.get("long_term_debt") or 0.0) + (self.get("short_term_debt") or 0.0)

    @property
    def free_cash_flow(self) -> float:
        return self["cfo"] - (self.get("capex") or 0.0)

    @property
    def working_capital(self) -> float:
        return self["current_assets"] - self["current_liabilities"]

    @property
    def eps(self) -> float:
        return self["net_income"] / self["shares"]

    def balance_gap(self) -> Optional[float]:
        """assets - (liabilities + equity) as a fraction of assets; ~0 for a consistent balance sheet."""
        a, l, e = self.get("total_assets"), self.get("total_liabilities"), self.get("equity")
        return None if None in (a, l, e) or not a else (a - l - e) / a


class FundamentalsStore:
    def __init__(self) -> None:
        self._by_symbol: Dict[str, List[Statement]] = {}

    def add(self, st: Statement) -> None:
        self._by_symbol.setdefault(st.symbol, []).append(st)

    def symbols(self) -> List[str]:
        return sorted(self._by_symbol)

    def known(self, symbol: str, on: date, lag_days: int = 0) -> List[Statement]:
        """Latest known version of every period that was public on ``on`` (after an extra safety lag), oldest first."""
        cutoff = on - timedelta(days=lag_days)
        best: Dict[date, Statement] = {}
        for st in self._by_symbol.get(symbol, []):
            if st.announced <= cutoff and (st.period_end not in best or (st.announced, st.version) > (best[st.period_end].announced, best[st.period_end].version)):
                best[st.period_end] = st
        return [best[k] for k in sorted(best)]

    def asof(self, symbol: str, on: date, lag_days: int = 0) -> Optional[Statement]:
        k = self.known(symbol, on, lag_days)
        return k[-1] if k else None

    def previous(self, symbol: str, current: Statement, on: date, lag_days: int = 0, min_gap_days: int = 300) -> Optional[Statement]:
        """The comparable earlier-year statement, as known on ``on`` (for year-over-year ratios)."""
        for st in reversed(self.known(symbol, on, lag_days)):
            if (current.period_end - st.period_end).days >= min_gap_days:
                return st
        return None
