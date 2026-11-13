"""Corporate actions: adjust price history so that a split or dividend is not mistaken for a crash.

Back-adjustment keeps the most recent prices as they traded and rescales everything *before* each ex-date. That is the
right series for indicators and returns; it is the wrong one for anything that needs the price actually seen on a past
date (order simulation against old limit prices, point-in-time screens on price level), which should use raw prices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List

import numpy as np
import pandas as pd

from ...foundation.core.exceptions import InstrumentError


@dataclass(frozen=True)
class CorporateAction:
    symbol: str
    ex_date: Any                 # first day the stock trades without the entitlement
    kind: str                    # "split" | "dividend"
    value: float                 # split: new shares per old share (2.0 = 2-for-1, 0.1 = 1-for-10 reverse); dividend: cash per share

    def __post_init__(self) -> None:
        if self.kind not in ("split", "dividend"):
            raise InstrumentError(f"unknown corporate action '{self.kind}'")
        if not np.isfinite(self.value) or self.value <= 0:
            raise InstrumentError(f"{self.symbol}: corporate action value must be positive")
        object.__setattr__(self, "ex_date", pd.Timestamp(self.ex_date))


def adjustment_factors(df: pd.DataFrame, actions: Iterable[CorporateAction]) -> pd.DataFrame:
    """Per-bar multipliers ``price`` and ``volume`` that convert raw history to back-adjusted history."""
    price = pd.Series(1.0, index=df.index)
    volume = pd.Series(1.0, index=df.index)
    for a in sorted(actions, key=lambda a: a.ex_date):
        before = df.index < a.ex_date
        if not before.any() or before.all():
            continue                                               # outside the data: nothing to adjust
        if a.kind == "split":
            price[before] /= a.value
            volume[before] *= a.value
        else:
            prev_close = float(df["close"][before].iloc[-1])       # last cum-dividend close
            if a.value >= prev_close:
                raise InstrumentError(f"{a.symbol}: dividend {a.value} is not below the previous close {prev_close}")
            price[before] *= 1.0 - a.value / prev_close
    return pd.DataFrame({"price": price, "volume": volume})


def adjust_ohlcv(df: pd.DataFrame, actions: Iterable[CorporateAction]) -> pd.DataFrame:
    """Back-adjusted copy of an OHLCV frame. Returns across each ex-date equal the economic (total) return."""
    actions = list(actions)
    f = adjustment_factors(df, actions)
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        if col in out:
            out[col] = out[col] * f["price"]
    if "volume" in out:
        out["volume"] = out["volume"] * f["volume"]
    return out


def unadjusted_position(quantity: float, avg_price: float, action: CorporateAction) -> tuple:
    """What a split does to an open position: (new quantity, new average price). Position value is unchanged."""
    if action.kind != "split":
        return quantity, avg_price
    return quantity * action.value, avg_price / action.value


def dividend_cash(quantity: float, actions: Iterable[CorporateAction]) -> float:
    """Cash credited (long) or debited (short) for dividends on a position held through each ex-date."""
    return float(sum(quantity * a.value for a in actions if a.kind == "dividend"))


def actions_from_records(records: Iterable[dict]) -> List[CorporateAction]:
    return [CorporateAction(r["symbol"], r["ex_date"], r["kind"], float(r["value"])) for r in records]
