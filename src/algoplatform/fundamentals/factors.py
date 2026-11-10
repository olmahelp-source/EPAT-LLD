"""Cross-sectional factor scoring and a fundamentals-driven long/short strategy."""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from ..core.exceptions import StrategyError, ValidationError
from ..core.models import Signal
from ..strategy.base import STRATEGIES, Strategy, StrategyContext
from . import ratios, scores
from .statements import FundamentalsStore

DIRECTION = {"earnings_yield": 1, "book_to_price": 1, "fcf_yield": 1, "roe": 1, "gross_margin": 1, "piotroski": 1,
             "revenue_growth": 1, "debt_to_equity": -1, "accruals": -1}


def winsorize(s: pd.Series, p: float = 0.05) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=1)
    return (s - s.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else s * 0.0


def sector_neutralize(s: pd.Series, sectors: Mapping[str, str]) -> pd.Series:
    """Subtract each sector's mean so the factor ranks companies against their own industry."""
    sec = pd.Series({k: sectors.get(k, "?") for k in s.index})
    return s - s.groupby(sec).transform("mean")


def composite_score(factors: pd.DataFrame, weights: Optional[Mapping[str, float]] = None,
                    sectors: Optional[Mapping[str, str]] = None, winsor: float = 0.05) -> pd.Series:
    """Winsorise -> (sector-neutralise) -> z-score each factor, sign by direction, then weighted mean over the factors
    available for each company (a missing factor lowers confidence, it does not zero the company)."""
    cols = [c for c in factors.columns if c in DIRECTION]
    if not cols:
        raise ValidationError("no recognised factor columns")
    z = pd.DataFrame(index=factors.index)
    for c in cols:
        s = factors[c].astype(float)
        if s.notna().sum() < 3:
            continue
        s = winsorize(s.dropna(), winsor).reindex(s.index)
        if sectors:
            s = sector_neutralize(s, sectors)
        z[c] = zscore(s) * DIRECTION[c]
    if z.empty:
        raise ValidationError("not enough data to score any factor")
    w = pd.Series({c: (weights or {}).get(c, 1.0) for c in z.columns})
    num = (z * w).sum(axis=1, skipna=True)
    den = (z.notna() * w).sum(axis=1)
    return (num / den.replace(0, np.nan)).rename("score")


def factor_frame(store: FundamentalsStore, prices: Mapping[str, float], on: date, lag_days: int = 0) -> pd.DataFrame:
    """One row per symbol, using only statements public on ``on`` (point-in-time)."""
    rows: Dict[str, Dict[str, Optional[float]]] = {}
    for sym, px in prices.items():
        cur = store.asof(sym, on, lag_days)
        if cur is None or px is None or px <= 0:
            continue
        prev = store.previous(sym, cur, on, lag_days)
        cap = px * cur["shares"]
        row: Dict[str, Optional[float]] = {
            "earnings_yield": cur["net_income"] / cap, "book_to_price": cur["equity"] / cap,
            "fcf_yield": cur.free_cash_flow / cap if cur.get("cfo") is not None else None,
            "roe": ratios.profitability(cur, prev)["roe"], "gross_margin": ratios.profitability(cur, prev)["gross_margin"],
            "debt_to_equity": ratios.leverage(cur)["debt_to_equity"],
            "revenue_growth": ratios.growth(cur, prev)["revenue_growth"] if prev else None,
            "piotroski": scores.piotroski_f(cur, prev)["score"] if prev else None,
            "accruals": scores.sloan_accruals(cur, prev) if cur.get("cfo") is not None else None}
        rows[sym] = row
    return pd.DataFrame.from_dict(rows, orient="index")


@STRATEGIES.register("factor_ls")
class FactorStrategy(Strategy):
    """Rank a universe on fundamentals (as known at each date) and hold the top quantile, optionally short the bottom.

    The fundamentals store is attached with :meth:`with_store` (it is data, not a parameter)."""
    name = "factor_ls"
    PARAMS = {"rebalance_every": 21, "top_frac": 0.34, "long_short": False, "lag_days": 1, "min_names": 3}
    PARAM_BOUNDS = {"rebalance_every": (5, 63), "top_frac": (0.1, 0.5)}

    def validate_params(self) -> None:
        if not 0 < self.p("top_frac") <= 0.5 or self.p("rebalance_every") < 1:
            raise StrategyError("top_frac must be in (0, 0.5] and rebalance_every >= 1", details=self.params)
        self.store: Optional[FundamentalsStore] = getattr(self, "store", None)
        self.sectors: Optional[Mapping[str, str]] = getattr(self, "sectors", None)
        self.weights: Optional[Mapping[str, float]] = getattr(self, "weights", None)
        self._bar, self._targets = 0, {}

    @classmethod
    def with_store(cls, store: FundamentalsStore, symbols, sectors=None, factor_weights=None, **params) -> "FactorStrategy":
        s = cls(symbols, **params)
        s.store, s.sectors, s.weights = store, sectors, factor_weights
        return s

    def clone(self, **overrides) -> "FactorStrategy":
        c = FactorStrategy.with_store(self.store, self.symbols, self.sectors, self.weights, **{**self.params, **overrides})
        c.id = self.id
        return c

    def on_bar(self, ctx: StrategyContext) -> List[Signal]:
        if self.store is None:
            raise StrategyError("attach fundamentals with FactorStrategy.with_store(...)")
        self._bar += 1
        if self._targets and self._bar % int(self.p("rebalance_every")) != 0:
            return [Signal(s, w, "hold") for s, w in self._targets.items()]
        prices = {s: ctx.prices.get(s) for s in self.symbols if s in ctx.prices}
        frame = factor_frame(self.store, prices, ctx.ts.date(), int(self.p("lag_days")))
        if len(frame) < int(self.p("min_names")):
            return []
        sc = composite_score(frame, self.weights, self.sectors).dropna().sort_values(ascending=False)
        n = max(1, int(round(len(sc) * float(self.p("top_frac")))))
        self._targets = {s: 0.0 for s in self.symbols}
        for s in sc.index[:n]:
            self._targets[s] = 1.0
        if self.p("long_short"):
            for s in sc.index[-n:]:
                self._targets[s] = -1.0
        return [Signal(s, w, f"score={sc.get(s, float('nan')):+.2f}") for s, w in self._targets.items()]
