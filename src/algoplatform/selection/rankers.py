"""Scores used to rank instruments. Each ranker takes one instrument's OHLCV frame (already cut off at the selection
date) and returns a single number where **higher is better**, or NaN when there is not enough history to judge."""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from ..core.exceptions import ValidationError

Ranker = Callable[..., float]
RANKERS: Dict[str, Ranker] = {}


def ranker(name: str) -> Callable[[Ranker], Ranker]:
    def deco(fn: Ranker) -> Ranker:
        RANKERS[name] = fn
        return fn
    return deco


def _log_returns(close: pd.Series, n: int) -> np.ndarray:
    c = close.to_numpy(float)[-(n + 1):]
    return np.diff(np.log(c)) if len(c) == n + 1 else np.array([])


@ranker("momentum")
def momentum(df: pd.DataFrame, lookback: int = 126, skip: int = 21) -> float:
    """Return over ``lookback`` bars ending ``skip`` bars ago. Skipping the most recent month avoids the well-known
    short-term reversal contaminating the medium-term trend (the classic "12-1" momentum is lookback=231, skip=21)."""
    c = df["close"].to_numpy(float)
    if lookback < 1 or skip < 0:
        raise ValidationError("need lookback >= 1 and skip >= 0")
    if len(c) < lookback + skip + 1:
        return float("nan")
    end = len(c) - 1 - skip
    return float(c[end] / c[end - lookback] - 1.0)


@ranker("volatility")
def volatility(df: pd.DataFrame, lookback: int = 63, periods: float = 252.0) -> float:
    """Annualised volatility (higher = more volatile). Use ``ascending=True`` in the selector to prefer calm names."""
    r = _log_returns(df["close"], lookback)
    return float(r.std(ddof=1) * math.sqrt(periods)) if len(r) > 2 else float("nan")


@ranker("low_volatility")
def low_volatility(df: pd.DataFrame, lookback: int = 63, periods: float = 252.0) -> float:
    return -volatility(df, lookback, periods)


@ranker("risk_adjusted_momentum")
def risk_adjusted_momentum(df: pd.DataFrame, lookback: int = 126, skip: int = 21) -> float:
    """Momentum per unit of volatility: prefers a steady climb over a single jump of the same size."""
    r = _log_returns(df["close"].iloc[:len(df) - skip] if skip else df["close"], lookback)
    return float(r.mean() / r.std(ddof=1) * math.sqrt(len(r))) if len(r) > 2 and r.std(ddof=1) > 0 else float("nan")


@ranker("trend_quality")
def trend_quality(df: pd.DataFrame, lookback: int = 90, periods: float = 252.0) -> float:
    """Annualised slope of a straight line through log prices, multiplied by the fit's R^2 (a smooth trend scores high,
    a volatile one with the same end points scores low)."""
    c = df["close"].to_numpy(float)[-lookback:]
    if len(c) < lookback or lookback < 5:
        return float("nan")
    y, t = np.log(c), np.arange(len(c), dtype=float)
    tc = t - t.mean()
    slope = float(tc @ (y - y.mean()) / (tc @ tc))
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = slope ** 2 * float(tc @ tc) / ss if ss > 0 else 0.0
    return float((math.exp(slope * periods) - 1.0) * r2)


@ranker("mean_reversion")
def mean_reversion(df: pd.DataFrame, lookback: int = 20) -> float:
    """How far below its recent average the price is, in standard deviations (higher = more oversold)."""
    c = df["close"].to_numpy(float)[-lookback:]
    if len(c) < lookback or c.std(ddof=1) == 0:
        return float("nan")
    return float(-(c[-1] - c.mean()) / c.std(ddof=1))


@ranker("liquidity")
def liquidity(df: pd.DataFrame, lookback: int = 20) -> float:
    """Average traded value per bar."""
    if len(df) < lookback:
        return float("nan")
    tail = df.iloc[-lookback:]
    return float((tail["close"] * tail["volume"]).mean())


@ranker("near_high")
def near_high(df: pd.DataFrame, lookback: int = 252) -> float:
    """Close as a fraction of the ``lookback`` high (1.0 = at the high)."""
    if len(df) < lookback:
        return float("nan")
    return float(df["close"].iloc[-1] / df["high"].iloc[-lookback:].max())


@ranker("relative_strength")
def relative_strength(df: pd.DataFrame, benchmark: Optional[pd.Series] = None, lookback: int = 126) -> float:
    """Return minus the benchmark's return over the same dates."""
    if benchmark is None:
        raise ValidationError("relative_strength needs a benchmark close series")
    b = benchmark.reindex(df.index).ffill()
    if len(df) < lookback + 1 or b.iloc[-lookback - 1:].isna().any():
        return float("nan")
    return float(df["close"].iloc[-1] / df["close"].iloc[-lookback - 1] - b.iloc[-1] / b.iloc[-lookback - 1])
