"""Volatility channels and range-state indicators. Causal, NaN during warm-up."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from ..quant.indicators import _a, _period, atr, bollinger, ema, rolling_max, rolling_min, true_range
from .trend import rolling_sum


def keltner(high, low, close, n: int = 20, mult: float = 2.0, atr_n: int = 10) -> Dict[str, np.ndarray]:
    mid, a = ema(close, n), atr(high, low, close, atr_n)
    return {"mid": mid, "upper": mid + mult * a, "lower": mid - mult * a}


def chandelier_exit(high, low, close, n: int = 22, mult: float = 3.0) -> Dict[str, np.ndarray]:
    """Trailing stops hung from the n-bar extreme: ``long_stop`` for longs, ``short_stop`` for shorts."""
    a = atr(high, low, close, n)
    return {"long_stop": rolling_max(high, n) - mult * a, "short_stop": rolling_min(low, n) + mult * a}


def natr(high, low, close, n: int = 14) -> np.ndarray:
    """ATR as a percentage of price (comparable across instruments)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * atr(high, low, close, n) / _a(close)


def choppiness(high, low, close, n: int = 14) -> np.ndarray:
    """Choppiness index: ~100 = sideways chop, ~0 = a straight-line trend. Common cut-offs are 61.8 and 38.2."""
    n = _period(n)
    span = rolling_max(high, n) - rolling_min(low, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * np.log10(rolling_sum(true_range(high, low, close), n) / np.where(span == 0, np.nan, span)) / math.log10(n)


def squeeze(high, low, close, n: int = 20, bb_k: float = 2.0, kc_mult: float = 1.5) -> np.ndarray:
    """1.0 while the Bollinger bands sit inside the Keltner channel (volatility compression that often precedes a
    breakout), 0.0 otherwise, NaN during warm-up."""
    bb, kc = bollinger(close, n, bb_k), keltner(high, low, close, n, kc_mult, n)
    known = ~(np.isnan(bb["upper"]) | np.isnan(kc["upper"]))
    with np.errstate(invalid="ignore"):
        inside = (bb["upper"] < kc["upper"]) & (bb["lower"] > kc["lower"])
    return np.where(known, inside.astype(float), np.nan)
