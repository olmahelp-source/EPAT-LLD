"""Momentum oscillators. Causal, NaN during warm-up, numpy in / numpy out."""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..quant.indicators import _a, _pad, _period, _windows, ema, rolling_max, rolling_min, rsi, sma
from .trend import ema_from_first_valid, rolling_sum


def momentum(x, n: int = 10) -> np.ndarray:
    """Price change over n bars (in price units; ``roc`` is the same thing as a fraction)."""
    x, n = _a(x), _period(n)
    out = np.full(len(x), np.nan)
    if len(x) > n:
        out[n:] = x[n:] - x[:-n]
    return out


def cci(high, low, close, n: int = 20) -> np.ndarray:
    """Commodity channel index: distance of typical price from its mean, in units of mean absolute deviation."""
    tp, n = (_a(high) + _a(low) + _a(close)) / 3.0, _period(n)
    w = _windows(tp, n)
    if not len(w):
        return np.full(len(tp), np.nan)
    mean = w.mean(axis=1)
    mad = np.abs(w - mean[:, None]).mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _pad((tp[n - 1:] - mean) / (0.015 * np.where(mad == 0, np.nan, mad)), len(tp), n)


def williams_r(high, low, close, n: int = 14) -> np.ndarray:
    """-100 (close at the n-bar low) .. 0 (close at the n-bar high)."""
    hh, ll, c = rolling_max(high, n), rolling_min(low, n), _a(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -100.0 * (hh - c) / np.where(hh == ll, np.nan, hh - ll)


def stoch_rsi(x, n: int = 14, k: int = 3, d: int = 3) -> Dict[str, np.ndarray]:
    r = rsi(x, n)
    lo, hi = rolling_min(r, n), rolling_max(r, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 100.0 * (r - lo) / np.where(hi == lo, np.nan, hi - lo)
    pk = sma(raw, k)
    return {"k": pk, "d": sma(pk, d)}


def tsi(x, long: int = 25, short: int = 13) -> np.ndarray:
    """True strength index: double-smoothed momentum over double-smoothed absolute momentum, -100..100."""
    x = _a(x)
    m = np.diff(x, prepend=np.nan)
    num = ema_from_first_valid(ema_from_first_valid(m, long), short)
    den = ema_from_first_valid(ema_from_first_valid(np.abs(m), long), short)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * num / np.where(den == 0, np.nan, den)


def ultimate_oscillator(high, low, close, n1: int = 7, n2: int = 14, n3: int = 28) -> np.ndarray:
    h, l, c = _a(high), _a(low), _a(close)
    prev = np.concatenate([[c[0]], c[:-1]]) if len(c) else c
    bp = c - np.minimum(l, prev)
    tr = np.maximum(h, prev) - np.minimum(l, prev)

    def avg(n: int) -> np.ndarray:
        den = rolling_sum(tr, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            return rolling_sum(bp, n) / np.where(den == 0, np.nan, den)
    return 100.0 * (4 * avg(n1) + 2 * avg(n2) + avg(n3)) / 7.0


def ppo(x, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
    """MACD as a percentage of the slow EMA, so it is comparable across instruments with different prices."""
    ef, es = ema(x, fast), ema(x, slow)
    with np.errstate(divide="ignore", invalid="ignore"):
        line = 100.0 * (ef - es) / es
    sig = ema_from_first_valid(line, signal)
    return {"ppo": line, "signal": sig, "hist": line - sig}


def cmo(x, n: int = 14) -> np.ndarray:
    """Chande momentum oscillator: (sum of gains - sum of losses) / (sum of both), -100..100."""
    x = _a(x)
    d = np.diff(x, prepend=np.nan)
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    up[0] = dn[0] = np.nan
    su, sd = rolling_sum(up, n), rolling_sum(dn, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * (su - sd) / np.where(su + sd == 0, np.nan, su + sd)


def awesome_oscillator(high, low, fast: int = 5, slow: int = 34) -> np.ndarray:
    mid = (_a(high) + _a(low)) / 2.0
    return sma(mid, fast) - sma(mid, slow)
