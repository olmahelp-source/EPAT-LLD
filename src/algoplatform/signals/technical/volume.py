"""Volume and money-flow indicators. Causal, NaN during warm-up."""
from __future__ import annotations

from typing import Dict

import numpy as np

from ...foundation.core.exceptions import ValidationError
from ...quantitative.quant.indicators import _a, _period, sma
from .trend import ema_from_first_valid, rolling_sum


def _close_location(high, low, close) -> np.ndarray:
    """Where the close sits in the bar's range: +1 at the high, -1 at the low, 0 for a zero-range bar."""
    h, l, c = _a(high), _a(low), _a(close)
    rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rng > 0, ((c - l) - (h - c)) / np.where(rng > 0, rng, 1.0), 0.0)


def mfi(high, low, close, volume, n: int = 14) -> np.ndarray:
    """Money flow index: an RSI weighted by traded value."""
    tp = (_a(high) + _a(low) + _a(close)) / 3.0
    flow = tp * _a(volume)
    d = np.diff(tp, prepend=np.nan)
    pos, neg = np.where(d > 0, flow, 0.0), np.where(d < 0, flow, 0.0)
    if len(tp):
        pos[0] = neg[0] = np.nan
    sp, sn = rolling_sum(pos, n), rolling_sum(neg, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100.0 - 100.0 / (1.0 + sp / sn)
    return np.where(sn == 0, np.where(np.isnan(sp), np.nan, 100.0), out)


def ad_line(high, low, close, volume) -> np.ndarray:
    """Accumulation / distribution line."""
    return np.cumsum(_close_location(high, low, close) * _a(volume))


def cmf(high, low, close, volume, n: int = 20) -> np.ndarray:
    """Chaikin money flow: -1..1, positive = closes near the highs on volume."""
    v = _a(volume)
    den = rolling_sum(v, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return rolling_sum(_close_location(high, low, close) * v, n) / np.where(den == 0, np.nan, den)


def force_index(close, volume, n: int = 13) -> np.ndarray:
    c = _a(close)
    return ema_from_first_valid(np.diff(c, prepend=np.nan) * _a(volume), n)


def ease_of_movement(high, low, volume, n: int = 14, scale: float = 1e6) -> np.ndarray:
    h, l, v = _a(high), _a(low), _a(volume)
    mid = (h + l) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        box = (v / scale) / np.where(h == l, np.nan, h - l)
        raw = np.diff(mid, prepend=np.nan) / box
    return sma(raw, n)


def pvt(close, volume) -> np.ndarray:
    """Price-volume trend: volume weighted by the bar's percentage change, accumulated."""
    c, v = _a(close), _a(volume)
    chg = np.zeros(len(c))
    with np.errstate(divide="ignore", invalid="ignore"):
        chg[1:] = c[1:] / c[:-1] - 1.0
    return np.cumsum(np.nan_to_num(chg) * v)


def relative_volume(volume, n: int = 20) -> np.ndarray:
    """Today's volume divided by the average of the *previous* n bars (the current bar is not in its own baseline)."""
    v, n = _a(volume), _period(n)
    base = np.full(len(v), np.nan)
    avg = sma(v, n)
    base[1:] = avg[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return v / np.where(base == 0, np.nan, base)


def volume_profile(price, volume, bins: int = 24, value_area: float = 0.7) -> Dict[str, object]:
    """Volume traded per price bucket over the window passed in.

    Returns the point of control (``poc``: price with the most volume), the value-area high / low that contain
    ``value_area`` of the volume around it, and the histogram. A summary of one window, so slice the data first."""
    p, v = _a(price), _a(volume)
    ok = np.isfinite(p) & np.isfinite(v)
    p, v = p[ok], v[ok]
    if len(p) < 2 or bins < 2 or not 0 < value_area <= 1:
        raise ValidationError("volume_profile needs >= 2 bars, bins >= 2 and value_area in (0, 1]")
    if p.max() == p.min():
        return {"poc": float(p[0]), "vah": float(p[0]), "val": float(p[0]), "edges": np.array([p[0], p[0]]), "volume": np.array([v.sum()])}
    hist, edges = np.histogram(p, bins=bins, weights=v)
    centers = (edges[:-1] + edges[1:]) / 2.0
    poc = int(np.argmax(hist))
    lo = hi = poc
    covered, total = hist[poc], hist.sum()
    while total > 0 and covered < value_area * total and (lo > 0 or hi < bins - 1):
        below = hist[lo - 1] if lo > 0 else -1.0
        above = hist[hi + 1] if hi < bins - 1 else -1.0
        if above >= below:
            hi += 1
            covered += hist[hi]
        else:
            lo -= 1
            covered += hist[lo]
    return {"poc": float(centers[poc]), "vah": float(edges[hi + 1]), "val": float(edges[lo]), "edges": edges, "volume": hist}
