"""Price structure: swings, support / resistance, pivots, Heikin-Ashi, Fibonacci levels.

Swings are the raw material for every chart pattern. A swing high at bar i needs ``right`` later bars to prove that it
was a high, so it only becomes *known* at bar ``i + right``. Everything here reports that confirmation bar, which is
what keeps pattern detection free of look-ahead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from ...foundation.core.exceptions import ValidationError
from ...quantitative.quant.indicators import _a


@dataclass(frozen=True)
class Swing:
    index: int          # bar of the extreme
    confirm: int        # first bar at which the swing is known (index + right)
    price: float
    kind: str           # "high" | "low"


def swing_points(high, low, left: int = 3, right: int = 3) -> List[Swing]:
    """Confirmed swing highs / lows in chronological order of confirmation.

    A bar is a swing high when its high is strictly above the ``left`` bars before it and not below the ``right`` bars
    after it (ties resolve to the earliest bar)."""
    h, l = _a(high), _a(low)
    if left < 1 or right < 1:
        raise ValidationError("left and right must be >= 1")
    out: List[Swing] = []
    for i in range(left, len(h) - right):
        if h[i] > h[i - left:i].max() and h[i] >= h[i + 1:i + right + 1].max():
            out.append(Swing(i, i + right, float(h[i]), "high"))
        if l[i] < l[i - left:i].min() and l[i] <= l[i + 1:i + right + 1].min():
            out.append(Swing(i, i + right, float(l[i]), "low"))
    out.sort(key=lambda s: (s.confirm, s.index))
    return out


def last_swing(high, low, kind: str = "high", left: int = 3, right: int = 3) -> np.ndarray:
    """Step series: at bar t, the price of the most recent swing of ``kind`` confirmed by t (NaN before the first).
    Handy in rules such as ``close > last_swing(high, low, "high")`` for a structure breakout."""
    if kind not in ("high", "low"):
        raise ValidationError("kind must be 'high' or 'low'")
    out = np.full(len(_a(high)), np.nan)
    for s in swing_points(high, low, left, right):
        if s.kind == kind:
            out[s.confirm:] = s.price
    return out


def support_resistance(high, low, close, left: int = 3, right: int = 3, tolerance: float = 0.01, min_touches: int = 2) -> List[Dict[str, Any]]:
    """Cluster confirmed swing prices that lie within ``tolerance`` (fraction of price) of each other into levels.

    Returns levels as of the last bar, strongest first. ``kind`` is 'support' when the level is below the last close."""
    c = _a(close)
    swings = swing_points(high, low, left, right)
    if not swings or not len(c):
        return []
    levels: List[Dict[str, Any]] = []
    for s in sorted(swings, key=lambda s: s.price):
        if levels and abs(s.price - levels[-1]["price"]) <= tolerance * levels[-1]["price"]:
            lv = levels[-1]
            lv["price"] = (lv["price"] * lv["touches"] + s.price) / (lv["touches"] + 1)
            lv["touches"] += 1
            lv["last_index"] = max(lv["last_index"], s.index)
        else:
            levels.append({"price": s.price, "touches": 1, "last_index": s.index})
    out = [dict(lv, kind="support" if lv["price"] < c[-1] else "resistance") for lv in levels if lv["touches"] >= min_touches]
    return sorted(out, key=lambda lv: (-lv["touches"], -lv["last_index"]))


def pivot_points(high, low, close, method: str = "classic") -> Dict[str, np.ndarray]:
    """Floor-trader pivots for bar t computed from bar t-1 (so they are known before bar t trades)."""
    h, l, c = _a(high), _a(low), _a(close)
    ph, pl, pc = (np.concatenate([[np.nan], x[:-1]]) for x in (h, l, c))
    rng = ph - pl
    p = (ph + pl + pc) / 3.0
    if method == "classic":
        return {"pivot": p, "r1": 2 * p - pl, "s1": 2 * p - ph, "r2": p + rng, "s2": p - rng, "r3": ph + 2 * (p - pl), "s3": pl - 2 * (ph - p)}
    if method == "fibonacci":
        return {"pivot": p, "r1": p + 0.382 * rng, "s1": p - 0.382 * rng, "r2": p + 0.618 * rng, "s2": p - 0.618 * rng,
                "r3": p + rng, "s3": p - rng}
    if method == "camarilla":
        return {"pivot": p, "r1": pc + rng * 1.1 / 12, "s1": pc - rng * 1.1 / 12, "r2": pc + rng * 1.1 / 6, "s2": pc - rng * 1.1 / 6,
                "r3": pc + rng * 1.1 / 4, "s3": pc - rng * 1.1 / 4}
    if method == "woodie":
        p = (ph + pl + 2 * pc) / 4.0
        return {"pivot": p, "r1": 2 * p - pl, "s1": 2 * p - ph, "r2": p + rng, "s2": p - rng, "r3": ph + 2 * (p - pl), "s3": pl - 2 * (ph - p)}
    raise ValidationError(f"unknown pivot method '{method}'", details={"valid": ["classic", "fibonacci", "camarilla", "woodie"]})


def heikin_ashi(open_, high, low, close) -> Dict[str, np.ndarray]:
    """Smoothed candles. They are an average, not tradable prices: use them for signals, never for fills."""
    o, h, l, c = _a(open_), _a(high), _a(low), _a(close)
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty(len(c))
    if len(c):
        ha_o[0] = (o[0] + c[0]) / 2.0
        for i in range(1, len(c)):
            ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    return {"open": ha_o, "high": np.maximum(np.maximum(h, ha_o), ha_c), "low": np.minimum(np.minimum(l, ha_o), ha_c), "close": ha_c}


def fibonacci_levels(swing_low: float, swing_high: float) -> Dict[str, float]:
    """Retracement (inside the move) and extension (beyond it) prices for an up-move from low to high."""
    if not swing_high > swing_low:
        raise ValidationError("swing_high must be above swing_low")
    span = swing_high - swing_low
    return {f"{r:g}": swing_high - r * span for r in (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)} | \
           {"ext_1.272": swing_low + 1.272 * span, "ext_1.618": swing_low + 1.618 * span}
