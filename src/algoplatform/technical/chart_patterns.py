"""Chart pattern identifiers built on confirmed swings (see ``levels.swing_points``).

Each detector returns :class:`PatternMatch` objects. ``confirm`` is the bar at which the pattern became *known*: the
neckline break for reversal patterns, the swing confirmation for divergences, the last bar for consolidation shapes.
A backtest may act on a match from ``confirm`` onwards and not a bar earlier. Detection is prefix-consistent: running a
detector on the first k bars returns exactly the full-history matches with ``confirm < k`` (tested).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..core.exceptions import ValidationError
from ..quant.indicators import _a, _period, rolling_max, rolling_min, rsi
from .levels import Swing, swing_points


@dataclass(frozen=True)
class PatternMatch:
    name: str
    direction: int                 # +1 bullish, -1 bearish, 0 neutral
    start: int                     # first bar of the formation
    end: int                       # bar of the last defining swing
    confirm: int                   # first bar at which the pattern is known / triggered
    level: float                   # key price: neckline or boundary that was broken
    target: float = float("nan")   # measured-move objective, when the pattern defines one
    meta: Dict[str, float] = field(default_factory=dict)


def _by_kind(swings: Sequence[Swing], kind: str) -> List[Swing]:
    return sorted((s for s in swings if s.kind == kind), key=lambda s: s.index)


def double_tops_bottoms(high, low, close, left: int = 3, right: int = 3, tolerance: float = 0.02, min_depth: float = 0.03,
                        max_width: int = 120, max_wait: int = 30) -> List[PatternMatch]:
    """Two similar swing highs (lows) with a pullback of at least ``min_depth`` between them, confirmed when a close
    breaks the pullback's extreme within ``max_wait`` bars. A move back beyond the peaks first cancels the setup."""
    h, l, c = _a(high), _a(low), _a(close)
    swings = swing_points(h, l, left, right)
    out: List[PatternMatch] = []
    for kind, sign in (("high", -1), ("low", 1)):
        pts = _by_kind(swings, kind)
        for a, b in zip(pts, pts[1:]):
            if b.index - a.index > max_width or abs(a.price - b.price) > tolerance * a.price:
                continue
            if kind == "high":
                extreme, neck = max(a.price, b.price), float(l[a.index:b.index + 1].min())
                depth = (extreme - neck) / extreme
            else:
                extreme, neck = min(a.price, b.price), float(h[a.index:b.index + 1].max())
                depth = (neck - extreme) / extreme
            if depth < min_depth:
                continue
            for j in range(b.index + 1, min(len(c), b.index + 1 + max_wait)):
                if (kind == "high" and h[j] > extreme * (1 + tolerance)) or (kind == "low" and l[j] < extreme * (1 - tolerance)):
                    break                                                 # a third, higher peak: not a double top any more
                if (kind == "high" and c[j] < neck) or (kind == "low" and c[j] > neck):
                    height = abs(extreme - neck)
                    out.append(PatternMatch("double_top" if kind == "high" else "double_bottom", sign, a.index, b.index,
                                            max(j, b.confirm), neck, neck + sign * height, {"extreme": extreme, "depth": depth}))
                    break
    return sorted(out, key=lambda m: m.confirm)


def head_and_shoulders(high, low, close, left: int = 3, right: int = 3, shoulder_tolerance: float = 0.03, head_margin: float = 0.01,
                       max_wait: int = 30) -> List[PatternMatch]:
    """Three swing highs with the middle one clearly highest and the outer two alike; confirmed by a close through the
    neckline drawn across the two troughs. Also finds the inverse (bullish) form on swing lows."""
    h, l, c = _a(high), _a(low), _a(close)
    swings = swing_points(h, l, left, right)
    out: List[PatternMatch] = []
    for kind, sign in (("high", -1), ("low", 1)):
        pts = _by_kind(swings, kind)
        for ls, hd, rs in zip(pts, pts[1:], pts[2:]):
            if kind == "high":
                if hd.price <= max(ls.price, rs.price) * (1 + head_margin):
                    continue
                opposite = l
                i1 = ls.index + int(np.argmin(opposite[ls.index:hd.index + 1]))
                i2 = hd.index + int(np.argmin(opposite[hd.index:rs.index + 1]))
            else:
                if hd.price >= min(ls.price, rs.price) * (1 - head_margin):
                    continue
                opposite = h
                i1 = ls.index + int(np.argmax(opposite[ls.index:hd.index + 1]))
                i2 = hd.index + int(np.argmax(opposite[hd.index:rs.index + 1]))
            if abs(ls.price - rs.price) > shoulder_tolerance * hd.price or i2 == i1:
                continue
            p1, p2 = float(opposite[i1]), float(opposite[i2])
            slope = (p2 - p1) / (i2 - i1)
            neck_at = lambda j: p1 + slope * (j - i1)                                  # noqa: E731
            for j in range(rs.index + 1, min(len(c), rs.index + 1 + max_wait)):
                if (kind == "high" and h[j] > hd.price) or (kind == "low" and l[j] < hd.price):
                    break
                neck = neck_at(j)
                if (kind == "high" and c[j] < neck) or (kind == "low" and c[j] > neck):
                    height = abs(hd.price - neck_at(hd.index))
                    out.append(PatternMatch("head_and_shoulders" if kind == "high" else "inverse_head_and_shoulders", sign, ls.index,
                                            rs.index, max(j, rs.confirm), float(neck), float(neck + sign * height),
                                            {"head": hd.price, "left_shoulder": ls.price, "right_shoulder": rs.price}))
                    break
    return sorted(out, key=lambda m: m.confirm)


def _line(points: Sequence[Swing]) -> tuple:
    x = np.array([p.index for p in points], float)
    y = np.array([p.price for p in points], float)
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    return slope, float(y.mean() - slope * x.mean())


def consolidation_pattern(high, low, close, left: int = 3, right: int = 3, lookback: int = 120, flat: float = 0.0004,
                          points: int = 3) -> Optional[PatternMatch]:
    """Shape of the current consolidation, from lines fitted through the last ``points`` swing highs and swing lows.

    Slopes are measured per bar as a fraction of price; below ``flat`` a line counts as horizontal. Returns one of
    ascending / descending / symmetrical triangle, rising / falling wedge, channel up / down, rectangle, or None when
    there are not enough swings. ``level`` is the upper boundary projected to the last bar."""
    h, l, c = _a(high), _a(low), _a(close)
    n = len(c)
    if n < 2:
        return None
    swings = [s for s in swing_points(h, l, left, right) if s.index >= n - lookback]
    highs, lows = _by_kind(swings, "high")[-points:], _by_kind(swings, "low")[-points:]
    if len(highs) < 2 or len(lows) < 2:
        return None
    (sh, bh), (sl, bl) = _line(highs), _line(lows)
    price = float(c[-1])
    rh, rl = sh / price, sl / price
    up_h, dn_h, up_l, dn_l = rh > flat, rh < -flat, rl > flat, rl < -flat
    if not (up_h or dn_h) and up_l:
        name, direction = "ascending_triangle", 1
    elif dn_h and not (up_l or dn_l):
        name, direction = "descending_triangle", -1
    elif dn_h and up_l:
        name, direction = "symmetrical_triangle", 0
    elif up_h and up_l:
        name, direction = ("rising_wedge", -1) if rl > rh * 1.25 else ("channel_up", 1)
    elif dn_h and dn_l:
        name, direction = ("falling_wedge", 1) if rh < rl * 1.25 else ("channel_down", -1)
    elif up_h and dn_l:
        name, direction = "broadening", 0
    else:
        name, direction = "rectangle", 0
    upper, lower = sh * (n - 1) + bh, sl * (n - 1) + bl
    start = min(highs[0].index, lows[0].index)
    return PatternMatch(name, direction, start, max(highs[-1].index, lows[-1].index), n - 1, float(upper),
                        meta={"upper": float(upper), "lower": float(lower), "slope_high": rh, "slope_low": rl})


def range_breakout(high, low, close, n: int = 20, max_width: float = 0.08) -> np.ndarray:
    """+1 / -1 on the bar whose close leaves a tight range: the previous ``n`` bars spanned at most ``max_width``
    (fraction of the range low) and the close is outside them. 0 otherwise. Array-valued, so usable in formulas."""
    h, l, c, n = _a(high), _a(low), _a(close), _period(n)
    hh, ll = np.full(len(c), np.nan), np.full(len(c), np.nan)
    hh[1:], ll[1:] = rolling_max(h, n)[:-1], rolling_min(l, n)[:-1]                  # the range excludes the current bar
    with np.errstate(divide="ignore", invalid="ignore"):
        tight = (hh - ll) / ll <= max_width
        return np.where(tight & (c > hh), 1, np.where(tight & (c < ll), -1, 0)).astype(int)


def divergences(high, low, oscillator, left: int = 3, right: int = 3, max_gap: int = 60) -> List[PatternMatch]:
    """Regular divergences between price swings and an oscillator sampled at the same bars: price makes a lower low
    while the oscillator makes a higher low (bullish), or a higher high against a lower oscillator high (bearish)."""
    h, l, osc = _a(high), _a(low), _a(oscillator)
    if len(osc) != len(h):
        raise ValidationError("oscillator must have the same length as price")
    swings = swing_points(h, l, left, right)
    out: List[PatternMatch] = []
    for kind, sign in (("low", 1), ("high", -1)):
        pts = _by_kind(swings, kind)
        for a, b in zip(pts, pts[1:]):
            oa, ob = osc[a.index], osc[b.index]
            if b.index - a.index > max_gap or not (np.isfinite(oa) and np.isfinite(ob)):
                continue
            if (kind == "low" and b.price < a.price and ob > oa) or (kind == "high" and b.price > a.price and ob < oa):
                out.append(PatternMatch("bullish_divergence" if sign > 0 else "bearish_divergence", sign, a.index, b.index, b.confirm,
                                        b.price, meta={"osc_first": float(oa), "osc_second": float(ob)}))
    return sorted(out, key=lambda m: m.confirm)


def scan_chart_patterns(data: Mapping[str, Sequence[float]], left: int = 3, right: int = 3) -> List[PatternMatch]:
    """All reversal patterns and RSI divergences in the data, ordered by confirmation bar, followed by the current
    consolidation shape (if any)."""
    missing = [k for k in ("high", "low", "close") if k not in data]
    if missing:
        raise ValidationError(f"chart pattern scan needs columns {missing}")
    h, l, c = data["high"], data["low"], data["close"]
    found = double_tops_bottoms(h, l, c, left, right) + head_and_shoulders(h, l, c, left, right) + divergences(h, l, rsi(c, 14), left, right)
    found.sort(key=lambda m: (m.confirm, m.name))
    shape = consolidation_pattern(h, l, c, left, right)
    return found + ([shape] if shape else [])


def pattern_signals(matches: Sequence[PatternMatch], n: int) -> np.ndarray:
    """Collapse matches into an int array of length n: the summed direction of patterns confirmed on each bar."""
    out = np.zeros(n, dtype=int)
    for m in matches:
        if 0 <= m.confirm < n:
            out[m.confirm] += m.direction
    return out
