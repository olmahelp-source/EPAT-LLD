"""Candlestick pattern identifiers.

Every function takes open/high/low/close and returns an int array aligned with the bars: +1 bullish, -1 bearish,
0 nothing. Direction-neutral patterns (doji, spinning top, inside / outside bar) return 1 when present. A pattern is
flagged on its *last* candle, so the value at t depends only on bars up to t.

Reversal patterns need something to reverse: where the classic definition says "after a downtrend", the previous close
must be below its ``trend`` -bar simple average (and above it for "after an uptrend").
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from ...foundation.core.exceptions import ValidationError
from ...quantitative.quant.indicators import _a, sma


def _parts(o, h, l, c):
    o, h, l, c = _a(o), _a(h), _a(l), _a(c)
    body = np.abs(c - o)
    rng = h - l
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    return o, h, l, c, body, np.where(rng > 0, rng, np.nan), upper, lower


def _prev(x: np.ndarray, k: int = 1) -> np.ndarray:
    """The series k bars ago (k = 0 is the series itself)."""
    if k == 0:
        return np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) > k:
        out[k:] = x[:-k]
    return out


def _trend(c: np.ndarray, n: int) -> np.ndarray:
    """+1 if the previous close was above its n-bar average, -1 if below, 0 while unknown."""
    ref = _prev(c) - _prev(sma(c, n))
    return np.where(np.isnan(ref), 0, np.sign(ref)).astype(int)


def _avg_body(body: np.ndarray, n: int = 10) -> np.ndarray:
    return _prev(sma(body, n))                                    # the candle being judged is not in its own baseline


def _flag(cond: np.ndarray, sign: int = 1) -> np.ndarray:
    return np.where(np.nan_to_num(cond.astype(float)) > 0, sign, 0).astype(int)


def doji(o, h, l, c, max_body: float = 0.1) -> np.ndarray:
    """Open and close almost equal: indecision."""
    *_, body, rng, _u, _l = _parts(o, h, l, c)
    with np.errstate(invalid="ignore"):
        return _flag(body <= max_body * rng)


def spinning_top(o, h, l, c, max_body: float = 0.3, min_shadow: float = 0.25) -> np.ndarray:
    *_, body, rng, upper, lower = _parts(o, h, l, c)
    with np.errstate(invalid="ignore"):
        return _flag((body <= max_body * rng) & (body > 0.1 * rng) & (upper >= min_shadow * rng) & (lower >= min_shadow * rng))


def marubozu(o, h, l, c, max_shadow: float = 0.05) -> np.ndarray:
    """A full-bodied candle with (almost) no shadows: one side was in control all bar."""
    o, h, l, c, body, rng, upper, lower = _parts(o, h, l, c)
    with np.errstate(invalid="ignore"):
        full = (upper <= max_shadow * rng) & (lower <= max_shadow * rng) & (body > 0)
    return _flag(full & (c > o)) - _flag(full & (c < o))


def _pin(o, h, l, c, tail: str, body_max: float, tail_min: float, other_max: float) -> np.ndarray:
    *_, body, rng, upper, lower = _parts(o, h, l, c)
    long_tail, short_tail = (lower, upper) if tail == "lower" else (upper, lower)
    with np.errstate(invalid="ignore"):
        return (body <= body_max * rng) & (long_tail >= tail_min * np.maximum(body, 1e-12)) & (short_tail <= other_max * rng) & (rng > 0)


def hammer(o, h, l, c, trend: int = 10) -> np.ndarray:
    """Small body at the top with a long lower shadow, after a decline (bullish)."""
    return _flag(_pin(o, h, l, c, "lower", 0.35, 2.0, 0.15) & (_trend(_a(c), trend) < 0))


def hanging_man(o, h, l, c, trend: int = 10) -> np.ndarray:
    """The hammer shape after an advance (bearish)."""
    return _flag(_pin(o, h, l, c, "lower", 0.35, 2.0, 0.15) & (_trend(_a(c), trend) > 0), -1)


def inverted_hammer(o, h, l, c, trend: int = 10) -> np.ndarray:
    return _flag(_pin(o, h, l, c, "upper", 0.35, 2.0, 0.15) & (_trend(_a(c), trend) < 0))


def shooting_star(o, h, l, c, trend: int = 10) -> np.ndarray:
    """Small body at the bottom with a long upper shadow, after an advance (bearish)."""
    return _flag(_pin(o, h, l, c, "upper", 0.35, 2.0, 0.15) & (_trend(_a(c), trend) > 0), -1)


def engulfing(o, h, l, c) -> np.ndarray:
    """The body swallows the previous, opposite-coloured body."""
    o, h, l, c, *_ = _parts(o, h, l, c)
    po, pc = _prev(o), _prev(c)
    with np.errstate(invalid="ignore"):
        bull = (pc < po) & (c > o) & (o <= pc) & (c >= po) & ((c - o) > (po - pc))
        bear = (pc > po) & (c < o) & (o >= pc) & (c <= po) & ((o - c) > (pc - po))
    return _flag(bull) - _flag(bear)


def harami(o, h, l, c) -> np.ndarray:
    """A small body inside the previous, opposite-coloured large body."""
    o, h, l, c, body, *_ = _parts(o, h, l, c)
    po, pc, pbody = _prev(o), _prev(c), _prev(body)
    with np.errstate(invalid="ignore"):
        inside = (np.maximum(o, c) < np.maximum(po, pc)) & (np.minimum(o, c) > np.minimum(po, pc)) & (body < 0.6 * pbody)
        big = pbody > _avg_body(body, 10) if len(body) > 11 else pbody > 0
    return _flag(inside & big & (pc < po) & (c > o)) - _flag(inside & big & (pc > po) & (c < o))


def piercing_line(o, h, l, c) -> np.ndarray:
    """After a down candle: opens below its low and closes above the middle of its body (bullish)."""
    o, h, l, c, *_ = _parts(o, h, l, c)
    po, pc, pl = _prev(o), _prev(c), _prev(l)
    with np.errstate(invalid="ignore"):
        return _flag((pc < po) & (c > o) & (o < pl) & (c > (po + pc) / 2.0) & (c < po))


def dark_cloud_cover(o, h, l, c) -> np.ndarray:
    o, h, l, c, *_ = _parts(o, h, l, c)
    po, pc, ph = _prev(o), _prev(c), _prev(h)
    with np.errstate(invalid="ignore"):
        return _flag((pc > po) & (c < o) & (o > ph) & (c < (po + pc) / 2.0) & (c > po), -1)


def _star(o, h, l, c, bullish: bool) -> np.ndarray:
    o, h, l, c, body, *_ = _parts(o, h, l, c)
    o1, c1, b1 = _prev(o, 2), _prev(c, 2), _prev(body, 2)
    o2, c2, b2 = _prev(o), _prev(c), _prev(body)
    avg = _prev(sma(body, 10), 3) if len(body) > 13 else np.full(len(body), 0.0)
    with np.errstate(invalid="ignore"):
        if bullish:
            return (c1 < o1) & (b1 > avg) & (b2 < 0.5 * b1) & (np.maximum(o2, c2) < c1) & (c > o) & (c > (o1 + c1) / 2.0)
        return (c1 > o1) & (b1 > avg) & (b2 < 0.5 * b1) & (np.minimum(o2, c2) > c1) & (c < o) & (c < (o1 + c1) / 2.0)


def morning_star(o, h, l, c) -> np.ndarray:
    """Big down candle, a small body gapping lower, then an up candle closing above the first one's midpoint."""
    return _flag(_star(o, h, l, c, True))


def evening_star(o, h, l, c) -> np.ndarray:
    return _flag(_star(o, h, l, c, False), -1)


def _three(o, h, l, c, up: bool) -> np.ndarray:
    o, h, l, c, body, rng, upper, lower = _parts(o, h, l, c)
    ok = np.ones(len(c), dtype=bool)
    with np.errstate(invalid="ignore"):
        for k in range(3):                                             # three solid candles of the same colour
            open_k, close_k = _prev(o, k), _prev(c, k)
            ok &= ((close_k > open_k) if up else (close_k < open_k)) & (_prev(body, k) >= 0.5 * _prev(rng, k))
        for k in range(2):                                             # each closes beyond, and opens inside, the one before
            open_k, close_k, open_b, close_b = _prev(o, k), _prev(c, k), _prev(o, k + 1), _prev(c, k + 1)
            if up:
                ok &= (close_k > close_b) & (open_k > open_b) & (open_k < close_b)
            else:
                ok &= (close_k < close_b) & (open_k < open_b) & (open_k > close_b)
    return ok


def three_white_soldiers(o, h, l, c) -> np.ndarray:
    return _flag(_three(o, h, l, c, True))


def three_black_crows(o, h, l, c) -> np.ndarray:
    return _flag(_three(o, h, l, c, False), -1)


def tweezer(o, h, l, c, tolerance: float = 0.001, trend: int = 10) -> np.ndarray:
    """Two bars with matching lows after a decline (+1) or matching highs after an advance (-1)."""
    o, h, l, c, *_ = _parts(o, h, l, c)
    tr = _trend(c, trend)
    with np.errstate(invalid="ignore"):
        bottom = (np.abs(l - _prev(l)) <= tolerance * l) & (_prev(c) < _prev(o)) & (c > o) & (tr < 0)
        top = (np.abs(h - _prev(h)) <= tolerance * h) & (_prev(c) > _prev(o)) & (c < o) & (tr > 0)
    return _flag(bottom) - _flag(top)


def inside_bar(o, h, l, c) -> np.ndarray:
    """The whole bar sits within the previous bar's range (contraction)."""
    _o, h, l, *_ = _parts(o, h, l, c)
    with np.errstate(invalid="ignore"):
        return _flag((h < _prev(h)) & (l > _prev(l)))


def outside_bar(o, h, l, c) -> np.ndarray:
    _o, h, l, *_ = _parts(o, h, l, c)
    with np.errstate(invalid="ignore"):
        return _flag((h > _prev(h)) & (l < _prev(l)))


def gap(o, h, l, c, min_pct: float = 0.0) -> np.ndarray:
    """+1 when the bar's low is above the previous high (gap up), -1 when its high is below the previous low."""
    _o, h, l, *_ = _parts(o, h, l, c)
    with np.errstate(invalid="ignore"):
        up = l > _prev(h) * (1 + min_pct)
        down = h < _prev(l) * (1 - min_pct)
    return _flag(up) - _flag(down)


CANDLE_PATTERNS: Dict[str, Callable[..., np.ndarray]] = {
    "doji": doji, "spinning_top": spinning_top, "marubozu": marubozu, "hammer": hammer, "hanging_man": hanging_man,
    "inverted_hammer": inverted_hammer, "shooting_star": shooting_star, "engulfing": engulfing, "harami": harami,
    "piercing_line": piercing_line, "dark_cloud_cover": dark_cloud_cover, "morning_star": morning_star,
    "evening_star": evening_star, "three_white_soldiers": three_white_soldiers, "three_black_crows": three_black_crows,
    "tweezer": tweezer, "inside_bar": inside_bar, "outside_bar": outside_bar, "gap": gap,
}
NEUTRAL_PATTERNS = frozenset({"doji", "spinning_top", "inside_bar", "outside_bar"})


def scan_candles(data: Mapping[str, Sequence[float]], patterns: Sequence[str] = ()) -> pd.DataFrame:
    """Run candlestick detectors over an OHLC frame / mapping. One int column per pattern, plus ``score``: the sum of the
    directional ones (neutral patterns are excluded from the score)."""
    names = list(patterns) or list(CANDLE_PATTERNS)
    unknown = [n for n in names if n not in CANDLE_PATTERNS]
    if unknown:
        raise ValidationError(f"unknown candlestick pattern(s) {unknown}", details={"valid": sorted(CANDLE_PATTERNS)})
    missing = [k for k in ("open", "high", "low", "close") if k not in data]
    if missing:
        raise ValidationError(f"candlestick scan needs columns {missing}")
    o, h, l, c = (data[k] for k in ("open", "high", "low", "close"))
    out = pd.DataFrame({n: CANDLE_PATTERNS[n](o, h, l, c) for n in names}, index=getattr(data, "index", None))
    out["score"] = out[[n for n in names if n not in NEUTRAL_PATTERNS]].sum(axis=1) if any(n not in NEUTRAL_PATTERNS for n in names) else 0
    return out
