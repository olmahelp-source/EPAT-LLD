"""Operators: the verbs that turn indicator series into conditions and signals.

Indicators answer "what is the value"; operators answer "did it cross, is it rising, how long since, what was it when".
All are vectorised over the whole series, causal, and NaN-safe: a comparison involving NaN is False, never an error and
never True. Scalars broadcast, so ``crossover(rsi, 30)`` works. Conditions are bool arrays, values are float arrays.
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ...foundation.core.exceptions import ValidationError
from ...quantitative.quant.indicators import _pad, _period, _windows, rolling_max, rolling_min


def _pair(a: Any, b: Any) -> Tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim == 0 and b.ndim == 0:
        return a.reshape(1), b.reshape(1)
    if a.ndim == 0:
        a = np.full(len(b), float(a))
    if b.ndim == 0:
        b = np.full(len(a), float(b))
    if len(a) != len(b):
        raise ValidationError("operands must have the same length", details={"left": len(a), "right": len(b)})
    return a, b


def _cond(x: Any) -> np.ndarray:
    """Anything -> bool array, with NaN counted as False."""
    a = np.asarray(x)
    if a.dtype == bool:
        return np.atleast_1d(a)
    a = np.atleast_1d(a.astype(float))
    return np.nan_to_num(a, nan=0.0) != 0


def _lag(x: np.ndarray, n: int = 1) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if n == 0:
        return x.astype(float)
    if len(x) > n:
        out[n:] = x[:-n]
    return out


# ---- comparisons -------------------------------------------------------------------------------------------------------
def above(a, b) -> np.ndarray:
    a, b = _pair(a, b)
    with np.errstate(invalid="ignore"):
        return a > b


def below(a, b) -> np.ndarray:
    a, b = _pair(a, b)
    with np.errstate(invalid="ignore"):
        return a < b


def between(x, low, high) -> np.ndarray:
    return and_(above_or_equal(x, low), above_or_equal(high, x))


def above_or_equal(a, b) -> np.ndarray:
    a, b = _pair(a, b)
    with np.errstate(invalid="ignore"):
        return a >= b


def crossover(a, b) -> np.ndarray:
    """True on the bar where ``a`` moves from at-or-below ``b`` to above it."""
    a, b = _pair(a, b)
    with np.errstate(invalid="ignore"):
        return (a > b) & (_lag(a) <= _lag(b))


def crossunder(a, b) -> np.ndarray:
    a, b = _pair(a, b)
    with np.errstate(invalid="ignore"):
        return (a < b) & (_lag(a) >= _lag(b))


def cross(a, b) -> np.ndarray:
    return crossover(a, b) | crossunder(a, b)


# ---- logic --------------------------------------------------------------------------------------------------------------
def and_(*conds) -> np.ndarray:
    out = _cond(conds[0])
    for c in conds[1:]:
        c = _cond(c)
        out = out & c if len(c) == len(out) or len(c) == 1 or len(out) == 1 else _mismatch(out, c)
    return out


def or_(*conds) -> np.ndarray:
    out = _cond(conds[0])
    for c in conds[1:]:
        c = _cond(c)
        out = out | c if len(c) == len(out) or len(c) == 1 or len(out) == 1 else _mismatch(out, c)
    return out


def _mismatch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    raise ValidationError("conditions must have the same length", details={"left": len(a), "right": len(b)})


def not_(cond) -> np.ndarray:
    return ~_cond(cond)


def where(cond, a, b) -> np.ndarray:
    """``a`` where the condition holds, otherwise ``b``."""
    c = _cond(cond)
    return np.where(c, np.asarray(a, float), np.asarray(b, float))


def nz(x, value: float = 0.0) -> np.ndarray:
    """Replace NaN by ``value``."""
    return np.nan_to_num(np.asarray(x, float), nan=value)


# ---- looking back ---------------------------------------------------------------------------------------------------------
def shift(x, n: int = 1) -> np.ndarray:
    """The value n bars ago (n must be >= 0: looking forward is not offered)."""
    if n < 0:
        raise ValidationError("shift must be >= 0; a negative shift would read the future")
    return _lag(np.asarray(x, float), int(n))


def change(x, n: int = 1) -> np.ndarray:
    x = np.asarray(x, float)
    return x - shift(x, n)


def pct_change(x, n: int = 1) -> np.ndarray:
    x = np.asarray(x, float)
    prev = shift(x, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return x / np.where(prev == 0, np.nan, prev) - 1.0


def highest(x, n: int) -> np.ndarray:
    return rolling_max(x, n)


def lowest(x, n: int) -> np.ndarray:
    return rolling_min(x, n)


def rising(x, n: int = 1) -> np.ndarray:
    """Strictly higher than the bar before, for each of the last n bars."""
    x = np.asarray(x, float)
    with np.errstate(invalid="ignore"):
        return all_true(x > _lag(x), n)


def falling(x, n: int = 1) -> np.ndarray:
    x = np.asarray(x, float)
    with np.errstate(invalid="ignore"):
        return all_true(x < _lag(x), n)


def count_true(cond, n: int) -> np.ndarray:
    """How many of the last n bars satisfied the condition (NaN until n bars exist)."""
    c, n = _cond(cond).astype(float), _period(n)
    return _pad(_windows(c, n).sum(axis=1), len(c), n)


def all_true(cond, n: int) -> np.ndarray:
    """The condition held on each of the last n bars (use to demand confirmation before acting)."""
    return np.nan_to_num(count_true(cond, n), nan=0.0) >= _period(n)


def any_true(cond, n: int) -> np.ndarray:
    """The condition held at least once in the last n bars (use to keep a trigger 'live' for a few bars)."""
    c, n = _cond(cond).astype(float), _period(n)
    csum = np.cumsum(c)
    before = np.zeros(len(c))
    if len(c) > n:
        before[n:] = csum[:-n]
    return (csum - before) > 0


def streak(cond) -> np.ndarray:
    """Length of the current run of consecutive True bars (0 when False)."""
    c = _cond(cond)
    out = np.zeros(len(c))
    run = 0
    for i, v in enumerate(c):
        run = run + 1 if v else 0
        out[i] = run
    return out


def barssince(cond) -> np.ndarray:
    """Bars elapsed since the condition was last True (0 on the bar itself, NaN before it ever happened)."""
    c = _cond(cond)
    idx = np.where(c, np.arange(len(c)), -1)
    last = np.maximum.accumulate(idx) if len(c) else idx
    return np.where(last >= 0, np.arange(len(c)) - last, np.nan)


def valuewhen(cond, x, occurrence: int = 0) -> np.ndarray:
    """Value of ``x`` on the most recent bar where the condition was True (``occurrence`` = 1 for the one before, ...)."""
    c, x = _cond(cond), np.asarray(x, float)
    if occurrence < 0:
        raise ValidationError("occurrence must be >= 0")
    if x.ndim == 0:
        x = np.full(len(c), float(x))
    out = np.full(len(c), np.nan)
    hits: list = []
    for i in range(len(c)):
        if c[i]:
            hits.append(x[i])
        if len(hits) > occurrence:
            out[i] = hits[-1 - occurrence]
    return out


def rising_edge(cond) -> np.ndarray:
    """True only on the first bar of each run of True values."""
    c = _cond(cond)
    prev = np.concatenate([[False], c[:-1]]) if len(c) else c
    return c & ~prev


def latch(set_cond, reset_cond) -> np.ndarray:
    """Flip-flop: becomes True on ``set_cond``, stays True until ``reset_cond``. Turns entry / exit events into an
    in-position state. If both fire on the same bar, reset wins (stand aside when in doubt)."""
    s, r = _cond(set_cond), _cond(reset_cond)
    if len(s) != len(r):
        _mismatch(s, r)
    out = np.zeros(len(s), dtype=bool)
    state = False
    for i in range(len(s)):
        if r[i]:
            state = False
        elif s[i]:
            state = True
        out[i] = state
    return out


def exrem(a, b) -> np.ndarray:
    """Remove repeated signals: keep the first True of ``a``, then ignore ``a`` until ``b`` has fired."""
    a, b = _cond(a), _cond(b)
    if len(a) != len(b):
        _mismatch(a, b)
    out = np.zeros(len(a), dtype=bool)
    armed = True
    for i in range(len(a)):
        if armed and a[i]:
            out[i], armed = True, False
        elif b[i]:
            armed = True
    return out


# ---- scaling ----------------------------------------------------------------------------------------------------------------
def percent_rank(x, n: int) -> np.ndarray:
    """Share of the previous n-1 values that the current value exceeds: 0 = lowest in the window, 1 = highest."""
    x, n = np.asarray(x, float), _period(n)
    if n < 2:
        raise ValidationError("percent_rank needs n >= 2")
    w = _windows(x, n)
    if not len(w):
        return np.full(len(x), np.nan)
    with np.errstate(invalid="ignore"):
        r = (w[:, :-1] < w[:, -1:]).sum(axis=1) / (n - 1)
    return _pad(np.where(np.isnan(w).any(axis=1), np.nan, r), len(x), n)


def normalize(x, n: int) -> np.ndarray:
    """Position within the n-bar range, 0..1."""
    x = np.asarray(x, float)
    lo, hi = rolling_min(x, n), rolling_max(x, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x - lo) / np.where(hi == lo, np.nan, hi - lo)


def clamp(x, low: float, high: float) -> np.ndarray:
    return np.clip(np.asarray(x, float), low, high)


def abs_(x) -> np.ndarray:
    return np.abs(np.asarray(x, float))


def sign(x) -> np.ndarray:
    return np.sign(np.asarray(x, float))


def max_(a, b) -> np.ndarray:
    a, b = _pair(a, b)
    return np.fmax(a, b)


def min_(a, b) -> np.ndarray:
    a, b = _pair(a, b)
    return np.fmin(a, b)


def log(x) -> np.ndarray:
    x = np.asarray(x, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)


def sqrt(x) -> np.ndarray:
    x = np.asarray(x, float)
    return np.where(x >= 0, np.sqrt(np.where(x >= 0, x, 0.0)), np.nan)


def cummax(x) -> np.ndarray:
    """Highest value so far (NaNs ignored)."""
    return np.fmax.accumulate(np.asarray(x, float))


def cummin(x) -> np.ndarray:
    return np.fmin.accumulate(np.asarray(x, float))
