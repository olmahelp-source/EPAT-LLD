"""Trend indicators beyond the core set in ``quant.indicators``.

Same contract as the core set: causal (value at t uses data up to t), NaN during warm-up, numpy in / numpy out.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from ..core.exceptions import ValidationError
from ..quant.indicators import _a, _pad, _period, _windows, atr, ema, rolling_max, rolling_min, true_range, wma


def rolling_sum(x, n: int) -> np.ndarray:
    x, n = _a(x), _period(n)
    return _pad(_windows(x, n).sum(axis=1), len(x), n)


def ema_from_first_valid(x, n: int) -> np.ndarray:
    """EMA of a series that starts with NaNs (the output of another indicator), seeded at its first valid value."""
    x, n = _a(x), _period(n)
    out = np.full(len(x), np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) >= n:
        out[valid[0]:] = ema(x[valid[0]:], n)
    return out


def dema(x, n: int) -> np.ndarray:
    """Double EMA: less lag than an EMA of the same length."""
    e1 = ema(x, n)
    return 2 * e1 - ema_from_first_valid(e1, n)


def tema(x, n: int) -> np.ndarray:
    e1 = ema(x, n)
    e2 = ema_from_first_valid(e1, n)
    return 3 * e1 - 3 * e2 + ema_from_first_valid(e2, n)


def hma(x, n: int) -> np.ndarray:
    """Hull moving average."""
    n = _period(n)
    if n < 2:
        raise ValidationError("hma period must be >= 2")
    return wma(2 * wma(x, max(n // 2, 1)) - wma(x, n), max(int(math.sqrt(n)), 1))


def kama(x, n: int = 10, fast: int = 2, slow: int = 30) -> np.ndarray:
    """Kaufman adaptive MA: fast in a clean trend, slow in noise (efficiency ratio = net move / path length)."""
    x, n = _a(x), _period(n)
    out = np.full(len(x), np.nan)
    if len(x) <= n:
        return out
    f, s = 2.0 / (fast + 1), 2.0 / (slow + 1)
    path = rolling_sum(np.abs(np.diff(x, prepend=x[0])), n)
    out[n - 1] = x[n - 1]
    for t in range(n, len(x)):
        er = abs(x[t] - x[t - n]) / path[t] if path[t] > 0 else 0.0
        sc = (er * (f - s) + s) ** 2
        out[t] = out[t - 1] + sc * (x[t] - out[t - 1])
    return out


def vwma(price, volume, n: int) -> np.ndarray:
    p, v = _a(price), _a(volume)
    den = rolling_sum(v, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        return rolling_sum(p * v, n) / np.where(den == 0, np.nan, den)


def supertrend(high, low, close, n: int = 10, mult: float = 3.0) -> Dict[str, np.ndarray]:
    """ATR trailing band. ``direction`` is +1 (price above the line, uptrend) or -1."""
    h, l, c = _a(high), _a(low), _a(close)
    a = atr(h, l, c, n)
    size = len(c)
    line, direction = np.full(size, np.nan), np.full(size, np.nan)
    start = _period(n) - 1
    if size <= start:
        return {"line": line, "direction": direction}
    mid = (h + l) / 2.0
    ub, lb = mid + mult * a, mid - mult * a
    fu, fl, d = ub[start], lb[start], 1.0
    line[start], direction[start] = fl, d
    for i in range(start + 1, size):
        fu = ub[i] if (ub[i] < fu or c[i - 1] > fu) else fu          # bands only ratchet towards price
        fl = lb[i] if (lb[i] > fl or c[i - 1] < fl) else fl
        if d > 0 and c[i] < fl:
            d = -1.0
        elif d < 0 and c[i] > fu:
            d = 1.0
        direction[i], line[i] = d, (fl if d > 0 else fu)
    return {"line": line, "direction": direction}


def ichimoku(high, low, close, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> Dict[str, np.ndarray]:
    """Ichimoku cloud, causal form: ``span_a`` / ``span_b`` at bar t are the values *drawn* at t (computed ``kijun``
    bars earlier). The lagging (chikou) span is price shifted backwards in time, i.e. future data, so it is not returned."""
    h, l = _a(high), _a(low)
    mid = lambda n: (rolling_max(h, n) + rolling_min(l, n)) / 2.0                    # noqa: E731
    ten, kij = mid(tenkan), mid(kijun)
    shift = _period(kijun)

    def forward(x: np.ndarray) -> np.ndarray:
        out = np.full(len(x), np.nan)
        if len(x) > shift:
            out[shift:] = x[:-shift]
        return out
    span_a, span_b = forward((ten + kij) / 2.0), forward(mid(senkou_b))
    c = _a(close)
    top, bottom = np.maximum(span_a, span_b), np.minimum(span_a, span_b)
    with np.errstate(invalid="ignore"):
        position = np.where(c > top, 1.0, np.where(c < bottom, -1.0, 0.0))        # +1 above the cloud, -1 below, 0 inside
    position[np.isnan(top)] = np.nan
    return {"tenkan": ten, "kijun": kij, "span_a": span_a, "span_b": span_b, "cloud_position": position}


def parabolic_sar(high, low, step: float = 0.02, max_step: float = 0.2) -> Dict[str, np.ndarray]:
    """Wilder's stop-and-reverse. ``direction`` +1 when the SAR is below price."""
    h, l = _a(high), _a(low)
    size = len(h)
    sar, direction = np.full(size, np.nan), np.full(size, np.nan)
    if size < 2:
        return {"sar": sar, "direction": direction}
    up = h[1] + l[1] >= h[0] + l[0]
    ep = h[1] if up else l[1]
    cur = l[0] if up else h[0]
    af = step
    sar[1], direction[1] = cur, 1.0 if up else -1.0
    for i in range(2, size):
        cur = cur + af * (ep - cur)
        if up:
            cur = min(cur, l[i - 1], l[i - 2])                            # never inside the last two bars
            if l[i] < cur:                                                # reversal
                up, cur, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_step)
        else:
            cur = max(cur, h[i - 1], h[i - 2])
            if h[i] > cur:
                up, cur, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, max_step)
        sar[i], direction[i] = cur, 1.0 if up else -1.0
    return {"sar": sar, "direction": direction}


def aroon(high, low, n: int = 25) -> Dict[str, np.ndarray]:
    """How recently the n-bar high / low was made (100 = this bar, 0 = n bars ago)."""
    h, l, n = _a(high), _a(low), _period(n)
    wh, wl = _windows(h, n + 1), _windows(l, n + 1)
    if not len(wh):
        nan = np.full(len(h), np.nan)
        return {"up": nan, "down": nan.copy(), "oscillator": nan.copy()}
    since_high = np.argmax(wh[:, ::-1], axis=1)                           # reversed window: index 0 is the current bar
    since_low = np.argmin(wl[:, ::-1], axis=1)
    up = _pad(100.0 * (n - since_high) / n, len(h), n + 1)
    down = _pad(100.0 * (n - since_low) / n, len(h), n + 1)
    return {"up": up, "down": down, "oscillator": up - down}


def trix(x, n: int = 15) -> np.ndarray:
    """1-bar rate of change (in %) of a triple-smoothed EMA."""
    e3 = ema_from_first_valid(ema_from_first_valid(ema(x, n), n), n)
    out = np.full(len(e3), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = 100.0 * (e3[1:] / e3[:-1] - 1.0)
    return out


def linreg(x, n: int = 20) -> Dict[str, np.ndarray]:
    """Rolling least-squares line over the last n bars: slope per bar, fitted value at the last bar, and R^2
    (R^2 near 1 with a positive slope = a clean uptrend)."""
    x, n = _a(x), _period(n)
    if n < 3:
        raise ValidationError("linreg period must be >= 3")
    w = _windows(x, n)
    size = len(x)
    if not len(w):
        nan = np.full(size, np.nan)
        return {"slope": nan, "value": nan.copy(), "r2": nan.copy()}
    t = np.arange(n, dtype=float)
    tc = t - t.mean()
    wm = w.mean(axis=1, keepdims=True)
    slope = (w - wm) @ tc / (tc @ tc)
    value = wm[:, 0] + slope * tc[-1]
    ss_tot = ((w - wm) ** 2).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(ss_tot > 0, slope ** 2 * (tc @ tc) / ss_tot, 0.0)
    return {"slope": _pad(slope, size, n), "value": _pad(value, size, n), "r2": _pad(r2, size, n)}


def vortex(high, low, close, n: int = 14) -> Dict[str, np.ndarray]:
    h, l, c = _a(high), _a(low), _a(close)
    size = len(c)
    vm_plus, vm_minus = np.zeros(size), np.zeros(size)
    vm_plus[1:], vm_minus[1:] = np.abs(h[1:] - l[:-1]), np.abs(l[1:] - h[:-1])
    tr = rolling_sum(true_range(h, l, c), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus, minus = rolling_sum(vm_plus, n) / tr, rolling_sum(vm_minus, n) / tr
    if size:
        plus[:n], minus[:n] = np.nan, np.nan                              # the first bar has no previous bar to compare with
    return {"plus": plus, "minus": minus}
