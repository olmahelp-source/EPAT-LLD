"""Technical indicators. All are causal: value at index t uses data up to and including t.

Warm-up positions are NaN. Functions take array-likes and return numpy arrays of equal
length, so they can be used both in vectorised research and bar-by-bar strategies.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from ..core.exceptions import ValidationError


def _a(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _period(n: int) -> int:
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValidationError("period must be a positive integer", details={"period": n})
    return int(n)


def _windows(x: np.ndarray, n: int) -> np.ndarray:
    """Rolling windows as a (len-n+1, n) view (no copy); empty if the series is shorter than n."""
    return np.lib.stride_tricks.sliding_window_view(x, n) if len(x) >= n else np.empty((0, n))


def _pad(values: np.ndarray, size: int, n: int) -> np.ndarray:
    out = np.full(size, np.nan)
    if len(values):
        out[n - 1:] = values
    return out


def sma(x, n: int) -> np.ndarray:
    """Simple moving average via cumulative sums: O(len), and value[t] depends only on x[:t+1]."""
    x, n = _a(x), _period(n)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        if np.isnan(x).any():                              # NaNs poison cumsum; fall back to windows
            return _pad(_windows(x, n).mean(axis=1), len(x), n)
        c = np.cumsum(np.concatenate([[0.0], x]))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def ema(x, n: int) -> np.ndarray:
    """EMA seeded with the SMA of the first ``n`` values (matches common charting platforms)."""
    x, n = _a(x), _period(n)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    alpha = 2.0 / (n + 1)
    out[n - 1] = np.nanmean(x[:n])
    for i in range(n, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def wma(x, n: int) -> np.ndarray:
    x, n = _a(x), _period(n)
    w = np.arange(1, n + 1, dtype=float)
    return _pad(_windows(x, n) @ w / w.sum(), len(x), n)


def rolling_std(x, n: int, ddof: int = 1) -> np.ndarray:
    x, n = _a(x), _period(n)
    if n - ddof <= 0:
        return np.full(len(x), np.nan)
    return _pad(_windows(x, n).std(axis=1, ddof=ddof), len(x), n)


def rolling_max(x, n: int) -> np.ndarray:
    x, n = _a(x), _period(n)
    return _pad(_windows(x, n).max(axis=1), len(x), n)


def rolling_min(x, n: int) -> np.ndarray:
    x, n = _a(x), _period(n)
    return _pad(_windows(x, n).min(axis=1), len(x), n)


def zscore(x, n: int) -> np.ndarray:
    x, n = _a(x), _period(n)
    w = _windows(x, n)
    if not len(w):
        return np.full(len(x), np.nan)
    sd = w.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x[n - 1:] - w.mean(axis=1)) / np.where(sd == 0, np.nan, sd)
    return _pad(z, len(x), n)


def roc(x, n: int) -> np.ndarray:
    x = _a(x)
    out = np.full(len(x), np.nan)
    n = _period(n)
    if len(x) > n:
        out[n:] = x[n:] / x[:-n] - 1.0
    return out


def rsi(x, n: int = 14) -> np.ndarray:
    """Wilder's RSI."""
    x, n = _a(x), _period(n)
    out = np.full(len(x), np.nan)
    if len(x) <= n:
        return out
    d = np.diff(x)
    up, dn = np.where(d > 0, d, 0.0), np.where(d < 0, -d, 0.0)
    au, ad = up[:n].mean(), dn[:n].mean()
    out[n] = 100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / ad)
    for i in range(n, len(d)):
        au = (au * (n - 1) + up[i]) / n
        ad = (ad * (n - 1) + dn[i]) / n
        out[i + 1] = 100.0 if ad == 0 else 100.0 - 100.0 / (1.0 + au / ad)
    return out


def true_range(high, low, close) -> np.ndarray:
    h, l, c = _a(high), _a(low), _a(close)
    prev = np.concatenate([[np.nan], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])
    tr[0] = h[0] - l[0]
    return tr


def atr(high, low, close, n: int = 14) -> np.ndarray:
    """Wilder's Average True Range."""
    tr, n = true_range(high, low, close), _period(n)
    out = np.full(len(tr), np.nan)
    if len(tr) < n:
        return out
    out[n - 1] = tr[:n].mean()
    for i in range(n, len(tr)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def macd(x, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
    if fast >= slow:
        raise ValidationError("fast period must be shorter than slow")
    line = ema(x, fast) - ema(x, slow)
    sig = np.full(len(line), np.nan)
    valid = ~np.isnan(line)
    if valid.sum() >= signal:
        sig[valid] = ema(line[valid], signal)
    return {"macd": line, "signal": sig, "hist": line - sig}


def bollinger(x, n: int = 20, k: float = 2.0) -> Dict[str, np.ndarray]:
    mid, sd = sma(x, n), rolling_std(x, n, ddof=0)
    up, lo = mid + k * sd, mid - k * sd
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_b = (_a(x) - lo) / (up - lo)
        width = (up - lo) / mid
    return {"mid": mid, "upper": up, "lower": lo, "pct_b": pct_b, "bandwidth": width}


def donchian(high, low, n: int = 20) -> Dict[str, np.ndarray]:
    up, lo = rolling_max(high, n), rolling_min(low, n)
    return {"upper": up, "lower": lo, "mid": (up + lo) / 2}


def stochastic(high, low, close, k: int = 14, d: int = 3) -> Dict[str, np.ndarray]:
    hh, ll, c = rolling_max(high, k), rolling_min(low, k), _a(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        pk = 100 * (c - ll) / (hh - ll)
    return {"k": pk, "d": sma(pk, d)}


def adx(high, low, close, n: int = 14) -> Dict[str, np.ndarray]:
    """Wilder's ADX with +DI / -DI."""
    h, l, c = _a(high), _a(low), _a(close)
    n = _period(n)
    size = len(c)
    out = {k: np.full(size, np.nan) for k in ("adx", "plus_di", "minus_di")}
    if size < 2 * n + 1:
        return out
    up, dn = np.diff(h), -np.diff(l)
    pdm, mdm = np.where((up > dn) & (up > 0), up, 0.0), np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(h, l, c)[1:]

    def wilder(v):
        r = np.full(len(v), np.nan)
        r[n - 1] = v[:n].sum()
        for i in range(n, len(v)):
            r[i] = r[i - 1] - r[i - 1] / n + v[i]
        return r
    str_, spdm, smdm = wilder(tr), wilder(pdm), wilder(mdm)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi, mdi = 100 * spdm / str_, 100 * smdm / str_
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    a = np.full(len(dx), np.nan)
    first = 2 * n - 2
    a[first] = np.nanmean(dx[n - 1:first + 1])
    for i in range(first + 1, len(dx)):
        a[i] = (a[i - 1] * (n - 1) + dx[i]) / n
    out["adx"][1:], out["plus_di"][1:], out["minus_di"][1:] = a, pdi, mdi
    return out


def vwap(price, volume, window: int = 0) -> np.ndarray:
    p, v = _a(price), _a(volume)
    if window:
        num = pd.Series(p * v).rolling(window).sum()
        den = pd.Series(v).rolling(window).sum()
        return (num / den.replace(0, np.nan)).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.cumsum(p * v) / np.cumsum(v)


def obv(close, volume) -> np.ndarray:
    c, v = _a(close), _a(volume)
    sign = np.concatenate([[0.0], np.sign(np.diff(c))])
    return np.cumsum(sign * v)


def crossed_above(a, b) -> bool:
    """True if series ``a`` crossed above ``b`` on the last bar."""
    a, b = _a(a), _a(b)
    if len(a) < 2 or len(b) < 2:
        return False
    return bool(a[-2] <= b[-2] and a[-1] > b[-1])


def crossed_below(a, b) -> bool:
    a, b = _a(a), _a(b)
    if len(a) < 2 or len(b) < 2:
        return False
    return bool(a[-2] >= b[-2] and a[-1] < b[-1])


INDICATORS = {"sma": sma, "ema": ema, "wma": wma, "rsi": rsi, "roc": roc, "zscore": zscore,
              "rolling_std": rolling_std, "rolling_max": rolling_max, "rolling_min": rolling_min}
