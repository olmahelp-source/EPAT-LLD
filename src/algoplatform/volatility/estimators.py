"""Volatility estimators: how much has it been moving.

All return an annualised, rolling series aligned with the bars (NaN during warm-up, causal). Range-based estimators use
the high and low as well as the close, so they extract far more information from each bar than close-to-close:

=================  ==========================  ==================================================================
estimator          uses                        note
=================  ==========================  ==================================================================
close_to_close     close                       the baseline; noisy
parkinson          high, low                   ~5x more efficient; assumes no drift and no gaps (biased low with gaps)
garman_klass       open, high, low, close      ~7x; still ignores overnight gaps
rogers_satchell    open, high, low, close      correct when the price trends
yang_zhang         open, high, low, close      handles gaps and drift; the usual choice for daily bars
=================  ==========================  ==================================================================
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from ..core.exceptions import ValidationError
from ..quant.indicators import _a, _pad, _period, _windows, sma


def _log_returns(c: np.ndarray) -> np.ndarray:
    out = np.full(len(c), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = np.log(c[1:] / c[:-1])
    return out


def _rolling_var(x: np.ndarray, n: int) -> np.ndarray:
    return _pad(_windows(x, n).var(axis=1, ddof=1), len(x), n) if n > 1 else np.full(len(x), np.nan)


def _annualise(variance: np.ndarray, periods: float) -> np.ndarray:
    return np.sqrt(np.clip(variance, 0.0, None) * periods)


def close_to_close(close, n: int = 20, periods: float = 252.0) -> np.ndarray:
    return _annualise(_rolling_var(_log_returns(_a(close)), _period(n)), periods)


def parkinson(high, low, n: int = 20, periods: float = 252.0) -> np.ndarray:
    hl = np.log(_a(high) / _a(low)) ** 2 / (4.0 * math.log(2.0))
    return _annualise(sma(hl, n), periods)


def _gk_bar(o, h, l, c) -> np.ndarray:
    return 0.5 * np.log(h / l) ** 2 - (2.0 * math.log(2.0) - 1.0) * np.log(c / o) ** 2


def _rs_bar(o, h, l, c) -> np.ndarray:
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


def garman_klass(open_, high, low, close, n: int = 20, periods: float = 252.0) -> np.ndarray:
    return _annualise(sma(_gk_bar(_a(open_), _a(high), _a(low), _a(close)), n), periods)


def rogers_satchell(open_, high, low, close, n: int = 20, periods: float = 252.0) -> np.ndarray:
    return _annualise(sma(_rs_bar(_a(open_), _a(high), _a(low), _a(close)), n), periods)


def yang_zhang(open_, high, low, close, n: int = 20, periods: float = 252.0) -> np.ndarray:
    """Overnight variance + a weighted mix of open-to-close variance and Rogers-Satchell."""
    o, h, l, c = _a(open_), _a(high), _a(low), _a(close)
    n = _period(n)
    if n < 3:
        raise ValidationError("yang_zhang needs n >= 3")
    overnight = np.full(len(c), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        overnight[1:] = np.log(o[1:] / c[:-1])
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    variance = _rolling_var(overnight, n) + k * _rolling_var(np.log(c / o), n) + (1 - k) * sma(_rs_bar(o, h, l, c), n)
    return _annualise(variance, periods)


def ewma(close, lam: float = 0.94, periods: float = 252.0) -> np.ndarray:
    """RiskMetrics volatility: yesterday's estimate decayed by ``lam`` plus today's squared return."""
    if not 0 < lam < 1:
        raise ValidationError("lam must be in (0, 1)")
    r = _log_returns(_a(close))
    out = np.full(len(r), np.nan)
    var = None
    for t in range(1, len(r)):
        if not np.isfinite(r[t]):
            continue
        var = r[t] ** 2 if var is None else lam * var + (1 - lam) * r[t] ** 2
        out[t] = math.sqrt(var * periods)
    return out


def variance_proxy(open_, high, low, close, kind: str = "garman_klass") -> np.ndarray:
    """Per-bar variance (not annualised): the input a variance-forecasting model such as HAR is fitted on."""
    o, h, l, c = _a(open_), _a(high), _a(low), _a(close)
    if kind == "squared_return":
        return _log_returns(c) ** 2
    if kind == "parkinson":
        return np.log(h / l) ** 2 / (4.0 * math.log(2.0))
    if kind == "garman_klass":
        return np.clip(_gk_bar(o, h, l, c), 0.0, None)
    raise ValidationError(f"unknown variance proxy '{kind}'", details={"valid": ["squared_return", "parkinson", "garman_klass"]})


def vol_of_vol(vol, n: int = 20) -> np.ndarray:
    """Rolling standard deviation of changes in log volatility: how unstable the volatility itself is."""
    v = _a(vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.diff(np.log(np.where(v > 0, v, np.nan)), prepend=np.nan)
    return np.sqrt(_rolling_var(d, _period(n)))


def volatility_cone(close, windows: Sequence[int] = (10, 21, 63, 126, 252), quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
                    periods: float = 252.0) -> pd.DataFrame:
    """For each horizon: the historical distribution of realised volatility and where it stands now. Compare an option's
    implied volatility with the cone for the matching horizon to judge whether it is rich or cheap by history."""
    rows = {}
    for w in windows:
        v = close_to_close(close, w, periods)
        v = v[np.isfinite(v)]
        if len(v) < 2:
            continue
        rows[w] = {"min": v.min(), **{f"q{int(q * 100):02d}": float(np.quantile(v, q)) for q in quantiles}, "max": v.max(), "current": v[-1],
                   "percentile": float((v[:-1] < v[-1]).mean())}
    if not rows:
        raise ValidationError("not enough history for any of the requested windows")
    return pd.DataFrame(rows).T.rename_axis("window")
