"""Who moves first: cross-correlation at leads and lags, and Granger causality."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd

from ..core.exceptions import StatisticsError
from ..stats.special import chi2_sf, norm_ppf
from .correlation import _aligned


def cross_correlation(a, b, max_lag: int = 10) -> pd.Series:
    """corr(a[t], b[t + lag]) for lag in -max_lag..max_lag. A peak at a positive lag means ``a`` leads ``b``."""
    a, b = _aligned(a, b, max_lag + 10)
    if max_lag < 1:
        raise StatisticsError("max_lag must be >= 1")
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[:len(a) - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[:len(b) + lag])
        out[lag] = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return pd.Series(out, name="correlation").rename_axis("lag")


def lead_lag(a, b, max_lag: int = 10) -> Dict[str, object]:
    """Strongest non-contemporaneous relation. ``significant`` uses the 1.96 / sqrt(n) band, Bonferroni-widened for the
    2 * max_lag lags inspected, so that picking the best of many lags is not mistaken for a finding."""
    cc = cross_correlation(a, b, max_lag).drop(0).dropna()
    if cc.empty:
        raise StatisticsError("cross-correlation undefined (constant series?)")
    best = int(cc.abs().idxmax())
    n = len(_aligned(a, b)[0]) - abs(best)
    band = _bonferroni_z(2 * max_lag) / math.sqrt(max(n, 1))
    sig = bool(abs(cc[best]) > band)
    return {"best_lag": best, "correlation": float(cc[best]), "leader": ("a" if best > 0 else "b") if sig else "none",
            "threshold": float(band), "significant": sig, "n": int(n)}


def _bonferroni_z(m: int) -> float:
    """Two-sided 5% normal critical value when m lags are inspected at once."""
    return float(norm_ppf(1 - 0.05 / (2 * m)))


def granger_causality(y, x, lags: int = 2) -> Dict[str, object]:
    """Does the past of ``x`` improve a forecast of ``y`` beyond y's own past?

    Compares an autoregression of y with the same regression plus ``lags`` lags of x (Wald test, chi-square with
    ``lags`` degrees of freedom). "Causes" here means *predictive precedence*, not economic causation."""
    y, x = _aligned(y, x, 3 * lags + 10)
    if lags < 1:
        raise StatisticsError("lags must be >= 1")
    n = len(y) - lags
    Y = y[lags:]
    own = np.column_stack([np.ones(n)] + [y[lags - k:len(y) - k] for k in range(1, lags + 1)])
    full = np.column_stack([own] + [x[lags - k:len(x) - k] for k in range(1, lags + 1)])

    def rss(A: np.ndarray) -> float:
        beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
        return float(((Y - A @ beta) ** 2).sum())
    rss_r, rss_u = rss(own), rss(full)
    if rss_u <= 0:
        raise StatisticsError("degenerate regression (perfect fit)")
    chi2 = max(n * (rss_r - rss_u) / rss_u, 0.0)
    p = chi2_sf(chi2, lags)
    return {"chi2": float(chi2), "p_value": float(p), "lags": lags, "n": int(n), "causes_5pct": bool(p < 0.05)}


def lead_lag_matrix(returns: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """Cell [i, j] > 0: i leads j by that many bars (significant relations only; 0 otherwise)."""
    cols = list(returns.columns)
    out = pd.DataFrame(0, index=cols, columns=cols, dtype=int)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            try:
                res = lead_lag(returns[a].to_numpy(), returns[b].to_numpy(), max_lag)
            except StatisticsError:
                continue
            if res["significant"]:
                out.loc[a, b], out.loc[b, a] = res["best_lag"], -res["best_lag"]
    return out
