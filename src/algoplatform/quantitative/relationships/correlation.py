"""Co-movement between instruments: correlation in its several forms, beta, and how both behave in the tails.

Inputs are *returns* (never price levels: two unrelated trending prices correlate almost perfectly).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ...foundation.core.exceptions import StatisticsError
from ..stats.regression import ols


def _aligned(a, b, min_len: int = 10):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b):
        raise StatisticsError("series must have equal length", details={"a": len(a), "b": len(b)})
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_len:
        raise StatisticsError(f"need at least {min_len} overlapping observations", details={"n": int(ok.sum())})
    return a[ok], b[ok]


def correlation_matrix(returns: pd.DataFrame, method: str = "pearson", min_periods: int = 30) -> pd.DataFrame:
    """Pairwise correlation. ``spearman`` (ranks) is robust to outliers and non-linear but monotone relations."""
    if method not in ("pearson", "spearman", "kendall"):
        raise StatisticsError(f"unknown correlation method '{method}'")
    if returns.shape[1] < 2:
        raise StatisticsError("need at least two series")
    return returns.corr(method=method, min_periods=min_periods)          # type: ignore[arg-type]  # validated just above


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 60) -> pd.Series:
    return a.rolling(window).corr(b)


def ewma_correlation(a, b, lam: float = 0.94) -> np.ndarray:
    """Exponentially weighted correlation: reacts faster than a rolling window when the relationship changes."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b) or not 0 < lam < 1:
        raise StatisticsError("series must have equal length and 0 < lam < 1")
    out = np.full(len(a), np.nan)
    va = vb = cov = 0.0
    started = False
    for t in range(len(a)):
        if not (np.isfinite(a[t]) and np.isfinite(b[t])):
            continue
        if not started:
            va, vb, cov, started = float(a[t] ** 2), float(b[t] ** 2), float(a[t] * b[t]), True
        else:
            va, vb, cov = lam * va + (1 - lam) * a[t] ** 2, lam * vb + (1 - lam) * b[t] ** 2, lam * cov + (1 - lam) * a[t] * b[t]
            out[t] = cov / np.sqrt(va * vb) if va > 0 and vb > 0 else np.nan
    return out


def partial_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """Correlation of each pair after removing what all the *other* series explain (from the precision matrix).
    Two stocks that correlate only because both follow the index have a partial correlation near zero."""
    clean = returns.dropna()
    if len(clean) <= clean.shape[1] + 1:
        raise StatisticsError("need more observations than series for partial correlation")
    prec = np.linalg.pinv(np.cov(clean.to_numpy(), rowvar=False))
    d = np.sqrt(np.diag(prec))
    pc = -prec / np.outer(d, d)
    np.fill_diagonal(pc, 1.0)
    return pd.DataFrame(pc, index=returns.columns, columns=returns.columns)


def beta(asset, market) -> Dict[str, float]:
    """Market model ``asset = alpha + beta * market``, plus beta measured separately on up and down market days."""
    a, m = _aligned(asset, market, 20)
    fit = ols(a, m, names=["market"])
    up, down = m > 0, m < 0

    def side(mask: np.ndarray) -> float:
        return float(np.cov(a[mask], m[mask])[0, 1] / m[mask].var(ddof=1)) if mask.sum() > 5 and m[mask].var(ddof=1) > 0 else float("nan")
    return {"beta": fit["market"], "alpha": fit["const"], "r2": fit.r2, "correlation": float(np.corrcoef(a, m)[0, 1]),
            "beta_up": side(up), "beta_down": side(down), "n": int(len(a))}


def rolling_beta(asset: pd.Series, market: pd.Series, window: int = 60) -> pd.Series:
    return asset.rolling(window).cov(market) / market.rolling(window).var()


def tail_dependence(a, b, q: float = 0.05) -> Dict[str, float]:
    """How often both are in their own worst (best) ``q`` together, relative to how often one is. Independent series give
    about ``q``; 1.0 means they always crash together. Correlation understates this when diversification fails in a crisis."""
    a, b = _aligned(a, b, 50)
    if not 0 < q < 0.5:
        raise StatisticsError("q must be in (0, 0.5)")
    lo = (a <= np.quantile(a, q)) & (b <= np.quantile(b, q))
    hi = (a >= np.quantile(a, 1 - q)) & (b >= np.quantile(b, 1 - q))
    return {"lower": float(lo.mean() / q), "upper": float(hi.mean() / q)}


def stress_correlation(a, b, market: Optional[np.ndarray] = None, q: float = 0.1) -> Dict[str, float]:
    """Correlation on the market's worst ``q`` of days versus all other days (``market`` defaults to the pair's average)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = (a + b) / 2.0 if market is None else np.asarray(market, float)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(m)
    a, b, m = a[ok], b[ok], m[ok]
    stress = m <= np.quantile(m, q)
    if stress.sum() < 5 or (~stress).sum() < 5:
        raise StatisticsError("not enough observations in one of the regimes")
    return {"stress": float(np.corrcoef(a[stress], b[stress])[0, 1]), "calm": float(np.corrcoef(a[~stress], b[~stress])[0, 1]),
            "n_stress": int(stress.sum())}


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Metric distance sqrt((1 - rho) / 2): 0 for identical series, 1 for perfectly opposite ones."""
    d = np.sqrt(np.clip((1.0 - corr.to_numpy()) / 2.0, 0.0, 1.0))
    np.fill_diagonal(d, 0.0)
    return pd.DataFrame(d, index=corr.index, columns=corr.columns)


def average_correlation(corr: pd.DataFrame) -> float:
    """Mean off-diagonal correlation: one number for "how diversified is this set"."""
    c = corr.to_numpy()
    n = len(c)
    return float((np.nansum(c) - np.nansum(np.diag(c))) / (n * (n - 1))) if n > 1 else float("nan")
