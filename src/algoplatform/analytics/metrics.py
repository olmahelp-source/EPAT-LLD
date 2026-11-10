"""Performance metrics from return series, equity curves and trades."""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..core.exceptions import ValidationError
from ..stats.tests import excess_kurtosis, skewness
from ..utils.helpers import safe_div


def to_returns(equity: pd.Series) -> pd.Series:
    """Simple per-bar returns from an equity curve."""
    if len(equity) < 2:
        return pd.Series(dtype=float)
    return equity.pct_change().dropna()


def infer_periods_per_year(index: pd.Index) -> float:
    """Estimate annualisation factor from the median spacing of a DatetimeIndex."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return 252.0
    days = np.median(np.diff(index.values).astype("timedelta64[s]").astype(float)) / 86400.0
    if days <= 0:
        return 252.0
    if days >= 25:
        return 12.0
    if days >= 6:
        return 52.0
    if days >= 0.9:
        return 252.0
    return 252.0 * (6.5 / 24.0) / days if days < 0.5 else 252.0


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown_series(equity).min()) if len(equity) else 0.0


def drawdown_periods(equity: pd.Series) -> pd.DataFrame:
    """Each underwater episode: start, trough date, recovery date (NaT if unrecovered), depth, length."""
    dd = drawdown_series(equity)
    rows, start, trough = [], None, None
    for ts, v in dd.items():
        if v < 0:
            if start is None:
                start, trough = ts, ts
            if v < dd[trough]:
                trough = ts
        elif start is not None:
            rows.append((start, trough, ts, float(dd[trough]), int(((dd.index >= start) & (dd.index < ts)).sum())))
            start = trough = None
    if start is not None:
        rows.append((start, trough, pd.NaT, float(dd[trough]), int((dd.index >= start).sum())))
    return pd.DataFrame(rows, columns=["start", "trough", "recovery", "depth", "bars"])


def annualized_return(equity: pd.Series, periods_per_year: float) -> float:
    n = len(equity) - 1
    if n <= 0 or equity.iloc[0] <= 0:
        return 0.0
    ratio = equity.iloc[-1] / equity.iloc[0]
    return float(ratio ** (periods_per_year / n) - 1.0) if ratio > 0 else -1.0


def sharpe(returns: pd.Series, periods_per_year: float, rf: float = 0.0) -> float:
    r = returns.dropna() - rf / periods_per_year
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd * math.sqrt(periods_per_year)) if sd and sd > 0 else 0.0


def sortino(returns: pd.Series, periods_per_year: float, mar: float = 0.0) -> float:
    r = returns.dropna() - mar / periods_per_year
    downside = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2))) if len(r) else 0.0
    return float(r.mean() / downside * math.sqrt(periods_per_year)) if downside > 0 else 0.0


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    r = returns.dropna() - threshold
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return float(gains / losses) if losses > 0 else math.inf if gains > 0 else 0.0


def tail_ratio(returns: pd.Series) -> float:
    r = returns.dropna()
    lo = abs(np.percentile(r, 5)) if len(r) else 0.0
    return float(np.percentile(r, 95) / lo) if lo > 0 else 0.0


def ulcer_index(equity: pd.Series) -> float:
    dd = drawdown_series(equity) * 100
    return float(math.sqrt(np.mean(dd ** 2))) if len(dd) else 0.0


def performance_metrics(equity: pd.Series, periods_per_year: Optional[float] = None, rf: float = 0.0,
                        benchmark: Optional[pd.Series] = None) -> Dict[str, float]:
    """The standard set. ``benchmark`` is a *returns* series aligned by index."""
    if len(equity) < 3:
        raise ValidationError("need at least 3 equity points to compute metrics")
    ppy = periods_per_year or infer_periods_per_year(equity.index)
    r = to_returns(equity)
    dd = drawdown_series(equity)
    ann_ret = annualized_return(equity, ppy)
    vol = float(r.std(ddof=1) * math.sqrt(ppy)) if len(r) > 1 else 0.0
    mdd = float(dd.min())
    out = {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": ann_ret, "volatility": vol, "sharpe": sharpe(r, ppy, rf), "sortino": sortino(r, ppy),
        "max_drawdown": mdd, "calmar": safe_div(ann_ret, abs(mdd)),
        "ulcer_index": ulcer_index(equity), "omega": omega_ratio(r), "tail_ratio": tail_ratio(r),
        "skew": skewness(r) if len(r) > 3 else 0.0, "kurtosis": excess_kurtosis(r) if len(r) > 4 else 0.0,
        "var_95": float(max(-np.percentile(r, 5), 0.0)) if len(r) else 0.0,
        "cvar_95": float(max(-r[r <= np.percentile(r, 5)].mean(), 0.0)) if len(r) else 0.0,
        "best_bar": float(r.max()), "worst_bar": float(r.min()),
        "pct_positive_bars": float((r > 0).mean()), "bars": int(len(equity)), "periods_per_year": float(ppy),
        "longest_drawdown_bars": int(drawdown_periods(equity)["bars"].max()) if (dd < 0).any() else 0,
    }
    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        rr = r.reindex(b.index)
        if len(b) > 5 and b.var() > 0:
            beta = float(np.cov(rr, b, ddof=1)[0, 1] / b.var(ddof=1))
            alpha = float((rr.mean() - beta * b.mean()) * ppy)
            te = float((rr - b).std(ddof=1) * math.sqrt(ppy))
            out.update(beta=beta, alpha=alpha, correlation=float(np.corrcoef(rr, b)[0, 1]), tracking_error=te,
                       information_ratio=safe_div(float((rr - b).mean() * ppy), te))
            up, dn = b > 0, b < 0
            out["up_capture"] = safe_div(float(rr[up].mean()), float(b[up].mean())) if up.any() else 0.0
            out["down_capture"] = safe_div(float(rr[dn].mean()), float(b[dn].mean())) if dn.any() else 0.0
    return out


def rolling_sharpe(returns: pd.Series, window: int, periods_per_year: float) -> pd.Series:
    m, s = returns.rolling(window).mean(), returns.rolling(window).std(ddof=1)
    return (m / s.replace(0, np.nan)) * math.sqrt(periods_per_year)


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """Year x month table of compounded returns."""
    r = to_returns(equity)
    if r.empty or not isinstance(r.index, pd.DatetimeIndex):
        return pd.DataFrame()
    m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    table = m.unstack()
    table.index.name, table.columns.name = "year", "month"
    return table
