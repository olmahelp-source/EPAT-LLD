"""Signal analytics: does a cross-sectional score actually predict forward returns?"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from ..core.exceptions import ValidationError
from ..stats.special import t_two_sided_p


def forward_returns(close: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Return from t to t+horizon, stored at t. The final ``horizon`` rows are NaN (the future is unknown)."""
    if horizon < 1:
        raise ValidationError("horizon must be >= 1")
    return close.shift(-horizon) / close - 1.0


def information_coefficient(signal: pd.DataFrame, fwd: pd.DataFrame, method: str = "spearman") -> pd.Series:
    """Cross-sectional correlation between today's signal and the forward return, one value per timestamp."""
    if method not in ("spearman", "pearson"):
        raise ValidationError("method must be spearman or pearson")
    s, f = signal.align(fwd, join="inner")
    if method == "spearman":
        s, f = s.rank(axis=1), f.rank(axis=1)
    mask = s.notna() & f.notna()
    s, f = s.where(mask), f.where(mask)
    n = mask.sum(axis=1)
    sm, fm = s.sub(s.mean(axis=1), axis=0), f.sub(f.mean(axis=1), axis=0)
    cov = (sm * fm).sum(axis=1)
    denom = np.sqrt((sm ** 2).sum(axis=1) * (fm ** 2).sum(axis=1))
    ic = (cov / denom.replace(0, np.nan)).where(n >= 3)
    return ic.dropna().rename("ic")


def ic_summary(ic: pd.Series) -> Dict[str, float]:
    ic = ic.dropna()
    n = len(ic)
    if n < 3:
        raise ValidationError("need at least 3 IC observations")
    mean, sd = ic.mean(), ic.std(ddof=1)
    t = mean / (sd / np.sqrt(n)) if sd > 0 else 0.0
    return {"ic_mean": float(mean), "ic_std": float(sd), "ic_ir": float(mean / sd) if sd > 0 else 0.0,
            "t_stat": float(t), "p_value": t_two_sided_p(float(t), n - 1), "hit_rate": float((ic > 0).mean()), "n": n}


def ic_decay(signal: pd.DataFrame, close: pd.DataFrame, horizons: Sequence[int] = (1, 2, 5, 10, 20)) -> pd.Series:
    """Mean IC as the prediction horizon grows. Fast decay means the signal must be traded quickly (costs matter)."""
    return pd.Series({h: float(information_coefficient(signal, forward_returns(close, h)).mean()) for h in horizons},
                     name="mean_ic")


def quantile_returns(signal: pd.DataFrame, fwd: pd.DataFrame, quantiles: int = 5) -> Dict[str, object]:
    """Average forward return by signal quantile (1 = lowest). A usable signal is monotonic across quantiles."""
    s, f = signal.align(fwd, join="inner")
    ranks = s.rank(axis=1, pct=True)
    bucket = np.ceil(ranks * quantiles).clip(1, quantiles)
    by_q = {q: float(f.where(bucket == q).stack().mean()) for q in range(1, quantiles + 1)}
    series = pd.Series(by_q, name="mean_fwd_return")
    mono = float(series.rank().corr(pd.Series(range(1, quantiles + 1), index=series.index), method="pearson"))
    return {"by_quantile": series, "spread": float(series.iloc[-1] - series.iloc[0]), "monotonicity": mono}


def signal_autocorrelation(signal: pd.DataFrame, lag: int = 1) -> float:
    """Average cross-sectional rank autocorrelation. Near 1 = slow, cheap-to-trade signal; near 0 = high turnover."""
    r = signal.rank(axis=1)
    return float(r.corrwith(r.shift(lag), axis=1).mean())


def signal_turnover(signal: pd.DataFrame) -> float:
    return 1.0 - signal_autocorrelation(signal)


def signal_correlation(signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average cross-sectional rank correlation between signals: highly correlated signals add little in a blend."""
    names = list(signals)
    out = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x, y = signals[a].align(signals[b], join="inner")
            c = float(x.rank(axis=1).corrwith(y.rank(axis=1), axis=1).mean())
            out.loc[a, b] = out.loc[b, a] = c
    return out
