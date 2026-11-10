"""Causal feature engineering: every value at row t uses data up to and including t only."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from ..core.exceptions import ValidationError
from ..quant import indicators as ind
from ..stats.tests import adf_test


def ffd_weights(d: float, thresh: float = 1e-4, max_len: int = 10_000) -> np.ndarray:
    """Weights of the fixed-width-window fractional difference (oldest first)."""
    w = [1.0]
    k = 1
    while k < max_len:
        nxt = -w[-1] * (d - k + 1) / k
        if abs(nxt) < thresh:
            break
        w.append(nxt)
        k += 1
    return np.asarray(w[::-1])


def frac_diff(series: pd.Series, d: float, thresh: float = 1e-4) -> pd.Series:
    """Fractional differentiation: d=0 is the series itself, d=1 the first difference. A value between 0 and 1
    makes prices stationary while keeping much of their long memory (plain returns throw it all away)."""
    if d < 0:
        raise ValidationError("d must be >= 0")
    w = ffd_weights(d, thresh)
    x = series.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    width = len(w)
    if len(x) >= width:
        out[width - 1:] = np.convolve(x, w[::-1], mode="valid")
    return pd.Series(out, index=series.index, name=series.name)


def min_stationary_d(series: pd.Series, grid: Sequence[float] = tuple(np.round(np.arange(0.0, 1.01, 0.1), 2)),
                     thresh: float = 1e-4) -> Dict[str, float]:
    """Smallest d on the grid for which the ADF test rejects a unit root at 5%."""
    for d in grid:
        s = frac_diff(series, d, thresh).dropna()
        if len(s) > 30 and adf_test(s.to_numpy(), regression="c")["reject_5pct"]:
            return {"d": float(d), "corr_with_original": float(np.corrcoef(s, series.reindex(s.index))[0, 1])}
    return {"d": float(grid[-1]), "corr_with_original": float("nan")}


FEATURE_NAMES = ["ret_1", "ret_5", "ret_10", "ret_20", "vol_20", "rsi_14", "z_20", "atr_pct", "macd_hist_pct",
                 "donchian_pos", "vol_ratio", "fd_logprice"]


def build_features(df: pd.DataFrame, fd_d: float = 0.4, fd_thresh: float = 1e-3) -> pd.DataFrame:
    """OHLCV frame -> feature frame (same index). Warm-up rows are NaN. Pure function of the past.

    ``fd_thresh`` 1e-3 gives a ~55-bar fractional-difference window; the textbook 1e-4 needs ~280 bars of warm-up."""
    need = {"open", "high", "low", "close", "volume"}
    if not need <= set(df.columns):
        raise ValidationError(f"missing columns {sorted(need - set(df.columns))}")
    c, h, l, v = (df[k].to_numpy(dtype=float) for k in ("close", "high", "low", "volume"))
    s = df["close"]
    f = pd.DataFrame(index=df.index)
    for n in (1, 5, 10, 20):
        f[f"ret_{n}"] = s.pct_change(n)
    f["vol_20"] = s.pct_change().rolling(20).std()
    f["rsi_14"] = ind.rsi(c, 14) / 100.0
    f["z_20"] = ind.zscore(c, 20)
    f["atr_pct"] = ind.atr(h, l, c, 14) / c
    f["macd_hist_pct"] = ind.macd(c)["hist"] / c
    hi, lo = ind.rolling_max(h, 20), ind.rolling_min(l, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        f["donchian_pos"] = (c - lo) / (hi - lo)
        f["vol_ratio"] = v / ind.sma(v, 20)
    f["fd_logprice"] = frac_diff(np.log(s), fd_d, fd_thresh)
    return f.replace([np.inf, -np.inf], np.nan)


class Scaler:
    """Standardise + winsorise. Fit on the TRAIN rows only; ``transform`` then applies the train statistics to
    anything else, so test-period information can never leak into the scale."""

    def __init__(self, clip: float = 5.0) -> None:
        self.clip, self.mean_, self.std_ = clip, None, None

    def fit(self, X: np.ndarray) -> "Scaler":
        X = np.asarray(X, float)
        self.mean_ = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=1)
        self.std_ = np.where(sd > 1e-12, sd, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise ValidationError("scaler not fitted")
        return np.clip((np.asarray(X, float) - self.mean_) / self.std_, -self.clip, self.clip)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
