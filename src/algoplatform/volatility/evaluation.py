"""Scoring volatility forecasts out of sample. True variance is never observed, so forecasts are compared with a noisy
proxy (squared return or a range-based estimate); QLIKE is the loss that ranks models correctly despite that noise."""
from __future__ import annotations

from typing import Callable, Dict, Mapping

import numpy as np
import pandas as pd

from ..core.exceptions import PlatformError, StatisticsError
from ..stats.regression import ols
from .models import VolModel


def _pair(realized, forecast):
    r, f = np.asarray(realized, float), np.asarray(forecast, float)
    ok = np.isfinite(r) & np.isfinite(f) & (f > 0) & (r >= 0)
    if ok.sum() < 5:
        raise StatisticsError("need at least 5 valid (realized, forecast) pairs")
    return r[ok], f[ok]


def qlike(realized_var, forecast_var) -> float:
    """Mean of r/f - log(r/f) - 1. Zero for a perfect forecast; punishes under-prediction more than over-prediction,
    which matches the cost of being under-hedged."""
    r, f = _pair(realized_var, forecast_var)
    ratio = np.clip(r / f, 1e-12, None)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mse(realized_var, forecast_var) -> float:
    r, f = _pair(realized_var, forecast_var)
    return float(np.mean((r - f) ** 2))


def mincer_zarnowitz(realized_var, forecast_var) -> Dict[str, float]:
    """Regress realised on forecast. An unbiased forecast has intercept 0 and slope 1; R^2 is the share explained."""
    r, f = _pair(realized_var, forecast_var)
    fit = ols(r, f, names=["forecast"])
    return {"intercept": fit["const"], "slope": fit["forecast"], "r2": fit.r2, "n": fit.n}


def walk_forward_forecasts(returns, factory: Callable[[], VolModel], train: int = 500, step: int = 20, horizon: int = 1,
                           expanding: bool = True) -> pd.DataFrame:
    """Refit every ``step`` bars on data up to t, forecast the next ``horizon`` bars, compare with what happened.
    Returns one row per forecast origin: ``forecast_var`` and ``realized_var`` (mean squared return over the horizon)."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < train + horizon + 1 or step < 1 or horizon < 1:
        raise StatisticsError("need len(returns) > train + horizon and step, horizon >= 1")
    rows, model = [], None
    for t in range(train, len(r) - horizon + 1):
        if model is None or (t - train) % step == 0:
            model = factory().fit(r[:t] if expanding else r[t - train:t])
            origin = t
        if t != origin:
            continue                                        # one forecast per refit keeps origins non-overlapping in information
        rows.append({"t": t, "forecast_var": float(model.forecast(horizon).variance_path.mean()),
                     "realized_var": float(np.mean(r[t:t + horizon] ** 2))})
    return pd.DataFrame(rows).set_index("t")


def compare_vol_models(returns, factories: Mapping[str, Callable[[], VolModel]], train: int = 500, step: int = 20,
                       horizon: int = 1) -> pd.DataFrame:
    """Out-of-sample league table (best QLIKE first). A model that fails to fit is listed with its error, not dropped."""
    rows = []
    for name, factory in factories.items():
        try:
            wf = walk_forward_forecasts(returns, factory, train, step, horizon)
            mz = mincer_zarnowitz(wf["realized_var"], wf["forecast_var"])
            rows.append({"model": name, "qlike": qlike(wf["realized_var"], wf["forecast_var"]), "mse": mse(wf["realized_var"], wf["forecast_var"]),
                         "mz_slope": mz["slope"], "mz_r2": mz["r2"], "forecasts": len(wf), "error": None})
        except PlatformError as exc:
            rows.append({"model": name, "qlike": np.nan, "mse": np.nan, "mz_slope": np.nan, "mz_r2": np.nan, "forecasts": 0, "error": exc.message})
    return pd.DataFrame(rows).sort_values("qlike", na_position="last").reset_index(drop=True)
