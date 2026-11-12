"""Volatility management: turning a volatility view into how much to hold.

The core idea is volatility targeting: hold more when the market is quiet and less when it is wild, so that the
position's *risk* stays roughly constant instead of its notional. On top of that, a regime classifier cuts exposure
further when volatility is extreme by its own history.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

from ..core.exceptions import ValidationError
from .models import make_model


def vol_target_scalar(forecast_vol: float, target_vol: float, max_leverage: float = 2.0, min_scalar: float = 0.0) -> float:
    """Exposure multiplier = target / forecast, clipped to [min_scalar, max_leverage]. An unusable forecast gives the
    floor (when volatility is unknown, be small)."""
    if target_vol <= 0 or max_leverage <= 0 or min_scalar < 0:
        raise ValidationError("target_vol and max_leverage must be positive, min_scalar >= 0")
    if not (isinstance(forecast_vol, (int, float)) and math.isfinite(forecast_vol) and forecast_vol > 0):
        return float(min_scalar)
    return float(min(max(target_vol / forecast_vol, min_scalar), max_leverage))


def vol_target_weights(returns: pd.Series, target_vol: float = 0.10, window: int = 20, max_leverage: float = 2.0,
                       periods: float = 252.0) -> pd.Series:
    """Exposure series for a volatility-targeted version of a return stream. The weight applied to bar t's return is
    computed from returns up to t-1 only (shifted by one bar), so ``returns * weights`` is a fair backtest."""
    if window < 2:
        raise ValidationError("window must be >= 2")
    realized = returns.rolling(window).std() * math.sqrt(periods)
    w = (target_vol / realized.where(realized > 0)).clip(upper=max_leverage)
    return w.shift(1)


def variance_risk_premium(implied_vol: float, realized_vol: float) -> float:
    """Implied minus realised variance. Usually positive: option sellers are paid for bearing volatility risk."""
    return implied_vol ** 2 - realized_vol ** 2


class VolRegime:
    """Label each bar low / normal / high by where volatility sits in its own trailing ``lookback`` distribution.

    Hysteresis: once in a regime, the percentile must retreat by ``buffer`` past the cut to leave it, which stops the
    label (and the position size hanging off it) flapping when volatility hovers at a threshold."""
    LABELS = ("low", "normal", "high")

    def __init__(self, lookback: int = 252, low_cut: float = 0.25, high_cut: float = 0.80, buffer: float = 0.05, min_history: int = 60) -> None:
        if not 0 < low_cut < high_cut < 1 or not 0 <= buffer < min(low_cut, 1 - high_cut):
            raise ValidationError("need 0 < low_cut < high_cut < 1 and a buffer smaller than both tails")
        self.lookback, self.low_cut, self.high_cut, self.buffer, self.min_history = lookback, low_cut, high_cut, buffer, min_history

    def percentiles(self, vol: Sequence[float]) -> np.ndarray:
        """Share of the previous ``lookback`` volatilities below the current one (NaN until ``min_history`` exist)."""
        v = np.asarray(vol, float)
        out = np.full(len(v), np.nan)
        for t in range(len(v)):
            hist = v[max(0, t - self.lookback):t]
            hist = hist[np.isfinite(hist)]
            if np.isfinite(v[t]) and len(hist) >= self.min_history:
                out[t] = float((hist < v[t]).mean())
        return out

    def classify(self, vol: Sequence[float]) -> np.ndarray:
        pct = self.percentiles(vol)
        out = np.empty(len(pct), dtype=object)
        state = "normal"
        for t, p in enumerate(pct):
            if not np.isfinite(p):
                out[t] = state = "normal"
                continue
            if state == "high":
                state = "high" if p >= self.high_cut - self.buffer else ("low" if p <= self.low_cut else "normal")
            elif state == "low":
                state = "low" if p <= self.low_cut + self.buffer else ("high" if p >= self.high_cut else "normal")
            else:
                state = "high" if p >= self.high_cut else ("low" if p <= self.low_cut else "normal")
            out[t] = state
        return out


class VolatilityManager:
    """One call that answers "given these returns, how much exposure should I run".

    ``assess`` fits the chosen forecaster, classifies the regime and returns the exposure scalar: the volatility-target
    multiplier, further multiplied by ``regime_scale[regime]`` (by default halved when volatility is in its top fifth)."""

    def __init__(self, target_vol: float = 0.10, model: str = "ewma", horizon: int = 5, max_leverage: float = 2.0,
                 regime: VolRegime = None, regime_scale: Dict[str, float] = None, periods: float = 252.0, **model_kwargs: Any) -> None:
        self.target_vol, self.model, self.horizon, self.max_leverage, self.periods = target_vol, model, horizon, max_leverage, periods
        self.regime = regime or VolRegime()
        self.regime_scale = {"low": 1.0, "normal": 1.0, "high": 0.5, **(regime_scale or {})}
        self.model_kwargs = model_kwargs
        make_model(model, periods=periods, **model_kwargs)              # fail at construction, not on the first bar

    def assess(self, returns: Sequence[float]) -> Dict[str, Any]:
        fitted = make_model(self.model, periods=self.periods, **self.model_kwargs).fit(returns)
        fc = fitted.forecast(self.horizon)
        history = fitted.conditional_vol()
        regime = str(self.regime.classify(np.append(history, fc.next_vol))[-1])
        pct = float(self.regime.percentiles(np.append(history, fc.next_vol))[-1])
        base = vol_target_scalar(fc.vol, self.target_vol, self.max_leverage)
        return {"forecast_vol": fc.vol, "next_vol": fc.next_vol, "regime": regime, "percentile": pct, "base_scalar": base,
                "scalar": base * self.regime_scale[regime], "model": fitted.name, "params": fc.params}
