"""Volatility forecasting models: how much will it move.

Every model has the same three calls: ``fit(returns)``, ``forecast(horizon)`` and ``conditional_vol()``. ``forecast``
returns a :class:`VolForecast` whose ``vol`` is the annualised volatility expected on average over the horizon, which
is the number position sizing and option pricing need.

=========  ==================================================================================================
model      idea
=========  ==================================================================================================
hist       volatility stays what it was over the last ``window`` bars
ewma       same, with recent days weighted more (RiskMetrics)
garch      clusters and mean-reverts to a long-run level (Gaussian maximum likelihood)
gjr        garch + leverage effect: a fall raises tomorrow's volatility more than a rise of the same size
har        tomorrow's variance is a blend of the last day, week and month (captures long memory with plain OLS)
=========  ==================================================================================================
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from ...foundation.core.exceptions import ModelError, StatisticsError, ValidationError
from ...foundation.utils.numeric import nelder_mead


@dataclass
class VolForecast:
    model: str
    horizon: int
    variance_path: np.ndarray                 # per-bar variance for bars T+1 .. T+horizon
    periods: float = 252.0
    params: Dict[str, float] = field(default_factory=dict)

    @property
    def vol(self) -> float:
        """Annualised volatility over the whole horizon."""
        return float(math.sqrt(max(self.variance_path.mean(), 0.0) * self.periods))

    @property
    def next_vol(self) -> float:
        return float(math.sqrt(max(self.variance_path[0], 0.0) * self.periods))


def _clean(returns, min_len: int) -> np.ndarray:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < min_len:
        raise StatisticsError(f"need at least {min_len} returns", details={"n": len(r)})
    return r


class VolModel(abc.ABC):
    name = "vol"
    min_obs = 30

    def __init__(self, periods: float = 252.0) -> None:
        self.periods = periods
        self._var: Optional[np.ndarray] = None            # in-sample conditional variance, one per return
        self._r: Optional[np.ndarray] = None
        self.params: Dict[str, float] = {}

    def fit(self, returns) -> "VolModel":
        self._r = _clean(returns, self.min_obs)
        self._fit(self._r)
        return self

    @abc.abstractmethod
    def _fit(self, r: np.ndarray) -> None: ...

    @abc.abstractmethod
    def _path(self, horizon: int) -> np.ndarray: ...

    def forecast(self, horizon: int = 1) -> VolForecast:
        if self._var is None:
            raise ModelError(f"{self.name} model is not fitted")
        if horizon < 1:
            raise ValidationError("horizon must be >= 1")
        return VolForecast(self.name, horizon, np.clip(self._path(int(horizon)), 1e-18, None), self.periods, dict(self.params))

    def conditional_vol(self) -> np.ndarray:
        """Annualised in-sample volatility; element t is the estimate made *before* seeing return t."""
        if self._var is None:
            raise ModelError(f"{self.name} model is not fitted")
        return np.sqrt(self._var * self.periods)


class HistoricalVol(VolModel):
    name = "hist"

    def __init__(self, window: int = 20, periods: float = 252.0) -> None:
        super().__init__(periods)
        self.window, self.min_obs = window, window + 1

    def _fit(self, r: np.ndarray) -> None:
        sq = r ** 2
        c = np.concatenate([[0.0], np.cumsum(sq)])
        var = np.full(len(r), sq[:self.window].mean())
        var[self.window:] = (c[self.window:-1] - c[:-self.window - 1]) / self.window
        self._var, self._next = var, float(sq[-self.window:].mean())

    def _path(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._next)


class EWMAVol(VolModel):
    name = "ewma"

    def __init__(self, lam: float = 0.94, periods: float = 252.0) -> None:
        super().__init__(periods)
        if not 0 < lam < 1:
            raise ValidationError("lam must be in (0, 1)")
        self.lam = lam

    def _fit(self, r: np.ndarray) -> None:
        var = np.empty(len(r))
        var[0] = r[:20].var() if len(r) >= 20 else r.var()
        for t in range(1, len(r)):
            var[t] = self.lam * var[t - 1] + (1 - self.lam) * r[t - 1] ** 2
        self._var, self._next = var, float(self.lam * var[-1] + (1 - self.lam) * r[-1] ** 2)
        self.params = {"lam": self.lam}

    def _path(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self._next)                 # EWMA has no mean reversion: the forecast is flat


class GarchVol(VolModel):
    """GARCH(1,1), or GJR-GARCH(1,1) with ``asymmetric=True``:

        h[t] = omega + (alpha + gamma * 1[r[t-1] < 0]) * r[t-1]^2 + beta * h[t-1]

    fitted by Gaussian maximum likelihood on de-meaned returns. Persistence is alpha + beta + gamma / 2."""
    min_obs = 250

    def __init__(self, asymmetric: bool = False, periods: float = 252.0) -> None:
        super().__init__(periods)
        self.asymmetric = asymmetric
        self.name = "gjr" if asymmetric else "garch"

    @staticmethod
    def _filter(e: np.ndarray, omega: float, alpha: float, beta: float, gamma: float, h0: float) -> np.ndarray:
        h = np.empty(len(e) + 1)
        h[0] = h0
        for t in range(len(e)):
            h[t + 1] = omega + (alpha + (gamma if e[t] < 0 else 0.0)) * e[t] ** 2 + beta * h[t]
        return h

    def _fit(self, r: np.ndarray) -> None:
        scale = 1.0 / max(r.std(), 1e-12)                    # work on unit-variance data so the optimiser is well conditioned
        e = (r - r.mean()) * scale
        v = float(e.var())

        def unpack(x: np.ndarray):
            return x[0], x[1], x[2], (x[3] if self.asymmetric else 0.0)

        def nll(x: np.ndarray) -> float:
            omega, alpha, beta, gamma = unpack(x)
            if omega <= 1e-8 or alpha < 0 or beta < 0 or gamma < 0 or alpha + beta + gamma / 2.0 >= 0.9995:
                return math.inf
            h = self._filter(e, omega, alpha, beta, gamma, v)[:-1]
            return float(0.5 * np.sum(np.log(h) + e ** 2 / h))
        best_x, best_f = None, math.inf
        for a0, b0 in ((0.05, 0.90), (0.10, 0.80), (0.15, 0.60)):          # a few starts: the likelihood surface is flat in places
            x0 = [v * (1 - a0 - b0), a0, b0] + ([0.05] if self.asymmetric else [])
            x, f = nelder_mead(nll, x0, step=[v * 0.02, 0.03, 0.05] + ([0.03] if self.asymmetric else []), max_iter=1500, tol=1e-9)
            if f < best_f:
                best_x, best_f = x, f
        if best_x is None or not math.isfinite(best_f):
            raise StatisticsError("GARCH fit failed to find a feasible point")
        omega, alpha, beta, gamma = unpack(best_x)
        h = self._filter(e, omega, alpha, beta, gamma, v)
        s2 = scale ** 2
        persistence = alpha + beta + gamma / 2.0
        self._var, self._next = h[:-1] / s2, float(h[-1] / s2)
        self.params = {"omega": float(omega / s2), "alpha": float(alpha), "beta": float(beta), "gamma": float(gamma),
                       "persistence": float(persistence), "long_run_vol": math.sqrt(omega / s2 / (1 - persistence) * self.periods),
                       "loglik": -best_f, "half_life": math.log(0.5) / math.log(persistence) if 0 < persistence < 1 else math.inf}

    def _path(self, horizon: int) -> np.ndarray:
        p = self.params
        long_run = p["omega"] / (1 - p["persistence"])
        return long_run + p["persistence"] ** np.arange(horizon) * (self._next - long_run)    # decays towards the long-run level


class HARVol(VolModel):
    """Heterogeneous autoregression of variance on its daily, weekly and monthly averages."""
    name = "har"

    def __init__(self, lags: Sequence[int] = (1, 5, 22), periods: float = 252.0) -> None:
        super().__init__(periods)
        if list(lags) != sorted(set(lags)) or lags[0] < 1:
            raise ValidationError("lags must be increasing positive integers")
        self.lags, self.min_obs = tuple(int(x) for x in lags), max(lags) * 4

    def fit(self, returns, variance=None) -> "HARVol":
        """``variance``: optional per-bar variance proxy (e.g. ``estimators.variance_proxy``); squared returns otherwise."""
        self._r = _clean(returns, self.min_obs)
        rv = self._r ** 2 if variance is None else _clean(variance, self.min_obs)
        self._fit(rv)
        return self

    def _features(self, rv: np.ndarray) -> np.ndarray:
        c = np.concatenate([[0.0], np.cumsum(rv)])
        m = max(self.lags)
        return np.column_stack([np.ones(len(rv) - m + 1)] + [(c[m:] - c[m - k:len(c) - k]) / k for k in self.lags])   # row i <-> bar i + m - 1

    def _fit(self, rv: np.ndarray) -> None:
        m = max(self.lags)
        X = self._features(rv)
        beta, *_ = np.linalg.lstsq(X[:-1], rv[m:], rcond=None)
        fitted = np.clip(X[:-1] @ beta, 1e-18, None)
        self._var = np.concatenate([np.full(m, rv[:m].mean()), fitted])
        self._beta, self._rv = beta, rv.copy()
        self.params = {"const": float(beta[0]), **{f"b{k}": float(b) for k, b in zip(self.lags, beta[1:])}}

    def _path(self, horizon: int) -> np.ndarray:
        rv, out = list(self._rv[-max(self.lags):]), []
        for _ in range(horizon):
            x = [1.0] + [float(np.mean(rv[-k:])) for k in self.lags]
            nxt = max(float(np.dot(self._beta, x)), 1e-18)
            out.append(nxt)
            rv.append(nxt)
        return np.array(out)


MODELS: Dict[str, Callable[..., VolModel]] = {"hist": HistoricalVol, "ewma": EWMAVol, "garch": GarchVol, "gjr": lambda **kw: GarchVol(True, **kw),
                                              "har": HARVol}


def make_model(name: str, **kwargs: Any) -> VolModel:
    if name not in MODELS:
        raise ValidationError(f"unknown volatility model '{name}'", details={"valid": sorted(MODELS)})
    return MODELS[name](**kwargs)


def forecast_volatility(returns, model: str = "garch", horizon: int = 1, **kwargs: Any) -> VolForecast:
    """Fit ``model`` on the returns and forecast ``horizon`` bars ahead."""
    return make_model(model, **kwargs).fit(returns).forecast(horizon)
