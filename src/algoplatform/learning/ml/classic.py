"""Classical models that complement the linear ones in ``research.models``: k-nearest neighbours, Gaussian naive Bayes
and the lasso. Same estimator convention (``predict_proba`` is the probability of class 1)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ...foundation.core.exceptions import ModelError, ValidationError
from .tree import _check_x, _check_xy


class KNN:
    """k-nearest neighbours: "what happened after the k most similar past situations". Distances are Euclidean, so scale
    the features first (``research.features.Scaler``). ``weights="distance"`` lets closer analogues count for more."""

    def __init__(self, k: int = 15, task: str = "classification", weights: str = "uniform") -> None:
        if k < 1 or task not in ("classification", "regression") or weights not in ("uniform", "distance"):
            raise ValidationError("need k >= 1, task classification|regression, weights uniform|distance")
        self.k, self.task, self.weights = k, task, weights
        self.n_features_: Optional[int] = None

    def fit(self, X, y, sample_weight=None) -> "KNN":
        self._X, self._y, self._w = _check_xy(X, y, sample_weight, self.task == "classification")
        if len(self._y) < self.k:
            raise ValidationError("fewer training rows than k", details={"rows": len(self._y), "k": self.k})
        self.n_features_ = self._X.shape[1]
        return self

    def predict_value(self, X, chunk: int = 512) -> np.ndarray:
        X = _check_x(X, self.n_features_)
        out = np.empty(len(X))
        sq = (self._X ** 2).sum(axis=1)
        for i in range(0, len(X), chunk):
            q = X[i:i + chunk]
            d2 = np.clip((q ** 2).sum(axis=1)[:, None] - 2 * q @ self._X.T + sq[None, :], 0, None)
            nn = np.argpartition(d2, self.k - 1, axis=1)[:, :self.k]
            w = self._w[nn]
            if self.weights == "distance":
                w = w / (np.sqrt(np.take_along_axis(d2, nn, axis=1)) + 1e-9)
            out[i:i + chunk] = (w * self._y[nn]).sum(axis=1) / w.sum(axis=1)
        return out

    def predict_proba(self, X) -> np.ndarray:
        if self.task != "classification":
            raise ModelError("predict_proba is only available for classification")
        return self.predict_value(X)

    def predict(self, X) -> np.ndarray:
        v = self.predict_value(X)
        return (v >= 0.5).astype(float) if self.task == "classification" else v


class GaussianNB:
    """Naive Bayes with a normal distribution per feature and class. Fast, hard to overfit, and a useful baseline:
    if a deep model cannot beat this out of sample, the extra complexity bought nothing."""

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing
        self.n_features_: Optional[int] = None

    def fit(self, X, y, sample_weight=None) -> "GaussianNB":
        X, y, w = _check_xy(X, y, sample_weight, True)
        if len(np.unique(y)) < 2:
            raise ValidationError("training data contains a single class")
        self.n_features_ = X.shape[1]
        eps = self.var_smoothing * X.var(axis=0).max() + 1e-12
        self._stats = []
        for c in (0.0, 1.0):
            m = y == c
            mu = np.average(X[m], axis=0, weights=w[m])
            var = np.average((X[m] - mu) ** 2, axis=0, weights=w[m]) + eps
            self._stats.append((mu, var, float(np.log(w[m].sum() / w.sum()))))
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = _check_x(X, self.n_features_)
        ll = [prior - 0.5 * (np.log(2 * np.pi * var) + (X - mu) ** 2 / var).sum(axis=1) for mu, var, prior in self._stats]
        return 1.0 / (1.0 + np.exp(np.clip(ll[0] - ll[1], -500, 500)))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(float)


class LassoModel:
    """L1-regularised linear regression by coordinate descent. The L1 penalty sets weak coefficients exactly to zero,
    so it doubles as a feature selector (``selected_``). Features are standardised internally; ``coef_`` is reported on
    the original scale with the intercept first, like ``RidgeModel``."""

    def __init__(self, alpha: float = 0.01, max_iter: int = 1000, tol: float = 1e-7) -> None:
        if alpha < 0:
            raise ValidationError("alpha must be >= 0")
        self.alpha, self.max_iter, self.tol = alpha, max_iter, tol
        self.coef_: Optional[np.ndarray] = None

    def fit(self, X, y, sample_weight=None) -> "LassoModel":
        X, y, w = _check_xy(X, y, sample_weight)
        w = w / w.sum()
        mu, ybar = w @ X, float(w @ y)
        sd = np.sqrt(w @ (X - mu) ** 2)
        live = sd > 1e-12
        Z = np.where(live, (X - mu) / np.where(live, sd, 1.0), 0.0)
        r = y - ybar
        beta = np.zeros(X.shape[1])
        for _ in range(self.max_iter):
            biggest = 0.0
            for j in np.flatnonzero(live):
                rho = float(w @ (Z[:, j] * (r + Z[:, j] * beta[j])))       # correlation with the partial residual (column variance is 1)
                new = np.sign(rho) * max(abs(rho) - self.alpha, 0.0)
                if new != beta[j]:
                    r -= Z[:, j] * (new - beta[j])
                    biggest = max(biggest, abs(new - beta[j]))
                    beta[j] = new
            if biggest < self.tol:
                break
        coef = np.where(live, beta / np.where(live, sd, 1.0), 0.0)
        self.coef_ = np.concatenate([[ybar - coef @ mu], coef])
        self.selected_ = np.flatnonzero(coef != 0)
        return self

    def predict(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise ModelError("model is not fitted")
        return self.coef_[0] + _check_x(X, len(self.coef_) - 1) @ self.coef_[1:]

    predict_proba = predict
