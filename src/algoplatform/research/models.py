"""Small dependency-free models plus evaluation helpers (calibration, importance, drift)."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..core.exceptions import ValidationError


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


class LogisticModel:
    """L2-regularised logistic regression by Newton/IRLS, with per-sample weights (e.g. label uniqueness)."""

    def __init__(self, l2: float = 1.0, max_iter: int = 50, tol: float = 1e-8) -> None:
        self.l2, self.max_iter, self.tol = l2, max_iter, tol
        self.coef_: Optional[np.ndarray] = None

    def fit(self, X, y, sample_weight=None) -> "LogisticModel":
        X, y = np.asarray(X, float), np.asarray(y, float)
        if X.ndim != 2 or len(X) != len(y):
            raise ValidationError("X must be 2-D and match y")
        if not np.isfinite(X).all():
            raise ValidationError("X contains NaN/inf")
        if set(np.unique(y)) - {0.0, 1.0}:
            raise ValidationError("y must be 0/1")
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        A = np.column_stack([np.ones(len(X)), X])
        pen = np.eye(A.shape[1]) * self.l2
        pen[0, 0] = 0.0                                                         # never shrink the intercept
        beta = np.zeros(A.shape[1])
        for _ in range(self.max_iter):
            p = _sigmoid(A @ beta)
            grad = A.T @ (sw * (y - p)) - pen @ beta
            H = A.T @ (A * (sw * p * (1 - p))[:, None]) + pen
            step = np.linalg.solve(H + 1e-9 * np.eye(len(beta)), grad)
            beta = beta + step
            if np.max(np.abs(step)) < self.tol:
                break
        self.coef_ = beta
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise ValidationError("model not fitted")
        X = np.atleast_2d(np.asarray(X, float))
        return _sigmoid(self.coef_[0] + X @ self.coef_[1:])


class RidgeModel:
    """Closed-form ridge regression (continuous target, e.g. forward return)."""

    def __init__(self, l2: float = 1.0) -> None:
        self.l2, self.coef_ = l2, None

    def fit(self, X, y, sample_weight=None) -> "RidgeModel":
        X, y = np.asarray(X, float), np.asarray(y, float)
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        A = np.column_stack([np.ones(len(X)), X])
        pen = np.eye(A.shape[1]) * self.l2
        pen[0, 0] = 0.0
        self.coef_ = np.linalg.solve(A.T @ (A * sw[:, None]) + pen, A.T @ (sw * y))
        return self

    def predict(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise ValidationError("model not fitted")
        return self.coef_[0] + np.atleast_2d(np.asarray(X, float)) @ self.coef_[1:]

    predict_proba = predict


class SklearnAdapter:
    """Wrap any estimator with ``fit`` / ``predict_proba`` (scikit-learn, xgboost, lightgbm...) without importing
    them here. Passes sample weights when the estimator accepts them."""

    def __init__(self, estimator) -> None:
        self.est = estimator

    def fit(self, X, y, sample_weight=None) -> "SklearnAdapter":
        try:
            self.est.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            self.est.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = np.asarray(self.est.predict_proba(X))
        return p[:, 1] if p.ndim == 2 else p


def log_loss(y, p, eps: float = 1e-12) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def accuracy(y, p, threshold: float = 0.5) -> float:
    return float(np.mean((np.asarray(p) >= threshold) == (np.asarray(y) >= 0.5)))


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed frequency per probability bin. A calibrated model's two columns agree."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = [(edges[b], edges[b + 1], int((idx == b).sum()), float(p[idx == b].mean()), float(y[idx == b].mean()))
            for b in range(bins) if (idx == b).any()]
    return pd.DataFrame(rows, columns=["low", "high", "n", "mean_pred", "observed"])


def permutation_importance(predict: Callable[[np.ndarray], np.ndarray], X: np.ndarray, y: np.ndarray,
                           metric: Callable[[np.ndarray, np.ndarray], float] = log_loss, n_repeats: int = 5,
                           seed: Optional[int] = 0, names: Optional[list] = None) -> pd.Series:
    """Increase in loss when one feature is shuffled (use OUT-of-sample data: in-sample importance rewards overfit)."""
    X, rng = np.asarray(X, float), np.random.default_rng(seed)
    base = metric(y, predict(X))
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            out[j] += metric(y, predict(Xp)) - base
    return pd.Series(out / n_repeats, index=names or [f"x{j}" for j in range(X.shape[1])]).sort_values(ascending=False)


def psi(expected, actual, bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index. <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift (retrain or stand down)."""
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    e, a = e[np.isfinite(e)], a[np.isfinite(a)]
    if len(e) < bins or len(a) < 2:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e)
    pa = np.histogram(a, edges)[0] / len(a)
    pe, pa = np.clip(pe, eps, None), np.clip(pa, eps, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def out_of_range_share(train, recent, q: float = 0.01) -> np.ndarray:
    """Per feature: share of ``recent`` values outside the [q, 1-q] quantile range seen in training.

    Robust to the strong autocorrelation of rolling-window features, where PSI (which assumes independent samples)
    is inflated even when nothing has changed. ~q means stable; a large share means the model is extrapolating."""
    train, recent = np.asarray(train, float), np.asarray(recent, float)
    lo, hi = np.nanquantile(train, q, axis=0), np.nanquantile(train, 1 - q, axis=0)
    ok = np.isfinite(recent)
    out = ((recent < lo) | (recent > hi)) & ok
    return out.sum(axis=0) / np.maximum(ok.sum(axis=0), 1)
