"""Unsupervised learning: PCA for compressing correlated features, k-means for grouping similar observations."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ...foundation.core.exceptions import ModelError, ValidationError


def _matrix(X) -> np.ndarray:
    X = np.asarray(X, float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValidationError("X must be a finite 2-D array")
    return X


class PCA:
    """Principal components. Fit on training rows only and reuse the fitted transform on later data: refitting on the
    full history leaks the future into the past."""

    def __init__(self, n_components: int = 2, standardize: bool = True) -> None:
        if n_components < 1:
            raise ValidationError("n_components must be >= 1")
        self.n_components, self.standardize = n_components, standardize
        self.components_: Optional[np.ndarray] = None

    def fit(self, X) -> "PCA":
        X = _matrix(X)
        if self.n_components > X.shape[1] or len(X) < 2:
            raise ValidationError("n_components exceeds the number of features, or fewer than 2 rows")
        self.mean_ = X.mean(axis=0)
        sd = X.std(axis=0, ddof=1)
        self.scale_ = np.where(sd > 0, sd, 1.0) if self.standardize else np.ones(X.shape[1])
        vals, vecs = np.linalg.eigh(np.cov((X - self.mean_) / self.scale_, rowvar=False))
        order = np.argsort(vals)[::-1][:self.n_components]
        vecs = vecs[:, order]
        self.components_ = (vecs * np.where(np.abs(vecs).max(axis=0) == vecs.max(axis=0), 1.0, -1.0)).T     # largest loading positive
        self.explained_variance_ = np.clip(vals[order], 0, None)
        self.explained_variance_ratio_ = self.explained_variance_ / max(np.clip(vals, 0, None).sum(), 1e-300)
        return self

    def transform(self, X) -> np.ndarray:
        if self.components_ is None:
            raise ModelError("PCA is not fitted")
        return ((np.atleast_2d(np.asarray(X, float)) - self.mean_) / self.scale_) @ self.components_.T

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z) -> np.ndarray:
        if self.components_ is None:
            raise ModelError("PCA is not fitted")
        return np.atleast_2d(np.asarray(Z, float)) @ self.components_ * self.scale_ + self.mean_


class KMeans:
    """Lloyd's algorithm with k-means++ seeding and several restarts (the best inertia wins)."""

    def __init__(self, k: int = 3, n_init: int = 5, max_iter: int = 200, tol: float = 1e-8, seed: Optional[int] = 0) -> None:
        if k < 1 or n_init < 1:
            raise ValidationError("k and n_init must be >= 1")
        self.k, self.n_init, self.max_iter, self.tol, self.seed = k, n_init, max_iter, tol, seed
        self.centers_: Optional[np.ndarray] = None

    @staticmethod
    def _dist2(X: np.ndarray, C: np.ndarray) -> np.ndarray:
        return np.clip((X ** 2).sum(axis=1)[:, None] - 2 * X @ C.T + (C ** 2).sum(axis=1)[None, :], 0, None)

    def fit(self, X) -> "KMeans":
        X = _matrix(X)
        if len(X) < self.k:
            raise ValidationError("fewer rows than clusters")
        rng = np.random.default_rng(self.seed)
        best: Tuple[float, Optional[np.ndarray], Optional[np.ndarray]] = (np.inf, None, None)
        for _ in range(self.n_init):
            centers = [X[rng.integers(len(X))]]
            for _ in range(1, self.k):                                   # k-means++: spread the initial centres out
                d2 = self._dist2(X, np.array(centers)).min(axis=1)
                centers.append(X[rng.choice(len(X), p=d2 / d2.sum())] if d2.sum() > 0 else X[rng.integers(len(X))])
            C = np.array(centers)
            for _ in range(self.max_iter):
                labels = self._dist2(X, C).argmin(axis=1)
                new = np.array([X[labels == j].mean(axis=0) if (labels == j).any() else C[j] for j in range(self.k)])
                shift = float(((new - C) ** 2).sum())
                C = new
                if shift < self.tol:
                    break
            d2 = self._dist2(X, C)
            labels = d2.argmin(axis=1)
            inertia = float(d2[np.arange(len(X)), labels].sum())
            if inertia < best[0]:
                best = (inertia, C, labels)
        self.inertia_, self.centers_, self.labels_ = best
        return self

    def predict(self, X) -> np.ndarray:
        if self.centers_ is None:
            raise ModelError("KMeans is not fitted")
        return self._dist2(np.atleast_2d(np.asarray(X, float)), self.centers_).argmin(axis=1)
