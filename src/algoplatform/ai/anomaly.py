"""Anomaly detection: flag what does not look like the rest.

Uses in a trading platform: bad ticks and fat-finger prices before they reach a strategy, days whose feature vector is
unlike anything a model was trained on, and unusual order or fill patterns for surveillance.
"""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from ..core.exceptions import ModelError, ValidationError
from ..quant.indicators import _period, _windows

_MAD_TO_SIGMA = 1.4826               # MAD * this = standard deviation, for normal data


def robust_zscore(x) -> np.ndarray:
    """Distance from the median in units of MAD-implied sigma. Unlike the ordinary z-score, the outlier being tested
    cannot inflate the scale it is measured against."""
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * _MAD_TO_SIGMA
    return (x - med) / mad if mad > 0 else np.zeros(len(x))


def rolling_robust_zscore(x, n: int = 50) -> np.ndarray:
    """Causal version: each value is scored against the ``n`` values *before* it."""
    x, n = np.asarray(x, float), _period(n)
    out = np.full(len(x), np.nan)
    w = _windows(x, n)
    if len(w) < 2:
        return out
    med = np.median(w[:-1], axis=1)
    mad = np.median(np.abs(w[:-1] - med[:, None]), axis=1) * _MAD_TO_SIGMA
    with np.errstate(divide="ignore", invalid="ignore"):
        out[n:] = np.where(mad > 0, (x[n:] - med) / np.where(mad > 0, mad, 1.0), 0.0)
    return out


def mahalanobis(X, reference=None) -> np.ndarray:
    """Multivariate distance of each row from the reference sample's centre, accounting for correlations."""
    X = np.atleast_2d(np.asarray(X, float))
    ref = X if reference is None else np.atleast_2d(np.asarray(reference, float))
    if len(ref) <= ref.shape[1]:
        raise ValidationError("reference needs more rows than columns")
    d = X - ref.mean(axis=0)
    inv = np.linalg.pinv(np.cov(ref, rowvar=False).reshape(ref.shape[1], ref.shape[1]))
    return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", d, inv, d), 0, None))


def _avg_path(n: float) -> float:
    """Average depth of an unsuccessful search in a binary tree of n points (the normaliser of isolation scores)."""
    return 0.0 if n <= 1 else 1.0 if n == 2 else 2.0 * (math.log(n - 1.0) + 0.5772156649) - 2.0 * (n - 1.0) / n


class IsolationForest:
    """Anomalies are few and different, so random splits isolate them in few steps. ``score`` is in (0, 1): about 0.5
    for ordinary points, towards 1 for anomalies. No distributional assumption and no labels needed."""

    def __init__(self, n_trees: int = 100, sample_size: int = 256, contamination: float = 0.01, seed: Optional[int] = 0) -> None:
        if n_trees < 1 or sample_size < 2 or not 0 < contamination < 0.5:
            raise ValidationError("need n_trees >= 1, sample_size >= 2, 0 < contamination < 0.5")
        self.n_trees, self.sample_size, self.contamination, self.seed = n_trees, sample_size, contamination, seed
        self._trees: List[tuple] = []

    def _grow(self, X: np.ndarray, rng: np.random.Generator, depth: int, limit: int):
        n = len(X)
        if depth >= limit or n <= 1:
            return ("leaf", n)
        spread = X.max(axis=0) - X.min(axis=0)
        live = np.flatnonzero(spread > 0)
        if not len(live):
            return ("leaf", n)
        j = int(rng.choice(live))
        cut = float(rng.uniform(X[:, j].min(), X[:, j].max()))
        left = X[:, j] < cut
        return ("node", j, cut, self._grow(X[left], rng, depth + 1, limit), self._grow(X[~left], rng, depth + 1, limit))

    def fit(self, X) -> "IsolationForest":
        X = np.asarray(X, float)
        if X.ndim != 2 or len(X) < 8 or not np.isfinite(X).all():
            raise ValidationError("X must be a finite 2-D array with at least 8 rows")
        rng = np.random.default_rng(self.seed)
        self._psi = min(self.sample_size, len(X))
        limit = int(math.ceil(math.log2(self._psi)))
        self._trees = [self._grow(X[rng.choice(len(X), self._psi, replace=False)], rng, 0, limit) for _ in range(self.n_trees)]
        self.n_features_ = X.shape[1]
        self.threshold_ = float(np.quantile(self.score(X), 1 - self.contamination))
        return self

    def _depths(self, node, X: np.ndarray, rows: np.ndarray, depth: int, out: np.ndarray) -> None:
        if node[0] == "leaf":
            out[rows] += depth + _avg_path(node[1])
            return
        _, j, cut, left, right = node
        go = X[rows, j] < cut
        if go.any():
            self._depths(left, X, rows[go], depth + 1, out)
        if (~go).any():
            self._depths(right, X, rows[~go], depth + 1, out)

    def score(self, X) -> np.ndarray:
        if not self._trees:
            raise ModelError("IsolationForest is not fitted")
        X = np.atleast_2d(np.asarray(X, float))
        if X.shape[1] != getattr(self, "n_features_", X.shape[1]):
            raise ValidationError("wrong number of features")
        total = np.zeros(len(X))
        rows = np.arange(len(X))
        for tree in self._trees:
            self._depths(tree, X, rows, 0, total)
        return np.power(2.0, -(total / len(self._trees)) / _avg_path(self._psi))

    def is_anomaly(self, X) -> np.ndarray:
        """True where the score exceeds the level that flagged ``contamination`` of the training data."""
        return self.score(X) > self.threshold_
