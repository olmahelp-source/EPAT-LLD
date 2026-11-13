"""Tree models in numpy: a CART decision tree, a random forest and gradient boosting.

All follow the estimator convention of ``research.models``: ``fit(X, y, sample_weight=None)``, ``predict(X)`` and, for
classifiers, ``predict_proba(X)`` returning the probability of class 1 as a 1-D array. Labels are 0/1.

Trees need no feature scaling and pick up interactions and thresholds ("momentum works only when volatility is low")
that linear models cannot. They also overfit financial data very easily: keep them shallow, keep leaves large, and judge
them only on purged out-of-sample folds (``research.cv``).
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from ...foundation.core.exceptions import ModelError, ValidationError
from ...foundation.utils.numeric import sigmoid


def _check_xy(X, y, sample_weight=None, classification: bool = False):
    X, y = np.asarray(X, float), np.asarray(y, float).ravel()
    if X.ndim != 2 or len(X) != len(y) or len(y) == 0:
        raise ValidationError("X must be 2-D with one row per label", details={"X": X.shape, "y": y.shape})
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValidationError("X and y must be finite (drop or impute NaN rows first)")
    if classification and set(np.unique(y)) - {0.0, 1.0}:
        raise ValidationError("classification labels must be 0/1")
    w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float).ravel()
    if len(w) != len(y) or (w < 0).any() or w.sum() <= 0:
        raise ValidationError("sample_weight must be non-negative, match y, and not sum to zero")
    return X, y, w


def _check_x(X, n_features: Optional[int]) -> np.ndarray:
    if n_features is None:
        raise ModelError("model is not fitted")
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[1] != n_features:
        raise ValidationError(f"expected {n_features} features, got {X.shape[1]}")
    return X


class DecisionTree:
    """CART with weighted variance reduction (for 0/1 labels this is the Gini criterion)."""

    def __init__(self, task: str = "classification", max_depth: int = 4, min_samples_leaf: int = 10,
                 max_features: Union[None, int, float, str] = None, seed: Optional[int] = 0) -> None:
        if task not in ("classification", "regression"):
            raise ValidationError("task must be 'classification' or 'regression'")
        if max_depth < 1 or min_samples_leaf < 1:
            raise ValidationError("max_depth and min_samples_leaf must be >= 1")
        self.task, self.max_depth, self.min_samples_leaf, self.max_features, self.seed = task, max_depth, min_samples_leaf, max_features, seed
        self.n_features_: Optional[int] = None

    def _n_try(self, d: int) -> int:
        mf = self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if isinstance(mf, float):
            return max(1, min(d, int(round(mf * d))))
        return max(1, min(d, int(mf)))

    def _best_split(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, features: np.ndarray):
        """-> (gain, feature, threshold) of the best admissible split, or None."""
        n, leaf = len(y), self.min_samples_leaf
        if n < 2 * leaf:
            return None
        total_w, total_wy = w.sum(), (w * y).sum()
        base = total_wy ** 2 / total_w
        best = None
        for j in features:
            order = np.argsort(X[:, j], kind="mergesort")
            xs, ws, wys = X[order, j], np.cumsum(w[order]), np.cumsum(w[order] * y[order])
            pos = np.arange(leaf - 1, n - leaf)                           # split after position i: left = [0..i]
            pos = pos[xs[pos] < xs[pos + 1]]                              # only between distinct values
            if not len(pos):
                continue
            wl, wyl = ws[pos], wys[pos]
            wr, wyr = total_w - wl, total_wy - wyl
            ok = (wl > 0) & (wr > 0)
            if not ok.any():
                continue
            gain = np.where(ok, wyl ** 2 / np.where(ok, wl, 1.0) + wyr ** 2 / np.where(ok, wr, 1.0) - base, -np.inf)
            k = int(np.argmax(gain))
            if gain[k] > 1e-12 and (best is None or gain[k] > best[0]):
                best = (float(gain[k]), int(j), float((xs[pos[k]] + xs[pos[k] + 1]) / 2.0))
        return best

    def fit(self, X, y, sample_weight=None) -> "DecisionTree":
        X, y, w = _check_xy(X, y, sample_weight, self.task == "classification")
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        self.n_features_ = d
        feature: List[int] = []
        threshold: List[float] = []
        left: List[int] = []
        right: List[int] = []
        value: List[float] = []
        importance = np.zeros(d)
        stack = [(np.arange(len(y)), 0, -1, False)]                       # (rows, depth, parent, is_right_child)
        while stack:
            rows, depth, parent, is_right = stack.pop()
            node = len(feature)
            if parent >= 0:
                (right if is_right else left)[parent] = node
            feature.append(-1)
            threshold.append(0.0)
            left.append(-1)
            right.append(-1)
            value.append(float(np.average(y[rows], weights=w[rows])) if w[rows].sum() > 0 else float(y[rows].mean()))
            if depth >= self.max_depth:
                continue
            feats = np.arange(d) if self._n_try(d) == d else rng.choice(d, self._n_try(d), replace=False)
            split = self._best_split(X[rows], y[rows], w[rows], feats)
            if split is None:
                continue
            gain, j, thr = split
            go_left = X[rows, j] <= thr
            feature[node], threshold[node] = j, thr
            importance[j] += gain
            stack.append((rows[~go_left], depth + 1, node, True))
            stack.append((rows[go_left], depth + 1, node, False))
        self.feature_, self.threshold_ = np.array(feature), np.array(threshold)
        self.left_, self.right_, self.value_ = np.array(left), np.array(right), np.array(value)
        self.feature_importances_ = importance / importance.sum() if importance.sum() > 0 else importance
        return self

    def apply(self, X) -> np.ndarray:
        """Index of the leaf each row lands in."""
        X = _check_x(X, self.n_features_)
        node = np.zeros(len(X), dtype=int)
        rows = np.arange(len(X))
        while True:
            f = self.feature_[node]
            inner = f >= 0
            if not inner.any():
                return node
            go_left = X[rows, np.where(inner, f, 0)] <= self.threshold_[node]
            node = np.where(inner, np.where(go_left, self.left_[node], self.right_[node]), node)

    def predict_value(self, X) -> np.ndarray:
        leaves = self.apply(X)                          # first: raises ModelError when the tree has not been fitted
        return self.value_[leaves]

    def predict_proba(self, X) -> np.ndarray:
        if self.task != "classification":
            raise ModelError("predict_proba is only available for classification trees")
        return self.predict_value(X)

    def predict(self, X) -> np.ndarray:
        v = self.predict_value(X)
        return (v >= 0.5).astype(float) if self.task == "classification" else v

    @property
    def n_leaves(self) -> int:
        return int((self.feature_ < 0).sum())


class RandomForest:
    """Bagged trees with random feature subsets. ``oob_score_`` is measured on the rows each tree did not see
    (accuracy for classification, R^2 for regression): a free, if optimistic for time series, generalisation check."""

    def __init__(self, task: str = "classification", n_trees: int = 100, max_depth: int = 5, min_samples_leaf: int = 10,
                 max_features: Union[None, int, float, str] = "sqrt", seed: Optional[int] = 0) -> None:
        if n_trees < 1:
            raise ValidationError("n_trees must be >= 1")
        self.task, self.n_trees, self.max_depth, self.min_samples_leaf, self.max_features, self.seed = \
            task, n_trees, max_depth, min_samples_leaf, max_features, seed
        self.trees_: List[DecisionTree] = []
        self.n_features_: Optional[int] = None

    def fit(self, X, y, sample_weight=None) -> "RandomForest":
        X, y, w = _check_xy(X, y, sample_weight, self.task == "classification")
        rng = np.random.default_rng(self.seed)
        n = len(y)
        self.n_features_ = X.shape[1]
        self.trees_ = []
        oob_sum, oob_cnt = np.zeros(n), np.zeros(n)
        p = w / w.sum()
        for _ in range(self.n_trees):
            idx = rng.choice(n, n, replace=True, p=p)                     # weights enter through the bootstrap
            tree = DecisionTree(self.task, self.max_depth, self.min_samples_leaf, self.max_features, int(rng.integers(2 ** 31)))
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
            out = np.ones(n, dtype=bool)
            out[idx] = False
            if out.any():
                oob_sum[out] += tree.predict_value(X[out])
                oob_cnt[out] += 1
        seen = oob_cnt > 0
        self.oob_score_ = float("nan")
        if seen.sum() > 1:
            pred = oob_sum[seen] / oob_cnt[seen]
            if self.task == "classification":
                self.oob_score_ = float(((pred >= 0.5) == (y[seen] >= 0.5)).mean())
            elif y[seen].var() > 0:
                self.oob_score_ = float(1 - ((y[seen] - pred) ** 2).mean() / y[seen].var())
        self.feature_importances_ = np.mean([t.feature_importances_ for t in self.trees_], axis=0)
        return self

    def predict_value(self, X) -> np.ndarray:
        X = _check_x(X, self.n_features_)
        return np.mean([t.predict_value(X) for t in self.trees_], axis=0)

    def predict_proba(self, X) -> np.ndarray:
        if self.task != "classification":
            raise ModelError("predict_proba is only available for classification forests")
        return self.predict_value(X)

    def predict(self, X) -> np.ndarray:
        v = self.predict_value(X)
        return (v >= 0.5).astype(float) if self.task == "classification" else v


class GradientBoosting:
    """Stage-wise additive trees: each new shallow tree fits what the ensemble still gets wrong.

    Classification uses logistic loss with a Newton step per leaf; regression uses squared loss. ``subsample`` < 1 gives
    stochastic boosting. With ``validation_fraction`` > 0 the **last** part of the data (time order is assumed) is held
    out, training stops after ``patience`` rounds without improvement there, and the model is cut back to the best round."""

    def __init__(self, task: str = "classification", n_estimators: int = 100, learning_rate: float = 0.05, max_depth: int = 3,
                 min_samples_leaf: int = 10, subsample: float = 1.0, validation_fraction: float = 0.0, patience: int = 10,
                 seed: Optional[int] = 0) -> None:
        if task not in ("classification", "regression"):
            raise ValidationError("task must be 'classification' or 'regression'")
        if n_estimators < 1 or not 0 < learning_rate <= 1 or not 0 < subsample <= 1 or not 0 <= validation_fraction < 0.9:
            raise ValidationError("need n_estimators >= 1, 0 < learning_rate <= 1, 0 < subsample <= 1, 0 <= validation_fraction < 0.9")
        self.task, self.n_estimators, self.learning_rate, self.max_depth = task, n_estimators, learning_rate, max_depth
        self.min_samples_leaf, self.subsample, self.validation_fraction, self.patience, self.seed = \
            min_samples_leaf, subsample, validation_fraction, patience, seed
        self.trees_: List[DecisionTree] = []
        self.n_features_: Optional[int] = None

    def _loss(self, y: np.ndarray, f: np.ndarray) -> float:
        if self.task == "regression":
            return float(np.mean((y - f) ** 2))
        p = np.clip(sigmoid(f), 1e-12, 1 - 1e-12)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def fit(self, X, y, sample_weight=None) -> "GradientBoosting":
        X, y, w = _check_xy(X, y, sample_weight, self.task == "classification")
        rng = np.random.default_rng(self.seed)
        n_val = int(len(y) * self.validation_fraction)
        Xt, yt, wt = (X[:-n_val], y[:-n_val], w[:-n_val]) if n_val else (X, y, w)
        if len(yt) < 2 * self.min_samples_leaf:
            raise ValidationError("too few training rows for min_samples_leaf", details={"rows": len(yt)})
        self.n_features_ = X.shape[1]
        mean = float(np.average(yt, weights=wt))
        if self.task == "classification":
            mean = min(max(mean, 1e-6), 1 - 1e-6)
            self.init_ = float(np.log(mean / (1 - mean)))
        else:
            self.init_ = mean
        f = np.full(len(yt), self.init_)
        fv = np.full(n_val, self.init_)
        self.trees_, self.validation_loss_ = [], []
        importance = np.zeros(X.shape[1])
        best, best_round = np.inf, 0
        for m in range(self.n_estimators):
            p = sigmoid(f) if self.task == "classification" else None
            resid = yt - (p if p is not None else f)
            rows = np.arange(len(yt)) if self.subsample >= 1 else rng.choice(len(yt), max(2 * self.min_samples_leaf, int(self.subsample * len(yt))), replace=False)
            tree = DecisionTree("regression", self.max_depth, self.min_samples_leaf, None, int(rng.integers(2 ** 31)))
            tree.fit(Xt[rows], resid[rows], wt[rows])
            if p is not None:                                           # Newton leaf values: sum(residual) / sum(p (1 - p))
                leaves = tree.apply(Xt[rows])
                num = np.bincount(leaves, weights=wt[rows] * resid[rows], minlength=len(tree.value_))
                den = np.bincount(leaves, weights=wt[rows] * p[rows] * (1 - p[rows]), minlength=len(tree.value_))
                tree.value_ = np.where(den > 1e-12, num / np.where(den > 1e-12, den, 1.0), 0.0)
                tree.value_ = np.clip(tree.value_, -4.0, 4.0)           # bound the step when a leaf is almost pure
            f += self.learning_rate * tree.predict_value(Xt)
            self.trees_.append(tree)
            importance += tree.feature_importances_
            if n_val:
                fv += self.learning_rate * tree.predict_value(X[-n_val:])
                loss = self._loss(y[-n_val:], fv)
                self.validation_loss_.append(loss)
                if loss < best - 1e-9:
                    best, best_round = loss, m + 1
                elif m + 1 - best_round >= self.patience:
                    break
        if n_val:
            self.trees_ = self.trees_[:max(best_round, 1)]
        self.best_iteration_ = len(self.trees_)
        self.feature_importances_ = importance / importance.sum() if importance.sum() > 0 else importance
        return self

    def decision_function(self, X) -> np.ndarray:
        X = _check_x(X, self.n_features_)
        f = np.full(len(X), self.init_)
        for tree in self.trees_:
            f += self.learning_rate * tree.predict_value(X)
        return f

    def predict_proba(self, X) -> np.ndarray:
        if self.task != "classification":
            raise ModelError("predict_proba is only available for classification")
        return sigmoid(self.decision_function(X))

    def predict(self, X) -> np.ndarray:
        f = self.decision_function(X)
        return (f >= 0).astype(float) if self.task == "classification" else f
