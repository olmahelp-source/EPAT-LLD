"""Ready-made deep models with the platform's estimator convention (``fit`` / ``predict`` / ``predict_proba``).

Each wrapper standardises its inputs with statistics from the training rows only, builds the network from a seed (same
seed, same data -> same model) and trains with early stopping on the most recent part of the data.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ...foundation.core.exceptions import ModelError, ValidationError
from ...foundation.utils.numeric import sigmoid
from .nn import LSTM, Activation, Adam, Dense, Dropout, Layer, Sequential


def make_sequences(X, y=None, lookback: int = 20) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Turn a time-ordered feature matrix into overlapping windows.

    Window k covers rows ``k .. k + lookback - 1`` and is paired with the label of its **last** row, so a prediction for
    row t uses rows up to t and nothing later. Windows containing NaN are dropped. Returns
    ``(windows (n, lookback, d), labels or None, index of each window's last row)``."""
    X = np.asarray(X, float)
    if X.ndim != 2 or lookback < 1:
        raise ValidationError("X must be 2-D and lookback >= 1")
    if len(X) < lookback:
        return np.empty((0, lookback, X.shape[1])), (None if y is None else np.empty(0)), np.empty(0, dtype=int)
    win = np.lib.stride_tricks.sliding_window_view(X, (lookback, X.shape[1]))[:, 0]
    last = np.arange(lookback - 1, len(X))
    ok = np.isfinite(win).all(axis=(1, 2))
    if y is not None:
        y = np.asarray(y, float).ravel()
        if len(y) != len(X):
            raise ValidationError("y must have one entry per row of X")
        ok &= np.isfinite(y[last])
        return win[ok].copy(), y[last][ok], last[ok]
    return win[ok].copy(), None, last[ok]


class _Scaled:
    """Shared plumbing: input standardisation fitted on the training rows."""
    _ymu: float = 0.0
    _ysd: float = 1.0

    def _fit_scale(self, X: np.ndarray) -> None:
        flat = X.reshape(-1, X.shape[-1])
        self._mu = flat.mean(axis=0)
        sd = flat.std(axis=0)
        self._sd = np.where(sd > 1e-12, sd, 1.0)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mu) / self._sd


class MLP(_Scaled):
    """Feed-forward network for tabular features. ``task="classification"`` predicts P(class 1) for 0/1 labels;
    ``"regression"`` predicts a continuous target (standardised internally, Huber loss)."""

    def __init__(self, task: str = "classification", hidden: Sequence[int] = (32, 16), dropout: float = 0.1, lr: float = 3e-3,
                 epochs: int = 200, batch_size: int = 64, weight_decay: float = 1e-4, patience: int = 15, validation_fraction: float = 0.2,
                 seed: Optional[int] = 0) -> None:
        if task not in ("classification", "regression"):
            raise ValidationError("task must be 'classification' or 'regression'")
        if not hidden or min(hidden) < 1:
            raise ValidationError("hidden must list at least one positive layer size")
        self.task, self.hidden, self.dropout, self.lr, self.epochs, self.batch_size = task, tuple(hidden), dropout, lr, epochs, batch_size
        self.weight_decay, self.patience, self.validation_fraction, self.seed = weight_decay, patience, validation_fraction, seed
        self.net_: Optional[Sequential] = None

    def _build(self, n_in: int) -> Sequential:
        rng = np.random.default_rng(self.seed)
        layers: List[Layer] = []
        prev = n_in
        for h in self.hidden:
            layers += [Dense(prev, h, rng), Activation("relu")] + ([Dropout(self.dropout, rng)] if self.dropout > 0 else [])
            prev = h
        layers.append(Dense(prev, 1, rng, init="glorot"))
        return Sequential(layers, "bce" if self.task == "classification" else "huber", Adam(self.lr, weight_decay=self.weight_decay), self.seed)

    def fit(self, X, y, sample_weight=None) -> "MLP":
        X, y = np.asarray(X, float), np.asarray(y, float).ravel()
        if X.ndim != 2 or len(X) != len(y):
            raise ValidationError("X must be 2-D with one row per label")
        if self.task == "classification" and set(np.unique(y)) - {0.0, 1.0}:
            raise ValidationError("classification labels must be 0/1")
        n_tr = len(X) - int(len(X) * self.validation_fraction)
        self._fit_scale(X[:n_tr])
        self._ymu, self._ysd = (0.0, 1.0) if self.task == "classification" else (float(y[:n_tr].mean()), float(y[:n_tr].std()) or 1.0)
        self.net_ = self._build(X.shape[1])
        self.net_.fit(self._scale(X), (y - self._ymu) / self._ysd, self.epochs, self.batch_size, self.validation_fraction, self.patience,
                      sample_weight=sample_weight)
        return self

    def _raw(self, X) -> np.ndarray:
        if self.net_ is None:
            raise ModelError("model is not fitted")
        return self.net_.predict(self._scale(np.atleast_2d(np.asarray(X, float))))[:, 0]

    def predict_proba(self, X) -> np.ndarray:
        if self.task != "classification":
            raise ModelError("predict_proba is only available for classification")
        return sigmoid(self._raw(X))

    def predict(self, X) -> np.ndarray:
        raw = self._raw(X)
        return (raw >= 0).astype(float) if self.task == "classification" else raw * self._ysd + self._ymu


class LSTMForecaster(_Scaled):
    """Sequence model: reads the last ``lookback`` rows of features to predict the label of the current row.

    ``fit(X, y)`` takes the plain time-ordered feature matrix and builds the windows itself; ``predict`` returns one
    value per row of X, NaN for the first ``lookback - 1`` rows (and for any window containing NaN)."""

    def __init__(self, lookback: int = 20, hidden: int = 16, task: str = "regression", dropout: float = 0.0, lr: float = 5e-3,
                 epochs: int = 60, batch_size: int = 32, weight_decay: float = 1e-4, patience: int = 10, validation_fraction: float = 0.2,
                 seed: Optional[int] = 0) -> None:
        if task not in ("classification", "regression") or lookback < 2 or hidden < 1:
            raise ValidationError("need task classification|regression, lookback >= 2, hidden >= 1")
        self.lookback, self.hidden, self.task, self.dropout, self.lr, self.epochs = lookback, hidden, task, dropout, lr, epochs
        self.batch_size, self.weight_decay, self.patience, self.validation_fraction, self.seed = batch_size, weight_decay, patience, validation_fraction, seed
        self.net_: Optional[Sequential] = None

    def fit(self, X, y, sample_weight=None) -> "LSTMForecaster":
        seq, labels, _ = make_sequences(X, y, self.lookback)
        assert labels is not None                                       # labels are always returned when y is given
        target: np.ndarray = labels
        if len(seq) < 20:
            raise ValidationError("too few complete windows to train on", details={"windows": len(seq), "lookback": self.lookback})
        if self.task == "classification" and set(np.unique(target)) - {0.0, 1.0}:
            raise ValidationError("classification labels must be 0/1")
        n_tr = len(seq) - int(len(seq) * self.validation_fraction)
        self._fit_scale(seq[:n_tr])
        self._ymu, self._ysd = (0.0, 1.0) if self.task == "classification" else (float(target[:n_tr].mean()), float(target[:n_tr].std()) or 1.0)
        rng = np.random.default_rng(self.seed)
        layers: List[Layer] = [LSTM(seq.shape[2], self.hidden, rng)]
        if self.dropout > 0:
            layers.append(Dropout(self.dropout, rng))
        layers.append(Dense(self.hidden, 1, rng, init="glorot"))
        self.net_ = Sequential(layers, "bce" if self.task == "classification" else "huber", Adam(self.lr, weight_decay=self.weight_decay), self.seed)
        self.net_.fit(self._scale(seq), (target - self._ymu) / self._ysd, self.epochs, self.batch_size, self.validation_fraction, self.patience)
        return self

    def _raw(self, X) -> np.ndarray:
        if self.net_ is None:
            raise ModelError("model is not fitted")
        X = np.asarray(X, float)
        seq, _, last = make_sequences(X, None, self.lookback)
        out = np.full(len(X), np.nan)
        if len(seq):
            out[last] = self.net_.predict(self._scale(seq))[:, 0]
        return out

    def predict_proba(self, X) -> np.ndarray:
        if self.task != "classification":
            raise ModelError("predict_proba is only available for classification")
        return sigmoid(self._raw(X))

    def predict(self, X) -> np.ndarray:
        raw = self._raw(X)
        if self.task == "classification":
            return np.where(np.isnan(raw), np.nan, (raw >= 0).astype(float))
        return raw * self._ysd + self._ymu


class Autoencoder(_Scaled):
    """Learns to compress the feature vector to ``encoding_dim`` numbers and rebuild it. Two uses: ``encode`` gives
    non-linear factors; ``reconstruction_error`` is high for rows unlike anything seen in training, i.e. an anomaly /
    regime-novelty score (a model trained on calm markets reconstructs a crash badly)."""

    def __init__(self, encoding_dim: int = 2, hidden: int = 16, lr: float = 3e-3, epochs: int = 200, batch_size: int = 64, patience: int = 15,
                 seed: Optional[int] = 0) -> None:
        if encoding_dim < 1 or hidden < 1:
            raise ValidationError("encoding_dim and hidden must be >= 1")
        self.encoding_dim, self.hidden, self.lr, self.epochs, self.batch_size, self.patience, self.seed = encoding_dim, hidden, lr, epochs, batch_size, patience, seed
        self.net_: Optional[Sequential] = None

    def fit(self, X, y: Any = None) -> "Autoencoder":
        X = np.asarray(X, float)
        if X.ndim != 2 or len(X) < 10:
            raise ValidationError("X must be 2-D with at least 10 rows")
        self._fit_scale(X[:len(X) - int(len(X) * 0.2)])
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]
        self._encoder = [Dense(d, self.hidden, rng), Activation("tanh"), Dense(self.hidden, self.encoding_dim, rng, init="glorot")]
        decoder = [Activation("tanh"), Dense(self.encoding_dim, self.hidden, rng), Activation("tanh"), Dense(self.hidden, d, rng, init="glorot")]
        self.net_ = Sequential(self._encoder + decoder, "mse", Adam(self.lr), self.seed)
        Z = self._scale(X)
        self.net_.fit(Z, Z, self.epochs, self.batch_size, 0.2, self.patience)
        self.threshold_ = float(np.quantile(self.reconstruction_error(X), 0.99))        # 99th percentile of training error
        return self

    def _check(self) -> Sequential:
        if self.net_ is None:
            raise ModelError("model is not fitted")
        return self.net_

    def encode(self, X) -> np.ndarray:
        self._check()
        out = self._scale(np.atleast_2d(np.asarray(X, float)))
        for layer in self._encoder:
            out = layer.forward(out)
        return out

    def reconstruction_error(self, X) -> np.ndarray:
        Z = self._scale(np.atleast_2d(np.asarray(X, float)))
        return ((self._check().predict(Z) - Z) ** 2).mean(axis=1)

    def is_anomaly(self, X) -> np.ndarray:
        return self.reconstruction_error(X) > self.threshold_
