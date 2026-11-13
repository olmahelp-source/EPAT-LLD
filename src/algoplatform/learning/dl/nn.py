"""A small neural-network library in numpy: layers with hand-written backward passes, Adam, and a training loop.

Why not a framework: the platform's rule is numpy + pandas only, and the networks that make sense on daily financial
data are small (thousands of parameters, not millions), so numpy trains them in seconds. Every backward pass here is
verified against numerical gradients in the tests. For large models, train elsewhere and plug the result in through
``research.models.SklearnAdapter`` or ``strategy.PolicyStrategy``.

Shapes: Dense works on the last axis, so it accepts (batch, features) and (batch, time, features). LSTM takes
(batch, time, features) and returns (batch, hidden), or (batch, time, hidden) with ``return_sequences=True``.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...foundation.core.exceptions import ModelError, ValidationError
from ...foundation.utils.numeric import sigmoid


class Layer:
    """Base layer: ``forward`` caches what ``backward`` needs; ``params`` and ``grads`` line up one to one."""
    params: List[np.ndarray]
    grads: List[np.ndarray]

    def __init__(self) -> None:
        self.params, self.grads = [], []

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, n_in: int, n_out: int, rng: Optional[np.random.Generator] = None, init: str = "he") -> None:
        super().__init__()
        if n_in < 1 or n_out < 1:
            raise ValidationError("layer sizes must be >= 1")
        rng = rng or np.random.default_rng(0)
        scale = math.sqrt(2.0 / n_in) if init == "he" else math.sqrt(2.0 / (n_in + n_out))
        self.W, self.b = rng.normal(0.0, scale, (n_in, n_out)), np.zeros(n_out)
        self.params, self.grads = [self.W, self.b], [np.zeros_like(self.W), np.zeros_like(self.b)]

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        n_in, n_out = self.W.shape
        self.grads[0][...] = self._x.reshape(-1, n_in).T @ grad.reshape(-1, n_out)
        self.grads[1][...] = grad.reshape(-1, n_out).sum(axis=0)
        return grad @ self.W.T


class Activation(Layer):
    def __init__(self, kind: str = "relu") -> None:
        super().__init__()
        if kind not in ("relu", "tanh", "sigmoid", "leaky_relu"):
            raise ValidationError(f"unknown activation '{kind}'")
        self.kind = kind

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self._x = x
        if self.kind == "relu":
            self._y = np.maximum(x, 0.0)
        elif self.kind == "leaky_relu":
            self._y = np.where(x > 0, x, 0.01 * x)
        elif self.kind == "tanh":
            self._y = np.tanh(x)
        else:
            self._y = sigmoid(x)
        return self._y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.kind == "relu":
            return grad * (self._x > 0)
        if self.kind == "leaky_relu":
            return grad * np.where(self._x > 0, 1.0, 0.01)
        if self.kind == "tanh":
            return grad * (1.0 - self._y ** 2)
        return grad * self._y * (1.0 - self._y)


class Dropout(Layer):
    """Inverted dropout: active only while training, so prediction needs no rescaling."""

    def __init__(self, rate: float = 0.1, rng: Optional[np.random.Generator] = None) -> None:
        super().__init__()
        if not 0 <= rate < 1:
            raise ValidationError("dropout rate must be in [0, 1)")
        self.rate, self._rng = rate, rng or np.random.default_rng(0)
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        if not training or self.rate == 0:
            self._mask = None
            return x
        self._mask = (self._rng.random(x.shape) >= self.rate) / (1.0 - self.rate)
        return x * self._mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad if self._mask is None else grad * self._mask


class LSTM(Layer):
    """Long short-term memory layer with full back-propagation through time. Gate order in the weights: input, forget,
    cell candidate, output. The forget-gate bias starts at 1 so that memory is kept by default early in training."""

    def __init__(self, n_in: int, hidden: int, rng: Optional[np.random.Generator] = None, return_sequences: bool = False) -> None:
        super().__init__()
        if n_in < 1 or hidden < 1:
            raise ValidationError("layer sizes must be >= 1")
        rng = rng or np.random.default_rng(0)
        self.n_in, self.hidden, self.return_sequences = n_in, hidden, return_sequences
        self.W = rng.normal(0.0, 1.0 / math.sqrt(n_in + hidden), (n_in + hidden, 4 * hidden))
        self.b = np.zeros(4 * hidden)
        self.b[hidden:2 * hidden] = 1.0
        self.params, self.grads = [self.W, self.b], [np.zeros_like(self.W), np.zeros_like(self.b)]

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        if x.ndim != 3 or x.shape[2] != self.n_in:
            raise ValidationError(f"LSTM expects (batch, time, {self.n_in}), got {x.shape}")
        B, T, _ = x.shape
        H = self.hidden
        h, c = np.zeros((B, H)), np.zeros((B, H))
        self._cache = []
        outs = np.empty((B, T, H))
        for t in range(T):
            z_in = np.concatenate([x[:, t], h], axis=1)
            z = z_in @ self.W + self.b
            i, f, g, o = sigmoid(z[:, :H]), sigmoid(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), sigmoid(z[:, 3 * H:])
            c_prev = c
            c = f * c_prev + i * g
            tc = np.tanh(c)
            h = o * tc
            outs[:, t] = h
            self._cache.append((z_in, i, f, g, o, c_prev, tc))
        return outs if self.return_sequences else h

    def backward(self, grad: np.ndarray) -> np.ndarray:
        T, H = len(self._cache), self.hidden
        B = self._cache[0][0].shape[0]
        dW, db = np.zeros_like(self.W), np.zeros_like(self.b)
        dx = np.empty((B, T, self.n_in))
        dh_next, dc_next = np.zeros((B, H)), np.zeros((B, H))
        for t in range(T - 1, -1, -1):
            z_in, i, f, g, o, c_prev, tc = self._cache[t]
            dh = dh_next + (grad[:, t] if self.return_sequences else (grad if t == T - 1 else 0.0))
            dc = dh * o * (1.0 - tc ** 2) + dc_next
            dz = np.concatenate([dc * g * i * (1 - i), dc * c_prev * f * (1 - f), dc * i * (1 - g ** 2), dh * tc * o * (1 - o)], axis=1)
            dW += z_in.T @ dz
            db += dz.sum(axis=0)
            d_in = dz @ self.W.T
            dx[:, t], dh_next, dc_next = d_in[:, :self.n_in], d_in[:, self.n_in:], dc * f
        self.grads[0][...], self.grads[1][...] = dW, db
        return dx


# ---- losses: each returns (mean loss, gradient w.r.t. the network output) ---------------------------------------------------
def _weights(w: Optional[np.ndarray], n: int) -> np.ndarray:
    return np.full((n, 1), 1.0 / n) if w is None else (np.asarray(w, float) / np.sum(w)).reshape(n, 1)


def mse_loss(out: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    wn = _weights(w, len(y))
    d = out - y
    return float((wn * d ** 2).sum() / out.shape[1]), 2.0 * wn * d / out.shape[1]


def huber_loss(out: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None, delta: float = 1.0) -> Tuple[float, np.ndarray]:
    """Quadratic near zero, linear in the tails: one fat-tailed return cannot dominate the gradient."""
    wn = _weights(w, len(y))
    d = out - y
    small = np.abs(d) <= delta
    loss = np.where(small, 0.5 * d ** 2, delta * (np.abs(d) - 0.5 * delta))
    return float((wn * loss).sum() / out.shape[1]), wn * np.where(small, d, delta * np.sign(d)) / out.shape[1]


def bce_logits_loss(out: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """Binary cross-entropy on logits (numerically stable form)."""
    wn = _weights(w, len(y))
    loss = np.maximum(out, 0.0) - out * y + np.log1p(np.exp(-np.abs(out)))
    return float((wn * loss).sum() / out.shape[1]), wn * (sigmoid(out) - y) / out.shape[1]


LOSSES: Dict[str, Callable[..., Tuple[float, np.ndarray]]] = {"mse": mse_loss, "huber": huber_loss, "bce": bce_logits_loss}


class Adam:
    """Adam with decoupled weight decay (AdamW) and global-norm gradient clipping (essential for recurrent layers)."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0,
                 clip_norm: Optional[float] = 5.0) -> None:
        if lr <= 0 or weight_decay < 0:
            raise ValidationError("lr must be positive and weight_decay >= 0")
        self.lr, self.beta1, self.beta2, self.eps, self.weight_decay, self.clip_norm = lr, beta1, beta2, eps, weight_decay, clip_norm
        self._m: List[np.ndarray] = []
        self._v: List[np.ndarray] = []
        self._t = 0

    def step(self, params: Sequence[np.ndarray], grads: Sequence[np.ndarray]) -> None:
        if not self._m:
            self._m, self._v = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
        scale = 1.0
        if self.clip_norm:
            norm = math.sqrt(sum(float((g ** 2).sum()) for g in grads))
            scale = min(1.0, self.clip_norm / (norm + 1e-12))
        self._t += 1
        for p, g, m, v in zip(params, grads, self._m, self._v):
            g = g * scale
            m[...] = self.beta1 * m + (1 - self.beta1) * g
            v[...] = self.beta2 * v + (1 - self.beta2) * g ** 2
            m_hat, v_hat = m / (1 - self.beta1 ** self._t), v / (1 - self.beta2 ** self._t)
            p -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + (self.weight_decay * p if p.ndim > 1 else 0.0))   # no decay on biases


class Sequential:
    """A stack of layers plus the training loop.

    ``fit`` holds out the **last** ``validation_fraction`` of the rows (data is assumed to be in time order, so the
    validation set is strictly later than the training set), stops when validation loss has not improved for
    ``patience`` epochs and restores the best weights. A non-finite loss raises ``ModelError`` instead of silently
    producing NaN predictions."""

    def __init__(self, layers: Sequence[Layer], loss: str = "mse", optimizer: Optional[Adam] = None, seed: Optional[int] = 0) -> None:
        if not layers:
            raise ValidationError("a network needs at least one layer")
        if loss not in LOSSES:
            raise ValidationError(f"unknown loss '{loss}'", details={"valid": sorted(LOSSES)})
        self.layers, self.loss_name, self.optimizer = list(layers), loss, optimizer or Adam()
        self._rng = np.random.default_rng(seed)
        self.history: Dict[str, List[float]] = {"loss": [], "val_loss": []}

    # ---- plumbing ---------------------------------------------------------------------------------------------------------
    @property
    def params(self) -> List[np.ndarray]:
        return [p for layer in self.layers for p in layer.params]

    @property
    def grads(self) -> List[np.ndarray]:
        return [g for layer in self.layers for g in layer.grads]

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))

    def get_weights(self) -> List[np.ndarray]:
        return [p.copy() for p in self.params]

    def set_weights(self, weights: Sequence[np.ndarray]) -> None:
        params = self.params
        if len(weights) != len(params) or any(w.shape != p.shape for w, p in zip(weights, params)):
            raise ModelError("weights do not match the network's shapes")
        for p, w in zip(params, weights):
            p[...] = w

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def loss(self, X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
        return LOSSES[self.loss_name](self.forward(X), self._target(y), w)[0]

    @staticmethod
    def _target(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, float)
        return y.reshape(len(y), -1)

    # ---- training ---------------------------------------------------------------------------------------------------------
    def fit(self, X, y, epochs: int = 100, batch_size: int = 32, validation_fraction: float = 0.2, patience: int = 10,
            shuffle: bool = True, sample_weight=None) -> "Sequential":
        X, y = np.asarray(X, float), self._target(y)
        if len(X) != len(y) or len(X) < 2:
            raise ValidationError("X and y must have the same number of rows (at least 2)")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValidationError("X and y must be finite")
        if epochs < 1 or batch_size < 1 or not 0 <= validation_fraction < 0.9:
            raise ValidationError("need epochs >= 1, batch_size >= 1, 0 <= validation_fraction < 0.9")
        w = None if sample_weight is None else np.asarray(sample_weight, float)
        n_val = int(len(X) * validation_fraction)
        n_tr = len(X) - n_val
        loss_fn = LOSSES[self.loss_name]
        best, best_weights, stale = math.inf, self.get_weights(), 0
        for _ in range(epochs):
            order = self._rng.permutation(n_tr) if shuffle else np.arange(n_tr)
            total = 0.0
            for i in range(0, n_tr, batch_size):
                idx = order[i:i + batch_size]
                value, grad = loss_fn(self.forward(X[idx], training=True), y[idx], None if w is None else w[idx])
                if not math.isfinite(value):
                    self.set_weights(best_weights)
                    raise ModelError("training diverged (non-finite loss); lower the learning rate or scale the inputs")
                self.backward(grad)
                self.optimizer.step(self.params, self.grads)
                total += value * len(idx)
            train_loss = total / n_tr
            val_loss = self.loss(X[n_tr:], y[n_tr:], None if w is None else w[n_tr:]) if n_val else train_loss
            self.history["loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            if val_loss < best - 1e-9:
                best, best_weights, stale = val_loss, self.get_weights(), 0
            else:
                stale += 1
                if stale >= patience:
                    break
        self.set_weights(best_weights)
        self.best_val_loss_ = best
        return self

    def predict(self, X, batch_size: int = 1024) -> np.ndarray:
        X = np.asarray(X, float)
        return np.concatenate([self.forward(X[i:i + batch_size]) for i in range(0, len(X), batch_size)]) if len(X) else np.empty((0, 1))


def gradient_check(model: Sequential, X: np.ndarray, y: np.ndarray, eps: float = 1e-5, n_checks: int = 30, seed: int = 0) -> float:
    """Largest relative difference between back-propagated and numerical gradients over ``n_checks`` random weights.
    Values around 1e-6 or smaller mean the backward pass is correct. (Use a network without dropout.)"""
    rng = np.random.default_rng(seed)
    yt = model._target(y)
    _, grad = LOSSES[model.loss_name](model.forward(X), yt, None)
    model.backward(grad)
    analytic = [g.copy() for g in model.grads]
    worst = 0.0
    for _ in range(n_checks):
        k = int(rng.integers(len(model.params)))
        p = model.params[k]
        idx: Any = tuple(int(rng.integers(s)) for s in p.shape)
        old = p[idx]
        p[idx] = old + eps
        up = model.loss(X, yt)
        p[idx] = old - eps
        down = model.loss(X, yt)
        p[idx] = old
        numeric = (up - down) / (2 * eps)
        worst = max(worst, abs(numeric - analytic[k][idx]) / max(abs(numeric) + abs(analytic[k][idx]), 1e-8))
    return worst
