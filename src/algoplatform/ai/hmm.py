"""Hidden Markov models for market regimes.

A Gaussian HMM assumes the market is in one of a few hidden states (calm, turbulent, ...), each with its own return
distribution, and switches between them with fixed probabilities. Fitting it (Baum-Welch) recovers the states and the
switching matrix from returns alone.

Two ways to read the states back, and the difference matters:

* :meth:`GaussianHMM.filter`  P(state at t | data up to t). **Causal**: safe to trade on and to backtest with.
* :meth:`GaussianHMM.predict_proba` / :meth:`predict`  smoothed / Viterbi, conditioned on the *whole* sample, including
  the future of each bar. For analysing history only; using them in a backtest is look-ahead.

The model parameters themselves are estimated on a training sample: fit on the past, then ``filter`` forward.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.exceptions import ModelError, ValidationError


def _obs(X) -> np.ndarray:
    X = np.asarray(X, float)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValidationError("observations must be a finite 1-D or 2-D array")
    return X


class GaussianHMM:
    """HMM with diagonal-covariance Gaussian emissions, fitted by expectation-maximisation with several restarts."""

    def __init__(self, n_states: int = 2, n_iter: int = 100, tol: float = 1e-5, n_init: int = 3, min_var: float = 1e-10,
                 seed: Optional[int] = 0) -> None:
        if n_states < 2 or n_iter < 1 or n_init < 1:
            raise ValidationError("need n_states >= 2, n_iter >= 1, n_init >= 1")
        self.n_states, self.n_iter, self.tol, self.n_init, self.min_var, self.seed = n_states, n_iter, tol, n_init, min_var, seed
        self.means_: Optional[np.ndarray] = None

    # ---- core recursions (scaled, so long series do not underflow) ---------------------------------------------------------
    def _emission(self, X: np.ndarray):
        """-> (b, shift): b[t, k] is the emission density divided by exp(shift[t]) (each row's maximum is 1)."""
        d = X[:, None, :] - self.means_[None, :, :]
        logb = -0.5 * (np.log(2 * math.pi * self.vars_)[None, :, :] + d ** 2 / self.vars_[None, :, :]).sum(axis=2)
        shift = logb.max(axis=1)
        return np.exp(logb - shift[:, None]), shift

    def _forward(self, b: np.ndarray):
        T, K = b.shape
        alpha, scale = np.empty((T, K)), np.empty(T)
        a = self.start_ * b[0]
        for t in range(T):
            if t:
                a = (alpha[t - 1] @ self.transition_) * b[t]
            scale[t] = a.sum()
            if scale[t] <= 0 or not np.isfinite(scale[t]):
                raise ModelError("HMM forward pass degenerated (an observation is impossible under every state)")
            alpha[t] = a / scale[t]
        return alpha, scale

    def _backward(self, b: np.ndarray, scale: np.ndarray) -> np.ndarray:
        T, K = b.shape
        beta = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            beta[t] = (self.transition_ @ (b[t + 1] * beta[t + 1])) / scale[t + 1]
        return beta

    def _init_params(self, X: np.ndarray, rng: np.random.Generator) -> None:
        K = self.n_states
        order = np.argsort(np.abs(X[:, 0] - np.median(X[:, 0])) + rng.normal(0, 1e-9 + 0.1 * X[:, 0].std(), len(X)))
        chunks = np.array_split(order, K)                               # seed states by distance from the median: quiet vs wild
        self.means_ = np.array([X[c].mean(axis=0) for c in chunks])
        self.vars_ = np.array([np.maximum(X[c].var(axis=0), self.min_var) + 1e-12 for c in chunks])
        self.transition_ = np.full((K, K), 0.1 / (K - 1)) + np.eye(K) * (0.9 - 0.1 / (K - 1))
        self.start_ = np.full(K, 1.0 / K)

    def fit(self, X) -> "GaussianHMM":
        X = _obs(X)
        if len(X) < 10 * self.n_states:
            raise ValidationError("too few observations for this many states", details={"n": len(X), "states": self.n_states})
        rng = np.random.default_rng(self.seed)
        best: Optional[tuple] = None
        for _ in range(self.n_init):
            self._init_params(X, rng)
            prev = -math.inf
            try:
                for it in range(self.n_iter):
                    b, shift = self._emission(X)
                    alpha, scale = self._forward(b)
                    loglik = float(np.log(scale).sum() + shift.sum())
                    beta = self._backward(b, scale)
                    gamma = alpha * beta
                    gamma /= gamma.sum(axis=1, keepdims=True)
                    xi = (alpha[:-1, :, None] * self.transition_[None] * (b[1:] * beta[1:])[:, None, :]) / scale[1:, None, None]
                    self.start_ = gamma[0] / gamma[0].sum()
                    trans = xi.sum(axis=0)
                    self.transition_ = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1e-300)
                    w = gamma.sum(axis=0)
                    self.means_ = (gamma.T @ X) / np.maximum(w[:, None], 1e-300)
                    d = X[:, None, :] - self.means_[None]
                    self.vars_ = np.maximum((gamma[:, :, None] * d ** 2).sum(axis=0) / np.maximum(w[:, None], 1e-300), self.min_var)
                    if loglik - prev < self.tol * max(1.0, abs(loglik)) and it > 0:
                        break
                    prev = loglik
            except ModelError:
                continue                                                # a degenerate restart: try the next one
            final = self.score(X)
            if best is None or final > best[0]:
                best = (final, self.start_.copy(), self.transition_.copy(), self.means_.copy(), self.vars_.copy())
        if best is None:
            raise ModelError("HMM could not be fitted from any starting point")
        self.loglik_, self.start_, self.transition_, self.means_, self.vars_ = best
        self.n_features_ = X.shape[1]
        return self

    def _check(self, X) -> np.ndarray:
        if self.means_ is None:
            raise ModelError("HMM is not fitted")
        X = _obs(X)
        if X.shape[1] != self.means_.shape[1]:
            raise ValidationError(f"expected {self.means_.shape[1]} feature(s), got {X.shape[1]}")
        return X

    # ---- inference -----------------------------------------------------------------------------------------------------------
    def score(self, X) -> float:
        """Log-likelihood of the observations."""
        b, shift = self._emission(self._check(X))
        return float(np.log(self._forward(b)[1]).sum() + shift.sum())

    def filter(self, X) -> np.ndarray:
        """Causal state probabilities: row t uses observations 0..t only."""
        return self._forward(self._emission(self._check(X))[0])[0]

    def predict_proba(self, X) -> np.ndarray:
        """Smoothed state probabilities (uses the whole sample: analysis only, not for trading decisions)."""
        b, _ = self._emission(self._check(X))
        alpha, scale = self._forward(b)
        gamma = alpha * self._backward(b, scale)
        return gamma / gamma.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """Most likely state path (Viterbi). Uses the whole sample: analysis only."""
        b, _ = self._emission(self._check(X))
        T, K = b.shape
        with np.errstate(divide="ignore"):
            logb, logA, logpi = np.log(b), np.log(self.transition_), np.log(self.start_)
        delta, back = logpi + logb[0], np.zeros((T, K), dtype=int)
        for t in range(1, T):
            cand = delta[:, None] + logA
            back[t] = cand.argmax(axis=0)
            delta = cand.max(axis=0) + logb[t]
        path = np.empty(T, dtype=int)
        path[-1] = int(delta.argmax())
        for t in range(T - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path

    def next_state_proba(self, X) -> np.ndarray:
        """P(state at the next bar | data so far)."""
        return self.filter(X)[-1] @ self.transition_

    def expected_durations(self) -> np.ndarray:
        """Average number of bars spent in each state once entered: 1 / (1 - P(stay))."""
        if self.means_ is None:
            raise ModelError("HMM is not fitted")
        return 1.0 / np.maximum(1.0 - np.diag(self.transition_), 1e-12)


class RegimeDetector:
    """Market regimes from returns, with states ordered from calmest (0) to most volatile.

    ``fit`` learns the regimes on a training sample; ``regimes`` / ``current`` then label any series causally."""

    def __init__(self, n_states: int = 2, periods: float = 252.0, seed: Optional[int] = 0, **hmm_kwargs: Any) -> None:
        self.n_states, self.periods = n_states, periods
        self.hmm = GaussianHMM(n_states, seed=seed, **hmm_kwargs)
        self._order: Optional[np.ndarray] = None

    @staticmethod
    def _returns(returns) -> np.ndarray:
        r = np.asarray(returns, float).ravel()
        if not np.isfinite(r).all():
            raise ValidationError("returns must be finite (drop the first NaN from pct_change)")
        return r

    def fit(self, returns) -> "RegimeDetector":
        self.hmm.fit(self._returns(returns))
        self._order = np.argsort(self.hmm.vars_[:, 0])
        return self

    @property
    def labels(self) -> List[str]:
        return ["calm", "turbulent"] if self.n_states == 2 else ["calm"] + [f"regime_{i}" for i in range(1, self.n_states - 1)] + ["turbulent"]

    def probabilities(self, returns) -> np.ndarray:
        """Causal regime probabilities, columns ordered calm -> turbulent."""
        if self._order is None:
            raise ModelError("RegimeDetector is not fitted")
        return self.hmm.filter(self._returns(returns))[:, self._order]

    def regimes(self, returns) -> np.ndarray:
        return self.probabilities(returns).argmax(axis=1)

    def current(self, returns) -> Dict[str, Any]:
        p = self.probabilities(returns)[-1]
        k = int(p.argmax())
        return {"state": k, "label": self.labels[k], "probability": float(p[k]), "probabilities": dict(zip(self.labels, map(float, p)))}

    def describe(self) -> List[Dict[str, float]]:
        """Per regime: annualised mean and volatility, and how long it typically lasts."""
        if self._order is None:
            raise ModelError("RegimeDetector is not fitted")
        dur = self.hmm.expected_durations()
        return [{"label": self.labels[i], "ann_return": float(self.hmm.means_[k, 0] * self.periods),
                 "ann_vol": float(math.sqrt(self.hmm.vars_[k, 0] * self.periods)), "expected_bars": float(dur[k])}
                for i, k in enumerate(self._order)]
