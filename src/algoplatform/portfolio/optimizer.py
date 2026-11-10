"""Portfolio optimisers (numpy only): long-only, capped, no external solver.

All routines take a covariance matrix ``cov`` (n x n) and return weights that sum
to 1. Constraints are enforced by projecting onto the capped simplex.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.exceptions import OptimizationError, ValidationError


def _check_cov(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValidationError("cov must be square")
    if not np.all(np.isfinite(cov)):
        raise ValidationError("cov contains NaN/inf")
    return (cov + cov.T) / 2.0


def project_capped_simplex(v: np.ndarray, cap: float = 1.0) -> np.ndarray:
    """Euclidean projection onto {w: sum w = 1, 0 <= w <= cap} by bisection on the shift."""
    n = len(v)
    if cap * n < 1 - 1e-12:
        raise OptimizationError("cap too small: weights cannot sum to 1", details={"cap": cap, "n": n})
    lo, hi = v.min() - cap, v.max()
    for _ in range(100):
        tau = (lo + hi) / 2
        s = np.clip(v - tau, 0.0, cap).sum()
        if s > 1:
            lo = tau
        else:
            hi = tau
    w = np.clip(v - (lo + hi) / 2, 0.0, cap)
    return w / w.sum()


def _pg(grad, w0: np.ndarray, lr: float, cap: float, iters: int = 5000, tol: float = 1e-11) -> np.ndarray:
    w = w0
    for _ in range(iters):
        nw = project_capped_simplex(w - lr * grad(w), cap)
        if np.max(np.abs(nw - w)) < tol:
            return nw
        w = nw
    return w


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def inverse_volatility(cov: np.ndarray) -> np.ndarray:
    cov = _check_cov(cov)
    iv = 1.0 / np.sqrt(np.clip(np.diag(cov), 1e-16, None))
    return iv / iv.sum()


def min_variance(cov: np.ndarray, cap: float = 1.0) -> np.ndarray:
    cov = _check_cov(cov)
    n = len(cov)
    lr = 1.0 / (2.0 * max(np.linalg.eigvalsh(cov).max(), 1e-12))
    return _pg(lambda w: 2 * cov @ w, equal_weight(n), lr, cap)


def mean_variance(mu: np.ndarray, cov: np.ndarray, risk_aversion: float = 5.0, cap: float = 1.0) -> np.ndarray:
    """Maximise mu'w - (lambda/2) w'Sigma w over the capped simplex."""
    cov, mu = _check_cov(cov), np.asarray(mu, dtype=float)
    if len(mu) != len(cov):
        raise ValidationError("mu and cov size mismatch")
    lr = 1.0 / (risk_aversion * max(np.linalg.eigvalsh(cov).max(), 1e-12))
    return _pg(lambda w: -(mu - risk_aversion * cov @ w), equal_weight(len(mu)), lr, cap)


def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0, cap: float = 1.0, grid: int = 40) -> np.ndarray:
    """Best Sharpe on the mean-variance frontier (long-only), scanning risk aversion."""
    mu = np.asarray(mu, dtype=float)
    cov = _check_cov(cov)
    best_w, best_s = equal_weight(len(mu)), -np.inf
    for lam in np.geomspace(0.05, 500.0, grid):
        w = mean_variance(mu - rf, cov, lam, cap)
        vol = float(np.sqrt(w @ cov @ w))
        s = float((w @ mu - rf) / vol) if vol > 0 else -np.inf
        if s > best_s:
            best_w, best_s = w, s
    return best_w


def risk_parity(cov: np.ndarray, budget: Optional[np.ndarray] = None, iters: int = 5000, tol: float = 1e-10) -> np.ndarray:
    """Equal (or budgeted) risk contribution via multiplicative fixed-point updates."""
    cov = _check_cov(cov)
    n = len(cov)
    b = np.full(n, 1.0 / n) if budget is None else np.asarray(budget, float) / np.sum(budget)
    w = inverse_volatility(cov)
    for _ in range(iters):
        rc = w * (cov @ w)
        total = rc.sum()
        if total <= 0:
            raise OptimizationError("degenerate covariance in risk parity")
        ratio = (b * total) / np.clip(rc, 1e-18, None)
        nw = w * ratio ** 0.5
        nw /= nw.sum()
        if np.max(np.abs(nw - w)) < tol:
            return nw
        w = nw
    return w


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    cov = _check_cov(cov)
    port_var = float(w @ cov @ w)
    return w * (cov @ w) / port_var if port_var > 0 else np.zeros_like(w)


def black_litterman(cov: np.ndarray, market_weights: np.ndarray, views_P: np.ndarray, views_Q: np.ndarray,
                    risk_aversion: float = 2.5, tau: float = 0.05, view_conf: Optional[np.ndarray] = None) -> np.ndarray:
    """Posterior expected returns given absolute/relative views (P mu = Q)."""
    cov = _check_cov(cov)
    P, Q = np.atleast_2d(views_P).astype(float), np.asarray(views_Q, float)
    pi = risk_aversion * cov @ market_weights                    # implied equilibrium returns
    omega = np.diag(np.diag(tau * P @ cov @ P.T)) if view_conf is None else np.diag(view_conf)
    inv = np.linalg.inv(np.linalg.inv(tau * cov) + P.T @ np.linalg.inv(omega) @ P)
    return inv @ (np.linalg.inv(tau * cov) @ pi + P.T @ np.linalg.inv(omega) @ Q)


OPTIMIZERS: Dict[str, str] = {"equal": "equal_weight", "inverse_vol": "inverse_volatility",
                              "min_variance": "min_variance", "risk_parity": "risk_parity"}
