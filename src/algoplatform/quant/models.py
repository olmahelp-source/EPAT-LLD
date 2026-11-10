"""Quantitative models: options pricing, volatility estimators, covariance, Kalman filter."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from ..core.exceptions import StatisticsError, ValidationError
from ..stats.special import norm_cdf, norm_pdf


# ---- Black-Scholes-Merton ---------------------------------------------------------------
def _d1d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    if min(S, K, T, sigma) <= 0:
        raise ValidationError("S, K, T, sigma must be positive")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def bs_price(S: float, K: float, T: float, r: float, sigma: float, kind: str = "call", q: float = 0.0) -> float:
    if kind not in ("call", "put"):
        raise ValidationError("kind must be 'call' or 'put'")
    if T <= 0:                                               # at expiry: intrinsic value
        return max(S - K, 0.0) if kind == "call" else max(K - S, 0.0)
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    if kind == "call":
        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, kind: str = "call", q: float = 0.0) -> Dict[str, float]:
    """Delta, gamma, vega (per 1 vol point = 0.01), theta (per day), rho (per 1% rate)."""
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    sq = math.sqrt(T)
    pdf = norm_pdf(d1)
    disc_q, disc_r = math.exp(-q * T), math.exp(-r * T)
    gamma = disc_q * pdf / (S * sigma * sq)
    vega = S * disc_q * pdf * sq / 100.0
    if kind == "call":
        delta = disc_q * norm_cdf(d1)
        theta = (-S * disc_q * pdf * sigma / (2 * sq) - r * K * disc_r * norm_cdf(d2) + q * S * disc_q * norm_cdf(d1)) / 365.0
        rho = K * T * disc_r * norm_cdf(d2) / 100.0
    else:
        delta = -disc_q * norm_cdf(-d1)
        theta = (-S * disc_q * pdf * sigma / (2 * sq) + r * K * disc_r * norm_cdf(-d2) - q * S * disc_q * norm_cdf(-d1)) / 365.0
        rho = -K * T * disc_r * norm_cdf(-d2) / 100.0
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(price: float, S: float, K: float, T: float, r: float, kind: str = "call", q: float = 0.0,
                tol: float = 1e-8, max_iter: int = 200) -> float:
    """Bisection implied volatility (robust); raises if the price violates no-arbitrage bounds."""
    lo_p = bs_price(S, K, T, r, 1e-6, kind, q)
    hi_p = bs_price(S, K, T, r, 5.0, kind, q)
    if not lo_p - 1e-9 <= price <= hi_p + 1e-9:
        raise StatisticsError("price outside no-arbitrage range for implied vol", details={"price": price})
    lo, hi = 1e-6, 5.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        p = bs_price(S, K, T, r, mid, kind, q)
        if abs(p - price) < tol:
            return mid
        lo, hi = (mid, hi) if p < price else (lo, mid)
    return (lo + hi) / 2


def put_call_parity_gap(call: float, put: float, S: float, K: float, T: float, r: float, q: float = 0.0) -> float:
    """C - P - (S e^{-qT} - K e^{-rT}); ~0 when no arbitrage."""
    return call - put - (S * math.exp(-q * T) - K * math.exp(-r * T))


# ---- volatility ------------------------------------------------------------------------------
def ewma_volatility(returns, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA volatility series (per-period)."""
    r = np.asarray(returns, float)
    var = np.empty(len(r))
    var[0] = np.nanvar(r[: min(len(r), 20)]) or r[0] ** 2
    for i in range(1, len(r)):
        var[i] = lam * var[i - 1] + (1 - lam) * r[i - 1] ** 2
    return np.sqrt(var)


def parkinson_vol(high, low) -> float:
    h, l = np.asarray(high, float), np.asarray(low, float)
    return float(math.sqrt(np.mean(np.log(h / l) ** 2) / (4 * math.log(2))))


def garman_klass_vol(open_, high, low, close) -> float:
    o, h, l, c = (np.asarray(v, float) for v in (open_, high, low, close))
    v = 0.5 * np.log(h / l) ** 2 - (2 * math.log(2) - 1) * np.log(c / o) ** 2
    return float(math.sqrt(max(np.mean(v), 0.0)))


def realized_vol(returns, periods_per_year: float = 252.0) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    return float(r.std(ddof=1) * math.sqrt(periods_per_year)) if len(r) > 1 else 0.0


def garch11_filter(returns, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Conditional volatility from GARCH(1,1) with given parameters (requires alpha+beta<1)."""
    if alpha < 0 or beta < 0 or omega <= 0 or alpha + beta >= 1:
        raise ValidationError("need omega>0, alpha,beta>=0, alpha+beta<1")
    r = np.asarray(returns, float)
    var = np.empty(len(r))
    var[0] = omega / (1 - alpha - beta)
    for t in range(1, len(r)):
        var[t] = omega + alpha * r[t - 1] ** 2 + beta * var[t - 1]
    return np.sqrt(var)


def garch11_fit(returns, grid: int = 12) -> Dict[str, float]:
    """Coarse variance-targeting grid MLE for GARCH(1,1); dependency-free and robust."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if len(r) < 100:
        raise StatisticsError("need >= 100 returns to fit GARCH")
    v = r.var()
    best = (-np.inf, None)
    for a in np.linspace(0.02, 0.25, grid):
        for b in np.linspace(0.5, 0.97, grid):
            if a + b >= 0.999:
                continue
            omega = v * (1 - a - b)
            sig = garch11_filter(r, omega, a, b)
            ll = -0.5 * np.sum(np.log(sig ** 2) + r ** 2 / sig ** 2)
            if ll > best[0]:
                best = (ll, (omega, a, b))
    omega, a, b = best[1]
    return {"omega": float(omega), "alpha": float(a), "beta": float(b), "loglik": float(best[0]),
            "persistence": float(a + b)}


# ---- covariance ----------------------------------------------------------------------------------
def ewma_covariance(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    r = np.asarray(returns, float)
    w = lam ** np.arange(len(r))[::-1]
    w /= w.sum()
    mu = w @ r
    x = r - mu
    return (x * w[:, None]).T @ x


def ledoit_wolf(returns: np.ndarray) -> Dict[str, object]:
    """Ledoit-Wolf shrinkage towards a scaled identity; returns shrunk covariance + intensity."""
    X = np.asarray(returns, float)
    n, p = X.shape
    if n < 3:
        raise StatisticsError("need at least 3 observations")
    X = X - X.mean(axis=0)
    S = X.T @ X / n
    mu = np.trace(S) / p
    d2 = np.sum((S - mu * np.eye(p)) ** 2) / p
    b2 = min(sum(np.sum((np.outer(x, x) - S) ** 2) for x in X) / (n ** 2 * p), d2)
    shrink = b2 / d2 if d2 > 0 else 0.0
    return {"cov": shrink * mu * np.eye(p) + (1 - shrink) * S, "shrinkage": float(shrink)}


def nearest_psd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Clip negative eigenvalues so the matrix is a valid covariance."""
    c = (cov + cov.T) / 2
    w, v = np.linalg.eigh(c)
    return (v * np.clip(w, eps, None)) @ v.T


def correlation_from_cov(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    return cov / np.outer(d, d)


# ---- Kalman filter ----------------------------------------------------------------------------------
@dataclass
class KalmanHedge:
    """Dynamic regression y_t = alpha_t + beta_t x_t + eps with random-walk coefficients.

    Used for adaptive pair-trading hedge ratios; ``update`` is O(1) so it works bar by bar."""
    delta: float = 1e-4          # state noise (larger = faster adaptation)
    obs_var: float = 1e-3

    def __post_init__(self) -> None:
        self.theta = np.zeros(2)                       # [alpha, beta]
        self.P = np.eye(2)
        self.Vw = self.delta / (1 - self.delta) * np.eye(2)
        self.n = 0

    def update(self, x: float, y: float) -> Dict[str, float]:
        H = np.array([1.0, x])
        self.P = self.P + self.Vw
        yhat = float(H @ self.theta)
        S = float(H @ self.P @ H) + self.obs_var
        K = self.P @ H / S
        err = y - yhat
        self.theta = self.theta + K * err
        self.P = self.P - np.outer(K, H) @ self.P
        self.n += 1
        return {"alpha": float(self.theta[0]), "beta": float(self.theta[1]), "error": float(err),
                "error_std": math.sqrt(S)}


def kalman_smooth_1d(x, q: float = 1e-5, r: float = 1e-2) -> np.ndarray:
    """Local-level Kalman filter (causal denoiser)."""
    x = np.asarray(x, float)
    est, p = np.empty(len(x)), 1.0
    est[0] = x[0]
    for t in range(1, len(x)):
        p += q
        k = p / (p + r)
        est[t] = est[t - 1] + k * (x[t] - est[t - 1])
        p *= (1 - k)
    return est
