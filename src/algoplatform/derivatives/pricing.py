"""Lattice and Monte Carlo option pricing, and implied-volatility surface tools."""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..core.exceptions import StatisticsError, ValidationError
from ..quant.models import implied_vol


def binomial_price(S: float, K: float, T: float, r: float, sigma: float, kind: str = "call", american: bool = False,
                   steps: int = 300, q: float = 0.0) -> float:
    """Cox-Ross-Rubinstein tree. European values converge to Black-Scholes; American adds the early-exercise right
    (valuable for puts, and for calls only when dividends are paid)."""
    if kind not in ("call", "put") or min(S, K, T, sigma) <= 0 or steps < 1:
        raise ValidationError("bad option parameters")
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d, disc = 1 / u, math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    if not 0 < p < 1:
        raise ValidationError("tree is not arbitrage-free: increase steps or check r/q/sigma")
    j = np.arange(steps + 1)
    ST = S * u ** (steps - j) * d ** j
    v = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
    for i in range(steps - 1, -1, -1):
        v = disc * (p * v[:-1] + (1 - p) * v[1:])
        if american:
            Si = S * u ** (i - np.arange(i + 1)) * d ** np.arange(i + 1)
            v = np.maximum(v, (Si - K) if kind == "call" else (K - Si))
    return float(v[0])


def _paths(S: float, T: float, r: float, sigma: float, n: int, steps: int, seed: Optional[int], q: float = 0.0,
           antithetic: bool = True) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = n // 2 if antithetic else n
    z = rng.standard_normal((half, steps))
    if antithetic:
        z = np.concatenate([z, -z])
    dt = T / steps
    inc = (r - q - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    return S * np.exp(np.cumsum(inc, axis=1))


def mc_price(S: float, K: float, T: float, r: float, sigma: float, kind: str = "call", payoff: str = "european",
             n: int = 40_000, steps: int = 64, seed: Optional[int] = 0, barrier: Optional[float] = None,
             q: float = 0.0) -> Dict[str, float]:
    """Monte Carlo with antithetic variates. ``payoff``: european | asian (arithmetic average) | barrier_out (down/up
    knock-out, ``barrier`` below S for down-and-out or above S for up-and-out). Returns price and standard error."""
    if min(S, K, T, sigma) <= 0 or n < 100:
        raise ValidationError("bad option parameters")
    paths = _paths(S, T, r, sigma, n, steps, seed, q)
    ST = paths[:, -1]
    ref = {"european": ST, "asian": paths.mean(axis=1), "barrier_out": ST}.get(payoff)
    if ref is None:
        raise ValidationError("payoff must be european, asian or barrier_out")
    pay = np.maximum(ref - K, 0.0) if kind == "call" else np.maximum(K - ref, 0.0)
    if payoff == "barrier_out":
        if barrier is None:
            raise ValidationError("barrier_out needs a barrier level")
        knocked = (paths.min(axis=1) <= barrier) if barrier < S else (paths.max(axis=1) >= barrier)
        pay = np.where(knocked, 0.0, pay)
    disc = math.exp(-r * T)
    return {"price": float(disc * pay.mean()), "stderr": float(disc * pay.std(ddof=1) / math.sqrt(len(pay)))}


# ---- implied volatility surface ---------------------------------------------------------------------------------
def svi_total_variance(k: np.ndarray, a: float, b: float, rho: float, m: float, sigma: float) -> np.ndarray:
    """Gatheral's SVI: total implied variance w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2)), k = log-moneyness."""
    k = np.asarray(k, float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def _nelder_mead(f: Callable[[np.ndarray], float], x0: np.ndarray, step: np.ndarray, iters: int = 2500, tol: float = 1e-14) -> np.ndarray:
    n = len(x0)
    pts = [x0] + [x0 + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c) if fr >= vals[-1] else c + 0.5 * (xr - c)
            fc = f(xc)
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    return pts[int(np.argmin(vals))]


def fit_svi(k: Sequence[float], total_var: Sequence[float], restarts: int = 6, seed: int = 0) -> Dict[str, object]:
    """Least-squares SVI fit of total variance with the no-arbitrage-friendly constraints b > 0, |rho| < 1, sigma > 0 and
    a + b sigma sqrt(1 - rho^2) >= 0 (enforced by penalty). Returns parameters, RMSE and a callable."""
    k, w = np.asarray(k, float), np.asarray(total_var, float)
    if len(k) < 5 or len(k) != len(w):
        raise StatisticsError("need at least 5 matching (k, w) points to fit SVI")

    def loss(x: np.ndarray) -> float:
        a, b, rho, m, s = x
        pen = 0.0
        if b <= 0 or s <= 1e-4 or abs(rho) >= 0.999:
            return 1e6 + 1e3 * (max(0, -b) + max(0, 1e-4 - s) + max(0, abs(rho) - 0.999))
        floor = a + b * s * math.sqrt(1 - rho ** 2)
        if floor < 0:
            pen = 1e3 * (-floor)
        return float(np.mean((svi_total_variance(k, *x) - w) ** 2) + pen)

    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        x0 = np.array([w.min() * 0.5, 0.1 + 0.2 * rng.random(), rng.uniform(-0.6, 0.2), float(np.mean(k)), 0.1 + 0.2 * rng.random()])
        x = _nelder_mead(loss, x0, np.array([0.01, 0.05, 0.1, 0.05, 0.05]))
        if best is None or loss(x) < loss(best):
            best = x
    a, b, rho, m, s = best
    rmse = math.sqrt(float(np.mean((svi_total_variance(k, *best) - w) ** 2)))
    return {"a": float(a), "b": float(b), "rho": float(rho), "m": float(m), "sigma": float(s), "rmse": rmse,
            "w": lambda kk: svi_total_variance(kk, *best)}


def check_no_arbitrage(strikes: Sequence[float], calls: Sequence[float], S: float, r: float, T: float, tol: float = 1e-8) -> List[str]:
    """Static no-arbitrage conditions on a call price slice: price bounds, monotonic decrease in strike (<=1 slope),
    and convexity in strike (non-negative butterflies). Returns human-readable violations (empty = clean)."""
    K, C = np.asarray(strikes, float), np.asarray(calls, float)
    if len(K) != len(C) or len(K) < 3 or not (np.diff(K) > 0).all():
        raise ValidationError("need >= 3 matching, strictly increasing strikes")
    bad: List[str] = []
    disc = math.exp(-r * T)
    for k, c in zip(K, C):
        if c < max(S - k * disc, 0.0) - tol:
            bad.append(f"K={k:g}: call below intrinsic bound")
        if c > S + tol:
            bad.append(f"K={k:g}: call above spot")
    slope = np.diff(C) / np.diff(K)
    for i, s in enumerate(slope):
        if s > tol:
            bad.append(f"K={K[i]:g}-{K[i + 1]:g}: call price rises with strike (negative vertical spread)")
        if s < -disc - tol:
            bad.append(f"K={K[i]:g}-{K[i + 1]:g}: vertical spread worth more than its max payoff")
    for i in range(len(slope) - 1):
        if slope[i + 1] < slope[i] - tol:
            bad.append(f"K={K[i + 1]:g}: butterfly arbitrage (price not convex in strike)")
    return bad


def implied_vol_slice(strikes: Sequence[float], prices: Sequence[float], S: float, T: float, r: float, kind: str = "call") -> np.ndarray:
    """Implied vols for a slice; NaN where a quote violates no-arbitrage bounds."""
    out = []
    for k, p in zip(strikes, prices):
        try:
            out.append(implied_vol(p, S, k, T, r, kind))
        except (StatisticsError, ValidationError):
            out.append(float("nan"))
    return np.asarray(out)
