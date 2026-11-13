"""Numerical building blocks shared by the model packages (no scipy)."""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def logsumexp(a: np.ndarray, axis: Optional[int] = None):
    """log(sum(exp(a))) without overflow; a float when ``axis`` is None, else an array."""
    a = np.asarray(a, float)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return float(out.reshape(-1)[0]) if axis is None else np.squeeze(out, axis=axis)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(z, float)
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def nelder_mead(f: Callable[[np.ndarray], float], x0, step=None, max_iter: int = 2000, tol: float = 1e-10) -> Tuple[np.ndarray, float]:
    """Derivative-free minimiser (downhill simplex). Returns (x, f(x)). ``f`` may return inf for infeasible points,
    which is how callers impose constraints."""
    x0 = np.asarray(x0, float)
    n = len(x0)
    step = np.full(n, 0.1) if step is None else np.broadcast_to(np.asarray(step, float), (n,))
    simplex = np.vstack([x0] + [x0 + np.eye(n)[i] * step[i] for i in range(n)])
    values = np.array([f(p) for p in simplex])
    for _ in range(max_iter):
        order = np.argsort(values)
        simplex, values = simplex[order], values[order]
        if np.isfinite(values[-1]) and abs(values[-1] - values[0]) <= tol * (abs(values[0]) + tol):
            break
        centroid = simplex[:-1].mean(axis=0)
        reflected = centroid + (centroid - simplex[-1])
        fr = f(reflected)
        if fr < values[0]:
            expanded = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(expanded)
            simplex[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = reflected, fr
        else:
            contracted = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(contracted)
            if fc < values[-1]:
                simplex[-1], values[-1] = contracted, fc
            else:                                                       # shrink towards the best vertex
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                values[1:] = [f(p) for p in simplex[1:]]
    best = int(np.argmin(values))
    return simplex[best], float(values[best])


def spearman(a, b) -> float:
    """Rank correlation of two equal-length vectors (NaN pairs dropped; average ranks for ties)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = rank(a[ok]), rank(b[ok])
    sa, sb = ra.std(), rb.std()
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb)) if sa > 0 and sb > 0 else float("nan")


def rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), like ``scipy.stats.rankdata``."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    sx = x[order]
    start = 0
    for i in range(1, len(x) + 1):
        if i == len(x) or sx[i] != sx[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + 1 + i) / 2.0
            start = i
    return ranks
