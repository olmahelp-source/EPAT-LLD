"""Change-point detection: notice that the process generating the data has changed.

For a strategy this is the question "is the edge I measured still there". :func:`cusum` is the classic sequential test
for a shift in the mean; :func:`bocpd` (Bayesian online change-point detection, Adams & MacKay 2007) keeps a full
probability distribution over how long the current regime has lasted. Both are online: the answer at t uses data up to t.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from ..core.exceptions import ValidationError


def cusum(x, threshold: float = 5.0, drift: float = 0.5, warmup: int = 30) -> Dict[str, object]:
    """Two-sided CUSUM on a series standardised with the mean and deviation of the current regime's first ``warmup``
    points. An alarm fires when the cumulative deviation exceeds ``threshold`` sigmas; the statistic then restarts and
    the reference level is re-estimated. ``drift`` is the slack (in sigmas) ignored each step: higher = fewer false
    alarms, slower detection. Returns alarm indices, their direction and the two running statistics."""
    x = np.asarray(x, float)
    if threshold <= 0 or drift < 0 or warmup < 5:
        raise ValidationError("need threshold > 0, drift >= 0, warmup >= 5")
    up, down = np.zeros(len(x)), np.zeros(len(x))
    alarms: List[int] = []
    direction: List[int] = []
    start, gp, gn = 0, 0.0, 0.0
    mu = sd = None
    for t in range(len(x)):
        if not np.isfinite(x[t]):
            continue
        if t - start < warmup:
            continue
        if mu is None:
            ref = x[start:start + warmup]
            ref = ref[np.isfinite(ref)]
            mu, sd = float(ref.mean()), float(ref.std(ddof=1)) if len(ref) > 1 else 0.0
            sd = sd if sd > 0 else 1.0
        z = (x[t] - mu) / sd
        gp, gn = max(0.0, gp + z - drift), max(0.0, gn - z - drift)
        up[t], down[t] = gp, gn
        if gp > threshold or gn > threshold:
            alarms.append(t)
            direction.append(1 if gp > threshold else -1)
            start, gp, gn, mu, sd = t + 1, 0.0, 0.0, None, None
    return {"alarms": alarms, "direction": direction, "up": up, "down": down}


def bocpd(x, hazard: float = 1.0 / 250.0, mu0: float = 0.0, kappa0: float = 1.0, alpha0: float = 1.0, beta0: float = 1.0,
          max_run: int = 500) -> Dict[str, np.ndarray]:
    """Bayesian online change-point detection for data that is normal with unknown mean and variance within each regime.

    ``hazard`` is the prior chance per step that a new regime starts (1/250 = about once a year on daily data). Put the
    data on a sensible scale (e.g. standardise returns) or set ``beta0`` to its typical variance.

    Returns, per step: ``run_length`` (most probable number of steps since the last change), ``expected_run_length``
    and ``change_prob`` (probability that a change happened within the last 5 steps). A sudden collapse of
    ``run_length`` towards zero marks a change."""
    x = np.asarray(x, float)
    if not 0 < hazard < 1 or kappa0 <= 0 or alpha0 <= 0 or beta0 <= 0 or max_run < 10:
        raise ValidationError("need 0 < hazard < 1, positive priors and max_run >= 10")
    if not np.isfinite(x).all():
        raise ValidationError("x must be finite")
    T = len(x)
    lg = np.array([math.lgamma(alpha0 + k / 2.0) for k in range(max_run + 3)])       # lgamma(alpha) for every run length
    lg_half = np.array([math.lgamma(alpha0 + k / 2.0 + 0.5) for k in range(max_run + 3)])
    probs = np.array([1.0])
    mu, kappa, alpha, beta = np.array([mu0]), np.array([kappa0]), np.array([alpha0]), np.array([beta0])
    run, expected, change = np.zeros(T, dtype=int), np.zeros(T), np.zeros(T)
    for t in range(T):
        k = np.arange(len(probs))
        scale2 = beta * (kappa + 1.0) / (alpha * kappa)                           # Student-t predictive, 2 * alpha degrees of freedom
        nu = 2.0 * alpha
        logpred = lg_half[k] - lg[k] - 0.5 * np.log(nu * math.pi * scale2) - (nu + 1.0) / 2.0 * np.log1p((x[t] - mu) ** 2 / (nu * scale2))
        pred = np.exp(logpred)
        growth = probs * pred * (1.0 - hazard)
        new = np.concatenate([[(probs * pred * hazard).sum()], growth])
        total = new.sum()
        new = new / total if total > 0 else np.concatenate([[1.0], np.zeros(len(growth))])
        mu_n = (kappa * mu + x[t]) / (kappa + 1.0)                                # conjugate update of every run length's posterior
        beta_n = beta + kappa * (x[t] - mu) ** 2 / (2.0 * (kappa + 1.0))
        mu, kappa = np.concatenate([[mu0], mu_n]), np.concatenate([[kappa0], kappa + 1.0])
        alpha, beta = np.concatenate([[alpha0], alpha + 0.5]), np.concatenate([[beta0], beta_n])
        if len(new) > max_run + 1:                                                # fold the tail into the longest run kept
            new[max_run] += new[max_run + 1:].sum()
            new, mu, kappa, alpha, beta = new[:max_run + 1], mu[:max_run + 1], kappa[:max_run + 1], alpha[:max_run + 1], beta[:max_run + 1]
        probs = new
        run[t] = int(np.argmax(probs))
        expected[t] = float((np.arange(len(probs)) * probs).sum())
        change[t] = float(probs[:6].sum()) if t >= 6 else 0.0
    return {"run_length": run, "expected_run_length": expected, "change_prob": change}


def changepoints_from_run_length(run_length, min_drop: int = 20) -> List[int]:
    """Indices where the most probable run length falls by at least ``min_drop`` in one step, reported as the estimated
    *start* of the new regime."""
    r = np.asarray(run_length, int)
    return [int(t - r[t]) for t in range(1, len(r)) if r[t - 1] - r[t] >= min_drop]
