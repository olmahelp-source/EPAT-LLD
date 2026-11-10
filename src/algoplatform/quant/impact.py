"""Market-impact models and the Almgren-Chriss trading trajectory."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..core.exceptions import StatisticsError, ValidationError


def sqrt_impact(sigma: float, quantity: float, adv: float, coef: float = 0.1) -> float:
    """Expected impact as a return fraction: coef * sigma * sqrt(Q / ADV)  (square-root law)."""
    if adv <= 0:
        raise ValidationError("adv must be positive")
    return coef * sigma * math.sqrt(abs(quantity) / adv)


def calibrate_impact(participations: Sequence[float], slippage: Sequence[float], sigma: Sequence[float]) -> float:
    """Fit ``slippage = coef * sigma * sqrt(participation)`` through the origin from your own fills.

    ``slippage`` is the adverse price move as a return fraction. Returns ``coef``."""
    p, s, v = (np.asarray(x, float) for x in (participations, slippage, sigma))
    if not (len(p) == len(s) == len(v)) or len(p) < 5:
        raise StatisticsError("need at least 5 matching observations to calibrate impact")
    x = v * np.sqrt(np.clip(p, 0, None))
    denom = float(x @ x)
    if denom <= 0:
        raise StatisticsError("degenerate regressors for impact calibration")
    return float(x @ s / denom)


def ac_trajectory(n: int, kappa: float) -> np.ndarray:
    """Remaining fraction x_0..x_n of the order under Almgren-Chriss: sinh(k(n-t))/sinh(k n).

    kappa -> 0 is a straight line (TWAP); larger kappa front-loads trading to cut risk."""
    if n < 1:
        raise ValidationError("n must be >= 1")
    t = np.arange(n + 1)
    if abs(kappa) < 1e-9:
        return 1.0 - t / n
    return np.sinh(kappa * (n - t)) / math.sinh(kappa * n)


def kappa_from_risk(risk_aversion: float, sigma: float, temp_impact: float, bar_fraction: float = 1.0) -> float:
    """kappa = sqrt(lambda * sigma^2 / eta): urgency from risk aversion, volatility and temporary impact."""
    if temp_impact <= 0:
        raise ValidationError("temp_impact must be positive")
    return math.sqrt(max(risk_aversion, 0.0) * sigma ** 2 / temp_impact) * bar_fraction
