"""Risk measures: VaR, Expected Shortfall, VaR backtests, stress tests."""
from __future__ import annotations

import math
from typing import Dict, Mapping, Optional

import numpy as np

from ..stats.special import chi2_sf, norm_pdf, norm_ppf
from ..stats.tests import excess_kurtosis, skewness
from ..utils.helpers import drop_nan, require, to_array


def _returns(x) -> np.ndarray:
    return drop_nan(to_array(x, "returns", min_len=10))


def historical_var(returns, confidence: float = 0.99) -> float:
    """Loss (positive number, as a return fraction) not exceeded with ``confidence``."""
    r = _returns(returns)
    return float(max(-np.percentile(r, 100 * (1 - confidence)), 0.0))


def parametric_var(returns, confidence: float = 0.99) -> float:
    r = _returns(returns)
    return float(max(-(r.mean() + norm_ppf(1 - confidence) * r.std(ddof=1)), 0.0))


def cornish_fisher_var(returns, confidence: float = 0.99) -> float:
    """Gaussian VaR adjusted for skew and kurtosis."""
    r = _returns(returns)
    z, s, k = norm_ppf(1 - confidence), skewness(r), excess_kurtosis(r)
    zcf = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
    return float(max(-(r.mean() + zcf * r.std(ddof=1)), 0.0))


def monte_carlo_var(mu: float, sigma: float, confidence: float = 0.99, n: int = 100_000,
                    seed: Optional[int] = 0, df: Optional[float] = None) -> float:
    """Simulated VaR for a single return stream; ``df`` switches to fat-tailed Student-t shocks."""
    rng = np.random.default_rng(seed)
    if df:
        shocks = rng.standard_t(df, n) * math.sqrt((df - 2) / df) if df > 2 else rng.standard_t(df, n)
    else:
        shocks = rng.standard_normal(n)
    sims = mu + sigma * shocks
    return float(max(-np.percentile(sims, 100 * (1 - confidence)), 0.0))


def expected_shortfall(returns, confidence: float = 0.975) -> float:
    """Average loss in the worst (1 - confidence) tail."""
    r = _returns(returns)
    cutoff = np.percentile(r, 100 * (1 - confidence))
    tail = r[r <= cutoff]
    return float(max(-tail.mean(), 0.0)) if len(tail) else 0.0


def parametric_es(returns, confidence: float = 0.975) -> float:
    r = _returns(returns)
    a = 1 - confidence
    return float(max(-(r.mean() - r.std(ddof=1) * norm_pdf(norm_ppf(a)) / a), 0.0))


def portfolio_var(weights, cov, confidence: float = 0.99, horizon: int = 1, mu=None) -> float:
    w, cov = np.asarray(weights, float), np.asarray(cov, float)
    sigma = math.sqrt(float(w @ cov @ w) * horizon)
    m = float(np.asarray(mu) @ w) * horizon if mu is not None else 0.0
    return float(max(-(m + norm_ppf(1 - confidence) * sigma), 0.0))


def kupiec_pof(violations: int, n: int, confidence: float = 0.99) -> Dict[str, float]:
    """Kupiec proportion-of-failures test. Small p => VaR model mis-calibrated."""
    require(n > 0 and 0 <= violations <= n, "bad violation count")
    p = 1 - confidence
    x = violations
    if x == 0:
        lr = -2 * n * math.log(1 - p)
    elif x == n:
        lr = -2 * n * math.log(p)
    else:
        ph = x / n
        lr = -2 * ((n - x) * math.log(1 - p) + x * math.log(p)
                   - (n - x) * math.log(1 - ph) - x * math.log(ph))
    return {"lr": lr, "p_value": chi2_sf(lr, 1), "violations": x, "expected": n * p}


def var_backtest(returns, var_series, confidence: float = 0.99) -> Dict[str, float]:
    """Count days the realised loss exceeded the forecast VaR (given as positive numbers)."""
    r, v = np.asarray(returns, float), np.asarray(var_series, float)
    require(len(r) == len(v), "returns and var_series length mismatch")
    viol = int(np.sum(-r > v))
    return kupiec_pof(viol, len(r), confidence)


def stress_test(positions_value: Mapping[str, float], shocks: Mapping[str, float]) -> Dict[str, float]:
    """Apply percentage shocks to position market values; returns P&L per symbol and 'total'.

    Symbols without an explicit shock use the ``"*"`` shock (default 0)."""
    default = shocks.get("*", 0.0)
    out = {s: v * shocks.get(s, default) for s, v in positions_value.items()}
    out["total"] = sum(out.values())
    return out


HISTORICAL_SCENARIOS: Dict[str, Dict[str, float]] = {
    "equity_crash_-20%": {"*": -0.20},
    "black_monday_-22%": {"*": -0.22},
    "flash_crash_-9%": {"*": -0.09},
    "rally_+10%": {"*": 0.10},
}


def reverse_stress_loss(equity: float, gross_exposure: float, loss_limit_pct: float) -> float:
    """Uniform adverse market move that would hit ``loss_limit_pct`` of equity at current gross exposure."""
    require(gross_exposure >= 0 and equity > 0, "bad inputs")
    return float("inf") if gross_exposure == 0 else equity * loss_limit_pct / gross_exposure
