"""Delta hedging: bandwidth rules and a path simulator that shows the cost/risk trade-off."""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np

from ..core.exceptions import ValidationError
from ..quant.models import bs_greeks, bs_price


def whalley_wilmott_band(S: float, gamma: float, cost_rate: float, risk_aversion: float = 1.0) -> float:
    """Half-width (in shares of delta) inside which re-hedging is not worth its cost:
    H = (3 * cost_rate * S * gamma^2 / (2 * risk_aversion)) ** (1/3)   (Whalley-Wilmott asymptotic approximation).
    Higher costs or lower gamma widen the no-trade band; higher risk aversion narrows it."""
    if cost_rate < 0 or risk_aversion <= 0:
        raise ValidationError("cost_rate >= 0 and risk_aversion > 0 required")
    return (1.5 * cost_rate * S * gamma ** 2 / risk_aversion) ** (1.0 / 3.0)


def hedge_trade(position_delta: float, current_hedge: float, band: float) -> float:
    """Shares to trade so net delta returns to zero, or 0 while inside the band."""
    net = position_delta + current_hedge
    return -net if abs(net) > band else 0.0


def simulate_delta_hedge(paths: np.ndarray, K: float, T: float, r: float, sigma: float, kind: str = "call",
                         rebalance_every: int = 1, band: Optional[float] = None, cost_bps: float = 0.0, qty: float = 1.0,
                         mult: float = 100.0) -> Dict[str, np.ndarray]:
    """Sell ``qty`` options at Black-Scholes value and delta-hedge along each simulated path (rows of ``paths`` with
    columns S_0..S_n, equally spaced to maturity T). Returns terminal P&L and trade counts per path.
    ``rebalance_every`` = time-based schedule; ``band`` (if given) = hedge only when |net delta| exceeds it."""
    paths = np.atleast_2d(np.asarray(paths, float))
    n = paths.shape[1] - 1
    dt = T / n
    pnl, trades, cost = np.zeros(len(paths)), np.zeros(len(paths)), np.zeros(len(paths))
    for p, S in enumerate(paths):
        premium = bs_price(S[0], K, T, r, sigma, kind) * qty * mult
        cash, hedge = premium, 0.0
        for i in range(n):
            tau = T - i * dt
            delta = -bs_greeks(S[i], K, tau, r, sigma, kind)["delta"] * qty * mult           # short option delta
            go = (i % rebalance_every == 0) if band is None else abs(delta + hedge) > band
            if go:
                trade = -(delta + hedge)
                if abs(trade) > 1e-12:
                    c = abs(trade) * S[i] * cost_bps / 1e4
                    cash -= trade * S[i] + c
                    cost[p] += c
                    hedge += trade
                    trades[p] += 1
            cash *= math.exp(r * dt)
        payoff = max(S[-1] - K, 0.0) if kind == "call" else max(K - S[-1], 0.0)
        pnl[p] = cash + hedge * S[-1] - payoff * qty * mult
    return {"pnl": pnl, "trades": trades, "cost": cost}
