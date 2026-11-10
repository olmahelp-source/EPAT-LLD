"""Limit-price logic and pre-trade cost estimates (bar data has no quotes, so the touch is modelled)."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from ..brokers.costs import CostModel
from ..core.enums import Side
from ..core.exceptions import ValidationError
from ..quant.impact import sqrt_impact


def modelled_quote(price: float, spread_bps: float) -> Tuple[float, float]:
    half = price * spread_bps / 2e4
    return price - half, price + half


def _round(side: Side, px: float, tick: float) -> float:
    """Round passively: buys down, sells up (never pay more than intended)."""
    steps = px / tick
    return round((math.floor(steps + 1e-9) if side is Side.BUY else math.ceil(steps - 1e-9)) * tick, 10)


def limit_price(side: Side, ref: float, tick: float = 0.01, mode: str = "touch", spread_bps: float = 2.0,
                improve_ticks: int = 1, bars_waited: int = 0, chase_bps_per_bar: float = 0.0,
                chase_cap_bps: float = 10.0) -> float:
    """Pick a limit price. Modes: ``touch`` (join own side), ``improve`` (step inside the spread, never past
    mid), ``mid``, ``aggressive`` (cross the spread). Waiting orders chase toward the market by
    ``chase_bps_per_bar`` each bar, capped at ``chase_cap_bps`` so a stale order cannot run away."""
    if ref <= 0 or tick <= 0:
        raise ValidationError("ref and tick must be positive")
    bid, ask = modelled_quote(ref, spread_bps)
    own, far = (bid, ask) if side is Side.BUY else (ask, bid)
    mid = (bid + ask) / 2
    if mode == "touch":
        px = own
    elif mode == "mid":
        px = mid
    elif mode == "improve":
        px = own + side.sign * improve_ticks * tick
        px = min(px, mid) if side is Side.BUY else max(px, mid)
    elif mode == "aggressive":
        px = far
    else:
        raise ValidationError(f"unknown pricing mode '{mode}'")
    chase = min(chase_bps_per_bar * bars_waited, chase_cap_bps) / 1e4
    px = px * (1 + side.sign * chase)
    return _round(side, px, tick)


def estimate_cost(side: Side, quantity: float, price: float, adv: Optional[float], sigma: Optional[float],
                  costs: CostModel, spread_bps: Optional[float] = None) -> Dict[str, float]:
    """Pre-trade cost in currency and bps: commission + spread/slippage + impact + tax."""
    notional = abs(quantity) * price
    if notional <= 0:
        raise ValidationError("quantity and price must be positive")
    c = costs.cfg
    spread = (spread_bps if spread_bps is not None else c.spread_bps) / 2e4 * notional
    slippage = c.slippage_bps / 1e4 * notional
    impact = sqrt_impact(sigma, quantity, adv, c.impact_coef) * notional if (sigma and adv) else 0.0
    commission = costs.commission(quantity, price)
    tax = costs.tax(side, notional)
    total = spread + slippage + impact + commission + tax
    return {"commission": commission, "spread": spread, "slippage": slippage, "impact": impact, "tax": tax,
            "total": total, "total_bps": total / notional * 1e4, "notional": notional}


def go_no_go(expected_alpha_bps: float, cost: Dict[str, float], margin: float = 1.5) -> Dict[str, Any]:
    """Trade only if expected alpha beats cost by ``margin`` x (default 1.5x)."""
    need = cost["total_bps"] * margin
    return {"go": expected_alpha_bps >= need, "alpha_bps": expected_alpha_bps, "required_bps": need,
            "cost_bps": cost["total_bps"]}
