"""Transaction-cost model shared by the paper broker and the backtester."""
from __future__ import annotations

import math
from typing import Optional

from ..core.config import CostConfig
from ..core.enums import Side


class CostModel:
    """Commission + spread/slippage + square-root market impact + transaction tax.

    All adverse price effects are applied against the trader: buys fill higher,
    sells fill lower.
    """

    def __init__(self, cfg: Optional[CostConfig] = None) -> None:
        self.cfg = cfg or CostConfig()

    def commission(self, quantity: float, price: float) -> float:
        c = self.cfg
        fee = abs(quantity) * c.commission_per_share + abs(quantity) * price * c.commission_pct
        return max(fee, c.min_commission) if quantity else 0.0

    def fill_price(self, side: Side, ref_price: float, quantity: float = 0.0,
                   adv: Optional[float] = None, volatility: Optional[float] = None) -> float:
        """Reference price moved by half-spread, slippage and (optionally) impact."""
        c = self.cfg
        bps = c.slippage_bps + c.spread_bps / 2.0
        impact = 0.0
        if adv and adv > 0 and volatility:
            # Almgren-style square-root law: impact ~ coef * sigma * sqrt(Q / ADV)
            impact = c.impact_coef * volatility * math.sqrt(abs(quantity) / adv)
        move = bps / 1e4 + impact
        return ref_price * (1.0 + side.sign * move)

    def tax(self, side: Side, notional: float) -> float:
        """Transaction taxes (e.g. STT) are typically charged on sells only."""
        return notional * self.cfg.transaction_tax_bps / 1e4 if side is Side.SELL else 0.0

    def total_cost(self, side: Side, quantity: float, ref_price: float, **kw) -> float:
        px = self.fill_price(side, ref_price, quantity, **kw)
        slip = abs(px - ref_price) * abs(quantity)
        return slip + self.commission(quantity, px) + self.tax(side, abs(quantity) * px)
