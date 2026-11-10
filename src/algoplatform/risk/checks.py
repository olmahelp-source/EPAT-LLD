"""Pre-trade risk checks. Each check is small, independent and returns a reason or None."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

from ..core.config import RiskConfig
from ..core.enums import Side
from ..core.models import Order
from ..utils.timeutils import TradingCalendar


@dataclass
class RiskContext:
    """Everything a check may need, passed in so checks stay pure and testable."""
    equity: float
    cash: float
    positions: Dict[str, float]                 # signed quantity per symbol
    prices: Dict[str, float]                    # last known price
    now: datetime
    pending: Dict[str, float] = field(default_factory=dict)       # signed qty working at broker
    multipliers: Dict[str, float] = field(default_factory=dict)
    price_times: Dict[str, datetime] = field(default_factory=dict)

    def mult(self, symbol: str) -> float:
        return self.multipliers.get(symbol, 1.0)

    def exposure(self, include_pending: bool = True) -> Tuple[float, float]:
        """(gross, net) notional including working orders."""
        gross = net = 0.0
        for s in sorted(set(self.positions) | set(self.pending)):      # fixed order: float sums must be reproducible
            q = self.positions.get(s, 0.0) + (self.pending.get(s, 0.0) if include_pending else 0.0)
            v = q * self.prices.get(s, 0.0) * self.mult(s)
            gross += abs(v)
            net += v
        return gross, net


class Check:
    name = "check"

    def check(self, order: Order, ctx: RiskContext) -> Optional[str]:
        raise NotImplementedError


def _ref_price(order: Order, ctx: RiskContext) -> Optional[float]:
    return order.limit_price or ctx.prices.get(order.symbol)


class QuantityCheck(Check):
    name = "max_order_quantity"

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def check(self, order, ctx):
        if order.quantity > self.cfg.max_order_quantity:
            return f"quantity {order.quantity} exceeds max {self.cfg.max_order_quantity}"


class NotionalCheck(Check):
    name = "max_order_notional"

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def check(self, order, ctx):
        px = _ref_price(order, ctx)
        if px is None:
            return "no reference price for notional check"
        notional = order.quantity * px * ctx.mult(order.symbol)
        if notional > self.cfg.max_order_notional:
            return f"notional {notional:,.0f} exceeds max {self.cfg.max_order_notional:,.0f}"


class PriceCollarCheck(Check):
    """Fat-finger guard: a limit/stop far from the market is almost always a typo."""
    name = "price_collar"

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def check(self, order, ctx):
        last = ctx.prices.get(order.symbol)
        for label, px in (("limit", order.limit_price), ("stop", order.stop_price)):
            if px is not None and last:
                dev = abs(px - last) / last
                if dev > self.cfg.price_collar_pct:
                    return f"{label} price {px} is {dev:.1%} from last {last} (collar {self.cfg.price_collar_pct:.0%})"


class RestrictedListCheck(Check):
    name = "restricted_list"

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def check(self, order, ctx):
        if order.symbol in self.cfg.restricted_symbols:
            return f"{order.symbol} is on the restricted list"


class ShortSellCheck(Check):
    name = "short_sell"

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def check(self, order, ctx):
        pos = ctx.positions.get(order.symbol, 0.0)
        after = pos + order.side.sign * order.quantity
        if order.reduce_only:
            if abs(after) >= abs(pos) - 1e-12:
                return "reduce-only order would not reduce the position"
            if after * pos < 0:
                return "reduce-only order would flip the position"
        if not self.cfg.allow_short and after < -1e-12:
            return "short selling not allowed"


class PositionLimitCheck(Check):
    name = "max_position_notional"

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def check(self, order, ctx):
        px = _ref_price(order, ctx)
        if px is None:
            return None
        q = ctx.positions.get(order.symbol, 0.0) + ctx.pending.get(order.symbol, 0.0) + order.side.sign * order.quantity
        notional = abs(q) * px * ctx.mult(order.symbol)
        if notional > self.cfg.max_position_notional:
            return f"projected position {notional:,.0f} exceeds max {self.cfg.max_position_notional:,.0f}"


class ExposureCheck(Check):
    """Gross and net exposure caps as a multiple of equity; risk-reducing orders always pass."""
    name = "exposure"

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def check(self, order, ctx):
        px = ctx.prices.get(order.symbol) or order.limit_price
        if px is None or ctx.equity <= 0:
            return None if ctx.equity > 0 else "non-positive equity"
        gross, net = ctx.exposure()
        pos = ctx.positions.get(order.symbol, 0.0) + ctx.pending.get(order.symbol, 0.0)
        delta = order.side.sign * order.quantity * px * ctx.mult(order.symbol)
        cur = pos * px * ctx.mult(order.symbol)
        new_gross = gross - abs(cur) + abs(cur + delta)
        new_net = net + delta
        if new_gross > gross and new_gross > self.cfg.max_gross_exposure_pct * ctx.equity:
            return f"gross exposure {new_gross / ctx.equity:.2f}x exceeds {self.cfg.max_gross_exposure_pct:.2f}x"
        if abs(new_net) > abs(net) and abs(new_net) > self.cfg.max_net_exposure_pct * ctx.equity:
            return f"net exposure {abs(new_net) / ctx.equity:.2f}x exceeds {self.cfg.max_net_exposure_pct:.2f}x"


class BuyingPowerCheck(Check):
    """For cash (unleveraged) accounts: opening buys must be covered by cash on hand."""
    name = "buying_power"

    def __init__(self, cash_account: bool = False) -> None:
        self.cash_account = cash_account

    def check(self, order, ctx):
        if not self.cash_account or order.side is not Side.BUY:
            return None
        px = _ref_price(order, ctx)
        if px is None:
            return "no reference price for buying-power check"
        covering = max(-ctx.positions.get(order.symbol, 0.0), 0.0)   # buys that close a short need no cash
        need = max(order.quantity - covering, 0.0) * px * ctx.mult(order.symbol)
        if need > ctx.cash:
            return f"insufficient cash: need {need:,.0f}, have {ctx.cash:,.0f}"


class OrderRateCheck(Check):
    name = "order_rate"

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self._times: Deque[datetime] = deque()

    def check(self, order, ctx):
        while self._times and (ctx.now - self._times[0]).total_seconds() >= 60:
            self._times.popleft()
        if len(self._times) >= self.cfg.max_orders_per_minute:
            return f"order rate limit {self.cfg.max_orders_per_minute}/min reached"
        self._times.append(ctx.now)


class DuplicateOrderCheck(Check):
    """Runaway-loop guard: same strategy/symbol/side/qty repeated within a short window."""
    name = "duplicate_order"

    def __init__(self, window_sec: float = 1.0, max_repeats: int = 3) -> None:
        self.window, self.max_repeats = window_sec, max_repeats
        self._recent: Deque[Tuple[datetime, tuple]] = deque(maxlen=200)

    def check(self, order, ctx):
        key = (order.strategy_id, order.symbol, order.side, round(order.quantity, 8), order.order_type)
        while self._recent and (ctx.now - self._recent[0][0]).total_seconds() > self.window:
            self._recent.popleft()
        if sum(1 for _, k in self._recent if k == key) >= self.max_repeats:
            return f"{self.max_repeats}+ identical orders within {self.window}s (runaway loop?)"
        self._recent.append((ctx.now, key))


class StaleDataCheck(Check):
    name = "stale_data"

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def check(self, order, ctx):
        t = ctx.price_times.get(order.symbol)
        if t is not None and (ctx.now - t).total_seconds() > self.cfg.max_data_staleness_sec:
            return f"price for {order.symbol} is {(ctx.now - t).total_seconds():.0f}s old"


class MarketHoursCheck(Check):
    name = "market_hours"

    def __init__(self, calendar: TradingCalendar) -> None:
        self.calendar = calendar

    def check(self, order, ctx):
        if not self.calendar.is_open(ctx.now):
            return "market is closed"


def default_checks(cfg: RiskConfig) -> List[Check]:
    # cheap, stateless checks first; stateful rate/duplicate counters last so rejected orders aren't counted
    return [RestrictedListCheck(cfg), QuantityCheck(cfg), NotionalCheck(cfg), PriceCollarCheck(cfg),
            StaleDataCheck(cfg), ShortSellCheck(cfg), PositionLimitCheck(cfg), ExposureCheck(cfg),
            BuyingPowerCheck(), DuplicateOrderCheck(), OrderRateCheck(cfg)]
