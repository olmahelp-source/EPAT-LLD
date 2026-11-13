"""Core domain models shared by every module."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .enums import OrderStatus, OrderType, Side, TimeInForce
from .exceptions import ValidationError


@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick_size: float = 0.01
    lot_size: float = 1.0
    multiplier: float = 1.0
    currency: str = "USD"
    asset_class: str = "equity"
    shortable: bool = True

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValidationError("symbol required")
        if self.tick_size <= 0 or self.lot_size <= 0 or self.multiplier <= 0:
            raise ValidationError("tick_size, lot_size, multiplier must be positive",
                                  details={"symbol": self.symbol})

    def round_price(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 10)

    def round_qty(self, qty: float) -> float:
        """Round toward zero to a whole number of lots (never round up risk)."""
        lots = math.floor(abs(qty) / self.lot_size + 1e-9)
        return math.copysign(lots * self.lot_size, qty) if lots else 0.0


@dataclass(frozen=True)
class Bar:
    ts: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def __post_init__(self) -> None:
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValidationError("bar prices must be positive", details={"symbol": self.symbol})
        if self.high < max(self.open, self.close, self.low) - 1e-12 or \
                self.low > min(self.open, self.close, self.high) + 1e-12:
            raise ValidationError("inconsistent OHLC", details={"symbol": self.symbol, "ts": str(self.ts)})


@dataclass
class Signal:
    """A strategy's desired exposure for one symbol (target-position model).

    ``target_weight`` is a fraction of strategy capital in [-1, 1]; the position
    sizer decides the actual quantity, so strategies never touch order logic.
    """
    symbol: str
    target_weight: float
    reason: str = ""
    stop_price: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not -1.0 - 1e-9 <= self.target_weight <= 1.0 + 1e-9:
            raise ValidationError("target_weight must be in [-1, 1]",
                                  details={"symbol": self.symbol, "w": self.target_weight})


@dataclass
class Fill:
    fill_id: str
    client_order_id: str
    symbol: str
    side: Side
    quantity: float
    price: float
    ts: datetime
    commission: float = 0.0
    strategy_id: str = ""
    account_id: str = ""
    broker_order_id: str = ""

    @property
    def signed_qty(self) -> float:
        return self.side.sign * self.quantity

    @property
    def notional(self) -> float:
        return self.quantity * self.price


@dataclass
class Order:
    client_order_id: str
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tif: TimeInForce = TimeInForce.DAY
    strategy_id: str = ""
    account_id: str = ""
    reduce_only: bool = False
    status: OrderStatus = OrderStatus.NEW
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    broker_order_id: str = ""
    reject_reason: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    parent_id: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    fills: List[Fill] = field(default_factory=list)

    def validate(self) -> None:
        if self.quantity <= 0 or not math.isfinite(self.quantity):
            raise ValidationError("quantity must be positive and finite", details={"qty": self.quantity})
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
            if self.limit_price is None or self.limit_price <= 0:
                raise ValidationError("limit order needs positive limit_price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            if self.stop_price is None or self.stop_price <= 0:
                raise ValidationError("stop order needs positive stop_price")
        if self.order_type is OrderType.TRAILING_STOP and not (self.tags.get("trail_pct") or self.tags.get("trail_amount")):
            raise ValidationError("trailing stop needs tags trail_pct or trail_amount")
        if self.order_type is OrderType.MARKET and self.limit_price is not None:
            raise ValidationError("market order cannot carry limit_price")

    @property
    def remaining(self) -> float:
        return max(self.quantity - self.filled_qty, 0.0)

    @property
    def signed_remaining(self) -> float:
        return self.side.sign * self.remaining

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("fills", None)
        for k in ("side", "order_type", "tif", "status"):
            d[k] = d[k].value if hasattr(d[k], "value") else d[k]
        return d


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0          # signed
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    last_price: float = 0.0
    multiplier: float = 1.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price * self.multiplier

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.last_price - self.avg_price) * self.multiplier

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-12
