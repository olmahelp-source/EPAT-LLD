"""Parent order: one trading intent worked as many child orders by an algo."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..core.enums import Side

ACTIVE, DONE, CANCELLED, FAILED, EXPIRED = "ACTIVE", "DONE", "CANCELLED", "FAILED", "EXPIRED"


@dataclass
class ParentOrder:
    parent_id: str
    symbol: str
    side: Side
    total: float
    algo: Any
    strategy_id: str = ""
    arrival_price: float = 0.0
    start_ts: Optional[datetime] = None
    max_bars: int = 200
    status: str = ACTIVE
    bars_elapsed: int = 0
    filled: float = 0.0
    fill_notional: float = 0.0
    commission: float = 0.0
    rejections: int = 0
    children: List[str] = field(default_factory=list)
    child_bar: Dict[str, int] = field(default_factory=dict)       # child id -> bar index when sent
    on_complete: Optional[Callable[["ParentOrder"], None]] = None
    reason: str = ""

    @property
    def avg_price(self) -> float:
        return self.fill_notional / self.filled if self.filled else 0.0

    @property
    def signed_remaining(self) -> float:
        return self.side.sign * max(self.total - self.filled, 0.0)

    @property
    def is_active(self) -> bool:
        return self.status == ACTIVE

    def record_fill(self, qty: float, price: float, commission: float = 0.0) -> None:
        self.filled += qty
        self.fill_notional += qty * price
        self.commission += commission

    def finish(self, status: str, reason: str = "") -> None:
        if self.status != ACTIVE:
            return
        self.status, self.reason = status, reason
        if self.on_complete:
            self.on_complete(self)
