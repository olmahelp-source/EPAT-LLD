"""AlgoExecutionManager: works parent orders bar by bar through the normal risk-checked path."""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from ..core import events as ev
from ..core.enums import OrderStatus, OrderType, Side
from ..core.exceptions import ValidationError
from ..core.logging import get_logger
from ..core.models import Bar, Fill, Instrument
from ..orders.execution import ExecutionService
from ..orders.oms import OrderManager
from ..utils.helpers import IdGenerator
from .algos import ExecutionAlgo, SliceContext, make_algo
from .parent import CANCELLED, DONE, EXPIRED, FAILED, ParentOrder

log = get_logger(__name__)


class AlgoExecutionManager:
    MAX_CONSECUTIVE_REJECTS = 3

    def __init__(self, execution: ExecutionService, oms: OrderManager, bus, instruments: Optional[Dict[str, Instrument]] = None,
                 volume_lookback: int = 5) -> None:
        self.execution, self.oms = execution, oms
        self.instruments = instruments if instruments is not None else {}
        self.volume_lookback = volume_lookback
        self.parents: Dict[str, ParentOrder] = {}
        self._child_to_parent: Dict[str, str] = {}
        self._volumes: Dict[str, List[float]] = {}
        self._ids = IdGenerator("PAR")
        bus.subscribe(ev.FILL, self._on_fill)

    # ---- lifecycle ----------------------------------------------------------------------------
    def start(self, symbol: str, side: Side, quantity: float, algo: "ExecutionAlgo | str", arrival_price: float,
              strategy_id: str = "", start_ts=None, on_complete: Optional[Callable] = None, **algo_params) -> ParentOrder:
        if quantity <= 0:
            raise ValidationError("quantity must be positive")
        if isinstance(algo, str):
            algo = make_algo(algo, **algo_params)
        p = ParentOrder(self._ids.next(), symbol, side, float(quantity), algo, strategy_id, arrival_price, start_ts,
                        max_bars=algo.horizon(), on_complete=on_complete)
        self.parents[p.parent_id] = p
        return p

    def active(self, symbol: Optional[str] = None) -> List[ParentOrder]:
        return [p for p in self.parents.values() if p.is_active and (symbol is None or p.symbol == symbol)]

    def cancel_parent(self, parent: ParentOrder, reason: str = "cancelled", status: str = CANCELLED) -> None:
        for cid in parent.children:
            o = self.oms.get(cid)
            if o.status.is_open:
                self.execution.cancel(o)
        parent.finish(status, reason)

    def cancel_all(self, symbol: Optional[str] = None, reason: str = "cancelled") -> int:
        ps = self.active(symbol)
        for p in ps:
            self.cancel_parent(p, reason)
        return len(ps)

    def signed_unsent(self, symbol: str) -> float:
        """Quantity parents still intend to trade but have not yet sent (working children are already
        counted by the OMS pending exposure, so they are excluded here to avoid double counting)."""
        total = 0.0
        for p in self.active(symbol):
            working = sum(self.oms.get(c).remaining for c in p.children if self.oms.get(c).status.is_open)
            total += p.side.sign * max(p.total - p.filled - working, 0.0)
        return total

    # ---- events -------------------------------------------------------------------------------
    def _on_fill(self, _topic: str, fill: Fill) -> None:
        pid = self._child_to_parent.get(fill.client_order_id)
        if pid:
            self.parents[pid].record_fill(fill.quantity, fill.price, fill.commission)

    # ---- per-bar work --------------------------------------------------------------------------
    def on_bars(self, bars: Dict[str, Bar]) -> None:
        for b in bars.values():
            self._volumes.setdefault(b.symbol, []).append(b.volume)
        for p in list(self.active()):
            bar = bars.get(p.symbol)
            if bar is not None:
                self._work(p, bar)

    def _children(self, p: ParentOrder):
        return [self.oms.get(c) for c in p.children]

    def _ctx(self, p: ParentOrder, bar: Bar) -> SliceContext:
        kids = self._children(p)
        open_kids = [o for o in kids if o.status.is_open]
        working = sum(o.remaining for o in open_kids)
        age = max((p.bars_elapsed - p.child_bar[o.client_order_id] for o in open_kids), default=0)
        vols = self._volumes.get(p.symbol, [])[-self.volume_lookback:]
        inst = self.instruments.get(p.symbol) or Instrument(p.symbol)
        return SliceContext(p, bar, p.bars_elapsed, max(p.total - p.filled - working, 0.0), working, age,
                            float(np.mean(vols)) if vols else bar.volume, inst.lot_size, inst.tick_size)

    def work_now(self, p: ParentOrder, bar: Bar) -> None:
        """Send a parent's first slice immediately (used when a strategy starts it mid-bar)."""
        if p.is_active:
            self._work(p, bar)

    def _work(self, p: ParentOrder, bar: Bar) -> None:
        kids = self._children(p)
        # a rejected child means risk or the broker said no: count it, and stop if it keeps happening
        if kids:
            last = kids[-1]
            if last.status is OrderStatus.REJECTED:
                if not last.tags.get("_counted"):
                    last.tags["_counted"] = True
                    p.rejections += 1
            else:
                p.rejections = 0
        if p.rejections >= self.MAX_CONSECUTIVE_REJECTS:
            self.cancel_parent(p, "too many rejected children", FAILED)
            return
        ctx = self._ctx(p, bar)
        if ctx.uncommitted < ctx.lot - 1e-9 and ctx.working == 0:
            p.finish(DONE)                                       # nothing left (remainder below one lot is dropped)
            return
        if p.bars_elapsed >= p.max_bars:
            self.cancel_parent(p, "horizon reached", EXPIRED)
            return
        d = p.algo.slice(ctx)
        if d.cancel_working:
            for o in self._children(p):
                if o.status.is_open:
                    self.execution.cancel(o)
            ctx = self._ctx(p, bar)
            d = p.algo.slice(ctx)
        if d.qty > 0:
            otype = OrderType.LIMIT if d.limit_price else OrderType.MARKET
            child = self.execution.submit(p.symbol, p.side, d.qty, otype, d.limit_price, strategy_id=p.strategy_id,
                                          parent_id=p.parent_id, tags={"arrival_price": p.arrival_price,
                                                                       "parent": p.parent_id})
            p.children.append(child.client_order_id)
            p.child_bar[child.client_order_id] = p.bars_elapsed
            self._child_to_parent[child.client_order_id] = p.parent_id
        p.bars_elapsed += 1
