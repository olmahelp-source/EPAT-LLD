"""Paper (simulated) broker.

It doubles as the simulated exchange for backtests: the same ``submit_order`` /
event interface as a live gateway, so strategy, risk and OMS code run unchanged.
Fills are driven by bars (``process_bar``) or quotes (``update_quote``).

Supported beyond plain market/limit/stop: trailing stops, OCO groups (``tags["oco"]``),
OTO children held until their parent fills (``tags["oto"]``), post-only and minimum-quantity
flags (``tags["post_only"]``, ``tags["min_qty"]``), IOC/FOK, volume-participation partial fills,
and square-root market impact scaled by recent volatility.
"""
from __future__ import annotations

import dataclasses
import math
import threading
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

from ..core.config import CostConfig
from ..core.enums import OrderType, Side, TimeInForce
from ..core.exceptions import OrderRejectedError
from ..core.models import Bar, Fill, Order
from ..utils.helpers import IdGenerator
from .base import BROKERS, BrokerEvent, BrokerGateway
from .costs import CostModel

_STOPS = (OrderType.STOP, OrderType.TRAILING_STOP)


@BROKERS.register("paper")
class PaperBroker(BrokerGateway):
    name = "paper"

    def __init__(self, initial_cash: float = 100_000.0, cost_model: Optional[CostModel] = None,
                 max_participation: Optional[float] = None, allow_short: bool = True,
                 immediate_market: bool = False, credentials=None, paper: bool = True) -> None:
        super().__init__()
        self.cash = float(initial_cash)
        self.costs = cost_model or CostModel(CostConfig())
        self.max_participation = max_participation   # fraction of bar volume per bar
        self.allow_short = allow_short
        self.immediate_market = immediate_market     # fill market orders at last quote on submit
        self._positions: Dict[str, float] = {}
        self._last: Dict[str, float] = {}
        self._open: Dict[str, Order] = {}
        self._held: Dict[str, Order] = {}            # OTO children waiting for their parent's fill
        self._oco: Dict[str, set] = {}
        self._done: set = set()
        self._wm: Dict[str, float] = {}              # trailing-stop watermarks
        self._rets: Dict[str, Deque[float]] = {}
        self._vol: Dict[str, float] = {}
        self._seen_client_ids: set = set()
        self._ids = IdGenerator("BRK")
        self._lock = threading.RLock()
        self._fill_ids = IdGenerator("FIL")
        self._fail_next: List[Exception] = []
        self.now = None  # set by caller (backtest) to stamp fills; falls back to bar ts

    # ---- lifecycle ----------------------------------------------------------------------
    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    # ---- testing hook: simulate outages ---------------------------------------------------
    def fail_next(self, exc: Exception, times: int = 1) -> None:
        self._fail_next.extend([exc] * times)

    def restore(self, fills, open_orders, last_prices=None) -> None:
        """Rebuild simulator state from a journal after a restart (a real broker keeps its own state; this one
        lives in process memory). ``fills`` fix positions and cash; ``open_orders`` become live resting orders."""
        with self._lock:
            for f in fills:
                self._positions[f.symbol] = self._positions.get(f.symbol, 0.0) + f.signed_qty
                self.cash -= f.signed_qty * f.price + f.commission
                self._seen_client_ids.add(f.client_order_id)
            for o in open_orders:
                self._seen_client_ids.add(o.client_order_id)
                self._open[o.client_order_id] = dataclasses.replace(o, fills=[])
            self._last.update(last_prices or {})

    # ---- order entry --------------------------------------------------------------------
    def submit_order(self, order: Order) -> str:
        with self._lock:
            self._require_connected()
            if self._fail_next:
                raise self._fail_next.pop(0)
            if order.client_order_id in self._seen_client_ids:
                raise OrderRejectedError("duplicate client order id", code="DUPLICATE_ID")
            if order.quantity <= 0:
                raise OrderRejectedError("non-positive quantity")
            tags = order.tags or {}
            if order.order_type is OrderType.TRAILING_STOP and not (tags.get("trail_pct") or tags.get("trail_amount")):
                raise OrderRejectedError("trailing stop needs tags trail_pct or trail_amount")
            last = self._last.get(order.symbol)
            if tags.get("post_only") and order.order_type is OrderType.LIMIT and last is not None:
                crosses = order.limit_price >= last if order.side is Side.BUY else order.limit_price <= last
                if crosses:
                    raise OrderRejectedError("post-only order would take liquidity", code="POST_ONLY")
            parent = tags.get("oto")
            if parent and parent not in self._open and parent not in self._held and parent not in self._done:
                raise OrderRejectedError("unknown OTO parent order", code="BAD_OTO")
            self._seen_client_ids.add(order.client_order_id)
            bid = self._ids.next()
            # the broker keeps its OWN copy; the OMS copy is only changed via events
            mine = dataclasses.replace(order, broker_order_id=bid, filled_qty=0.0, fills=[])
            if parent and parent not in self._done:
                self._held[order.client_order_id] = mine
            else:
                self._open[order.client_order_id] = mine
            if tags.get("oco"):
                self._oco.setdefault(tags["oco"], set()).add(order.client_order_id)
            self._emit(BrokerEvent("ack", order.client_order_id, bid))
            if self.immediate_market and mine.order_type is OrderType.MARKET and mine.symbol in self._last \
                    and order.client_order_id in self._open:
                self._execute(mine, self._last[mine.symbol], mine.remaining, ts=self.now)
            return bid

    def cancel_order(self, order: Order) -> None:
        with self._lock:
            self._require_connected()
            mine = self._open.pop(order.client_order_id, None) or self._held.pop(order.client_order_id, None)
            if mine is None:
                raise OrderRejectedError("order not open at broker", code="UNKNOWN_ORDER")
            self._emit(BrokerEvent("cancelled", order.client_order_id, mine.broker_order_id))
            self._cancel_children(order.client_order_id, "parent cancelled")

    def get_positions(self) -> Dict[str, float]:
        with self._lock:
            return {s: q for s, q in self._positions.items() if abs(q) > 1e-12}

    def get_cash(self) -> float:
        return self.cash

    def open_order_ids(self) -> set:
        return set(self._open) | set(self._held)

    # ---- linked-order helpers -------------------------------------------------------------
    def _cancel_children(self, parent_cid: str, reason: str) -> None:
        for cid, o in list(self._held.items()):
            if (o.tags or {}).get("oto") == parent_cid:
                self._held.pop(cid)
                self._emit(BrokerEvent("cancelled", cid, o.broker_order_id, reason=reason))
                self._cancel_children(cid, reason)

    def _release_children(self, parent_cid: str) -> None:
        for cid, o in list(self._held.items()):
            if (o.tags or {}).get("oto") == parent_cid:
                self._open[cid] = self._held.pop(cid)

    def _cancel_siblings(self, order: Order) -> None:
        group = (order.tags or {}).get("oco")
        if not group:
            return
        for cid in list(self._oco.get(group, ())):
            if cid == order.client_order_id:
                continue
            sib = self._open.pop(cid, None) or self._held.pop(cid, None)
            self._oco[group].discard(cid)
            if sib is not None:
                self._emit(BrokerEvent("cancelled", cid, sib.broker_order_id, reason="OCO sibling filled"))

    def _end(self, order: Order, kind: str, reason: str) -> None:
        """Terminate an order at the broker (expire/reject) and drop anything waiting on it."""
        self._open.pop(order.client_order_id, None)
        self._emit(BrokerEvent(kind, order.client_order_id, order.broker_order_id, reason=reason))
        self._cancel_children(order.client_order_id, reason)

    # ---- market-driven matching ----------------------------------------------------------
    def update_quote(self, symbol: str, price: float, ts=None, volume: float = 0.0) -> None:
        """Tick-style update: treat as a degenerate bar."""
        self.process_bar(Bar(ts, symbol, price, price, price, price, volume))

    def process_bar(self, bar: Bar) -> None:
        with self._lock:
            prev = self._last.get(bar.symbol)
            self._last[bar.symbol] = bar.close
            if prev:
                rets = self._rets.setdefault(bar.symbol, deque(maxlen=20))
                rets.append(math.log(bar.close / prev))
                if len(rets) >= 5:
                    self._vol[bar.symbol] = float(np.std(rets, ddof=1))
            capacity = (bar.volume * self.max_participation) if (self.max_participation and bar.volume > 0) else None
            candidates = [o for o in self._open.values() if o.symbol == bar.symbol]
            # conservative: within an OCO pair the protective stop is assumed to trigger first
            candidates.sort(key=lambda o: 0 if ((o.tags or {}).get("oco") and o.order_type in _STOPS) else 1)
            for order in candidates:
                if order.client_order_id not in self._open:          # cancelled by an OCO sibling this bar
                    continue
                px = self._match_price(order, bar)
                if px is not None:
                    qty = order.remaining if capacity is None else min(order.remaining, capacity)
                    min_qty = (order.tags or {}).get("min_qty")
                    if min_qty and qty < min(min_qty, order.remaining):
                        continue                                      # too small to execute under min-quantity
                    if capacity is not None:
                        capacity -= qty
                    if qty > 0:
                        if order.tif is TimeInForce.FOK and qty < order.remaining:
                            self._end(order, "expired", "FOK not fully fillable")
                            continue
                        self._execute(order, px, qty, ts=bar.ts, bar=bar)
                if order.client_order_id in self._open and order.tif in (TimeInForce.IOC, TimeInForce.FOK):
                    self._end(order, "expired", "IOC remainder cancelled")

    def _match_price(self, order: Order, bar: Bar) -> Optional[float]:
        """Price at which ``order`` would trade during ``bar``, or None."""
        buy = order.side is Side.BUY
        t = order.order_type
        if t is OrderType.MARKET:
            return bar.open
        if t is OrderType.LIMIT:
            lp = order.limit_price
            if buy and bar.low <= lp:
                return min(bar.open, lp)
            if not buy and bar.high >= lp:
                return max(bar.open, lp)
            return None
        if t is OrderType.TRAILING_STOP:
            tags = order.tags
            wm = self._wm.setdefault(order.client_order_id, bar.open)
            dist = tags.get("trail_amount") or wm * tags["trail_pct"]
            if buy:                                                   # protects a short: stop trails the lows
                stop = wm + dist
                if bar.high >= stop:
                    return max(bar.open, stop)
                self._wm[order.client_order_id] = min(wm, bar.low)
            else:
                stop = wm - dist
                if bar.low <= stop:
                    return min(bar.open, stop)
                self._wm[order.client_order_id] = max(wm, bar.high)
            return None
        sp = order.stop_price
        triggered = (buy and bar.high >= sp) or (not buy and bar.low <= sp)
        if not triggered:
            return None
        stop_px = max(bar.open, sp) if buy else min(bar.open, sp)   # gaps fill at open
        if t is OrderType.STOP:
            return stop_px
        lp = order.limit_price                                      # STOP_LIMIT
        if buy and stop_px <= lp:
            return stop_px
        if not buy and stop_px >= lp:
            return stop_px
        return None

    def _execute(self, order: Order, ref_price: float, qty: float, ts, bar: Optional[Bar] = None) -> None:
        pos = self._positions.get(order.symbol, 0.0)
        new_pos = pos + order.side.sign * qty
        if not self.allow_short and new_pos < -1e-12:
            self._end(order, "rejected", "short selling disabled")
            return
        is_limit_like = order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT)
        # passive limit fills get no slippage; marketable ones pay spread, slippage and impact
        px = ref_price if is_limit_like else self.costs.fill_price(
            order.side, ref_price, qty, adv=(bar.volume if bar and bar.volume > 0 else None),
            volatility=self._vol.get(order.symbol))
        commission = self.costs.commission(qty, px) + self.costs.tax(order.side, qty * px)
        self.cash -= order.side.sign * qty * px + commission
        self._positions[order.symbol] = new_pos
        fill = Fill(self._fill_ids.next(), order.client_order_id, order.symbol, order.side, qty, px,
                    ts, commission, order.strategy_id, order.account_id, order.broker_order_id)
        order.filled_qty += qty
        complete = order.filled_qty >= order.quantity - 1e-12
        if complete:
            self._open.pop(order.client_order_id, None)
            self._done.add(order.client_order_id)
        self._emit(BrokerEvent("fill", order.client_order_id, order.broker_order_id, fill=fill))
        self._cancel_siblings(order)                                  # first fill of an OCO leg cancels the rest
        if complete:
            self._release_children(order.client_order_id)              # OTO children go live
