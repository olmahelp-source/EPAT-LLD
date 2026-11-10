"""Order Management System: order store, state machine, fills, reconciliation."""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from ..brokers.base import BrokerEvent
from ..core import events as ev
from ..core.enums import OrderStatus as S, OrderType, Side, TimeInForce
from ..core.events import EventBus
from ..core.exceptions import (InvalidTransitionError, OrderNotFoundError)
from ..core.logging import AuditLog, get_logger
from ..core.models import Fill, Order
from ..utils.helpers import IdGenerator
from ..utils.timeutils import Clock, SystemClock

log = get_logger(__name__)

# Legal transitions. Fills may arrive before the ack (PENDING -> FILLED) and a
# cancel request may lose the race against a fill (PENDING_CANCEL -> FILLED).
TRANSITIONS: Dict[S, Set[S]] = {
    S.NEW: {S.PENDING, S.REJECTED, S.CANCELLED},
    S.PENDING: {S.ACKNOWLEDGED, S.PARTIALLY_FILLED, S.FILLED, S.REJECTED, S.CANCELLED,
                S.EXPIRED, S.PENDING_CANCEL},
    S.ACKNOWLEDGED: {S.PARTIALLY_FILLED, S.FILLED, S.PENDING_CANCEL, S.CANCELLED, S.EXPIRED, S.REJECTED},
    S.PARTIALLY_FILLED: {S.PARTIALLY_FILLED, S.FILLED, S.PENDING_CANCEL, S.CANCELLED, S.EXPIRED},
    S.PENDING_CANCEL: {S.CANCELLED, S.FILLED, S.PARTIALLY_FILLED, S.ACKNOWLEDGED, S.EXPIRED},
    S.FILLED: set(), S.CANCELLED: set(), S.REJECTED: set(), S.EXPIRED: set(),
}


class OrderManager:
    def __init__(self, bus: Optional[EventBus] = None, audit: Optional[AuditLog] = None,
                 clock: Optional[Clock] = None, id_prefix: str = "ORD") -> None:
        self.bus = bus or EventBus()
        self.audit = audit or AuditLog()
        self.clock = clock or SystemClock()
        self._orders: Dict[str, Order] = {}
        self._open: Dict[str, Order] = {}          # index of non-terminal orders (keeps queries O(open))
        self._idem: Dict[str, str] = {}
        self._fill_ids: Set[str] = set()
        self._ids = IdGenerator(id_prefix)
        self._lock = threading.RLock()
        self.anomalies: List[Dict[str, Any]] = []

    # ---- creation ---------------------------------------------------------------------
    def create(self, symbol: str, side: Side, quantity: float, order_type: OrderType = OrderType.MARKET,
               limit_price: Optional[float] = None, stop_price: Optional[float] = None,
               tif: TimeInForce = TimeInForce.DAY, strategy_id: str = "", account_id: str = "",
               reduce_only: bool = False, idempotency_key: Optional[str] = None,
               parent_id: Optional[str] = None, tags: Optional[Dict[str, Any]] = None) -> Order:
        """Create an order. Re-using an ``idempotency_key`` returns the original order
        instead of creating a duplicate (safe to call again after a timeout)."""
        with self._lock:
            if idempotency_key and idempotency_key in self._idem:
                return self._orders[self._idem[idempotency_key]]
            now = self.clock.now()
            order = Order(self._ids.next(), symbol, side, float(quantity), order_type, limit_price, stop_price,
                          tif, strategy_id, account_id, reduce_only, created_at=now, updated_at=now,
                          parent_id=parent_id, tags=dict(tags or {}))
            order.validate()
            if idempotency_key:
                order.tags["idempotency_key"] = idempotency_key        # persisted, so recovery can rebuild the key index
            self._orders[order.client_order_id] = order
            self._open[order.client_order_id] = order
            if idempotency_key:
                self._idem[idempotency_key] = order.client_order_id
        self.audit.record(strategy_id or "oms", "order.created", order=order.client_order_id,
                          symbol=symbol, side=side.value, qty=quantity)
        self._notify(order)
        return order

    def restore(self, orders) -> None:
        """Load previously persisted orders (restart recovery). Rebuilds the open-order index, the
        idempotency-key index and the fill-id set so replayed events stay idempotent."""
        with self._lock:
            for o in orders:
                self._orders[o.client_order_id] = o
                if o.status.is_open:
                    self._open[o.client_order_id] = o
                else:
                    self._open.pop(o.client_order_id, None)
                key = o.tags.get("idempotency_key")
                if key:
                    self._idem[key] = o.client_order_id
                for f in o.fills:
                    self._fill_ids.add(f.fill_id)

    def create_group_id(self) -> str:
        return self._ids.next()

    def get(self, client_order_id: str) -> Order:
        try:
            return self._orders[client_order_id]
        except KeyError:
            raise OrderNotFoundError(f"unknown order {client_order_id}") from None

    # ---- state machine ---------------------------------------------------------------
    def _transition(self, order: Order, new: S, reason: str = "") -> None:
        if new is order.status and new is not S.PARTIALLY_FILLED:
            return
        if new not in TRANSITIONS[order.status]:
            raise InvalidTransitionError(f"{order.status.value} -> {new.value} not allowed",
                                         details={"order": order.client_order_id})
        order.status = new
        order.updated_at = self.clock.now()
        if new.is_terminal:
            self._open.pop(order.client_order_id, None)
        if reason:
            order.reject_reason = reason

    def mark_sent(self, order: Order) -> None:
        with self._lock:
            self._transition(order, S.PENDING)
        self._notify(order)

    def mark_rejected(self, order: Order, reason: str) -> None:
        with self._lock:
            if order.status.is_terminal:
                return
            self._transition(order, S.REJECTED, reason)
        self.audit.record("oms", "order.rejected", order=order.client_order_id, reason=reason)
        self._notify(order)

    def set_broker_id(self, order: Order, broker_order_id: str) -> None:
        order.broker_order_id = broker_order_id

    def request_cancel(self, order: Order) -> bool:
        """Move to PENDING_CANCEL. Returns False if the order is already finished."""
        with self._lock:
            if order.status.is_terminal:
                return False
            if order.status is S.NEW:
                self._transition(order, S.CANCELLED)
            else:
                self._transition(order, S.PENDING_CANCEL)
        self._notify(order)
        return True

    # ---- broker events ------------------------------------------------------------------
    def on_broker_event(self, event: BrokerEvent) -> None:
        """Entry point wired to ``BrokerGateway.set_event_handler``."""
        try:
            order = self.get(event.client_order_id)
        except OrderNotFoundError:
            self.anomalies.append({"type": "unknown_order_event", "event": event})
            log.error("event for unknown order %s (%s)", event.client_order_id, event.kind)
            return
        with self._lock:
            if event.broker_order_id and not order.broker_order_id:
                order.broker_order_id = event.broker_order_id
            kind = event.kind
            if kind == "ack":
                if order.status in (S.PENDING, S.NEW):
                    if order.status is S.NEW:
                        self._transition(order, S.PENDING)
                    self._transition(order, S.ACKNOWLEDGED)
            elif kind == "fill" and event.fill is not None:
                self._apply_fill_locked(order, event.fill)
            elif kind == "rejected":
                if not order.status.is_terminal:
                    self._transition(order, S.REJECTED, event.reason)
            elif kind == "cancelled":
                if order.status is S.PENDING_CANCEL or order.status.is_open:
                    self._transition(order, S.CANCELLED, event.reason)
            elif kind == "cancel_rejected":
                # the cancel lost (e.g. already filled): back to the live state it was in
                if order.status is S.PENDING_CANCEL:
                    self._transition(order, S.PARTIALLY_FILLED if order.filled_qty > 0 else S.ACKNOWLEDGED, event.reason)
            elif kind == "expired":
                if order.status.is_open:
                    self._transition(order, S.EXPIRED, event.reason)
            else:
                log.warning("unhandled broker event kind %s", kind)
        if kind == "fill" and event.fill is not None:
            return  # _apply_fill_locked already published
        self._notify(order)

    def _apply_fill_locked(self, order: Order, fill: Fill) -> None:
        if fill.fill_id in self._fill_ids:               # duplicate delivery -> idempotent
            self.anomalies.append({"type": "duplicate_fill", "fill": fill.fill_id})
            return
        if fill.quantity <= 0 or fill.price <= 0:
            self.anomalies.append({"type": "bad_fill", "fill": fill.fill_id})
            return
        new_qty = order.filled_qty + fill.quantity
        if new_qty > order.quantity + 1e-9:              # overfill: broker bug; do not trust blindly
            self.anomalies.append({"type": "overfill", "fill": fill.fill_id, "order": order.client_order_id})
            log.error("overfill on %s: %s > %s", order.client_order_id, new_qty, order.quantity)
        self._fill_ids.add(fill.fill_id)
        prev_notional = order.avg_fill_price * order.filled_qty
        order.filled_qty = new_qty
        order.avg_fill_price = (prev_notional + fill.price * fill.quantity) / new_qty
        order.fills.append(fill)
        if order.status.is_terminal and order.status is not S.FILLED:
            # late fill after cancel/expire: position truth wins, status is left alone
            self.anomalies.append({"type": "late_fill", "fill": fill.fill_id, "order": order.client_order_id})
            log.warning("late fill %s on %s order %s", fill.fill_id, order.status.value, order.client_order_id)
        else:
            done = order.filled_qty >= order.quantity - 1e-9
            self._transition(order, S.FILLED if done else S.PARTIALLY_FILLED)
        order.updated_at = self.clock.now()
        self.audit.record("broker", "order.fill", order=order.client_order_id, qty=fill.quantity, price=fill.price)
        self.bus.publish(ev.FILL, fill)
        self.bus.publish(ev.ORDER_UPDATED, order)

    def _notify(self, order: Order) -> None:
        self.bus.publish(ev.ORDER_UPDATED, order)

    # ---- queries ---------------------------------------------------------------------
    def open_orders(self, symbol: Optional[str] = None, strategy_id: Optional[str] = None) -> List[Order]:
        with self._lock:
            return [o for o in self._open.values()
                    if (symbol is None or o.symbol == symbol)
                    and (strategy_id is None or o.strategy_id == strategy_id)]

    def all_orders(self) -> List[Order]:
        with self._lock:
            return list(self._orders.values())

    def pending_exposure(self, exclude: Optional[str] = None) -> Dict[str, float]:
        """Signed quantity still working per symbol (so risk sees orders in flight).
        ``exclude`` omits one order, e.g. the one currently being risk-checked."""
        out: Dict[str, float] = defaultdict(float)
        for o in self.open_orders():
            if o.client_order_id != exclude:
                out[o.symbol] += o.signed_remaining
        return dict(out)

    # ---- housekeeping -----------------------------------------------------------------
    def expire_day_orders(self) -> List[Order]:
        expired = []
        with self._lock:
            for o in self._orders.values():
                if o.status.is_open and o.tif is TimeInForce.DAY and o.status is not S.NEW:
                    self._transition(o, S.EXPIRED, "end of day")
                    expired.append(o)
        for o in expired:
            self._notify(o)
        return expired

    def reconcile(self, broker_open_ids: Set[str]) -> Dict[str, List[str]]:
        """Compare local open orders against the broker's list.

        ``ghosts``: open locally, unknown at broker (lost ack / dropped order).
        ``orphans``: open at broker, unknown locally (placed outside this system).
        Nothing is changed automatically; the caller decides how to resolve.
        """
        with self._lock:
            local_open = {o.client_order_id for o in self._orders.values()
                          if o.status in (S.ACKNOWLEDGED, S.PARTIALLY_FILLED, S.PENDING_CANCEL)}
        report = {"ghosts": sorted(local_open - broker_open_ids),
                  "orphans": sorted(broker_open_ids - set(self._orders))}
        if report["ghosts"] or report["orphans"]:
            self.audit.record("oms", "order.reconcile_break", **report)
            log.error("reconciliation break: %s", report)
        return report

    def resolve_ghosts(self, ghosts: List[str], reason: str = "missing at broker") -> None:
        with self._lock:
            for cid in ghosts:
                o = self._orders.get(cid)
                if o and o.status.is_open:
                    o.status = S.CANCELLED          # administrative close, bypasses the table
                    self._open.pop(cid, None)
                    o.reject_reason = reason
                    o.updated_at = self.clock.now()
        for cid in ghosts:
            self._notify(self.get(cid))
