"""Journal: subscribes to the event bus and writes every order update, fill and kill-switch change to the store."""
from __future__ import annotations

from typing import Callable, Optional

from ..core import events as ev
from ..core.events import EventBus
from ..core.logging import get_logger
from ..core.models import Fill, Order
from .store import SqliteStore

log = get_logger(__name__)


class Journal:
    """A failing write is never silent: ``on_error`` is called (the engine wires it to the kill switch), because
    trading on after state can no longer be persisted would make a restart unsafe."""

    def __init__(self, store: SqliteStore, bus: EventBus, on_error: Optional[Callable[[Exception], None]] = None) -> None:
        self.store, self.on_error = store, on_error
        self.errors = 0
        self._unsub = [bus.subscribe(ev.ORDER_UPDATED, self._order), bus.subscribe(ev.FILL, self._fill),
                       bus.subscribe(ev.KILL_SWITCH, self._kill)]

    def close(self) -> None:
        for u in self._unsub:
            u()

    def _guard(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            log.error("journal write failed: %r", exc)
            if self.on_error:
                self.on_error(exc)

    def _order(self, _t: str, order: Order) -> None:
        def w() -> None:
            self.store.save_order(order)
            self.store.append_event("order", {"id": order.client_order_id, "status": order.status.value,
                                              "filled": order.filled_qty})
        self._guard(w)

    def _fill(self, _t: str, fill: Fill) -> None:
        def w() -> None:
            if self.store.save_fill(fill):
                self.store.append_event("fill", {"id": fill.fill_id, "order": fill.client_order_id, "symbol": fill.symbol,
                                                 "qty": fill.signed_qty, "price": fill.price})
        self._guard(w)

    def _kill(self, _t: str, payload: dict) -> None:
        def w() -> None:
            self.store.set_kill_switch(payload.get("state", ""), payload.get("reason", ""), payload.get("source", ""))
            self.store.append_event("kill_switch", payload)
        self._guard(w)
