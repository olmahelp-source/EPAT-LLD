"""Restart recovery: rebuild OMS, portfolio and kill-switch state from the store, then reconcile with the broker."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..brokers.base import BrokerEvent
from ..core.enums import OrderStatus as S
from ..core.logging import get_logger
from .store import SqliteStore

log = get_logger(__name__)


@dataclass
class RecoveryReport:
    orders: int = 0
    open_orders: int = 0
    fills: int = 0
    positions: Dict[str, float] = field(default_factory=dict)
    kill_switch: str = "ARMED"
    orphan_fills: int = 0
    never_sent: List[str] = field(default_factory=list)          # NEW orders: created but never reached the broker
    unresolved: List[str] = field(default_factory=list)          # PENDING at the crash and unknown to the broker
    reconcile: Dict[str, Any] = field(default_factory=dict)
    halted: bool = False
    halt_reason: str = ""

    @property
    def clean(self) -> bool:
        return not (self.unresolved or self.reconcile.get("positions") or self.reconcile.get("orders", {}).get("ghosts")
                    or self.reconcile.get("orders", {}).get("orphans") or self.orphan_fills)


def recover(engine, store: SqliteStore, halt_on_break: bool = True) -> RecoveryReport:
    """Call once after constructing the engine and before ``start()``.

    1. orders and fills are reloaded into the OMS (idempotency keys and fill ids included, so replays stay idempotent)
    2. the portfolio is rebuilt by replaying fills in their original order
    3. a persisted TRIGGERED kill switch stays triggered (restored without re-running its emergency actions)
    4. orders and positions are reconciled with the broker; any break halts trading for a human to review
    """
    rep = RecoveryReport()
    orders = {o.client_order_id: o for o in store.load_orders()}
    fills = store.load_fills()
    for f in fills:
        o = orders.get(f.client_order_id)
        if o is None:
            rep.orphan_fills += 1                                    # fill without its order: keep position truth anyway
            continue
        o.fills.append(f)
    for o in orders.values():
        if o.status is S.NEW:                                        # never sent: safe to close
            o.status = S.CANCELLED
            o.reject_reason = "never sent before restart"
            rep.never_sent.append(o.client_order_id)
    engine.oms.restore(orders.values())
    for f in fills:
        engine.portfolio.apply_fill(f)
    rep.orders, rep.fills = len(orders), len(fills)
    rep.open_orders = len(engine.oms.open_orders())
    rep.positions = {s: p.quantity for s, p in engine.portfolio.open_positions().items()}

    ks = store.get_kill_switch()
    if ks and ks["state"] == "TRIGGERED":
        engine.risk.kill_switch.restore(ks["reason"], ks["source"])
    rep.kill_switch = engine.risk.kill_switch.state.value

    try:
        broker_open = engine.broker.open_order_ids()
        for oid in [o.client_order_id for o in engine.oms.open_orders() if o.status is S.PENDING]:
            if oid in broker_open:
                engine.oms.on_broker_event(BrokerEvent("ack", oid))
            else:
                rep.unresolved.append(oid)
        rep.reconcile = engine.execution.reconcile()
    except Exception as exc:  # noqa: BLE001 - the broker may be unreachable right after a restart
        rep.reconcile = {"error": repr(exc)}
        rep.halt_reason = f"could not reconcile with broker: {exc!r}"
    if halt_on_break and (not rep.clean or rep.halt_reason):
        reason = rep.halt_reason or "restart reconciliation found differences from the broker"
        engine.risk.kill_switch.restore(reason, "recovery")
        rep.halted, rep.halt_reason = True, reason
        log.error("recovery halted trading: %s", reason)
    return rep
