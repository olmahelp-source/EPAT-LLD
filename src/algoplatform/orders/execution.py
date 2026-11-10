"""Execution service: the only path from a decision to a broker.

    strategy decision -> create order (OMS) -> risk gate -> broker -> events -> OMS -> portfolio

Every order, including rejected ones, is recorded in the OMS so the audit trail is complete.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..brokers.base import BrokerGateway
from ..core.enums import OrderStatus, OrderType, Side, TimeInForce
from ..core.exceptions import (BrokerConnectionError, BrokerError, OrderRejectedError, RateLimitError, ValidationError)
from ..core.logging import get_logger
from ..core.models import Order
from ..portfolio.portfolio import Portfolio
from ..risk.checks import RiskContext
from ..risk.manager import RiskManager
from ..utils.timeutils import Clock
from .oms import OrderManager

log = get_logger(__name__)


class ExecutionService:
    def __init__(self, oms: OrderManager, broker: BrokerGateway, risk: RiskManager, portfolio: Portfolio,
                 clock: Clock, prices: Callable[[], Dict[str, float]],
                 price_times: Optional[Callable[[], Dict[str, Any]]] = None) -> None:
        self.oms, self.broker, self.risk, self.portfolio, self.clock = oms, broker, risk, portfolio, clock
        self._prices = prices
        self._price_times = price_times or (lambda: {})
        self.broker.set_event_handler(self.oms.on_broker_event)

    # ---- context --------------------------------------------------------------------------
    def risk_context(self, exclude: Optional[str] = None) -> RiskContext:
        return RiskContext(
            equity=self.portfolio.equity, cash=self.portfolio.cash,
            positions={s: p.quantity for s, p in self.portfolio.open_positions().items()},
            prices=dict(self._prices()), now=self.clock.now(),
            pending=self.oms.pending_exposure(exclude),
            multipliers={s: i.multiplier for s, i in self.portfolio.instruments.items()},
            price_times=dict(self._price_times()))

    # ---- submit ---------------------------------------------------------------------------
    def submit(self, symbol: str, side: Side, quantity: float, order_type: OrderType = OrderType.MARKET,
               limit_price: Optional[float] = None, stop_price: Optional[float] = None,
               tif: TimeInForce = TimeInForce.DAY, strategy_id: str = "", account_id: str = "",
               reduce_only: bool = False, idempotency_key: Optional[str] = None,
               tags: Optional[Dict[str, Any]] = None, parent_id: Optional[str] = None) -> Order:
        """Create, risk-check and send an order. Never raises for business rejections: inspect
        ``order.status``. Raises only for programming errors such as invalid parameters."""
        order = self.oms.create(symbol, side, quantity, order_type, limit_price, stop_price, tif, strategy_id,
                                account_id, reduce_only, idempotency_key, parent_id=parent_id, tags=tags)
        if order.status is not OrderStatus.NEW:          # idempotent replay of an earlier submit
            return order
        decision = self.risk.evaluate(order, self.risk_context(exclude=order.client_order_id))
        if not decision.allowed:
            self.oms.mark_rejected(order, "; ".join(decision.reasons))
            return order
        try:
            self.oms.mark_sent(order)
            bid = self.broker.submit_order(order)
            self.oms.set_broker_id(order, bid)
            self.risk.monitor.record_success()
        except OrderRejectedError as exc:
            self.oms.mark_rejected(order, f"broker rejected: {exc.message}")
        except (BrokerConnectionError, RateLimitError) as exc:
            self.oms.mark_rejected(order, f"broker unavailable: {exc.message}")
            self.risk.monitor.record_error(f"broker: {exc.message}")
        except BrokerError as exc:
            self.oms.mark_rejected(order, f"broker error: {exc.message}")
            self.risk.monitor.record_error(f"broker: {exc.message}")
        except Exception as exc:  # noqa: BLE001 - unknown failure leaves state ambiguous: fail safe
            log.exception("unexpected error sending %s", order.client_order_id)
            self.oms.mark_rejected(order, f"internal error: {exc!r}")
            self.risk.monitor.record_error(repr(exc))
        return order

    # ---- linked orders ---------------------------------------------------------------------
    def submit_oco(self, symbol: str, side: Side, quantity: float, limit_price: float, stop_price: float,
                   strategy_id: str = "", oto_parent: Optional[str] = None, tags: Optional[Dict[str, Any]] = None) -> List[Order]:
        """One-cancels-other: a limit and a stop on the same side; the first fill cancels the other."""
        group = f"OCO-{self.oms.create_group_id()}"
        base = {**(tags or {}), "oco": group, **({"oto": oto_parent} if oto_parent else {})}
        a = self.submit(symbol, side, quantity, OrderType.LIMIT, limit_price, strategy_id=strategy_id,
                        tags=dict(base), parent_id=oto_parent)
        b = self.submit(symbol, side, quantity, OrderType.STOP, stop_price=stop_price, strategy_id=strategy_id,
                        tags=dict(base), parent_id=oto_parent)
        if a.status is OrderStatus.REJECTED and b.status is not OrderStatus.REJECTED:
            self.cancel(b)                                    # never leave a naked half of an OCO pair
        elif b.status is OrderStatus.REJECTED and a.status is not OrderStatus.REJECTED:
            self.cancel(a)
        return [a, b]

    def submit_bracket(self, symbol: str, side: Side, quantity: float, take_profit: float, stop_loss: float,
                       entry_limit: Optional[float] = None, strategy_id: str = "",
                       tags: Optional[Dict[str, Any]] = None) -> List[Order]:
        """Entry + OCO(take-profit, stop-loss) that only goes live once the entry fills (OTO)."""
        ref = entry_limit or self._prices().get(symbol)
        if ref is not None:
            ok = (stop_loss < ref < take_profit) if side is Side.BUY else (take_profit < ref < stop_loss)
            if not ok:
                raise ValidationError("bracket levels must straddle the entry price (stop on the losing side)",
                                      details={"entry": ref, "take_profit": take_profit, "stop_loss": stop_loss})
        entry = self.submit(symbol, side, quantity, OrderType.LIMIT if entry_limit else OrderType.MARKET,
                            entry_limit, strategy_id=strategy_id, tags=dict(tags or {}))
        if entry.status is OrderStatus.REJECTED:
            return [entry]
        exits = self.submit_oco(symbol, side.opposite, quantity, take_profit, stop_loss, strategy_id,
                                oto_parent=entry.client_order_id, tags=tags)
        return [entry, *exits]

    # ---- cancel / emergency ---------------------------------------------------------------
    def cancel(self, order: Order) -> bool:
        if not self.oms.request_cancel(order):
            return False
        if order.status is OrderStatus.CANCELLED:        # never reached the broker
            return True
        try:
            self.broker.cancel_order(order)
        except OrderRejectedError:
            # broker no longer knows it: it filled or expired; events will settle the state
            log.warning("cancel for %s rejected by broker", order.client_order_id)
        except BrokerError as exc:
            log.error("cancel failed for %s: %r", order.client_order_id, exc)
            self.risk.monitor.record_error("cancel failed")
            return False
        return True

    def cancel_all(self, strategy_id: Optional[str] = None) -> int:
        n = 0
        for o in self.oms.open_orders(strategy_id=strategy_id):
            if self.cancel(o):
                n += 1
        return n

    def flatten_all(self, reason: str = "flatten") -> List[Order]:
        """Close every position with reduce-only market orders (allowed even when halted)."""
        pending = self.oms.pending_exposure()
        orders = []
        for sym, pos in self.portfolio.open_positions().items():
            remaining = pos.quantity + pending.get(sym, 0.0)
            if abs(remaining) < 1e-9:
                continue
            side = Side.SELL if remaining > 0 else Side.BUY
            orders.append(self.submit(sym, side, abs(remaining), OrderType.MARKET, reduce_only=True,
                                      strategy_id="risk", tags={"flatten": True, "reason": reason}))
        return orders

    def reconcile(self) -> Dict[str, Any]:
        """Compare orders and positions with the broker; report (and log) any break."""
        orders = self.oms.reconcile(self.broker.open_order_ids())
        positions = self.portfolio.reconcile(self.broker.get_positions())
        if positions:
            log.error("position break vs broker: %s", positions)
        return {"orders": orders, "positions": positions}
