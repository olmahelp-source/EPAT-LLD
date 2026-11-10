import unittest

from helpers import bar, free_cost_config
from algoplatform.brokers import BrokerEvent, CostModel, PaperBroker, ResilientGateway, create_gateway
from algoplatform.core import Fill, OrderStatus, OrderType, Side, TimeInForce, EventBus
from algoplatform.core import events as ev
from algoplatform.core.config import CostConfig
from algoplatform.core.exceptions import (BrokerConnectionError, InvalidTransitionError,
                                          OrderNotFoundError, OrderRejectedError)
from algoplatform.orders import OrderManager
from algoplatform.utils import SimClock

S = OrderStatus


def wired(**broker_kw):
    bus = EventBus()
    fills = []
    bus.subscribe(ev.FILL, lambda t, f: fills.append(f))
    oms = OrderManager(bus, clock=SimClock())
    broker = PaperBroker(100_000, CostModel(free_cost_config()), **broker_kw)
    broker.set_event_handler(oms.on_broker_event)
    broker.connect()
    return oms, broker, fills


def send(oms, broker, *args, **kw):
    o = oms.create(*args, **kw)
    oms.mark_sent(o)
    oms.set_broker_id(o, broker.submit_order(o))
    return o


class OMSTests(unittest.TestCase):
    def setUp(self):
        self.oms = OrderManager(EventBus(), clock=SimClock())

    def fill(self, o, qty, price, fid):
        return Fill(fid, o.client_order_id, o.symbol, o.side, qty, price, None)

    def test_idempotent_create(self):
        a = self.oms.create("X", Side.BUY, 10, idempotency_key="k1")
        b = self.oms.create("X", Side.BUY, 10, idempotency_key="k1")
        self.assertIs(a, b)
        self.assertEqual(len(self.oms.all_orders()), 1)

    def test_lifecycle_with_partial_fills(self):
        o = self.oms.create("X", Side.BUY, 100)
        self.oms.mark_sent(o)
        self.oms.on_broker_event(BrokerEvent("ack", o.client_order_id, "B1"))
        self.assertEqual(o.status, S.ACKNOWLEDGED)
        self.oms.on_broker_event(BrokerEvent("fill", o.client_order_id, fill=self.fill(o, 40, 10.0, "f1")))
        self.assertEqual(o.status, S.PARTIALLY_FILLED)
        self.oms.on_broker_event(BrokerEvent("fill", o.client_order_id, fill=self.fill(o, 60, 11.0, "f2")))
        self.assertEqual(o.status, S.FILLED)
        self.assertAlmostEqual(o.avg_fill_price, 10.6)             # (40*10 + 60*11) / 100

    def test_fill_before_ack_is_accepted(self):
        o = self.oms.create("X", Side.BUY, 10)
        self.oms.mark_sent(o)
        self.oms.on_broker_event(BrokerEvent("fill", o.client_order_id, fill=self.fill(o, 10, 5.0, "f1")))
        self.oms.on_broker_event(BrokerEvent("ack", o.client_order_id, "B1"))     # late ack must not regress state
        self.assertEqual(o.status, S.FILLED)

    def test_invalid_transition_rejected(self):
        o = self.oms.create("X", Side.BUY, 10)
        with self.assertRaises(InvalidTransitionError):
            self.oms._transition(o, S.FILLED)                      # NEW -> FILLED is illegal

    def test_duplicate_fill_ignored(self):
        o = self.oms.create("X", Side.BUY, 10)
        self.oms.mark_sent(o)
        f = self.fill(o, 4, 10.0, "dup")
        self.oms.on_broker_event(BrokerEvent("fill", o.client_order_id, fill=f))
        self.oms.on_broker_event(BrokerEvent("fill", o.client_order_id, fill=f))
        self.assertEqual(o.filled_qty, 4)
        self.assertEqual(self.oms.anomalies[-1]["type"], "duplicate_fill")

    def test_late_fill_after_cancel_is_recorded(self):
        o = self.oms.create("X", Side.BUY, 10)
        self.oms.mark_sent(o)
        self.oms.on_broker_event(BrokerEvent("ack", o.client_order_id, "B"))
        self.oms.request_cancel(o)
        self.oms.on_broker_event(BrokerEvent("cancelled", o.client_order_id))
        self.assertEqual(o.status, S.CANCELLED)
        self.oms.on_broker_event(BrokerEvent("fill", o.client_order_id, fill=self.fill(o, 3, 10.0, "late")))
        self.assertEqual(o.status, S.CANCELLED)                    # status unchanged
        self.assertEqual(o.filled_qty, 3)                           # but the position truth is kept
        self.assertEqual(self.oms.anomalies[-1]["type"], "late_fill")

    def test_cancel_loses_race_to_fill(self):
        o = self.oms.create("X", Side.BUY, 10)
        self.oms.mark_sent(o)
        self.oms.on_broker_event(BrokerEvent("ack", o.client_order_id, "B"))
        self.oms.request_cancel(o)
        self.assertEqual(o.status, S.PENDING_CANCEL)
        self.oms.on_broker_event(BrokerEvent("fill", o.client_order_id, fill=self.fill(o, 10, 9.0, "f")))
        self.assertEqual(o.status, S.FILLED)

    def test_unknown_order_event_is_logged_not_raised(self):
        self.oms.on_broker_event(BrokerEvent("ack", "nope"))
        self.assertEqual(self.oms.anomalies[-1]["type"], "unknown_order_event")
        with self.assertRaises(OrderNotFoundError):
            self.oms.get("nope")

    def test_reconcile_finds_ghosts_and_orphans(self):
        o = self.oms.create("X", Side.BUY, 10)
        self.oms.mark_sent(o)
        self.oms.on_broker_event(BrokerEvent("ack", o.client_order_id, "B"))
        rep = self.oms.reconcile({"EXT-1"})
        self.assertEqual(rep["ghosts"], [o.client_order_id])
        self.assertEqual(rep["orphans"], ["EXT-1"])
        self.oms.resolve_ghosts(rep["ghosts"])
        self.assertEqual(o.status, S.CANCELLED)

    def test_pending_exposure_and_exclude(self):
        a = self.oms.create("X", Side.BUY, 10)
        b = self.oms.create("X", Side.SELL, 4)
        self.assertEqual(self.oms.pending_exposure()["X"], 6)
        self.assertEqual(self.oms.pending_exposure(exclude=a.client_order_id)["X"], -4)

    def test_expire_day_orders_only_sent(self):
        d = self.oms.create("X", Side.BUY, 1)
        g = self.oms.create("X", Side.BUY, 1, tif=TimeInForce.GTC)
        for o in (d, g):
            self.oms.mark_sent(o)
        expired = self.oms.expire_day_orders()
        self.assertEqual([o.client_order_id for o in expired], [d.client_order_id])
        self.assertEqual(g.status, S.PENDING)


class PaperBrokerTests(unittest.TestCase):
    def test_market_order_fills_at_next_open_and_updates_state(self):
        oms, broker, fills = wired()
        o = send(oms, broker, "X", Side.BUY, 10)
        self.assertEqual(o.status, S.ACKNOWLEDGED)
        broker.process_bar(bar("2024-01-02", o=50.0, h=51, l=49, c=50.5))
        self.assertEqual(o.status, S.FILLED)
        self.assertEqual(o.avg_fill_price, 50.0)
        self.assertEqual(broker.get_positions(), {"X": 10})
        self.assertAlmostEqual(broker.get_cash(), 100_000 - 500)

    def test_costs_move_price_against_trader(self):
        bus = EventBus()
        oms = OrderManager(bus)
        broker = PaperBroker(100_000, CostModel(CostConfig(slippage_bps=10, spread_bps=0, commission_per_share=0.01,
                                                           min_commission=0)))
        broker.set_event_handler(oms.on_broker_event)
        broker.connect()
        buy = send(oms, broker, "X", Side.BUY, 100)
        broker.process_bar(bar("2024-01-02", o=100.0))
        self.assertAlmostEqual(buy.avg_fill_price, 100.10)          # 10 bps worse
        sell = send(oms, broker, "X", Side.SELL, 100)
        broker.process_bar(bar("2024-01-03", o=100.0))
        self.assertAlmostEqual(sell.avg_fill_price, 99.90)

    def test_limit_order_only_fills_when_touched(self):
        oms, broker, _ = wired()
        o = send(oms, broker, "X", Side.BUY, 10, OrderType.LIMIT, limit_price=95.0)
        broker.process_bar(bar("2024-01-02", o=100, h=101, l=96, c=100))
        self.assertEqual(o.status, S.ACKNOWLEDGED)                  # low 96 > 95: no fill
        broker.process_bar(bar("2024-01-03", o=100, h=101, l=94, c=97))
        self.assertEqual(o.status, S.FILLED)
        self.assertEqual(o.avg_fill_price, 95.0)

    def test_limit_gap_improves_price(self):
        oms, broker, _ = wired()
        o = send(oms, broker, "X", Side.BUY, 10, OrderType.LIMIT, limit_price=95.0)
        broker.process_bar(bar("2024-01-02", o=90, h=92, l=88, c=91))   # opens below the limit
        self.assertEqual(o.avg_fill_price, 90.0)

    def test_stop_gap_fills_at_open_not_stop(self):
        oms, broker, _ = wired()
        send(oms, broker, "X", Side.BUY, 10)
        broker.process_bar(bar("2024-01-02", o=100))
        s = send(oms, broker, "X", Side.SELL, 10, OrderType.STOP, stop_price=95.0)
        broker.process_bar(bar("2024-01-03", o=90, h=92, l=88, c=89))
        self.assertEqual(s.avg_fill_price, 90.0)                    # gapped through the stop

    def test_partial_fills_by_participation(self):
        oms, broker, fills = wired(max_participation=0.1)
        o = send(oms, broker, "X", Side.BUY, 1000)
        broker.process_bar(bar("2024-01-02", o=10, v=5000))         # capacity 500
        self.assertEqual(o.status, S.PARTIALLY_FILLED)
        self.assertEqual(o.filled_qty, 500)
        broker.process_bar(bar("2024-01-03", o=10, v=5000))
        self.assertEqual(o.status, S.FILLED)
        self.assertEqual(len(fills), 2)

    def test_ioc_expires_unfilled_remainder(self):
        oms, broker, _ = wired(max_participation=0.1)
        o = send(oms, broker, "X", Side.BUY, 1000, tif=TimeInForce.IOC)
        broker.process_bar(bar("2024-01-02", o=10, v=5000))
        self.assertEqual(o.filled_qty, 500)
        self.assertEqual(o.status, S.EXPIRED)

    def test_duplicate_client_id_rejected(self):
        oms, broker, _ = wired()
        o = send(oms, broker, "X", Side.BUY, 1)
        with self.assertRaises(OrderRejectedError):
            broker.submit_order(o)

    def test_disconnected_and_injected_failure(self):
        oms, broker, _ = wired()
        broker.fail_next(BrokerConnectionError("blip"))
        o = oms.create("X", Side.BUY, 1)
        with self.assertRaises(BrokerConnectionError):
            broker.submit_order(o)
        broker.disconnect()
        with self.assertRaises(BrokerConnectionError):
            broker.submit_order(oms.create("X", Side.BUY, 1))

    def test_cancel_open_order(self):
        oms, broker, _ = wired()
        o = send(oms, broker, "X", Side.BUY, 5, OrderType.LIMIT, limit_price=1.0)
        oms.request_cancel(o)
        broker.cancel_order(o)
        self.assertEqual(o.status, S.CANCELLED)
        self.assertEqual(broker.open_order_ids(), set())

    def test_no_short_when_disabled(self):
        oms, broker, _ = wired(allow_short=False)
        o = send(oms, broker, "X", Side.SELL, 5)
        broker.process_bar(bar("2024-01-02", o=10))
        self.assertEqual(o.status, S.REJECTED)

    def test_registry_creates_paper_gateway(self):
        self.assertIsInstance(create_gateway("paper"), PaperBroker)


class ResilientTests(unittest.TestCase):
    def build(self):
        oms = OrderManager(EventBus())
        primary, secondary = PaperBroker(1000), PaperBroker(1000)
        gw = ResilientGateway(primary, secondary, max_attempts=3, failure_threshold=2)
        gw.set_event_handler(oms.on_broker_event)
        gw.connect()
        return oms, primary, secondary, gw

    def test_retries_transient_failure_on_primary(self):
        oms, p, s, gw = self.build()
        p.fail_next(BrokerConnectionError("blip"), times=1)
        o = oms.create("X", Side.BUY, 1)
        oms.mark_sent(o)
        gw.submit_order(o)
        self.assertIn(o.client_order_id, p.open_order_ids())
        self.assertEqual(s.open_order_ids(), set())

    def test_fails_over_when_primary_is_down(self):
        oms, p, s, gw = self.build()
        p.fail_next(BrokerConnectionError("down"), times=10)
        o = oms.create("X", Side.BUY, 1)
        oms.mark_sent(o)
        gw.submit_order(o)
        self.assertIn(o.client_order_id, s.open_order_ids())
        oms.request_cancel(o)
        gw.cancel_order(o)                                          # cancel routed to the owning broker
        self.assertEqual(s.open_order_ids(), set())

    def test_rejection_is_not_failed_over(self):
        oms, p, s, gw = self.build()
        o = oms.create("X", Side.BUY, 1)
        oms.mark_sent(o)
        gw.submit_order(o)
        with self.assertRaises(OrderRejectedError):
            gw.submit_order(o)                                      # duplicate id => rejection from primary
        self.assertEqual(s.open_order_ids(), set())


if __name__ == "__main__":
    unittest.main()
