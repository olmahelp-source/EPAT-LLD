import os
import sqlite3
import tempfile
import unittest
from datetime import datetime

import numpy as np

from helpers import bar, free_config, market_from_closes
from algoplatform.app.engine import TradingEngine
from algoplatform.brokers import BrokerEvent, CostModel, PaperBroker
from algoplatform.core import EventBus, Fill, Order, OrderStatus, OrderType, Side, TimeInForce
from algoplatform.core import events as ev
from algoplatform.core.exceptions import ValidationError
from algoplatform.persistence import (SCHEMA_VERSION, Journal, SqliteStore, StoreError, fill_from_dict, fill_to_dict)
from algoplatform.strategy import SMACrossover
from algoplatform.utils import SimClock

S = OrderStatus


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "t.db")

    def tearDown(self):
        self.dir.cleanup()

    def test_schema_and_in_place_migration(self):
        old = SqliteStore(self.path, _max_version=1)
        self.assertEqual(old.version, 1)
        old.append_event("order", {"id": "o1"})
        old.close()
        new = SqliteStore(self.path)
        self.assertEqual(new.version, SCHEMA_VERSION)
        new.set_kill_switch("TRIGGERED", "why", "test")                                # table added by migration 2
        self.assertEqual(new.get_kill_switch()["reason"], "why")
        self.assertEqual(new.last_seq(), 1)                                            # v1 data survived the upgrade

    def test_refuses_a_database_from_the_future(self):
        con = sqlite3.connect(self.path)
        con.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        con.execute("INSERT INTO schema_version VALUES (99)")
        con.commit()
        con.close()
        with self.assertRaises(StoreError):
            SqliteStore(self.path)

    def test_journal_ordering_filters_and_corruption(self):
        s = SqliteStore(self.path)
        for i in range(5):
            s.append_event("fill" if i % 2 else "order", {"n": i})
        rows, bad = s.events()
        self.assertEqual([r[3]["n"] for r in rows], [0, 1, 2, 3, 4])
        self.assertEqual([r[3]["n"] for r in s.events(topics=["fill"])[0]], [1, 3])
        self.assertEqual([r[3]["n"] for r in s.events(after_seq=3)[0]], [3, 4])
        s._db.execute("UPDATE events SET payload='{not json' WHERE seq=2")             # damaged row
        rows, bad = s.events()
        self.assertEqual((len(rows), bad), (4, 1))                                     # skipped and counted, not fatal
        self.assertEqual(s.sequence_gaps(), [])
        s._db.execute("DELETE FROM events WHERE seq=3")
        self.assertEqual(s.sequence_gaps(), [3])                                       # a missing row is detectable

    def test_orders_upsert_and_fills_are_idempotent(self):
        s = SqliteStore(self.path)
        o = Order("o1", "X", Side.BUY, 10)
        s.save_order(o)
        o.status = S.FILLED
        s.save_order(o)
        self.assertEqual([x.status for x in s.load_orders()], [S.FILLED])
        f = Fill("f1", "o1", "X", Side.BUY, 10, 5.0, datetime(2024, 1, 2))
        self.assertTrue(s.save_fill(f))
        self.assertFalse(s.save_fill(f))
        self.assertEqual(len(s.load_fills()), 1)

    def test_codec_roundtrip_preserves_everything_that_matters(self):
        o = Order("o1", "X", Side.SELL, 12.5, OrderType.LIMIT, limit_price=101.25, tif=TimeInForce.GTC, strategy_id="s",
                  status=S.PARTIALLY_FILLED, filled_qty=5.0, avg_fill_price=101.3, parent_id="P1", reduce_only=True,
                  created_at=datetime(2024, 1, 2, 9, 30), tags={"arrival_price": np.float64(100.5), "oco": "G1"})
        s = SqliteStore(self.path)
        s.save_order(o)
        back = s.load_orders()[0]
        for field in ("client_order_id", "symbol", "side", "quantity", "order_type", "limit_price", "tif", "strategy_id",
                      "status", "filled_qty", "avg_fill_price", "parent_id", "reduce_only", "created_at"):
            self.assertEqual(getattr(back, field), getattr(o, field), field)
        self.assertEqual(back.tags, {"arrival_price": 100.5, "oco": "G1"})
        f = Fill("f", "o1", "X", Side.SELL, 5, 101.3, datetime(2024, 1, 2), 0.5, "s", "acc", "B1")
        self.assertEqual(fill_from_dict(fill_to_dict(f)), f)

    def test_kv_and_closed_store_errors(self):
        s = SqliteStore(self.path)
        s.put("session", {"symbols": ["A"], "seed": 3})
        self.assertEqual(s.get("session"), {"symbols": ["A"], "seed": 3})
        self.assertIsNone(s.get("missing"))
        s.close()
        with self.assertRaises(StoreError):
            s.append_event("x", {})

    def test_journal_records_bus_events(self):
        bus, s = EventBus(), SqliteStore(self.path)
        Journal(s, bus)
        o = Order("o1", "X", Side.BUY, 1)
        bus.publish(ev.ORDER_UPDATED, o)
        bus.publish(ev.FILL, Fill("f1", "o1", "X", Side.BUY, 1, 9.0, None))
        bus.publish(ev.FILL, Fill("f1", "o1", "X", Side.BUY, 1, 9.0, None))            # duplicate delivery
        bus.publish(ev.KILL_SWITCH, {"state": "TRIGGERED", "reason": "r", "source": "t"})
        topics = [r[2] for r in s.events()[0]]
        self.assertEqual(topics, ["order", "fill", "kill_switch"])
        self.assertEqual(len(s.load_fills()), 1)
        self.assertEqual(s.get_kill_switch()["state"], "TRIGGERED")


def make_engine(path, broker=None, cfg=None):
    cfg = cfg or free_config()
    broker = broker or PaperBroker(cfg.initial_capital, CostModel(cfg.costs))
    store = SqliteStore(path)
    eng = TradingEngine(cfg, broker, {}, SimClock(), store=store)
    return eng, broker, store


def feed(eng, day, o=100.0, h=None, l=None, c=None):
    eng.on_bars({"X": bar(f"2024-03-{day:02d}", o=o, h=h, l=l, c=c)})


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "engine.db")

    def tearDown(self):
        self.dir.cleanup()

    def crash_scenario(self):
        e1, broker, s1 = make_engine(self.path)
        e1.start()
        feed(e1, 1)
        buy = e1.execution.submit("X", Side.BUY, 10, idempotency_key="entry-1")
        feed(e1, 2, o=100)                                                              # fills 10 @ 100
        feed(e1, 3, o=104, h=105, c=104)
        sell = e1.execution.submit("X", Side.SELL, 4)
        feed(e1, 4, o=106, c=106)                                                       # realises a profit
        resting = e1.execution.submit("X", Side.BUY, 5, OrderType.LIMIT, limit_price=98.0)
        return e1, broker, s1, buy, sell, resting

    def test_state_is_rebuilt_exactly_after_a_crash(self):
        e1, broker, s1, buy, sell, resting = self.crash_scenario()
        s1.close()                                                                      # process dies
        e2, _, s2 = make_engine(self.path, broker)                                      # restart; the broker kept its state
        rep = e2.recover()
        self.assertTrue(rep.clean, rep)
        self.assertEqual(rep.orders, 3)
        self.assertEqual(rep.fills, 2)
        self.assertEqual(rep.positions, {"X": 6})
        self.assertAlmostEqual(e2.portfolio.cash, e1.portfolio.cash)
        self.assertAlmostEqual(e2.portfolio.realized_pnl, e1.portfolio.realized_pnl)
        self.assertEqual(e2.portfolio.position("X").avg_price, e1.portfolio.position("X").avg_price)
        self.assertEqual(len(e2.portfolio.trades), len(e1.portfolio.trades))
        ro = e2.oms.get(resting.client_order_id)
        self.assertEqual(ro.status, S.ACKNOWLEDGED)
        self.assertEqual([o.client_order_id for o in e2.oms.open_orders()], [resting.client_order_id])
        self.assertFalse(e2.risk.kill_switch.is_triggered)
        # life goes on: the recovered engine can cancel the resting order at the real broker
        e2.start()
        self.assertTrue(e2.execution.cancel(ro))
        self.assertEqual(ro.status, S.CANCELLED)
        self.assertEqual(broker.open_order_ids(), set())

    def test_idempotency_keys_and_fill_ids_survive_restart(self):
        e1, broker, s1, buy, *_ = self.crash_scenario()
        s1.close()
        e2, _, _ = make_engine(self.path, broker)
        e2.recover()
        again = e2.oms.create("X", Side.BUY, 10, idempotency_key="entry-1")
        self.assertEqual(again.client_order_id, buy.client_order_id)                    # no duplicate order after restart
        old_fill = e2.oms.get(buy.client_order_id).fills[0]
        e2.oms.on_broker_event(BrokerEvent("fill", buy.client_order_id, fill=old_fill)) # broker replays an old report
        self.assertEqual(e2.portfolio.quantity("X"), 6)
        self.assertEqual(e2.oms.anomalies[-1]["type"], "duplicate_fill")

    def test_kill_switch_stays_tripped_without_rerunning_emergency_actions(self):
        e1, broker, s1, *_ = self.crash_scenario()
        e1.risk.kill_switch.trip("loss limit", source="monitor:daily_loss")
        n_orders = len(e1.oms.all_orders())
        s1.close()
        e2, _, _ = make_engine(self.path, broker)
        rep = e2.recover()
        self.assertEqual(rep.kill_switch, "TRIGGERED")
        self.assertTrue(e2.risk.kill_switch.is_triggered)
        self.assertEqual(e2.risk.kill_switch.reason, "loss limit")
        self.assertEqual(len(e2.oms.all_orders()), n_orders)                            # no second flatten was fired
        blocked = e2.execution.submit("X", Side.BUY, 1)
        self.assertEqual(blocked.status, S.REJECTED)

    def test_position_break_with_the_broker_halts_trading(self):
        e1, broker, s1, *_ = self.crash_scenario()
        s1.close()
        lost_broker = PaperBroker(100_000, CostModel(free_config().costs))             # broker state does not match our records
        e2, _, _ = make_engine(self.path, lost_broker)
        rep = e2.recover()
        self.assertFalse(rep.clean)
        self.assertTrue(rep.halted)
        self.assertIn("X", rep.reconcile["positions"])
        self.assertTrue(e2.risk.kill_switch.is_triggered)
        self.assertEqual(e2.risk.kill_switch.source, "recovery")
        self.assertEqual(e2.execution.submit("X", Side.BUY, 1).status, S.REJECTED)
        e3, *_ = make_engine(os.path.join(self.dir.name, "x.db"))
        self.assertIsNotNone(e3)

    def test_halt_can_be_declined_but_the_break_is_still_reported(self):
        e1, broker, s1, *_ = self.crash_scenario()
        s1.close()
        e2, _, _ = make_engine(self.path, PaperBroker(100_000, CostModel(free_config().costs)))
        rep = e2.recover(halt_on_break=False)
        self.assertFalse(rep.clean)
        self.assertFalse(rep.halted)
        self.assertFalse(e2.risk.kill_switch.is_triggered)

    def test_never_sent_and_unresolved_orders_are_classified(self):
        e1, broker, s1 = make_engine(self.path)
        e1.start()
        feed(e1, 1)
        never = e1.oms.create("X", Side.BUY, 1)                                         # created, never sent
        ambiguous = e1.oms.create("X", Side.BUY, 2)
        e1.oms.mark_sent(ambiguous)                                                     # sent, crash before any ack
        s1.close()
        e2, _, _ = make_engine(self.path, broker)
        rep = e2.recover()
        self.assertEqual(rep.never_sent, [never.client_order_id])
        self.assertEqual(e2.oms.get(never.client_order_id).status, S.CANCELLED)
        self.assertEqual(rep.unresolved, [ambiguous.client_order_id])                   # may or may not have reached the broker
        self.assertTrue(rep.halted)

    def test_pending_order_known_to_the_broker_is_resolved_not_flagged(self):
        e1, broker, s1 = make_engine(self.path)
        e1.start()
        feed(e1, 1)
        o = e1.oms.create("X", Side.BUY, 3, OrderType.LIMIT, limit_price=95.0)
        e1.oms.mark_sent(o)
        broker.submit_order(o)                                                          # the broker got it; the ack never reached us
        s1.close()
        e2, _, _ = make_engine(self.path, broker)
        rep = e2.recover()
        self.assertEqual(rep.unresolved, [])
        self.assertEqual(e2.oms.get(o.client_order_id).status, S.ACKNOWLEDGED)
        self.assertTrue(rep.clean, rep)

    def test_strategy_run_recovers_to_identical_state(self):
        md = market_from_closes(X=100 * np.exp(np.cumsum(np.random.default_rng(5).normal(0.001, 0.012, 200))))
        cfg = free_config()
        broker = PaperBroker(cfg.initial_capital, CostModel(cfg.costs))
        store = SqliteStore(self.path)
        e1 = TradingEngine(cfg, broker, dict(md.instruments), SimClock(), store=store)
        e1.add_strategy(SMACrossover(["X"], fast=5, slow=20))
        e1.start()
        for bars in md.bar_batches():
            e1.on_bars(bars)
        store.close()
        store2 = SqliteStore(self.path)
        e2 = TradingEngine(cfg, broker, dict(md.instruments), SimClock(), store=store2)
        rep = e2.recover()
        self.assertTrue(rep.clean, rep)
        self.assertEqual(len(e2.oms.all_orders()), len(e1.oms.all_orders()))
        self.assertAlmostEqual(e2.portfolio.cash, e1.portfolio.cash, places=6)
        self.assertAlmostEqual(e2.portfolio.realized_pnl, e1.portfolio.realized_pnl, places=6)
        self.assertEqual({s: p.quantity for s, p in e2.portfolio.open_positions().items()},
                         {s: p.quantity for s, p in e1.portfolio.open_positions().items()})
        self.assertEqual(store2.sequence_gaps(), [])

    def test_journal_failure_trips_the_kill_switch(self):
        e1, broker, s1 = make_engine(self.path)
        e1.start()
        feed(e1, 1)
        s1.close()                                                                      # disk gone / db unusable
        o = e1.execution.submit("X", Side.BUY, 1)
        self.assertTrue(e1.risk.kill_switch.is_triggered)
        self.assertEqual(e1.risk.kill_switch.source, "journal")
        self.assertGreater(e1.journal.errors, 0)

    def test_recover_requires_a_store(self):
        cfg = free_config()
        eng = TradingEngine(cfg, PaperBroker(1e5), {}, SimClock())
        with self.assertRaises(ValidationError):
            eng.recover()


if __name__ == "__main__":
    unittest.main()
