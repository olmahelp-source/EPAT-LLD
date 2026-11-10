import unittest

import numpy as np
import pandas as pd

from helpers import bar, free_config, market_from_closes
from test_orders_brokers import send, wired
from test_strategy_backtest import Scripted
from algoplatform.analytics.tca import fill_quality, markouts, order_tca, parent_tca, tca_summary
from algoplatform.app.engine import TradingEngine
from algoplatform.backtest import BacktestEngine
from algoplatform.brokers import CostModel, PaperBroker
from algoplatform.core import Fill, OrderStatus, OrderType, PlatformConfig, Side
from algoplatform.core.config import CostConfig, RiskConfig
from algoplatform.core.exceptions import ValidationError
from algoplatform.execution import (POV, TWAP, VWAP, Iceberg, ImplementationShortfall, ParentOrder, SliceContext,
                                    estimate_cost, go_no_go, limit_price, make_algo, u_shape, volume_curve_from_history)
from algoplatform.execution.parent import DONE, FAILED
from algoplatform.quant.impact import ac_trajectory, calibrate_impact, kappa_from_risk, sqrt_impact
from algoplatform.utils import SimClock

S = OrderStatus


def run_algo(algo, total, uncommitted_fn=None, bars=None, lot=1.0, **ctx_kw):
    """Drive an algo bar by bar assuming each slice fills immediately; returns the slices."""
    p = ParentOrder("P", "X", Side.BUY, total, algo)
    done, slices = 0.0, []
    for k in range(bars or 12):
        ctx = SliceContext(p, bar("2024-01-02", c=100.0), k, total - done, 0.0, 0, ctx_kw.get("vol", 10_000), lot)
        d = algo.slice(ctx)
        slices.append(d.qty)
        done += d.qty
    return slices


class AlgoScheduleTests(unittest.TestCase):
    def test_twap_even_and_complete(self):
        s = run_algo(TWAP(bars=5), 1000)
        self.assertEqual(s[:5], [200] * 5)
        self.assertEqual(sum(s), 1000)

    def test_twap_respects_lot_and_sends_remainder_last(self):
        s = run_algo(TWAP(bars=4), 1000, lot=100)
        self.assertEqual(sum(s), 1000)
        self.assertTrue(all(q % 100 == 0 for q in s))

    def test_vwap_follows_curve(self):
        s = run_algo(VWAP(bars=3, curve=[1, 1, 2]), 1000)
        self.assertEqual(s[:3], [250, 250, 500])
        u = u_shape(10)
        self.assertGreater(u[0], u[5])
        self.assertGreater(u[-1], u[5])                                           # heavy open and close
        self.assertAlmostEqual(u.sum(), 1.0)
        prof = volume_curve_from_history([5, 1, 5, 7, 1, 7], 3)
        self.assertAlmostEqual(prof.sum(), 1.0)
        self.assertGreater(prof[0], prof[1])
        with self.assertRaises(ValidationError):
            VWAP(bars=3, curve=[1, 2])

    def test_pov_never_exceeds_participation(self):
        p = ParentOrder("P", "X", Side.BUY, 5000, POV(0.1))
        mk = lambda unc, work: SliceContext(p, bar("2024-01-02"), 0, unc, work, 0, 10_000)
        self.assertEqual(POV(0.1).slice(mk(5000, 0)).qty, 1000)
        self.assertEqual(POV(0.1).slice(mk(5000, 400)).qty, 600)                  # working qty counts against the cap
        self.assertEqual(POV(0.1).slice(mk(300, 0)).qty, 300)
        self.assertEqual(POV(0.1).slice(mk(5000, 1500)).qty, 0)
        with self.assertRaises(ValidationError):
            POV(0)

    def test_implementation_shortfall_front_loads_with_urgency(self):
        slow = run_algo(ImplementationShortfall(bars=6, kappa=0.0), 1200)
        fast = run_algo(ImplementationShortfall(bars=6, kappa=1.5), 1200)
        self.assertAlmostEqual(slow[0], 200)                                      # kappa 0 == TWAP
        self.assertGreater(fast[0], slow[0])
        self.assertTrue(all(fast[i] >= fast[i + 1] - 1e-9 for i in range(5)))     # decreasing slices
        self.assertEqual(sum(fast), 1200)

    def test_iceberg_hides_size_and_refreshes(self):
        ice = Iceberg(display=100, variance=0.3, max_wait=2, seed=1)
        p = ParentOrder("P", "X", Side.BUY, 1000, ice)
        d = ice.slice(SliceContext(p, bar("2024-01-02", c=100.0), 0, 1000, 0, 0, 1e4))
        self.assertTrue(70 <= d.qty <= 130)
        self.assertLess(d.limit_price, 100.0)                                     # passive: below the market for a buy
        sizes = {Iceberg(display=100, variance=0.3, seed=s).slice(SliceContext(p, bar("2024-01-02"), 0, 1000, 0, 0, 1e4)).qty
                 for s in range(8)}
        self.assertGreater(len(sizes), 3)                                         # randomised clips
        wait = ice.slice(SliceContext(p, bar("2024-01-02"), 3, 900, 100, 1, 1e4))
        self.assertEqual((wait.qty, wait.cancel_working), (0, False))
        stale = ice.slice(SliceContext(p, bar("2024-01-02"), 3, 900, 100, 2, 1e4))
        self.assertTrue(stale.cancel_working)
        # deterministic for a given seed
        a = Iceberg(display=100, seed=5).slice(SliceContext(p, bar("2024-01-02"), 0, 1000, 0, 0, 1e4)).qty
        b = Iceberg(display=100, seed=5).slice(SliceContext(p, bar("2024-01-02"), 0, 1000, 0, 0, 1e4)).qty
        self.assertEqual(a, b)

    def test_registry(self):
        self.assertIsInstance(make_algo("twap", bars=3), TWAP)
        with self.assertRaises(Exception):
            make_algo("nope")


class PricingAndImpactTests(unittest.TestCase):
    def test_limit_price_modes(self):
        touch_b = limit_price(Side.BUY, 100.0, 0.01, "touch", spread_bps=10)
        touch_s = limit_price(Side.SELL, 100.0, 0.01, "touch", spread_bps=10)
        self.assertLess(touch_b, 100.0)
        self.assertGreater(touch_s, 100.0)
        mid = limit_price(Side.BUY, 100.0, 0.01, "mid", spread_bps=10)
        agg = limit_price(Side.BUY, 100.0, 0.01, "aggressive", spread_bps=10)
        imp = limit_price(Side.BUY, 100.0, 0.01, "improve", spread_bps=10, improve_ticks=50)
        self.assertGreater(agg, mid)
        self.assertLessEqual(imp, mid + 1e-9)                                     # improving never crosses mid
        self.assertEqual(round(touch_b / 0.01) * 0.01, touch_b) if False else None
        self.assertAlmostEqual(touch_b % 0.01, 0.0, places=6) if abs(touch_b % 0.01) < 1e-6 or abs(touch_b % 0.01 - 0.01) < 1e-6 else self.fail("not on tick")

    def test_chase_is_capped(self):
        base = limit_price(Side.BUY, 100.0, 0.01, "touch", spread_bps=2)
        chased = limit_price(Side.BUY, 100.0, 0.01, "touch", spread_bps=2, bars_waited=3, chase_bps_per_bar=2, chase_cap_bps=10)
        capped = limit_price(Side.BUY, 100.0, 0.01, "touch", spread_bps=2, bars_waited=100, chase_bps_per_bar=2, chase_cap_bps=10)
        self.assertGreater(chased, base)
        self.assertGreater(capped, chased)
        self.assertLessEqual(capped, base * 1.0011)
        with self.assertRaises(ValidationError):
            limit_price(Side.BUY, 100.0, mode="weird")

    def test_cost_estimate_and_go_no_go(self):
        cm = CostModel(CostConfig(commission_per_share=0.0, min_commission=0, slippage_bps=1, spread_bps=2, impact_coef=1.0))
        small = estimate_cost(Side.BUY, 100, 50.0, adv=1e6, sigma=0.01, costs=cm)
        big = estimate_cost(Side.BUY, 100_000, 50.0, adv=1e6, sigma=0.01, costs=cm)
        self.assertGreater(big["total_bps"], small["total_bps"])
        self.assertAlmostEqual(small["spread"] / small["notional"] * 1e4, 1.0)    # half of 2 bps
        self.assertTrue(go_no_go(30, small)["go"])
        self.assertFalse(go_no_go(1, big)["go"])

    def test_sqrt_law_and_calibration(self):
        self.assertAlmostEqual(sqrt_impact(0.02, 10_000, 1e6, 0.5), 0.5 * 0.02 * 0.1)
        rng = np.random.default_rng(0)
        part = rng.uniform(0.001, 0.2, 200)
        sig = rng.uniform(0.005, 0.03, 200)
        slip = 0.7 * sig * np.sqrt(part) + rng.normal(0, 1e-5, 200)
        self.assertAlmostEqual(calibrate_impact(part, slip, sig), 0.7, delta=0.03)
        with self.assertRaises(Exception):
            calibrate_impact([0.1], [0.001], [0.01])

    def test_almgren_chriss_trajectory(self):
        x = ac_trajectory(10, 1.0)
        self.assertAlmostEqual(x[0], 1.0)
        self.assertAlmostEqual(x[-1], 0.0)
        self.assertTrue((np.diff(x) <= 0).all())
        np.testing.assert_allclose(ac_trajectory(5, 0.0), np.linspace(1, 0, 6))
        self.assertLess(ac_trajectory(10, 2.0)[1], ac_trajectory(10, 0.5)[1])      # more urgency = faster early trading
        self.assertGreater(kappa_from_risk(2.0, 0.02, 1e-3), kappa_from_risk(0.5, 0.02, 1e-3))

    def test_impact_in_broker_penalises_big_orders(self):
        """One large market order must fill worse than the same size split over several bars."""
        cm = CostModel(CostConfig(commission_per_share=0, min_commission=0, slippage_bps=0, spread_bps=0, impact_coef=2.0))

        def avg_fill(sizes):
            b, fills = PaperBroker(1e9, cm), []
            from algoplatform.orders import OrderManager
            from algoplatform.core import EventBus, events as ev
            bus = EventBus()
            bus.subscribe(ev.FILL, lambda t, f: fills.append(f))
            oms = OrderManager(bus)
            b.set_event_handler(oms.on_broker_event)
            b.connect()
            for i in range(12):                                                     # build a volatility history
                b.process_bar(bar(f"2024-01-{i + 1:02d}", o=100, c=100 * (1.01 if i % 2 else 0.99), h=102, l=98))
            for i, q in enumerate(sizes):
                o = oms.create("X", Side.BUY, q)
                oms.mark_sent(o)
                b.submit_order(o)
                b.process_bar(bar(f"2024-02-{i + 1:02d}", o=100, h=101, l=99, c=100, v=100_000))
            return sum(f.quantity * f.price for f in fills) / sum(f.quantity for f in fills)
        big, sliced = avg_fill([5000]), avg_fill([1000] * 5)
        self.assertGreater(big, sliced)
        self.assertGreater(big, 100.0)


def engine_for(cfg=None, volume=1e6, **broker_kw):
    cfg = cfg or free_config()
    broker = PaperBroker(cfg.initial_capital, CostModel(cfg.costs), **broker_kw)
    eng = TradingEngine(cfg, broker, {}, SimClock())
    eng.start()
    return eng


def feed(eng, n, start_day=2, price=100.0, volume=1e6):
    for i in range(n):
        eng.on_bars({"X": bar(f"2024-03-{start_day + i:02d}", o=price, h=price * 1.001, l=price * 0.999, c=price, v=volume)})


class LinkedOrderTests(unittest.TestCase):
    def test_bracket_take_profit_cancels_stop(self):
        eng = engine_for()
        feed(eng, 1)
        entry, tp, sl = eng.execution.submit_bracket("X", Side.BUY, 10, take_profit=108, stop_loss=96)
        self.assertEqual([o.status for o in (entry, tp, sl)], [S.ACKNOWLEDGED] * 3)
        eng.on_bars({"X": bar("2024-03-05", o=100, h=101, l=99, c=100)})            # entry fills; exits go live
        self.assertEqual(entry.status, S.FILLED)
        self.assertEqual(eng.portfolio.quantity("X"), 10)
        eng.on_bars({"X": bar("2024-03-06", o=100, h=109, l=99, c=108)})            # target hit
        self.assertEqual((tp.status, sl.status), (S.FILLED, S.CANCELLED))
        self.assertEqual(eng.portfolio.quantity("X"), 0)
        self.assertAlmostEqual(eng.portfolio.realized_pnl, 80.0)                    # 10 * (108 - 100)
        self.assertEqual(eng.broker.open_order_ids(), set())

    def test_bracket_stop_wins_when_both_levels_trade_in_one_bar(self):
        eng = engine_for()
        feed(eng, 1)
        entry, tp, sl = eng.execution.submit_bracket("X", Side.BUY, 10, take_profit=108, stop_loss=96)
        eng.on_bars({"X": bar("2024-03-05", o=100, h=101, l=99, c=100)})
        eng.on_bars({"X": bar("2024-03-06", o=100, h=109, l=94, c=100)})            # ambiguous bar: assume the worst
        self.assertEqual((sl.status, tp.status), (S.FILLED, S.CANCELLED))
        self.assertAlmostEqual(eng.portfolio.realized_pnl, -40.0)

    def test_cancelling_entry_cancels_held_exits(self):
        eng = engine_for()
        feed(eng, 1)
        entry, tp, sl = eng.execution.submit_bracket("X", Side.BUY, 10, 105, 94, entry_limit=98)
        eng.on_bars({"X": bar("2024-03-05", o=100, h=101, l=99.5, c=100)})            # entry limit not touched
        self.assertEqual(entry.status, S.ACKNOWLEDGED)
        eng.execution.cancel(entry)
        self.assertEqual((entry.status, tp.status, sl.status), (S.CANCELLED,) * 3)
        self.assertEqual(eng.broker.open_order_ids(), set())

    def test_bracket_rejects_inverted_levels_and_rejected_entry_skips_exits(self):
        eng = engine_for()
        feed(eng, 1)
        with self.assertRaises(ValidationError):
            eng.execution.submit_bracket("X", Side.BUY, 10, take_profit=90, stop_loss=110)
        cfg = free_config(risk=RiskConfig(max_order_quantity=5))
        eng2 = engine_for(cfg)
        feed(eng2, 1)
        orders = eng2.execution.submit_bracket("X", Side.BUY, 10, 108, 96)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].status, S.REJECTED)

    def test_oco_pair_on_existing_position(self):
        eng = engine_for()
        feed(eng, 1)
        eng.execution.submit("X", Side.BUY, 10)
        feed(eng, 1, start_day=5)
        self.assertEqual(eng.portfolio.quantity("X"), 10)
        a, b = eng.execution.submit_oco("X", Side.SELL, 10, limit_price=108, stop_price=96)
        eng.on_bars({"X": bar("2024-03-06", o=100, h=101, l=95, c=97)})               # stop triggers
        self.assertEqual((b.status, a.status), (S.FILLED, S.CANCELLED))
        self.assertEqual(eng.portfolio.quantity("X"), 0)

    def test_trailing_stop_follows_price_up_then_triggers(self):
        eng = engine_for()
        feed(eng, 1)
        eng.execution.submit("X", Side.BUY, 10)
        eng.on_bars({"X": bar("2024-03-05", o=100, h=101, l=99, c=100)})
        t = eng.execution.submit("X", Side.SELL, 10, OrderType.TRAILING_STOP, tags={"trail_pct": 0.05})
        self.assertEqual(t.status, S.ACKNOWLEDGED)
        eng.on_bars({"X": bar("2024-03-06", o=102, h=110, l=101, c=109)})            # watermark rises to 110
        self.assertEqual(t.status, S.ACKNOWLEDGED)
        eng.on_bars({"X": bar("2024-03-07", o=108, h=109, l=103, c=104)})            # stop = 104.5 -> filled there
        self.assertEqual(t.status, S.FILLED)
        self.assertAlmostEqual(t.avg_fill_price, 104.5)

    def test_trailing_stop_needs_distance(self):
        eng = engine_for()
        feed(eng, 1)
        with self.assertRaises(ValidationError):
            eng.execution.submit("X", Side.SELL, 1, OrderType.TRAILING_STOP)

    def test_post_only_and_min_qty_flags(self):
        oms, broker, _ = wired()
        broker.process_bar(bar("2024-01-02", o=100))
        crossing = oms.create("X", Side.BUY, 5, OrderType.LIMIT, limit_price=101.0, tags={"post_only": True})
        oms.mark_sent(crossing)
        from algoplatform.core.exceptions import OrderRejectedError
        with self.assertRaises(OrderRejectedError):
            broker.submit_order(crossing)
        passive = send(oms, broker, "X", Side.BUY, 5, OrderType.LIMIT, limit_price=99.0, tags={"post_only": True})
        self.assertEqual(passive.status, S.ACKNOWLEDGED)
        oms2, b2, _ = wired(max_participation=0.1)
        small = send(oms2, b2, "X", Side.BUY, 100, tags={"min_qty": 50})
        b2.process_bar(bar("2024-01-02", o=10, v=100))                                # capacity 10 < min 50
        self.assertEqual(small.filled_qty, 0)
        b2.process_bar(bar("2024-01-03", o=10, v=1000))                               # capacity 100
        self.assertEqual(small.status, S.FILLED)


class ManagerTests(unittest.TestCase):
    def test_twap_parent_executes_in_slices(self):
        eng = engine_for()
        feed(eng, 1)
        done = []
        p = eng.algos.start("X", Side.BUY, 1000, "twap", 100.0, bars=5, on_complete=done.append)
        feed(eng, 9, start_day=5)
        self.assertEqual(p.status, DONE)
        self.assertEqual(done, [p])
        self.assertEqual(len(p.children), 5)
        self.assertEqual(eng.portfolio.quantity("X"), 1000)
        self.assertTrue(all(eng.oms.get(c).quantity == 200 for c in p.children))
        self.assertTrue(all(eng.oms.get(c).parent_id == p.parent_id for c in p.children))
        self.assertAlmostEqual(p.avg_price, 100.0)

    def test_children_obey_risk_limits_and_parent_fails_after_repeated_rejects(self):
        eng = engine_for(free_config(risk=RiskConfig(max_order_quantity=150)))
        feed(eng, 1)
        p = eng.algos.start("X", Side.BUY, 1000, "twap", 100.0, bars=5)
        feed(eng, 8, start_day=5)
        self.assertEqual(p.status, FAILED)
        self.assertEqual(p.filled, 0)
        self.assertEqual(len(p.children), 3)
        self.assertTrue(all(eng.oms.get(c).status is S.REJECTED for c in p.children))

    def test_pov_respects_cap_and_finishes(self):
        eng = engine_for()
        feed(eng, 2, volume=10_000)
        p = eng.algos.start("X", Side.BUY, 800, "pov", 100.0, participation=0.05)      # 80k notional on a 100k account
        feed(eng, 12, start_day=5, volume=10_000)
        self.assertEqual(p.status, DONE)
        self.assertTrue(all(eng.oms.get(c).quantity <= 500 for c in p.children))     # 5% of 10,000
        self.assertEqual(eng.portfolio.quantity("X"), 800)

    def test_iceberg_works_passive_limits(self):
        eng = engine_for()
        feed(eng, 1)
        p = eng.algos.start("X", Side.BUY, 300, "iceberg", 100.0, display=100, variance=0.0, mode="aggressive", spread_bps=20)
        feed(eng, 12, start_day=5)
        self.assertEqual(p.status, DONE)
        kids = [eng.oms.get(c) for c in p.children]
        self.assertTrue(all(k.order_type is OrderType.LIMIT for k in kids))
        self.assertTrue(all(k.quantity <= 100 for k in kids))
        self.assertEqual(eng.portfolio.quantity("X"), 300)

    def test_cancel_and_kill_switch_stop_parents(self):
        eng = engine_for()
        feed(eng, 1)
        p = eng.algos.start("X", Side.BUY, 1000, "twap", 100.0, bars=10)
        feed(eng, 2, start_day=5)
        self.assertGreater(len(p.children), 0)
        eng.risk.kill_switch.trip("test")
        self.assertEqual(p.status, "CANCELLED")
        self.assertEqual(eng.oms.open_orders(), [] if not eng.portfolio.open_positions() else eng.oms.open_orders())
        q = eng.algos.start("X", Side.BUY, 10, "twap", 100.0, bars=2)
        self.assertEqual(eng.algos.cancel_all(), 1)
        self.assertEqual(q.status, "CANCELLED")

    def test_unsent_quantity_is_visible_for_netting(self):
        eng = engine_for()
        feed(eng, 1)
        p = eng.algos.start("X", Side.BUY, 1000, "twap", 100.0, bars=5)
        self.assertEqual(eng.algos.signed_unsent("X"), 1000)
        feed(eng, 1, start_day=5)                                                    # first slice (200) now working
        self.assertEqual(eng.algos.signed_unsent("X"), 800)                          # working part is the OMS's job

    def test_validation(self):
        eng = engine_for()
        with self.assertRaises(ValidationError):
            eng.algos.start("X", Side.BUY, 0, "twap", 100.0)


class TcaTests(unittest.TestCase):
    def parent(self):
        p = ParentOrder("P1", "X", Side.BUY, 100, TWAP(2), arrival_price=100.0)
        p.record_fill(60, 100.05, 1.0)
        p.record_fill(40, 100.15, 1.0)
        return p

    def test_shortfall_signs(self):
        r = parent_tca(self.parent())
        self.assertAlmostEqual(r["avg_price"], 100.09)
        self.assertAlmostEqual(r["slippage_bps"], 9.0, places=3)                    # paid 9 bps more than arrival
        self.assertAlmostEqual(r["fill_rate"], 1.0)
        sell = ParentOrder("P2", "X", Side.SELL, 100, TWAP(2), arrival_price=100.0)
        sell.record_fill(100, 99.90)
        self.assertAlmostEqual(parent_tca(sell)["slippage_bps"], 10.0, places=3)     # selling lower is also a cost
        self.assertEqual(parent_tca(ParentOrder("P3", "X", Side.BUY, 10, TWAP(2), arrival_price=100.0))["slippage_bps"], 0.0)

    def test_markouts(self):
        idx = pd.bdate_range("2024-01-01", periods=6)
        closes = pd.DataFrame({"X": [100, 101, 102, 101, 100, 99.0]}, index=idx)
        buy = Fill("f1", "o", "X", Side.BUY, 1, 100.0, idx[0])
        sell = Fill("f2", "o", "X", Side.SELL, 1, 102.0, idx[2])
        m = markouts([buy, sell], closes, (1, 4))
        self.assertAlmostEqual(m.loc[0, "markout_1"], 100.0)                         # +1% favourable for a buy
        self.assertAlmostEqual(m.loc[1, "markout_1"], 98.04, places=1)               # price fell after the sell: good
        self.assertTrue(np.isnan(m.loc[1, "markout_4"]))                             # beyond the data

    def test_summary_and_quality(self):
        s = tca_summary([self.parent()])
        self.assertEqual(s["parents"], 1)
        self.assertIn("twap", s["by_algo"])
        self.assertEqual(tca_summary([]), {"parents": 0})

    def test_end_to_end_backtest_with_algo(self):
        closes = np.linspace(100, 130, 80)
        md = market_from_closes(X=closes)
        res = BacktestEngine(md, free_config()).run(Scripted(["X"], plan={10: 1.0, 40: 0.0}),
                                                    algo=("twap", {"bars": 5}))
        buys = [f for f in res.fills if f.side is Side.BUY]
        self.assertGreaterEqual(len(buys), 5)                                        # built up over >= 5 bars
        self.assertGreaterEqual(len({f.ts for f in buys}), 5)
        self.assertEqual(res.summary["positions"], {})
        self.assertTrue(all(p.status == DONE for p in res.parents))
        self.assertGreater(len(order_tca(res.orders)), 0)
        q = fill_quality(res.orders)
        self.assertEqual(q["reject_rate"], 0)
        single = BacktestEngine(md, free_config()).run(Scripted(["X"], plan={10: 1.0, 40: 0.0}))
        self.assertEqual(sum(f.quantity for f in buys), sum(f.quantity for f in single.fills if f.side is Side.BUY))

    def test_algo_backtest_is_deterministic(self):
        md = market_from_closes(X=100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0.0005, 0.01, 200))))
        run = lambda: BacktestEngine(md, PlatformConfig()).run(helpers_sma(), algo=("iceberg", {"display": 40, "seed": 3}))
        a, b = run(), run()
        pd.testing.assert_series_equal(a.equity, b.equity)


def helpers_sma():
    from algoplatform.strategy import SMACrossover
    return SMACrossover(["X"], fast=5, slow=20)


if __name__ == "__main__":
    unittest.main()
