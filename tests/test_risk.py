import unittest
from datetime import datetime, timedelta

import numpy as np

import helpers
from algoplatform.core import EventBus, KillSwitchState, Order, Role, Side, OrderType, RiskAction
from algoplatform.core.config import RiskConfig
from algoplatform.core.exceptions import (AuthorizationError, KillSwitchActive, RiskError, RiskLimitBreach)
from algoplatform.risk import (KillSwitch, RiskContext, RiskManager, cornish_fisher_var, expected_shortfall,
                               historical_var, kupiec_pof, monte_carlo_var, parametric_var, portfolio_var,
                               reverse_stress_loss, stress_test)
from algoplatform.utils import SimClock

T0 = datetime(2024, 1, 2, 12, 0)


def ctx(equity=100_000, cash=100_000, positions=None, prices=None, pending=None, now=T0, **kw):
    return RiskContext(equity, cash, positions or {}, prices or {"X": 100.0}, now, pending or {}, **kw)


def order(qty=10, side=Side.BUY, **kw):
    return Order("o1", kw.pop("symbol", "X"), side, qty, strategy_id=kw.pop("strategy_id", "s"), **kw)


class PreTradeTests(unittest.TestCase):
    def mgr(self, **cfg):
        return RiskManager(RiskConfig(**cfg), clock=SimClock())

    def test_clean_order_passes(self):
        d = self.mgr().evaluate(order(), ctx())
        self.assertEqual(d.action, RiskAction.ALLOW)

    def test_quantity_and_notional_limits(self):
        m = self.mgr(max_order_quantity=50, max_order_notional=2000)
        self.assertIn("max_order_quantity", m.evaluate(order(51), ctx()).reasons[0])
        self.assertIn("max_order_notional", m.evaluate(order(30), ctx()).reasons[0])

    def test_fat_finger_price_collar(self):
        m = self.mgr()
        o = order(1, order_type=OrderType.LIMIT, limit_price=150.0)            # 50% away from last
        self.assertIn("price_collar", m.evaluate(o, ctx()).reasons[0])
        o2 = order(1, order_type=OrderType.LIMIT, limit_price=101.0)
        self.assertTrue(m.evaluate(o2, ctx()).allowed)

    def test_position_limit_includes_pending(self):
        m = self.mgr(max_position_notional=5000)
        d = m.evaluate(order(30), ctx(positions={"X": 10}, pending={"X": 15}))   # 55 * 100 > 5000
        self.assertFalse(d.allowed)

    def test_exposure_caps_but_reducing_orders_pass(self):
        m = self.mgr(max_gross_exposure_pct=1.0, max_net_exposure_pct=1.0, max_order_notional=1e9,
                     max_position_notional=1e9)
        c = ctx(positions={"X": 1000})                                           # already 100% gross
        self.assertFalse(m.evaluate(order(10), c).allowed)
        self.assertTrue(m.evaluate(order(10, Side.SELL), c).allowed)             # reduces risk

    def test_short_sell_and_reduce_only(self):
        m = self.mgr(allow_short=False)
        self.assertFalse(m.evaluate(order(5, Side.SELL), ctx()).allowed)
        m2 = self.mgr()
        self.assertFalse(m2.evaluate(order(5, reduce_only=True), ctx(positions={"X": 10})).allowed)
        self.assertTrue(m2.evaluate(order(5, Side.SELL, reduce_only=True), ctx(positions={"X": 10})).allowed)
        self.assertFalse(m2.evaluate(order(20, Side.SELL, reduce_only=True), ctx(positions={"X": 10})).allowed)  # flips

    def test_restricted_and_stale(self):
        m = self.mgr(restricted_symbols=("X",))
        self.assertIn("restricted", m.evaluate(order(), ctx()).reasons[0])
        m2 = self.mgr(max_data_staleness_sec=10)
        d = m2.evaluate(order(), ctx(price_times={"X": T0 - timedelta(seconds=60)}))
        self.assertIn("stale_data", d.reasons[0])

    def test_rate_limit(self):
        m = self.mgr(max_orders_per_minute=3)
        results = [m.evaluate(order(1, strategy_id=f"s{i}"), ctx()).allowed for i in range(5)]
        self.assertEqual(results, [True, True, True, False, False])

    def test_runaway_duplicate_detection(self):
        m = self.mgr()
        res = [m.evaluate(order(7), ctx()).allowed for _ in range(5)]
        self.assertEqual(res, [True, True, True, False, False])

    def test_broken_check_fails_closed(self):
        from algoplatform.risk import Check

        class Boom(Check):
            name = "boom"

            def check(self, order, ctx):
                raise RuntimeError("bug")
        m = RiskManager(RiskConfig(), checks=[Boom()])
        d = m.evaluate(order(), ctx())
        self.assertFalse(d.allowed)
        self.assertIn("errored", d.reasons[0])

    def test_validate_raises_typed_errors(self):
        m = self.mgr(max_order_quantity=1)
        with self.assertRaises(RiskLimitBreach):
            m.validate(order(5), ctx())
        m.kill_switch.trip("test")
        with self.assertRaises(KillSwitchActive):
            m.validate(order(1), ctx())


class KillSwitchTests(unittest.TestCase):
    def setUp(self):
        self.um = helpers.fast_user_manager()
        self.admin = self.um.register("root_admin", helpers.STRONG_PW, Role.ADMIN)
        self.rm1 = self.um.register("rm1", helpers.STRONG_PW, Role.RISK_MANAGER, actor=self.admin)
        self.rm2 = self.um.register("rm2", helpers.STRONG_PW, Role.RISK_MANAGER, actor=self.admin)
        self.trader = self.um.register("trader", helpers.STRONG_PW)

    def test_trip_is_idempotent_and_runs_callbacks_even_if_one_fails(self):
        ks, calls = KillSwitch(), []
        ks.on_trip(lambda r: (_ for _ in ()).throw(RuntimeError("cb fails")))
        ks.on_trip(lambda r: calls.append(r))
        self.assertTrue(ks.trip("loss"))
        self.assertFalse(ks.trip("again"))
        self.assertEqual(calls, ["loss"])
        self.assertEqual(ks.state, KillSwitchState.TRIGGERED)
        with self.assertRaises(KillSwitchActive):
            ks.check()

    def test_reset_requires_permission_and_reason(self):
        ks = KillSwitch()
        ks.trip("x")
        with self.assertRaises(AuthorizationError):
            ks.reset(self.trader, "all good now")
        with self.assertRaises(RiskError):
            ks.reset(self.rm1, "ok")
        ks.reset(self.rm1, "reviewed and cleared")
        self.assertEqual(ks.state, KillSwitchState.ARMED)

    def test_two_person_reset(self):
        ks = KillSwitch(two_person_reset=True)
        ks.trip("x")
        with self.assertRaises(AuthorizationError):
            ks.reset(self.rm1, "reviewed and cleared")
        with self.assertRaises(AuthorizationError):
            ks.reset(self.rm1, "reviewed and cleared", approver=self.rm1)       # self-approval
        with self.assertRaises(AuthorizationError):
            ks.reset(self.rm1, "reviewed and cleared", approver=self.trader)    # approver lacks permission
        ks.reset(self.rm1, "reviewed and cleared", approver=self.rm2)
        self.assertFalse(ks.is_triggered)

    def test_monitor_trips_on_daily_loss_and_drawdown(self):
        m = RiskManager(RiskConfig(max_daily_loss_pct=0.05, max_drawdown_pct=0.20), clock=SimClock())
        m.monitor.start_day(100_000)
        m.monitor.update_equity(97_000)
        self.assertFalse(m.kill_switch.is_triggered)
        m.monitor.update_equity(94_000)
        self.assertTrue(m.kill_switch.is_triggered)
        self.assertIn("daily loss", m.kill_switch.reason)

    def test_monitor_trips_on_consecutive_errors_and_resets_on_success(self):
        m = RiskManager(RiskConfig(max_consecutive_errors=3))
        m.monitor.record_error()
        m.monitor.record_error()
        m.monitor.record_success()
        m.monitor.record_error()
        self.assertFalse(m.kill_switch.is_triggered)
        m.monitor.record_error()
        m.monitor.record_error()
        self.assertTrue(m.kill_switch.is_triggered)

    def test_halted_blocks_everything_except_flatten(self):
        m = RiskManager(RiskConfig())
        m.kill_switch.trip("halt")
        self.assertFalse(m.evaluate(order(1), ctx()).allowed)
        self.assertTrue(m.evaluate(order(1, Side.SELL, tags={"flatten": True}), ctx(positions={"X": 1})).allowed)

    def test_warning_published_at_80_percent(self):
        bus, seen = EventBus(), []
        bus.subscribe("risk.breach", lambda t, p: seen.append(p))
        m = RiskManager(RiskConfig(max_daily_loss_pct=0.05), bus=bus)
        m.monitor.start_day(100_000)
        m.monitor.update_equity(95_900)            # -4.1% = 82% of the limit
        self.assertEqual(seen[0]["level"], "warning")
        self.assertFalse(m.kill_switch.is_triggered)

    def test_exposure_scalar_shrinks_with_drawdown(self):
        m = RiskManager(RiskConfig(max_drawdown_pct=0.20))
        m.monitor.start_day(100_000)
        m.monitor.update_equity(100_000)
        self.assertEqual(m.exposure_scalar(), 1.0)
        m.monitor.update_equity(90_000)
        self.assertLess(m.exposure_scalar(), 1.0)
        self.assertGreater(m.exposure_scalar(), 0.0)


class MeasureTests(unittest.TestCase):
    def setUp(self):
        self.r = np.random.default_rng(1).normal(0.0005, 0.01, 5000)

    def test_var_orderings_and_values(self):
        self.assertAlmostEqual(historical_var(self.r, 0.99), parametric_var(self.r, 0.99), delta=0.003)
        self.assertGreater(expected_shortfall(self.r, 0.99), historical_var(self.r, 0.99))
        self.assertAlmostEqual(parametric_var(self.r, 0.95), 1.645 * 0.01 - 0.0005, delta=0.0015)
        self.assertAlmostEqual(monte_carlo_var(0.0, 0.01, 0.99), 0.0233, delta=0.001)
        self.assertGreater(cornish_fisher_var(self.r, 0.99), 0)

    def test_fat_tails_raise_cornish_fisher_var(self):
        fat = np.random.default_rng(2).standard_t(3, 20_000) * 0.01
        self.assertGreater(cornish_fisher_var(fat, 0.99), parametric_var(fat, 0.99) * 0.95)

    def test_portfolio_var_diversification(self):
        cov = np.array([[0.0004, 0.0], [0.0, 0.0004]])
        w = np.array([0.5, 0.5])
        self.assertAlmostEqual(portfolio_var(w, cov, 0.99), 2.326 * np.sqrt(0.0002), places=3)

    def test_kupiec(self):
        ok = kupiec_pof(10, 1000, 0.99)
        self.assertGreater(ok["p_value"], 0.9)
        bad = kupiec_pof(40, 1000, 0.99)
        self.assertLess(bad["p_value"], 0.001)

    def test_stress_and_reverse_stress(self):
        out = stress_test({"A": 1000, "B": -500}, {"A": -0.2, "*": 0.1})
        self.assertAlmostEqual(out["A"], -200)
        self.assertAlmostEqual(out["B"], -50)
        self.assertAlmostEqual(out["total"], -250)
        self.assertAlmostEqual(reverse_stress_loss(100_000, 200_000, 0.1), 0.05)


if __name__ == "__main__":
    unittest.main()
