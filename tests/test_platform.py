import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np

import helpers
from helpers import bar, free_config, market_from_closes
from algoplatform.app import TradingPlatform
from algoplatform.app.cli import main as cli_main
from algoplatform.core import OrderStatus, OrderType, Role, Side
from algoplatform.core.exceptions import AuthorizationError, ConfigError
from algoplatform.optimize import Param, ParameterSpace
from algoplatform.strategy import StrategyBuilder
from algoplatform.users import PasswordHasher


class PlatformTests(unittest.TestCase):
    def setUp(self):
        self.p = TradingPlatform(free_config(log_level="CRITICAL"))
        self.p.users.hasher = PasswordHasher(n=2 ** 4)                           # fast hashing for tests
        u = self.p.users
        self.admin = u.register("root_admin", helpers.STRONG_PW, Role.ADMIN)
        self.trader = u.register("trader", helpers.STRONG_PW)
        self.viewer = u.register("viewer", helpers.STRONG_PW, Role.VIEWER, actor=self.admin)
        self.risk = u.register("riskman", helpers.STRONG_PW, Role.RISK_MANAGER, actor=self.admin)
        self.md = market_from_closes(X=100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.001, 0.01, 300))))

    # ---- permissions -------------------------------------------------------------------------
    def test_viewer_cannot_research_or_trade(self):
        with self.assertRaises(AuthorizationError):
            self.p.backtest(self.viewer, "sma_crossover", self.md)
        with self.assertRaises(AuthorizationError):
            self.p.create_strategy(self.viewer, "sma_crossover", ["X"])
        with self.assertRaises(AuthorizationError):
            self.p.open_session(self.viewer, "ACC-1", [])

    def test_backtest_by_name_and_by_spec_is_audited(self):
        r1 = self.p.backtest(self.trader, "sma_crossover", self.md, params={"fast": 5, "slow": 20})
        spec = (StrategyBuilder("rule").indicator("f", "sma", period=5).indicator("s", "sma", period=20)
                .long_entry(("f", ">", "s")).long_exit(("f", "<=", "s")).build())
        r2 = self.p.backtest(self.trader, spec, self.md)
        self.assertGreater(len(r1.trades), 0)
        self.assertGreater(len(r2.trades), 0)
        self.assertEqual(len(self.p.audit.entries("backtest.run")), 2)

    def test_optimize_through_facade(self):
        sp = ParameterSpace([Param("fast", "int", 3, 10), Param("slow", "int", 15, 40)])
        res = self.p.optimize(self.trader, "sma_crossover", self.md, sp, n_iter=5)
        self.assertEqual(res.n_trials, 5)
        self.assertEqual(len(self.p.audit.entries("optimize.run")), 1)

    # ---- live-style session -----------------------------------------------------------------
    def session(self):
        acc = self.p.accounts.link(self.trader, "paper", {})
        return self.p.open_session(self.trader, acc.account_id, [])

    def test_manual_order_flow_goes_through_risk_oms_and_portfolio(self):
        eng = self.session()
        eng.start()
        eng.on_bars({"X": bar("2024-01-02", o=100.0)})
        o = self.p.place_manual_order(self.trader, eng, "X", Side.BUY, 10)
        self.assertEqual(o.status, OrderStatus.ACKNOWLEDGED)
        self.assertEqual(o.strategy_id, "manual:trader")
        eng.on_bars({"X": bar("2024-01-03", o=101.0)})
        self.assertEqual(o.status, OrderStatus.FILLED)
        self.assertEqual(eng.portfolio.quantity("X"), 10)
        self.assertEqual(eng.broker.get_positions(), {"X": 10})
        self.assertEqual(eng.execution.reconcile(), {"orders": {"ghosts": [], "orphans": []}, "positions": {}})

    def test_risk_blocks_manual_fat_finger(self):
        eng = self.session()
        eng.start()
        eng.on_bars({"X": bar("2024-01-02", o=100.0)})
        o = self.p.place_manual_order(self.trader, eng, "X", Side.BUY, 10, OrderType.LIMIT, limit_price=500.0)
        self.assertEqual(o.status, OrderStatus.REJECTED)
        self.assertIn("price_collar", o.reject_reason)
        self.assertEqual(eng.broker.open_order_ids(), set())                        # never reached the broker

    def test_kill_switch_lifecycle_with_permissions(self):
        eng = self.session()
        eng.start()
        eng.on_bars({"X": bar("2024-01-02", o=100.0)})
        held = self.p.place_manual_order(self.trader, eng, "X", Side.BUY, 5, OrderType.LIMIT, limit_price=99.0)
        self.assertTrue(self.p.trip_kill_switch(self.trader, eng, "manual halt"))
        self.assertEqual(held.status, OrderStatus.CANCELLED)                         # resting orders pulled on trip
        o = self.p.place_manual_order(self.trader, eng, "X", Side.BUY, 1)
        self.assertEqual(o.status, OrderStatus.REJECTED)
        self.assertIn("kill switch", o.reject_reason)
        with self.assertRaises(AuthorizationError):
            self.p.reset_kill_switch(self.trader, eng, "I want to trade again")      # traders cannot reset
        self.p.reset_kill_switch(self.risk, eng, "reviewed by risk desk")
        ok = self.p.place_manual_order(self.trader, eng, "X", Side.BUY, 1)
        self.assertEqual(ok.status, OrderStatus.ACKNOWLEDGED)
        self.assertEqual(len(self.p.audit.entries("killswitch.reset")), 1)

    def test_kill_switch_flattens_open_position(self):
        eng = self.session()
        eng.start()
        eng.on_bars({"X": bar("2024-01-02", o=100.0)})
        self.p.place_manual_order(self.trader, eng, "X", Side.BUY, 10)
        eng.on_bars({"X": bar("2024-01-03", o=100.0)})
        self.assertEqual(eng.portfolio.quantity("X"), 10)
        self.p.trip_kill_switch(self.trader, eng, "emergency")
        eng.on_bars({"X": bar("2024-01-04", o=100.0)})
        self.assertEqual(eng.portfolio.quantity("X"), 0)

    def test_live_account_blocked_in_paper_environment(self):
        acc = self.p.accounts.link(self.trader, "paper", {"api_key": "k"}, paper=False)
        with self.assertRaises(ConfigError):
            self.p.open_session(self.trader, acc.account_id, [])

    def test_allocation_requires_risk_permission(self):
        with self.assertRaises(AuthorizationError):
            self.p.allocate_capital(self.trader, {"a": 0.5, "b": 0.5})
        out = self.p.allocate_capital(self.risk, {"a": 0.5, "b": 0.5})
        self.assertAlmostEqual(sum(out.values()), self.p.allocator.investable)

    def test_session_cancel_all_needs_permission(self):
        eng = self.session()
        eng.start()
        eng.on_bars({"X": bar("2024-01-02", o=100.0)})
        self.p.place_manual_order(self.trader, eng, "X", Side.BUY, 5, OrderType.LIMIT, limit_price=95.0)
        with self.assertRaises(AuthorizationError):
            self.p.cancel_all(self.viewer, eng)
        self.assertEqual(self.p.cancel_all(self.trader, eng), 1)


class CliTests(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = cli_main(list(argv))
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def test_strategies_and_backtest(self):
        code, out, _ = self.run_cli("strategies")
        self.assertEqual(code, 0)
        self.assertIn("sma_crossover", out)
        code, out, _ = self.run_cli("backtest", "--bars", "300", "--param", "fast=5", "--param", "slow=20")
        self.assertEqual(code, 0)
        self.assertIn("PERFORMANCE", out)

    def test_backtest_json_and_optimize(self):
        import json
        code, out, _ = self.run_cli("backtest", "--bars", "300", "--json")
        self.assertEqual(code, 0)
        self.assertIn("performance", json.loads(out))
        code, out, _ = self.run_cli("optimize", "--bars", "300", "--method", "random", "--n-iter", "5")
        self.assertEqual(code, 0)
        self.assertIn("best_params", out)

    def test_errors_return_nonzero_not_tracebacks(self):
        code, _, err = self.run_cli("backtest", "--bars", "300", "--param", "fast=50", "--param", "slow=10")
        self.assertEqual(code, 2)
        self.assertIn("error", err)
        code, _, _ = self.run_cli("backtest", "--strategy", "nope")
        self.assertNotEqual(code, 0)                                                 # argparse rejects unknown choice
        code, _, err = self.run_cli("backtest", "--csv", "A=/does/not/exist.csv")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
