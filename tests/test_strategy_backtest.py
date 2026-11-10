import json
import unittest

import numpy as np
import pandas as pd

from helpers import free_config, free_cost_config, market_from_closes
from algoplatform.backtest import BacktestEngine
from algoplatform.core import PlatformConfig, Signal
from algoplatform.core.config import RiskConfig
from algoplatform.core.exceptions import BacktestError, ConfigError, StrategyError
from algoplatform.data import MarketData, synthetic
from algoplatform.strategy import (STRATEGIES, MomentumRotation, PairsTrading, SMACrossover,
                                   Strategy, StrategyBuilder, create_from_spec, spec_from_json, spec_to_json,
                                   validate_spec)


class Scripted(Strategy):
    """Emits fixed weights at chosen bar indices (index counts bars seen so far, from 0)."""
    name = "scripted"
    PARAMS = {"plan": None}

    def on_bar(self, ctx):
        out = []
        for s in self.symbols:
            if s in ctx.bars:
                idx = ctx.bars_seen(s) - 1
                if idx in (self.p("plan") or {}):
                    out.append(Signal(s, self.p("plan")[idx], f"plan@{idx}"))
        return out


class Spy(Strategy):
    name = "spy"
    PARAMS = {}

    def __init__(self, symbols, **kw):
        super().__init__(symbols, **kw)
        self.seen = []

    def on_bar(self, ctx):
        s = self.symbols[0]
        self.seen.append((ctx.bars_seen(s), len(ctx.closes(s)), ctx.closes(s)[-1], ctx.bars[s].close, ctx.ts))
        return []


class Crashy(Strategy):
    name = "crashy"
    PARAMS = {}

    def on_bar(self, ctx):
        raise RuntimeError("user bug")


def trend_closes(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.012, n)))


class BuilderTests(unittest.TestCase):
    def base(self):
        return (StrategyBuilder("t").param("fast", 5).param("slow", 20)
                .indicator("fast", "sma", period="$fast").indicator("slow", "sma", period="$slow")
                .long_entry(("fast", ">", "slow")).long_exit(("fast", "<=", "slow")))

    def test_valid_spec_and_json_roundtrip(self):
        spec = self.base().build()
        self.assertEqual(spec_from_json(spec_to_json(spec)), json.loads(json.dumps(spec)))

    def test_validation_errors_are_specific(self):
        cases = {
            "unknown indicator type": lambda b: b.indicator("x", "magic", period=3),
            "missing period": lambda b: b.indicator("x", "sma"),
            "bad operator": lambda b: b.long_entry(("fast", "~", "slow")),
            "unknown operand": lambda b: b.long_entry(("nonexistent", ">", 1)),
            "undefined param": lambda b: b.indicator("y", "ema", period="$nope"),
            "bad weight": lambda b: b.weight(3),
            "price field clash": lambda b: b.indicator("close", "sma", period=2),
        }
        for label, mutate in cases.items():
            with self.assertRaises(StrategyError, msg=label):
                mutate(self.base()).build()
        with self.assertRaises(StrategyError):
            StrategyBuilder("empty").build()                                  # no entry rules
        with self.assertRaises(StrategyError):
            spec_from_json("{not json")

    def test_rule_strategy_close_to_builtin_sma(self):
        closes = trend_closes(400, 1)
        md = market_from_closes(X=closes)
        cfg = free_config()
        rule = BacktestEngine(md, cfg).run(create_from_spec(self.base().build(), ["X"], fast=5, slow=20))
        builtin = BacktestEngine(md, cfg).run(SMACrossover(["X"], fast=5, slow=20))
        self.assertGreater(len(rule.trades), 3)
        self.assertAlmostEqual(rule.metrics()["total_return"], builtin.metrics()["total_return"], delta=0.03)

    def test_parameter_override_and_unknown_param(self):
        spec = self.base().build()
        s = create_from_spec(spec, ["X"], fast=3)
        self.assertEqual(s.params["fast"], 3)
        with self.assertRaises(StrategyError):
            create_from_spec(spec, ["X"], bogus=1)

    def test_or_groups_and_lagged_operands_and_stop(self):
        spec = (StrategyBuilder("combo").param("n", 10)
                .indicator("rsi", "rsi", period=14).indicator("atr", "atr", period=14)
                .indicator("hi", "rolling_max", period="$n")
                .long_entry({"any": [("rsi", "<", 30), ["close", ">", "hi@1"]]})
                .long_exit(("rsi", ">", 70)).stop_atr("atr", 2.0).build())
        md = market_from_closes(X=trend_closes(400, 3))
        res = BacktestEngine(md, free_config()).run(create_from_spec(spec, ["X"]))
        self.assertGreater(len(res.trades), 0)

    def test_short_side_rules(self):
        spec = (StrategyBuilder("shorty").indicator("s", "sma", period=10)
                .short_entry(("close", "<", "s")).short_exit(("close", ">", "s")).build())
        closes = 100 * np.exp(-np.arange(200) * 0.004)
        res = BacktestEngine(market_from_closes(X=closes), free_config()).run(create_from_spec(spec, ["X"]))
        self.assertGreater(res.trade_stats()["short_pnl"], 0)                 # profits from a falling market

    def test_indicator_output_selection(self):
        spec = (StrategyBuilder("bb").indicator("lo", "bollinger", period=20, k=2.0, output="lower")
                .long_entry(("close", "<", "lo")).build())
        validate_spec(spec)
        bad = (StrategyBuilder("bb").indicator("lo", "bollinger", period=20, output="nonsense")
               .long_entry(("close", "<", "lo")))
        with self.assertRaises(StrategyError):
            bad.build()


class LibraryTests(unittest.TestCase):
    def test_registry_and_param_validation(self):
        self.assertIn("sma_crossover", STRATEGIES.names())
        with self.assertRaises(StrategyError):
            SMACrossover(["X"], fast=50, slow=10)
        with self.assertRaises(StrategyError):
            SMACrossover(["X"], nope=1)
        with self.assertRaises(StrategyError):
            SMACrossover([])
        with self.assertRaises(ConfigError):
            STRATEGIES.get("missing")

    def test_every_builtin_runs_on_synthetic_data(self):
        data = synthetic.universe(["A", "B", "C"], n=400, seed=2)
        md = MarketData()
        for s, df in data.items():
            md.add(s, df)
        needs_data = {"factor_ls", "sentiment"}                         # need a fundamentals store / news signal attached
        for name in STRATEGIES.names():
            if name in needs_data:
                res = BacktestEngine(md, PlatformConfig()).run(STRATEGIES.create(name, ["A", "B", "C"]))
                self.assertTrue(res.kill_switch_triggered, name)          # fails loudly, contained, and halts trading
                continue
            if name == "pairs_trading":
                strat = STRATEGIES.create(name, ["A", "B"])
            else:
                strat = STRATEGIES.create(name, ["A", "B", "C"])
            res = BacktestEngine(md, PlatformConfig()).run(strat)
            self.assertEqual(res.summary["strategy_errors"], 0, name)
            self.assertGreater(len(res.equity), 100, name)

    def test_pairs_trading_profits_on_cointegrated_pair(self):
        y, x = synthetic.cointegrated_pair(n=1000, beta=1.5, theta=0.08, seed=5)
        md = MarketData().add("Y", y).add("X", x)
        res = BacktestEngine(md, free_config()).run(PairsTrading(["Y", "X"], lookback=60, entry_z=2.0, exit_z=0.3))
        self.assertGreater(len(res.trades), 5)
        self.assertGreater(res.trade_stats()["total_pnl"], 0)

    def test_momentum_picks_the_winner(self):
        n = 300
        md = market_from_closes(UP=100 * np.exp(np.arange(n) * 0.004), FLAT=np.full(n, 100.0),
                                DOWN=100 * np.exp(-np.arange(n) * 0.004))
        res = BacktestEngine(md, free_config()).run(MomentumRotation(["UP", "FLAT", "DOWN"], lookback=40, skip=2,
                                                                      top_n=1, rebalance_every=10))
        held = {f.symbol for f in res.fills}
        self.assertEqual(held, {"UP"})

    def test_clone_resets_state(self):
        s = SMACrossover(["X"], fast=3, slow=9)
        c = s.clone(fast=4)
        self.assertEqual((c.params["fast"], c.params["slow"]), (4, 9))


class BacktestIntegrityTests(unittest.TestCase):
    def test_strategy_never_sees_the_future(self):
        md = market_from_closes(X=trend_closes(120, 4))
        spy = Spy(["X"])
        BacktestEngine(md, free_config()).run(spy)
        for i, (seen, n, last_close, bar_close, ts) in enumerate(spy.seen):
            self.assertEqual(seen, i + 1)
            self.assertEqual(n, i + 1)                                         # history length == bars delivered
            self.assertEqual(last_close, bar_close)
        window_is_readonly = md.arrays("X")["close"]                           # arrays untouched by strategies
        self.assertEqual(len(window_is_readonly), 120)

    def test_altering_future_data_does_not_change_past_results(self):
        closes = trend_closes(300, 6)
        altered = closes.copy()
        altered[200:] *= 3.0                                                   # wreck the future
        r1 = BacktestEngine(market_from_closes(X=closes), free_config()).run(SMACrossover(["X"], fast=5, slow=20))
        r2 = BacktestEngine(market_from_closes(X=altered), free_config()).run(SMACrossover(["X"], fast=5, slow=20))
        cut = r1.equity.index[198]
        pd.testing.assert_series_equal(r1.equity.loc[:cut], r2.equity.loc[:cut])

    def test_signal_fills_on_next_bar_open(self):
        closes = np.linspace(100, 130, 30)
        md = market_from_closes(X=closes)
        res = BacktestEngine(md, free_config()).run(Scripted(["X"], plan={5: 1.0, 15: 0.0}), )
        buy, sell = res.fills[0], res.fills[1]
        idx = md.frame("X").index
        self.assertEqual(buy.ts, idx[6])                                       # decided on bar 5, filled on bar 6
        self.assertAlmostEqual(buy.price, md.frame("X").open.iloc[6])
        self.assertEqual(sell.ts, idx[16])
        self.assertAlmostEqual(sell.price, md.frame("X").open.iloc[16])

    def test_start_end_window_uses_warmup_without_trading_it(self):
        md = market_from_closes(X=trend_closes(300, 2))
        idx = md.frame("X").index
        res = BacktestEngine(md, free_config()).run(SMACrossover(["X"], fast=5, slow=20), start=idx[100], end=idx[200])
        self.assertGreaterEqual(res.equity.index[0], idx[100])
        self.assertLessEqual(res.equity.index[-1], idx[200])
        self.assertTrue(all(f.ts >= idx[100] for f in res.fills))

    def test_costs_reduce_returns(self):
        md = market_from_closes(X=trend_closes(400, 7))
        free = BacktestEngine(md, free_config()).run(SMACrossover(["X"], fast=3, slow=10))
        costly_cfg = PlatformConfig(costs=free_cost_config().__class__(commission_per_share=0.05, slippage_bps=20,
                                                                         spread_bps=10, min_commission=1.0))
        costly = BacktestEngine(md, costly_cfg).run(SMACrossover(["X"], fast=3, slow=10))
        self.assertLess(costly.metrics()["total_return"], free.metrics()["total_return"])
        self.assertGreater(costly.summary["commission"], 0)

    def test_equity_identity_and_trade_pnl_reconcile(self):
        md = market_from_closes(X=trend_closes(300, 8))
        res = BacktestEngine(md, PlatformConfig()).run(SMACrossover(["X"], fast=5, slow=20))
        net_trades = sum(t.pnl for t in res.trades)
        self.assertAlmostEqual(res.equity.iloc[-1] - res.config.initial_capital, net_trades, delta=1.0)

    def test_opposite_strategies_net_to_zero_orders(self):
        md = market_from_closes(X=np.linspace(100, 120, 40))
        a, b = Scripted(["X"], strategy_id="A", plan={5: 1.0}), Scripted(["X"], strategy_id="B", plan={5: -1.0})
        res = BacktestEngine(md, free_config()).run([(a, 0.5), (b, 0.5)])
        self.assertEqual(len(res.fills), 0)                                    # internal crossing: nothing sent out
        self.assertAlmostEqual(res.equity.iloc[-1], 100_000)

    def test_strategy_exceptions_are_contained_then_halt_trading(self):
        md = market_from_closes(X=np.linspace(100, 120, 40))
        res = BacktestEngine(md, free_config()).run(Crashy(["X"]))
        self.assertGreaterEqual(res.summary["strategy_errors"], 5)
        self.assertTrue(res.kill_switch_triggered)                              # 5 consecutive errors => halt

    def test_kill_switch_flattens_and_caps_loss_on_crash(self):
        closes = np.concatenate([np.full(40, 100.0), [70, 60, 50, 40, 30, 30, 30]])
        md = market_from_closes(X=closes)
        cfg = free_config(risk=RiskConfig(max_daily_loss_pct=0.05, max_drawdown_pct=0.5))
        res = BacktestEngine(md, cfg).run(Scripted(["X"], plan={10: 1.0}))
        self.assertTrue(res.kill_switch_triggered)
        self.assertEqual(res.summary["positions"], {})                          # flattened
        final_loss = 1 - res.equity.iloc[-1] / 100_000
        self.assertAlmostEqual(final_loss, 0.075, delta=0.005)                  # stopped at the first -30% bar, not -70%
        unprotected = BacktestEngine(md, free_config(risk=RiskConfig(max_daily_loss_pct=1.0, max_drawdown_pct=1.0))) \
            .run(Scripted(["X"], plan={10: 1.0}))
        self.assertLess(unprotected.equity.iloc[-1], res.equity.iloc[-1])

    def test_results_do_not_depend_on_hash_seed(self):
        """Regression: iterating a set of symbols made results vary from process to process."""
        import os, subprocess, sys
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "from algoplatform.data import MarketData, synthetic\n"
            "from algoplatform.backtest import BacktestEngine\n"
            "from algoplatform.strategy import SMACrossover\n"
            "from algoplatform.core import PlatformConfig\n"
            "from algoplatform.core.config import RiskConfig\n"
            "md = MarketData()\n"
            "for s, df in synthetic.universe(['AAA','BBB','CCC','DDD'], n=300, seed=11).items(): md.add(s, df)\n"
            "cfg = PlatformConfig(risk=RiskConfig(max_net_exposure_pct=0.5))   # cap binds: submission order matters\n"
            "r = BacktestEngine(md, cfg).run(SMACrossover(md.symbols, fast=5, slow=32))\n"
            "print(repr(round(float(r.equity.iloc[-1]), 6)), len(r.fills))\n"
        ) % os.path.join(os.path.dirname(__file__), "..", "src")
        outs = set()
        for seed in ("1", "2", "3", "4", "5"):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            outs.add(subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True,
                                    check=True).stdout.strip())
        self.assertEqual(len(outs), 1, outs)

    def test_errors_for_bad_inputs(self):
        md = market_from_closes(X=np.linspace(100, 110, 30))
        with self.assertRaises(BacktestError):
            BacktestEngine(md, free_config()).run(SMACrossover(["MISSING"]))
        with self.assertRaises(BacktestError):
            BacktestEngine(MarketData(), free_config())
        with self.assertRaises(BacktestError):
            BacktestEngine(md, free_config()).run(SMACrossover(["X"], fast=2, slow=5), start="2030-01-01")

    def test_result_report_and_benchmark(self):
        md = market_from_closes(X=trend_closes(300, 9))
        res = BacktestEngine(md, PlatformConfig()).run(SMACrossover(["X"], fast=5, slow=20))
        rep = res.report(n_trials=10)
        self.assertIn("beta", rep["performance"])
        self.assertIn("deflated_sharpe", rep["significance"])
        self.assertLessEqual(rep["significance"]["deflated_sharpe"], rep["significance"]["probabilistic_sharpe_vs_0"] + 1e-9)


if __name__ == "__main__":
    unittest.main()
