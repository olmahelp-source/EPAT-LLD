import math
import os
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from helpers import df_from_closes, free_config, market_from_closes
from algoplatform.analytics import (build_report, drawdown_periods, drawdown_series, html_report, infer_periods_per_year,
                                    max_drawdown, monthly_returns, performance_metrics, rolling_sharpe, text_report,
                                    to_json, trade_statistics, contribution_by, trades_frame)
from algoplatform.backtest import BacktestEngine
from algoplatform.core.exceptions import DataError, MissingDataError, OptimizationError, ValidationError
from algoplatform.data import MarketData, adjust_for_split, clean_ohlcv, find_gaps, synthetic, validate_ohlcv
from algoplatform.optimize import (Param, ParameterSpace, backtest_objective, genetic_search, grid_search, optimize,
                                   random_search, walk_forward)
from algoplatform.portfolio import Trade
from algoplatform.strategy import SMACrossover


# ---------------------------------------------------------------------------- optimisation
def quad(p):
    return -((p["x"] - 7) ** 2) - ((p["y"] - 3) ** 2)


class SpaceTests(unittest.TestCase):
    def test_param_validation(self):
        with self.assertRaises(OptimizationError):
            Param("a", "int", 5, 5)
        with self.assertRaises(OptimizationError):
            Param("a", "log", 0, 10)
        with self.assertRaises(OptimizationError):
            Param("a", "choice")
        with self.assertRaises(OptimizationError):
            ParameterSpace([Param("a", "int", 0, 3), Param("a", "int", 0, 3)])

    def test_grid_sampling_and_constraints(self):
        sp = ParameterSpace([Param("fast", "int", 2, 10, step=2), Param("slow", "int", 4, 12, step=4)],
                            constraint=lambda p: p["fast"] < p["slow"])
        pts = list(sp.grid())
        self.assertTrue(all(p["fast"] < p["slow"] for p in pts))
        rng = np.random.default_rng(0)
        for _ in range(50):
            self.assertTrue(sp.valid(sp.sample(rng)))
        log = Param("lr", "log", 1e-4, 1e-1)
        vals = log.values(4)
        self.assertAlmostEqual(vals[1] / vals[0], vals[2] / vals[1])               # geometric spacing
        impossible = ParameterSpace([Param("a", "int", 0, 3)], constraint=lambda p: False)
        with self.assertRaises(OptimizationError):
            impossible.sample(rng)

    def test_from_bounds(self):
        sp = ParameterSpace.from_bounds({"n": (5, 20), "z": (0.5, 2.5)})
        self.assertEqual([p.kind for p in sp.params], ["int", "float"])


class OptimizerTests(unittest.TestCase):
    space = ParameterSpace([Param("x", "int", 0, 14), Param("y", "int", 0, 6)])

    def test_grid_finds_exact_optimum(self):
        res = grid_search(quad, ParameterSpace([Param("x", "int", 0, 14, step=1), Param("y", "int", 0, 6, step=1)]))
        self.assertEqual(res.best_params, {"x": 7, "y": 3})
        self.assertEqual(res.best_score, 0)
        self.assertEqual(res.n_trials, 15 * 7)

    def test_random_is_reproducible_and_genetic_converges(self):
        a, b = random_search(quad, self.space, 30, seed=1), random_search(quad, self.space, 30, seed=1)
        self.assertEqual(a.best_params, b.best_params)
        g = genetic_search(quad, self.space, population=14, generations=8, seed=3)
        self.assertGreaterEqual(g.best_score, -2)                                    # within 1-2 steps of the optimum
        self.assertLessEqual(g.n_trials, 14 * 9)                                     # caching avoids re-evaluation

    def test_failed_trials_are_isolated(self):
        def obj(p):
            if p["x"] % 2:
                raise RuntimeError("boom")
            if p["x"] == 4:
                return float("nan")
            return float(p["x"])
        res = grid_search(obj, ParameterSpace([Param("x", "int", 0, 8, step=1)]))
        self.assertEqual(res.best_params["x"], 8)
        self.assertEqual(res.summary()["failed"], 5)                                 # x=1,3,5,7 raise; x=4 is NaN
        with self.assertRaises(OptimizationError):
            random_search(lambda p: 1 / 0, self.space, 5)                            # everything failed

    def test_plateau_score_distinguishes_spike_from_plateau(self):
        sp = ParameterSpace([Param("x", "int", 0, 8, step=1)])
        plateau = grid_search(lambda p: 10 - abs(p["x"] - 4) * 0.2, sp)
        spike = grid_search(lambda p: 10 if p["x"] == 4 else 1, sp)
        self.assertGreater(plateau.plateau_score(sp), 0.9)
        self.assertLess(spike.plateau_score(sp), 0.2)

    def test_plateau_score_works_for_random_search(self):
        sp = ParameterSpace([Param("x", "int", 0, 40), Param("y", "int", 0, 40)])
        smooth = random_search(lambda p: 100 - 0.05 * ((p["x"] - 20) ** 2 + (p["y"] - 20) ** 2), sp, 200, seed=0)
        self.assertGreater(smooth.plateau_score(sp), 0.8)
        self.assertTrue(math.isfinite(smooth.plateau_score(sp)))

    def test_unknown_method_and_oversized_grid(self):
        with self.assertRaises(OptimizationError):
            optimize(quad, self.space, "bayes")
        big = ParameterSpace([Param(f"p{i}", "int", 0, 40, step=1) for i in range(5)])
        with self.assertRaises(OptimizationError):
            grid_search(lambda p: 0, big)

    def test_backtest_objective_end_to_end(self):
        md = market_from_closes(X=100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.001, 0.012, 400))))
        eng = BacktestEngine(md, free_config())
        sp = ParameterSpace([Param("fast", "int", 3, 15), Param("slow", "int", 16, 60)],
                            constraint=lambda p: p["fast"] < p["slow"])
        obj = backtest_objective(eng, lambda **p: SMACrossover(["X"], **p), "sharpe", min_trades=2)
        res = random_search(obj, sp, 10, seed=0)
        self.assertLess(res.best_params["fast"], res.best_params["slow"])
        self.assertTrue(math.isfinite(res.best_score))
        never_trades = backtest_objective(eng, lambda **p: SMACrossover(["X"], **p), "sharpe", min_trades=10_000)
        self.assertEqual(never_trades({"fast": 5, "slow": 20}), -math.inf)           # "optimal" that never trades is rejected
        with self.assertRaises(Exception):
            backtest_objective(eng, lambda **p: SMACrossover(["X"], **p), "no_such_metric", min_trades=0)({"fast": 5, "slow": 20})

    def test_invalid_strategy_params_become_failed_trials_not_crashes(self):
        md = market_from_closes(X=100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0.001, 0.012, 300))))
        eng = BacktestEngine(md, free_config())
        sp = ParameterSpace([Param("fast", "int", 3, 30), Param("slow", "int", 10, 40)])        # no constraint: some invalid
        res = random_search(backtest_objective(eng, lambda **p: SMACrossover(["X"], **p), min_trades=1), sp, 15, seed=1)
        self.assertGreater(res.summary()["failed"], 0)
        self.assertLess(res.best_params["fast"], res.best_params["slow"])


class WalkForwardTests(unittest.TestCase):
    def test_walk_forward_stitches_out_of_sample(self):
        md = market_from_closes(X=100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0.0007, 0.012, 600))))
        eng = BacktestEngine(md, free_config())
        sp = ParameterSpace([Param("fast", "int", 3, 15), Param("slow", "int", 16, 60)],
                            constraint=lambda p: p["fast"] < p["slow"])
        wf = walk_forward(eng, lambda **p: SMACrossover(["X"], **p), sp, train_bars=200, test_bars=100, n_iter=6,
                          min_trades=1)
        self.assertGreaterEqual(len(wf.folds), 3)
        self.assertTrue((wf.folds["test_start"] > wf.folds["train_end"]).all())      # test strictly after train
        self.assertTrue(wf.oos_equity.index.is_monotonic_increasing)
        self.assertEqual(set(wf.param_stability), {"fast", "slow"})
        self.assertIn("oos_sharpe", wf.summary())

    def test_bad_windows(self):
        md = market_from_closes(X=np.linspace(100, 120, 100))
        sp = ParameterSpace([Param("fast", "int", 3, 5), Param("slow", "int", 6, 9)])
        with self.assertRaises(OptimizationError):
            walk_forward(BacktestEngine(md, free_config()), lambda **p: SMACrossover(["X"], **p), sp, 90, 50)


# ---------------------------------------------------------------------------- analytics
class MetricTests(unittest.TestCase):
    def curve(self, values, start="2024-01-01"):
        return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)

    def test_drawdown_periods_and_depth(self):
        eq = self.curve([100, 120, 90, 100, 130, 125])
        self.assertAlmostEqual(max_drawdown(eq), -0.25)
        dd = drawdown_periods(eq)
        self.assertEqual(len(dd), 2)
        self.assertAlmostEqual(dd.loc[0, "depth"], -0.25)
        self.assertTrue(pd.notna(dd.loc[0, "recovery"]))
        self.assertTrue(pd.isna(dd.loc[1, "recovery"]))                              # still underwater at the end
        self.assertAlmostEqual(drawdown_series(eq).iloc[-1], 125 / 130 - 1)

    def test_known_metrics(self):
        eq = self.curve([100, 110, 99, 108.9, 119.79])
        m = performance_metrics(eq, periods_per_year=252)
        self.assertAlmostEqual(m["total_return"], 0.1979, places=4)
        self.assertAlmostEqual(m["max_drawdown"], -0.10)
        self.assertAlmostEqual(m["best_bar"], 0.10)
        self.assertAlmostEqual(m["worst_bar"], -0.10)
        self.assertGreater(m["sortino"], m["sharpe"] * 0)                             # defined and finite
        flat = performance_metrics(self.curve([100] * 10))
        self.assertEqual(flat["sharpe"], 0.0)
        with self.assertRaises(ValidationError):
            performance_metrics(self.curve([100, 101]))

    def test_periods_per_year_inference(self):
        self.assertEqual(infer_periods_per_year(pd.bdate_range("2024-01-01", periods=50)), 252.0)
        self.assertEqual(infer_periods_per_year(pd.date_range("2024-01-01", periods=50, freq="W")), 52.0)
        self.assertEqual(infer_periods_per_year(pd.date_range("2024-01-01", periods=50, freq="MS")), 12.0)

    def test_benchmark_beta_alpha(self):
        rng = np.random.default_rng(0)
        bench = pd.Series(rng.normal(0.0004, 0.01, 500), index=pd.bdate_range("2022-01-01", periods=500))
        strat = 1.5 * bench + 0.0002
        eq = 100 * (1 + strat).cumprod()
        eq = pd.concat([pd.Series([100.0], index=[eq.index[0] - pd.Timedelta(days=1)]), eq])
        m = performance_metrics(eq, 252, benchmark=bench)
        self.assertAlmostEqual(m["beta"], 1.5, places=2)
        self.assertAlmostEqual(m["correlation"], 1.0, places=3)
        self.assertAlmostEqual(m["alpha"], 0.0002 * 252, delta=0.005)

    def test_monthly_table_and_rolling_sharpe(self):
        eq = self.curve(100 * np.cumprod(1 + np.random.default_rng(1).normal(0.001, 0.01, 300)))
        mt = monthly_returns(eq)
        self.assertGreater(mt.shape[1], 6)
        rs = rolling_sharpe(eq.pct_change().dropna(), 60, 252)
        self.assertTrue(rs.iloc[:58].isna().all())


def trade(pnl, direction=1, sym="X", strat="s", entry_px=100.0, qty=10, days=5):
    t0 = datetime(2024, 1, 1)
    return Trade(sym, direction, qty, entry_px, entry_px + pnl / qty, t0, t0 + timedelta(days=days), pnl, 1.0, strat, 2.0, 5.0)


class TradeStatsTests(unittest.TestCase):
    def test_known_statistics(self):
        ts = [trade(100), trade(-50), trade(200), trade(-50)]
        s = trade_statistics(ts)
        self.assertEqual(s["trades"], 4)
        self.assertAlmostEqual(s["win_rate"], 0.5)
        self.assertAlmostEqual(s["avg_win"], 150)
        self.assertAlmostEqual(s["avg_loss"], 50)
        self.assertAlmostEqual(s["payoff_ratio"], 3.0)
        self.assertAlmostEqual(s["profit_factor"], 3.0)
        self.assertAlmostEqual(s["expectancy"], 50)
        self.assertEqual(s["max_consecutive_losses"], 1)
        self.assertAlmostEqual(s["avg_hold_days"], 5.0)
        self.assertEqual(trade_statistics([]), {"trades": 0})

    def test_streaks_and_dependence_on_top_trades(self):
        ts = [trade(10), trade(-5), trade(-5), trade(-5), trade(1000)]
        s = trade_statistics(ts)
        self.assertEqual(s["max_consecutive_losses"], 3)
        self.assertAlmostEqual(s["pnl_without_top5"], 0.0)                           # all five trades removed
        s2 = trade_statistics([trade(10)] * 8 + [trade(500)])
        self.assertLess(s2["pnl_without_top5"], s2["total_pnl"])

    def test_contribution_grouping(self):
        ts = [trade(100, sym="A"), trade(-20, sym="A"), trade(50, sym="B", direction=-1)]
        c = contribution_by(ts, "symbol")
        self.assertAlmostEqual(c.loc["A", "pnl"], 80)
        self.assertAlmostEqual(c.loc["A", "win_rate"], 0.5)
        self.assertAlmostEqual(contribution_by(ts, "direction").loc[-1, "pnl"], 50)
        self.assertEqual(len(trades_frame(ts)), 3)


class ReportTests(unittest.TestCase):
    def test_reports_render(self):
        md = market_from_closes(X=100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.001, 0.012, 300))))
        res = BacktestEngine(md, free_config()).run(SMACrossover(["X"], fast=5, slow=20))
        rep = build_report(res.equity, res.trades, n_trials=5)
        self.assertIn("PERFORMANCE", text_report(rep))
        page = html_report(res.equity, rep, "My <Strategy>")
        self.assertIn("<svg", page)
        self.assertIn("&lt;Strategy&gt;", page)                                      # titles are escaped
        self.assertNotIn("<Strategy>", page)
        import json
        json.loads(to_json(rep))


# ---------------------------------------------------------------------------- data
class DataTests(unittest.TestCase):
    def good(self, n=30):
        return df_from_closes(np.linspace(100, 110, n))

    def test_validation_catches_each_problem(self):
        df = self.good()
        validate_ohlcv(df)
        dup = pd.concat([df, df.iloc[[0]]])
        with self.assertRaises(DataError):
            validate_ohlcv(dup)
        with self.assertRaises(DataError):
            validate_ohlcv(df.iloc[::-1])
        nan = df.copy()
        nan.iloc[3, 3] = np.nan
        with self.assertRaises(DataError):
            validate_ohlcv(nan)
        neg = df.copy()
        neg.iloc[3, 0] = -1
        with self.assertRaises(DataError):
            validate_ohlcv(neg)
        bad = df.copy()
        bad.iloc[5, 1] = bad.iloc[5, 2] * 0.5                                        # high below low
        with self.assertRaises(DataError):
            validate_ohlcv(bad)
        with self.assertRaises(DataError):
            validate_ohlcv(df.drop(columns=["close"]))
        with self.assertRaises(DataError):
            validate_ohlcv(pd.DataFrame())

    def test_cleaning_removes_spike_and_repairs(self):
        df = self.good(80)
        df.iloc[40, df.columns.get_loc("close")] *= 5                                # one-bar bad tick
        df.iloc[40, df.columns.get_loc("high")] = df.iloc[40]["close"]
        clean, rep = clean_ohlcv(df)
        self.assertEqual(rep["outliers"], 1)
        self.assertEqual(len(clean), 79)
        messy = pd.concat([self.good(10), self.good(10).iloc[[2]]]).sort_index()
        c2, r2 = clean_ohlcv(messy)
        self.assertEqual(r2["duplicates"], 1)
        self.assertFalse(c2.index.has_duplicates)

    def test_real_jump_is_not_removed(self):
        df = self.good(80)
        df.iloc[40:, :4] *= 1.5                                                      # permanent 50% re-rating (e.g. takeover)
        _, rep = clean_ohlcv(df)
        self.assertEqual(rep["outliers"], 0)

    def test_split_adjustment_and_gap_finder(self):
        df = self.good(20)
        split_date = df.index[10]
        adj = adjust_for_split(df, split_date, 2.0)
        self.assertAlmostEqual(adj.close.iloc[0], df.close.iloc[0] / 2)
        self.assertAlmostEqual(adj.close.iloc[10], df.close.iloc[10])
        self.assertAlmostEqual(adj.volume.iloc[0], df.volume.iloc[0] * 2)
        with self.assertRaises(ValidationError):
            adjust_for_split(df, split_date, 0)
        gappy = pd.concat([df.iloc[:5], df.iloc[12:]])
        self.assertEqual(len(find_gaps(gappy, pd.Timedelta(days=4))), 1)

    def test_market_data_store_csv_and_batches(self):
        md = MarketData()
        md.add("A", self.good(20))
        md.add("B", self.good(20).iloc[5:])                                          # starts later
        batches = md.bar_batches()
        self.assertEqual(len(batches), 20)
        self.assertEqual(set(batches[0]), {"A"})
        self.assertEqual(set(batches[10]), {"A", "B"})
        self.assertIs(md.bar_batches(), batches)                                      # cached
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.csv")
            md.to_csv("A", path)
            md2 = MarketData().add_csv("A2", path)
            pd.testing.assert_frame_equal(md.frame("A"), md2.frame("A2"), check_freq=False)
            with self.assertRaises(MissingDataError):
                MarketData().add_csv("Z", os.path.join(d, "missing.csv"))
        with self.assertRaises(MissingDataError):
            md.frame("NOPE")
        self.assertEqual(len(md.slice("2021-01-04", "2021-01-08").frame("A")), 5)

    def test_synthetic_is_reproducible_and_valid(self):
        a, b = synthetic.gbm(200, seed=5), synthetic.gbm(200, seed=5)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(a.equals(synthetic.gbm(200, seed=6)))
        validate_ohlcv(a)
        df, regimes = synthetic.regime_switching(500, seed=1)
        validate_ohlcv(df)
        self.assertEqual(set(np.unique(regimes)), {0, 1})
        for d in synthetic.universe(["A", "B"], n=100, seed=1).values():
            validate_ohlcv(d)
        with self.assertRaises(ValidationError):
            synthetic.gbm(1)


if __name__ == "__main__":
    unittest.main()
