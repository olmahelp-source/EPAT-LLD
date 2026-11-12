import math
import unittest

import numpy as np
import pandas as pd

import helpers  # noqa: F401
from helpers import free_config
from algoplatform.analytics import (EvaluationLimits, bootstrap_metrics, compare_strategies, drawdown_probability, evaluate_backtest,
                                    evaluate_strategy, regime_breakdown, rolling_stability, trade_concentration)
from algoplatform.backtest import BacktestEngine
from algoplatform.core.exceptions import OptimizationError, ValidationError
from algoplatform.data import MarketData, synthetic
from algoplatform.optimize import (ADVANCED_METHODS, Param, ParameterSpace, UnitCube, backtest_objective, bayesian_search, cma_es,
                                   differential_evolution, multi_objective_search, optimize, pareto_front, pso_search, random_search,
                                   simulated_annealing, successive_halving, tpe_search)
from algoplatform.selection import RANKERS, UniverseSelector, ranker, select_one_per_cluster, select_pairs
from algoplatform.strategy import SMACrossover

SPACE2 = ParameterSpace([Param("x", "float", -5, 5), Param("y", "float", -5, 5)])
MIXED = ParameterSpace([Param("x", "float", -5, 5), Param("y", "float", -5, 5), Param("n", "int", 1, 20), Param("mode", "choice", choices=("a", "b", "c"))])
BUDGETS = {"bayesian": dict(n_iter=40), "tpe": dict(n_iter=60), "pso": dict(particles=10, iterations=6), "de": dict(population=10, generations=5),
           "annealing": dict(n_iter=60), "cmaes": dict(generations=8)}


def bowl(p):
    return -((p["x"] - 1.5) ** 2 + (p["y"] + 2) ** 2)


def mixed(p):
    if p["x"] > 4.5:
        raise ValueError("this region crashes")
    return -((p["x"] - 1.5) ** 2 + (p["y"] + 2) ** 2 + 0.05 * (p["n"] - 7) ** 2 + (0 if p["mode"] == "b" else 1.0))


class UnitCubeTests(unittest.TestCase):
    def test_round_trip_for_every_parameter_kind(self):
        space = ParameterSpace([Param("f", "float", -2, 6), Param("s", "float", 0, 1, step=0.25), Param("i", "int", 3, 9),
                                Param("lg", "log", 1e-4, 1.0), Param("c", "choice", choices=("x", "y", "z"))])
        cube = UnitCube(space)
        self.assertEqual(cube.dim, 4 + 3)                                             # a choice takes one coordinate per option
        point = {"f": 1.0, "s": 0.75, "i": 6, "lg": 0.01, "c": "y"}
        vec = cube.encode(point)
        self.assertTrue(((vec >= 0) & (vec <= 1)).all())
        self.assertAlmostEqual(vec[3], 0.5)                                           # log scale: 0.01 is half way from 1e-4 to 1
        back = cube.decode(vec)
        self.assertEqual({k: back[k] for k in ("s", "i", "c")}, {"s": 0.75, "i": 6, "c": "y"})
        self.assertAlmostEqual(back["f"], 1.0)
        self.assertAlmostEqual(back["lg"], 0.01)
        edge = cube.decode(np.full(cube.dim, 7.0))                                    # out-of-range vectors are clipped into the space
        self.assertEqual((edge["f"], edge["i"], edge["lg"]), (6.0, 9, 1.0))
        self.assertIsInstance(edge["i"], int)

    def test_constraint_is_respected(self):
        space = ParameterSpace([Param("fast", "int", 2, 50), Param("slow", "int", 2, 50)], constraint=lambda p: p["fast"] < p["slow"])
        cube, rng = UnitCube(space), np.random.default_rng(0)
        for _ in range(50):
            p = cube.valid_decode(rng.random(cube.dim), rng)
            self.assertLess(p["fast"], p["slow"])
        for method, kw in BUDGETS.items():
            res = optimize(lambda p: -abs(p["fast"] - 10) - abs(p["slow"] - 30), space, method, **kw)
            self.assertTrue((res.trials["fast"] < res.trials["slow"]).all(), method)


class AdvancedOptimizerTests(unittest.TestCase):
    def test_every_method_gets_close_on_a_smooth_problem(self):
        for method, kw in BUDGETS.items():
            best = np.median([optimize(bowl, SPACE2, method, seed=s, **kw).best_score for s in range(3)])
            self.assertGreater(best, -0.6, method)                                    # the optimum is 0; a random point averages about -19

    def test_model_based_methods_beat_random_search_at_equal_budget(self):
        seeds = range(6)
        rnd = np.median([random_search(bowl, SPACE2, n_iter=40, seed=s).best_score for s in seeds])
        bayes = np.median([bayesian_search(bowl, SPACE2, n_iter=40, seed=s).best_score for s in seeds])
        tpe = np.median([tpe_search(bowl, SPACE2, n_iter=40, seed=s).best_score for s in seeds])
        self.assertGreater(bayes, rnd / 3)                                            # at least three times closer to the optimum
        self.assertGreater(tpe, rnd)

    def test_mixed_space_with_crashing_trials(self):
        for method, kw in BUDGETS.items():
            res = optimize(mixed, MIXED, method, **kw)
            self.assertTrue(math.isfinite(res.best_score), method)
            self.assertLess(res.best_params["x"], 4.5, method)
            failed = res.trials[res.trials["error"].notna()]
            self.assertTrue((failed["score"] == -math.inf).all(), method)             # a crash is a failed trial, not a failed search
            self.assertIn(res.best_params["mode"], ("a", "b", "c"))
            self.assertIsInstance(res.best_params["n"], int)
        self.assertGreater(bayesian_search(mixed, MIXED, n_iter=60).best_score, -1.0)

    def test_deterministic_and_seed_dependent(self):
        for fn, kw in ((bayesian_search, dict(n_iter=20)), (tpe_search, dict(n_iter=25)), (pso_search, dict(particles=6, iterations=3)),
                       (differential_evolution, dict(population=6, generations=3)), (simulated_annealing, dict(n_iter=25)),
                       (cma_es, dict(generations=4))):
            a, b, c = fn(mixed, MIXED, seed=1, **kw), fn(mixed, MIXED, seed=1, **kw), fn(mixed, MIXED, seed=2, **kw)
            self.assertTrue(a.trials.equals(b.trials), fn.__name__)
            self.assertFalse(a.trials[["x", "y"]].equals(c.trials[["x", "y"]]), fn.__name__)

    def test_results_feed_the_overfitting_bookkeeping(self):
        res = bayesian_search(bowl, SPACE2, n_iter=30)
        summary = res.summary()
        self.assertEqual((summary["method"], summary["trials"]), ("bayesian", 30))
        self.assertTrue(math.isfinite(summary["score_variance_for_dsr"]))
        self.assertEqual(res.n_trials, len(res.trials))
        self.assertEqual(len(res.top(5)), 5)
        self.assertEqual(set(ADVANCED_METHODS), set(BUDGETS))

    def test_argument_validation_and_total_failure(self):
        for call in (lambda: bayesian_search(bowl, SPACE2, n_iter=5, n_init=10), lambda: tpe_search(bowl, SPACE2, gamma=0.9),
                     lambda: pso_search(bowl, SPACE2, particles=1), lambda: differential_evolution(bowl, SPACE2, population=3),
                     lambda: simulated_annealing(bowl, SPACE2, cooling=1.5), lambda: cma_es(bowl, SPACE2, sigma=0),
                     lambda: optimize(bowl, SPACE2, "quantum")):
            with self.assertRaises(OptimizationError):
                call()
        with self.assertRaises(OptimizationError):
            pso_search(lambda p: 1 / 0, SPACE2, particles=4, iterations=2)            # every trial failed: say so

    def test_works_on_a_real_backtest_objective(self):
        md = MarketData()
        md.add("AAA", synthetic.gbm(400, mu=0.15, sigma=0.2, seed=2))
        space = ParameterSpace([Param("fast", "int", 3, 20), Param("slow", "int", 25, 80)])
        objective = backtest_objective(BacktestEngine(md, free_config()), lambda **p: SMACrossover(["AAA"], **p), min_trades=1)
        res = optimize(objective, space, "bayesian", n_iter=14, n_init=8)
        self.assertEqual(res.n_trials, 14)
        self.assertLess(res.best_params["fast"], res.best_params["slow"])


class MultiFidelityAndParetoTests(unittest.TestCase):
    def test_successive_halving_spends_budget_on_survivors(self):
        calls = []

        def objective(p, budget):
            calls.append(budget)
            return bowl(p) + np.random.default_rng(int(abs(p["x"]) * 1e6)).normal(0, 3.0 / budget)   # noisier at a small budget
        res = successive_halving(objective, SPACE2, n_configs=27, min_budget=1, max_budget=9, eta=3)
        self.assertEqual(res.trials.groupby("budget").size().to_dict(), {1.0: 27, 3.0: 9, 9.0: 3})
        self.assertEqual(sum(calls), 27 * 1 + 9 * 3 + 3 * 9)                          # 81 budget units instead of 243
        final = res.trials[res.trials["budget"] == 9.0]
        self.assertEqual(res.best_score, final["score"].max())
        survivors = set(zip(final["x"], final["y"]))
        first = res.trials[res.trials["budget"] == 1.0].nlargest(9, "score")
        self.assertTrue(survivors <= set(zip(first["x"], first["y"])))                # only the best of the first round moved on
        with self.assertRaises(OptimizationError):
            successive_halving(objective, SPACE2, n_configs=2, eta=3)
        with self.assertRaises(OptimizationError):
            successive_halving(lambda p, b: 1 / 0, SPACE2, n_configs=9)

    def test_pareto_front_keeps_only_undominated_rows(self):
        df = pd.DataFrame({"ret": [0.10, 0.20, 0.15, 0.05, 0.20, np.nan], "dd": [0.05, 0.20, 0.30, 0.02, 0.25, 0.01]}, index=list("abcdef"))
        front = pareto_front(df, {"ret": "max", "dd": "min"})
        self.assertEqual(list(front.index), ["a", "b", "d"])                          # c and e are beaten on both; f has no return
        self.assertEqual(list(pareto_front(df, {"ret": "max"}).index), ["b", "e"])
        with self.assertRaises(OptimizationError):
            pareto_front(df, {"ret": "up"})
        with self.assertRaises(OptimizationError):
            pareto_front(df, {"sharpe": "max"})

    def test_multi_objective_search(self):
        def evaluate(p):
            if p["y"] > 4:
                raise RuntimeError("bad region")
            return {"ret": -(p["x"] - 1) ** 2, "risk": abs(p["x"]) + abs(p["y"])}
        out = multi_objective_search(evaluate, SPACE2, {"ret": "max", "risk": "min"}, n_iter=80)
        front, trials = out["front"], out["trials"]
        self.assertTrue(0 < len(front) < len(trials))
        self.assertTrue(trials["error"].notna().any())
        ordered = front.sort_values("risk")
        self.assertTrue(ordered["ret"].is_monotonic_increasing)                       # on the front, more return always costs more risk
        with self.assertRaises(OptimizationError):
            multi_objective_search(lambda p: 1 / 0, SPACE2, {"ret": "max"}, n_iter=5)


def universe(n=500, seed=5):
    md = MarketData()
    for sym, df in synthetic.universe([f"S{i}" for i in range(12)], n=n, seed=seed).items():
        md.add(sym, df)
    md.add("SHORT", synthetic.gbm(100, seed=9))
    md.add("PENNY", synthetic.gbm(n, s0=0.5, seed=10))
    return md


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.md = universe()
        cls.asof = cls.md.frame("S0").index[400]
        cls.sectors = {f"S{i}": "tech" if i < 6 else "energy" for i in range(12)}

    def test_filters_explain_every_rejection(self):
        res = UniverseSelector().require_history(252).price_between(5, None).min_dollar_volume(1e6).volatility_between(0.0, 5.0).max_stale_share().select(self.md)
        self.assertEqual(sorted(res.symbols), sorted(f"S{i}" for i in range(12)))
        self.assertIn("history", res.rejected["SHORT"])
        self.assertIn("price", res.rejected["PENNY"])
        none = UniverseSelector().min_dollar_volume(1e15).select(self.md)
        self.assertEqual(none.symbols, [])
        self.assertEqual(len(none.rejected), 14)
        self.assertEqual(none.weights(), {})

    def test_ranking_is_a_weighted_blend_of_percentile_ranks(self):
        res = UniverseSelector().require_history(252).price_between(5, None).rank_by("momentum", lookback=126).rank_by("low_volatility", weight=0.5).select(self.md)
        t = res.scores
        np.testing.assert_allclose(t["composite"], (t["momentum_rank"] + 0.5 * t["low_volatility_rank"]) / 1.5)
        self.assertEqual(res.symbols, list(t.index))
        self.assertTrue(t["composite"].is_monotonic_decreasing)
        self.assertEqual(t["momentum"].idxmax(), t["momentum_rank"].idxmax())
        asc = UniverseSelector().require_history(252).price_between(5, None).rank_by("volatility", ascending=True).top(1).select(self.md)
        self.assertEqual(asc.symbols[0], t["low_volatility"].idxmax())                # ascending volatility == the calmest name
        top3 = UniverseSelector().require_history(252).rank_by("momentum").top(3).select(self.md)
        self.assertEqual(len(top3.symbols), 3)
        self.assertEqual(sum("cut-off" in r for r in top3.rejected.values()), len(top3.scores) - 3)
        q = UniverseSelector().require_history(252).rank_by("momentum").top_quantile(0.25).select(self.md)
        self.assertEqual(len(q.symbols), math.ceil(0.25 * len(q.scores)))

    def test_selection_is_point_in_time(self):
        sel = UniverseSelector().require_history(252).price_between(5, None).rank_by("momentum").rank_by("trend_quality").diversify(0.9).top(4)
        base = sel.select(self.md, asof=self.asof)
        future = MarketData()
        for sym in self.md.symbols:
            df = self.md.frame(sym).copy()
            df.loc[df.index > self.asof, ["open", "high", "low", "close"]] *= 3.0     # rewrite everything after the selection date
            future.add(sym, df)
        again = sel.select(future, asof=self.asof)
        self.assertEqual(again.symbols, base.symbols)
        pd.testing.assert_frame_equal(again.scores, base.scores)
        self.assertNotEqual(sel.select(future).symbols, sel.select(self.md).symbols) if sel.select(future).scores["momentum"].equals(
            sel.select(self.md).scores["momentum"]) is False else None

    def test_concentration_limits(self):
        ranked = UniverseSelector().require_history(252).price_between(5, None).rank_by("momentum").select(self.md).symbols
        capped = UniverseSelector().require_history(252).price_between(5, None).rank_by("momentum").per_group(self.sectors, 2).top(4).select(self.md)
        counts = pd.Series([self.sectors[s] for s in capped.symbols]).value_counts()
        self.assertEqual(counts.to_dict(), {"tech": 2, "energy": 2})
        self.assertTrue(any("already has 2" in r for r in capped.rejected.values()))
        twin = MarketData()
        for sym in ranked[:3]:
            twin.add(sym, self.md.frame(sym))
        twin.add("TWIN", self.md.frame(ranked[0]) * 1.0)                              # a perfect copy of the top name
        div = UniverseSelector().rank_by("momentum").diversify(0.95).select(twin)
        self.assertEqual(len({ranked[0], "TWIN"} & set(div.symbols)), 1)              # only one of the pair survives
        self.assertIn("correlation", [r for s, r in div.rejected.items() if s in (ranked[0], "TWIN")][0])

    def test_external_scores_custom_rankers_and_weights(self):
        fundamentals = {f"S{i}": float(i) for i in range(12)}                         # e.g. a factor composite computed elsewhere
        res = UniverseSelector().require_history(252).rank_by_scores("quality", fundamentals).top(3).select(self.md)
        self.assertEqual(res.symbols, ["S11", "S10", "S9"])
        self.assertIn("no score", res.rejected["PENNY"])
        custom = UniverseSelector().require_history(252).rank_by(lambda df, k=1: float(df["close"].iloc[-1]) * k, k=2).top(2).select(self.md)
        closes = {s: self.md.frame(s)["close"].iloc[-1] for s in self.md.symbols if len(self.md.frame(s)) >= 252}
        self.assertEqual(custom.symbols[0], max(closes, key=closes.get))
        w = res.weights("score")
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertGreater(w["S11"], w["S9"])
        self.assertEqual(res.weights(), {s: 1 / 3 for s in res.symbols})
        with self.assertRaises(ValidationError):
            res.weights("magic")

    def test_a_broken_ranker_rejects_only_that_symbol(self):
        def fragile(df):
            if len(df) < 200:
                raise RuntimeError("not enough data")
            return float(df["close"].iloc[-1])
        res = UniverseSelector().rank_by(fragile).select(self.md)
        self.assertIn("failed", res.rejected["SHORT"])
        self.assertEqual(len(res.symbols), 13)

    def test_rankers(self):
        up = synthetic.gbm(400, mu=0.5, sigma=0.1, seed=1)
        down = synthetic.gbm(400, mu=-0.5, sigma=0.1, seed=1)
        self.assertGreater(RANKERS["momentum"](up), 0)
        self.assertLess(RANKERS["momentum"](down), 0)
        self.assertGreater(RANKERS["trend_quality"](up), RANKERS["trend_quality"](down))
        self.assertGreater(RANKERS["near_high"](up), RANKERS["near_high"](down))
        self.assertTrue(math.isnan(RANKERS["momentum"](up.iloc[:50])))                # not enough history: no score, not a guess
        self.assertAlmostEqual(RANKERS["volatility"](up, 252), 0.1, delta=0.03)
        self.assertAlmostEqual(RANKERS["momentum"](up, 20, 0), up["close"].iloc[-1] / up["close"].iloc[-21] - 1)
        bench = up["close"]
        self.assertAlmostEqual(RANKERS["relative_strength"](up, benchmark=bench), 0.0)
        with self.assertRaises(ValidationError):
            RANKERS["relative_strength"](up)

        @ranker("test_last_close")
        def last_close(df):
            return float(df["close"].iloc[-1])
        self.assertIn("test_last_close", RANKERS)
        with self.assertRaises(ValidationError):
            UniverseSelector().rank_by("nope")

    def test_cluster_and_pair_selection(self):
        res = select_one_per_cluster(self.md, 3)
        self.assertEqual(len(res.symbols), 3)
        self.assertEqual(res.scores.loc[res.symbols, "cluster"].nunique(), 3)         # one from each cluster
        self.assertIn("SHORT", res.rejected)
        y, x = synthetic.cointegrated_pair(600, seed=2)
        md = MarketData()
        md.add("Y", y)
        md.add("X", x)
        md.add("Z", synthetic.gbm(600, seed=8))
        pairs = select_pairs(md, level="1%", top=1)
        self.assertEqual({pairs.iloc[0]["y"], pairs.iloc[0]["x"]}, {"X", "Y"})
        with self.assertRaises(ValidationError):
            select_one_per_cluster(self.md, 50)
        with self.assertRaises(ValidationError):
            UniverseSelector().select({})


def daily(mean, sd, n=1260, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(mean, sd, n), index=pd.bdate_range("2019-01-01", periods=n))


class ResultAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.good, cls.noise = daily(0.0008, 0.008), daily(0.0, 0.01, seed=1)

    def test_bootstrap_intervals_bracket_the_estimate_and_expose_luck(self):
        b = bootstrap_metrics(self.good, n_boot=400)
        for key in ("sharpe", "cagr", "max_drawdown", "volatility"):
            self.assertLessEqual(b[key]["low"], b[key]["median"])
            self.assertLessEqual(b[key]["median"], b[key]["high"])
        self.assertTrue(b["sharpe"]["low"] < b["sharpe"]["observed"] < b["sharpe"]["high"])
        self.assertGreater(b["sharpe"]["low"], 0)                                     # five years of a real edge: the interval excludes zero
        n = bootstrap_metrics(self.noise - self.noise.mean(), n_boot=400)["sharpe"]   # a sample whose mean is exactly zero
        self.assertTrue(n["low"] < 0 < n["high"])                                     # no edge: zero is inside the interval
        short = bootstrap_metrics(self.good.iloc[:120], n_boot=400)["sharpe"]
        self.assertGreater(short["high"] - short["low"], b["sharpe"]["high"] - b["sharpe"]["low"])   # less data, less certainty
        pd.testing.assert_frame_equal(pd.DataFrame(bootstrap_metrics(self.good, 100, seed=3)), pd.DataFrame(bootstrap_metrics(self.good, 100, seed=3)))
        with self.assertRaises(ValidationError):
            bootstrap_metrics(self.good.iloc[:5])

    def test_drawdown_probability(self):
        small, large = drawdown_probability(self.good, 0.05), drawdown_probability(self.good, 0.30)
        self.assertGreater(small["probability"], large["probability"])
        self.assertLess(large["probability"], 0.05)
        self.assertLessEqual(small["p95_drawdown"], small["median_drawdown"])
        risky = drawdown_probability(daily(0.0008, 0.03), 0.20)
        self.assertGreater(risky["probability"], drawdown_probability(self.good, 0.20)["probability"])
        self.assertEqual(small["horizon"], 252)
        with self.assertRaises(ValidationError):
            drawdown_probability(self.good, 1.5)

    def test_stability_separates_steady_from_one_lucky_stretch(self):
        steady = rolling_stability(self.good)
        lucky = pd.Series(np.concatenate([np.random.default_rng(2).normal(0.0, 0.006, 1000), np.random.default_rng(3).normal(0.004, 0.006, 260)]),
                          index=self.good.index)
        spike = rolling_stability(lucky)
        self.assertGreater(steady["positive_windows"], spike["positive_windows"])
        self.assertGreater(steady["equity_r2"], spike["equity_r2"])
        self.assertTrue(0 <= steady["positive_windows"] <= 1 and 0 <= steady["equity_r2"] <= 1)
        with self.assertRaises(ValidationError):
            rolling_stability(self.good.iloc[:100], 126)

    def test_regime_breakdown_shows_dependence_on_one_regime(self):
        rng = np.random.default_rng(4)
        labels = np.where(np.arange(1260) % 4 == 0, "crisis", "normal")
        r = pd.Series(np.where(labels == "crisis", rng.normal(-0.002, 0.02, 1260), rng.normal(0.0012, 0.006, 1260)), index=self.good.index)
        table = regime_breakdown(r, labels)
        self.assertLess(table.loc["crisis", "sharpe"], 0 < table.loc["normal", "sharpe"])
        self.assertAlmostEqual(table["share_of_time"].sum(), 1.0)
        self.assertAlmostEqual(table["pnl_share"].sum(), 1.0)
        self.assertGreater(table.loc["normal", "pnl_share"], 1.0)                     # all the profit, and then some, came from one regime
        self.assertEqual(int(table["bars"].sum()), 1260)
        with self.assertRaises(ValidationError):
            regime_breakdown(r, labels[:10])

    def test_trade_concentration(self):
        even = trade_concentration([10.0] * 50 + [-5.0] * 30)
        self.assertAlmostEqual(even["top_share"], 50 / 350)
        self.assertAlmostEqual(even["profit_factor"], 500 / 150)
        lottery = trade_concentration([1000.0] + [-5.0] * 100)
        self.assertGreater(lottery["top_share"], 1.0)                                 # without the one winner it loses money
        self.assertLess(lottery["pnl_without_top"], 0)
        self.assertEqual(trade_concentration([])["trades"], 0)
        self.assertEqual(trade_concentration([-1.0, -2.0])["top_share"], float("inf"))

    def test_comparison_flags_duplicates_and_deflates_for_the_number_compared(self):
        dup = self.good * 0.5 + self.noise * 0.05
        table = compare_strategies({"good": self.good, "noise": self.noise, "dup": dup})
        self.assertEqual(table.index[-1], "noise")
        self.assertGreater(table.attrs["correlation"].loc["good", "dup"], 0.9)        # two "different" strategies that are one bet
        self.assertGreater(table.loc["dup", "avg_correlation"], table.loc["noise", "avg_correlation"])
        self.assertLess(table.loc["noise", "deflated_sharpe"], 0.5)
        self.assertTrue(table["sharpe"].is_monotonic_decreasing)
        with self.assertRaises(ValidationError):
            compare_strategies({"only": self.good})


class ScorecardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.good, cls.noise = daily(0.0008, 0.008), daily(0.0, 0.01, seed=1)
        cls.trades = [100.0] * 100 + [-80.0] * 60                                     # profit factor 2.1, no dependence on a few trades

    def test_a_sound_result_passes_and_noise_fails_with_reasons(self):
        v = evaluate_strategy(self.good, self.trades)
        self.assertTrue(v.passed)
        self.assertEqual((v.grade, v.score, v.failures), ("A", 1.0, []))
        self.assertEqual({c.name for c in v.checks}, {"track_record", "sharpe", "skill_probability", "max_drawdown", "drawdown_risk", "consistency",
                                                      "trade_count", "profit_factor", "trade_concentration"})
        bad = evaluate_strategy(self.noise, list(np.random.default_rng(6).normal(0, 300, 12)))
        self.assertFalse(bad.passed)
        self.assertIn(bad.grade, ("D", "F"))
        failed = {f.split(":")[0] for f in bad.failures}
        self.assertTrue({"sharpe", "skill_probability", "trade_count"} <= failed)
        d = bad.to_dict()
        self.assertEqual((d["passed"], len(d["checks"])), (False, 9))

    def test_the_bar_rises_with_the_number_of_things_tried(self):
        one = evaluate_strategy(self.good, self.trades, n_trials=1)
        many = evaluate_strategy(self.good, self.trades, n_trials=5000)
        self.assertGreater(one.metrics["skill_probability"], many.metrics["skill_probability"])
        self.assertTrue(one.passed)
        self.assertFalse(many.passed)                                                 # the same result, found by trying 5000 variants
        self.assertIn("deflated Sharpe over 5000 trials", many.failures[0])
        with self.assertRaises(ValidationError):
            evaluate_strategy(self.good, n_trials=0)

    def test_limits_are_configurable_and_missing_evidence_fails(self):
        strict = evaluate_strategy(self.good, self.trades, limits=EvaluationLimits(min_sharpe=3.0, max_drawdown=0.01))
        self.assertEqual({f.split(":")[0] for f in strict.failures} & {"sharpe", "max_drawdown"}, {"sharpe", "max_drawdown"})
        short = evaluate_strategy(self.good.iloc[:60], self.trades)
        self.assertIn("track_record", {f.split(":")[0] for f in short.failures})
        no_trades = evaluate_strategy(self.good)
        self.assertEqual(len(no_trades.checks), 6)                                    # trade checks are absent, not silently passed
        lottery = evaluate_strategy(self.good, [5000.0] + [-10.0] * 100)
        self.assertIn("trade_concentration", {f.split(":")[0] for f in lottery.failures})

    def test_scorecard_from_a_backtest(self):
        md = MarketData()
        md.add("AAA", synthetic.gbm(750, seed=1))
        result = BacktestEngine(md, free_config()).run(SMACrossover(["AAA"]))
        v = evaluate_backtest(result, n_trials=50)
        self.assertEqual(v.metrics["n_trials"], 50)
        self.assertEqual(v.metrics["trades"]["trades"], len(result.trades))
        self.assertFalse(v.passed)                                                    # a moving-average cross on a random walk has no edge
        self.assertEqual(v.grade, evaluate_backtest(result, n_trials=50).grade)


if __name__ == "__main__":
    unittest.main()
