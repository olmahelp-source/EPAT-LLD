import os
import pickle
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))      # spawned workers must be able to import this module

import numpy as np
import pandas as pd

from helpers import free_config, market_from_closes
from algoplatform.core.exceptions import OptimizationError
from algoplatform.optimize import (Param, ParameterSpace, ParallelBacktestObjective, genetic_search, grid_search,
                                   make_parallel_objective, random_search)
from algoplatform.strategy import StrategyBuilder


def quad(p):
    return -((p["x"] - 7) ** 2) - ((p["y"] - 3) ** 2)


class SlowObjective:
    """Picklable objective that is slow for one parameter value."""

    def __call__(self, p):
        if p["x"] == 1:
            time.sleep(5)
        return float(p["x"])


class CrashObjective:
    def __call__(self, p):
        os._exit(1)                                                       # the worker process dies


def space():
    return ParameterSpace([Param("fast", "int", 3, 15), Param("slow", "int", 16, 60)])


def data():
    return market_from_closes(X=100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.001, 0.012, 350))))


class ParallelTests(unittest.TestCase):
    def setUp(self):
        self.md = data()
        self.obj = make_parallel_objective(self.md, "sma_crossover", ["X"], free_config(), min_trades=2)

    def test_objective_is_picklable_and_serial_call_works(self):
        again = pickle.loads(pickle.dumps(self.obj))
        self.assertIsNone(again._engine)                                    # engine is rebuilt per process, not shipped
        a, b = self.obj({"fast": 5, "slow": 20}), again({"fast": 5, "slow": 20})
        self.assertEqual(a, b)
        self.assertTrue(np.isfinite(a))

    def test_random_search_parallel_equals_serial_exactly(self):
        serial = random_search(self.obj, space(), n_iter=8, seed=4)
        par = random_search(self.obj, space(), n_iter=8, seed=4, n_jobs=2)
        pd.testing.assert_frame_equal(serial.trials, par.trials)             # same points, same order, same scores
        self.assertEqual((serial.best_params, serial.best_score), (par.best_params, par.best_score))

    def test_failed_trials_are_isolated_in_workers_too(self):
        sp = ParameterSpace([Param("fast", "int", 3, 30), Param("slow", "int", 10, 40)])      # no constraint: some trials invalid
        serial = random_search(self.obj, sp, n_iter=8, seed=2)
        par = random_search(self.obj, sp, n_iter=8, seed=2, n_jobs=2)
        self.assertGreater(par.summary()["failed"], 0)
        pd.testing.assert_frame_equal(serial.trials, par.trials)

    def test_genetic_and_grid_parallel_equal_serial(self):
        g1 = genetic_search(self.obj, space(), population=6, generations=2, seed=1)
        g2 = genetic_search(self.obj, space(), population=6, generations=2, seed=1, n_jobs=2)
        pd.testing.assert_frame_equal(g1.trials, g2.trials)
        small = ParameterSpace([Param("fast", "int", 4, 8, step=2), Param("slow", "int", 20, 30, step=5)])
        s1, s2 = grid_search(self.obj, small), grid_search(self.obj, small, n_jobs=2)
        pd.testing.assert_frame_equal(s1.trials, s2.trials)

    def test_rule_spec_objective_in_parallel(self):
        spec = (StrategyBuilder("rule").param("fast", 5).param("slow", 20).indicator("f", "sma", period="$fast")
                .indicator("s", "sma", period="$slow").long_entry(("f", ">", "s")).long_exit(("f", "<=", "s")).build())
        obj = ParallelBacktestObjective({s: self.md.frame(s) for s in self.md.symbols}, None, ["X"], free_config().to_dict(),
                                        min_trades=2, spec=spec)
        sp = ParameterSpace([Param("fast", "int", 3, 10), Param("slow", "int", 15, 40)])
        pd.testing.assert_frame_equal(random_search(obj, sp, 6, seed=3).trials, random_search(obj, sp, 6, seed=3, n_jobs=2).trials)

    def test_unpicklable_objective_is_rejected_with_a_clear_message(self):
        sp = ParameterSpace([Param("x", "int", 0, 14), Param("y", "int", 0, 6)])
        self.assertEqual(len(random_search(lambda p: quad(p), sp, 6, seed=0).trials), 6)      # serial path needs no pickling
        with self.assertRaisesRegex(OptimizationError, "picklable"):
            random_search(lambda p: quad(p), sp, 6, seed=0, n_jobs=2)

    def test_trial_timeout_fails_the_slow_trial_not_the_search(self):
        sp = ParameterSpace([Param("x", "int", 0, 4, step=1)])
        res = grid_search(SlowObjective(), sp, n_jobs=2, trial_timeout=1.0)
        slow = res.trials[res.trials["x"] == 1].iloc[0]
        self.assertIn("timed out", slow["error"])
        self.assertEqual(res.best_params["x"], 4)                             # the rest completed normally

    def test_worker_crash_fails_trials_without_hanging(self):
        sp = ParameterSpace([Param("x", "int", 0, 4, step=1)])
        t0 = time.time()
        with self.assertRaisesRegex(OptimizationError, "every trial failed"):
            grid_search(CrashObjective(), sp, n_jobs=2, trial_timeout=20)
        self.assertLess(time.time() - t0, 30)

    def test_constructor_validation(self):
        with self.assertRaises(OptimizationError):
            ParallelBacktestObjective({}, None, ["X"], {})
        with self.assertRaises(OptimizationError):
            ParallelBacktestObjective({}, "sma_crossover", ["X"], {}, spec={"name": "x"})


if __name__ == "__main__":
    unittest.main()
