"""Parameter optimisers: grid, random, genetic. Objectives are plain callables params -> score."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestEngine
from ..core.exceptions import OptimizationError, PlatformError
from ..core.logging import get_logger
from .space import ParameterSpace

log = get_logger(__name__)
Objective = Callable[[Dict[str, Any]], float]


@dataclass
class OptimizationResult:
    trials: pd.DataFrame                       # one row per evaluation: params..., score, error
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    method: str
    elapsed: float = 0.0

    def top(self, n: int = 10) -> pd.DataFrame:
        return self.trials.sort_values("score", ascending=False).head(n)

    def summary(self) -> Dict[str, Any]:
        s = self.trials["score"].replace([-np.inf, np.inf], np.nan).dropna()
        return {"method": self.method, "trials": self.n_trials, "failed": int(self.trials["error"].notna().sum()),
                "best_score": self.best_score, "best_params": self.best_params,
                "score_mean": float(s.mean()) if len(s) else float("nan"),
                "score_std": float(s.std()) if len(s) > 1 else float("nan"),
                "score_variance_for_dsr": float(s.var()) if len(s) > 1 else float("nan"), "seconds": round(self.elapsed, 2)}

    def plateau_score(self, space: ParameterSpace, params: Optional[Dict[str, Any]] = None, points: int = 5,
                      radius: float = 0.2) -> float:
        """Mean score of nearby points divided by the point's own score. Near 1 => a plateau
        (robust); near 0 => a lone spike (likely overfit). NaN when there is nothing to compare.

        Uses adjacent grid points when they were evaluated (grid search); otherwise falls back to
        evaluated trials within ``radius`` of each axis range (random / genetic search)."""
        params = params or self.best_params
        own = self._lookup(params)
        if own is None or own <= 0:
            return float("nan")
        neigh = [self._lookup(n) for n in space.neighbors(params, points)]
        neigh = [v for v in neigh if v is not None and math.isfinite(v)]
        if not neigh:
            neigh = self._nearby(space, params, radius)
        return float(np.mean(neigh) / own) if neigh else float("nan")

    def _nearby(self, space: ParameterSpace, params: Dict[str, Any], radius: float) -> List[float]:
        df = self.trials[np.isfinite(self.trials["score"])]
        mask = np.ones(len(df), dtype=bool)
        same = np.ones(len(df), dtype=bool)
        for p in space.params:
            col = df[p.name]
            if p.kind == "choice":
                mask &= (col == params[p.name]).to_numpy()
            else:
                mask &= ((col - params[p.name]).abs() <= radius * (p.high - p.low)).to_numpy()
            same &= (col == params[p.name]).to_numpy()
        return [float(v) for v in df.loc[mask & ~same, "score"]]

    def _lookup(self, params: Dict[str, Any]) -> Optional[float]:
        df = self.trials
        mask = np.ones(len(df), dtype=bool)
        for k, v in params.items():
            if k in df.columns:
                mask &= (df[k] == v).to_numpy()
        return float(df.loc[mask, "score"].iloc[0]) if mask.any() else None


def _evaluate_safely(objective: Objective, params: Dict[str, Any]):
    """-> (score, error). One bad parameter set must never kill the search, in-process or in a worker."""
    try:
        score = float(objective(params))
        if not math.isfinite(score):
            return -math.inf, "non-finite score"
        return score, None
    except PlatformError as exc:
        return -math.inf, exc.message or exc.code
    except Exception as exc:  # noqa: BLE001
        log.warning("trial %s failed: %r", params, exc)
        return -math.inf, repr(exc)


_WORKER_OBJECTIVE: Optional[Objective] = None


def _worker_init(objective: Objective) -> None:          # runs once per worker process: data is shipped once, not per trial
    global _WORKER_OBJECTIVE
    _WORKER_OBJECTIVE = objective


def _worker_eval(params: Dict[str, Any]):
    return _evaluate_safely(_WORKER_OBJECTIVE, params)


class _Runner:
    """Evaluates points with caching and failure isolation (a crashing trial scores -inf).

    ``n_jobs > 1`` evaluates each batch of points in a process pool. Rows are always recorded in the order the points were
    requested, so a parallel run produces exactly the same trial table as a serial one."""

    def __init__(self, objective: Objective, space: ParameterSpace, n_jobs: int = 1, trial_timeout: Optional[float] = None) -> None:
        self.objective, self.space = objective, space
        self.n_jobs, self.trial_timeout = max(1, int(n_jobs)), trial_timeout
        self.cache: Dict[tuple, float] = {}
        self.rows: List[Dict[str, Any]] = []
        self._pool = None

    def _key(self, params: Dict[str, Any]) -> tuple:
        return tuple(params[n] for n in self.space.names)

    def _record(self, params: Dict[str, Any], score: float, err: Optional[str]) -> None:
        self.cache[self._key(params)] = score
        self.rows.append({**params, "score": score, "error": err})

    def evaluate(self, params: Dict[str, Any]) -> float:
        return self.evaluate_many([params])[0]

    def evaluate_many(self, points: List[Dict[str, Any]]) -> List[float]:
        todo, seen = [], set()
        for p in points:
            k = self._key(p)
            if k not in self.cache and k not in seen:
                todo.append(p)
                seen.add(k)
        if todo:
            if self.n_jobs > 1 and len(todo) > 1:
                results = self._parallel(todo)
            else:
                results = [_evaluate_safely(self.objective, p) for p in todo]
            for p, (score, err) in zip(todo, results):
                self._record(p, score, err)
        return [self.cache[self._key(p)] for p in points]

    def _parallel(self, todo: List[Dict[str, Any]]):
        import concurrent.futures as cf
        import multiprocessing as mp
        import pickle
        if self._pool is None:
            try:
                pickle.dumps(self.objective)
            except Exception as exc:  # noqa: BLE001
                raise OptimizationError("objective is not picklable, so it cannot run in a process pool; use "
                                        "make_parallel_objective(...) or n_jobs=1", details={"error": repr(exc)}) from None
            try:                                              # spawn: safe even when the parent has threads (e.g. the web server)
                self._pool = cf.ProcessPoolExecutor(self.n_jobs, mp_context=mp.get_context("spawn"),
                                                    initializer=_worker_init, initargs=(self.objective,))
            except (OSError, ValueError) as exc:
                log.warning("process pool unavailable (%r); running serially", exc)
                self.n_jobs = 1
                return [_evaluate_safely(self.objective, p) for p in todo]
        futures = [self._pool.submit(_worker_eval, p) for p in todo]
        out = []
        for p, f in zip(todo, futures):
            try:
                out.append(f.result(timeout=self.trial_timeout))
            except cf.TimeoutError:
                f.cancel()
                out.append((-math.inf, f"timed out after {self.trial_timeout}s"))
            except Exception as exc:  # noqa: BLE001 - a worker crash (e.g. killed) fails the trial, not the search
                out.append((-math.inf, f"worker failed: {exc!r}"))
        return out

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def result(self, method: str, t0: float) -> OptimizationResult:
        self.close()
        df = pd.DataFrame(self.rows)
        if df.empty:
            raise OptimizationError("no trials were evaluated")
        ok = df[np.isfinite(df["score"])]
        if ok.empty:
            raise OptimizationError("every trial failed", details={"first_error": str(df["error"].iloc[0])})
        best = ok.loc[ok["score"].idxmax()]
        bp = {n: _py(best[n]) for n in self.space.names}
        return OptimizationResult(df, bp, float(best["score"]), len(df), method, time.time() - t0)


def _py(v: Any) -> Any:
    return v.item() if hasattr(v, "item") else v


def grid_search(objective: Objective, space: ParameterSpace, points: int = 5, max_trials: int = 2000, n_jobs: int = 1,
                trial_timeout: Optional[float] = None) -> OptimizationResult:
    t0 = time.time()
    if space.grid_size(points) > max_trials * 10:
        raise OptimizationError("grid too large; use random_search or reduce points",
                                details={"size": space.grid_size(points)})
    run = _Runner(objective, space, n_jobs, trial_timeout)
    pts = []
    for i, p in enumerate(space.grid(points)):
        if i >= max_trials:
            log.warning("grid truncated at %d trials", max_trials)
            break
        pts.append(p)
    try:
        run.evaluate_many(pts)
    except BaseException:
        run.close()
        raise
    return run.result("grid", t0)


def random_search(objective: Objective, space: ParameterSpace, n_iter: int = 50, seed: Optional[int] = 0, n_jobs: int = 1,
                  trial_timeout: Optional[float] = None) -> OptimizationResult:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    pts = [space.sample(rng) for _ in range(n_iter)]          # sampling is independent of results: same points serial or parallel
    run = _Runner(objective, space, n_jobs, trial_timeout)
    try:
        run.evaluate_many(pts)
    except BaseException:
        run.close()
        raise
    return run.result("random", t0)


def genetic_search(objective: Objective, space: ParameterSpace, population: int = 16, generations: int = 8,
                   elite: int = 2, mutation_rate: float = 0.3, tournament: int = 3,
                   seed: Optional[int] = 0, n_jobs: int = 1, trial_timeout: Optional[float] = None) -> OptimizationResult:
    """Simple GA: tournament selection, uniform crossover, Gaussian mutation, elitism. Each generation is evaluated as
    one batch, so ``n_jobs`` parallelises across the population."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    run = _Runner(objective, space, n_jobs, trial_timeout)
    try:
        pop = [space.sample(rng) for _ in range(population)]
        for _ in range(generations):
            scores = run.evaluate_many(pop)
            order = np.argsort(scores)[::-1]
            nxt = [pop[i] for i in order[:elite]]
            while len(nxt) < population:
                def pick():
                    idx = rng.integers(len(pop), size=tournament)
                    return pop[int(idx[np.argmax([scores[i] for i in idx])])]
                a, b = pick(), pick()
                child = {n: (a[n] if rng.random() < 0.5 else b[n]) for n in space.names}
                for p in space.params:
                    if rng.random() < mutation_rate:
                        child[p.name] = p.perturb(child[p.name], rng)
                nxt.append(child if space.valid(child) else space.sample(rng))
            pop = nxt
        run.evaluate_many(pop)
    except BaseException:
        run.close()
        raise
    return run.result("genetic", t0)


def optimize(objective: Objective, space: ParameterSpace, method: str = "random", **kwargs: Any) -> OptimizationResult:
    methods = {"grid": grid_search, "random": random_search, "genetic": genetic_search}
    if method not in methods:
        raise OptimizationError(f"unknown method '{method}'", details={"valid": sorted(methods)})
    return methods[method](objective, space, **kwargs)


# ---- backtest objectives ---------------------------------------------------------------------
def backtest_objective(engine: BacktestEngine, factory: Callable[..., Any], metric: str = "sharpe",
                       start=None, end=None, min_trades: int = 5, max_drawdown: Optional[float] = None) -> Objective:
    """Score = chosen metric of a backtest; penalised to -inf when it trades too little or
    breaches a drawdown cap (guards against 'optimal' parameters that never trade)."""
    def objective(params: Dict[str, Any]) -> float:
        res = engine.run(factory(**params), start=start, end=end)
        if len(res.trades) < min_trades:
            return -math.inf
        m = res.metrics()
        if max_drawdown is not None and m["max_drawdown"] < -abs(max_drawdown):
            return -math.inf
        if metric not in m:
            raise OptimizationError(f"unknown metric '{metric}'", details={"valid": sorted(m)})
        return float(m[metric])
    return objective
