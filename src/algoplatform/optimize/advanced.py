"""Advanced parameter optimisers.

Grid, random and genetic search live in ``optimizers``. The methods here spend an evaluation budget more cleverly,
which matters because one evaluation is one full backtest:

==================  ====================================================================================================
bayesian_search     Gaussian-process model of score vs parameters; evaluates where expected improvement is highest.
                    Best when evaluations are expensive and there are few (< ~10) parameters.
tpe_search          Tree-structured Parzen estimator: models "good" and "bad" regions separately. Handles categorical
                    and many parameters better than a GP.
pso_search          Particle swarm: a population that is pulled towards its own and the swarm's best points.
differential_evolution  Population search using scaled differences between members as mutations; few knobs, robust.
simulated_annealing Random local moves, accepting some worse ones early on to escape local optima.
cma_es              Evolves a full covariance matrix of the search distribution: learns which parameters interact.
successive_halving  Many candidates on a small budget (short history, few symbols), survivors get more: the core of
                    Hyperband. For when a *cheap approximation* of the objective exists.
pareto_front        Several objectives at once (return and drawdown and turnover): returns the non-dominated set.
==================  ====================================================================================================

All of them reuse the trial runner of ``optimizers`` (caching, failure isolation: a crashing trial scores -inf and the
search continues), accept an optional parameter-space constraint, are seeded, and return the same
``OptimizationResult``, so ``plateau_score`` and the deflated-Sharpe bookkeeping (number of trials, score variance)
work unchanged. A smarter optimiser finds the in-sample optimum faster; it does **not** make that optimum less
overfitted. Walk-forward validation is still required.
"""
from __future__ import annotations

import functools
import math
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.exceptions import OptimizationError
from ..stats.special import norm_cdf, norm_pdf, norm_ppf
from ..utils.numeric import rank
from .optimizers import Objective, OptimizationResult, _evaluate_safely, _py, _Runner
from .space import Param, ParameterSpace


class UnitCube:
    """Maps a parameter space to [0, 1]^d and back, so continuous optimisers can treat ints, log-scaled floats and
    choices uniformly. A choice becomes one coordinate per option (the largest wins when decoding)."""

    def __init__(self, space: ParameterSpace) -> None:
        self.space = space
        self._slices: List[Tuple[Param, slice]] = []
        d = 0
        for p in space.params:
            width = len(p.choices) if p.kind == "choice" else 1
            self._slices.append((p, slice(d, d + width)))
            d += width
        self.dim = d

    @staticmethod
    def to_unit(p: Param, v: Any) -> float:
        """A numeric parameter's value as a position in [0, 1] (log-scaled where the parameter is)."""
        if p.kind == "log":
            return (math.log(v) - math.log(p.low)) / (math.log(p.high) - math.log(p.low))
        return (v - p.low) / (p.high - p.low)

    @staticmethod
    def from_unit(p: Param, u: float) -> Any:
        u = min(max(float(u), 0.0), 1.0)
        if p.kind == "log":
            return float(math.exp(math.log(p.low) + u * (math.log(p.high) - math.log(p.low))))
        v = p.low + u * (p.high - p.low)
        if p.kind == "float" and p.step:
            v = round(round(v / p.step) * p.step, 10)
        return p.clip(v)

    def encode(self, params: Mapping[str, Any]) -> np.ndarray:
        x = np.zeros(self.dim)
        for p, sl in self._slices:
            if p.kind == "choice":
                x[sl.start + list(p.choices).index(params[p.name])] = 1.0
            else:
                x[sl] = self.to_unit(p, params[p.name])
        return x

    def decode(self, x: np.ndarray) -> Dict[str, Any]:
        x = np.clip(np.asarray(x, float), 0.0, 1.0)
        return {p.name: (p.choices[int(np.argmax(x[sl]))] if p.kind == "choice" else self.from_unit(p, x[sl][0])) for p, sl in self._slices}

    def valid_decode(self, x: np.ndarray, rng: np.random.Generator) -> Dict[str, Any]:
        """Decode; if the constraint rejects the point, substitute a random valid one."""
        params = self.decode(x)
        return params if self.space.valid(params) else self.space.sample(rng)


def _finish(run: _Runner, method: str, t0: float) -> OptimizationResult:
    return run.result(method, t0)


def _guard(fn: Callable[[_Runner], None], run: _Runner) -> None:
    try:
        fn(run)
    except BaseException:
        run.close()
        raise


def _normal_scores(scores: Sequence[float]) -> np.ndarray:
    """Rank-based normal scores: the k-th best of n maps to the matching normal quantile.

    Backtest scores are heavy-tailed (a few disastrous parameter sets), and on the raw scale those outliers swamp the
    differences among the good points that the surrogate actually needs to resolve. Ranks keep the ordering and give
    every part of the range equal resolution; failures share the lowest rank."""
    y = _finite(scores)
    return np.array([norm_ppf((r - 0.5) / len(y)) for r in rank(y)])


def _finite(scores: Sequence[float]) -> np.ndarray:
    """Scores with failures (-inf) replaced by something worse than every success, so models can still be fitted."""
    s = np.asarray(scores, float)
    ok = np.isfinite(s)
    if not ok.any():
        return np.zeros(len(s))
    floor = s[ok].min() - (s[ok].std() if ok.sum() > 1 and s[ok].std() > 0 else 1.0)
    return np.where(ok, s, floor)


# ---- Bayesian optimisation ---------------------------------------------------------------------------------------------
def _gp_posterior(X: np.ndarray, y: np.ndarray, Xs: np.ndarray, length: float, noise: float = 1e-4):
    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d2 = (a ** 2).sum(1)[:, None] - 2 * a @ b.T + (b ** 2).sum(1)[None, :]
        return np.exp(-0.5 * np.clip(d2, 0, None) / length ** 2)
    K = kernel(X, X) + noise * np.eye(len(X))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    Ks = kernel(X, Xs)
    v = np.linalg.solve(L, Ks)
    mean = Ks.T @ alpha
    var = np.clip(1.0 - (v ** 2).sum(axis=0), 1e-12, None)
    loglik = -0.5 * y @ alpha - np.log(np.diag(L)).sum()
    return mean, np.sqrt(var), float(loglik)


def bayesian_search(objective: Objective, space: ParameterSpace, n_iter: int = 40, n_init: int = 10, candidates: int = 600,
                    xi: float = 0.01, seed: Optional[int] = 0, n_jobs: int = 1, trial_timeout: Optional[float] = None) -> OptimizationResult:
    """Gaussian-process optimisation with expected improvement. ``n_init`` random points seed the model (evaluated as
    one batch, so ``n_jobs`` helps there); after that each trial is chosen using everything learned so far. The kernel
    length scale is re-selected by marginal likelihood every round."""
    if n_iter <= n_init or n_init < 2:
        raise OptimizationError("need n_iter > n_init >= 2")
    t0, rng, cube = time.time(), np.random.default_rng(seed), UnitCube(space)
    run = _Runner(objective, space, n_jobs, trial_timeout)

    def body(run: _Runner) -> None:
        points = [space.sample(rng) for _ in range(n_init)]
        scores = list(run.evaluate_many(points))
        for _ in range(n_iter - n_init):
            X = np.array([cube.encode(p) for p in points])
            yn = _normal_scores(scores)
            best_i = int(np.argmax(yn))
            pool = [space.sample(rng) for _ in range(candidates // 2)]
            pool += [{p.name: p.perturb(points[best_i][p.name], rng, 0.1) for p in space.params} for _ in range(candidates // 2)]   # exploit near the incumbent
            pool = [c for c in pool if space.valid(c) and run._key(c) not in run.cache]
            if not pool:
                break
            Xs = np.array([cube.encode(c) for c in pool])
            fitted = None
            for length in (0.1, 0.2, 0.4, 0.8):
                try:
                    m, s, ll = _gp_posterior(X, yn, Xs, length * math.sqrt(cube.dim))
                except np.linalg.LinAlgError:
                    continue
                if fitted is None or ll > fitted[2]:
                    fitted = (m, s, ll)
            if fitted is None:                                         # numerically stuck: fall back to a random probe
                choice = pool[int(rng.integers(len(pool)))]
            else:
                m, s, _ = fitted
                z = (m - yn.max() - xi) / s
                ei = (m - yn.max() - xi) * np.array([norm_cdf(v) for v in z]) + s * np.array([norm_pdf(v) for v in z])
                choice = pool[int(np.argmax(ei))]
            points.append(choice)
            scores.append(run.evaluate(choice))
    _guard(body, run)
    return _finish(run, "bayesian", t0)


# ---- TPE ---------------------------------------------------------------------------------------------------------------
def tpe_search(objective: Objective, space: ParameterSpace, n_iter: int = 60, n_init: int = 15, gamma: float = 0.25, candidates: int = 48,
               seed: Optional[int] = 0, n_jobs: int = 1, trial_timeout: Optional[float] = None) -> OptimizationResult:
    """Tree-structured Parzen estimator. Trials are split into a small "good" set (the best ``gamma * sqrt(n)``, as in
    the reference implementation) and the rest; each parameter gets a density for both groups; candidates are drawn
    from the good density and the one with the highest good/bad density ratio is evaluated.

    The good set is kept small on purpose: with a fixed share of all trials it grows with the search, its density
    centres on the middle of a cluster of merely decent points, and the search stalls there instead of following the
    best ones."""
    if n_iter <= n_init or n_init < 4 or not 0 < gamma < 0.5:
        raise OptimizationError("need n_iter > n_init >= 4 and 0 < gamma < 0.5")
    t0, rng, cube = time.time(), np.random.default_rng(seed), UnitCube(space)
    run = _Runner(objective, space, n_jobs, trial_timeout)

    def bandwidth(obs: np.ndarray) -> float:
        return max(float(obs.std()) * len(obs) ** -0.2, 0.03) if len(obs) > 1 else 0.25

    def log_density(u: float, obs: np.ndarray, bw: float) -> float:
        k = np.exp(-0.5 * ((u - obs) / bw) ** 2) / (bw * math.sqrt(2 * math.pi))
        return math.log((k.sum() + 1.0) / (len(obs) + 1.0))             # "+1": a uniform prior component keeps every region reachable

    def body(run: _Runner) -> None:
        points = [space.sample(rng) for _ in range(n_init)]
        scores = list(run.evaluate_many(points))
        for _ in range(n_iter - n_init):
            order = np.argsort(_finite(scores))[::-1]
            n_good = max(2, min(25, int(math.ceil(gamma * math.sqrt(len(points))))))     # deliberately few: see docstring
            good, bad = [points[i] for i in order[:n_good]], [points[i] for i in order[n_good:]]
            model: Dict[str, Any] = {}
            for p in space.params:                                       # one density pair per parameter
                if p.kind == "choice":
                    cg = np.array([sum(g[p.name] == c for g in good) + 1.0 for c in p.choices])
                    cb = np.array([sum(b[p.name] == c for b in bad) + 1.0 for c in p.choices])
                    model[p.name] = (cg / cg.sum(), cb / cb.sum())
                else:
                    ug = np.array([cube.to_unit(p, g[p.name]) for g in good])
                    ub = np.array([cube.to_unit(p, b[p.name]) for b in bad]) if bad else np.array([0.5])
                    model[p.name] = (ug, bandwidth(ug), ub, bandwidth(ub))
            best, best_ratio = None, -math.inf
            for _ in range(candidates):                                  # draw from the "good" density, keep the best good/bad ratio
                cand: Dict[str, Any] = {}
                ratio = 0.0
                for p in space.params:
                    if p.kind == "choice":
                        pg, pb = model[p.name]
                        k = int(rng.choice(len(p.choices), p=pg))
                        cand[p.name] = p.choices[k]
                        ratio += math.log(pg[k] / pb[k])
                    else:
                        ug, bg, ub, bb = model[p.name]
                        cand[p.name] = cube.from_unit(p, ug[int(rng.integers(len(ug)))] + rng.normal(0.0, bg))
                        u = cube.to_unit(p, cand[p.name])
                        ratio += log_density(u, ug, bg) - log_density(u, ub, bb)
                if space.valid(cand) and run._key(cand) not in run.cache and ratio > best_ratio:
                    best, best_ratio = cand, ratio
            choice = best if best is not None else space.sample(rng)
            points.append(choice)
            scores.append(run.evaluate(choice))
    _guard(body, run)
    return _finish(run, "tpe", t0)


# ---- population methods in the unit cube ---------------------------------------------------------------------------------
def pso_search(objective: Objective, space: ParameterSpace, particles: int = 12, iterations: int = 10, inertia: float = 0.7,
               cognitive: float = 1.5, social: float = 1.5, seed: Optional[int] = 0, n_jobs: int = 1,
               trial_timeout: Optional[float] = None) -> OptimizationResult:
    """Particle swarm optimisation. Each iteration evaluates the whole swarm as one batch (parallel with ``n_jobs``)."""
    if particles < 3 or iterations < 1:
        raise OptimizationError("need particles >= 3 and iterations >= 1")
    t0, rng, cube = time.time(), np.random.default_rng(seed), UnitCube(space)
    run = _Runner(objective, space, n_jobs, trial_timeout)

    def body(run: _Runner) -> None:
        pos = np.array([cube.encode(space.sample(rng)) for _ in range(particles)])
        vel = rng.uniform(-0.1, 0.1, pos.shape)
        best_pos, best_score = pos.copy(), np.full(particles, -math.inf)
        for _ in range(iterations):
            decoded = [cube.valid_decode(x, rng) for x in pos]
            scores = np.array(run.evaluate_many(decoded))
            pos = np.array([cube.encode(d) for d in decoded])              # snap to what was actually evaluated
            better = scores > best_score
            best_pos[better], best_score[better] = pos[better], scores[better]
            g = best_pos[int(np.argmax(best_score))]
            r1, r2 = rng.random(pos.shape), rng.random(pos.shape)
            vel = np.clip(inertia * vel + cognitive * r1 * (best_pos - pos) + social * r2 * (g - pos), -0.5, 0.5)
            pos = np.clip(pos + vel, 0.0, 1.0)
    _guard(body, run)
    return _finish(run, "pso", t0)


def differential_evolution(objective: Objective, space: ParameterSpace, population: int = 15, generations: int = 10, f: float = 0.7,
                           cr: float = 0.8, seed: Optional[int] = 0, n_jobs: int = 1, trial_timeout: Optional[float] = None) -> OptimizationResult:
    """DE/rand/1/bin: a trial vector is ``a + f * (b - c)`` crossed with its parent; it replaces the parent if better."""
    if population < 4 or generations < 1 or not 0 < f <= 2 or not 0 <= cr <= 1:
        raise OptimizationError("need population >= 4, generations >= 1, 0 < f <= 2, 0 <= cr <= 1")
    t0, rng, cube = time.time(), np.random.default_rng(seed), UnitCube(space)
    run = _Runner(objective, space, n_jobs, trial_timeout)

    def body(run: _Runner) -> None:
        decoded = [space.sample(rng) for _ in range(population)]
        pop = np.array([cube.encode(d) for d in decoded])
        scores = np.array(run.evaluate_many(decoded))
        for _ in range(generations):
            trials = np.empty_like(pop)
            for i in range(population):
                a, b, c = pop[rng.choice([j for j in range(population) if j != i], 3, replace=False)]
                cross = rng.random(cube.dim) < cr
                cross[int(rng.integers(cube.dim))] = True                   # at least one coordinate comes from the mutant
                trials[i] = np.where(cross, np.clip(a + f * (b - c), 0.0, 1.0), pop[i])
            decoded = [cube.valid_decode(x, rng) for x in trials]
            trial_scores = np.array(run.evaluate_many(decoded))
            win = trial_scores >= scores
            pop[win], scores[win] = np.array([cube.encode(d) for d in decoded])[win], trial_scores[win]
    _guard(body, run)
    return _finish(run, "de", t0)


def simulated_annealing(objective: Objective, space: ParameterSpace, n_iter: int = 100, t_start: float = 1.0, cooling: float = 0.95,
                        seed: Optional[int] = 0, trial_timeout: Optional[float] = None) -> OptimizationResult:
    """Local search that accepts a worse neighbour with probability exp(-loss / temperature). The temperature is in
    units of the score's own spread (estimated from the first few trials), and the step size shrinks as it cools."""
    if n_iter < 10 or t_start <= 0 or not 0 < cooling < 1:
        raise OptimizationError("need n_iter >= 10, t_start > 0, 0 < cooling < 1")
    t0, rng = time.time(), np.random.default_rng(seed)
    run = _Runner(objective, space, 1, trial_timeout)

    def body(run: _Runner) -> None:
        warm = [space.sample(rng) for _ in range(min(8, n_iter // 2))]
        ws = run.evaluate_many(warm)
        fin = [s for s in ws if math.isfinite(s)]
        scale = float(np.std(fin)) if len(fin) > 1 and np.std(fin) > 0 else 1.0
        cur, cur_score = max(zip(warm, ws), key=lambda t: t[1])
        temp = t_start
        for _ in range(n_iter - len(warm)):
            step = max(0.03, 0.25 * temp / t_start)
            cand = {p.name: (p.perturb(cur[p.name], rng, step) if rng.random() < 0.7 else cur[p.name]) for p in space.params}
            if not space.valid(cand):
                continue
            score = run.evaluate(cand)
            if score >= cur_score or (math.isfinite(score) and rng.random() < math.exp((score - cur_score) / (scale * temp))):
                cur, cur_score = cand, score
            temp = max(temp * cooling, 1e-6)
    _guard(body, run)
    return _finish(run, "annealing", t0)


def cma_es(objective: Objective, space: ParameterSpace, generations: int = 12, population: Optional[int] = None, sigma: float = 0.3,
           seed: Optional[int] = 0, n_jobs: int = 1, trial_timeout: Optional[float] = None) -> OptimizationResult:
    """Covariance matrix adaptation evolution strategy (Hansen's (mu/mu_w, lambda)-CMA-ES) in the unit cube. It learns
    the scale and correlation of the search distribution from the ranking of each generation, so it copes with
    parameters whose effects are coupled (e.g. fast and slow lookbacks) without being told."""
    if generations < 1 or sigma <= 0:
        raise OptimizationError("need generations >= 1 and sigma > 0")
    t0, rng, cube = time.time(), np.random.default_rng(seed), UnitCube(space)
    n = cube.dim
    lam = population or 4 + int(3 * math.log(n))
    if lam < 4:
        raise OptimizationError("population must be >= 4")
    mu = lam // 2
    w = math.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1.0 / (w ** 2).sum()
    cc, cs = (4 + mueff / n) / (n + 4 + 2 * mueff / n), (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
    run = _Runner(objective, space, n_jobs, trial_timeout)

    def body(run: _Runner) -> None:
        nonlocal sigma
        mean = cube.encode(space.sample(rng))
        pc, ps, C, B, D = np.zeros(n), np.zeros(n), np.eye(n), np.eye(n), np.ones(n)
        for g in range(generations):
            y = (rng.normal(size=(lam, n)) * D) @ B.T
            decoded = [cube.valid_decode(mean + sigma * yk, rng) for yk in y]
            scores = np.array(run.evaluate_many(decoded))
            order = np.argsort(_finite(list(scores)))[::-1][:mu]
            y_sel = (np.array([cube.encode(decoded[i]) for i in order]) - mean) / sigma    # steps actually taken, after clipping
            yw = w @ y_sel
            mean = np.clip(mean + sigma * yw, 0.0, 1.0)
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (B @ ((B.T @ yw) / D))
            hsig = float(np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chi < 1.4 + 2 / (n + 1))
            pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * yw
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * (y_sel.T * w) @ y_sel
            sigma = float(np.clip(sigma * math.exp((cs / damps) * (np.linalg.norm(ps) / chi - 1)), 1e-4, 1.0))
            vals, B = np.linalg.eigh((C + C.T) / 2)
            D = np.sqrt(np.clip(vals, 1e-12, None))
    _guard(body, run)
    return _finish(run, "cmaes", t0)


# ---- multi-fidelity ---------------------------------------------------------------------------------------------------------
def _with_budget(objective: Callable[[Dict[str, Any], float], float], params: Dict[str, Any], budget: float) -> float:
    return objective(params, budget)


def successive_halving(objective: Callable[[Dict[str, Any], float], float], space: ParameterSpace, n_configs: int = 27, min_budget: float = 1.0,
                       max_budget: float = 9.0, eta: int = 3, seed: Optional[int] = 0) -> OptimizationResult:
    """``objective(params, budget)`` scores a configuration using ``budget`` units of some resource (years of history,
    number of symbols, Monte-Carlo paths). All ``n_configs`` start at ``min_budget``; each round keeps the best
    ``1 / eta`` and multiplies the budget by ``eta`` until ``max_budget``. ``best_params`` is the winner at the largest
    budget reached. The trial table has one row per (configuration, budget)."""
    if n_configs < eta or eta < 2 or not 0 < min_budget <= max_budget:
        raise OptimizationError("need n_configs >= eta >= 2 and 0 < min_budget <= max_budget")
    t0, rng = time.time(), np.random.default_rng(seed)
    configs = [space.sample(rng) for _ in range(n_configs)]
    budget, rows = float(min_budget), []
    while True:
        scored = []
        for params in configs:
            score, err = _evaluate_safely(functools.partial(_with_budget, objective, budget=budget), params)
            rows.append({**params, "budget": budget, "score": score, "error": err})
            scored.append((score, params))
        scored.sort(key=lambda t: t[0], reverse=True)
        if budget >= max_budget or len(configs) <= 1:
            break
        configs = [p for s, p in scored[:max(1, len(scored) // eta)] if math.isfinite(s)] or [scored[0][1]]
        budget = min(budget * eta, float(max_budget))
    df = pd.DataFrame(rows)
    final = df[(df["budget"] == budget) & np.isfinite(df["score"])]
    if final.empty:
        raise OptimizationError("every configuration failed at the final budget", details={"first_error": str(df["error"].dropna().iloc[0])})
    top = final.loc[final["score"].idxmax()]
    return OptimizationResult(df, {n: _py(top[n]) for n in space.names}, float(top["score"]), len(df), "successive_halving", time.time() - t0)


# ---- several objectives -------------------------------------------------------------------------------------------------------
def pareto_front(trials: pd.DataFrame, objectives: Mapping[str, str]) -> pd.DataFrame:
    """Rows not dominated by any other: no other row is at least as good on every objective and better on one.
    ``objectives`` maps a column to ``"max"`` or ``"min"``, e.g. ``{"sharpe": "max", "max_drawdown": "max", "turnover": "min"}``."""
    if not objectives or any(v not in ("max", "min") for v in objectives.values()):
        raise OptimizationError("objectives must map column -> 'max' | 'min'")
    missing = [c for c in objectives if c not in trials]
    if missing:
        raise OptimizationError(f"trial table has no column(s) {missing}")
    vals = np.column_stack([trials[c].to_numpy(float) * (1.0 if d == "max" else -1.0) for c, d in objectives.items()])
    ok = np.isfinite(vals).all(axis=1)
    keep = np.zeros(len(trials), dtype=bool)
    for i in np.flatnonzero(ok):
        dominated = ((vals[ok] >= vals[i]).all(axis=1) & (vals[ok] > vals[i]).any(axis=1)).any()
        keep[i] = not dominated
    return trials[keep]


def multi_objective_search(evaluate: Callable[[Dict[str, Any]], Mapping[str, float]], space: ParameterSpace, objectives: Mapping[str, str],
                           n_iter: int = 60, seed: Optional[int] = 0) -> Dict[str, pd.DataFrame]:
    """Random search that records several metrics per trial and returns ``{"trials", "front"}``. Choosing a point on
    the front is a preference (how much return for how much drawdown), which is a human decision, so none is made here."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_iter):
        params = space.sample(rng)
        try:
            metrics = {k: float(v) for k, v in evaluate(params).items()}
            rows.append({**params, **metrics, "error": None})
        except Exception as exc:  # noqa: BLE001 - one bad trial must not end the search
            rows.append({**params, **{k: float("nan") for k in objectives}, "error": repr(exc)})
    trials = pd.DataFrame(rows)
    if trials[list(objectives)].notna().all(axis=1).sum() == 0:
        raise OptimizationError("every trial failed", details={"first_error": str(trials["error"].iloc[0])})
    return {"trials": trials, "front": pareto_front(trials, objectives)}


ADVANCED_METHODS: Dict[str, Callable[..., OptimizationResult]] = {
    "bayesian": bayesian_search, "tpe": tpe_search, "pso": pso_search, "de": differential_evolution,
    "annealing": simulated_annealing, "cmaes": cma_es}
