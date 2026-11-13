"""Genetic programming: let evolution search the space of formulas for one that predicts returns.

A candidate "alpha" is an expression tree over price fields, e.g. ``sub(ts_mean(close, 5), ts_mean(close, 20))``.
A population of random trees is evolved by tournament selection, subtree crossover and mutation; fitness is the rank
correlation between the formula's value at t and the return that followed (the information coefficient), measured on
several sub-periods so that a formula has to work consistently, minus a penalty on size.

This is the most powerful overfitting machine in the platform, and is built accordingly:

* every primitive is causal (windows end at the current bar; there is no way to read forward);
* the last ``validation_fraction`` of the data is never shown to evolution and is used once, to report ``test_ic``;
* ``n_evaluated`` is returned so the number of formulas tried can be fed to the deflated Sharpe ratio.

Expect ``test_ic`` to be much smaller than ``train_ic``. That gap is the honest result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ...foundation.core.exceptions import ValidationError
from ...quantitative.quant.indicators import rolling_max, rolling_min, rolling_std, sma
from ...signals.technical import operators as ops
from ...foundation.utils.numeric import spearman

Tree = Tuple[Any, ...]          # ("x", name) | ("const", value) | (unary, child) | (binary, left, right) | (ts_op, child, window)


def _div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > 1e-9, a / np.where(np.abs(b) > 1e-9, b, 1.0), np.nan)


BINARY: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": _div}
UNARY: Dict[str, Callable[[np.ndarray], np.ndarray]] = {"neg": np.negative, "abs": np.abs, "sign": np.sign}
TS: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "delay": lambda x, n: ops.shift(x, n), "delta": lambda x, n: ops.change(x, n), "ts_mean": sma, "ts_std": rolling_std,
    "ts_max": rolling_max, "ts_min": rolling_min, "ts_rank": ops.percent_rank,
    "ts_zscore": lambda x, n: _div(x - sma(x, n), rolling_std(x, n))}
WINDOWS = (3, 5, 10, 20, 40)


def evaluate_tree(tree: Tree, data: Mapping[str, np.ndarray]) -> np.ndarray:
    op = tree[0]
    if op == "x":
        return np.asarray(data[tree[1]], float)
    if op == "const":
        return np.full(len(next(iter(data.values()))), float(tree[1]))
    with np.errstate(all="ignore"):
        if op in UNARY:
            return UNARY[op](evaluate_tree(tree[1], data))
        if op in BINARY:
            return BINARY[op](evaluate_tree(tree[1], data), evaluate_tree(tree[2], data))
        if op in TS:
            return TS[op](evaluate_tree(tree[1], data), int(tree[2]))
    raise ValidationError(f"unknown primitive '{op}'")


def to_string(tree: Tree) -> str:
    op = tree[0]
    if op == "x":
        return str(tree[1])
    if op == "const":
        return f"{tree[1]:g}"
    if op in TS:
        return f"{op}({to_string(tree[1])}, {tree[2]})"
    return f"{op}({', '.join(to_string(c) for c in tree[1:])})"


def size(tree: Tree) -> int:
    return 1 if tree[0] in ("x", "const") else 1 + sum(size(c) for c in tree[1:] if isinstance(c, tuple))


def depth(tree: Tree) -> int:
    return 1 if tree[0] in ("x", "const") else 1 + max(depth(c) for c in tree[1:] if isinstance(c, tuple))


@dataclass
class GPResult:
    formula: str
    tree: Tree
    fitness: float
    train_ic: float
    test_ic: float
    n_evaluated: int
    history: List[float] = field(default_factory=list)                 # best fitness per generation
    hall_of_fame: List[Tuple[str, float]] = field(default_factory=list)

    def values(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        """The formula evaluated on new data, oriented so that higher means "expect a higher return"."""
        return evaluate_tree(self.tree, data) * (1.0 if self.train_ic >= 0 else -1.0)


class SymbolicAlpha:
    def __init__(self, population: int = 200, generations: int = 15, max_depth: int = 5, tournament: int = 5, crossover: float = 0.7,
                 mutation: float = 0.25, parsimony: float = 0.002, n_folds: int = 4, validation_fraction: float = 0.3, elite: int = 2,
                 seed: Optional[int] = 0) -> None:
        if population < 10 or generations < 1 or max_depth < 2 or tournament < 2 or n_folds < 2:
            raise ValidationError("need population >= 10, generations >= 1, max_depth >= 2, tournament >= 2, n_folds >= 2")
        if not 0 <= crossover <= 1 or not 0 <= mutation <= 1 or not 0.1 <= validation_fraction <= 0.5:
            raise ValidationError("crossover and mutation must be in [0, 1], validation_fraction in [0.1, 0.5]")
        self.population, self.generations, self.max_depth, self.tournament = population, generations, max_depth, tournament
        self.crossover, self.mutation, self.parsimony, self.n_folds = crossover, mutation, parsimony, n_folds
        self.validation_fraction, self.elite, self.seed = validation_fraction, elite, seed

    # ---- random trees -----------------------------------------------------------------------------------------------------
    def _random(self, rng: np.random.Generator, names: Sequence[str], max_depth: int, grow: bool = True) -> Tree:
        if max_depth <= 1 or (grow and rng.random() < 0.3):
            return ("x", names[int(rng.integers(len(names)))]) if rng.random() < 0.9 else ("const", round(float(rng.uniform(-2, 2)), 2))
        kind = rng.random()
        if kind < 0.45:
            return (list(TS)[int(rng.integers(len(TS)))], self._random(rng, names, max_depth - 1, grow), int(WINDOWS[int(rng.integers(len(WINDOWS)))]))
        if kind < 0.9:
            return (list(BINARY)[int(rng.integers(len(BINARY)))], self._random(rng, names, max_depth - 1, grow), self._random(rng, names, max_depth - 1, grow))
        return (list(UNARY)[int(rng.integers(len(UNARY)))], self._random(rng, names, max_depth - 1, grow))

    @staticmethod
    def _paths(tree: Tree, prefix: Tuple[int, ...] = ()) -> List[Tuple[int, ...]]:
        out = [prefix]
        for i, c in enumerate(tree[1:], start=1):
            if isinstance(c, tuple):
                out += SymbolicAlpha._paths(c, prefix + (i,))
        return out

    @staticmethod
    def _get(tree: Tree, path: Tuple[int, ...]) -> Tree:
        for i in path:
            tree = tree[i]
        return tree

    @staticmethod
    def _set(tree: Tree, path: Tuple[int, ...], sub: Tree) -> Tree:
        if not path:
            return sub
        return tree[:path[0]] + (SymbolicAlpha._set(tree[path[0]], path[1:], sub),) + tree[path[0] + 1:]

    def _pick(self, rng: np.random.Generator, tree: Tree) -> Tuple[int, ...]:
        paths = self._paths(tree)
        return paths[int(rng.integers(len(paths)))]

    # ---- fitness -------------------------------------------------------------------------------------------------------------
    def _fold_ics(self, values: np.ndarray, target: np.ndarray) -> np.ndarray:
        ics = []
        for idx in np.array_split(np.arange(len(target)), self.n_folds):
            v, y = values[idx], target[idx]
            ok = np.isfinite(v) & np.isfinite(y)
            if ok.sum() < 20 or np.nanstd(v[ok]) < 1e-12:
                return np.array([])                                         # undefined or constant somewhere: unusable
            ics.append(spearman(v[ok], y[ok]))
        return np.array(ics)

    def _fitness(self, tree: Tree, data: Mapping[str, np.ndarray], target: np.ndarray) -> float:
        try:
            ics = self._fold_ics(evaluate_tree(tree, data), target)
        except (ValidationError, FloatingPointError, ValueError):
            return -1.0
        if not len(ics) or not np.isfinite(ics).all():
            return -1.0
        return float(abs(ics.mean()) - 0.5 * ics.std() - self.parsimony * size(tree))      # consistent across folds, and small

    # ---- evolution -----------------------------------------------------------------------------------------------------------
    def fit(self, data: Mapping[str, Sequence[float]], target: Sequence[float]) -> GPResult:
        """``data``: name -> series (e.g. open/high/low/close/volume, or any causal features). ``target``: the forward
        return aligned with each bar (NaN where unknown, as from ``ml.forward_returns``)."""
        cols = {k: np.asarray(v, float) for k, v in data.items()}
        y = np.asarray(target, float)
        if not cols or any(len(v) != len(y) for v in cols.values()):
            raise ValidationError("every data series must have the same length as the target")
        n_val = int(len(y) * self.validation_fraction)
        n_tr = len(y) - n_val
        if n_tr < 40 * self.n_folds:
            raise ValidationError("too little data for the requested folds", details={"train_rows": n_tr})
        train = {k: v[:n_tr] for k, v in cols.items()}
        rng = np.random.default_rng(self.seed)
        names = sorted(cols)
        pop = [self._random(rng, names, int(rng.integers(2, self.max_depth + 1)), grow=bool(i % 2)) for i in range(self.population)]
        cache: Dict[Tree, float] = {}

        def score(t: Tree) -> float:
            if t not in cache:
                cache[t] = self._fitness(t, train, y[:n_tr])
            return cache[t]
        history: List[float] = []
        for _ in range(self.generations):
            fit = np.array([score(t) for t in pop])
            history.append(float(fit.max()))
            order = np.argsort(fit)[::-1]
            nxt = [pop[i] for i in order[:self.elite]]

            def select() -> Tree:
                idx = rng.integers(len(pop), size=self.tournament)
                return pop[int(idx[np.argmax(fit[idx])])]
            while len(nxt) < self.population:
                child = select()
                if rng.random() < self.crossover:
                    donor = select()
                    child = self._set(child, self._pick(rng, child), self._get(donor, self._pick(rng, donor)))
                if rng.random() < self.mutation:
                    child = self._set(child, self._pick(rng, child), self._random(rng, names, 2))
                nxt.append(child if depth(child) <= self.max_depth else select())           # reject bloat
            pop = nxt
        ranked = sorted(cache.items(), key=lambda kv: -kv[1])
        best, best_fit = ranked[0]
        train_ic = self._ic(best, train, y[:n_tr])
        test_ic = self._ic(best, cols, y, start=n_tr) * (1.0 if train_ic >= 0 else -1.0)     # oriented by the training sign
        return GPResult(to_string(best), best, float(best_fit), train_ic, test_ic, len(cache), history,
                        [(to_string(t), float(f)) for t, f in ranked[:10]])

    @staticmethod
    def _ic(tree: Tree, data: Mapping[str, np.ndarray], target: np.ndarray, start: int = 0) -> float:
        values = evaluate_tree(tree, data)[start:]                          # evaluated on full history, so windows are warm at `start`
        return spearman(values, target[start:])
