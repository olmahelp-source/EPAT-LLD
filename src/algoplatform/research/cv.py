"""Cross-validation for overlapping financial labels: purging, embargo, combinatorial CV, and PBO."""
from __future__ import annotations

import itertools
from math import comb
from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd

from ..core.exceptions import ValidationError


class PurgedKFold:
    """K contiguous test folds over time-ordered events.

    ``t1`` is a Series indexed by event start time (sorted) whose values are the label end times. For each test fold
    the train set drops (purges) every event whose label span overlaps the test period, and additionally drops
    ``embargo_pct`` of the sample right after the test period (serial correlation leaks information forward)."""

    def __init__(self, n_splits: int, t1: pd.Series, embargo_pct: float = 0.01) -> None:
        if n_splits < 2:
            raise ValidationError("n_splits must be >= 2")
        if not t1.index.is_monotonic_increasing:
            raise ValidationError("t1 must be indexed by sorted event times")
        if n_splits > len(t1):
            raise ValidationError("more splits than samples")
        self.n_splits, self.t1, self.embargo = n_splits, t1, int(len(t1) * embargo_pct)
        self._t0 = t1.index.to_numpy()
        self._t1 = pd.Index(t1.values).to_numpy()

    def _train_for(self, test_blocks: List[np.ndarray]) -> np.ndarray:
        n = len(self.t1)
        keep = np.ones(n, dtype=bool)
        for blk in test_blocks:
            start_t, end_t = self._t0[blk[0]], self._t1[blk].max()
            overlap = (self._t0 <= end_t) & (self._t1 >= start_t)              # any intersection with the test period
            keep &= ~overlap
            after = np.searchsorted(self._t0, end_t, side="right")
            keep[after:after + self.embargo] = False
        return np.flatnonzero(keep)

    def split(self, X=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for blk in np.array_split(np.arange(len(self.t1)), self.n_splits):
            yield self._train_for([blk]), blk


class CombinatorialPurgedCV(PurgedKFold):
    """N groups, every combination of ``k`` groups used as test. Yields many train/test paths from one history,
    so a strategy's performance becomes a distribution instead of one lucky split."""

    def __init__(self, n_splits: int, n_test_groups: int, t1: pd.Series, embargo_pct: float = 0.01) -> None:
        super().__init__(n_splits, t1, embargo_pct)
        if not 1 <= n_test_groups < n_splits:
            raise ValidationError("need 1 <= n_test_groups < n_splits")
        self.k = n_test_groups

    @property
    def n_combinations(self) -> int:
        return comb(self.n_splits, self.k)

    @property
    def n_paths(self) -> int:
        return self.n_combinations * self.k // self.n_splits

    def split(self, X=None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        groups = np.array_split(np.arange(len(self.t1)), self.n_splits)
        for combo in itertools.combinations(range(self.n_splits), self.k):
            blocks = [groups[g] for g in combo]
            yield self._train_for(blocks), np.concatenate(blocks)


def cv_scores(fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray], y: np.ndarray, cv: PurgedKFold,
              scorer: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """``fit_predict(train_idx, test_idx)`` returns predictions for the test indices; ``scorer(y_true, pred)``."""
    out = []
    for tr, te in cv.split():
        if len(tr) == 0:
            continue
        out.append(scorer(y[te], fit_predict(tr, te)))
    return np.asarray(out)


def probability_of_backtest_overfitting(perf: pd.DataFrame, n_blocks: int = 8) -> Dict[str, object]:
    """CSCV (Bailey, Borwein, Lopez de Prado, Zhu). ``perf`` is T x N per-period returns of N candidate strategies
    (every configuration you tried). For each split of the history into in-sample/out-of-sample halves, pick the best
    in-sample strategy and see where it ranks out-of-sample. PBO is the share of splits where it ranks below median."""
    if n_blocks < 4 or n_blocks % 2:
        raise ValidationError("n_blocks must be an even number >= 4")
    T, N = perf.shape
    if N < 2 or T < n_blocks * 5:
        raise ValidationError("need at least 2 strategies and 5 observations per block")
    blocks = np.array_split(np.arange(T), n_blocks)
    X = perf.to_numpy(dtype=float)

    def sharpe(rows: np.ndarray) -> np.ndarray:
        sub = X[rows]
        sd = sub.std(axis=0, ddof=1)
        return np.where(sd > 0, sub.mean(axis=0) / np.where(sd > 0, sd, 1), 0.0)

    logits, oos_of_best = [], []
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        is_rows = np.concatenate([blocks[i] for i in combo])
        oos_rows = np.concatenate([blocks[i] for i in range(n_blocks) if i not in combo])
        best = int(np.argmax(sharpe(is_rows)))
        oos_sr = sharpe(oos_rows)
        rank = (oos_sr < oos_sr[best]).sum() + 1                        # 1 = worst .. N = best
        omega = rank / (N + 1)
        logits.append(np.log(omega / (1 - omega)))
        oos_of_best.append(oos_sr[best])
    logits = np.asarray(logits)
    return {"pbo": float((logits <= 0).mean()), "logits": logits, "n_splits": len(logits),
            "mean_oos_sharpe_of_selected": float(np.mean(oos_of_best))}
