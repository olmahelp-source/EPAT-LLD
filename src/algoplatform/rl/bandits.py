"""Multi-armed bandits: learn which of several options pays best *while* using them.

In trading the "arms" are strategies, parameter sets or execution venues, and the reward is each period's (risk-
adjusted) result. A bandit shifts allocation towards what is working but keeps testing the rest, instead of committing
to last year's winner. ``discount`` < 1 forgets old evidence, which matters because edges decay.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.exceptions import ValidationError


class _Bandit:
    def __init__(self, arms: Sequence[str], discount: float = 1.0, seed: Optional[int] = 0) -> None:
        if len(arms) < 2 or len(set(arms)) != len(arms):
            raise ValidationError("need at least two distinct arms")
        if not 0 < discount <= 1:
            raise ValidationError("discount must be in (0, 1]")
        self.arms, self.discount = list(arms), discount
        self._rng = np.random.default_rng(seed)
        k = len(arms)
        self.n, self.total, self.total_sq = np.zeros(k), np.zeros(k), np.zeros(k)
        self.pulls = 0

    def _index(self, arm: str) -> int:
        try:
            return self.arms.index(arm)
        except ValueError:
            raise ValidationError(f"unknown arm '{arm}'", details={"arms": self.arms}) from None

    def update(self, arm: str, reward: float) -> None:
        if not math.isfinite(reward):
            raise ValidationError("reward must be finite")
        i = self._index(arm)
        if self.discount < 1:
            self.n *= self.discount
            self.total *= self.discount
            self.total_sq *= self.discount
        self.n[i] += 1
        self.total[i] += reward
        self.total_sq[i] += reward ** 2
        self.pulls += 1

    @property
    def means(self) -> np.ndarray:
        return np.where(self.n > 0, self.total / np.where(self.n > 0, self.n, 1.0), 0.0)

    def best(self) -> str:
        return self.arms[int(np.argmax(self.means))]

    def summary(self) -> List[Dict[str, float]]:
        return [{"arm": a, "n": float(n), "mean": float(m)} for a, n, m in zip(self.arms, self.n, self.means)]

    def select(self) -> str:
        raise NotImplementedError


class EpsilonGreedyBandit(_Bandit):
    """Play the best arm so far, except a fraction ``epsilon`` of the time when a random arm is tried."""

    def __init__(self, arms: Sequence[str], epsilon: float = 0.1, discount: float = 1.0, seed: Optional[int] = 0) -> None:
        super().__init__(arms, discount, seed)
        if not 0 <= epsilon <= 1:
            raise ValidationError("epsilon must be in [0, 1]")
        self.epsilon = epsilon

    def select(self) -> str:
        untried = np.flatnonzero(self.n == 0)
        if len(untried):
            return self.arms[int(untried[0])]
        if self._rng.random() < self.epsilon:
            return self.arms[int(self._rng.integers(len(self.arms)))]
        return self.best()


class UCB1Bandit(_Bandit):
    """Optimism under uncertainty: choose the arm with the highest mean + exploration bonus (large when rarely tried)."""

    def __init__(self, arms: Sequence[str], c: float = 2.0, discount: float = 1.0, seed: Optional[int] = 0) -> None:
        super().__init__(arms, discount, seed)
        self.c = c

    def select(self) -> str:
        untried = np.flatnonzero(self.n == 0)
        if len(untried):
            return self.arms[int(untried[0])]
        bonus = np.sqrt(self.c * math.log(max(self.n.sum(), 1.0 + 1e-9)) / self.n)
        return self.arms[int(np.argmax(self.means + bonus))]


class ThompsonBandit(_Bandit):
    """Gaussian Thompson sampling: draw a plausible mean for each arm from its posterior and play the best draw.
    :meth:`weights` turns the same posterior into capital weights (probability that each arm is the best)."""

    def __init__(self, arms: Sequence[str], prior_mean: float = 0.0, prior_strength: float = 1.0, discount: float = 1.0,
                 seed: Optional[int] = 0) -> None:
        super().__init__(arms, discount, seed)
        if prior_strength <= 0:
            raise ValidationError("prior_strength must be positive")
        self.prior_mean, self.prior_strength = prior_mean, prior_strength

    def _posterior(self):
        n = self.n + self.prior_strength
        mean = (self.total + self.prior_mean * self.prior_strength) / n
        raw_var = np.where(self.n > 1, self.total_sq / np.where(self.n > 0, self.n, 1.0) - self.means ** 2, np.nan)
        pooled = float(np.nanmean(raw_var)) if np.isfinite(raw_var).any() else 1.0
        var = np.where(np.isfinite(raw_var) & (raw_var > 0), raw_var, max(pooled, 1e-12))
        return mean, np.sqrt(var / n)

    def select(self) -> str:
        mean, sd = self._posterior()
        return self.arms[int(np.argmax(self._rng.normal(mean, sd)))]

    def weights(self, draws: int = 4000) -> Dict[str, float]:
        mean, sd = self._posterior()
        wins = np.bincount(np.argmax(self._rng.normal(mean, sd, size=(draws, len(self.arms))), axis=1), minlength=len(self.arms))
        return {a: float(w) / draws for a, w in zip(self.arms, wins)}
