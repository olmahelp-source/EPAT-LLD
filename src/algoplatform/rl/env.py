"""A single-instrument trading environment for reinforcement learning.

Timeline of one step: at the close of bar t the agent sees ``features[t]`` (and its current position), picks a target
position, holds it over bar t+1 and is paid

    reward = position * return[t+1]  -  cost * |change in position|  -  risk_aversion * (position * return[t+1])^2

so the only information available to a decision is what was known when it was made, and costs are charged on every
change. Features must be causal (built from data up to t) and should be standardised with statistics from the training
period only (see :func:`standardize`).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..core.exceptions import ValidationError


def standardize(train_features, features=None) -> np.ndarray:
    """Scale ``features`` (default: the training features themselves) with the training mean and deviation."""
    tr = np.asarray(train_features, float)
    mu, sd = np.nanmean(tr, axis=0), np.nanstd(tr, axis=0)
    return (np.asarray(tr if features is None else features, float) - mu) / np.where(sd > 1e-12, sd, 1.0)


class TradingEnv:
    def __init__(self, returns, features, actions: Sequence[float] = (-1.0, 0.0, 1.0), cost_bps: float = 5.0,
                 episode_length: Optional[int] = None, include_position: bool = True, reward_scale: float = 100.0,
                 risk_aversion: float = 0.0, seed: Optional[int] = 0) -> None:
        self.returns, self.features = np.asarray(returns, float).ravel(), np.asarray(features, float)
        if self.features.ndim != 2 or len(self.features) != len(self.returns):
            raise ValidationError("features must be 2-D with one row per return")
        if len(self.returns) < 3 or not np.isfinite(self.returns).all() or not np.isfinite(self.features).all():
            raise ValidationError("returns and features must be finite and have at least 3 rows (drop indicator warm-up rows first)")
        if len(actions) < 2 or min(actions) < -1 or max(actions) > 1:
            raise ValidationError("need at least two actions, each a target position in [-1, 1]")
        if cost_bps < 0 or risk_aversion < 0 or (episode_length is not None and episode_length < 2):
            raise ValidationError("cost_bps and risk_aversion must be >= 0 and episode_length >= 2")
        self.actions, self.cost = np.asarray(actions, float), cost_bps / 1e4
        self.episode_length, self.include_position = episode_length, include_position
        self.reward_scale, self.risk_aversion = reward_scale, risk_aversion
        self._rng = np.random.default_rng(seed)
        self._t = self._end = 0
        self.position = 0.0

    @classmethod
    def from_prices(cls, close, features, **kwargs: Any) -> "TradingEnv":
        """Build from closes and a feature matrix aligned with them; the first row (no return yet) is dropped, as are
        leading rows where any feature is NaN."""
        c, f = np.asarray(close, float), np.asarray(features, float)
        if len(c) != len(f):
            raise ValidationError("close and features must have the same length")
        r = np.concatenate([[np.nan], c[1:] / c[:-1] - 1.0])
        ok = np.isfinite(f).all(axis=1) & np.isfinite(r)
        first = int(np.argmax(ok)) if ok.any() else len(c)
        if not ok[first:].all():
            raise ValidationError("features contain NaN after their warm-up period")
        return cls(r[first:], f[first:], **kwargs)

    # ---- spaces ---------------------------------------------------------------------------------------------------------
    @property
    def n_actions(self) -> int:
        return len(self.actions)

    @property
    def obs_dim(self) -> int:
        return self.features.shape[1] + (1 if self.include_position else 0)

    def _obs(self) -> np.ndarray:
        f = self.features[self._t]
        return np.append(f, self.position) if self.include_position else f.copy()

    # ---- dynamics -----------------------------------------------------------------------------------------------------------
    def reset(self, start: Optional[int] = None) -> np.ndarray:
        """Start an episode. With ``episode_length`` set and no ``start``, a random window is drawn (decorrelates
        training); otherwise the episode runs over the whole series."""
        n = len(self.returns)
        if start is None:
            start = int(self._rng.integers(0, n - self.episode_length)) if self.episode_length and self.episode_length < n - 1 else 0
        if not 0 <= start < n - 1:
            raise ValidationError("start must leave at least one step", details={"start": start, "rows": n})
        self._t = start
        self._end = n - 1 if not self.episode_length else min(n - 1, start + self.episode_length)
        self.position = 0.0
        return self._obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if not 0 <= int(action) < self.n_actions:
            raise ValidationError(f"action must be in 0..{self.n_actions - 1}", details={"action": action})
        if self._t >= self._end:
            raise ValidationError("episode is over; call reset()")
        target = float(self.actions[int(action)])
        cost = self.cost * abs(target - self.position)
        self.position = target
        self._t += 1
        gross = target * float(self.returns[self._t])
        pnl = gross - cost
        reward = (pnl - self.risk_aversion * gross ** 2) * self.reward_scale
        done = self._t >= self._end
        return self._obs(), reward, done, {"pnl": pnl, "cost": cost, "position": target, "t": self._t}
