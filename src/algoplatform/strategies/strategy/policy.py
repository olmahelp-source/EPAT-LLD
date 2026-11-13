"""PolicyStrategy: plug any model into the strategy SDK.

Machine-learning, deep-learning, reinforcement-learning and regime models all end up as the same thing: a function from
"what I can see now" to "how much I want to hold". ``PolicyStrategy`` adapts such a function to the ``Strategy``
contract, so the model inherits the platform's causal data view, sizing, risk checks, execution and backtester instead
of re-implementing them (and cannot bypass them).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ...foundation.core.exceptions import StrategyError
from ...foundation.core.models import Signal
from .base import Strategy, StrategyContext

Policy = Callable[[Dict[str, np.ndarray]], float]


class PolicyStrategy(Strategy):
    """``policy(window) -> target weight in [-1, 1]`` where ``window`` is the causal OHLCV history of one symbol.

    A policy that raises or returns a non-finite number makes the strategy stand aside for that bar (target 0) and is
    counted in ``policy_errors``; the engine's own error budget still applies on top."""
    name = "policy"
    PARAMS = {"history": 250, "warmup": 60, "step": 0.0}

    def __init__(self, symbols: Sequence[str], policy: Policy, strategy_id: Optional[str] = None, **params: Any) -> None:
        if not callable(policy):
            raise StrategyError("policy must be callable")
        self.policy = policy
        self.policy_errors = 0
        super().__init__(symbols, strategy_id, **params)

    def validate_params(self) -> None:
        if self.p("history") < 2 or not 1 <= self.p("warmup") <= self.p("history") or self.p("step") < 0:
            raise StrategyError("need history >= 2, 1 <= warmup <= history, step >= 0", details=self.params)

    @property
    def warmup(self) -> int:
        return int(self.p("warmup"))

    def clone(self, **overrides: Any) -> "PolicyStrategy":
        return PolicyStrategy(self.symbols, self.policy, self.id, **{**self.params, **overrides})

    def on_bar(self, ctx: StrategyContext) -> List[Signal]:
        out: List[Signal] = []
        step = float(self.p("step"))
        for s in self.symbols:
            if s not in ctx.bars or not self.ready(ctx, s):
                continue
            try:
                raw = float(self.policy(ctx.window(s, int(self.p("history")))))
            except Exception as exc:  # noqa: BLE001 - a failing model must not stop the other symbols
                self.policy_errors += 1
                out.append(Signal(s, 0.0, f"policy error: {type(exc).__name__}"))
                continue
            if not np.isfinite(raw):
                self.policy_errors += 1
                raw = 0.0
            w = float(np.clip(raw, -1.0, 1.0))
            out.append(Signal(s, round(w / step) * step if step > 0 else w, "policy"))
        return out


def probability_policy(predict: Callable[[np.ndarray], float], features: Callable[[Dict[str, np.ndarray]], np.ndarray],
                       edge: float = 0.05, allow_short: bool = False) -> Policy:
    """Turn a probability-of-up model into a policy: long when p > 0.5 + edge, short (if allowed) when p < 0.5 - edge,
    sized by distance from 0.5. ``features(window)`` must return the latest feature row (causal by construction, since
    the window ends at the current bar)."""
    def policy(window: Dict[str, np.ndarray]) -> float:
        x = np.asarray(features(window), float)
        if not np.isfinite(x).all():
            return 0.0
        p = float(np.ravel(predict(x.reshape(1, -1)))[0])
        if p > 0.5 + edge:
            return min(1.0, (p - 0.5) * 2)
        if p < 0.5 - edge and allow_short:
            return -min(1.0, (0.5 - p) * 2)
        return 0.0
    return policy
