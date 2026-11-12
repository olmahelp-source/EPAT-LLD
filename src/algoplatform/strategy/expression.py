"""Formula strategies: the second strategy creator, built on the full function catalogue.

``StrategyBuilder`` (``strategy.builder``) composes rules from a fixed indicator list. This creator takes formulas in
the safe expression language of ``technical.expression``, so anything in the catalogue (70+ indicators, candlestick and
chart patterns, operators such as ``crossover`` / ``barssince`` / ``latch``) can be used without writing a class.

A spec is plain data (JSON-serialisable)::

    {"name": "trend_pullback",
     "params": {"fast": 20, "slow": 100},  "bounds": {"fast": [5, 60], "slow": [60, 250]},
     "long_entry": "ema(close, fast) > ema(close, slow) and crossover(rsi(close, 14), 40)",
     "long_exit":  "crossunder(close, ema(close, slow)) or rsi(close, 14) > 80",
     "stop_pct": 0.08, "weight": 1.0, "history": 400}

or, instead of entry / exit rules, a ``target`` formula giving the desired weight in [-1, 1] directly::

    {"name": "zscore_fade", "target": "clamp(-zscore(close, 20) / 2, -1, 1)"}
"""
from __future__ import annotations

import copy
import json
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..core.exceptions import ExpressionError, StrategyError
from ..core.models import Signal
from ..technical.expression import Expression
from .base import Strategy, StrategyContext

RULE_KEYS = ("long_entry", "long_exit", "short_entry", "short_exit")
_KNOWN = set(RULE_KEYS) | {"name", "params", "bounds", "target", "weight", "history", "warmup", "stop_pct", "step"}


def validate_expression_spec(spec: Dict[str, Any]) -> Dict[str, Expression]:
    """Check a spec and return its compiled formulas. Raises StrategyError naming the offending rule."""
    if not isinstance(spec, dict) or not spec.get("name"):
        raise StrategyError("spec needs a 'name'")
    unknown = sorted(set(spec) - _KNOWN)
    if unknown:
        raise StrategyError(f"unknown spec key(s) {unknown}", details={"valid": sorted(_KNOWN)})
    has_rules = any(spec.get(k) for k in ("long_entry", "short_entry"))
    if bool(spec.get("target")) == has_rules:
        raise StrategyError("spec needs either a 'target' formula or at least one of long_entry / short_entry (not both)")
    params = spec.get("params", {})
    for name, lh in spec.get("bounds", {}).items():
        if name not in params or len(lh) != 2 or not lh[0] < lh[1]:
            raise StrategyError(f"bounds for '{name}' must be [low, high] of a declared parameter")
    w = spec.get("weight", 1.0)
    if not isinstance(w, (int, float)) or not 0 < w <= 1:
        raise StrategyError("weight must be in (0, 1]")
    stop = spec.get("stop_pct")
    if stop is not None and not 0 < stop < 1:
        raise StrategyError("stop_pct must be in (0, 1)")
    compiled: Dict[str, Expression] = {}
    for key in RULE_KEYS + ("target",):
        if spec.get(key):
            try:
                compiled[key] = Expression(spec[key], params)
            except ExpressionError as exc:
                raise StrategyError(f"{key}: {exc.message}", details={**exc.details, "rule": key}) from exc
    return compiled


class ExpressionStrategy(Strategy):
    """Interprets a validated formula spec. One instance per run (holds per-symbol position state)."""
    name = "expression"

    def __init__(self, symbols: Sequence[str], spec: Dict[str, Any], strategy_id: Optional[str] = None, **params: Any) -> None:
        self._rules = validate_expression_spec(spec)
        self.spec = copy.deepcopy(spec)
        self.PARAMS = dict(self.spec.get("params", {}))                 # instance-level defaults
        self.PARAM_BOUNDS = {k: tuple(v) for k, v in self.spec.get("bounds", {}).items()}
        self.name = self.spec["name"]
        self._history = int(self.spec.get("history", 300))
        self._warmup = int(self.spec.get("warmup", min(self._history, 60)))
        self._state: Dict[str, Dict[str, float]] = {}
        super().__init__(symbols, strategy_id, **params)

    def validate_params(self) -> None:
        if self._history < 5 or not 1 <= self._warmup <= self._history:
            raise StrategyError("need history >= 5 and 1 <= warmup <= history")

    @property
    def warmup(self) -> int:
        return self._warmup

    def clone(self, **overrides: Any) -> "ExpressionStrategy":
        return ExpressionStrategy(self.symbols, self.spec, self.id, **{**self.params, **overrides})

    def _holds(self, key: str, window: Dict[str, np.ndarray]) -> bool:
        rule = self._rules.get(key)
        return bool(rule is not None and rule.holds(window, self.params))

    def on_bar(self, ctx: StrategyContext) -> List[Signal]:
        out: List[Signal] = []
        weight = float(self.spec.get("weight", 1.0))
        stop_pct = self.spec.get("stop_pct")
        for s in self.symbols:
            if s not in ctx.bars or not self.ready(ctx, s):
                continue
            window = ctx.window(s, self._history)
            if "target" in self._rules:
                raw = self._rules["target"].last(window, self.params)
                step = float(self.spec.get("step", 0.0))
                tgt = 0.0 if not np.isfinite(raw) else float(np.clip(raw, -1.0, 1.0))      # unknown => flat
                if step > 0:
                    tgt = round(tgt / step) * step                                         # coarse steps limit churn
                out.append(Signal(s, tgt * weight, "target"))
                continue
            st = self._state.setdefault(s, {"pos": 0.0, "entry": float("nan"), "best": float("nan")})
            pos, close, reason = st["pos"], float(window["close"][-1]), ""
            if pos != 0 and stop_pct:                                                      # trailing percentage stop
                st["best"] = max(st["best"], close) if pos > 0 else min(st["best"], close)
                if (pos > 0 and close <= st["best"] * (1 - stop_pct)) or (pos < 0 and close >= st["best"] * (1 + stop_pct)):
                    pos, reason = 0.0, "stop"
            if reason != "stop":
                if pos > 0 and self._holds("long_exit", window):
                    pos, reason = 0.0, "long_exit"
                elif pos < 0 and self._holds("short_exit", window):
                    pos, reason = 0.0, "short_exit"
                if pos == 0:
                    if self._holds("long_entry", window):
                        pos, reason = 1.0, "long_entry"
                    elif self._holds("short_entry", window):
                        pos, reason = -1.0, "short_entry"
                elif pos > 0 and "long_exit" not in self._rules and self._holds("short_entry", window):
                    pos, reason = -1.0, "reverse_short"                                    # no exit rule: the opposite entry flips
                elif pos < 0 and "short_exit" not in self._rules and self._holds("long_entry", window):
                    pos, reason = 1.0, "reverse_long"
            if pos != st["pos"]:
                st["entry"] = st["best"] = close
            st["pos"] = pos
            out.append(Signal(s, pos * weight, reason or "hold"))
        return out


def create_from_expressions(spec: Dict[str, Any], symbols: Sequence[str], strategy_id: Optional[str] = None,
                            **params: Any) -> ExpressionStrategy:
    return ExpressionStrategy(symbols, spec, strategy_id, **params)


def expression_factory(spec: Dict[str, Any], symbols: Sequence[str]) -> Callable[..., ExpressionStrategy]:
    """``f(**overrides) -> strategy`` for the optimisers."""
    validate_expression_spec(spec)
    return lambda **overrides: ExpressionStrategy(symbols, spec, **overrides)


def expression_spec_to_json(spec: Dict[str, Any]) -> str:
    validate_expression_spec(spec)
    return json.dumps(spec, indent=2, sort_keys=True)


def expression_spec_from_json(text: str) -> Dict[str, Any]:
    try:
        spec = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StrategyError(f"invalid JSON strategy spec: {exc}") from exc
    validate_expression_spec(spec)
    return spec
