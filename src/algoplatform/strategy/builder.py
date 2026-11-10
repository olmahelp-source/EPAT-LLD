"""Custom strategy creator: define a strategy as data (JSON-serialisable), no code, no eval.

Example::

    spec = (StrategyBuilder("trend_rsi")
            .param("fast", 10).param("slow", 40)
            .indicator("fast", "sma", period="$fast")
            .indicator("slow", "sma", period="$slow")
            .indicator("rsi", "rsi", period=14)
            .indicator("atr", "atr", period=14)
            .long_entry(("fast", "crosses_above", "slow"), ("rsi", "<", 70))
            .long_exit(("fast", "crosses_below", "slow"))
            .stop_atr("atr", 2.5)
            .build())
    strat = create_from_spec(spec, ["AAPL"], fast=12)

Rules are lists of ``[left, op, right]`` triples joined by AND; wrap in ``{"any": [...]}``
for OR or ``{"all": [...]}`` for nesting. Operands are indicator names, price fields
(open/high/low/close/volume), numbers, or ``"name@k"`` for the value k bars ago.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..core.exceptions import StrategyError
from ..core.models import Signal
from ..quant import indicators as ind
from .base import Strategy, StrategyContext

PRICE_FIELDS = {"open", "high", "low", "close", "volume"}
OPS = {">", "<", ">=", "<=", "==", "!=", "crosses_above", "crosses_below"}

# indicator type -> (function, input columns, allowed outputs or None, required params, optional params)
_SPECS: Dict[str, Dict[str, Any]] = {
    "sma": dict(fn=ind.sma, inputs=["source"], req=["period"]),
    "ema": dict(fn=ind.ema, inputs=["source"], req=["period"]),
    "wma": dict(fn=ind.wma, inputs=["source"], req=["period"]),
    "rsi": dict(fn=ind.rsi, inputs=["source"], req=["period"]),
    "roc": dict(fn=ind.roc, inputs=["source"], req=["period"]),
    "zscore": dict(fn=ind.zscore, inputs=["source"], req=["period"]),
    "rolling_std": dict(fn=ind.rolling_std, inputs=["source"], req=["period"]),
    "rolling_max": dict(fn=ind.rolling_max, inputs=["source"], req=["period"]),
    "rolling_min": dict(fn=ind.rolling_min, inputs=["source"], req=["period"]),
    "atr": dict(fn=ind.atr, inputs=["high", "low", "close"], req=["period"]),
    "bollinger": dict(fn=ind.bollinger, inputs=["source"], req=["period"], opt=["k"],
                      outputs=["mid", "upper", "lower", "pct_b", "bandwidth"]),
    "macd": dict(fn=ind.macd, inputs=["source"], req=["fast", "slow", "signal"],
                 outputs=["macd", "signal", "hist"]),
    "donchian": dict(fn=ind.donchian, inputs=["high", "low"], req=["period"], outputs=["upper", "lower", "mid"]),
    "stochastic": dict(fn=ind.stochastic, inputs=["high", "low", "close"], req=["k"], opt=["d"],
                       outputs=["k", "d"]),
    "adx": dict(fn=ind.adx, inputs=["high", "low", "close"], req=["period"],
                outputs=["adx", "plus_di", "minus_di"]),
}
_REF = re.compile(r"^\$([A-Za-z_]\w*)$")
_LAG = re.compile(r"^([A-Za-z_]\w*)@(\d+)$")


def _resolve(value: Any, params: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        m = _REF.match(value)
        if m:
            if m.group(1) not in params:
                raise StrategyError(f"unknown parameter reference {value}")
            return params[m.group(1)]
    return value


def validate_spec(spec: Dict[str, Any]) -> None:
    """Static validation with precise messages; raises StrategyError."""
    if not isinstance(spec, dict) or not spec.get("name"):
        raise StrategyError("spec needs a 'name'")
    inds = spec.get("indicators", {})
    params = spec.get("params", {})
    for name, d in inds.items():
        if name in PRICE_FIELDS:
            raise StrategyError(f"indicator name '{name}' clashes with a price field")
        t = d.get("type")
        if t not in _SPECS:
            raise StrategyError(f"indicator '{name}': unknown type '{t}'", details={"valid": sorted(_SPECS)})
        sp = _SPECS[t]
        for req in sp["req"]:
            if req not in d:
                raise StrategyError(f"indicator '{name}' ({t}) missing '{req}'")
        if "source" in sp["inputs"] and d.get("source", "close") not in PRICE_FIELDS:
            raise StrategyError(f"indicator '{name}': bad source '{d.get('source')}'")
        if "outputs" in sp and d.get("output", sp["outputs"][0]) not in sp["outputs"]:
            raise StrategyError(f"indicator '{name}': output must be one of {sp['outputs']}")
        for k, v in d.items():
            if isinstance(v, str) and _REF.match(v) and _REF.match(v).group(1) not in params:
                raise StrategyError(f"indicator '{name}' references undefined parameter {v}")

    known = set(inds) | PRICE_FIELDS

    def check_node(node: Any, where: str) -> None:
        if isinstance(node, dict):
            if set(node) - {"any", "all"} or len(node) != 1:
                raise StrategyError(f"{where}: dict rule must have exactly one of 'any'/'all'")
            for child in next(iter(node.values())):
                check_node(child, where)
            return
        if not (isinstance(node, (list, tuple)) and len(node) == 3):
            raise StrategyError(f"{where}: condition must be [left, op, right], got {node!r}")
        left, op, right = node
        if op not in OPS:
            raise StrategyError(f"{where}: unknown operator '{op}'", details={"valid": sorted(OPS)})
        for side in (left, right):
            if isinstance(side, (int, float)) and not isinstance(side, bool):
                continue
            if isinstance(side, str):
                if _REF.match(side):
                    if _REF.match(side).group(1) not in params:
                        raise StrategyError(f"{where}: undefined parameter {side}")
                    continue
                m = _LAG.match(side)
                base = m.group(1) if m else side
                if base not in known:
                    raise StrategyError(f"{where}: unknown operand '{side}'", details={"known": sorted(known)})
                continue
            raise StrategyError(f"{where}: bad operand {side!r}")

    rule_keys = ("long_entry", "long_exit", "short_entry", "short_exit")
    if not any(spec.get(k) for k in ("long_entry", "short_entry")):
        raise StrategyError("spec needs at least one of long_entry / short_entry")
    for key in rule_keys:
        for cond in spec.get(key, []):
            check_node(cond, key)
    stop = spec.get("stop_atr")
    if stop and stop.get("indicator") not in inds:
        raise StrategyError("stop_atr.indicator must be a defined indicator")
    w = spec.get("weight", 1.0)
    if not isinstance(w, (int, float)) or not 0 < w <= 1:
        raise StrategyError("weight must be in (0, 1]")


class RuleBasedStrategy(Strategy):
    """Interprets a validated spec. One instance per run (holds per-symbol state)."""
    name = "rule_based"

    def __init__(self, symbols: Sequence[str], spec: Dict[str, Any], strategy_id: Optional[str] = None,
                 **params: Any) -> None:
        validate_spec(spec)
        self.spec = copy.deepcopy(spec)
        self.PARAMS = dict(self.spec.get("params", {}))        # instance-level defaults
        self.name = self.spec["name"]
        self.PARAM_BOUNDS = {k: tuple(v) for k, v in self.spec.get("bounds", {}).items()}
        self._state: Dict[str, Dict[str, float]] = {}
        super().__init__(symbols, strategy_id, **params)
        self._indicators = self._resolve_indicators()
        self._history_len = max(self.spec.get("history", 0), self._max_period() * 3 + 50)

    # ---- spec resolution --------------------------------------------------------------------
    def _resolve_indicators(self) -> Dict[str, Dict[str, Any]]:
        out = {}
        for name, d in self.spec.get("indicators", {}).items():
            r = {k: _resolve(v, self.params) for k, v in d.items()}
            int_keys = {"period", "fast", "slow", "signal"}
            if r["type"] == "stochastic":
                int_keys |= {"k", "d"}                      # for bollinger "k" stays a float multiplier
            for k in int_keys & set(r):
                r[k] = int(r[k])
                if r[k] < 1:
                    raise StrategyError(f"indicator '{name}': {k} must be >= 1")
            out[name] = r
        return out

    def clone(self, **overrides: Any) -> "RuleBasedStrategy":
        return RuleBasedStrategy(self.symbols, self.spec, self.id, **{**self.params, **overrides})

    def _max_period(self) -> int:
        vals = [int(v) for d in self._indicators.values() for k, v in d.items()
                if k in ("period", "slow", "k") and isinstance(v, (int, float))]
        return max(vals) if vals else 1

    @property
    def warmup(self) -> int:
        return self._max_period() + 2

    # ---- evaluation ----------------------------------------------------------------------------
    def _compute(self, w: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        series: Dict[str, np.ndarray] = {f: w[f] for f in PRICE_FIELDS}
        for name, d in self._indicators.items():
            sp = _SPECS[d["type"]]
            args = [w[d.get("source", "close")] if i == "source" else w[i] for i in sp["inputs"]]
            kwargs = {k: d[k] for k in sp["req"] + sp.get("opt", []) if k in d}
            if d["type"] in ("stochastic",):
                kwargs = {"k": d["k"], **({"d": d["d"]} if "d" in d else {})}
            if d["type"] in ("sma", "ema", "wma", "rsi", "roc", "zscore", "rolling_std", "rolling_max",
                             "rolling_min", "atr", "donchian", "adx", "bollinger"):
                kwargs["n"] = kwargs.pop("period")
            res = sp["fn"](*args, **kwargs)
            series[name] = res[d.get("output", sp["outputs"][0])] if "outputs" in sp else res
        return series

    @staticmethod
    def _value(operand: Any, series: Dict[str, np.ndarray], params: Dict[str, Any], shift: int = 0) -> float:
        if isinstance(operand, (int, float)):
            return float(operand)
        operand = _resolve(operand, params)
        if isinstance(operand, (int, float)):
            return float(operand)
        m = _LAG.match(operand)
        name, lag = (m.group(1), int(m.group(2))) if m else (operand, 0)
        arr = series[name]
        idx = -1 - lag - shift
        return float(arr[idx]) if len(arr) >= -idx else float("nan")

    def _holds(self, node: Any, series: Dict[str, np.ndarray]) -> bool:
        if isinstance(node, dict):
            kind, children = next(iter(node.items()))
            results = (self._holds(c, series) for c in children)
            return any(results) if kind == "any" else all(results)
        left, op, right = node
        l0, r0 = self._value(left, series, self.params), self._value(right, series, self.params)
        if np.isnan(l0) or np.isnan(r0):
            return False                                      # unknown => condition is not met
        if op == ">":
            return l0 > r0
        if op == "<":
            return l0 < r0
        if op == ">=":
            return l0 >= r0
        if op == "<=":
            return l0 <= r0
        if op == "==":
            return l0 == r0
        if op == "!=":
            return l0 != r0
        l1, r1 = self._value(left, series, self.params, 1), self._value(right, series, self.params, 1)
        if np.isnan(l1) or np.isnan(r1):
            return False
        return (l1 <= r1 and l0 > r0) if op == "crosses_above" else (l1 >= r1 and l0 < r0)

    def _all(self, key: str, series: Dict[str, np.ndarray]) -> bool:
        rules = self.spec.get(key, [])
        return bool(rules) and all(self._holds(r, series) for r in rules)

    def on_bar(self, ctx: StrategyContext) -> List[Signal]:
        out: List[Signal] = []
        weight = float(self.spec.get("weight", 1.0))
        stop_cfg = self.spec.get("stop_atr")
        for s in self.symbols:
            if s not in ctx.bars or not self.ready(ctx, s):
                continue
            series = self._compute(ctx.window(s, self._history_len))
            st = self._state.setdefault(s, {"pos": 0.0, "stop": float("nan")})
            close = series["close"][-1]
            pos = st["pos"]
            reason = ""
            if pos != 0 and stop_cfg:                           # trailing ATR stop, ratcheted in favour
                a = self._value(stop_cfg["indicator"], series, self.params)
                if np.isfinite(a):
                    mult = float(_resolve(stop_cfg.get("multiple", 2.0), self.params))
                    new_stop = close - pos * mult * a
                    st["stop"] = new_stop if np.isnan(st["stop"]) else (max(st["stop"], new_stop) if pos > 0 else min(st["stop"], new_stop))
                    if (pos > 0 and close <= st["stop"]) or (pos < 0 and close >= st["stop"]):
                        pos, reason = 0.0, "stop"
            if reason != "stop":
                if pos > 0 and (self._all("long_exit", series) or (self._all("short_entry", series) and not self.spec.get("long_exit"))):
                    pos, reason = 0.0, "long_exit"
                elif pos < 0 and (self._all("short_exit", series) or (self._all("long_entry", series) and not self.spec.get("short_exit"))):
                    pos, reason = 0.0, "short_exit"
                if pos == 0 and reason != "stop":
                    if self._all("long_entry", series):
                        pos, reason = 1.0, "long_entry"
                    elif self._all("short_entry", series):
                        pos, reason = -1.0, "short_entry"
            if pos != st["pos"]:
                st["stop"] = float("nan")
            st["pos"] = pos
            out.append(Signal(s, pos * weight, reason or "hold"))
        return out


class StrategyBuilder:
    """Fluent helper that produces (and validates) a spec dict."""

    def __init__(self, name: str) -> None:
        self._spec: Dict[str, Any] = {"name": name, "params": {}, "indicators": {}, "bounds": {}}

    def param(self, name: str, default: Any, bounds: Optional[Tuple[float, float]] = None) -> "StrategyBuilder":
        self._spec["params"][name] = default
        if bounds:
            self._spec["bounds"][name] = list(bounds)
        return self

    def indicator(self, name: str, type_: str, **kwargs: Any) -> "StrategyBuilder":
        self._spec["indicators"][name] = {"type": type_, **kwargs}
        return self

    def _rules(self, key: str, rules: Sequence[Any]) -> "StrategyBuilder":
        self._spec.setdefault(key, []).extend(list(r) if isinstance(r, tuple) else r for r in rules)
        return self

    def long_entry(self, *rules: Any) -> "StrategyBuilder":
        return self._rules("long_entry", rules)

    def long_exit(self, *rules: Any) -> "StrategyBuilder":
        return self._rules("long_exit", rules)

    def short_entry(self, *rules: Any) -> "StrategyBuilder":
        return self._rules("short_entry", rules)

    def short_exit(self, *rules: Any) -> "StrategyBuilder":
        return self._rules("short_exit", rules)

    def stop_atr(self, indicator: str, multiple: Union[float, str] = 2.0) -> "StrategyBuilder":
        self._spec["stop_atr"] = {"indicator": indicator, "multiple": multiple}
        return self

    def weight(self, w: float) -> "StrategyBuilder":
        self._spec["weight"] = w
        return self

    def build(self) -> Dict[str, Any]:
        validate_spec(self._spec)
        return copy.deepcopy(self._spec)


def create_from_spec(spec: Dict[str, Any], symbols: Sequence[str], strategy_id: Optional[str] = None,
                     **params: Any) -> RuleBasedStrategy:
    return RuleBasedStrategy(symbols, spec, strategy_id, **params)


def spec_factory(spec: Dict[str, Any], symbols: Sequence[str]) -> Callable[..., RuleBasedStrategy]:
    validate_spec(spec)
    return lambda **overrides: RuleBasedStrategy(symbols, spec, **overrides)


def spec_to_json(spec: Dict[str, Any]) -> str:
    validate_spec(spec)
    return json.dumps(spec, indent=2, sort_keys=True)


def spec_from_json(text: str) -> Dict[str, Any]:
    try:
        spec = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StrategyError(f"invalid JSON strategy spec: {exc}") from exc
    validate_spec(spec)
    return spec
