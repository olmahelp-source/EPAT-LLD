"""One catalogue for every function a rule may call: indicators, patterns and operators.

The catalogue is what lets the rest of the platform treat "an indicator" as data instead of code:

* :func:`compute` runs any indicator by name against an OHLCV frame, picking the right input columns itself;
* :func:`indicator_frame` builds a whole feature table in one call, isolating a failing column instead of losing the lot;
* the formula engine (``technical.expression``) resolves function names here, so a new indicator registered with
  :func:`register_function` is immediately usable in strategy formulas, screens and alerts;
* :func:`describe` is the menu a UI shows.
"""
from __future__ import annotations

import difflib
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...foundation.core.exceptions import ConfigError, MissingDataError, PlatformError, ValidationError
from ...quantitative.quant import indicators as core
from . import bands, candles, chart_patterns, levels, momentum, operators, trend, volume

PRICE_FIELDS = ("open", "high", "low", "close", "volume")
REQUIRED = inspect.Parameter.empty
_OHLC, _HLC, _HL, _C, _CV, _HLCV = ("open", "high", "low", "close"), ("high", "low", "close"), ("high", "low"), ("close",), \
    ("close", "volume"), ("high", "low", "close", "volume")


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    fn: Callable[..., Any]
    kind: str                           # indicator | pattern | operator
    category: str
    inputs: Tuple[str, ...]             # price columns passed positionally by ``compute`` (empty for operators)
    params: Mapping[str, Any]           # tunable parameters -> default (REQUIRED when there is none)
    outputs: Tuple[str, ...]            # names of the series returned; empty for a single series
    doc: str

    def signature(self) -> str:
        args = list(self.inputs) + [k if v is REQUIRED else f"{k}={v!r}" for k, v in self.params.items()]
        return f"{self.name}({', '.join(args)})" + (f" -> {{{', '.join(self.outputs)}}}" if self.outputs else "")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "category": self.category, "inputs": list(self.inputs),
                "params": {k: (None if v is REQUIRED else v) for k, v in self.params.items()},
                "required": [k for k, v in self.params.items() if v is REQUIRED], "outputs": list(self.outputs),
                "signature": self.signature(), "doc": self.doc}


CATALOG: Dict[str, FunctionSpec] = {}


def register_function(name: str, fn: Callable[..., Any], kind: str = "indicator", category: str = "custom",
                      inputs: Sequence[str] = _C, outputs: Sequence[str] = (), overwrite: bool = False) -> FunctionSpec:
    """Add a function to the catalogue. ``inputs`` are the price columns it takes first, in order; every remaining
    parameter becomes a tunable. Operators take explicit arguments, so they declare no inputs."""
    if not name.isidentifier() or name.startswith("_") or name in PRICE_FIELDS:
        raise ConfigError(f"'{name}' is not a usable function name")
    if kind not in ("indicator", "pattern", "operator"):
        raise ConfigError(f"unknown function kind '{kind}'")
    if name in CATALOG and not overwrite:
        raise ConfigError(f"function '{name}' already registered")
    bad = [c for c in inputs if c not in PRICE_FIELDS]
    if bad:
        raise ConfigError(f"inputs must be price fields, got {bad}")
    sig = list(inspect.signature(fn).parameters.values())
    positional = [p for p in sig if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if len(positional) < len(inputs):
        raise ConfigError(f"'{name}' takes fewer arguments than its {len(inputs)} declared inputs")
    params = {p.name: p.default for p in sig[len(inputs):] if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    doc = (inspect.getdoc(fn) or "").split("\n\n")[0].replace("\n", " ")
    spec = FunctionSpec(name, fn, kind, category, tuple(inputs), params, tuple(outputs), doc)
    CATALOG[name] = spec
    return spec


def _many(module: Any, category: str, items: Mapping[str, Tuple[Sequence[str], Sequence[str]]], kind: str = "indicator") -> None:
    for fn_name, (inputs, outputs) in items.items():
        register_function(fn_name.rstrip("_") if kind != "operator" else fn_name, getattr(module, fn_name), kind, category, inputs, outputs)


_many(core, "average", {"sma": (_C, ()), "ema": (_C, ()), "wma": (_C, ())})
_many(core, "statistic", {"zscore": (_C, ()), "rolling_std": (_C, ()), "rolling_max": (_C, ()), "rolling_min": (_C, ())})
_many(core, "momentum", {"rsi": (_C, ()), "roc": (_C, ()), "macd": (_C, ("macd", "signal", "hist")), "stochastic": (_HLC, ("k", "d"))})
_many(core, "volatility", {"atr": (_HLC, ()), "true_range": (_HLC, ()), "bollinger": (_C, ("mid", "upper", "lower", "pct_b", "bandwidth")),
                           "donchian": (_HL, ("upper", "lower", "mid"))})
_many(core, "trend", {"adx": (_HLC, ("adx", "plus_di", "minus_di"))})
_many(core, "volume", {"vwap": (_CV, ()), "obv": (_CV, ())})
_many(trend, "average", {"dema": (_C, ()), "tema": (_C, ()), "hma": (_C, ()), "kama": (_C, ()), "vwma": (_CV, ())})
_many(trend, "statistic", {"rolling_sum": (_C, ()), "linreg": (_C, ("slope", "value", "r2"))})
_many(trend, "trend", {"supertrend": (_HLC, ("line", "direction")), "parabolic_sar": (_HL, ("sar", "direction")),
                       "ichimoku": (_HLC, ("tenkan", "kijun", "span_a", "span_b", "cloud_position")),
                       "aroon": (_HL, ("up", "down", "oscillator")), "trix": (_C, ()), "vortex": (_HLC, ("plus", "minus"))})
_many(momentum, "momentum", {"momentum": (_C, ()), "cci": (_HLC, ()), "williams_r": (_HLC, ()), "stoch_rsi": (_C, ("k", "d")), "tsi": (_C, ()),
                             "ultimate_oscillator": (_HLC, ()), "ppo": (_C, ("ppo", "signal", "hist")), "cmo": (_C, ()),
                             "awesome_oscillator": (_HL, ())})
_many(bands, "volatility", {"keltner": (_HLC, ("mid", "upper", "lower")), "chandelier_exit": (_HLC, ("long_stop", "short_stop")),
                            "natr": (_HLC, ()), "choppiness": (_HLC, ()), "squeeze": (_HLC, ())})
_many(volume, "volume", {"mfi": (_HLCV, ()), "ad_line": (_HLCV, ()), "cmf": (_HLCV, ()), "force_index": (_CV, ()),
                         "ease_of_movement": (("high", "low", "volume"), ()), "pvt": (_CV, ()), "relative_volume": (("volume",), ())})
_many(levels, "structure", {"pivot_points": (_HLC, ("pivot", "r1", "s1", "r2", "s2", "r3", "s3")),
                            "heikin_ashi": (_OHLC, ("open", "high", "low", "close")), "last_swing": (_HL, ())})
_many(candles, "candlestick", {name: (_OHLC, ()) for name in candles.CANDLE_PATTERNS}, kind="pattern")
_many(chart_patterns, "chart", {"range_breakout": (_HLC, ())}, kind="pattern")
_many(operators, "compare", {k: ((), ()) for k in ("above", "below", "above_or_equal", "between", "crossover", "crossunder", "cross")}, "operator")
_many(operators, "logic", {k: ((), ()) for k in ("and_", "or_", "not_", "where", "nz")}, "operator")
_many(operators, "lookback", {k: ((), ()) for k in ("shift", "change", "pct_change", "highest", "lowest", "rising", "falling", "count_true",
                                                    "all_true", "any_true", "streak", "barssince", "valuewhen", "rising_edge", "latch",
                                                    "exrem")}, "operator")
_many(operators, "scale", {k: ((), ()) for k in ("percent_rank", "normalize", "clamp", "abs_", "sign", "max_", "min_", "log", "sqrt",
                                                 "cummax", "cummin")}, "operator")
for _alias, _target in (("abs", "abs_"), ("max", "max_"), ("min", "min_"), ("if_else", "where")):   # friendlier names inside formulas
    register_function(_alias, CATALOG[_target].fn, "operator", CATALOG[_target].category, (), ())


# ---- lookup ---------------------------------------------------------------------------------------------------------------
def suggest(name: str, n: int = 3) -> List[str]:
    return difflib.get_close_matches(name, list(CATALOG), n=n, cutoff=0.6)


def get_function(name: str) -> FunctionSpec:
    try:
        return CATALOG[name]
    except KeyError:
        raise ValidationError(f"unknown function '{name}'", details={"did_you_mean": suggest(str(name))}) from None


def describe(kind: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Catalogue entries as plain dicts (for a UI, docs or ``python -m algoplatform functions``)."""
    return [s.to_dict() for s in sorted(CATALOG.values(), key=lambda s: (s.kind, s.category, s.name))
            if (kind is None or s.kind == kind) and (category is None or s.category == category)]


# ---- execution ------------------------------------------------------------------------------------------------------------
def _column(data: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in data:
        raise MissingDataError(f"data has no '{name}' column", details={"available": [k for k in PRICE_FIELDS if k in data]})
    return np.asarray(data[name], float)


def compute(name: str, data: Mapping[str, Any], *, output: Optional[str] = None, source: Optional[str] = None, **params: Any):
    """Run indicator / pattern ``name`` on an OHLCV frame (or a mapping of arrays).

    ``source`` replaces the input column of single-input indicators (``compute("sma", df, source="volume", n=20)``).
    Multi-output indicators return a dict unless ``output`` names one series."""
    spec = get_function(name)
    if not spec.inputs:
        raise ValidationError(f"'{name}' is an operator: it takes explicit arguments, use call('{name}', ...)")
    unknown = sorted(set(params) - set(spec.params))
    if unknown:
        raise ValidationError(f"{name}: unknown parameter(s) {unknown}", details={"signature": spec.signature()})
    missing = [k for k, v in spec.params.items() if v is REQUIRED and k not in params]
    if missing:
        raise ValidationError(f"{name}: missing parameter(s) {missing}", details={"signature": spec.signature()})
    columns = spec.inputs
    if source is not None:
        if len(spec.inputs) != 1:
            raise ValidationError(f"{name} takes {list(spec.inputs)}; 'source' only applies to single-input indicators")
        columns = (source,)
    result = call(name, *[_column(data, c) for c in columns], **params)
    if isinstance(result, dict):
        if output is None:
            return result
        if output not in result:
            raise ValidationError(f"{name} has no output '{output}'", details={"outputs": list(result)})
        return result[output]
    if output is not None:
        raise ValidationError(f"{name} returns a single series; it has no output '{output}'")
    return result


def call(name: str, *args: Any, **kwargs: Any) -> Any:
    """Call any catalogue function with explicit arguments, turning unexpected failures into a ``ValidationError``
    that names the function (bad arguments are by far the most common cause)."""
    spec = get_function(name)
    try:
        return spec.fn(*args, **kwargs)
    except PlatformError as exc:
        exc.details.setdefault("function", name)
        raise
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"{name} failed: {exc}", details={"function": name, "signature": spec.signature(),
                                                                "cause": type(exc).__name__}) from exc


def indicator_frame(data: Mapping[str, Any], columns: Mapping[str, Any], on_error: str = "raise") -> pd.DataFrame:
    """Build a feature table. ``columns`` maps an output column to ``"name"`` or ``("name", {params})`` where params may
    include ``output`` / ``source``::

        indicator_frame(df, {"rsi": "rsi", "fast": ("ema", {"n": 10}), "hist": ("macd", {"output": "hist"})})

    With ``on_error="nan"`` a failing column is filled with NaN and reported in ``frame.attrs["errors"]`` so that one
    bad definition does not cost the rest of the table."""
    if on_error not in ("raise", "nan"):
        raise ValidationError("on_error must be 'raise' or 'nan'")
    n = len(_column(data, "close"))
    out: Dict[str, np.ndarray] = {}
    errors: Dict[str, str] = {}
    for col, definition in columns.items():
        name, params = (definition, {}) if isinstance(definition, str) else (definition[0], dict(definition[1]))
        try:
            values = compute(name, data, **params)
            if isinstance(values, dict):
                raise ValidationError(f"{name} has several outputs; choose one with 'output'", details={"outputs": list(values)})
            out[col] = np.asarray(values, float)
        except PlatformError as exc:
            if on_error == "raise":
                exc.details.setdefault("column", col)
                raise
            errors[col], out[col] = str(exc), np.full(n, np.nan)
    frame = pd.DataFrame(out, index=getattr(data, "index", None))
    frame.attrs["errors"] = errors
    return frame
