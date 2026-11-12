"""A safe formula language over the function catalogue.

    crossover(ema(close, fast), ema(close, slow)) and rsi(close, 14) < 70
    supertrend(high, low, close, 10, 3).direction > 0 and relative_volume(volume, 20) > 1.5
    clamp(-zscore(close, 20) / 2, -1, 1)

The text is parsed with Python's own parser but **never executed by it**: the syntax tree is walked by the small
interpreter below, which knows only arithmetic, comparisons, and/or/not, ``a if c else b``, catalogue function calls
and picking one output of a multi-output function (``macd(close).hist``). There is no attribute access on values, no
indexing, no imports, no lambdas and no names other than data columns, declared parameters and catalogue functions, so
a formula typed into a UI or loaded from JSON cannot reach the interpreter, the file system or the process.

Semantics follow ``technical.operators``: everything is evaluated over the whole series, and any comparison involving
NaN (indicator warm-up) is False, so an unknown value never fires a rule.
"""
from __future__ import annotations

import ast
from typing import Any, Callable, Dict, Mapping, Optional, Set

import numpy as np

from ..core.exceptions import ExpressionError, PlatformError
from . import operators as ops
from .catalog import CATALOG, PRICE_FIELDS, suggest

MAX_LENGTH, MAX_NODES, MAX_DEPTH, MAX_POWER = 2000, 400, 40, 8.0
_ALLOWED = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd, ast.BinOp, ast.Add, ast.Sub,
            ast.Mult, ast.Div, ast.Pow, ast.Compare, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq, ast.Call, ast.keyword,
            ast.Name, ast.Load, ast.Constant, ast.Attribute, ast.IfExp)
_CONSTANTS = {"nan": float("nan"), "true": True, "false": False}


def _depth(node: ast.AST) -> int:
    return 1 + max((_depth(c) for c in ast.iter_child_nodes(node)), default=0)


class Expression:
    """A compiled formula. Compile once, evaluate against any OHLCV frame (or mapping of arrays)."""

    def __init__(self, text: str, params: Optional[Mapping[str, Any]] = None,
                 functions: Optional[Mapping[str, Callable[..., Any]]] = None) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ExpressionError("expression must be a non-empty string")
        if len(text) > MAX_LENGTH:
            raise ExpressionError(f"expression longer than {MAX_LENGTH} characters")
        self.text = text.strip()
        self.params: Dict[str, Any] = dict(params or {})
        clash = sorted(set(self.params) & (set(PRICE_FIELDS) | set(_CONSTANTS)))
        if clash:
            raise ExpressionError(f"parameter name(s) {clash} clash with data columns or constants")
        self._functions: Dict[str, Callable[..., Any]] = {name: spec.fn for name, spec in CATALOG.items()}
        self._functions.update(functions or {})
        try:
            self._tree = ast.parse(self.text, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(f"syntax error at column {exc.offset}: {exc.msg}", details={"expression": self.text}) from None
        self.names: Set[str] = set()         # data columns and parameters the formula reads
        self.calls: Set[str] = set()         # catalogue functions it uses
        self._validate()

    # ---- static checks: everything that can be rejected before any data is seen ------------------------------------------
    def _validate(self) -> None:
        nodes = list(ast.walk(self._tree))
        if len(nodes) > MAX_NODES or _depth(self._tree) > MAX_DEPTH:
            raise ExpressionError("expression is too large or too deeply nested")
        called = {id(n.func) for n in nodes if isinstance(n, ast.Call)}
        for node in nodes:
            if not isinstance(node, _ALLOWED):
                raise ExpressionError(f"'{type(node).__name__}' is not allowed in expressions", details={"expression": self.text})
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ExpressionError("only catalogue functions can be called")
                if node.func.id not in self._functions:
                    raise ExpressionError(f"unknown function '{node.func.id}'", details={"did_you_mean": suggest(node.func.id)})
                if any(k.arg is None for k in node.keywords):
                    raise ExpressionError("'**' arguments are not allowed")
                self.calls.add(node.func.id)
            elif isinstance(node, ast.Attribute):
                if not isinstance(node.value, ast.Call) or node.attr.startswith("_"):
                    raise ExpressionError("'.' may only pick an output of a function call, e.g. macd(close).hist")
                spec = CATALOG.get(node.value.func.id) if isinstance(node.value.func, ast.Name) else None
                if spec is not None and spec.outputs and node.attr not in spec.outputs:
                    raise ExpressionError(f"{spec.name} has no output '{node.attr}'", details={"outputs": list(spec.outputs)})
            elif isinstance(node, ast.Constant):
                if not isinstance(node.value, (int, float, bool, str)) or isinstance(node.value, complex):
                    raise ExpressionError(f"constant {node.value!r} is not allowed")
            elif isinstance(node, ast.Name) and id(node) not in called:
                if node.id.startswith("_"):
                    raise ExpressionError(f"name '{node.id}' is not allowed")
                self.names.add(node.id)

    @property
    def columns(self) -> Set[str]:
        """Names the formula expects to find in the data (everything that is not a parameter or constant)."""
        return {n for n in self.names if n not in self.params and n not in _CONSTANTS}

    # ---- evaluation -----------------------------------------------------------------------------------------------------
    def evaluate(self, data: Mapping[str, Any], params: Optional[Mapping[str, Any]] = None) -> Any:
        """Value of the formula over the whole series: a float array, a bool array, or a scalar for constant formulas."""
        env = {**self.params, **(params or {})}
        cache: Dict[str, np.ndarray] = {}
        try:
            return self._eval(self._tree.body, data, env, cache)
        except PlatformError as exc:
            exc.details.setdefault("expression", self.text)
            raise

    def mask(self, data: Mapping[str, Any], params: Optional[Mapping[str, Any]] = None) -> np.ndarray:
        """The formula as a condition: a bool array with NaN treated as False."""
        value = self._series(self.evaluate(data, params))
        out = ops._cond(value)
        return np.full(self._length(data), bool(out[0])) if out.shape == (1,) and self._length(data) != 1 else out

    def last(self, data: Mapping[str, Any], params: Optional[Mapping[str, Any]] = None) -> float:
        """The formula's value on the final bar (NaN if there is none): what a live strategy asks every bar."""
        value = np.atleast_1d(np.asarray(self._series(self.evaluate(data, params)), float))
        return float(value[-1]) if len(value) else float("nan")

    def holds(self, data: Mapping[str, Any], params: Optional[Mapping[str, Any]] = None) -> bool:
        v = self.last(data, params)
        return bool(np.isfinite(v) and v != 0)

    @staticmethod
    def _length(data: Mapping[str, Any]) -> int:
        for k in PRICE_FIELDS:
            if k in data:
                return len(data[k])
        return 1

    def _series(self, value: Any) -> Any:
        if isinstance(value, dict):
            raise ExpressionError("the formula ends in a multi-output function; pick one output, e.g. macd(close).hist",
                                  details={"outputs": list(value), "expression": self.text})
        if isinstance(value, str):
            raise ExpressionError("a formula cannot evaluate to text", details={"expression": self.text})
        return value

    def _name(self, name: str, data: Mapping[str, Any], env: Mapping[str, Any], cache: Dict[str, np.ndarray]) -> Any:
        if name in env:
            return env[name]
        if name in cache:
            return cache[name]
        if name in data:
            cache[name] = np.asarray(data[name], float)
            return cache[name]
        if name in _CONSTANTS:
            return _CONSTANTS[name]
        known = sorted(set(env) | {k for k in PRICE_FIELDS if k in data})
        raise ExpressionError(f"unknown name '{name}'", details={"known": known, "did_you_mean": suggest(name)})

    def _number(self, value: Any, what: str) -> Any:
        if isinstance(value, (dict, str)):
            raise ExpressionError(f"{what} needs a number or series" + ("; pick one output of the function first" if isinstance(value, dict) else ""),
                                  details={"expression": self.text})
        return np.asarray(value, float) if isinstance(value, np.ndarray) and value.dtype == bool else value

    def _eval(self, node: ast.AST, data: Mapping[str, Any], env: Mapping[str, Any], cache: Dict[str, np.ndarray]) -> Any:
        ev = lambda n: self._eval(n, data, env, cache)                                  # noqa: E731
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self._name(node.id, data, env, cache)
        if isinstance(node, ast.BoolOp):
            values = [self._number(ev(v), "and / or") for v in node.values]
            return ops.and_(*values) if isinstance(node.op, ast.And) else ops.or_(*values)
        if isinstance(node, ast.UnaryOp):
            value = self._number(ev(node.operand), "a unary operator")
            if isinstance(node.op, ast.Not):
                return ops.not_(value)
            return -np.asarray(value, float) if isinstance(node.op, ast.USub) else np.asarray(value, float)
        if isinstance(node, ast.BinOp):
            return self._binary(node.op, self._number(ev(node.left), "arithmetic"), self._number(ev(node.right), "arithmetic"))
        if isinstance(node, ast.Compare):
            out, left = None, self._number(ev(node.left), "a comparison")
            for op, right_node in zip(node.ops, node.comparators):
                right = self._number(ev(right_node), "a comparison")
                piece = self._compare(op, left, right)
                out = piece if out is None else ops.and_(out, piece)
                left = right
            return out
        if isinstance(node, ast.IfExp):
            return ops.where(self._number(ev(node.test), "a condition"), self._number(ev(node.body), "if/else"), self._number(ev(node.orelse), "if/else"))
        if isinstance(node, ast.Call):
            name = node.func.id                                                         # type: ignore[attr-defined]
            args = [ev(a) for a in node.args]
            kwargs = {k.arg: ev(k.value) for k in node.keywords}
            try:
                return self._functions[name](*args, **kwargs)
            except PlatformError as exc:
                exc.details.setdefault("function", name)
                raise
            except Exception as exc:  # noqa: BLE001 - wrong argument count / types surface as a formula error
                sig = CATALOG[name].signature() if name in CATALOG else name
                raise ExpressionError(f"{name}(...) failed: {exc}", details={"signature": sig, "expression": self.text}) from exc
        if isinstance(node, ast.Attribute):
            value = ev(node.value)
            if not isinstance(value, dict) or node.attr not in value:
                raise ExpressionError(f"no output '{node.attr}' here", details={"outputs": list(value) if isinstance(value, dict) else []})
            return value[node.attr]
        raise ExpressionError(f"cannot evaluate '{type(node).__name__}'")               # unreachable after _validate

    @staticmethod
    def _binary(op: ast.operator, a: Any, b: Any) -> Any:
        a, b = np.asarray(a, float), np.asarray(b, float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if isinstance(op, ast.Add):
                return a + b
            if isinstance(op, ast.Sub):
                return a - b
            if isinstance(op, ast.Mult):
                return a * b
            if isinstance(op, ast.Div):
                return np.where(b == 0, np.nan, a / np.where(b == 0, 1.0, b))            # x / 0 is unknown, not an error
            if b.ndim != 0 or abs(float(b)) > MAX_POWER:
                raise ExpressionError(f"the exponent of '**' must be a constant with magnitude <= {MAX_POWER:g}")
            return np.power(a, float(b))

    @staticmethod
    def _compare(op: ast.cmpop, a: Any, b: Any) -> np.ndarray:
        a, b = ops._pair(a, b)
        known = ~(np.isnan(a) | np.isnan(b))
        with np.errstate(invalid="ignore"):
            if isinstance(op, ast.Gt):
                out = a > b
            elif isinstance(op, ast.GtE):
                out = a >= b
            elif isinstance(op, ast.Lt):
                out = a < b
            elif isinstance(op, ast.LtE):
                out = a <= b
            elif isinstance(op, ast.Eq):
                out = a == b
            else:
                out = a != b
        return out & known

    def __repr__(self) -> str:
        return f"Expression({self.text!r})"


def compile_expression(text: str, params: Optional[Mapping[str, Any]] = None) -> Expression:
    return Expression(text, params)


def evaluate(text: str, data: Mapping[str, Any], **params: Any) -> Any:
    """One-shot convenience: compile and evaluate."""
    return Expression(text, params).evaluate(data)
