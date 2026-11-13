"""Fuzzy logic: turn trader's rules of thumb into a smooth score.

A crisp rule ("RSI below 30 and trend up -> buy") flips from nothing to everything at RSI 30.00. A fuzzy rule lets
"oversold" be true to a degree, combines several such partial truths, and outputs a graded conviction that position
sizing can use directly. This is a zero-order Sugeno system: each rule proposes a number, and the answer is the average
of the proposals weighted by how strongly each rule fired.

    system = FuzzySystem([
        FuzzyVariable("rsi", {"oversold": left_shoulder(25, 40), "overbought": right_shoulder(60, 75)}),
        FuzzyVariable("trend", {"down": left_shoulder(-1, 0), "up": right_shoulder(0, 1)})])
    system.add_rule("IF rsi IS oversold AND trend IS up THEN 1.0")
    system.add_rule("IF rsi IS overbought AND trend IS down THEN -1.0")
    system.evaluate(rsi=28, trend=0.6)      # -> a conviction in [-1, 1]
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

from ...foundation.core.exceptions import ValidationError

Membership = Callable[[float], float]


def triangle(a: float, b: float, c: float) -> Membership:
    """0 at a, 1 at b, 0 at c."""
    if not a <= b <= c or a == c:
        raise ValidationError("triangle needs a <= b <= c with a < c")
    return lambda x: max(0.0, min((x - a) / (b - a) if b > a else 1.0, (c - x) / (c - b) if c > b else 1.0))


def trapezoid(a: float, b: float, c: float, d: float) -> Membership:
    """0 at a, 1 between b and c, 0 at d."""
    if not a <= b <= c <= d or a == d:
        raise ValidationError("trapezoid needs a <= b <= c <= d with a < d")
    return lambda x: max(0.0, min((x - a) / (b - a) if b > a else 1.0, 1.0, (d - x) / (d - c) if d > c else 1.0))


def left_shoulder(full: float, zero: float) -> Membership:
    """1 at or below ``full``, falling to 0 at ``zero`` ("low", "oversold", "down")."""
    if not full < zero:
        raise ValidationError("left_shoulder needs full < zero")
    return lambda x: 1.0 if x <= full else 0.0 if x >= zero else (zero - x) / (zero - full)


def right_shoulder(zero: float, full: float) -> Membership:
    """0 at or below ``zero``, rising to 1 at ``full`` ("high", "overbought", "up")."""
    if not zero < full:
        raise ValidationError("right_shoulder needs zero < full")
    return lambda x: 0.0 if x <= zero else 1.0 if x >= full else (x - zero) / (full - zero)


def gaussian(mean: float, sigma: float) -> Membership:
    if sigma <= 0:
        raise ValidationError("sigma must be positive")
    return lambda x: math.exp(-0.5 * ((x - mean) / sigma) ** 2)


@dataclass
class FuzzyVariable:
    name: str
    sets: Mapping[str, Membership]

    def __post_init__(self) -> None:
        if not self.name.isidentifier() or not self.sets:
            raise ValidationError("a fuzzy variable needs an identifier name and at least one set")

    def fuzzify(self, x: float) -> Dict[str, float]:
        return {label: float(mf(x)) for label, mf in self.sets.items()}


@dataclass
class FuzzyRule:
    conditions: Sequence[Tuple[str, str, bool]]     # (variable, set label, negated)
    output: float
    operator: str = "and"                           # and -> min of the degrees, or -> max
    weight: float = 1.0
    text: str = field(default="", compare=False)

    def strength(self, degrees: Mapping[str, Mapping[str, float]]) -> float:
        vals = [(1.0 - degrees[v][s]) if neg else degrees[v][s] for v, s, neg in self.conditions]
        return self.weight * (min(vals) if self.operator == "and" else max(vals))


_RULE = re.compile(r"^\s*IF\s+(?P<body>.+?)\s+THEN\s+(?P<out>[-+]?\d*\.?\d+)\s*(?:WEIGHT\s+(?P<w>\d*\.?\d+))?\s*$", re.IGNORECASE)
_TERM = re.compile(r"^\s*(?P<var>[A-Za-z_]\w*)\s+IS\s+(?P<neg>NOT\s+)?(?P<set>[A-Za-z_]\w*)\s*$", re.IGNORECASE)


class FuzzySystem:
    def __init__(self, variables: Sequence[FuzzyVariable], default: float = 0.0) -> None:
        names = [v.name for v in variables]
        if not variables or len(set(names)) != len(names):
            raise ValidationError("need at least one variable, with unique names")
        self.variables = {v.name: v for v in variables}
        self.rules: List[FuzzyRule] = []
        self.default = default

    def add_rule(self, rule) -> "FuzzySystem":
        """Accepts a :class:`FuzzyRule` or text: ``IF a IS x AND b IS NOT y THEN 0.5 [WEIGHT 2]`` (all AND or all OR)."""
        if isinstance(rule, str):
            rule = self._parse(rule)
        if not rule.conditions or rule.weight <= 0 or rule.operator not in ("and", "or"):
            raise ValidationError("a rule needs conditions, a positive weight and operator and/or")
        for var, label, _ in rule.conditions:
            if var not in self.variables or label not in self.variables[var].sets:
                raise ValidationError(f"rule refers to unknown '{var} IS {label}'",
                                      details={"variables": {k: list(v.sets) for k, v in self.variables.items()}})
        self.rules.append(rule)
        return self

    @staticmethod
    def _parse(text: str) -> FuzzyRule:
        m = _RULE.match(text)
        if not m:
            raise ValidationError(f"cannot parse rule: {text!r}", details={"format": "IF <var> IS [NOT] <set> AND|OR ... THEN <number> [WEIGHT <number>]"})
        body = m["body"]
        has_and, has_or = re.search(r"\sAND\s", body, re.I), re.search(r"\sOR\s", body, re.I)
        if has_and and has_or:
            raise ValidationError("a rule may use AND or OR, not both; split it into two rules", details={"rule": text})
        conds = []
        for part in re.split(r"\s+(?:AND|OR)\s+", body, flags=re.I):
            t = _TERM.match(part)
            if not t:
                raise ValidationError(f"cannot parse condition '{part.strip()}'", details={"rule": text})
            conds.append((t["var"], t["set"], bool(t["neg"])))
        return FuzzyRule(conds, float(m["out"]), "or" if has_or else "and", float(m["w"]) if m["w"] else 1.0, text.strip())

    def _degrees(self, inputs: Mapping[str, float]) -> Dict[str, Dict[str, float]]:
        missing = [n for n in self.variables if n not in inputs]
        if missing:
            raise ValidationError(f"missing input(s) {missing}")
        bad = [n for n in self.variables if not math.isfinite(float(inputs[n]))]
        if bad:
            raise ValidationError(f"non-finite input(s) {bad}")
        return {n: v.fuzzify(float(inputs[n])) for n, v in self.variables.items()}

    def explain(self, **inputs: float) -> List[Dict[str, object]]:
        """Every rule with how strongly it fired: the audit trail for a fuzzy decision."""
        deg = self._degrees(inputs)
        return [{"rule": r.text or str(r.conditions), "strength": r.strength(deg), "output": r.output} for r in self.rules]

    def evaluate(self, **inputs: float) -> float:
        """Weighted average of the fired rules' outputs; ``default`` when nothing fires."""
        if not self.rules:
            raise ValidationError("the system has no rules")
        deg = self._degrees(inputs)
        num = den = 0.0
        for r in self.rules:
            s = r.strength(deg)
            num, den = num + s * r.output, den + s
        return num / den if den > 1e-12 else self.default
