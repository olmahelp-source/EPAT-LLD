"""Parameter spaces for optimisation."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

from ..core.exceptions import OptimizationError


@dataclass(frozen=True)
class Param:
    name: str
    kind: str                         # int | float | log | choice
    low: float = 0.0
    high: float = 0.0
    step: Optional[float] = None
    choices: Sequence[Any] = ()

    def __post_init__(self) -> None:
        if self.kind not in ("int", "float", "log", "choice"):
            raise OptimizationError(f"bad param kind '{self.kind}'")
        if self.kind == "choice":
            if not self.choices:
                raise OptimizationError(f"param '{self.name}' needs choices")
        elif self.low >= self.high:
            raise OptimizationError(f"param '{self.name}': low must be < high")
        if self.kind == "log" and self.low <= 0:
            raise OptimizationError(f"param '{self.name}': log scale needs low > 0")

    def values(self, n: int = 5) -> List[Any]:
        """Discrete candidate values (a grid axis)."""
        if self.kind == "choice":
            return list(self.choices)
        if self.kind == "int":
            step = int(self.step or max(1, (self.high - self.low) // max(n - 1, 1)))
            return list(range(int(self.low), int(self.high) + 1, max(step, 1)))
        if self.step:
            return [round(v, 10) for v in np.arange(self.low, self.high + self.step / 2, self.step)]
        if self.kind == "log":
            return [float(v) for v in np.geomspace(self.low, self.high, n)]
        return [float(v) for v in np.linspace(self.low, self.high, n)]

    def sample(self, rng: np.random.Generator) -> Any:
        if self.kind == "choice":
            return self.choices[int(rng.integers(len(self.choices)))]
        if self.kind == "int":
            return int(rng.integers(int(self.low), int(self.high) + 1))
        if self.kind == "log":
            return float(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))
        v = float(rng.uniform(self.low, self.high))
        return round(round(v / self.step) * self.step, 10) if self.step else v

    def clip(self, v: Any) -> Any:
        if self.kind == "choice":
            return v
        v = min(max(v, self.low), self.high)
        return int(round(v)) if self.kind == "int" else float(v)

    def perturb(self, v: Any, rng: np.random.Generator, scale: float = 0.15) -> Any:
        if self.kind == "choice":
            return self.sample(rng)
        if self.kind == "log":
            return self.clip(v * math.exp(rng.normal(0, scale)))
        span = self.high - self.low
        out = self.clip(v + rng.normal(0, scale * span))
        if self.kind == "float" and self.step:
            out = self.clip(round(round(out / self.step) * self.step, 10))
        return out


class ParameterSpace:
    def __init__(self, params: Sequence[Param], constraint: Optional[Callable[[Dict[str, Any]], bool]] = None) -> None:
        if not params:
            raise OptimizationError("empty parameter space")
        names = [p.name for p in params]
        if len(set(names)) != len(names):
            raise OptimizationError("duplicate parameter names")
        self.params = list(params)
        self.constraint = constraint

    @classmethod
    def from_bounds(cls, bounds: Dict[str, tuple], constraint=None) -> "ParameterSpace":
        """Build from ``Strategy.PARAM_BOUNDS``: ints if both bounds are ints, else floats."""
        ps = [Param(n, "int" if all(isinstance(b, int) for b in lh) else "float", lh[0], lh[1])
              for n, lh in bounds.items()]
        return cls(ps, constraint)

    @property
    def names(self) -> List[str]:
        return [p.name for p in self.params]

    def valid(self, params: Dict[str, Any]) -> bool:
        return self.constraint(params) if self.constraint else True

    def grid(self, points: int = 5) -> Iterator[Dict[str, Any]]:
        axes = [p.values(points) for p in self.params]
        for combo in itertools.product(*axes):
            d = dict(zip(self.names, combo))
            if self.valid(d):
                yield d

    def grid_size(self, points: int = 5) -> int:
        return int(np.prod([len(p.values(points)) for p in self.params]))

    def sample(self, rng: np.random.Generator, max_tries: int = 200) -> Dict[str, Any]:
        for _ in range(max_tries):
            d = {p.name: p.sample(rng) for p in self.params}
            if self.valid(d):
                return d
        raise OptimizationError("could not sample a valid point; constraint too strict")

    def neighbors(self, point: Dict[str, Any], points: int = 5) -> List[Dict[str, Any]]:
        """Adjacent grid points (one axis shifted one step), used for plateau/robustness checks."""
        out = []
        for p in self.params:
            vals = p.values(points)
            if point[p.name] in vals:
                i = vals.index(point[p.name])
                for j in (i - 1, i + 1):
                    if 0 <= j < len(vals):
                        n = dict(point)
                        n[p.name] = vals[j]
                        if self.valid(n):
                            out.append(n)
        return out
