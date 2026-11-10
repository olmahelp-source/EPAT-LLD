"""Small pure helpers: ids, validation, numeric utilities."""
from __future__ import annotations

import itertools
import math
import threading
import uuid
from typing import Any, Iterable, List, Optional, Sequence, TypeVar

import numpy as np

from ..core.exceptions import ValidationError

T = TypeVar("T")


class IdGenerator:
    """Monotonic, thread-safe, human-readable ids: ``PREFIX-<run>-<seq>``.

    The run token keeps ids unique across restarts, which matters for idempotent
    client order ids.
    """

    def __init__(self, prefix: str = "ORD", run_token: Optional[str] = None) -> None:
        self.prefix = prefix
        self.run = run_token or uuid.uuid4().hex[:6]
        self._counter = itertools.count(1)
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            return f"{self.prefix}-{self.run}-{next(self._counter):08d}"


# ---- validation ---------------------------------------------------------------------
def require(cond: bool, message: str, **details: Any) -> None:
    if not cond:
        raise ValidationError(message, details=details)


def require_positive(value: float, name: str) -> float:
    require(isinstance(value, (int, float)) and math.isfinite(value) and value > 0,
            f"{name} must be a positive finite number", value=value)
    return float(value)


def require_in_range(value: float, lo: float, hi: float, name: str) -> float:
    require(math.isfinite(value) and lo <= value <= hi, f"{name} must be in [{lo}, {hi}]", value=value)
    return float(value)


def require_non_empty(seq: Sequence, name: str) -> Sequence:
    require(seq is not None and len(seq) > 0, f"{name} must not be empty")
    return seq


# ---- numerics ------------------------------------------------------------------------
def safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        if b == 0 or not math.isfinite(b):
            return default
        out = a / b
        return out if math.isfinite(out) else default
    except (TypeError, ZeroDivisionError):
        return default


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def to_array(x: Iterable[float], name: str = "x", min_len: int = 1) -> np.ndarray:
    arr = np.asarray(list(x) if not hasattr(x, "__len__") else x, dtype=float)
    require(arr.ndim == 1, f"{name} must be 1-D")
    require(len(arr) >= min_len, f"{name} needs at least {min_len} observations", n=len(arr))
    return arr


def drop_nan(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def chunked(seq: Sequence[T], n: int) -> List[Sequence[T]]:
    require(n > 0, "chunk size must be positive")
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def round_to(x: float, step: float) -> float:
    return round(round(x / step) * step, 10)


def annualize_return(total_return: float, periods: int, per_year: float) -> float:
    if periods <= 0 or total_return <= -1:
        return -1.0 if total_return <= -1 else 0.0
    return (1.0 + total_return) ** (per_year / periods) - 1.0
