"""Helpers shared by the web layer's route modules: JSON-safe conversion, chart series, small parsing helpers, and the
strategy lists the UI may run."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from ..core.exceptions import ValidationError


NEEDS_DATA = {"factor_ls", "sentiment"}          # need a fundamentals store / news feed: not runnable from the UI
SLOW = {"ml_logistic"}                            # retrains repeatedly: keep optimisation small


def clean(o: Any) -> Any:
    """Make anything JSON-safe: numpy -> python, NaN/inf -> null, timestamps -> ISO strings."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)) or hasattr(o, "isoformat"):
        return o.isoformat()
    if hasattr(o, "value") and not isinstance(o, (str, int)):          # Enum
        return o.value
    return o


def series_points(s: pd.Series, max_points: int = 400) -> List[list]:
    if len(s) > max_points:
        s = s.iloc[np.linspace(0, len(s) - 1, max_points).astype(int)]
    return [[t.strftime("%Y-%m-%d"), float(v)] for t, v in zip(pd.DatetimeIndex(s.index), s.to_numpy(float))]


def last_value(x: Any) -> float:
    a = np.asarray(x, float)
    return float(a[-1]) if len(a) else float("nan")


def date_at(x: Any, i: int) -> str:
    """The date of row ``i`` of a frame or an index, as YYYY-MM-DD."""
    index = x if isinstance(x, pd.Index) else x.index
    return pd.Timestamp(index[i]).strftime("%Y-%m-%d")


def bounded(b: Dict[str, Any], key: str, default: float, lo: float, hi: float, kind: Callable[[Any], Any] = float) -> Any:
    value = kind(b.get(key, default))
    if not lo <= value <= hi:
        raise ValidationError(f"{key} must be between {lo:g} and {hi:g}", details={key: value})
    return value


__all__: List[str] = ["NEEDS_DATA", "SLOW", "clean", "series_points", "last_value", "date_at", "bounded"]
