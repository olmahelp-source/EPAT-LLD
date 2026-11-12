"""Walk-forward prediction: the only honest way to produce model outputs for a backtest.

The model is refitted on the past, predicts the next block, then the window moves on. Every prediction is therefore
made by a model that had not seen that row, nor (with ``embargo``) any training label that overlaps it in time.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from ..core.exceptions import ValidationError
from ..core.logging import get_logger
from ..research.features import Scaler
from .ensemble import _score

log = get_logger(__name__)


def walk_forward_predict(factory: Callable[[], Any], X, y, train_size: int = 250, step: int = 20, embargo: int = 0,
                         expanding: bool = True, scale: bool = True, sample_weight=None) -> Dict[str, Any]:
    """Out-of-sample predictions for rows ``train_size + embargo`` onwards.

    ``embargo``: number of rows dropped between the end of training and the first predicted row. Set it to the label
    horizon: a label built from the next h bars must not be in training when predicting inside those h bars.
    ``scale``: standardise features with statistics from the training window only.
    A block whose fit fails is left as NaN and counted in ``failed_fits`` (the previous model is *not* reused silently).
    Returns ``{"pred": array (NaN where not predicted), "fits": int, "failed_fits": int}``."""
    X, y = np.asarray(X, float), np.asarray(y, float).ravel()
    if X.ndim != 2 or len(X) != len(y):
        raise ValidationError("X must be 2-D with one row per label")
    if train_size < 10 or step < 1 or embargo < 0:
        raise ValidationError("need train_size >= 10, step >= 1, embargo >= 0")
    if len(y) <= train_size + embargo:
        raise ValidationError("not enough rows to predict anything", details={"rows": len(y), "needed": train_size + embargo + 1})
    sw = None if sample_weight is None else np.asarray(sample_weight, float)
    pred = np.full(len(y), np.nan)
    fits = failed = 0
    for start in range(train_size + embargo, len(y), step):
        end_train = start - embargo
        lo = 0 if expanding else end_train - train_size
        rows = np.arange(lo, end_train)
        rows = rows[np.isfinite(X[rows]).all(axis=1) & np.isfinite(y[rows])]
        block = np.arange(start, min(start + step, len(y)))
        usable = block[np.isfinite(X[block]).all(axis=1)]
        if len(rows) < 10 or not len(usable):
            continue
        try:
            scaler = Scaler().fit(X[rows]) if scale else None
            tx = (lambda a: scaler.transform(a)) if scaler else (lambda a: a)
            model = factory().fit(tx(X[rows]), y[rows], None if sw is None else sw[rows])
            pred[usable] = _score(model, tx(X[usable]))
            fits += 1
        except Exception as exc:  # noqa: BLE001 - e.g. a training window with a single class
            failed += 1
            log.warning("walk-forward fit at row %d failed: %r", start, exc)
    return {"pred": pred, "fits": fits, "failed_fits": failed}


def forward_returns(close, horizon: int = 1) -> np.ndarray:
    """Return over the next ``horizon`` bars, aligned to the bar on which a decision would be made (the last ``horizon``
    entries are NaN: their future has not happened). This is a *label*, never a feature."""
    c = np.asarray(close, float)
    if horizon < 1:
        raise ValidationError("horizon must be >= 1")
    out = np.full(len(c), np.nan)
    if len(c) > horizon:
        out[:-horizon] = c[horizon:] / c[:-horizon] - 1.0
    return out


def signal_from_probability(prob, edge: float = 0.05, allow_short: bool = True) -> np.ndarray:
    """Map P(up) to a position in [-1, 1]: flat inside the no-trade band 0.5 +/- edge, scaled by conviction outside it."""
    p = np.asarray(prob, float)
    out = np.where(p > 0.5 + edge, np.minimum(1.0, (p - 0.5) * 2), np.where((p < 0.5 - edge) & allow_short, -np.minimum(1.0, (0.5 - p) * 2), 0.0))
    return np.where(np.isfinite(p), out, 0.0)
