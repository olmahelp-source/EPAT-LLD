"""Combining models. Averaging several mediocre, differently-wrong models is usually more robust out of sample than
selecting the single best one in sample (which is itself a form of overfitting)."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from ...foundation.core.exceptions import ModelError, ValidationError
from ...foundation.core.logging import get_logger

log = get_logger(__name__)


def _score(model: Any, X) -> np.ndarray:
    out = model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X)
    out = np.asarray(out, float)
    return out[:, 1] if out.ndim == 2 else out


class VotingEnsemble:
    """Weighted average of the members' predictions. A member that fails to fit is dropped with a warning (and listed in
    ``failed_``); only if every member fails does ``fit`` raise."""

    def __init__(self, models: Sequence[Any], weights: Optional[Sequence[float]] = None) -> None:
        if not models:
            raise ValidationError("ensemble needs at least one model")
        if weights is not None and (len(weights) != len(models) or min(weights) < 0 or sum(weights) <= 0):
            raise ValidationError("weights must be non-negative, one per model, and not all zero")
        self.models, self.weights = list(models), list(weights) if weights is not None else [1.0] * len(models)
        self.fitted_: List[Any] = []
        self.failed_: List[str] = []

    def fit(self, X, y, sample_weight=None) -> "VotingEnsemble":
        self.fitted_, self.failed_, self._w = [], [], []
        for model, w in zip(self.models, self.weights):
            try:
                model.fit(X, y, sample_weight)
                self.fitted_.append(model)
                self._w.append(w)
            except Exception as exc:  # noqa: BLE001 - one bad member must not sink the ensemble
                self.failed_.append(f"{type(model).__name__}: {exc}")
                log.warning("ensemble member %s failed to fit: %r", type(model).__name__, exc)
        if not self.fitted_ or sum(self._w) <= 0:
            raise ModelError("every ensemble member failed to fit", details={"errors": self.failed_})
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self.fitted_:
            raise ModelError("ensemble is not fitted")
        return np.average([_score(m, X) for m in self.fitted_], axis=0, weights=self._w)

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(float)


class StackingEnsemble:
    """A meta-model learns how to weigh the base models.

    To keep the meta-model honest it is trained on base predictions for rows the base models have **not** seen: the
    bases are first fitted on the earlier part of the (time-ordered) data and predict the later ``holdout`` part; the
    meta-model is fitted there; finally the bases are refitted on everything."""

    def __init__(self, base_models: Sequence[Any], meta_model: Any, holdout: float = 0.3) -> None:
        if not base_models or not 0.1 <= holdout <= 0.6:
            raise ValidationError("need at least one base model and holdout in [0.1, 0.6]")
        self.base_models, self.meta_model, self.holdout = list(base_models), meta_model, holdout
        self._fitted = False

    def fit(self, X, y, sample_weight=None) -> "StackingEnsemble":
        X, y = np.asarray(X, float), np.asarray(y, float)
        cut = int(len(y) * (1 - self.holdout))
        if cut < 10 or len(y) - cut < 10:
            raise ValidationError("too few rows to split into base and meta training sets", details={"rows": len(y)})
        sw = None if sample_weight is None else np.asarray(sample_weight, float)
        for m in self.base_models:
            m.fit(X[:cut], y[:cut], None if sw is None else sw[:cut])
        Z = np.column_stack([_score(m, X[cut:]) for m in self.base_models])
        self.meta_model.fit(Z, y[cut:], None if sw is None else sw[cut:])
        for m in self.base_models:
            m.fit(X, y, sw)
        self._fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self._fitted:
            raise ModelError("ensemble is not fitted")
        return _score(self.meta_model, np.column_stack([_score(m, X) for m in self.base_models]))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(float)
