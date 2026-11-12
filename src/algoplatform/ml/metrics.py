"""Model evaluation metrics (complements ``research.models``: log_loss, brier_score, accuracy, calibration)."""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.exceptions import ValidationError
from ..utils.numeric import rank, spearman


def _yp(y, p):
    y, p = np.asarray(y, float).ravel(), np.asarray(p, float).ravel()
    if len(y) != len(p) or not len(y):
        raise ValidationError("y and predictions must be non-empty and the same length")
    ok = np.isfinite(y) & np.isfinite(p)
    return y[ok], p[ok]


def roc_auc(y, score) -> float:
    """Probability that a random positive is ranked above a random negative (0.5 = no skill). Threshold-free."""
    y, s = _yp(y, score)
    pos, neg = int((y >= 0.5).sum()), int((y < 0.5).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    return float((rank(s)[y >= 0.5].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def confusion(y, p, threshold: float = 0.5) -> Dict[str, int]:
    y, p = _yp(y, p)
    yt, yp = y >= 0.5, p >= threshold
    return {"tp": int((yt & yp).sum()), "fp": int((~yt & yp).sum()), "tn": int((~yt & ~yp).sum()), "fn": int((yt & ~yp).sum())}


def precision_recall_f1(y, p, threshold: float = 0.5) -> Dict[str, float]:
    """Precision matters most for a trading signal: of the trades it told you to take, how many were right."""
    c = confusion(y, p, threshold)
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else float("nan")
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1}


def r2_score(y, pred) -> float:
    y, pred = _yp(y, pred)
    var = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - pred) ** 2).sum() / var) if var > 0 else float("nan")


def information_coefficient(forecast, outcome) -> float:
    """Rank correlation between forecasts and what followed: the standard measure of a signal's predictive power.
    0.02-0.05 on daily data is already a useful signal; much more usually means leakage."""
    return spearman(forecast, outcome)


def directional_accuracy(forecast, outcome) -> float:
    f, o = _yp(forecast, outcome)
    live = (f != 0) & (o != 0)
    return float((np.sign(f[live]) == np.sign(o[live])).mean()) if live.any() else float("nan")
