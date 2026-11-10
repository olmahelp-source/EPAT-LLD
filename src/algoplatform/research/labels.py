"""Financial labelling: triple-barrier, meta-labels, sample uniqueness and sequential bootstrap.

Labels look FORWARD in time by construction (that is what a label is). The rule that keeps research honest is that a
label at t0 is only usable for training once its barrier time t1 is in the past, which the purged CV in ``cv.py`` and
the rolling trainer in ``strategy/ml.py`` both enforce.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..core.exceptions import ValidationError


def daily_vol(close: pd.Series, span: int = 50) -> pd.Series:
    """EWM standard deviation of simple returns: the usual dynamic barrier width."""
    return close.pct_change().ewm(span=span, min_periods=max(5, span // 5)).std()


def triple_barrier(close: pd.Series, t_events: pd.Index, target: pd.Series, horizon: int,
                   pt_sl: Tuple[float, float] = (1.0, 1.0), min_ret: float = 0.0,
                   side: Optional[pd.Series] = None) -> pd.DataFrame:
    """Label each event by the first barrier touched within ``horizon`` bars.

    Barriers are ``pt_sl[0] * target`` above and ``pt_sl[1] * target`` below the entry (set a multiplier to 0 to
    disable it). Checks use closes. Events without ``horizon`` bars of future data are dropped rather than given a
    truncated label. With ``side`` (+1/-1) returns are side-adjusted and the label becomes the meta-label
    1 (the primary bet made money) / 0 (it did not).

    Columns: t1 (barrier time), ret, label, barrier ('pt' | 'sl' | 'time'), target.
    """
    if horizon < 1:
        raise ValidationError("horizon must be >= 1")
    pos = close.index.get_indexer(t_events)
    if (pos < 0).any():
        raise ValidationError("every event time must exist in the close index")
    vals = close.to_numpy(dtype=float)
    rows = []
    for t0, p0 in zip(t_events, pos):
        end = p0 + horizon
        if end >= len(vals):
            continue
        tgt = float(target.loc[t0]) if t0 in target.index else np.nan
        if not np.isfinite(tgt) or tgt <= 0 or tgt < min_ret:
            continue
        s = float(side.loc[t0]) if side is not None else 1.0
        path = (vals[p0 + 1:end + 1] / vals[p0] - 1.0) * s
        up = pt_sl[0] * tgt if pt_sl[0] > 0 else np.inf
        dn = -pt_sl[1] * tgt if pt_sl[1] > 0 else -np.inf
        hit_up, hit_dn = np.flatnonzero(path >= up), np.flatnonzero(path <= dn)
        fu = hit_up[0] if len(hit_up) else horizon + 1
        fd = hit_dn[0] if len(hit_dn) else horizon + 1
        if fu > horizon - 1 and fd > horizon - 1:
            barrier, k, ret = "time", horizon - 1, float(path[-1])
            label = float(np.sign(ret)) if abs(ret) >= min_ret else 0.0
        elif fd <= fu:                                        # tie => stop first (the conservative assumption)
            barrier, k, ret, label = "sl", int(fd), float(path[fd]), -1.0
        else:
            barrier, k, ret, label = "pt", int(fu), float(path[fu]), 1.0
        if side is not None:
            label = 1.0 if ret > 0 else 0.0
        rows.append((t0, close.index[p0 + 1 + k], ret, label, barrier, tgt))
    out = pd.DataFrame(rows, columns=["t0", "t1", "ret", "label", "barrier", "target"]).set_index("t0")
    return out


# ---- sample uniqueness --------------------------------------------------------------------------------------
def _spans(t1: pd.Series, close_index: pd.Index) -> Tuple[np.ndarray, np.ndarray]:
    start = close_index.get_indexer(t1.index)
    end = close_index.get_indexer(pd.Index(t1.values))
    if (start < 0).any() or (end < 0).any():
        raise ValidationError("event and barrier times must exist in the close index")
    return start, end


def num_concurrent(t1: pd.Series, close_index: pd.Index) -> pd.Series:
    """How many label spans cover each bar (overlap makes samples non-independent)."""
    s, e = _spans(t1, close_index)
    diff = np.zeros(len(close_index) + 1)
    np.add.at(diff, s, 1)
    np.add.at(diff, e + 1, -1)
    return pd.Series(np.cumsum(diff)[:-1], index=close_index)


def average_uniqueness(t1: pd.Series, close_index: pd.Index) -> pd.Series:
    """Mean of 1/concurrency over each label's span: 1 = no overlap, ->0 = heavily overlapped."""
    conc = num_concurrent(t1, close_index).to_numpy()
    s, e = _spans(t1, close_index)
    u = [float(np.mean(1.0 / np.maximum(conc[a:b + 1], 1))) for a, b in zip(s, e)]
    return pd.Series(u, index=t1.index)


def sample_weights(t1: pd.Series, close_index: pd.Index, ret: Optional[pd.Series] = None) -> pd.Series:
    """Training weights proportional to uniqueness (optionally x |return|), scaled to average 1."""
    w = average_uniqueness(t1, close_index)
    if ret is not None:
        w = w * ret.reindex(w.index).abs()
    return w * len(w) / w.sum() if w.sum() > 0 else w


def sequential_bootstrap(t1: pd.Series, close_index: pd.Index, n_draws: Optional[int] = None,
                         seed: Optional[int] = 0) -> np.ndarray:
    """Bootstrap indices where each draw favours events that overlap little with those already chosen, giving a
    far more independent sample than plain bootstrap on overlapping labels."""
    s, e = _spans(t1, close_index)
    n, m = len(t1), len(close_index)
    ind = np.zeros((n, m))
    for i, (a, b) in enumerate(zip(s, e)):
        ind[i, a:b + 1] = 1.0
    length = ind.sum(1)
    rng = np.random.default_rng(seed)
    chosen, c = [], np.zeros(m)
    for _ in range(n_draws or n):
        u = (ind / (c + ind).clip(min=1)).sum(1) / length           # uniqueness each candidate would have if added
        p = u / u.sum()
        i = int(rng.choice(n, p=p))
        chosen.append(i)
        c += ind[i]
    return np.asarray(chosen)
