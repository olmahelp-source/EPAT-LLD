"""Market structure: clusters, spanning trees and common factors hidden in a correlation matrix."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.exceptions import StatisticsError
from .correlation import correlation_distance


def hierarchical_clusters(corr: pd.DataFrame, n_clusters: Optional[int] = None, threshold: Optional[float] = None,
                          linkage: str = "average") -> Dict[str, object]:
    """Agglomerative clustering on correlation distance.

    Stop at ``n_clusters`` groups, or when the closest two groups are further apart than ``threshold`` (a distance in
    [0, 1]; 0.5 corresponds to zero correlation). Returns ``labels`` (Series symbol -> cluster id), ``order`` (leaves
    arranged so that similar instruments are adjacent: the quasi-diagonal order used by hierarchical risk parity) and
    ``merges``."""
    if linkage not in ("single", "complete", "average"):
        raise StatisticsError(f"unknown linkage '{linkage}'")
    if (n_clusters is None) == (threshold is None):
        raise StatisticsError("give exactly one of n_clusters / threshold")
    names = list(corr.index)
    n = len(names)
    if n < 2 or corr.shape[0] != corr.shape[1]:
        raise StatisticsError("need a square correlation matrix of at least 2 instruments")
    if n_clusters is not None and not 1 <= n_clusters <= n:
        raise StatisticsError("n_clusters must be between 1 and the number of instruments")
    dist = correlation_distance(corr.fillna(0.0)).to_numpy().copy()
    members: Dict[int, List[int]] = {i: [i] for i in range(n)}
    d = {(i, j): dist[i, j] for i in range(n) for j in range(i + 1, n)}
    merges: List[Tuple[List[str], List[str], float]] = []
    stopped: Optional[Dict[int, List[int]]] = None
    next_id = n
    while len(members) > 1:
        (a, b), best = min(d.items(), key=lambda kv: (kv[1], kv[0]))
        if stopped is None and ((n_clusters is not None and len(members) <= n_clusters) or (threshold is not None and best > threshold)):
            stopped = {k: list(v) for k, v in members.items()}       # remember the cut, keep merging to get the full leaf order
        merges.append(([names[i] for i in members[a]], [names[i] for i in members[b]], float(best)))
        na, nb = len(members[a]), len(members[b])
        new: Dict[Tuple[int, int], float] = {}
        for c in members:
            if c in (a, b):
                continue
            dac, dbc = d[(min(a, c), max(a, c))], d[(min(b, c), max(b, c))]
            new[(c, next_id)] = min(dac, dbc) if linkage == "single" else max(dac, dbc) if linkage == "complete" else (na * dac + nb * dbc) / (na + nb)
        d = {k: v for k, v in d.items() if a not in k and b not in k}
        d.update(new)
        members[next_id] = members.pop(a) + members.pop(b)
        next_id += 1
    final = stopped if stopped is not None else members
    labels = pd.Series(0, index=names, dtype=int)
    for cid, (_, idx) in enumerate(sorted(final.items(), key=lambda kv: min(kv[1]))):
        labels.iloc[idx] = cid
    order = [names[i] for i in next(iter(members.values()))]
    return {"labels": labels, "order": order, "merges": merges, "n_clusters": int(labels.nunique())}


def minimum_spanning_tree(corr: pd.DataFrame) -> List[Tuple[str, str, float]]:
    """The n-1 strongest links that connect every instrument (Prim's algorithm on correlation distance). Hubs of the
    tree are the instruments the rest of the market hangs off."""
    names = list(corr.index)
    n = len(names)
    if n < 2:
        raise StatisticsError("need at least 2 instruments")
    dist = correlation_distance(corr.fillna(0.0)).to_numpy()
    in_tree, edges = {0}, []
    best = {j: (dist[0, j], 0) for j in range(1, n)}
    while len(in_tree) < n:
        j = min(best, key=lambda k: best[k][0])
        dj, i = best.pop(j)
        in_tree.add(j)
        edges.append((names[i], names[j], float(dj)))
        for k in best:
            if dist[j, k] < best[k][0]:
                best[k] = (dist[j, k], j)
    return edges


def pca_factors(returns: pd.DataFrame, n_components: int = 3) -> Dict[str, object]:
    """Statistical factors of a return panel. The first is almost always "the market"; later ones are sector / style tilts."""
    clean = returns.dropna()
    k = clean.shape[1]
    if len(clean) < 3 or not 1 <= n_components <= k:
        raise StatisticsError("need at least 3 complete rows and 1 <= n_components <= number of series")
    z = (clean - clean.mean()) / clean.std(ddof=1).replace(0, np.nan)
    z = z.fillna(0.0)
    vals, vecs = np.linalg.eigh(np.cov(z.to_numpy(), rowvar=False))
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0, None), vecs[:, order]
    vecs = vecs * np.where(vecs.sum(axis=0) < 0, -1.0, 1.0)             # fix the sign: loadings sum to a positive number
    cols = [f"PC{i + 1}" for i in range(n_components)]
    return {"explained_ratio": pd.Series(vals[:n_components] / vals.sum(), index=cols),
            "loadings": pd.DataFrame(vecs[:, :n_components], index=clean.columns, columns=cols),
            "factors": pd.DataFrame(z.to_numpy() @ vecs[:, :n_components], index=clean.index, columns=cols)}


def absorption_ratio(returns: pd.DataFrame, n_components: Optional[int] = None) -> float:
    """Share of total variance explained by the top components (default: one fifth of the series). A rising value means
    the market is moving as one block, i.e. diversification is evaporating."""
    k = returns.shape[1]
    n = n_components or max(1, k // 5)
    ratio: pd.Series = pca_factors(returns, n)["explained_ratio"]          # type: ignore[assignment]
    return float(ratio.sum())
