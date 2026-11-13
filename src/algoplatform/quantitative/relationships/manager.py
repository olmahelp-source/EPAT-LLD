"""RelationshipManager: one place that knows how instruments are related, and what that means for a portfolio.

Relations come from three sources: declared facts (same sector, same issuer, supplier/customer, index membership),
statistics computed from prices (correlated, cointegrated, leads), and anything a user adds by hand. Consumers ask
questions instead of recomputing statistics: who are this symbol's peers, which pairs are tradable, and how many
independent bets does this set of positions really contain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Set

import numpy as np
import pandas as pd

from ...foundation.core.exceptions import PlatformError, ValidationError
from .cointegration import find_cointegrated_pairs
from .correlation import correlation_matrix
from .leadlag import lead_lag


@dataclass(frozen=True)
class Relation:
    a: str
    b: str
    kind: str                       # sector | correlated | cointegrated | leads | <anything declared by hand>
    weight: float = 1.0             # strength: correlation, test statistic, lag correlation ...
    directed: bool = False          # True: a -> b (e.g. a leads b)
    meta: Mapping[str, Any] = field(default_factory=dict)

    def other(self, symbol: str) -> str:
        return self.b if symbol == self.a else self.a


class RelationshipManager:
    def __init__(self, sectors: Optional[Mapping[str, str]] = None) -> None:
        self._relations: List[Relation] = []
        self.sectors: Dict[str, str] = dict(sectors or {})
        self.errors: List[str] = []

    # ---- building ---------------------------------------------------------------------------------------------------------
    def add(self, a: str, b: str, kind: str, weight: float = 1.0, directed: bool = False, **meta: Any) -> Relation:
        if a == b:
            raise ValidationError("a relation needs two different instruments")
        rel = Relation(a, b, kind, float(weight), directed, meta)
        self._relations = [r for r in self._relations if not (r.kind == kind and {r.a, r.b} == {a, b})] + [rel]
        return rel

    def clear(self, kind: Optional[str] = None) -> None:
        self._relations = [] if kind is None else [r for r in self._relations if r.kind != kind]

    def build(self, prices: pd.DataFrame, corr_threshold: float = 0.7, coint_level: str = "5%", max_lag: int = 5,
              method: str = "spearman") -> Dict[str, int]:
        """Derive statistical relations from a price panel (one column per symbol). Replaces earlier statistical
        relations; declared ones are kept. A pair whose statistic cannot be computed is recorded in ``errors`` and
        skipped, so one short or constant series does not stop the build."""
        if prices.shape[1] < 2:
            raise ValidationError("need at least two instruments")
        for kind in ("correlated", "cointegrated", "leads", "sector"):
            self.clear(kind)
        self.errors = []
        returns = prices.pct_change().iloc[1:]
        corr = correlation_matrix(returns, method)
        cols = list(prices.columns)
        for i, a in enumerate(cols):
            for b in cols[i + 1:]:
                c = float(corr.loc[a, b])                               # type: ignore[arg-type]
                if np.isfinite(c) and abs(c) >= corr_threshold:
                    self.add(a, b, "correlated", c)
                if self.sectors.get(a) and self.sectors.get(a) == self.sectors.get(b):
                    self.add(a, b, "sector", 1.0, sector=self.sectors[a])
                try:
                    ll = lead_lag(returns[a].to_numpy(), returns[b].to_numpy(), max_lag)
                    if ll["significant"]:
                        lag = int(ll["best_lag"])                           # type: ignore[call-overload]
                        lead, follow = (a, b) if lag > 0 else (b, a)
                        self.add(lead, follow, "leads", float(ll["correlation"]), directed=True, lag=abs(lag))   # type: ignore[arg-type]
                except PlatformError as exc:
                    self.errors.append(f"{a}/{b} lead-lag: {exc.message}")
        try:
            pairs = find_cointegrated_pairs(prices, coint_level)
            self.errors += [f"cointegration {msg}" for msg in pairs.attrs.get("skipped", [])]
            for rec in pairs.to_dict("records"):
                self.add(str(rec["y"]), str(rec["x"]), "cointegrated", float(rec["stat"]), hedge_ratio=float(rec["hedge_ratio"]),
                         half_life=float(rec["half_life"]))
        except PlatformError as exc:
            self.errors.append(f"cointegration scan: {exc.message}")
        return self.summary()

    # ---- questions ----------------------------------------------------------------------------------------------------------
    def relations(self, kind: Optional[str] = None) -> List[Relation]:
        return [r for r in self._relations if kind is None or r.kind == kind]

    def related(self, symbol: str, kind: Optional[str] = None) -> List[Relation]:
        return [r for r in self.relations(kind) if symbol in (r.a, r.b)]

    def peers(self, symbol: str, n: int = 5, kind: str = "correlated") -> List[str]:
        """The instruments most strongly related to ``symbol`` (by absolute weight)."""
        rel = sorted(self.related(symbol, kind), key=lambda r: -abs(r.weight))
        return [r.other(symbol) for r in rel[:n]]

    def pairs(self, kind: str = "cointegrated") -> List[Relation]:
        return sorted(self.relations(kind), key=lambda r: r.weight)

    def leaders(self, symbol: str) -> List[Relation]:
        """Instruments whose moves tend to precede ``symbol``'s."""
        return [r for r in self.relations("leads") if r.b == symbol]

    def groups(self, kind: str = "correlated", positive_only: bool = True) -> List[Set[str]]:
        """Connected components of the relation graph: sets of instruments that are, transitively, one exposure."""
        adj: Dict[str, Set[str]] = {}
        for r in self.relations(kind):
            if positive_only and r.weight < 0:
                continue
            adj.setdefault(r.a, set()).add(r.b)
            adj.setdefault(r.b, set()).add(r.a)
        seen: Set[str] = set()
        out: List[Set[str]] = []
        for start in sorted(adj):
            if start in seen:
                continue
            comp, stack = set(), [start]
            while stack:
                node = stack.pop()
                if node not in comp:
                    comp.add(node)
                    stack.extend(adj[node] - comp)
            seen |= comp
            out.append(comp)
        return sorted(out, key=lambda g: (-len(g), sorted(g)))

    def concentration(self, weights: Mapping[str, float], kind: str = "correlated") -> Dict[str, Any]:
        """How concentrated a set of positions is once related instruments are counted as one bet.

        ``effective_bets`` is the inverse Herfindahl index of *group* weights: ten equal positions in one correlated
        group are one bet, not ten."""
        gross = sum(abs(w) for w in weights.values())
        if gross <= 0:
            return {"groups": [], "largest_share": 0.0, "effective_bets": 0.0, "positions": 0}
        grouped = [g & set(weights) for g in self.groups(kind)]
        grouped = [g for g in grouped if g]
        covered = set().union(*grouped) if grouped else set()
        buckets = grouped + [{s} for s in weights if s not in covered and weights[s] != 0]
        shares = sorted(((sorted(g), abs(sum(weights[s] for s in g)) / gross) for g in buckets), key=lambda x: -x[1])
        hhi = sum(s ** 2 for _, s in shares)
        return {"groups": [{"symbols": g, "share": round(s, 6)} for g, s in shares], "largest_share": shares[0][1] if shares else 0.0,
                "effective_bets": 1.0 / hhi if hhi > 0 else 0.0, "positions": sum(1 for w in weights.values() if w != 0)}

    def summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for r in self._relations:
            out[r.kind] = out.get(r.kind, 0) + 1
        return out

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"a": r.a, "b": r.b, "kind": r.kind, "weight": r.weight, "directed": r.directed, **dict(r.meta)}
                             for r in self._relations])
