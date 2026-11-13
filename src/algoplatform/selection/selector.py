"""UniverseSelector: decide *what* to trade before deciding *when*.

A selection is a small pipeline, declared fluently and run point-in-time:

    result = (UniverseSelector()
              .require_history(252).price_between(5, None).min_dollar_volume(2e6)       # 1. eligibility filters
              .rank_by("momentum", lookback=126).rank_by("low_volatility", weight=0.5)  # 2. composite score
              .diversify(max_correlation=0.8).per_group(sectors, 2)                    # 3. concentration limits
              .top(10)                                                                # 4. how many
              .select(market_data, asof="2023-06-30"))

Selection methods available: liquidity / price / history / volatility screens, any custom predicate, ranking by any
registered ranker (momentum, risk-adjusted momentum, low volatility, trend quality, mean reversion, relative strength,
liquidity, proximity to highs) or by external scores (fundamental factor composites, sentiment), correlation-aware
diversification, per-sector caps, one-per-cluster selection and cointegrated pair selection.

Everything is evaluated on data up to ``asof`` only, so the same selector can be re-run through history without
look-ahead. Note what this cannot fix: if the *list of symbols you feed it* contains only today's survivors, the
backtest still has survivorship bias. Feed it the universe as it stood at each date.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.exceptions import ValidationError
from ..relationships.cointegration import find_cointegrated_pairs
from ..relationships.structure import hierarchical_clusters
from .rankers import RANKERS

Frames = Mapping[str, pd.DataFrame]


@dataclass
class SelectionResult:
    symbols: List[str]                                   # chosen, best first
    scores: pd.DataFrame                                 # eligible symbols: each ranker's raw value, percentile ranks, composite
    rejected: Dict[str, str] = field(default_factory=dict)   # symbol -> why it was left out
    asof: Any = None

    def weights(self, scheme: str = "equal") -> Dict[str, float]:
        """Starting weights for the selection: ``equal`` or ``score`` (proportional to composite score)."""
        if not self.symbols:
            return {}
        if scheme == "equal":
            return {s: 1.0 / len(self.symbols) for s in self.symbols}
        if scheme == "score":
            sc = self.scores.loc[self.symbols, "composite"].clip(lower=0)
            return {str(k): float(v) for k, v in (sc / sc.sum()).items()} if sc.sum() > 0 else {s: 1.0 / len(self.symbols) for s in self.symbols}
        raise ValidationError("scheme must be 'equal' or 'score'")


def _frames(data: Any, asof: Any) -> Dict[str, pd.DataFrame]:
    """Accept a MarketData-like object (``symbols`` + ``frame``) or a mapping, and cut every frame off at ``asof``."""
    raw = {s: data.frame(s) for s in data.symbols} if hasattr(data, "frame") else dict(data)
    if not raw:
        raise ValidationError("no instruments to select from")
    if asof is None:
        return raw
    ts = pd.Timestamp(asof)
    return {s: df.loc[:ts] for s, df in raw.items()}


class UniverseSelector:
    def __init__(self) -> None:
        self._filters: List[Tuple[str, Callable[[pd.DataFrame], bool]]] = []
        self._rankers: List[Tuple[str, Callable[[pd.DataFrame], float], float, bool]] = []
        self._external: List[Tuple[str, Mapping[str, float], float, bool]] = []
        self._n: Optional[int] = None
        self._quantile: Optional[float] = None
        self._min_score: Optional[float] = None
        self._max_corr: Optional[Tuple[float, int]] = None
        self._groups: Optional[Tuple[Mapping[str, str], int]] = None

    # ---- 1. eligibility ---------------------------------------------------------------------------------------------------
    def where(self, reason: str, predicate: Callable[[pd.DataFrame], bool]) -> "UniverseSelector":
        """Keep instruments for which ``predicate(frame)`` is true; ``reason`` is recorded for those that fail."""
        self._filters.append((reason, predicate))
        return self

    def require_history(self, bars: int) -> "UniverseSelector":
        return self.where(f"fewer than {bars} bars of history", lambda df: len(df) >= bars)

    def price_between(self, low: Optional[float] = None, high: Optional[float] = None) -> "UniverseSelector":
        return self.where(f"price outside [{low}, {high}]",
                          lambda df: len(df) > 0 and (low is None or df["close"].iloc[-1] >= low) and (high is None or df["close"].iloc[-1] <= high))

    def min_dollar_volume(self, value: float, window: int = 20) -> "UniverseSelector":
        return self.where(f"average traded value below {value:g}",
                          lambda df: len(df) >= window and float((df["close"] * df["volume"]).iloc[-window:].mean()) >= value)

    def volatility_between(self, low: float = 0.0, high: float = float("inf"), window: int = 63, periods: float = 252.0) -> "UniverseSelector":
        def ok(df: pd.DataFrame) -> bool:
            r = np.diff(np.log(df["close"].to_numpy(float)[-(window + 1):]))
            return len(r) == window and low <= float(r.std(ddof=1) * np.sqrt(periods)) <= high
        return self.where(f"volatility outside [{low:g}, {high:g}]", ok)

    def max_stale_share(self, share: float = 0.1, window: int = 60) -> "UniverseSelector":
        """Reject instruments that often do not trade (zero-volume bars), whose prices are not really tradable."""
        return self.where(f"more than {share:.0%} zero-volume bars",
                          lambda df: len(df) >= window and float((df["volume"].iloc[-window:] <= 0).mean()) <= share)

    # ---- 2. ranking -----------------------------------------------------------------------------------------------------------
    def rank_by(self, ranker: Any, weight: float = 1.0, ascending: bool = False, **params: Any) -> "UniverseSelector":
        """Add a ranking criterion: a registered ranker name or ``fn(frame) -> float``. By default higher is better;
        ``ascending=True`` prefers low values. Several criteria are combined as a weighted average of percentile ranks,
        so they need not share units."""
        if weight <= 0:
            raise ValidationError("weight must be positive")
        if callable(ranker):
            name, fn = getattr(ranker, "__name__", "custom"), (lambda df, f=ranker: f(df, **params))
        else:
            if ranker not in RANKERS:
                raise ValidationError(f"unknown ranker '{ranker}'", details={"valid": sorted(RANKERS)})
            name, fn = ranker, (lambda df, f=RANKERS[ranker]: f(df, **params))
        label = name if name not in [r[0] for r in self._rankers] else f"{name}_{len(self._rankers)}"
        self._rankers.append((label, fn, float(weight), ascending))
        return self

    def rank_by_scores(self, name: str, scores: Mapping[str, float], weight: float = 1.0, ascending: bool = False) -> "UniverseSelector":
        """Rank by numbers computed elsewhere (a fundamental factor composite, a sentiment score, a model forecast).
        The caller is responsible for those being point-in-time as of the selection date."""
        if weight <= 0:
            raise ValidationError("weight must be positive")
        self._external.append((name, scores, float(weight), ascending))
        return self

    # ---- 3. concentration limits --------------------------------------------------------------------------------------------------
    def diversify(self, max_correlation: float = 0.8, window: int = 126) -> "UniverseSelector":
        """Walk down the ranking and skip any instrument whose returns correlate above ``max_correlation`` with one
        already chosen: ten picks that are one trade are not ten picks."""
        if not -1 < max_correlation <= 1 or window < 20:
            raise ValidationError("need -1 < max_correlation <= 1 and window >= 20")
        self._max_corr = (max_correlation, window)
        return self

    def per_group(self, groups: Mapping[str, str], max_per_group: int = 2) -> "UniverseSelector":
        """At most ``max_per_group`` instruments from each group (sector, country, cluster). Ungrouped instruments each
        count as their own group."""
        if max_per_group < 1:
            raise ValidationError("max_per_group must be >= 1")
        self._groups = (groups, max_per_group)
        return self

    # ---- 4. how many ------------------------------------------------------------------------------------------------------------------
    def top(self, n: int) -> "UniverseSelector":
        if n < 1:
            raise ValidationError("n must be >= 1")
        self._n = n
        return self

    def top_quantile(self, q: float) -> "UniverseSelector":
        if not 0 < q <= 1:
            raise ValidationError("q must be in (0, 1]")
        self._quantile = q
        return self

    def min_score(self, value: float) -> "UniverseSelector":
        """Require a composite score (0..1 percentile scale) of at least ``value``."""
        self._min_score = value
        return self

    # ---- run -----------------------------------------------------------------------------------------------------------------------------
    def select(self, data: Any, asof: Any = None) -> SelectionResult:
        frames = _frames(data, asof)
        rejected: Dict[str, str] = {}
        eligible: List[str] = []
        for sym, df in frames.items():
            reason = None
            for why, pred in self._filters:
                try:
                    if not pred(df):
                        reason = why
                        break
                except Exception as exc:  # noqa: BLE001 - a broken frame rejects that instrument only
                    reason = f"filter error: {exc}"
                    break
            if reason:
                rejected[sym] = reason
            else:
                eligible.append(sym)
        raw: Dict[str, Dict[str, float]] = {s: {} for s in eligible}
        directions: Dict[str, Tuple[float, bool]] = {}
        for label, fn, weight, asc in self._rankers:
            directions[label] = (weight, asc)
            for s in eligible:
                try:
                    raw[s][label] = float(fn(frames[s]))
                except Exception as exc:  # noqa: BLE001
                    raw[s][label] = float("nan")
                    rejected.setdefault(s, f"ranker '{label}' failed: {exc}")
        for label, scores, weight, asc in self._external:
            directions[label] = (weight, asc)
            for s in eligible:
                raw[s][label] = float(scores.get(s, float("nan")))
        table = pd.DataFrame.from_dict(raw, orient="index")
        if directions:
            for s in table.index[table[list(directions)].isna().any(axis=1)]:
                rejected.setdefault(s, "no score (insufficient history for a ranker)")
            table = table.dropna(subset=list(directions))
            total = sum(w for w, _ in directions.values())
            composite = pd.Series(0.0, index=table.index)
            for label, (weight, asc) in directions.items():
                pct = table[label].rank(pct=True, ascending=not asc) if len(table) > 1 else pd.Series(1.0, index=table.index)
                table[f"{label}_rank"] = pct
                composite += weight * pct
            table["composite"] = composite / total
        else:
            table = pd.DataFrame(index=[s for s in eligible if s not in rejected])
            table["composite"] = 1.0
        table = table.sort_values("composite", ascending=False, kind="mergesort")
        ordered = list(table.index)
        if self._min_score is not None:
            for s in [s for s in ordered if table.loc[s, "composite"] < self._min_score]:
                rejected[s] = f"composite below {self._min_score:g}"
            ordered = [s for s in ordered if table.loc[s, "composite"] >= self._min_score]
        limit = self._n
        if self._quantile is not None:
            limit = min(limit or len(ordered), max(1, int(np.ceil(self._quantile * len(ordered)))))
        chosen = self._apply_limits(ordered, frames, rejected, limit)
        return SelectionResult(chosen, table, rejected, asof)

    def _apply_limits(self, ordered: Sequence[str], frames: Frames, rejected: Dict[str, str], limit: Optional[int]) -> List[str]:
        chosen: List[str] = []
        counts: Dict[str, int] = {}
        returns = None
        if self._max_corr and len(ordered) > 1:
            window = self._max_corr[1]
            returns = pd.DataFrame({s: frames[s]["close"].pct_change().iloc[-window:] for s in ordered})
        for s in ordered:
            if limit is not None and len(chosen) >= limit:
                rejected.setdefault(s, "ranked below the cut-off")
                continue
            if self._groups:
                group = self._groups[0].get(s, f"__{s}")
                if counts.get(group, 0) >= self._groups[1]:
                    rejected[s] = f"group '{group}' already has {self._groups[1]}"
                    continue
            if returns is not None and chosen:
                corr = returns[chosen].corrwith(returns[s])
                if (corr > self._max_corr[0]).any():                        # type: ignore[index]
                    rejected[s] = f"correlation {corr.max():.2f} with {corr.idxmax()} exceeds {self._max_corr[0]:g}"   # type: ignore[index]
                    continue
            chosen.append(s)
            if self._groups:
                counts[group] = counts.get(group, 0) + 1
        return chosen


def select_one_per_cluster(data: Any, n_clusters: int, ranker: str = "risk_adjusted_momentum", asof: Any = None, window: int = 126,
                           **params: Any) -> SelectionResult:
    """Group instruments by return correlation, then take the best-ranked one from each group: a diversified shortlist
    that needs no sector labels."""
    frames = _frames(data, asof)
    usable = {s: df for s, df in frames.items() if len(df) > window}
    if len(usable) < max(2, n_clusters):
        raise ValidationError("not enough instruments with history for the requested clusters", details={"usable": len(usable)})
    returns = pd.DataFrame({s: df["close"].pct_change().iloc[-window:] for s, df in usable.items()})
    labels: pd.Series = hierarchical_clusters(returns.corr(), n_clusters=n_clusters)["labels"]     # type: ignore[assignment]
    result = UniverseSelector().rank_by(ranker, **params).per_group({str(s): f"cluster_{c}" for s, c in labels.items()}, 1).select(usable)
    result.scores["cluster"] = labels.reindex(result.scores.index)
    result.asof = asof
    for s in frames:
        if s not in usable:
            result.rejected[s] = f"fewer than {window + 1} bars of history"
    return result


def select_pairs(data: Any, asof: Any = None, level: str = "5%", max_half_life: float = 60.0, top: Optional[int] = None) -> pd.DataFrame:
    """Cointegrated pairs among the instruments, strongest first (see ``relationships.find_cointegrated_pairs`` for the
    multiple-testing caveat that travels with the result)."""
    frames = _frames(data, asof)
    prices = pd.DataFrame({s: df["close"] for s, df in frames.items()})
    pairs = find_cointegrated_pairs(prices, level=level, max_half_life=max_half_life)
    return pairs.head(top) if top else pairs
