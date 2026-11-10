"""Point-in-time news store, de-duplication, source weighting and a decayed sentiment signal."""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Mapping, Optional, Sequence, Set


from ..core.exceptions import StrategyError, ValidationError
from ..core.models import Signal
from ..strategy.base import STRATEGIES, Strategy, StrategyContext
from .events import EntityMapper, classify_event
from .lexicon import score_text


@dataclass
class NewsItem:
    published: datetime
    source: str
    headline: str
    body: str = ""
    symbols: Set[str] = field(default_factory=set)
    latency: timedelta = timedelta(0)          # delay between publication and when YOUR system could act on it
    score: float = 0.0
    event_weight: float = 0.0
    events: List[str] = field(default_factory=list)

    @property
    def available_at(self) -> datetime:
        return self.published + self.latency


def _fingerprint(headline: str) -> str:
    norm = re.sub(r"[^a-z0-9 ]", "", headline.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    return hashlib.sha1(norm.encode()).hexdigest()


class NewsStore:
    """Annotates, de-duplicates and serves news strictly as-of a timestamp (never before ``available_at``)."""

    def __init__(self, mapper: EntityMapper, default_latency: timedelta = timedelta(seconds=30),
                 source_weights: Optional[Mapping[str, float]] = None, unknown_source_weight: float = 0.5,
                 dedupe_window: timedelta = timedelta(hours=12)) -> None:
        self.mapper, self.default_latency = mapper, default_latency
        self.source_weights, self.unknown_w = dict(source_weights or {}), unknown_source_weight
        self.dedupe_window = dedupe_window
        self._items: List[NewsItem] = []
        self._seen: Dict[str, datetime] = {}
        self.duplicates = 0

    def add(self, published: datetime, source: str, headline: str, body: str = "", latency: Optional[timedelta] = None) -> Optional[NewsItem]:
        """Returns the stored item, or None if it was a near-duplicate (the same story repeated by many outlets or
        bots must count once, otherwise a coordinated pump becomes a signal)."""
        if not headline.strip():
            raise ValidationError("empty headline")
        fp = _fingerprint(headline)
        last = self._seen.get(fp)
        if last is not None and abs(published - last) <= self.dedupe_window:
            self.duplicates += 1
            return None
        self._seen[fp] = published
        text = f"{headline}. {body}"
        ev = classify_event(headline)
        item = NewsItem(published, source, headline, body, self.mapper.symbols(text),
                        latency if latency is not None else self.default_latency,
                        score_text(headline)["score"] * 0.7 + score_text(body)["score"] * 0.3 if body else score_text(headline)["score"],
                        max(-1.0, min(1.0, sum(w for _, w in ev))), [n for n, _ in ev])
        self._items.append(item)
        return item

    def asof(self, symbol: str, ts: datetime, window: timedelta) -> List[NewsItem]:
        lo = ts - window
        return [i for i in self._items if symbol in i.symbols and lo <= i.available_at <= ts]

    def __len__(self) -> int:
        return len(self._items)


class SentimentSignal:
    """Decayed sum of item sentiment: recent news counts most (half-life), reputable sources count more, and an
    explicit event (guidance cut, FDA approval...) adds its own directional weight. Output is squashed to (-1, 1)."""

    def __init__(self, store: NewsStore, half_life: timedelta = timedelta(hours=24), window: timedelta = timedelta(days=5),
                 event_weight: float = 1.0, scale: float = 1.0) -> None:
        self.store, self.half_life, self.window, self.event_weight, self.scale = store, half_life, window, event_weight, scale

    def score(self, symbol: str, ts: datetime) -> float:
        total = 0.0
        for it in self.store.asof(symbol, ts, self.window):
            age = (ts - it.available_at).total_seconds()
            decay = math.exp(-math.log(2) * age / self.half_life.total_seconds())
            w = self.store.source_weights.get(it.source, self.store.unknown_w)
            total += w * decay * (it.score + self.event_weight * it.event_weight)
        return math.tanh(total * self.scale)

    def scores(self, symbols: Sequence[str], ts: datetime) -> Dict[str, float]:
        return {s: self.score(s, ts) for s in symbols}


@STRATEGIES.register("sentiment")
class SentimentStrategy(Strategy):
    """Go long names with strongly positive recent news (short strongly negative if allowed). Attach a
    :class:`SentimentSignal` with :meth:`with_signal`."""
    name = "sentiment"
    PARAMS = {"threshold": 0.3, "allow_short": False}
    PARAM_BOUNDS = {"threshold": (0.05, 0.9)}

    def validate_params(self) -> None:
        if not 0 < self.p("threshold") < 1:
            raise StrategyError("threshold must be in (0, 1)", details=self.params)
        self.signal: Optional[SentimentSignal] = getattr(self, "signal", None)

    @classmethod
    def with_signal(cls, signal: SentimentSignal, symbols, **params) -> "SentimentStrategy":
        s = cls(symbols, **params)
        s.signal = signal
        return s

    def clone(self, **overrides) -> "SentimentStrategy":
        c = SentimentStrategy.with_signal(self.signal, self.symbols, **{**self.params, **overrides})
        c.id = self.id
        return c

    def on_bar(self, ctx: StrategyContext) -> List[Signal]:
        if self.signal is None:
            raise StrategyError("attach a SentimentSignal with SentimentStrategy.with_signal(...)")
        out = []
        for s in self.symbols:
            if s not in ctx.bars:
                continue
            sc = self.signal.score(s, ctx.ts.to_pydatetime() if hasattr(ctx.ts, "to_pydatetime") else ctx.ts)
            thr = float(self.p("threshold"))
            w = 1.0 if sc >= thr else (-1.0 if sc <= -thr and self.p("allow_short") else 0.0)
            out.append(Signal(s, w, f"sentiment={sc:+.2f}"))
        return out
