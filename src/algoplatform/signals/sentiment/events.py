"""Headline event classification and ticker/entity mapping."""
from __future__ import annotations

import re
from typing import Iterable, List, Mapping, Set, Tuple

# (event name, regex, direction weight in [-1, 1])
EVENT_RULES: List[Tuple[str, "re.Pattern[str]", float]] = [(n, re.compile(p, re.I), w) for n, p, w in [
    ("earnings_beat", r"\b(beats?|tops?|exceeds?|surpass\w*)\b.*\b(estimates?|expectations?|forecasts?|consensus)\b", 0.7),
    ("earnings_miss", r"\b(misses|missed|falls? short|below)\b.*\b(estimates?|expectations?|forecasts?|consensus)\b", -0.7),
    ("guidance_raise", r"\b(raises?|lifts?|boosts?|hikes?)\b.*\b(guidance|outlook|forecast|target)\b", 0.6),
    ("guidance_cut", r"\b(cuts?|lowers?|slashes?|trims?|withdraws?)\b.*\b(guidance|outlook|forecast)\b", -0.7),
    ("upgrade", r"\bupgrad\w+", 0.5), ("downgrade", r"\bdowngrad\w+", -0.5),
    ("merger", r"\b(to acquire|acquires?|agrees? to buy|takeover|merger|buyout)\b", 0.4),
    ("buyback", r"\b(buyback|repurchase)\b", 0.3), ("dividend_hike", r"\b(raises?|increases?|hikes?)\b.*\bdividend\b", 0.4),
    ("dividend_cut", r"\b(cuts?|suspends?|eliminates?)\b.*\bdividend\b", -0.6),
    ("lawsuit", r"\b(lawsuit|sued|class action|subpoena|probe|investigation|sec charges?)\b", -0.5),
    ("recall", r"\brecall\w*", -0.5), ("bankruptcy", r"\b(bankruptcy|chapter 11|insolven\w+|default(s|ed)?)\b", -1.0),
    ("fraud", r"\b(fraud|accounting irregularit\w+|restat\w+)\b", -0.9),
    ("approval", r"\b(fda approv\w+|regulatory approval|approved by)\b", 0.6)]]


def classify_event(headline: str) -> List[Tuple[str, float]]:
    """All event types a headline matches, with their directional weights."""
    return [(name, w) for name, rx, w in EVENT_RULES if rx.search(headline)]


class EntityMapper:
    """Map headline text to tickers using aliases (company names, brands, tickers). Whole-word, case-insensitive;
    short all-caps tickers are matched case-sensitively so 'ON' or 'A' don't match ordinary words."""

    def __init__(self, aliases: Mapping[str, Iterable[str]]) -> None:
        self._rules: List[Tuple[str, "re.Pattern[str]"]] = []
        for symbol, names in aliases.items():
            # a 1-letter ticker would match the article "A"; such symbols match only through explicit aliases
            for name in list(names) + ([symbol] if len(symbol) > 1 else []):
                flags = 0 if (name.isupper() and len(name) <= 4) else re.I
                self._rules.append((symbol, re.compile(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", flags)))

    def symbols(self, text: str) -> Set[str]:
        return {sym for sym, rx in self._rules if rx.search(text)}
