"""Finance-flavoured lexicon sentiment with negation and intensifiers (transparent, auditable, no model download)."""
from __future__ import annotations

import re
from typing import Dict

POSITIVE = set("""beat beats exceeded exceeds surpass surpassed strong stronger strength growth grew grow gains gain gained rally
rallied surge surged soar soared record profit profitable improved improve improvement upgrade upgraded outperform outperformed
raise raised raises bullish optimistic robust resilient accelerate accelerated expansion expand approved approval win won
breakthrough dividend buyback recovery rebound upside favorable successful success momentum""".split())
NEGATIVE = set("""miss missed misses weak weaker weakness decline declined declines fall fell drop dropped plunge plunged slump slumped
loss losses lawsuit sued probe investigation fraud default bankruptcy downgrade downgraded underperform cut cuts slash slashed
warning warns bearish pessimistic concern concerns risk risks recall layoffs layoff delay delayed shortfall impairment writedown
restatement subpoena fine fined penalty halted suspended downside unfavorable failure failed volatile""".split())
NEGATORS = {"not", "no", "never", "without", "neither", "nor", "none", "hardly", "barely", "cannot", "cant", "wont", "didnt",
            "doesnt", "isnt", "wasnt", "fails", "failed"}
INTENSIFIERS = {"very": 1.5, "significantly": 1.5, "sharply": 1.5, "substantially": 1.5, "strongly": 1.5, "slightly": 0.5,
                "marginally": 0.5, "somewhat": 0.6}
# Multi-word phrases whose meaning differs from their parts ("profit" is good, "profit warning" is not).
PHRASES = {"profit warning": -1.5, "going concern": -1.5, "short seller": -1.0, "accounting irregularities": -1.5,
           "credit downgrade": -1.5, "class action": -1.0, "all time high": 1.5, "record high": 1.5, "beat expectations": 1.5,
           "raised guidance": 1.5, "price target raised": 1.0, "price target cut": -1.0}
_TOKEN = re.compile(r"[a-z']+")


def tokenize(text: str) -> list:
    return [t.replace("'", "") for t in _TOKEN.findall(text.lower())]


def score_text(text: str, negation_window: int = 3) -> Dict[str, float]:
    """Score in [-1, 1] = (pos - neg) / (pos + neg). A negator within ``negation_window`` words flips a word's sign
    ('not strong' is negative); intensifiers scale it. Text with no sentiment words scores 0 with ``n_terms`` = 0."""
    lowered = re.sub(r"[-_]", " ", text.lower())
    pos = neg = 0.0
    for phrase, w in PHRASES.items():
        n = lowered.count(phrase)
        if n:
            lowered = lowered.replace(phrase, " ")                 # consume it so its parts are not counted again
            pos, neg = (pos + w * n, neg) if w > 0 else (pos, neg - w * n)
    toks = tokenize(lowered)
    for i, t in enumerate(toks):
        sign = 1.0 if t in POSITIVE else -1.0 if t in NEGATIVE else 0.0
        if not sign:
            continue
        window = toks[max(0, i - negation_window):i]
        if any(w in NEGATORS for w in window):
            sign = -sign
        weight = 1.0
        for w in window:
            weight *= INTENSIFIERS.get(w, 1.0)
        if sign > 0:
            pos += weight
        else:
            neg += weight
    total = pos + neg
    return {"score": (pos - neg) / total if total else 0.0, "pos": pos, "neg": neg, "n_terms": int(round(total))}
