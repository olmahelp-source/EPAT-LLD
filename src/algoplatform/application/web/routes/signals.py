"""Signal libraries: the technical lab (indicators, patterns, formulas), news sentiment, the function catalogue.

Every route checks the caller's permission on the server (through the role desks or ``UserManager.authorize``); hiding a
page in the browser is a convenience, not the control."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ....foundation.core.exceptions import ValidationError
from ....signals.sentiment import classify_event, EntityMapper, score_text
from ....signals.technical import describe, Expression, operators as ops, scan_candles, scan_chart_patterns, support_resistance
from ....signals.technical.catalog import compute
from ....access.users.manager import User

from ..common import date_at, last_value, series_points


MAX_HEADLINES = 100


def register(app: Any) -> None:
    r, P = app.route, app.platform

    def research(u: User):
        return P.desk(u, "research")                              # raises AuthorizationError for roles without research

    def first_symbol(u: User, b: Dict[str, Any]):
        md, symbols = app._data(b, u)
        return md.frame(symbols[0]), symbols[0], md, symbols

    @r("GET", "/api/functions")
    def functions(_u, _b):
        return describe()

    @r("POST", "/api/lab/technical")
    def technical(u, b):
        research(u)
        df, sym, _, _ = first_symbol(u, b)
        h, l, c = df["high"], df["low"], df["close"]
        out: Dict[str, Any] = {"symbol": sym, "close": series_points(c)}
        st = compute("supertrend", df, n=10, mult=3.0)
        out["indicators"] = {"rsi": last_value(compute("rsi", df)), "adx": last_value(compute("adx", df, output="adx")), "supertrend": last_value(st["direction"]),
                             "natr": last_value(compute("natr", df)), "choppiness": last_value(compute("choppiness", df)),
                             "relative_volume": last_value(compute("relative_volume", df))}
        text = (b.get("formula") or "").strip()
        if text:
            expr = Expression(text)
            raw = expr.evaluate(df)
            value = np.asarray(raw)
            if value.ndim == 0:
                value = np.full(len(df), value.item())
            is_condition = value.dtype == bool
            mask = expr.mask(df)
            starts = np.flatnonzero(ops.rising_edge(mask))
            out["formula"] = {"text": expr.text, "condition": bool(is_condition), "true_bars": int(mask.sum()), "bars": len(df),
                              "columns": sorted(expr.columns), "functions": sorted(expr.calls),
                              "events": [{"date": date_at(df, int(i)), "close": float(c.iloc[int(i)])} for i in starts[-40:]][::-1],
                              "series": None if is_condition else series_points(pd.Series(value.astype(float), index=df.index).dropna())}
        found = scan_chart_patterns(df)
        out["chart_patterns"] = [{"name": m.name, "direction": m.direction, "date": date_at(df, min(m.confirm, len(df) - 1)), "level": m.level,
                                  "target": m.target} for m in found][-30:][::-1]
        recent = scan_candles(df).tail(int(b.get("candle_window", 20)))
        counts = (recent.drop(columns="score") != 0).sum()
        out["candles"] = [{"pattern": k, "count": int(v), "last_signal": int(recent[k][recent[k] != 0].iloc[-1])} for k, v in counts.items() if v > 0]
        out["candle_score"] = int(recent["score"].sum())
        out["levels"] = [{"price": lv["price"], "kind": lv["kind"], "touches": lv["touches"]} for lv in support_resistance(h, l, c)[:8]]
        return out

    @r("POST", "/api/lab/sentiment")
    def sentiment(u, b):
        research(u)
        lines = [x.strip()[:300] for x in str(b.get("headlines", "")).splitlines() if x.strip()]
        if not lines:
            raise ValidationError("enter at least one headline, one per line")
        if len(lines) > MAX_HEADLINES:
            raise ValidationError(f"at most {MAX_HEADLINES} headlines per request", details={"given": len(lines)})
        aliases: Dict[str, List[str]] = {}
        for line in str(b.get("aliases", "")).splitlines():
            if not line.strip():
                continue
            sym, _, names = line.partition("=")
            sym = sym.strip().upper()
            if not sym.isalnum() or len(sym) > 8:
                raise ValidationError(f"alias line '{line.strip()[:40]}': write SYMBOL = name, other name")
            aliases[sym] = [n.strip() for n in names.split(",") if n.strip()]
        mapper = EntityMapper(aliases) if aliases else None
        rows = []
        for text in lines:
            lex = score_text(text)
            events = classify_event(text)
            event_score = max(-1.0, min(1.0, sum(w for _, w in events)))
            # an identified event is stronger evidence than word counts, so it gets the larger share when present
            combined = 0.6 * event_score + 0.4 * lex["score"] if events else lex["score"]
            rows.append({"headline": text, "score": lex["score"], "terms": lex["n_terms"], "events": [name for name, _ in events],
                         "event_score": event_score, "combined": combined, "symbols": sorted(mapper.symbols(text)) if mapper else []})
        by_symbol: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            for sym in row["symbols"]:
                agg = by_symbol.setdefault(sym, {"symbol": sym, "headlines": 0, "total": 0.0, "events": []})
                agg["headlines"] += 1
                agg["total"] += row["combined"]
                agg["events"] += row["events"]
        for agg in by_symbol.values():
            agg["average"] = agg.pop("total") / agg["headlines"]
            agg["events"] = sorted(set(agg["events"]))
        scored = [x["combined"] for x in rows if x["terms"] or x["events"]]
        return {"headlines": rows, "by_symbol": sorted(by_symbol.values(), key=lambda a: -abs(a["average"])),
                "average": float(np.mean(scored)) if scored else 0.0, "neutral": len(rows) - len(scored)}
