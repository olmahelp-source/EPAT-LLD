"""API routes for the research lab, governance, alerts and the platform map.

Kept apart from ``server.py`` (sessions, backtests, data upload) so each file has one job. Every route goes through
``TradingPlatform`` and its role desks, so permissions are enforced on the server exactly as they are from Python.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..architecture import GROUPS, check_architecture
from ..core.enums import Permission, Severity
from ..core.exceptions import AuthorizationError, ValidationError
from ..notify.alerts import threshold
from ..notify.channels import MemoryChannel
from ..relationships.manager import RelationshipManager
from ..selection.rankers import RANKERS
from ..selection.selector import UniverseSelector
from ..technical import (Expression, describe, scan_candles, scan_chart_patterns, support_resistance)
from ..technical import operators as ops
from ..technical.catalog import compute
from ..users.manager import User, UserManager
from ..volatility import (EWMAVol, GarchVol, HARVol, HistoricalVol, VolatilityManager, compare_vol_models, estimators, volatility_cone)

VOL_MODELS = {"ewma": EWMAVol, "hist": HistoricalVol, "har": HARVol, "garch": GarchVol}
ALERT_METRICS = ("drawdown", "daily_pnl_pct", "equity", "gross_leverage", "strategy_errors")


def _last(x: Any) -> float:
    a = np.asarray(x, float)
    return float(a[-1]) if len(a) else float("nan")


def _date(df: pd.DataFrame, i: int) -> str:
    return pd.Timestamp(df.index[i]).strftime("%Y-%m-%d")


def register(app: Any) -> None:
    from .server import series_points                             # deferred: server imports this module
    r, P = app.route, app.platform

    def research(u: User):
        return P.desk(u, "research")                              # raises AuthorizationError for roles without research

    def first_symbol(u: User, b: Dict[str, Any]):
        md, symbols = app._data(b, u)
        return md.frame(symbols[0]), symbols[0], md, symbols

    # ---- platform map -------------------------------------------------------------------------------------------------
    violations: List[List[str]] = []                              # the source does not change while the server runs: scan once

    @r("GET", "/api/structure")
    def structure(u, _b):
        if not violations:
            violations.append(check_architecture())
        return {"role": u.role, "desks": sorted(P.desks(u)),
                "groups": [{"key": g.key, "title": g.title, "job": g.job, "packages": list(g.packages), "roles": [x.value for x in g.roles]} for g in GROUPS],
                "violations": violations[0]}

    @r("GET", "/api/functions")
    def functions(_u, _b):
        return describe()

    @r("GET", "/api/lab/options")
    def options(_u, _b):
        return {"rankers": sorted(RANKERS), "vol_models": sorted(VOL_MODELS), "alert_metrics": list(ALERT_METRICS)}

    # ---- research lab -------------------------------------------------------------------------------------------------
    @r("POST", "/api/lab/technical")
    def technical(u, b):
        research(u)
        df, sym, _, _ = first_symbol(u, b)
        h, l, c = df["high"], df["low"], df["close"]
        out: Dict[str, Any] = {"symbol": sym, "close": series_points(c)}
        st = compute("supertrend", df, n=10, mult=3.0)
        out["indicators"] = {"rsi": _last(compute("rsi", df)), "adx": _last(compute("adx", df, output="adx")), "supertrend": _last(st["direction"]),
                             "natr": _last(compute("natr", df)), "choppiness": _last(compute("choppiness", df)),
                             "relative_volume": _last(compute("relative_volume", df))}
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
                              "events": [{"date": _date(df, int(i)), "close": float(c.iloc[int(i)])} for i in starts[-40:]][::-1],
                              "series": None if is_condition else series_points(pd.Series(value.astype(float), index=df.index).dropna())}
        found = scan_chart_patterns(df)
        out["chart_patterns"] = [{"name": m.name, "direction": m.direction, "date": _date(df, min(m.confirm, len(df) - 1)), "level": m.level,
                                  "target": m.target} for m in found][-30:][::-1]
        recent = scan_candles(df).tail(int(b.get("candle_window", 20)))
        counts = (recent.drop(columns="score") != 0).sum()
        out["candles"] = [{"pattern": k, "count": int(v), "last_signal": int(recent[k][recent[k] != 0].iloc[-1])} for k, v in counts.items() if v > 0]
        out["candle_score"] = int(recent["score"].sum())
        out["levels"] = [{"price": lv["price"], "kind": lv["kind"], "touches": lv["touches"]} for lv in support_resistance(h, l, c)[:8]]
        return out

    @r("POST", "/api/lab/volatility")
    def volatility(u, b):
        research(u)
        df, sym, _, _ = first_symbol(u, b)
        o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
        rets = c.pct_change().dropna()
        window = int(b.get("window", 20))
        est = {"close_to_close": estimators.close_to_close(c, window), "parkinson": estimators.parkinson(h, l, window),
               "garman_klass": estimators.garman_klass(o, h, l, c, window), "rogers_satchell": estimators.rogers_satchell(o, h, l, c, window),
               "yang_zhang": estimators.yang_zhang(o, h, l, c, window), "ewma": estimators.ewma(c)}
        model = b.get("model", "ewma")
        if model not in VOL_MODELS:
            raise ValidationError(f"unknown volatility model '{model}'", details={"valid": sorted(VOL_MODELS)})
        target = float(b.get("target_vol", 0.10))
        view = VolatilityManager(target_vol=target, model=model).assess(rets.to_numpy())
        n = len(rets)
        train = max(250, n // 2)
        league = None
        if n >= train + 60:
            step = max(20, (n - train) // 6)                         # at most about six refits: keeps the request fast
            league = compare_vol_models(rets.to_numpy(), {k: v for k, v in VOL_MODELS.items() if b.get("league_garch", True) or k != "garch"},
                                        train=train, step=step, horizon=5).to_dict("records")
        cone = volatility_cone(c).reset_index().to_dict("records") if n > 300 else []
        return {"symbol": sym, "estimators": [{"name": k, "value": _last(v)} for k, v in est.items()], "assessment": view, "league": league,
                "cone": cone, "series": {"yang_zhang": series_points(pd.Series(est["yang_zhang"], index=df.index).dropna()),
                                         "ewma": series_points(pd.Series(est["ewma"], index=df.index).dropna())}}

    @r("POST", "/api/lab/select")
    def select(u, b):
        desk = research(u)
        md, symbols = app._data(b, u)
        if len(symbols) < 2:
            raise ValidationError("selection needs at least two instruments")
        sel = UniverseSelector().require_history(int(b.get("min_history", 130)))
        rankers = b.get("rankers") or [{"name": "momentum", "weight": 1.0}, {"name": "low_volatility", "weight": 0.5}]
        for rk in rankers:
            sel.rank_by(rk["name"], weight=float(rk.get("weight", 1.0)), ascending=bool(rk.get("ascending", False)))
        if b.get("max_correlation") not in (None, ""):
            sel.diversify(float(b["max_correlation"]))
        sel.top(int(b.get("top", 3)))
        res = desk.select_universe(sel, md)
        rel = RelationshipManager()
        summary = rel.build(md.closes(), corr_threshold=float(b.get("corr_threshold", 0.6)))
        weights = {s: 1.0 / len(res.symbols) for s in res.symbols} if res.symbols else {}
        corr = md.returns().corr()
        scores = res.scores.reset_index().rename(columns={"index": "symbol"})
        return {"selected": res.symbols, "scores": scores.to_dict("records"), "rejected": [{"symbol": s, "reason": why} for s, why in res.rejected.items()],
                "relationships": summary, "groups": [sorted(g) for g in rel.groups()],
                "pairs": [{"a": x.a, "b": x.b, "stat": x.weight, **dict(x.meta)} for x in rel.pairs()][:10],
                "leads": [{"leader": x.a, "follower": x.b, "lag": x.meta.get("lag"), "corr": x.weight} for x in rel.relations("leads")][:10],
                "concentration": rel.concentration(weights) if weights else None,
                "correlation": {s: corr.loc[s].to_dict() for s in corr.index}, "errors": rel.errors[:10]}

    # ---- governance ------------------------------------------------------------------------------------------------------
    def can(u: User) -> Dict[str, bool]:
        return {"register": u.has(Permission.MANAGE_STRATEGY), "propose": u.has(Permission.SUBMIT_STRATEGY),
                "approve": u.has(Permission.APPROVE_STRATEGY), "demote": u.has(Permission.APPROVE_STRATEGY) or u.has(Permission.MANAGE_RISK)}

    @r("GET", "/api/governance")
    def governance(u, _b):
        UserManager.authorize(u, Permission.VIEW_PORTFOLIO)
        rows = []
        for rec in P.lifecycle.records():
            p = rec.pending
            rows.append({"strategy_id": rec.strategy_id, "owner": rec.owner, "stage": rec.stage.value, "since": rec.stage_since,
                         "description": rec.description,
                         "pending": None if p is None else {"target": p.target.value, "by": p.submitted_by, "at": p.submitted_at, "note": p.note,
                                                            "grade": p.evidence.get("grade"), "failures": p.evidence.get("failures", [])},
                         "history": rec.history[-6:][::-1]})
        return {"strategies": sorted(rows, key=lambda x: x["strategy_id"]), "can": can(u), "me": u.username,
                "min_paper_days": P.lifecycle.policy.min_paper_days}

    def run_and_score(u: User, b: Dict[str, Any]):
        desk = research(u)
        md, symbols = app._data(b, u)
        app._check_strategy(b["strategy"])
        symbols = app._symbols_for(b["strategy"], symbols)
        result = desk.backtest(b["strategy"], md, symbols, b.get("params") or {})
        return desk, desk.evaluate(result, n_trials=max(1, int(b.get("n_trials", 1))))

    @r("POST", "/api/governance/evaluate")
    def evaluate(u, b):
        _, verdict = run_and_score(u, b)
        return verdict.to_dict()

    @r("POST", "/api/governance/register")
    def register_strategy(u, b):
        sid = str(b.get("strategy_id", "")).strip()
        if not sid or len(sid) > 40 or not sid.replace("_", "").replace("-", "").isalnum():
            raise ValidationError("strategy id: 1-40 letters, digits, '-' or '_'")
        research(u).register_strategy(sid, {"strategy": b.get("strategy"), "params": b.get("params") or {}})
        return {"ok": True}

    @r("POST", "/api/governance/propose")
    def propose(u, b):
        desk, verdict = run_and_score(u, b)
        desk.propose(b["strategy_id"], verdict, str(b.get("note", ""))[:300])
        return {"ok": True, "verdict": verdict.to_dict()}

    @r("POST", "/api/governance/decide")
    def decide(u, b):
        sid, action, reason = b["strategy_id"], b["action"], str(b.get("reason", "")).strip()
        lc = P.lifecycle
        if action == "approve":
            lc.approve(u, sid, reason)
        elif action == "reject":
            lc.reject(u, sid, reason)
        elif action == "demote":
            lc.demote(u, sid, reason)
        elif action == "retire":
            lc.retire(u, sid, reason or "retired")
        else:
            raise ValidationError("action must be approve, reject, demote or retire")
        P.autosave()
        return {"ok": True, "stage": lc.stage(sid).value}

    # ---- alerts and notifications -------------------------------------------------------------------------------------------
    def personal_inbox(u: User) -> MemoryChannel:
        """Each user's own channel for personal alerts. Reused if it was restored from the store after a restart."""
        name = f"inbox:{u.user_id}"
        for route in P.notifier.routes:
            if route.channel.name == name and route.user_id == u.user_id and isinstance(route.channel, MemoryChannel):
                return route.channel
        channel = MemoryChannel(name, maxlen=200)
        P.notifier.subscribe(u.user_id, channel, include_system=False)
        P.autosave()
        return channel

    def can_ack(u: User) -> bool:
        return u.has(Permission.MANAGE_RISK) or u.has(Permission.OPERATE_SYSTEM)

    @r("GET", "/api/inbox")
    def inbox(u, _b):
        UserManager.authorize(u, Permission.VIEW_RISK)
        mine = personal_inbox(u)
        return {"platform": [n.to_dict() for n in P.inbox.items[-60:]][::-1], "personal": [n.to_dict() for n in mine.items[-60:]][::-1],
                "open_critical": [n.to_dict() for n in P.notifier.unacknowledged()], "can_ack": can_ack(u),
                "alerts": P.alerts.describe(u.user_id), "undelivered": len(P.notifier.dead_letters),
                "can_add": {"personal": u.has(Permission.PLACE_ORDER) and u.has(Permission.MANAGE_ALERTS), "firm": u.has(Permission.MANAGE_ALERTS)},
                "metrics": list(ALERT_METRICS)}

    @r("POST", "/api/inbox/ack")
    def ack(u, b):
        if not can_ack(u):
            raise AuthorizationError("acknowledging a critical alert needs the risk-manager or operations role", code="FORBIDDEN")
        if not P.notifier.acknowledge(str(b["id"]), u.username):
            raise ValidationError("that notification is not open (already acknowledged?)")
        P.audit.record(u.username, "notify.acknowledge", id=str(b["id"]))
        P.autosave()
        return {"ok": True}

    @r("POST", "/api/alerts/add")
    def add_alert(u, b):
        severity = Severity(str(b.get("severity", "WARNING")).upper())
        for_count = int(b.get("for_count", 1))
        if b.get("kind") == "price":
            symbol = str(b["symbol"]).strip().upper()
            if not symbol.isalnum() or len(symbol) > 8:
                raise ValidationError("symbol: up to 8 letters or digits")
            direction = b.get("direction", "above")
            if direction not in ("above", "below"):
                raise ValidationError("direction must be 'above' or 'below'")
            metric, op, value = f"price.{symbol}", ">" if direction == "above" else "<", float(b["level"])
            label = f"{symbol} {direction} {value:g}"
        else:
            metric, op, value = str(b["metric"]), str(b.get("op", ">")), float(b["value"])
            if metric not in ALERT_METRICS:
                raise ValidationError(f"metric must be one of {list(ALERT_METRICS)}")
            label = f"{metric} {op} {value:g}"
        if b.get("scope", "personal") == "personal":
            name = f"{u.username}: {label}"
            rule = threshold(name, metric, op, value, severity=severity, for_count=for_count, topic="alert.personal", notify_resolved=False)
            personal_inbox(u)
            P.desk(u, "trading").add_alert(rule)
        else:
            rule = threshold(label, metric, op, value, severity=severity, for_count=for_count, topic="alert.firm")
            P.desk(u, "risk").add_alert(rule)
        P.audit.record(u.username, "alert.add", rule=rule.name, scope=b.get("scope", "personal"))
        return {"ok": True, "name": rule.name}

    @r("POST", "/api/alerts/remove")
    def remove_alert(u, b):
        name = str(b["name"])
        owner = P.alerts.owner(name)
        if owner is None:
            UserManager.authorize(u, Permission.MANAGE_ALERTS)
        elif owner != u.user_id:
            raise AuthorizationError("that is another user's personal alert", code="FORBIDDEN")
        P.alerts.remove(name)
        P.audit.record(u.username, "alert.remove", rule=name)
        P.autosave()
        return {"ok": True}


def session_metrics(engine: Any, prices: Dict[str, float]) -> Dict[str, Any]:
    """What alert rules are evaluated against after each paper-trading step."""
    pf = engine.portfolio
    equity = pf.equity
    out: Dict[str, Any] = {"drawdown": engine.risk.monitor.drawdown, "daily_pnl_pct": engine.risk.monitor.daily_pnl_pct, "equity": equity,
                           "gross_leverage": pf.gross_exposure / equity if equity > 0 else None, "strategy_errors": engine.strategy_errors}
    out.update({f"price.{s}": float(p) for s, p in prices.items()})
    return out


__all__: List[str] = ["register", "session_metrics", "ALERT_METRICS"]
