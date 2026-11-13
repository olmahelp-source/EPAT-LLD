"""Operations and control: strategy approvals, alerts and inbox, health checks and jobs, the audit log.

Every route checks the caller's permission on the server (through the role desks or ``UserManager.authorize``); hiding a
page in the browser is a convenience, not the control."""
from __future__ import annotations

from typing import Any, Dict

from ...core.enums import Permission, Severity
from ...core.exceptions import AuthorizationError, ValidationError
from ...notify.alerts import threshold
from ...notify.channels import MemoryChannel
from ...users.manager import User, UserManager

from ..session import ALERT_METRICS


def register(app: Any) -> None:
    r, P = app.route, app.platform

    def research(u: User):
        return P.desk(u, "research")                              # raises AuthorizationError for roles without research

    def can(u: User) -> Dict[str, bool]:
        return {"register": u.has(Permission.MANAGE_STRATEGY), "propose": u.has(Permission.SUBMIT_STRATEGY),
                "approve": u.has(Permission.APPROVE_STRATEGY), "demote": u.has(Permission.APPROVE_STRATEGY) or u.has(Permission.MANAGE_RISK)}

    def run_and_score(u: User, b: Dict[str, Any]):
        desk = research(u)
        md, symbols = app._data(b, u)
        app._check_strategy(b["strategy"])
        symbols = app._symbols_for(b["strategy"], symbols)
        result = desk.backtest(b["strategy"], md, symbols, b.get("params") or {})
        return desk, desk.evaluate(result, n_trials=max(1, int(b.get("n_trials", 1))))

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

    def store_check():
        if P.store is None:
            return True, "not configured: shared state is kept in memory only and is lost on restart"
        P.store.put("health.ping", {"ok": True})                  # raises if the database cannot be written
        return True, "writable"

    def sessions_check():
        halted = [s for s in list(app.sessions.values()) if s.engine.risk.kill_switch.state == "TRIGGERED"]
        return not halted, f"{len(app.sessions)} open, {len(halted)} halted by the kill switch"

    # health checks and a housekeeping job for the operations page; none of them is critical to serving requests
    P.health.register("state store", store_check, critical=False)
    P.health.register("notification delivery", lambda: (len(P.notifier.dead_letters) == 0, f"{len(P.notifier.dead_letters)} undelivered"),
                      critical=False)
    P.health.register("paper sessions", sessions_check, critical=False)

    def data_check():
        st = P.data.status()
        if st["sample_only"]:
            return True, "sample-data-only mode: no data provider is used"
        parts, ok = [], True
        for prov in st["providers"]:
            last = prov["status"]
            if not prov["connected"]:
                parts.append(f"{prov['label']}: not connected")
                continue
            if last.get("ok") is False:
                ok = False
            parts.append(f"{prov['label']}: " + ("never tested" if not last else "last call ok" if last.get("ok") else "last call failed"))
        return ok, "; ".join(parts) + f"; {len(st['library'])} downloaded series"

    P.health.register("market data", data_check, critical=False)
    if P.store is not None:
        P.scheduler.every(300, P.autosave, name="save platform state")


    @r("GET", "/api/audit")
    def audit(u, _b):
        if not u.has(Permission.VIEW_AUDIT):                  # the compliance desk: auditor, risk manager, admin
            raise AuthorizationError("the audit log needs the auditor, risk-manager or admin role")
        return list(reversed(P.audit.entries()[-150:]))

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

    @r("GET", "/api/ops/health")
    def health(u, _b):
        desk = P.desk(u, "operations")
        return {"health": desk.health(), "jobs": desk.jobs(), "undelivered": desk.undelivered(), "restore": P.restore_report,
                "persistent": P.store is not None, "sessions": len(app.sessions), "users": P.users.user_count(),
                "routes": [{"scope": "personal" if rt.user_id else "platform", "channel": rt.channel.name, "min_severity": rt.min_severity.value,
                            "topics": list(rt.topics or [])} for rt in P.notifier.routes]}

    @r("POST", "/api/ops/replay")
    def replay(u, _b):
        return P.desk(u, "operations").replay_undelivered()

    @r("POST", "/api/ops/run_jobs")
    def run_jobs(u, _b):
        return {"ran": P.desk(u, "operations").run_scheduled()}
