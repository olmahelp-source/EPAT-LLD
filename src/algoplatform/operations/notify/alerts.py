"""Alert rules: conditions on live metrics that turn into notifications.

A rule moves through OK -> PENDING -> FIRING. It must hold for ``for_count`` consecutive evaluations before it fires
(one noisy tick does not page anyone), it repeats at most every ``cooldown`` seconds while it keeps firing, and it
sends a "resolved" message when the condition clears. A rule that raises is reported once as broken and treated as not
firing: a bug in one alert must not silence the others or stop the loop that evaluates them.

    alerts = AlertManager(notifier)
    alerts.add(threshold("drawdown", "drawdown", ">", 0.10, severity=Severity.ERROR, for_count=2))
    alerts.add(price_alert("AAPL", 200.0, "above"))
    alerts.evaluate({"drawdown": 0.12, "price.AAPL": 198.4})     # call every bar / every few seconds
"""
from __future__ import annotations

import operator
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Mapping, Optional

from ...foundation.core.enums import Severity
from ...foundation.core.exceptions import ValidationError
from ...foundation.core.logging import get_logger
from .notifier import Notifier

log = get_logger(__name__)
Metrics = Mapping[str, Any]
_OPS: Dict[str, Callable[[Any, Any], bool]] = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le,
                                               "==": operator.eq, "!=": operator.ne}


class AlertState(str, Enum):
    OK = "OK"
    PENDING = "PENDING"
    FIRING = "FIRING"


@dataclass
class AlertRule:
    name: str
    condition: Callable[[Metrics], bool]
    severity: Severity = Severity.WARNING
    message: str = ""                         # str.format template over the metrics, e.g. "drawdown is {drawdown:.1%}"
    topic: str = "alert"
    for_count: int = 1
    cooldown: float = 300.0
    notify_resolved: bool = True
    user_id: Optional[str] = None             # private alert for one user (e.g. a personal price alert)
    definition: Optional[Dict[str, Any]] = None   # set for rules built from data (``threshold`` and friends): what makes them storable
    # runtime state
    state: AlertState = field(default=AlertState.OK, compare=False)
    streak: int = field(default=0, compare=False)
    last_sent: float = field(default=-1e18, compare=False)
    fired: int = field(default=0, compare=False)
    errors: int = field(default=0, compare=False)
    silenced_until: float = field(default=-1e18, compare=False)

    def __post_init__(self) -> None:
        if not self.name or self.for_count < 1 or self.cooldown < 0:
            raise ValidationError("an alert rule needs a name, for_count >= 1 and cooldown >= 0")

    def render(self, metrics: Metrics) -> str:
        try:
            return self.message.format(**metrics) if self.message else ""
        except (KeyError, IndexError, ValueError):
            return self.message                # a template naming a missing metric is shown as written


@dataclass(frozen=True)
class AlertEvent:
    rule: str
    transition: str                           # fired | repeated | resolved | broken
    severity: Severity
    message: str


def threshold(name: str, metric: str, op: str, value: float, **kwargs: Any) -> AlertRule:
    """Fire when ``metrics[metric] <op> value``. A missing or non-numeric metric means "unknown", which does not fire."""
    if op not in _OPS:
        raise ValidationError(f"unknown comparison '{op}'", details={"valid": sorted(_OPS)})
    compare = _OPS[op]

    def condition(m: Metrics) -> bool:
        v = m.get(metric)
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and compare(v, value)
    kwargs.setdefault("message", f"{metric} is {{{metric}}} (limit {op} {value:g})")
    return AlertRule(name, condition, definition={"kind": "threshold", "metric": metric, "op": op, "value": value}, **kwargs)


def price_alert(symbol: str, level: float, direction: str = "above", **kwargs: Any) -> AlertRule:
    """Price crossing a level; expects the metric ``price.<SYMBOL>``."""
    if direction not in ("above", "below"):
        raise ValidationError("direction must be 'above' or 'below'")
    kwargs.setdefault("severity", Severity.INFO)
    kwargs.setdefault("topic", "alert.price")
    kwargs.setdefault("notify_resolved", False)
    kwargs.setdefault("message", f"{symbol} is {direction} {level:g}")
    return threshold(f"price_{symbol}_{direction}_{level:g}", f"price.{symbol}", ">" if direction == "above" else "<", level, **kwargs)


def drawdown_alert(limit: float, **kwargs: Any) -> AlertRule:
    kwargs.setdefault("severity", Severity.ERROR)
    kwargs.setdefault("topic", "alert.risk")
    kwargs.setdefault("message", "drawdown {drawdown:.1%} exceeds " + f"{limit:.0%}")
    return threshold("drawdown", "drawdown", ">", limit, **kwargs)


def daily_loss_alert(limit: float, **kwargs: Any) -> AlertRule:
    """``daily_pnl_pct`` below ``-limit`` (limit given as a positive fraction of equity)."""
    kwargs.setdefault("severity", Severity.ERROR)
    kwargs.setdefault("topic", "alert.risk")
    kwargs.setdefault("message", "daily P&L {daily_pnl_pct:.2%} breaches " + f"-{limit:.1%}")
    return threshold("daily_loss", "daily_pnl_pct", "<", -abs(limit), **kwargs)


def stale_data_alert(max_age_seconds: float, **kwargs: Any) -> AlertRule:
    kwargs.setdefault("severity", Severity.ERROR)
    kwargs.setdefault("topic", "alert.data")
    kwargs.setdefault("message", "no market data for {data_age_seconds:.0f}s")
    return threshold("stale_data", "data_age_seconds", ">", max_age_seconds, **kwargs)


def reject_rate_alert(max_rate: float, **kwargs: Any) -> AlertRule:
    kwargs.setdefault("topic", "alert.orders")
    kwargs.setdefault("message", "order reject rate {reject_rate:.0%}")
    return threshold("reject_rate", "reject_rate", ">", max_rate, **kwargs)


class AlertManager:
    def __init__(self, notifier: Notifier, clock: Callable[[], float] = time.time) -> None:
        self.notifier, self._clock = notifier, clock
        self._rules: Dict[str, AlertRule] = {}

    def add(self, rule: AlertRule, replace: bool = False) -> AlertRule:
        if rule.name in self._rules and not replace:
            raise ValidationError(f"alert rule '{rule.name}' already exists")
        self._rules[rule.name] = rule
        return rule

    def remove(self, name: str) -> bool:
        return self._rules.pop(name, None) is not None

    def silence(self, name: str, seconds: float) -> None:
        """Keep evaluating but send nothing for a while (planned maintenance, a known incident being handled)."""
        self._rule(name).silenced_until = self._clock() + seconds

    def _rule(self, name: str) -> AlertRule:
        if name not in self._rules:
            raise ValidationError(f"unknown alert rule '{name}'", details={"rules": sorted(self._rules)})
        return self._rules[name]

    # ---- persistence ---------------------------------------------------------------------------------------------------------
    def export_state(self) -> Dict[str, Any]:
        """Rules as plain data. Only rules built from data (``threshold``, ``price_alert``, ``drawdown_alert`` ...) can be
        stored; a rule whose condition is arbitrary code is listed under ``not_storable`` and must be added again in
        code after a restart."""
        rules, skipped = [], []
        for r in self._rules.values():
            if r.definition is None:
                skipped.append(r.name)
                continue
            rules.append({"name": r.name, "definition": dict(r.definition), "severity": r.severity.value, "message": r.message, "topic": r.topic,
                          "for_count": r.for_count, "cooldown": r.cooldown, "notify_resolved": r.notify_resolved, "user_id": r.user_id,
                          "silenced_until": r.silenced_until if r.silenced_until > 0 else None})
        return {"rules": rules, "not_storable": skipped}

    def import_state(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        """Restore rules written by :meth:`export_state` (replacing rules of the same name). Each rule starts in the OK
        state: after a restart a condition must be seen again before it fires. A stored rule that no longer builds is
        skipped and reported."""
        restored, failed = 0, []
        for d in state.get("rules", []):
            try:
                definition = d["definition"]
                if definition.get("kind") != "threshold":
                    raise ValidationError(f"unknown rule kind '{definition.get('kind')}'")
                rule = threshold(d["name"], definition["metric"], definition["op"], definition["value"], severity=Severity(d["severity"]),
                                 message=d.get("message", ""), topic=d.get("topic", "alert"), for_count=int(d.get("for_count", 1)),
                                 cooldown=float(d.get("cooldown", 300.0)), notify_resolved=bool(d.get("notify_resolved", True)),
                                 user_id=d.get("user_id"))
                if d.get("silenced_until"):
                    rule.silenced_until = float(d["silenced_until"])
                self.add(rule, replace=True)
                restored += 1
            except (KeyError, TypeError, ValueError, ValidationError) as exc:
                failed.append({"rule": str(d.get("name", "?")), "error": str(exc)})
        return {"restored": restored, "failed": failed, "not_storable": list(state.get("not_storable", []))}

    def states(self) -> Dict[str, Dict[str, Any]]:
        return {n: {"state": r.state.value, "fired": r.fired, "errors": r.errors, "severity": r.severity.value} for n, r in self._rules.items()}

    def describe(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Rules as plain rows for a UI: firm-wide rules plus, when ``user_id`` is given, that user's personal ones
        (other users' personal rules are never included)."""
        return [{"name": r.name, "state": r.state.value, "severity": r.severity.value, "topic": r.topic, "personal": r.user_id is not None,
                 "fired": r.fired, "errors": r.errors, "condition": (f"{r.definition['metric']} {r.definition['op']} {r.definition['value']:g}"
                                                                     if r.definition else "defined in code"),
                 "for_count": r.for_count} for r in self._rules.values() if r.user_id is None or r.user_id == user_id]

    def owner(self, name: str) -> Optional[str]:
        return self._rule(name).user_id

    def firing(self) -> List[str]:
        return [n for n, r in self._rules.items() if r.state is AlertState.FIRING]

    def evaluate(self, metrics: Metrics) -> List[AlertEvent]:
        """Evaluate every rule against the current metrics; returns what changed."""
        now = self._clock()
        out: List[AlertEvent] = []
        for rule in list(self._rules.values()):
            try:
                hit = bool(rule.condition(metrics))
            except Exception as exc:  # noqa: BLE001 - a broken rule is reported, never propagated
                rule.errors += 1
                hit = False
                if rule.errors == 1:
                    log.error("alert rule %s is broken: %r", rule.name, exc)
                    self.notifier.notify(f"Alert rule '{rule.name}' is broken", repr(exc), Severity.WARNING, "alert.broken", key=f"broken|{rule.name}")
                    out.append(AlertEvent(rule.name, "broken", Severity.WARNING, repr(exc)))
            if hit:
                rule.streak += 1
                if rule.streak < rule.for_count:
                    rule.state = AlertState.PENDING
                    continue
                first = rule.state is not AlertState.FIRING
                rule.state = AlertState.FIRING
                if (first or now - rule.last_sent >= rule.cooldown) and now >= rule.silenced_until:
                    text = rule.render(metrics)
                    rule.last_sent, rule.fired = now, rule.fired + 1
                    self.notifier.notify(f"Alert: {rule.name}", text, rule.severity, rule.topic, key=f"alert|{rule.name}|{rule.fired}",
                                         user_id=rule.user_id, rule=rule.name)
                    out.append(AlertEvent(rule.name, "fired" if first else "repeated", rule.severity, text))
            else:
                was_firing = rule.state is AlertState.FIRING
                rule.state, rule.streak = AlertState.OK, 0
                if was_firing:
                    if rule.notify_resolved and now >= rule.silenced_until:
                        self.notifier.notify(f"Resolved: {rule.name}", "", Severity.INFO, rule.topic, key=f"resolved|{rule.name}|{rule.fired}",
                                             user_id=rule.user_id, rule=rule.name)
                    out.append(AlertEvent(rule.name, "resolved", Severity.INFO, ""))
        return out
