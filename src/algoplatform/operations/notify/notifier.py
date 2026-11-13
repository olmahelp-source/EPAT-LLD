"""Notifier: routes a notification to the right channels and makes sure a delivery problem never becomes a trading problem.

Guarantees:

* ``notify`` never raises because a channel failed. Each channel is tried independently; failures are counted, logged
  and kept in a dead-letter queue for :meth:`Notifier.replay_dead_letters`.
* Transient failures (``DeliveryError``) are retried with backoff; permanent ones are not.
* The same notification (same key) is sent once per ``dedupe_window``, so a breached limit evaluated every bar produces
  one message, not hundreds. The repeat count is kept.
* A notification addressed to a user reaches that user's channels only. Platform-wide notifications reach the
  platform channels and any user channel that opted in to system messages.
* CRITICAL notifications stay "open" until someone acknowledges them; :meth:`escalate_due` re-sends the ones nobody has.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple

from ...foundation.core import events
from ...foundation.core.enums import Severity
from ...foundation.core.events import EventBus
from ...foundation.core.exceptions import DeliveryError, ValidationError
from ...foundation.core.logging import get_logger
from ...foundation.utils.decorators import RateLimiter
from ...foundation.utils.resilience import DeadLetterQueue
from .channels import Channel
from .message import Notification

log = get_logger(__name__)


@dataclass
class Route:
    channel: Channel
    min_severity: Severity = Severity.INFO
    topics: Optional[Tuple[str, ...]] = None          # None = every topic; "risk" matches "risk" and "risk.*"
    user_id: Optional[str] = None                     # None = a platform channel
    include_system: bool = True                       # for user channels: also receive platform-wide notifications
    limiter: Optional[RateLimiter] = None

    def matches(self, n: Notification) -> bool:
        if not n.severity.at_least(self.min_severity):
            return False
        if self.topics is not None and not any(n.topic == t or n.topic.startswith(t + ".") for t in self.topics):
            return False
        if n.user_id is not None:
            return self.user_id == n.user_id          # private: only that user's own channels
        return self.user_id is None or self.include_system


@dataclass
class DeliveryReport:
    notification: Notification
    delivered: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)
    suppressed: bool = False                          # dropped as a duplicate inside the dedupe window
    throttled: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed


class Notifier:
    def __init__(self, dedupe_window: float = 60.0, max_attempts: int = 3, backoff: float = 0.5, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, dead_letters: Optional[DeadLetterQueue] = None, history: int = 500) -> None:
        if dedupe_window < 0 or max_attempts < 1 or backoff < 0:
            raise ValidationError("need dedupe_window >= 0, max_attempts >= 1, backoff >= 0")
        self.dedupe_window, self.max_attempts, self.backoff = dedupe_window, max_attempts, backoff
        self._clock, self._sleep = clock, sleep
        self.dead_letters = dead_letters or DeadLetterQueue()
        self.routes: List[Route] = []
        self.history: Deque[Notification] = deque(maxlen=history)
        self._seen: Dict[str, Tuple[float, int]] = {}                  # dedupe key -> (first sent, repeats suppressed)
        self._open: Dict[str, Tuple[Notification, float]] = {}         # unacknowledged CRITICAL: id -> (notification, last sent)
        self._lock = threading.RLock()
        self.stats = {"sent": 0, "suppressed": 0, "failed": 0, "retries": 0, "throttled": 0}

    # ---- configuration ------------------------------------------------------------------------------------------------------
    def add_channel(self, channel: Channel, min_severity: Severity = Severity.INFO, topics: Optional[Sequence[str]] = None,
                    user_id: Optional[str] = None, include_system: bool = True, rate_limit: Optional[Tuple[int, float]] = None) -> Route:
        """Register a channel. ``rate_limit=(max_messages, per_seconds)`` protects a chat channel from an alert storm;
        CRITICAL notifications are never rate-limited."""
        limiter = RateLimiter(rate_limit[0], rate_limit[1], self._clock) if rate_limit else None
        route = Route(channel, Severity(min_severity), tuple(topics) if topics else None, user_id, include_system, limiter)
        with self._lock:
            self.routes.append(route)
        return route

    def subscribe(self, user_id: str, channel: Channel, min_severity: Severity = Severity.INFO, topics: Optional[Sequence[str]] = None,
                  include_system: bool = True) -> Route:
        """A user's personal channel."""
        if not user_id:
            raise ValidationError("user_id required")
        return self.add_channel(channel, min_severity, topics, user_id, include_system)

    def remove_channel(self, name: str, user_id: Optional[str] = None) -> int:
        with self._lock:
            before = len(self.routes)
            self.routes = [r for r in self.routes if not (r.channel.name == name and r.user_id == user_id)]
            return before - len(self.routes)

    # ---- sending ----------------------------------------------------------------------------------------------------------------
    def notify(self, title: str, body: str = "", severity: Severity = Severity.INFO, topic: str = "general", key: Optional[str] = None,
               user_id: Optional[str] = None, **data: Any) -> DeliveryReport:
        return self.send(Notification(title, body, Severity(severity), topic, key, user_id, data, self._clock()))

    def send(self, n: Notification) -> DeliveryReport:
        report = DeliveryReport(n)
        now = self._clock()
        with self._lock:
            seen = self._seen.get(n.dedupe_key)
            if seen and now - seen[0] < self.dedupe_window:
                self._seen[n.dedupe_key] = (seen[0], seen[1] + 1)
                self.stats["suppressed"] += 1
                report.suppressed = True
                return report
            self._seen[n.dedupe_key] = (now, 0)
            if len(self._seen) > 5000:                                  # forget keys whose window has long passed
                self._seen = {k: v for k, v in self._seen.items() if now - v[0] < self.dedupe_window}
            self.history.append(n)
            routes = [r for r in self.routes if r.matches(n)]
            if n.severity is Severity.CRITICAL:
                self._open[n.id] = (n, now)
        for route in routes:
            name = route.channel.name
            if route.limiter and n.severity is not Severity.CRITICAL and not route.limiter.allow():
                report.throttled.append(name)
                self.stats["throttled"] += 1
                continue
            error = self._deliver(route.channel, n)
            if error is None:
                report.delivered.append(name)
                self.stats["sent"] += 1
            else:
                report.failed[name] = error
                self.stats["failed"] += 1
        return report

    def _deliver(self, channel: Channel, n: Notification) -> Optional[str]:
        """Try one channel. Returns None on success or the error text; never raises."""
        delay = self.backoff
        for attempt in range(1, self.max_attempts + 1):
            try:
                channel.send(n)
                return None
            except DeliveryError as exc:
                if attempt == self.max_attempts:
                    self.dead_letters.add((channel.name, n), exc, "notify", attempt)
                    log.error("notification %s not delivered via %s after %d attempts: %s", n.id, channel.name, attempt, exc)
                    return str(exc)
                self.stats["retries"] += 1
                self._sleep(delay)
                delay *= 2
            except Exception as exc:  # noqa: BLE001 - permanent (bad config) or a bug in a channel: do not retry, do not propagate
                self.dead_letters.add((channel.name, n), exc, "notify", attempt)
                log.error("notification %s failed permanently via %s: %s", n.id, channel.name, exc)
                return str(exc)
        return "unreachable"

    def replay_dead_letters(self) -> Tuple[int, int]:
        """Retry everything in the dead-letter queue against the channels currently registered under the same names."""
        by_name = {r.channel.name: r.channel for r in self.routes}

        def resend(item: Tuple[str, Notification]) -> None:
            name, n = item
            if name not in by_name:
                raise ValidationError(f"channel '{name}' is no longer registered")
            by_name[name].send(n)
        return self.dead_letters.replay(resend, "notify")

    # ---- acknowledgement and escalation ---------------------------------------------------------------------------------------------
    def acknowledge(self, notification_id: str, by: str) -> bool:
        """Mark a CRITICAL notification as seen by a person. Returns False if it was not open."""
        with self._lock:
            found = self._open.pop(notification_id, None)
        if found:
            log.info("notification %s acknowledged by %s", notification_id, by)
        return found is not None

    def unacknowledged(self) -> List[Notification]:
        with self._lock:
            return [n for n, _ in self._open.values()]

    def escalate_due(self, after: float = 300.0, channel: Optional[Channel] = None) -> List[Notification]:
        """Re-send every CRITICAL notification that has gone ``after`` seconds without acknowledgement: to ``channel``
        if given (e.g. a pager), otherwise through its normal routes. Call it periodically (see ``ops.Scheduler``)."""
        now = self._clock()
        with self._lock:
            due = [(nid, n) for nid, (n, last) in self._open.items() if now - last >= after]
            for nid, n in due:
                self._open[nid] = (n, now)
        for _, n in due:
            again = Notification(f"UNACKNOWLEDGED: {n.title}", n.body, n.severity, n.topic, f"escalation|{n.id}|{now}", n.user_id, n.data, now, n.id)
            if channel is not None:
                err = self._deliver(channel, again)
                self.stats["sent" if err is None else "failed"] += 1
            else:
                for route in [r for r in self.routes if r.matches(again)]:
                    err = self._deliver(route.channel, again)
                    self.stats["sent" if err is None else "failed"] += 1
        return [n for _, n in due]

    # ---- persistence -----------------------------------------------------------------------------------------------------------------
    def export_state(self) -> Dict[str, Any]:
        """Routes, unacknowledged critical notifications and undelivered notifications as plain data.

        A route is stored with its channel's own description where the channel can give one without exposing a secret
        (:meth:`Channel.config`); otherwise only the channel's name is stored and it must be supplied again on load."""
        with self._lock:
            routes = [{"channel": r.channel.name, "config": r.channel.config(), "min_severity": r.min_severity.value,
                       "topics": list(r.topics) if r.topics is not None else None, "user_id": r.user_id, "include_system": r.include_system,
                       "rate_limit": [r.limiter.max_calls, r.limiter.period] if r.limiter else None} for r in self.routes]
            open_ = [n.to_dict() for n, _ in self._open.values()]
        dead = [{"channel": d.item[0], "notification": d.item[1].to_dict(), "error": d.error, "attempts": d.attempts}
                for d in self.dead_letters.items("notify")]
        return {"routes": routes, "open": open_, "dead_letters": dead}

    def import_state(self, state: Mapping[str, Any], channels: Optional[Mapping[str, Channel]] = None) -> Dict[str, Any]:
        """Restore what :meth:`export_state` wrote. ``channels`` maps a name to a channel object for the routes that
        could not be stored whole (anything holding a secret). Routes whose channel is neither rebuildable nor supplied
        are returned under ``unresolved`` so the operator can register them again; they are not silently dropped.
        Routes already present (same channel name and user) are left as they are."""
        from .channels import channel_from_config
        supplied = dict(channels or {})
        restored, unresolved = 0, []
        with self._lock:
            existing = {(r.channel.name, r.user_id) for r in self.routes}
        for d in state.get("routes", []):
            name, user_id = d["channel"], d.get("user_id")
            if (name, user_id) in existing:
                continue
            channel = supplied.get(name)
            if channel is None and d.get("config"):
                try:
                    channel = supplied[name] = channel_from_config(d["config"])
                except Exception as exc:  # noqa: BLE001 - one bad stored route must not stop the others
                    unresolved.append({"channel": name, "user_id": user_id, "reason": str(exc)})
                    continue
            if channel is None:
                unresolved.append({"channel": name, "user_id": user_id, "reason": "holds a secret and was not stored; register it again"})
                continue
            limit = d.get("rate_limit")
            self.add_channel(channel, Severity(d.get("min_severity", "INFO")), d.get("topics"), user_id, bool(d.get("include_system", True)),
                             (int(limit[0]), float(limit[1])) if limit else None)
            existing.add((name, user_id))
            restored += 1
        now = self._clock()
        with self._lock:
            for nd in state.get("open", []):
                n = Notification.from_dict(nd)
                self._open.setdefault(n.id, (n, now))               # the escalation timer restarts from the moment of the restart
        for dd in state.get("dead_letters", []):
            self.dead_letters.add((dd["channel"], Notification.from_dict(dd["notification"])), RuntimeError(dd.get("error", "")), "notify",
                                  int(dd.get("attempts", 1)))
        return {"routes_restored": restored, "unresolved": unresolved, "open_critical": len(state.get("open", [])),
                "dead_letters": len(state.get("dead_letters", []))}

    # ---- event bus ------------------------------------------------------------------------------------------------------------------
    def bridge(self, bus: EventBus) -> List[Callable[[], None]]:
        """Turn the platform's own events into notifications: kill switch -> CRITICAL, risk breach -> WARNING, system
        error -> ERROR. Returns the unsubscribe functions."""
        def handler(severity: Severity, label: str) -> Callable[[str, Any], None]:
            def on_event(topic: str, payload: Any) -> None:
                detail = payload if isinstance(payload, dict) else {"detail": str(payload)}
                reason = str(detail.get("reason") or detail.get("message") or detail.get("detail") or "")
                self.notify(label, reason, severity, topic, key=f"{topic}|{reason}", **{k: str(v) for k, v in detail.items()})
            return on_event
        return [bus.subscribe(events.KILL_SWITCH, handler(Severity.CRITICAL, "Kill switch")),
                bus.subscribe(events.RISK_BREACH, handler(Severity.WARNING, "Risk limit breached")),
                bus.subscribe(events.ERROR, handler(Severity.ERROR, "System error"))]
