"""Kill switch: halts trading, runs emergency actions, requires authorised reset."""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ..core import events as ev
from ..core.enums import KillSwitchState, Permission
from ..core.events import EventBus
from ..core.exceptions import AuthorizationError, KillSwitchActive, RiskError
from ..core.logging import AuditLog, get_logger

log = get_logger(__name__)


class KillSwitch:
    """Thread-safe latch.

    * ``trip`` is idempotent, never raises because of a failing callback, and can be
      called from any thread (monitor, strategy error handler, human operator).
    * ``reset`` needs a user holding ``RESET_KILL_SWITCH``; with ``two_person_reset``
      a second, different authorised user must approve (four-eyes principle).
    * Emergency callbacks (cancel all orders, flatten positions) run on trip.
    """

    def __init__(self, bus: Optional[EventBus] = None, audit: Optional[AuditLog] = None,
                 two_person_reset: bool = False,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.bus = bus or EventBus()
        self.audit = audit or AuditLog()
        self.two_person_reset = two_person_reset
        self._clock = clock
        self._state = KillSwitchState.ARMED
        self._lock = threading.RLock()
        self._callbacks: List[Callable[[str], None]] = []
        self.reason = ""
        self.source = ""
        self.tripped_at: Optional[datetime] = None
        self.history: List[Dict[str, Any]] = []

    @property
    def state(self) -> KillSwitchState:
        return self._state

    @property
    def is_triggered(self) -> bool:
        return self._state is KillSwitchState.TRIGGERED

    def on_trip(self, callback: Callable[[str], None]) -> None:
        self._callbacks.append(callback)

    def check(self) -> None:
        if self.is_triggered:
            raise KillSwitchActive(f"trading halted: {self.reason}", details={"source": self.source})

    def trip(self, reason: str, source: str = "manual") -> bool:
        """Halt trading. Returns True if this call tripped it, False if it was already tripped."""
        with self._lock:
            if self._state is KillSwitchState.TRIGGERED:
                return False
            self._state = KillSwitchState.TRIGGERED
            self.reason, self.source, self.tripped_at = reason, source, self._clock()
            self.history.append({"event": "trip", "reason": reason, "source": source, "ts": self.tripped_at})
        log.critical("KILL SWITCH TRIPPED by %s: %s", source, reason)
        self.audit.record(source, "killswitch.trip", reason=reason)
        for cb in list(self._callbacks):
            try:
                cb(reason)
            except Exception as exc:  # noqa: BLE001 - one failing action must not block the others
                log.error("kill-switch callback %s failed: %r", getattr(cb, "__name__", cb), exc)
                self.audit.record("killswitch", "killswitch.callback_failed", error=repr(exc))
        self.bus.publish(ev.KILL_SWITCH, {"state": "TRIGGERED", "reason": reason, "source": source})
        return True

    def restore(self, reason: str, source: str = "recovery") -> None:
        """Re-enter the TRIGGERED state without running emergency callbacks (used after a restart, where the
        emergency actions already happened or are decided by a human). Still needs an authorised reset."""
        with self._lock:
            if self._state is KillSwitchState.TRIGGERED:
                return
            self._state = KillSwitchState.TRIGGERED
            self.reason, self.source, self.tripped_at = reason, source, self._clock()
            self.history.append({"event": "restore", "reason": reason, "source": source, "ts": self.tripped_at})
        log.critical("kill switch RESTORED as triggered (%s): %s", source, reason)
        self.audit.record(source, "killswitch.restored", reason=reason)

    def reset(self, actor, reason: str, approver=None) -> None:
        """Re-arm trading. ``actor``/``approver`` are users exposing ``has(Permission)`` and ``username``."""
        if not actor.has(Permission.RESET_KILL_SWITCH):
            raise AuthorizationError(f"{actor.username} may not reset the kill switch")
        if not reason or len(reason.strip()) < 5:
            raise RiskError("a written reason (>= 5 chars) is required to reset the kill switch")
        if self.two_person_reset:
            if approver is None or approver.username == actor.username:
                raise AuthorizationError("reset requires approval from a second, different user")
            if not approver.has(Permission.RESET_KILL_SWITCH):
                raise AuthorizationError(f"{approver.username} may not approve a reset")
        with self._lock:
            if self._state is not KillSwitchState.TRIGGERED:
                return
            self._state = KillSwitchState.ARMED
            self.history.append({"event": "reset", "by": actor.username, "reason": reason, "ts": self._clock()})
        log.warning("kill switch reset by %s: %s", actor.username, reason)
        self.audit.record(actor.username, "killswitch.reset", reason=reason,
                          approver=getattr(approver, "username", None))
        self.bus.publish(ev.KILL_SWITCH, {"state": "ARMED", "reason": reason, "source": actor.username})
