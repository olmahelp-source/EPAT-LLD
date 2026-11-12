"""Health monitoring: is every part of the platform alive, and who gets told when one is not.

Two kinds of signal:

* **checks**      functions the monitor calls ("can I reach the broker", "is the database writable");
* **heartbeats**  things that must keep happening on their own ("a bar arrived", "the strategy loop ran"). The component
                  calls ``beat``; the monitor notices when the beats stop.

A check that raises or exceeds its timeout is a failed check, not a crashed monitor. Overall status is DOWN if any
critical item fails, DEGRADED if only non-critical ones do. Status *changes* are sent to the notifier, so a flapping
dependency produces one message per transition rather than one per poll.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ..core.enums import Severity
from ..core.exceptions import ValidationError
from ..utils.resilience import call_with_timeout


class Status(str, Enum):
    UP = "UP"
    DEGRADED = "DEGRADED"
    DOWN = "DOWN"


@dataclass
class HealthResult:
    name: str
    ok: bool
    detail: str = ""
    latency_ms: float = 0.0
    critical: bool = True


class HealthMonitor:
    def __init__(self, notifier: Any = None, clock: Callable[[], float] = time.time) -> None:
        self.notifier, self._clock = notifier, clock
        self._checks: Dict[str, tuple] = {}
        self._beats: Dict[str, Dict[str, Any]] = {}
        self._last: Dict[str, bool] = {}
        self.last_status: Optional[Status] = None

    def register(self, name: str, check: Callable[[], Any], critical: bool = True, timeout: float = 5.0) -> None:
        """``check()`` may return a bool, ``(bool, detail)`` or nothing (no exception = healthy)."""
        if name in self._checks or name in self._beats:
            raise ValidationError(f"health item '{name}' already registered")
        if timeout <= 0:
            raise ValidationError("timeout must be positive")
        self._checks[name] = (check, critical, timeout)

    def expect_heartbeat(self, name: str, max_age: float, critical: bool = True) -> None:
        """Declare that ``beat(name)`` should be called at least every ``max_age`` seconds."""
        if name in self._checks or name in self._beats:
            raise ValidationError(f"health item '{name}' already registered")
        if max_age <= 0:
            raise ValidationError("max_age must be positive")
        self._beats[name] = {"max_age": max_age, "critical": critical, "last": None}

    def beat(self, name: str) -> None:
        if name not in self._beats:
            raise ValidationError(f"unknown heartbeat '{name}'", details={"known": sorted(self._beats)})
        self._beats[name]["last"] = self._clock()

    def _run_check(self, name: str) -> HealthResult:
        check, critical, timeout = self._checks[name]
        t0 = time.perf_counter()
        try:
            out = call_with_timeout(check, timeout)
            ok, detail = (bool(out[0]), str(out[1])) if isinstance(out, tuple) else (True if out is None else bool(out), "")
        except Exception as exc:  # noqa: BLE001 - a failing check is a result
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        return HealthResult(name, ok, detail, (time.perf_counter() - t0) * 1e3, critical)

    def run(self) -> Dict[str, Any]:
        """Evaluate everything and return ``{"status", "items", "failing"}``; notifies on changes."""
        results: List[HealthResult] = [self._run_check(n) for n in self._checks]
        now = self._clock()
        for name, hb in self._beats.items():
            if hb["last"] is None:
                results.append(HealthResult(name, False, "no heartbeat received yet", critical=hb["critical"]))
            else:
                age = now - hb["last"]
                results.append(HealthResult(name, age <= hb["max_age"], f"last beat {age:.0f}s ago (limit {hb['max_age']:.0f}s)", critical=hb["critical"]))
        failing = [r for r in results if not r.ok]
        status = Status.DOWN if any(r.critical for r in failing) else Status.DEGRADED if failing else Status.UP
        if self.notifier is not None:
            for r in results:
                was = self._last.get(r.name)
                if was is not None and was != r.ok or (was is None and not r.ok):
                    sev = Severity.INFO if r.ok else (Severity.CRITICAL if r.critical else Severity.WARNING)
                    self.notifier.notify(f"{r.name} {'recovered' if r.ok else 'is failing'}", r.detail, sev, "system.health",
                                         key=f"health|{r.name}|{r.ok}|{now}")
        self._last = {r.name: r.ok for r in results}
        self.last_status = status
        return {"status": status.value, "failing": [r.name for r in failing],
                "items": [{"name": r.name, "ok": r.ok, "detail": r.detail, "critical": r.critical, "latency_ms": round(r.latency_ms, 2)} for r in results]}
