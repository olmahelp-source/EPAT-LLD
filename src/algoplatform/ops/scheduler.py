"""A small job scheduler driven by the platform clock.

It has no thread of its own: whoever owns the main loop calls :meth:`Scheduler.run_pending` (every bar in a backtest
or simulation, every second live). That keeps scheduled work deterministic and testable with ``SimClock``, and means
jobs run on the same thread as the engine instead of racing it.

A job that raises is contained and counted; after ``max_failures`` consecutive failures it is disabled and
``on_disabled`` is called (wire it to the notifier). If the clock jumps past several run times (the process was paused,
the market was closed) the job runs once, not once per missed slot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta
from typing import Any, Callable, Dict, List, Optional

from ..core.exceptions import ValidationError
from ..core.logging import get_logger
from ..utils.timeutils import Clock, SystemClock, TradingCalendar

log = get_logger(__name__)


@dataclass
class Job:
    name: str
    fn: Callable[[], Any]
    interval: Optional[timedelta] = None          # run every interval ...
    at: Optional[dtime] = None                    # ... or once a day at this time
    calendar: Optional[TradingCalendar] = None    # daily jobs only: skip non-trading days
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    runs: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_error: str = ""
    enabled: bool = True

    def summary(self) -> Dict[str, Any]:
        return {"name": self.name, "enabled": self.enabled, "runs": self.runs, "failures": self.failures, "last_error": self.last_error,
                "next_run": self.next_run.isoformat() if self.next_run else None, "last_run": self.last_run.isoformat() if self.last_run else None}


class Scheduler:
    def __init__(self, clock: Optional[Clock] = None, max_failures: int = 3, on_disabled: Optional[Callable[[Job], None]] = None) -> None:
        if max_failures < 1:
            raise ValidationError("max_failures must be >= 1")
        self.clock, self.max_failures, self.on_disabled = clock or SystemClock(), max_failures, on_disabled
        self._jobs: Dict[str, Job] = {}

    def _add(self, job: Job) -> Job:
        if job.name in self._jobs:
            raise ValidationError(f"job '{job.name}' already scheduled")
        self._jobs[job.name] = job
        return job

    def every(self, seconds: float, fn: Callable[[], Any], name: Optional[str] = None, run_now: bool = False) -> Job:
        if seconds <= 0:
            raise ValidationError("interval must be positive")
        now = self.clock.now()
        return self._add(Job(name or getattr(fn, "__name__", "job"), fn, interval=timedelta(seconds=seconds),
                             next_run=now if run_now else now + timedelta(seconds=seconds)))

    def daily_at(self, hhmm: str, fn: Callable[[], Any], name: Optional[str] = None, calendar: Optional[TradingCalendar] = None) -> Job:
        """Run once a day at ``"HH:MM"`` (in the clock's own time). With ``calendar``, only on trading days."""
        try:
            hour, minute = (int(x) for x in hhmm.split(":"))
            at = dtime(hour, minute)
        except (ValueError, AttributeError):
            raise ValidationError(f"time must be 'HH:MM', got {hhmm!r}") from None
        job = Job(name or getattr(fn, "__name__", "job"), fn, at=at, calendar=calendar)
        job.next_run = self._next_daily(job, self.clock.now(), include_today=True)
        return self._add(job)

    @staticmethod
    def _next_daily(job: Job, now: datetime, include_today: bool) -> datetime:
        candidate = now.replace(hour=job.at.hour, minute=job.at.minute, second=0, microsecond=0)      # type: ignore[union-attr]
        if candidate <= now or not include_today:
            candidate += timedelta(days=1) if candidate <= now else timedelta(0)
        for _ in range(14):                                         # skip weekends / holidays
            if job.calendar is None or job.calendar.is_trading_day(candidate.date()):
                return candidate
            candidate += timedelta(days=1)
        return candidate

    def cancel(self, name: str) -> bool:
        return self._jobs.pop(name, None) is not None

    def enable(self, name: str) -> None:
        job = self._jobs.get(name)
        if job is None:
            raise ValidationError(f"unknown job '{name}'")
        job.enabled, job.consecutive_failures = True, 0

    def jobs(self) -> List[Dict[str, Any]]:
        return [j.summary() for j in self._jobs.values()]

    def run_pending(self) -> List[str]:
        """Run every enabled job that is due. Returns the names that ran (successfully or not)."""
        now = self.clock.now()
        ran: List[str] = []
        for job in list(self._jobs.values()):
            if not job.enabled or job.next_run is None or now < job.next_run:
                continue
            ran.append(job.name)
            job.last_run, job.runs = now, job.runs + 1
            try:
                job.fn()
                job.consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001 - one job must not stop the others or the loop that drives us
                job.failures += 1
                job.consecutive_failures += 1
                job.last_error = f"{type(exc).__name__}: {exc}"
                log.error("scheduled job %s failed (%d in a row): %r", job.name, job.consecutive_failures, exc)
                if job.consecutive_failures >= self.max_failures:
                    job.enabled = False
                    log.error("scheduled job %s disabled after %d consecutive failures", job.name, job.consecutive_failures)
                    if self.on_disabled:
                        try:
                            self.on_disabled(job)
                        except Exception as cb_exc:  # noqa: BLE001
                            log.error("on_disabled callback failed: %r", cb_exc)
            job.next_run = now + job.interval if job.interval else self._next_daily(job, now, include_today=False)   # from *now*: no catch-up burst
        return ran
