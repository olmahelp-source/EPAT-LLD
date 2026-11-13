"""Reusable resilience helpers: retry, circuit breaker, rate limiter, timing."""
from __future__ import annotations

import functools
import random
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Optional, Tuple, Type

from ..core.exceptions import BrokerConnectionError, PlatformError
from ..core.logging import get_logger

log = get_logger(__name__)


def retry(max_attempts: int = 4, base_delay: float = 0.5, max_delay: float = 8.0,
          backoff: float = 2.0, jitter: float = 0.1,
          retry_on: Tuple[Type[BaseException], ...] = (PlatformError, ConnectionError, TimeoutError),
          sleep: Callable[[float], None] = time.sleep) -> Callable:
    """Retry with exponential backoff.

    Exceptions that are ``PlatformError`` are retried only if ``retryable`` is True,
    so validation or risk errors fail immediately instead of being hammered.
    """
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except retry_on as exc:
                    if isinstance(exc, PlatformError) and not exc.retryable:
                        raise
                    if attempt == max_attempts:
                        log.error("%s failed after %d attempts: %r", fn.__name__, attempt, exc)
                        raise
                    wait = min(delay, max_delay) * (1 + random.uniform(0, jitter))
                    log.warning("%s attempt %d failed (%r); retry in %.2fs", fn.__name__, attempt, exc, wait)
                    sleep(wait)
                    delay *= backoff
        return wrapper
    return deco


class CircuitBreaker:
    """Stops calling a failing dependency for ``reset_timeout`` seconds."""

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 30.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self._clock = clock
        self._failures = 0
        self._opened_at: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return self.CLOSED
            if self._clock() - self._opened_at >= self.reset_timeout:
                return self.HALF_OPEN
            return self.OPEN

    def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        if self.state == self.OPEN:
            raise BrokerConnectionError("circuit open", code="CIRCUIT_OPEN")
        try:
            result = fn(*args, **kwargs)
        except Exception:
            with self._lock:
                self._failures += 1
                if self._failures >= self.failure_threshold or self._opened_at is not None:
                    self._opened_at = self._clock()
            raise
        with self._lock:
            self._failures = 0
            self._opened_at = None
        return result


class RateLimiter:
    """Sliding-window limiter: at most ``max_calls`` per ``period`` seconds."""

    def __init__(self, max_calls: int, period: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.max_calls, self.period, self._clock = max_calls, period, clock
        self._calls: Deque[float] = deque()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        now = self._clock()
        with self._lock:
            while self._calls and now - self._calls[0] >= self.period:
                self._calls.popleft()
            if len(self._calls) >= self.max_calls:
                return False
            self._calls.append(now)
            return True

    def current(self) -> int:
        now = self._clock()
        with self._lock:
            return sum(1 for t in self._calls if now - t < self.period)


def timed(fn: Callable) -> Callable:
    """Log the duration of a call at DEBUG level."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            log.debug("%s took %.3f ms", fn.__name__, (time.perf_counter() - t0) * 1e3)
    return wrapper


def safe_call(fn: Callable, *args: Any, default: Any = None, **kwargs: Any) -> Any:
    """Call ``fn`` and return ``default`` on any exception (logged)."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        log.warning("safe_call %s failed: %r", getattr(fn, "__name__", fn), exc)
        return default
