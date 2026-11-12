"""Advanced error handling: deciding what to do with a failure, not just raising it.

``core.exceptions`` says *what* went wrong and ``utils.decorators`` retries or breaks a circuit. This module adds the
patterns a trading system needs on top of those:

* :func:`classify` / :class:`ErrorPolicy`  one place that maps an exception to a category, severity and action
* :class:`Result` / :func:`attempt`         carry success-or-failure as a value through a pipeline
* :func:`first_success`                     fallback chain (primary feed -> backup feed -> cached value)
* :func:`call_with_timeout`, :class:`Deadline`   bound how long a step or a whole request may take
* :class:`Bulkhead`                         cap concurrency so one slow dependency cannot starve the rest
* :class:`ErrorBudget`, :class:`ErrorBoundary`   contain errors, count them, escalate when there are too many
* :class:`DeadLetterQueue`                  keep what could not be processed so it can be replayed, never dropped
* :class:`ErrorCollector`                   finish a batch, then report every failure at once
* :func:`error_context`                     attach "what were we doing" to any error that passes through

Rule used throughout: an error whose action is HALT (kill switch, bad configuration) is never swallowed.
"""
from __future__ import annotations

import contextlib
import functools
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, Generic, Iterator, List, Optional, Sequence, Tuple, Type, TypeVar

from ..core.enums import Severity
from ..core.exceptions import (AggregateError, AuthError, BrokerConnectionError, BrokerError, CapacityError, ConfigError,
                               DataError, DeliveryError, InvalidTransitionError, KillSwitchActive, MissingDataError,
                               OperationTimeout, OrderRejectedError, PlatformError, RateLimitError, RiskLimitBreach,
                               ValidationError)
from ..core.logging import get_logger

log = get_logger(__name__)
T = TypeVar("T")
U = TypeVar("U")


# ---- classification --------------------------------------------------------------------------------------------------
class ErrorAction(str, Enum):
    RETRY = "RETRY"          # transient: try again with backoff
    FALLBACK = "FALLBACK"    # use an alternative source / default
    SKIP = "SKIP"            # drop this item, carry on with the next
    ESCALATE = "ESCALATE"    # carry on, but a human must look
    HALT = "HALT"            # stop: continuing could lose money or corrupt state


@dataclass(frozen=True)
class ErrorInfo:
    error_type: str
    code: str
    message: str
    category: str
    severity: Severity
    action: ErrorAction
    retryable: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.error_type, "code": self.code, "message": self.message, "category": self.category,
                "severity": self.severity.value, "action": self.action.value, "retryable": self.retryable, "details": self.details}


# Most specific first; the first isinstance match wins.
_DEFAULT_RULES: List[Tuple[Type[BaseException], str, Severity, ErrorAction]] = [
    (KillSwitchActive, "risk", Severity.CRITICAL, ErrorAction.HALT),
    (RiskLimitBreach, "risk", Severity.WARNING, ErrorAction.SKIP),
    (RateLimitError, "broker", Severity.WARNING, ErrorAction.RETRY),
    (BrokerConnectionError, "broker", Severity.ERROR, ErrorAction.RETRY),
    (OrderRejectedError, "broker", Severity.WARNING, ErrorAction.SKIP),
    (BrokerError, "broker", Severity.ERROR, ErrorAction.ESCALATE),
    (InvalidTransitionError, "orders", Severity.ERROR, ErrorAction.ESCALATE),
    (AuthError, "security", Severity.ERROR, ErrorAction.ESCALATE),
    (ConfigError, "config", Severity.CRITICAL, ErrorAction.HALT),
    (ValidationError, "validation", Severity.WARNING, ErrorAction.SKIP),
    (MissingDataError, "data", Severity.WARNING, ErrorAction.FALLBACK),
    (DataError, "data", Severity.ERROR, ErrorAction.SKIP),
    (DeliveryError, "notification", Severity.WARNING, ErrorAction.RETRY),
    (OperationTimeout, "timeout", Severity.WARNING, ErrorAction.RETRY),
    (CapacityError, "capacity", Severity.WARNING, ErrorAction.RETRY),
    (PlatformError, "platform", Severity.ERROR, ErrorAction.ESCALATE),
    (TimeoutError, "timeout", Severity.WARNING, ErrorAction.RETRY),
    (ConnectionError, "network", Severity.WARNING, ErrorAction.RETRY),
    (MemoryError, "system", Severity.CRITICAL, ErrorAction.HALT),
    ((ArithmeticError, ValueError, TypeError, KeyError, IndexError, AttributeError), "bug", Severity.ERROR, ErrorAction.ESCALATE),  # type: ignore[list-item]
]


class ErrorPolicy:
    """Maps exceptions to a decision. Custom rules are consulted before the defaults; within either set the most specific
    class wins, so ``on(BrokerError, ...)`` does not hide the rule for ``RateLimitError``."""

    def __init__(self) -> None:
        self._custom: List[Tuple[Type[BaseException], str, Severity, ErrorAction]] = []

    def on(self, exc_type: Type[BaseException], action: ErrorAction, severity: Severity = Severity.ERROR,
           category: str = "custom") -> "ErrorPolicy":
        self._custom.append((exc_type, category, severity, action))
        self._custom.sort(key=lambda r: -len(r[0].__mro__))            # deeper in the hierarchy = more specific
        return self

    def decide(self, exc: BaseException) -> ErrorInfo:
        for rules in (self._custom, _DEFAULT_RULES):
            for exc_type, category, severity, action in rules:
                if isinstance(exc, exc_type):
                    return self._info(exc, category, severity, action)
        return self._info(exc, "unexpected", Severity.ERROR, ErrorAction.ESCALATE)

    @staticmethod
    def _info(exc: BaseException, category: str, severity: Severity, action: ErrorAction) -> ErrorInfo:
        if isinstance(exc, PlatformError):
            retryable = exc.retryable
            if action is ErrorAction.RETRY and not retryable:          # the exception itself says trying again is pointless
                action = ErrorAction.ESCALATE
            return ErrorInfo(type(exc).__name__, exc.code, exc.message, category, severity, action, retryable, dict(exc.details))
        return ErrorInfo(type(exc).__name__, type(exc).__name__, str(exc), category, severity, action, action is ErrorAction.RETRY)


_DEFAULT_POLICY = ErrorPolicy()


def classify(exc: BaseException) -> ErrorInfo:
    """Default decision for an exception (see :class:`ErrorPolicy` to customise)."""
    return _DEFAULT_POLICY.decide(exc)


# ---- Result ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Result(Generic[T]):
    """Success or failure as a value. Use where an exception would abort a loop that should keep going
    (one bad symbol in a 500-symbol scan) and the caller decides later what to do with the failures."""
    value: Optional[T] = None
    error: Optional[BaseException] = None

    @classmethod
    def success(cls, value: T) -> "Result[T]":
        return cls(value=value)

    @classmethod
    def failure(cls, error: BaseException) -> "Result[T]":
        return cls(error=error)

    @property
    def ok(self) -> bool:
        return self.error is None

    def unwrap(self) -> T:
        if self.error is not None:
            raise self.error
        return self.value  # type: ignore[return-value]

    def unwrap_or(self, default: T) -> T:
        return self.value if self.error is None else default  # type: ignore[return-value]

    def map(self, fn: Callable[[T], U]) -> "Result[U]":
        return self if self.error is not None else attempt(fn, self.value)  # type: ignore[return-value]

    def and_then(self, fn: Callable[[T], "Result[U]"]) -> "Result[U]":
        if self.error is not None:
            return self  # type: ignore[return-value]
        try:
            return fn(self.value)  # type: ignore[arg-type]
        except Exception as exc:  # noqa: BLE001
            return Result.failure(exc)

    def recover(self, fn: Callable[[BaseException], T]) -> "Result[T]":
        return self if self.error is None else attempt(fn, self.error)

    def info(self) -> Optional[ErrorInfo]:
        return None if self.error is None else classify(self.error)


def attempt(fn: Callable[..., T], *args: Any, catch: Tuple[Type[BaseException], ...] = (Exception,), **kwargs: Any) -> Result[T]:
    try:
        return Result.success(fn(*args, **kwargs))
    except catch as exc:
        return Result.failure(exc)


def partition(results: Sequence[Result[T]]) -> Tuple[List[T], List[BaseException]]:
    """Split a batch of results into (values, errors)."""
    return [r.value for r in results if r.ok], [r.error for r in results if not r.ok]  # type: ignore[misc]


# ---- fallback ----------------------------------------------------------------------------------------------------------
def first_success(*candidates: Callable[[], T], catch: Tuple[Type[BaseException], ...] = (Exception,),
                  on_error: Optional[Callable[[int, BaseException], None]] = None) -> T:
    """Call each candidate in order and return the first that works. If all fail, raise one :class:`AggregateError`
    holding every failure. A HALT-class error stops the chain at once: a fallback must not paper over a kill switch."""
    if not candidates:
        raise ValidationError("first_success needs at least one candidate")
    errors: List[BaseException] = []
    for i, fn in enumerate(candidates):
        try:
            return fn()
        except catch as exc:
            if classify(exc).action is ErrorAction.HALT:
                raise
            errors.append(exc)
            if on_error:
                on_error(i, exc)
            log.warning("fallback candidate %d failed: %r", i, exc)
    raise AggregateError("all fallback candidates failed", errors)


def with_fallback(fallback: Callable[..., T], catch: Tuple[Type[BaseException], ...] = (Exception,)) -> Callable:
    """Decorator: on failure call ``fallback`` with the same arguments."""
    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return first_success(lambda: fn(*args, **kwargs), lambda: fallback(*args, **kwargs), catch=catch)
        return wrapper
    return deco


# ---- time limits ---------------------------------------------------------------------------------------------------------
def call_with_timeout(fn: Callable[..., T], seconds: float, *args: Any, **kwargs: Any) -> T:
    """Run ``fn`` in a worker thread and give up waiting after ``seconds``.

    Python cannot kill a thread, so a timed-out call keeps running in the background until it returns; the caller is
    simply no longer blocked by it. Use it for I/O (a broker call that hangs), not to cancel CPU work."""
    if seconds <= 0:
        raise ValidationError("timeout must be positive", details={"seconds": seconds})
    box: Dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - transported to the caller below
            box["error"] = exc
    t = threading.Thread(target=run, daemon=True, name=f"timeout:{getattr(fn, '__name__', 'call')}")
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise OperationTimeout(f"{getattr(fn, '__name__', 'call')} exceeded {seconds:g}s", details={"seconds": seconds})
    if "error" in box:
        raise box["error"]
    return box["value"]


class Deadline:
    """A time budget shared by several steps: each step asks how much is left instead of having its own timeout."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock, self.seconds = clock, float(seconds)
        self._end = clock() + self.seconds

    @property
    def remaining(self) -> float:
        return max(0.0, self._end - self._clock())

    @property
    def expired(self) -> bool:
        return self.remaining <= 0.0

    def check(self, what: str = "operation") -> float:
        if self.expired:
            raise OperationTimeout(f"deadline of {self.seconds:g}s exceeded during {what}", details={"step": what})
        return self.remaining


# ---- bulkhead ------------------------------------------------------------------------------------------------------------
class Bulkhead:
    """At most ``max_concurrent`` callers at a time; the rest are refused immediately (or after ``wait`` seconds).
    Keeps a slow broker or data vendor from tying up every worker thread."""

    def __init__(self, max_concurrent: int, name: str = "bulkhead", wait: float = 0.0) -> None:
        if max_concurrent < 1:
            raise ValidationError("max_concurrent must be >= 1")
        self.name, self.max_concurrent, self.wait = name, max_concurrent, wait
        self._sem = threading.BoundedSemaphore(max_concurrent)
        self.rejected = 0

    def __enter__(self) -> "Bulkhead":
        got = self._sem.acquire(timeout=self.wait) if self.wait > 0 else self._sem.acquire(blocking=False)
        if not got:
            self.rejected += 1
            raise CapacityError(f"{self.name} is full ({self.max_concurrent} in flight)", details={"bulkhead": self.name})
        return self

    def __exit__(self, *exc: Any) -> None:
        self._sem.release()

    def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        with self:
            return fn(*args, **kwargs)


# ---- budgets and boundaries -------------------------------------------------------------------------------------------------
class ErrorBudget:
    """How many errors are tolerable in a sliding window before something must change."""

    def __init__(self, max_errors: int, window_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic) -> None:
        if max_errors < 1 or window_seconds <= 0:
            raise ValidationError("max_errors must be >= 1 and window_seconds positive")
        self.max_errors, self.window, self._clock = max_errors, window_seconds, clock
        self._events: Deque[float] = deque()
        self._lock = threading.Lock()

    def _trim(self, now: float) -> None:
        while self._events and now - self._events[0] >= self.window:
            self._events.popleft()

    def record(self) -> bool:
        """Count one error. Returns True when the budget is now exhausted."""
        now = self._clock()
        with self._lock:
            self._trim(now)
            self._events.append(now)
            return len(self._events) >= self.max_errors

    @property
    def used(self) -> int:
        with self._lock:
            self._trim(self._clock())
            return len(self._events)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.max_errors

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


class ErrorBoundary:
    """Contain failures of one component so they cannot take down its caller, and escalate when they pile up.

    * every caught error is classified, counted and handed to ``on_error``;
    * a HALT-class error is always re-raised, as is anything listed in ``reraise``;
    * when ``budget`` runs out, ``on_exhausted`` fires once (e.g. trip the kill switch, page someone) and fires again only
      after the budget has recovered.

    Usable three ways: ``boundary.run(fn, ...)``, ``with boundary: ...`` or ``@boundary``.
    """

    def __init__(self, name: str, *, catch: Tuple[Type[BaseException], ...] = (Exception,), policy: Optional[ErrorPolicy] = None,
                 budget: Optional[ErrorBudget] = None, on_error: Optional[Callable[[ErrorInfo, BaseException], None]] = None,
                 on_exhausted: Optional[Callable[[str, ErrorInfo], None]] = None,
                 reraise: Tuple[Type[BaseException], ...] = ()) -> None:
        self.name, self.catch, self.policy, self.budget = name, catch, policy or _DEFAULT_POLICY, budget
        self.on_error, self.on_exhausted, self.reraise = on_error, on_exhausted, reraise
        self.calls = self.errors = 0
        self.by_code: Counter = Counter()
        self.last_error: Optional[ErrorInfo] = None
        self._tripped = False

    def handle(self, exc: BaseException) -> bool:
        """Record ``exc``. Returns True if it was contained, False if the caller must re-raise it."""
        if not isinstance(exc, self.catch) or isinstance(exc, self.reraise):
            return False
        info = self.policy.decide(exc)
        self.errors += 1
        self.by_code[info.code] += 1
        self.last_error = info
        log.log(40 if info.severity.at_least(Severity.ERROR) else 30, "[%s] %s (%s -> %s)", self.name, exc, info.category, info.action.value)
        self._notify(self.on_error, info, exc)
        if self.budget is not None:
            if self.budget.record():
                if not self._tripped:
                    self._tripped = True
                    self._notify(self.on_exhausted, self.name, info)
            else:
                self._tripped = False
        return info.action is not ErrorAction.HALT

    def _notify(self, callback: Optional[Callable], *args: Any) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception as cb_exc:  # noqa: BLE001 - a broken callback must not turn handling into a new failure
            log.error("[%s] error callback failed: %r", self.name, cb_exc)

    def run(self, fn: Callable[..., T], *args: Any, default: Any = None, **kwargs: Any) -> Any:
        self.calls += 1
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001
            if self.handle(exc):
                return default
            raise
        if self.budget is not None and not self.budget.exhausted:
            self._tripped = False
        return result

    def __call__(self, fn: Callable[..., T]) -> Callable[..., Optional[T]]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
            return self.run(fn, *args, **kwargs)
        return wrapper

    def __enter__(self) -> "ErrorBoundary":
        self.calls += 1
        return self

    def __exit__(self, exc_type: Any, exc: Optional[BaseException], tb: Any) -> bool:
        return exc is not None and self.handle(exc)

    def stats(self) -> Dict[str, Any]:
        return {"name": self.name, "calls": self.calls, "errors": self.errors, "by_code": dict(self.by_code),
                "budget_used": self.budget.used if self.budget else None, "exhausted": self._tripped,
                "last_error": self.last_error.to_dict() if self.last_error else None}


# ---- dead letters ------------------------------------------------------------------------------------------------------------
@dataclass
class DeadLetter:
    item: Any
    error: str
    source: str
    attempts: int = 1
    ts: float = field(default_factory=time.time)


class DeadLetterQueue:
    """Bounded store for work that failed for good. Nothing is silently dropped: when the queue is full the oldest entry
    is evicted and counted in ``evicted``."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._items: Deque[DeadLetter] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.evicted = 0

    def add(self, item: Any, error: BaseException, source: str = "", attempts: int = 1) -> DeadLetter:
        letter = DeadLetter(item, repr(error), source, attempts)
        with self._lock:
            if len(self._items) == self._items.maxlen:
                self.evicted += 1
            self._items.append(letter)
        return letter

    def __len__(self) -> int:
        return len(self._items)

    def items(self, source: Optional[str] = None) -> List[DeadLetter]:
        with self._lock:
            return [d for d in self._items if source is None or d.source == source]

    def replay(self, handler: Callable[[Any], Any], source: Optional[str] = None) -> Tuple[int, int]:
        """Hand every stored item to ``handler`` again. Items that succeed leave the queue; the rest stay with one more
        attempt recorded. Returns (succeeded, still_failed)."""
        with self._lock:
            batch = [d for d in self._items if source is None or d.source == source]
        done: List[DeadLetter] = []
        for letter in batch:
            try:
                handler(letter.item)
                done.append(letter)
            except Exception as exc:  # noqa: BLE001
                letter.attempts += 1
                letter.error = repr(exc)
        with self._lock:
            for letter in done:
                if letter in self._items:
                    self._items.remove(letter)
        return len(done), len(batch) - len(done)


# ---- collect, then raise -----------------------------------------------------------------------------------------------------
class ErrorCollector:
    """Run every item of a batch, remember which ones failed, and raise a single :class:`AggregateError` at the end::

        errors = ErrorCollector("loading instruments")
        for row in rows:
            with errors.catch(row["symbol"]):
                master.add(parse(row))
        errors.raise_if_any()
    """

    def __init__(self, what: str = "batch", max_errors: Optional[int] = None) -> None:
        self.what, self.max_errors = what, max_errors
        self.errors: List[Tuple[str, BaseException]] = []

    @contextlib.contextmanager
    def catch(self, label: str = "", catch: Tuple[Type[BaseException], ...] = (Exception,)) -> Iterator[None]:
        try:
            yield
        except catch as exc:
            if classify(exc).action is ErrorAction.HALT:
                raise
            self.errors.append((label, exc))
            if self.max_errors is not None and len(self.errors) >= self.max_errors:
                self.raise_if_any()

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> List[Dict[str, str]]:
        return [{"item": label, "error": str(exc)} for label, exc in self.errors]

    def raise_if_any(self) -> None:
        if self.errors:
            raise AggregateError(f"{self.what}: {len(self.errors)} item(s) failed", [e for _, e in self.errors],
                                 details={"items": [label for label, _ in self.errors][:50]})


@contextlib.contextmanager
def error_context(operation: str, wrap: Type[PlatformError] = PlatformError, **details: Any) -> Iterator[None]:
    """Attach context to whatever fails inside the block.

    A ``PlatformError`` keeps its type and gains ``operation`` plus the given details (existing keys are not overwritten).
    Any other exception is wrapped in ``wrap`` with the original as ``__cause__``, so callers above only ever see the
    platform's own error tree."""
    try:
        yield
    except PlatformError as exc:
        exc.details.setdefault("operation", operation)
        for k, v in details.items():
            exc.details.setdefault(k, v)
        raise
    except Exception as exc:  # noqa: BLE001
        raise wrap(f"{operation} failed: {exc}", details={"operation": operation, "cause": type(exc).__name__, **details}) from exc
