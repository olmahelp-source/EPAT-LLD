"""Exception hierarchy for the whole platform.

Every error raised on purpose derives from :class:`PlatformError`, so callers can
catch one base class at a service boundary while inner code catches the specific
subclass it knows how to handle. ``retryable`` tells retry helpers whether trying
again can help (a network blip) or not (a validation failure).
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class PlatformError(Exception):
    """Base class for all platform errors."""

    retryable: bool = False

    def __init__(self, message: str = "", *, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = dict(details or {})

    def to_dict(self) -> Dict[str, Any]:
        return {"error": self.code, "message": self.message, "details": self.details}

    def __str__(self) -> str:  # include code for log readability
        return f"[{self.code}] {self.message}" if self.message else f"[{self.code}]"


# ---- configuration / validation ------------------------------------------------
class ConfigError(PlatformError):
    pass


class ValidationError(PlatformError):
    pass


# ---- data ------------------------------------------------------------------------
class DataError(PlatformError):
    pass


class MissingDataError(DataError):
    pass


class DataProviderError(DataError):
    """A market-data provider (Alpaca, Interactive Brokers, ...) refused or failed a request."""


class ProviderAuthError(DataProviderError):
    """The provider rejected the credentials: retrying will not help, the keys must be fixed."""


class ProviderUnavailableError(DataProviderError):
    """Rate limit, server error or network failure: worth retrying later."""
    retryable = True


# ---- security --------------------------------------------------------------------
class AuthError(PlatformError):
    pass


class AuthenticationError(AuthError):
    pass


class AuthorizationError(AuthError):
    pass


class UserError(PlatformError):
    pass


# ---- brokers ---------------------------------------------------------------------
class BrokerError(PlatformError):
    pass


class BrokerConnectionError(BrokerError):
    retryable = True


class OrderRejectedError(BrokerError):
    """The broker/exchange refused the order. Never retried automatically."""


class RateLimitError(BrokerError):
    retryable = True


# ---- orders ----------------------------------------------------------------------
class OrderError(PlatformError):
    pass


class InvalidTransitionError(OrderError):
    pass


class DuplicateOrderError(OrderError):
    pass


class OrderNotFoundError(OrderError):
    pass


# ---- risk ------------------------------------------------------------------------
class RiskError(PlatformError):
    pass


class RiskLimitBreach(RiskError):
    """A pre-trade check failed; the order must not be sent."""


class KillSwitchActive(RiskError):
    """Trading is halted by the kill switch."""


# ---- strategy / research -----------------------------------------------------------
class StrategyError(PlatformError):
    pass


class BacktestError(PlatformError):
    pass


class OptimizationError(PlatformError):
    pass


class StatisticsError(PlatformError):
    pass


class ExpressionError(StrategyError):
    """A formula / rule expression is malformed or uses something that is not allowed."""


class GovernanceError(PlatformError):
    """A lifecycle or approval rule was violated (e.g. the author approving their own strategy)."""


# ---- instruments / models -------------------------------------------------------------------
class InstrumentError(DataError):
    pass


class ModelError(PlatformError):
    """A learning model was misused: not fitted, wrong shapes, degenerate training data."""


# ---- notifications ----------------------------------------------------------------------------
class NotificationError(PlatformError):
    pass


class DeliveryError(NotificationError):
    """A channel could not deliver right now (network, rate limit); trying again can help."""
    retryable = True


# ---- resilience ---------------------------------------------------------------------------------
class OperationTimeout(PlatformError):
    retryable = True


class CapacityError(PlatformError):
    """A bulkhead / queue is full; the caller should back off."""
    retryable = True


class AggregateError(PlatformError):
    """Several failures reported as one, so a batch can finish before it complains."""

    def __init__(self, message: str = "", errors: Optional[list] = None, **kwargs: Any) -> None:
        self.errors = list(errors or [])
        details = dict(kwargs.pop("details", None) or {})
        details.setdefault("count", len(self.errors))
        details.setdefault("errors", [repr(e) for e in self.errors[:20]])
        super().__init__(message or f"{len(self.errors)} operation(s) failed", details=details, **kwargs)
