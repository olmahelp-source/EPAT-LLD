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
