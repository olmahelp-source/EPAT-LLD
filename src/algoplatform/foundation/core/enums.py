"""Shared enumerations."""
from __future__ import annotations

from enum import Enum


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY

    @classmethod
    def from_qty(cls, qty: float) -> "Side":
        return cls.BUY if qty >= 0 else cls.SELL


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"     # needs tags trail_pct or trail_amount


class TimeInForce(str, Enum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class OrderStatus(str, Enum):
    NEW = "NEW"                      # created locally, not yet sent
    PENDING = "PENDING"              # sent, awaiting broker acknowledgement
    ACKNOWLEDGED = "ACKNOWLEDGED"    # resting at the broker/exchange
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    PENDING_CANCEL = "PENDING_CANCEL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL

    @property
    def is_open(self) -> bool:
        return not self.is_terminal


_TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED}


class Role(str, Enum):
    """One platform, different jobs. Each role sees only the desk it needs (see ``app.workspaces``)."""
    ADMIN = "ADMIN"                          # users, roles, configuration
    QUANT = "QUANT"                          # research, models, backtests; never sends an order
    TRADER = "TRADER"                        # runs strategies and places orders on own broker accounts
    PORTFOLIO_MANAGER = "PORTFOLIO_MANAGER"  # allocates capital, approves strategies for live trading
    RISK_MANAGER = "RISK_MANAGER"            # limits, kill switch, risk sign-off; never sends an order
    OPERATIONS = "OPERATIONS"                # connectivity, instruments, alerts, health, schedules
    AUDITOR = "AUDITOR"                      # read-only access to the audit trail and approvals
    VIEWER = "VIEWER"                        # read-only positions and orders


class Permission(str, Enum):
    PLACE_ORDER = "order:place"
    CANCEL_ORDER = "order:cancel"
    VIEW_ORDERS = "order:view"
    MANAGE_STRATEGY = "strategy:manage"
    RUN_BACKTEST = "backtest:run"
    VIEW_PORTFOLIO = "portfolio:view"
    MANAGE_RISK = "risk:manage"
    RESET_KILL_SWITCH = "risk:reset_kill_switch"
    TRIGGER_KILL_SWITCH = "risk:trigger_kill_switch"
    MANAGE_USERS = "users:manage"
    MANAGE_BROKERS = "brokers:manage"
    TRAIN_MODEL = "model:train"
    SUBMIT_STRATEGY = "strategy:submit"      # propose a strategy for paper / live
    APPROVE_STRATEGY = "strategy:approve"    # sign it off (never the same person who submitted)
    MANAGE_ALLOCATION = "allocation:manage"
    VIEW_RISK = "risk:view"
    VIEW_AUDIT = "audit:view"
    MANAGE_ALERTS = "alerts:manage"
    MANAGE_INSTRUMENTS = "instruments:manage"
    OPERATE_SYSTEM = "system:operate"


class KillSwitchState(str, Enum):
    ARMED = "ARMED"          # normal operation, watching limits
    TRIGGERED = "TRIGGERED"  # trading halted
    DISARMED = "DISARMED"    # explicitly disabled (never default)


class RiskAction(str, Enum):
    ALLOW = "ALLOW"
    REJECT = "REJECT"
    REDUCE_ONLY = "REDUCE_ONLY"


class Severity(str, Enum):
    """Shared by error classification, alerts and notifications, so they can be compared."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self.value]

    def at_least(self, other: "Severity") -> bool:
        return self.rank >= other.rank


_SEVERITY_RANK = {"INFO": 0, "WARNING": 1, "ERROR": 2, "CRITICAL": 3}
