"""RiskManager: single entry point combining pre-trade checks, monitoring and kill switch."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core import events as ev
from ..core.config import RiskConfig
from ..core.enums import RiskAction
from ..core.events import EventBus
from ..core.exceptions import KillSwitchActive, RiskLimitBreach
from ..core.logging import AuditLog, get_logger
from ..core.models import Order
from ..portfolio.sizing import drawdown_scalar
from ..utils.timeutils import Clock, SystemClock
from .checks import Check, RiskContext, default_checks
from .killswitch import KillSwitch

log = get_logger(__name__)


@dataclass
class RiskDecision:
    action: RiskAction
    reasons: List[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.action is RiskAction.ALLOW


class RiskMonitor:
    """Watches equity and system health; trips the kill switch on hard-limit breaches."""

    def __init__(self, cfg: RiskConfig, kill_switch: KillSwitch, bus: Optional[EventBus] = None) -> None:
        self.cfg, self.kill_switch = cfg, kill_switch
        self.bus = bus or EventBus()
        self.peak_equity = 0.0
        self.day_start_equity = 0.0
        self.last_equity = 0.0
        self.consecutive_errors = 0
        self._warned: set = set()

    def start_day(self, equity: float) -> None:
        self.day_start_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        self._warned.clear()

    @property
    def drawdown(self) -> float:
        return (self.last_equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0.0

    @property
    def daily_pnl_pct(self) -> float:
        return (self.last_equity - self.day_start_equity) / self.day_start_equity if self.day_start_equity > 0 else 0.0

    def update_equity(self, equity: float) -> None:
        if self.peak_equity == 0:
            self.peak_equity = self.day_start_equity = equity
        self.last_equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        self._evaluate("daily_loss", -self.daily_pnl_pct, self.cfg.max_daily_loss_pct,
                       f"daily loss {self.daily_pnl_pct:.2%} breached -{self.cfg.max_daily_loss_pct:.2%}")
        self._evaluate("drawdown", -self.drawdown, self.cfg.max_drawdown_pct,
                       f"drawdown {self.drawdown:.2%} breached -{self.cfg.max_drawdown_pct:.2%}")

    def _evaluate(self, key: str, value: float, limit: float, message: str) -> None:
        if value >= limit:
            self.kill_switch.trip(message, source=f"monitor:{key}")
        elif value >= 0.8 * limit and key not in self._warned:   # early warning at 80% of the limit
            self._warned.add(key)
            log.warning("risk warning: %s at %.2f%% of %.2f%% limit", key, value * 100, limit * 100)
            self.bus.publish(ev.RISK_BREACH, {"level": "warning", "metric": key, "value": value, "limit": limit})

    def record_error(self, what: str = "") -> None:
        self.consecutive_errors += 1
        if self.consecutive_errors >= self.cfg.max_consecutive_errors:
            self.kill_switch.trip(f"{self.consecutive_errors} consecutive errors ({what})", source="monitor:errors")

    def record_success(self) -> None:
        self.consecutive_errors = 0

    def check_data_age(self, age_seconds: float, symbol: str = "") -> None:
        if age_seconds > self.cfg.max_data_staleness_sec * 5:
            self.kill_switch.trip(f"market data stale for {symbol or 'feed'}: {age_seconds:.0f}s", source="monitor:data")


class RiskManager:
    def __init__(self, cfg: Optional[RiskConfig] = None, bus: Optional[EventBus] = None,
                 audit: Optional[AuditLog] = None, clock: Optional[Clock] = None,
                 kill_switch: Optional[KillSwitch] = None, checks: Optional[List[Check]] = None) -> None:
        self.cfg = cfg or RiskConfig()
        self.bus = bus or EventBus()
        self.audit = audit or AuditLog()
        self.clock = clock or SystemClock()
        self.kill_switch = kill_switch or KillSwitch(self.bus, self.audit)
        self.monitor = RiskMonitor(self.cfg, self.kill_switch, self.bus)
        self.checks = checks if checks is not None else default_checks(self.cfg)
        self.rejections: Dict[str, int] = {}

    def add_check(self, check: Check) -> None:
        self.checks.append(check)

    def evaluate(self, order: Order, ctx: RiskContext) -> RiskDecision:
        """Never raises for a business rejection; returns the full list of reasons."""
        reasons: List[str] = []
        flatten = bool(order.tags.get("flatten"))
        if self.kill_switch.is_triggered:
            # while halted only risk-reducing flatten orders (created by the platform) may pass
            if not flatten:
                reasons.append(f"kill switch active: {self.kill_switch.reason}")
        else:
            for chk in self.checks:
                try:
                    msg = chk.check(order, ctx)
                except Exception as exc:  # noqa: BLE001 - a broken check must fail CLOSED
                    msg = f"check {chk.name} errored: {exc!r}"
                if msg:
                    reasons.append(f"{chk.name}: {msg}")
                    self.rejections[chk.name] = self.rejections.get(chk.name, 0) + 1
                    break                       # first failure is enough; avoids charging rate counters
        if reasons:
            self.audit.record("risk", "order.risk_rejected", order=order.client_order_id, reasons=reasons)
            self.bus.publish(ev.RISK_BREACH, {"level": "reject", "order": order.client_order_id, "reasons": reasons})
            return RiskDecision(RiskAction.REJECT, reasons)
        return RiskDecision(RiskAction.ALLOW)

    def validate(self, order: Order, ctx: RiskContext) -> None:
        """Raising variant for callers that prefer exceptions."""
        d = self.evaluate(order, ctx)
        if not d.allowed:
            if self.kill_switch.is_triggered and not order.tags.get("flatten"):
                raise KillSwitchActive(d.reasons[0])
            raise RiskLimitBreach("; ".join(d.reasons), details={"order": order.client_order_id})

    def exposure_scalar(self) -> float:
        """Multiplier in [0, 1] applied to position sizes as drawdown deepens."""
        return drawdown_scalar(self.monitor.drawdown, start=self.cfg.max_drawdown_pct * 0.25,
                               floor_at=self.cfg.max_drawdown_pct, min_scale=0.0)
