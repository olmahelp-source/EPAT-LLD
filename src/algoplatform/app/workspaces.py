"""Role desks: one platform, a different front door for each job.

``TradingPlatform`` owns the shared services. A *desk* is the slice of them that one role works with. Opening a desk
checks the role's permission once; every method that needs a further permission checks that too, so a UI can show a
user exactly the desks :func:`desks_for` returns and call them directly.

==============  ===================  =================================================================================
desk            typical role         what it is for
==============  ===================  =================================================================================
research        QUANT                backtest, optimise, score results, select instruments, register models, propose
trading         TRADER               open sessions, place and cancel orders, personal alerts
portfolio       PORTFOLIO_MANAGER    allocate capital, approve or reject promotions
risk            RISK_MANAGER         risk snapshot, halt / resume, demote a live strategy, risk alerts
operations      OPERATIONS           health, notification channels, dead letters, instruments, schedules
compliance      AUDITOR              audit trail, approval history, unacknowledged critical alerts (read-only)
admin           ADMIN                users and roles
==============  ===================  =================================================================================

The separation is the control: research cannot send an order, trading cannot approve its own strategy or reset its own
kill switch, risk and compliance cannot trade. A live (non-paper) session only accepts strategies the governance layer
has approved for live trading.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Type

from ..analytics.evaluation import EvaluationLimits, StrategyVerdict, evaluate_backtest
from ..backtest.engine import BacktestResult
from ..core.enums import OrderType, Permission, Role, Severity, Side
from ..core.exceptions import AuthorizationError, GovernanceError
from ..core.models import Order
from ..governance.lifecycle import StrategyRecord
from ..governance.models import ModelVersion
from ..notify.alerts import AlertRule
from ..notify.channels import Channel
from ..optimize.optimizers import OptimizationResult
from ..optimize.space import ParameterSpace
from ..relationships.manager import RelationshipManager
from ..selection.selector import SelectionResult, UniverseSelector
from ..users.manager import User, UserManager
from .engine import TradingEngine


class Desk:
    """A role-scoped view of the platform. Subclasses set ``name`` and the permission needed to open the desk."""
    name = "desk"
    requires: Permission

    def __init__(self, platform: Any, user: User) -> None:
        UserManager.authorize(user, self.requires)
        self.platform, self.user = platform, user

    def _need(self, permission: Permission) -> None:
        UserManager.authorize(self.user, permission)

    def _audit(self, action: str, **details: Any) -> None:
        self.platform.audit.record(self.user.username, action, **details)

    def _saved(self, result: Any = None) -> Any:
        """Persist the shared services after a change (a no-op without a store) and pass the result through."""
        self.platform.autosave()
        return result


class ResearchDesk(Desk):
    name, requires = "research", Permission.RUN_BACKTEST

    def backtest(self, name_or_spec: Any, data: Any, symbols: Optional[Sequence[str]] = None, params: Optional[Dict[str, Any]] = None,
                 start: Any = None, end: Any = None) -> BacktestResult:
        return self.platform.backtest(self.user, name_or_spec, data, symbols, params, start, end)

    def optimize(self, name_or_spec: Any, data: Any, space: ParameterSpace, method: str = "bayesian", **kwargs: Any) -> OptimizationResult:
        return self.platform.optimize(self.user, name_or_spec, data, space, method=method, **kwargs)

    def evaluate(self, result: BacktestResult, n_trials: int = 1, limits: Optional[EvaluationLimits] = None) -> StrategyVerdict:
        """Scorecard for a backtest. Pass the number of configurations that were tried (``OptimizationResult.n_trials``)."""
        return evaluate_backtest(result, n_trials, limits)

    def select_universe(self, selector: UniverseSelector, data: Any, asof: Any = None) -> SelectionResult:
        result = selector.select(data, asof)
        self._audit("research.select_universe", chosen=result.symbols, asof=str(asof))
        return result

    def relationships(self, prices: Any, sectors: Optional[Mapping[str, str]] = None, **kwargs: Any) -> RelationshipManager:
        manager = RelationshipManager(sectors)
        manager.build(prices, **kwargs)
        return manager

    def register_strategy(self, strategy_id: str, description: Optional[Dict[str, Any]] = None) -> StrategyRecord:
        return self._saved(self.platform.lifecycle.register(self.user, strategy_id, description))

    def propose(self, strategy_id: str, verdict: StrategyVerdict, note: str = "") -> Any:
        """Ask for the strategy to move to the next stage (research -> paper -> live). Someone else must approve."""
        return self._saved(self.platform.lifecycle.submit(self.user, strategy_id, verdict, note))

    def register_model(self, name: str, model: Any, metrics: Optional[Mapping[str, float]] = None, data: Any = None, **kwargs: Any) -> ModelVersion:
        self._need(Permission.TRAIN_MODEL)
        version = self.platform.models.register(name, model, metrics, data=data, created_by=self.user.username, **kwargs)
        self._audit("model.register", model=name, version=version.version)
        return self._saved(version)


class TradingDesk(Desk):
    name, requires = "trading", Permission.PLACE_ORDER

    def open_session(self, account_id: str, strategies: Sequence[tuple], **kwargs: Any) -> TradingEngine:
        """Start a trading engine on one of the user's broker accounts with ``[(strategy, capital_share), ...]``.
        On a live account every strategy must have been approved for live trading; paper accounts take anything."""
        account = self.platform.accounts._owned(self.user, account_id)
        if not account.paper:
            for strategy, _ in strategies:
                self.platform.lifecycle.require_live(strategy.id)
        return self.platform.open_session(self.user, account_id, strategies, **kwargs)

    def place_order(self, engine: TradingEngine, symbol: str, side: Side, quantity: float, order_type: OrderType = OrderType.MARKET,
                    limit_price: Optional[float] = None) -> Order:
        return self.platform.place_manual_order(self.user, engine, symbol, side, quantity, order_type, limit_price)

    def cancel_all(self, engine: TradingEngine) -> int:
        return self.platform.cancel_all(self.user, engine)

    def halt(self, engine: TradingEngine, reason: str) -> bool:
        """Any trader may stop trading. Starting again is the risk desk's decision."""
        return self.platform.trip_kill_switch(self.user, engine, reason)

    def add_alert(self, rule: AlertRule) -> AlertRule:
        """A personal alert: its notifications go to this user's own channels only."""
        self._need(Permission.MANAGE_ALERTS)
        rule.user_id = self.user.user_id
        return self._saved(self.platform.alerts.add(rule))

    def subscribe(self, channel: Channel, min_severity: Severity = Severity.INFO, topics: Optional[Sequence[str]] = None) -> None:
        self.platform.notifier.subscribe(self.user.user_id, channel, min_severity, topics)
        self._saved()


class PortfolioDesk(Desk):
    name, requires = "portfolio", Permission.MANAGE_ALLOCATION

    def allocate(self, weights: Dict[str, float]) -> Dict[str, float]:
        out = self.platform.allocator.allocate(weights)
        self._audit("allocation.set", weights=weights)
        return out

    def pending_promotions(self) -> List[StrategyRecord]:
        return self.platform.lifecycle.pending()

    def approve(self, strategy_id: str, comment: str = "") -> StrategyRecord:
        return self._saved(self.platform.lifecycle.approve(self.user, strategy_id, comment))

    def reject(self, strategy_id: str, reason: str) -> StrategyRecord:
        return self._saved(self.platform.lifecycle.reject(self.user, strategy_id, reason))


class RiskDesk(Desk):
    name, requires = "risk", Permission.VIEW_RISK

    def snapshot(self, engine: TradingEngine, relationships: Optional[RelationshipManager] = None) -> Dict[str, Any]:
        """Where the book stands: equity, exposure, kill-switch state, and (given a relationship map) how many
        independent bets the open positions really are."""
        pf = engine.portfolio
        equity = pf.equity
        positions = {s: p.market_value for s, p in pf.open_positions().items()}
        out: Dict[str, Any] = {"equity": equity, "gross_exposure": pf.gross_exposure, "net_exposure": pf.net_exposure,
                               "gross_leverage": pf.gross_exposure / equity if equity > 0 else float("inf"),
                               "realized_pnl": pf.realized_pnl, "unrealized_pnl": pf.unrealized_pnl, "positions": positions,
                               "kill_switch": engine.risk.kill_switch.state.value, "strategy_errors": engine.strategy_errors}
        if relationships is not None and positions:
            out["concentration"] = relationships.concentration(positions)
        return out

    def halt(self, engine: TradingEngine, reason: str) -> bool:
        return self.platform.trip_kill_switch(self.user, engine, reason)

    def resume(self, engine: TradingEngine, reason: str, approver: Optional[User] = None) -> None:
        self.platform.reset_kill_switch(self.user, engine, reason, approver)

    def demote(self, strategy_id: str, reason: str) -> StrategyRecord:
        """Take a live strategy back to paper trading."""
        return self._saved(self.platform.lifecycle.demote(self.user, strategy_id, reason))

    def approve(self, strategy_id: str, comment: str = "") -> StrategyRecord:
        return self._saved(self.platform.lifecycle.approve(self.user, strategy_id, comment))

    def add_alert(self, rule: AlertRule) -> AlertRule:
        self._need(Permission.MANAGE_ALERTS)
        return self._saved(self.platform.alerts.add(rule))


class OperationsDesk(Desk):
    name, requires = "operations", Permission.OPERATE_SYSTEM

    def health(self) -> Dict[str, Any]:
        return self.platform.health.run()

    def add_channel(self, channel: Channel, min_severity: Severity = Severity.INFO, topics: Optional[Sequence[str]] = None, **kwargs: Any) -> None:
        self._need(Permission.MANAGE_ALERTS)
        self.platform.notifier.add_channel(channel, min_severity, topics, **kwargs)
        self._audit("notify.add_channel", channel=channel.name, min_severity=Severity(min_severity).value)
        self._saved()

    def undelivered(self) -> List[Dict[str, Any]]:
        return [{"channel": d.item[0], "title": d.item[1].title, "error": d.error, "attempts": d.attempts} for d in self.platform.notifier.dead_letters.items()]

    def replay_undelivered(self) -> Dict[str, int]:
        ok, failed = self.platform.notifier.replay_dead_letters()
        return self._saved({"delivered": ok, "still_failing": failed})

    def load_instruments(self, records: Sequence[Mapping[str, Any]], on_error: str = "skip", overwrite: bool = False) -> Dict[str, Any]:
        """Load contract specs. ``overwrite``: a row for a symbol already in the master replaces it (to correct a spec);
        without it such a row is rejected, so a careless re-load cannot silently change live terms."""
        self._need(Permission.MANAGE_INSTRUMENTS)
        report = self.platform.instruments.load_records(records, overwrite=overwrite, on_error=on_error)
        self._audit("instruments.load", loaded=report["loaded"], rejected=len(report["rejected"]), overwrite=overwrite)
        return self._saved(report)

    def jobs(self) -> List[Dict[str, Any]]:
        return self.platform.scheduler.jobs()

    def run_scheduled(self) -> List[str]:
        return self.platform.scheduler.run_pending()


class ComplianceDesk(Desk):
    """Read-only by construction: it exposes no method that changes anything."""
    name, requires = "compliance", Permission.VIEW_AUDIT

    def audit_trail(self, action: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.platform.audit.entries(action)

    def strategy_history(self, strategy_id: str) -> List[Dict[str, Any]]:
        return list(self.platform.lifecycle.get(strategy_id).history)

    def strategies(self) -> List[Dict[str, Any]]:
        return [{"strategy": r.strategy_id, "owner": r.owner, "stage": r.stage.value, "pending": r.pending is not None}
                for r in self.platform.lifecycle.records()]

    def open_critical_alerts(self) -> List[Dict[str, Any]]:
        return [n.to_dict() for n in self.platform.notifier.unacknowledged()]


class AdminDesk(Desk):
    name, requires = "admin", Permission.MANAGE_USERS

    def create_user(self, username: str, password: str, role: Role = Role.TRADER) -> User:
        return self.platform.users.register(username, password, role, actor=self.user)

    def set_role(self, user_id: str, role: Role) -> None:
        if user_id == self.user.user_id and role is not Role.ADMIN:
            raise GovernanceError("an administrator cannot remove their own admin role (another admin must)", code="SELF_DEMOTION")
        self.platform.users.set_role(self.user, user_id, role)

    def deactivate(self, user_id: str) -> None:
        self.platform.users.deactivate(self.user, user_id)

    def users(self) -> List[User]:
        return self.platform.users.list_users(self.user)


DESKS: Dict[str, Type[Desk]] = {d.name: d for d in (ResearchDesk, TradingDesk, PortfolioDesk, RiskDesk, OperationsDesk, ComplianceDesk, AdminDesk)}


def open_desk(platform: Any, user: User, name: str) -> Desk:
    """Open one desk, or raise ``AuthorizationError`` if the user's role does not include it."""
    if name not in DESKS:
        raise AuthorizationError(f"unknown desk '{name}'", details={"desks": sorted(DESKS)})
    return DESKS[name](platform, user)


def desks_for(platform: Any, user: User) -> Dict[str, Desk]:
    """Every desk this user may open: what a UI should show them."""
    return {name: cls(platform, user) for name, cls in DESKS.items() if user.has(cls.requires)}
