"""TradingPlatform: the facade that wires users, brokers, strategies, risk and execution.

Every public method takes the acting ``User`` and enforces role-based permissions, so a UI or
API layer can call it directly without re-implementing access control.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from ..backtest.engine import BacktestEngine, BacktestResult
from ..brokers.base import BrokerGateway, create_gateway
from ..core.config import PlatformConfig
from ..core.enums import OrderType, Permission, Side
from ..core.events import EventBus
from ..core.exceptions import ConfigError
from ..core.logging import AuditLog, get_logger, setup_logging
from ..core.models import Order
from ..data.store import MarketData
from ..optimize.optimizers import OptimizationResult, backtest_objective, optimize
from ..optimize.space import ParameterSpace
from ..portfolio.allocation import FundAllocator
from ..strategy.base import STRATEGIES, Strategy
from ..strategy.builder import create_from_spec, spec_factory
from ..users.manager import BrokerAccountManager, User, UserManager
from .engine import TradingEngine

log = get_logger(__name__)


class TradingPlatform:
    def __init__(self, config: Optional[PlatformConfig] = None) -> None:
        self.config = config or PlatformConfig()
        setup_logging(self.config.log_level)
        self.bus = EventBus()
        self.audit = AuditLog(self.config.audit_path)
        self.users = UserManager(self.audit)
        self.accounts = BrokerAccountManager(audit=self.audit, gateway_factory=self._make_gateway)
        self.allocator = FundAllocator(self.config.initial_capital)
        self.engines: Dict[str, TradingEngine] = {}

    # ---- factories ---------------------------------------------------------------------------
    def _make_gateway(self, broker: str, credentials: Dict[str, str], paper: bool) -> BrokerGateway:
        if self.config.environment == "paper" and not paper:
            raise ConfigError("platform is in 'paper' environment; live broker accounts are disabled")
        return create_gateway(broker, credentials, paper)

    def create_strategy(self, user: User, name_or_spec: Any, symbols: Sequence[str], **params: Any) -> Strategy:
        UserManager.authorize(user, Permission.MANAGE_STRATEGY)
        if isinstance(name_or_spec, dict):
            return create_from_spec(name_or_spec, symbols, **params)
        return STRATEGIES.create(name_or_spec, symbols, **params)

    def _factory(self, name_or_spec: Any, symbols: Sequence[str]):
        if isinstance(name_or_spec, dict):
            return spec_factory(name_or_spec, symbols)
        cls = STRATEGIES.get(name_or_spec)
        return lambda **p: cls(symbols, **p)

    # ---- research ---------------------------------------------------------------------------
    def backtest(self, user: User, name_or_spec: Any, data: MarketData, symbols: Optional[Sequence[str]] = None,
                 params: Optional[Dict[str, Any]] = None, start=None, end=None) -> BacktestResult:
        UserManager.authorize(user, Permission.RUN_BACKTEST)
        symbols = list(symbols or data.symbols)
        strat = self._factory(name_or_spec, symbols)(**(params or {}))
        res = BacktestEngine(data, self.config).run(strat, start=start, end=end)
        self.audit.record(user.username, "backtest.run", strategy=strat.name, symbols=symbols)
        return res

    def optimize(self, user: User, name_or_spec: Any, data: MarketData, space: ParameterSpace,
                 symbols: Optional[Sequence[str]] = None, metric: str = "sharpe", method: str = "random",
                 start=None, end=None, **opt_kwargs: Any) -> OptimizationResult:
        UserManager.authorize(user, Permission.RUN_BACKTEST)
        symbols = list(symbols or data.symbols)
        eng = BacktestEngine(data, self.config)
        obj = backtest_objective(eng, self._factory(name_or_spec, symbols), metric, start, end)
        out = optimize(obj, space, method, **opt_kwargs)
        self.audit.record(user.username, "optimize.run", method=method, trials=out.n_trials)
        return out

    # ---- trading -----------------------------------------------------------------------------
    def open_session(self, user: User, account_id: str, strategies: Sequence[tuple], instruments=None,
                     clock=None, store=None) -> TradingEngine:
        """Create a trading engine on one of the user's broker accounts.
        ``strategies`` is ``[(Strategy, capital_pct), ...]``."""
        UserManager.authorize(user, Permission.PLACE_ORDER)
        gateway = self.accounts.open_gateway(user, account_id)
        engine = TradingEngine(self.config, gateway, instruments, clock, self.bus, self.audit, store=store)
        for strat, pct in strategies:
            engine.add_strategy(strat, pct)
        self.engines[account_id] = engine
        self.audit.record(user.username, "session.open", account=account_id)
        return engine

    def place_manual_order(self, user: User, engine: TradingEngine, symbol: str, side: Side, quantity: float,
                           order_type: OrderType = OrderType.MARKET, limit_price: Optional[float] = None) -> Order:
        """Human order entry: same risk gate and OMS as algorithmic orders."""
        UserManager.authorize(user, Permission.PLACE_ORDER)
        return engine.execution.submit(symbol, side, quantity, order_type, limit_price,
                                       strategy_id=f"manual:{user.username}")

    def cancel_all(self, user: User, engine: TradingEngine) -> int:
        UserManager.authorize(user, Permission.CANCEL_ORDER)
        return engine.execution.cancel_all()

    # ---- kill switch ---------------------------------------------------------------------------
    def trip_kill_switch(self, user: User, engine: TradingEngine, reason: str) -> bool:
        UserManager.authorize(user, Permission.TRIGGER_KILL_SWITCH)
        return engine.risk.kill_switch.trip(reason or "manual", source=f"user:{user.username}")

    def reset_kill_switch(self, user: User, engine: TradingEngine, reason: str, approver: Optional[User] = None) -> None:
        engine.risk.kill_switch.reset(user, reason, approver)

    # ---- allocation ------------------------------------------------------------------------------
    def allocate_capital(self, user: User, weights: Dict[str, float]) -> Dict[str, float]:
        UserManager.authorize(user, Permission.MANAGE_RISK)
        return self.allocator.allocate(weights)
