"""Event-driven backtester built on the same TradingEngine used for paper/live trading."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from ..analytics.metrics import performance_metrics, to_returns
from ..analytics.report import build_report
from ..analytics.trades import trade_statistics, trades_frame
from ..app.engine import TradingEngine
from ..brokers.costs import CostModel
from ..brokers.paper import PaperBroker
from ..core import events as ev
from ..core.config import PlatformConfig
from ..core.events import EventBus
from ..core.exceptions import BacktestError
from ..core.logging import AuditLog
from ..core.enums import OrderType, Side
from ..core.models import Bar, Fill
from ..data.store import MarketData
from ..portfolio.sizing import PositionSizeManager
from ..strategy.base import Strategy
from ..utils.timeutils import SimClock


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: List[Any]
    fills: List[Fill]
    summary: Dict[str, Any]
    strategy: Dict[str, Any]
    config: PlatformConfig
    benchmark: Optional[pd.Series] = None          # benchmark *returns*
    kill_switch_triggered: bool = False
    order_count: int = 0
    rejected_orders: int = 0
    parents: List[Any] = field(default_factory=list)
    orders: List[Any] = field(default_factory=list)

    @property
    def returns(self) -> pd.Series:
        return to_returns(self.equity)

    def metrics(self, rf: float = 0.0) -> Dict[str, float]:
        return performance_metrics(self.equity, benchmark=self.benchmark, rf=rf)

    def trade_stats(self) -> Dict[str, Any]:
        return trade_statistics(self.trades)

    def trades_frame(self) -> pd.DataFrame:
        return trades_frame(self.trades)

    def report(self, n_trials: int = 1) -> Dict[str, Any]:
        return build_report(self.equity, self.trades, benchmark=self.benchmark, n_trials=n_trials)


class BacktestEngine:
    """Replays ``MarketData`` through strategy -> sizing -> risk -> OMS -> simulated broker.

    ``start``/``end`` bound the *trading* window; earlier bars are fed as warm-up history only,
    so a strategy with a 200-bar lookback can trade from the first day of a test window without
    peeking ahead."""

    def __init__(self, data: MarketData, config: Optional[PlatformConfig] = None, max_participation: Optional[float] = None,
                 rebalance_band: float = 0.10, flatten_on_kill: bool = True) -> None:
        if not data.symbols:
            raise BacktestError("no market data loaded")
        self.data = data
        self.config = config or PlatformConfig(environment="paper")
        self.max_participation = max_participation
        self.rebalance_band = rebalance_band
        self.flatten_on_kill = flatten_on_kill

    def run(self, strategy: Union[Strategy, Sequence[Tuple[Strategy, float]]], start=None, end=None,
            sizer: Optional[PositionSizeManager] = None, close_at_end: bool = True,
            algo: Optional[Tuple[str, dict]] = None, algo_min_notional: float = 0.0) -> BacktestResult:
        """Run one strategy (single-use instance) or ``[(strategy, capital_pct), ...]``."""
        slots = [(strategy, 1.0)] if isinstance(strategy, Strategy) else list(strategy)
        if not slots:
            raise BacktestError("no strategy supplied")
        for strat, _ in slots:
            missing = [s for s in strat.symbols if s not in self.data.symbols]
            if missing:
                raise BacktestError(f"strategy {strat.id} needs data for {missing}")
        cfg = self.config
        bus = EventBus()
        fills: List[Fill] = []
        bus.subscribe(ev.FILL, lambda _t, f: fills.append(f))
        clock = SimClock()
        broker = PaperBroker(cfg.initial_capital, CostModel(cfg.costs), self.max_participation, cfg.risk.allow_short)
        engine = TradingEngine(cfg, broker, dict(self.data.instruments), clock, bus, AuditLog(max_memory=2000),
                               sizer=sizer, flatten_on_kill=self.flatten_on_kill, rebalance_band=self.rebalance_band,
                               algo=algo, algo_min_notional=algo_min_notional)
        for strat, pct in slots:
            engine.add_strategy(strat, pct)
        start_ts = pd.Timestamp(start) if start is not None else None
        end_ts = pd.Timestamp(end) if end is not None else None
        engine.start()
        try:
            last_bars = None
            for bars in self.data.bar_batches():
                ts = next(iter(bars.values())).ts
                if start_ts is not None and ts < start_ts:
                    engine.warm_up(bars)
                    continue
                if end_ts is not None and ts > end_ts:
                    break
                engine.on_bars(bars)
                last_bars = bars
            if close_at_end and last_bars is not None and not engine.risk.kill_switch.is_triggered:
                self._close_out(engine, last_bars)
        finally:
            engine.stop()
        eq = engine.portfolio.equity_curve()
        if len(eq) < 3:
            raise BacktestError("backtest produced fewer than 3 equity points; check dates and data")
        orders = engine.oms.all_orders()
        return BacktestResult(
            equity=eq, trades=list(engine.portfolio.trades), fills=fills, summary=engine.summary(),
            strategy=[s.describe() for s, _ in slots] if len(slots) > 1 else slots[0][0].describe(),
            config=copy.deepcopy(cfg), benchmark=self._benchmark(eq.index),
            kill_switch_triggered=engine.risk.kill_switch.is_triggered, order_count=len(orders),
            rejected_orders=sum(1 for o in orders if o.status.value == "REJECTED"),
            parents=list(engine.algos.parents.values()), orders=orders)

    @staticmethod
    def _close_out(engine: TradingEngine, last_bars) -> None:
        """Liquidate at the final close so open P&L is realised into the trade log. The extra
        fill happens at the last bar's close, charged normal costs."""
        engine.algos.cancel_all(reason="end of backtest")
        for sym, pos in list(engine.portfolio.open_positions().items()):
            b = last_bars.get(sym)
            if b is None:
                continue
            engine.execution.submit(sym, Side.SELL if pos.quantity > 0 else Side.BUY, abs(pos.quantity),
                                    OrderType.MARKET, reduce_only=True, strategy_id="close_out")
            engine.broker.process_bar(Bar(b.ts, b.symbol, b.close, b.close, b.close, b.close, b.volume))
        engine.portfolio.mark(engine.last_prices, engine.clock.now())

    def _benchmark(self, index: pd.Index) -> pd.Series:
        """Equal-weight buy-and-hold of the loaded universe, as a returns series."""
        rets = self.data.closes().pct_change().mean(axis=1)
        return rets.reindex(index).dropna()
