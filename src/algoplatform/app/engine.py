"""TradingEngine: one event loop used unchanged for backtest, paper and live trading.

Per bar:  broker matching -> mark-to-market -> risk monitor -> strategies -> sizing ->
          net target positions -> orders (risk-gated) -> next bar.
Signals computed on the close of bar t become orders that can only fill from bar t+1
onward, which removes the classic same-bar look-ahead bias.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from ..brokers.base import BrokerGateway
from ..core import events as ev
from ..core.config import PlatformConfig
from ..core.enums import OrderType, Side
from ..core.events import EventBus
from ..core.exceptions import StrategyError, ValidationError
from ..execution.manager import AlgoExecutionManager
from ..core.logging import AuditLog, get_logger
from ..core.models import Bar, Fill, Instrument, Signal
from ..orders.execution import ExecutionService
from ..orders.oms import OrderManager
from ..portfolio.portfolio import Portfolio
from ..portfolio.sizing import PositionSizeManager, SizingContext
from ..quant import indicators as ind
from ..risk.manager import RiskManager
from ..strategy.base import HistoryView, Strategy, StrategyContext
from ..utils.timeutils import Clock, SimClock, SystemClock

log = get_logger(__name__)


@dataclass
class _Slot:
    strategy: Strategy
    capital_pct: float
    targets: Dict[str, float] = field(default_factory=dict)   # symbol -> target quantity
    consecutive_errors: int = 0


class TradingEngine:
    def __init__(self, cfg: PlatformConfig, broker: BrokerGateway, instruments: Optional[Dict[str, Instrument]] = None,
                 clock: Optional[Clock] = None, bus: Optional[EventBus] = None, audit: Optional[AuditLog] = None,
                 risk: Optional[RiskManager] = None, sizer: Optional[PositionSizeManager] = None,
                 flatten_on_kill: bool = True, rebalance_band: float = 0.10, vol_lookback: int = 20,
                 algo: Optional[tuple] = None, algo_min_notional: float = 0.0, store=None) -> None:
        self.cfg = cfg
        self.clock = clock or SystemClock()
        self.bus = bus or EventBus()
        self.audit = audit or AuditLog(cfg.audit_path)
        self.instruments = dict(instruments or {})
        self.broker = broker
        self.portfolio = Portfolio(cfg.initial_capital, self.instruments)
        self.oms = OrderManager(self.bus, self.audit, self.clock)
        self.risk = risk or RiskManager(cfg.risk, self.bus, self.audit, self.clock)
        self.sizer = sizer or PositionSizeManager.from_config(cfg.sizing, cfg.risk.allow_short)
        self.history = HistoryView()
        self.last_prices: Dict[str, float] = {}
        self.price_times: Dict[str, datetime] = {}
        self.execution = ExecutionService(self.oms, broker, self.risk, self.portfolio, self.clock,
                                          lambda: self.last_prices, lambda: self.price_times)
        self.algos = AlgoExecutionManager(self.execution, self.oms, self.bus, self.instruments)
        self.algo = algo                       # e.g. ("twap", {"bars": 5}); None = single market orders
        self.algo_min_notional = algo_min_notional
        self._bars: Dict[str, Bar] = {}
        self._slots: Dict[str, _Slot] = {}
        self.flatten_on_kill = flatten_on_kill
        self.rebalance_band = rebalance_band
        self.vol_lookback = vol_lookback
        self._day = None
        self._last_equity: Optional[float] = None
        self.bars_processed = 0
        self.strategy_errors = 0
        self.store = store
        self.journal = None
        if store is not None:
            from ..persistence.journal import Journal
            self.journal = Journal(store, self.bus, on_error=lambda exc: self.risk.kill_switch.trip(
                f"persistence failure, state can no longer be saved: {exc!r}", source="journal"))
        self.bus.subscribe(ev.FILL, self._on_fill)
        self.risk.kill_switch.on_trip(self._on_kill)

    # ---- wiring ------------------------------------------------------------------------------
    def add_strategy(self, strategy: Strategy, capital_pct: float = 1.0) -> None:
        if not 0 < capital_pct <= 1:
            raise ValidationError("capital_pct must be in (0, 1]")
        if strategy.id in self._slots:
            raise ValidationError(f"strategy id '{strategy.id}' already added")
        if sum(s.capital_pct for s in self._slots.values()) + capital_pct > 1 + 1e-9:
            raise ValidationError("total strategy capital exceeds 100%")
        self._slots[strategy.id] = _Slot(strategy, capital_pct)
        for s in strategy.symbols:
            self.instruments.setdefault(s, Instrument(s))
            self.portfolio.instruments.setdefault(s, self.instruments[s])

    def recover(self, halt_on_break: bool = True):
        """Rebuild state from the persistence store (call before ``start``). Returns a RecoveryReport."""
        if self.store is None:
            raise ValidationError("engine has no store; pass store=SqliteStore(...)")
        from ..persistence.recovery import recover
        return recover(self, self.store, halt_on_break)

    def start(self) -> None:
        self.broker.connect()
        self.risk.monitor.start_day(self.portfolio.equity)

    def stop(self) -> None:
        try:
            self.algos.cancel_all(reason="engine stopped")
            self.execution.cancel_all()
        finally:
            for slot in self._slots.values():
                try:
                    slot.strategy.on_end()
                except Exception as exc:  # noqa: BLE001
                    log.error("strategy %s on_end failed: %r", slot.strategy.id, exc)
            self.broker.disconnect()

    # ---- event handlers ------------------------------------------------------------------------
    def _on_fill(self, _topic: str, fill: Fill) -> None:
        self.portfolio.apply_fill(fill)
        slot = self._slots.get(fill.strategy_id)
        if slot:
            try:
                slot.strategy.on_fill(fill)
            except Exception as exc:  # noqa: BLE001
                log.error("strategy on_fill failed: %r", exc)

    def _on_kill(self, reason: str) -> None:
        self.algos.cancel_all(reason="kill switch")
        self.execution.cancel_all()
        if self.flatten_on_kill:
            self.execution.flatten_all(reason)

    # ---- main loop -------------------------------------------------------------------------------
    def run(self, bar_batches: Iterable[Dict[str, Bar]]) -> None:
        self.start()
        try:
            for bars in bar_batches:
                self.on_bars(bars)
        finally:
            self.stop()

    def warm_up(self, bars: Dict[str, Bar]) -> None:
        """Feed history only: no fills, no decisions. Used for bars before the trading window."""
        for b in bars.values():
            self.history.append(b)
            self.last_prices[b.symbol] = b.close

    def on_bars(self, bars: Dict[str, Bar]) -> None:
        if not bars:
            return
        ts = max(b.ts for b in bars.values())
        if isinstance(self.clock, SimClock):
            self.clock.set(ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts)
        simulated = hasattr(self.broker, "process_bar")
        for b in bars.values():
            self.history.append(b)
            self.last_prices[b.symbol] = b.close
            self.price_times[b.symbol] = self.clock.now()
            if simulated:
                self.broker.process_bar(b)                   # fills from earlier decisions, at this bar's open
        day = ts.date()
        equity = self.portfolio.mark(self.last_prices, ts)
        if day != self._day:
            self._day = day
            # the day's reference is the PREVIOUS close, otherwise a gap-down bar hides its own loss
            self.risk.monitor.start_day(self._last_equity if self._last_equity is not None else equity)
        self.risk.monitor.update_equity(equity)
        self._last_equity = equity
        self.bars_processed += 1
        self._bars = bars
        self.algos.on_bars(bars)                              # keeps volume stats; works active parents below
        self.bus.publish(ev.BAR, bars)
        if self.risk.kill_switch.is_triggered:
            self.execution.flatten_all("kill switch active") if self.flatten_on_kill else None
            return
        # 1) every strategy updates its targets, 2) only then are net positions traded, so
        # opposing views cancel internally instead of generating two real orders
        touched: Dict[str, List[str]] = {}
        for slot in self._slots.values():
            for sym in self._run_strategy(slot, ts, bars):
                touched.setdefault(sym, []).append(slot.strategy.id)
        for sym, ids in touched.items():
            self._rebalance(sym, "+".join(sorted(set(ids))))     # netted orders are attributed to every strategy behind them

    # ---- strategy -> orders -------------------------------------------------------------------------
    def _run_strategy(self, slot: _Slot, ts: datetime, bars: Dict[str, Bar]) -> List[str]:
        """Run one strategy and update its target quantities. Returns the symbols it touched, in signal
        order (never a set: iteration order must not depend on string hashing, or runs diverge)."""
        strat = slot.strategy
        ctx = StrategyContext(ts, bars, self.history,
                              {s: p.quantity for s, p in self.portfolio.open_positions().items()},
                              self.portfolio.equity, self.portfolio.cash, dict(self.last_prices))
        try:
            signals = strat.on_bar(ctx) or []
            for sig in signals:
                if not isinstance(sig, Signal):
                    raise StrategyError(f"{strat.id} returned {type(sig).__name__}, expected Signal")
        except Exception as exc:  # noqa: BLE001 - user strategy code is untrusted
            self.strategy_errors += 1
            log.error("strategy %s failed at %s: %r", strat.id, ts, exc)
            self.audit.record(strat.id, "strategy.error", error=repr(exc))
            slot.consecutive_errors += 1          # per-strategy streak: a healthy peer must not mask a broken one
            if slot.consecutive_errors >= self.cfg.risk.max_consecutive_errors:
                self.risk.kill_switch.trip(f"strategy {strat.id} failed {slot.consecutive_errors} times in a row",
                                           source="engine:strategy")
            return []
        touched: Dict[str, None] = {}               # ordered set
        for sig in signals:
            if sig.symbol not in self.last_prices:
                continue
            slot.targets[sig.symbol] = self._target_qty(slot, sig)
            touched[sig.symbol] = None
        slot.consecutive_errors = 0
        return list(touched)

    def _target_qty(self, slot: _Slot, sig: Signal) -> float:
        sym, price = sig.symbol, self.last_prices[sig.symbol]
        inst = self.instruments.get(sym) or Instrument(sym)
        w = self.history.window(sym, max(self.vol_lookback, 15) + 1)
        closes = w["close"]
        vol = float(np.std(np.diff(np.log(closes)), ddof=1)) if len(closes) > 3 else None
        atr = ind.atr(w["high"], w["low"], w["close"], min(14, max(2, len(closes) - 1)))[-1] if len(closes) > 3 else None
        pos_qty = self.portfolio.quantity(sym)
        ctx = SizingContext(
            symbol=sym, equity=self.portfolio.equity * slot.capital_pct, price=price,
            volatility=vol if vol and math.isfinite(vol) and vol > 0 else None,
            atr=float(atr) if atr is not None and np.isfinite(atr) else None,
            stop_distance=(abs(price - sig.stop_price) if sig.stop_price else None),
            instrument=inst, adv=float(np.mean(self.history.window(sym, 20)["volume"])) or None,
            available_cash=self.portfolio.cash + max(pos_qty, 0.0) * price * inst.multiplier)
        return self.sizer.target_quantity(sig.target_weight, ctx, self.risk.exposure_scalar())

    def net_target(self, symbol: str) -> float:
        """Sum of all strategies' targets: offsetting views net out internally before any order."""
        return sum(s.targets.get(symbol, 0.0) for s in self._slots.values())

    def _rebalance(self, symbol: str, strategy_id: str) -> None:
        desired = self.net_target(symbol)
        current = self.portfolio.quantity(symbol)
        inst = self.instruments.get(symbol) or Instrument(symbol)
        # working children are in the OMS pending exposure; the unsent part of an active parent is added
        pending = self.oms.pending_exposure().get(symbol, 0.0) + self.algos.signed_unsent(symbol)
        delta = inst.round_qty(desired - (current + pending))
        if delta == 0:
            return
        # hysteresis: ignore small re-sizing of an existing same-side position to avoid churn
        if desired != 0 and current * desired > 0 and abs(delta) < self.rebalance_band * abs(current):
            return
        price = self.last_prices.get(symbol, 0.0)
        if self.algo is not None and abs(delta) * price >= self.algo_min_notional:
            if self.algos.active(symbol):                      # target moved: drop the old plan, re-plan from fills
                self.algos.cancel_all(symbol, "retargeted")
                pending = self.oms.pending_exposure().get(symbol, 0.0)
                delta = inst.round_qty(desired - (current + pending))
                if delta == 0:
                    return
            name, params = self.algo
            parent = self.algos.start(symbol, Side.from_qty(delta), abs(delta), name, price, strategy_id,
                                      start_ts=self.clock.now(), **params)
            bar = self._bars.get(symbol)
            if bar is not None:
                self.algos.work_now(parent, bar)
            return
        self.execution.submit(symbol, Side.from_qty(delta), abs(delta), OrderType.MARKET, strategy_id=strategy_id,
                              tags={"arrival_price": price})

    # ---- reporting -----------------------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        snap = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in self.portfolio.snapshot().items()}
        snap.update({"orders": len(self.oms.all_orders()), "trades": len(self.portfolio.trades),
                     "kill_switch": self.risk.kill_switch.state.value, "bars": self.bars_processed,
                     "strategy_errors": self.strategy_errors, "commission": self.portfolio.total_commission})
        return snap
