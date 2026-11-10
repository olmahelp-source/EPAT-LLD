"""Portfolio / position keeping: cash, positions, P&L, exposures, trade log."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..core.exceptions import ValidationError
from ..core.models import Fill, Instrument, Position


@dataclass
class Trade:
    """A closed (or partially closed) round trip, used by trade analytics."""
    symbol: str
    direction: int              # +1 long, -1 short
    quantity: float
    entry_price: float
    exit_price: float
    entry_ts: Optional[datetime]
    exit_ts: Optional[datetime]
    pnl: float                  # net of commissions
    commission: float
    strategy_id: str = ""
    mae: float = 0.0            # max adverse excursion per unit (price terms, >= 0)
    mfe: float = 0.0            # max favourable excursion per unit (>= 0)

    @property
    def return_pct(self) -> float:
        base = self.entry_price * self.quantity
        return self.pnl / base if base else 0.0

    @property
    def holding_time(self):
        if self.entry_ts is None or self.exit_ts is None:
            return None
        return self.exit_ts - self.entry_ts


class Portfolio:
    def __init__(self, initial_cash: float, instruments: Optional[Dict[str, Instrument]] = None,
                 name: str = "main") -> None:
        if initial_cash <= 0:
            raise ValidationError("initial_cash must be positive")
        self.name = name
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.instruments = instruments or {}
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.total_commission = 0.0
        self.realized_by_strategy: Dict[str, float] = {}
        self._equity_points: List[Tuple[datetime, float]] = []
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    # ---- fills --------------------------------------------------------------------------
    def _mult(self, symbol: str) -> float:
        inst = self.instruments.get(symbol)
        return inst.multiplier if inst else 1.0

    def apply_fill(self, fill: Fill) -> None:
        with self._lock:
            pos = self.positions.get(fill.symbol)
            if pos is None:
                pos = self.positions[fill.symbol] = Position(fill.symbol, multiplier=self._mult(fill.symbol))
            mult = pos.multiplier
            signed = fill.signed_qty
            self.cash -= signed * fill.price * mult + fill.commission
            self.total_commission += fill.commission
            old = pos.quantity
            if abs(old) < 1e-12 or old * signed > 0:                 # opening or adding
                new_qty = old + signed
                pos.avg_price = (pos.avg_price * abs(old) + fill.price * abs(signed)) / abs(new_qty)
                pos.quantity = new_qty
                meta = self._meta.setdefault(fill.symbol, {"ts": fill.ts, "hi": fill.price, "lo": fill.price,
                                                           "entry_comm": 0.0, "qty": 0.0})
                if abs(old) < 1e-12:
                    meta.update(ts=fill.ts, hi=fill.price, lo=fill.price, entry_comm=0.0, qty=0.0)
                meta["entry_comm"] += fill.commission
                meta["qty"] += abs(signed)
            else:                                                     # reducing, closing or flipping
                closing = min(abs(signed), abs(old))
                direction = 1 if old > 0 else -1
                gross = closing * (fill.price - pos.avg_price) * direction * mult
                meta = self._meta.get(fill.symbol, {"ts": None, "hi": pos.avg_price, "lo": pos.avg_price,
                                                    "entry_comm": 0.0, "qty": abs(old)})
                share = closing / meta["qty"] if meta["qty"] else 1.0
                entry_comm = meta["entry_comm"] * share
                exit_comm = fill.commission * closing / abs(signed)
                net = gross - entry_comm - exit_comm
                pos.realized_pnl += gross
                self.realized_by_strategy[fill.strategy_id] = self.realized_by_strategy.get(fill.strategy_id, 0.0) + net
                if direction > 0:
                    mae, mfe = max(pos.avg_price - meta["lo"], 0.0), max(meta["hi"] - pos.avg_price, 0.0)
                else:
                    mae, mfe = max(meta["hi"] - pos.avg_price, 0.0), max(pos.avg_price - meta["lo"], 0.0)
                self.trades.append(Trade(fill.symbol, direction, closing, pos.avg_price, fill.price,
                                         meta["ts"], fill.ts, net, entry_comm + exit_comm,
                                         fill.strategy_id, mae, mfe))
                meta["entry_comm"] -= entry_comm
                meta["qty"] -= closing
                new_qty = old + signed
                if abs(new_qty) < 1e-12:
                    pos.quantity, pos.avg_price = 0.0, 0.0
                    self._meta.pop(fill.symbol, None)
                elif new_qty * old < 0:                               # flipped: remainder opens fresh
                    pos.quantity, pos.avg_price = new_qty, fill.price
                    self._meta[fill.symbol] = {"ts": fill.ts, "hi": fill.price, "lo": fill.price,
                                               "entry_comm": fill.commission * (abs(signed) - closing) / abs(signed),
                                               "qty": abs(new_qty)}
                else:
                    pos.quantity = new_qty
            pos.last_price = fill.price

    # ---- marking ---------------------------------------------------------------------------
    def mark(self, prices: Dict[str, float], ts: Optional[datetime] = None, record: bool = True) -> float:
        with self._lock:
            for sym, px in prices.items():
                pos = self.positions.get(sym)
                if pos is not None and px > 0:
                    pos.last_price = px
                    m = self._meta.get(sym)
                    if m and not pos.is_flat:
                        m["hi"], m["lo"] = max(m["hi"], px), min(m["lo"], px)
            eq = self.equity
            if record and ts is not None:
                self._equity_points.append((ts, eq))
            return eq

    # ---- state -------------------------------------------------------------------------------
    def position(self, symbol: str) -> Position:
        return self.positions.get(symbol) or Position(symbol, multiplier=self._mult(symbol))

    def quantity(self, symbol: str) -> float:
        p = self.positions.get(symbol)
        return p.quantity if p else 0.0

    @property
    def market_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def equity(self) -> float:
        return self.cash + self.market_value

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.market_value) for p in self.positions.values())

    @property
    def net_exposure(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values()) - self.total_commission

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        return self.equity - self.initial_cash

    def open_positions(self) -> Dict[str, Position]:
        return {s: p for s, p in self.positions.items() if not p.is_flat}

    def equity_curve(self) -> pd.Series:
        if not self._equity_points:
            return pd.Series(dtype=float, name="equity")
        idx, vals = zip(*self._equity_points)
        s = pd.Series(vals, index=pd.DatetimeIndex(idx), name="equity")
        return s[~s.index.duplicated(keep="last")]

    def snapshot(self) -> Dict[str, Any]:
        return {"cash": self.cash, "equity": self.equity, "gross": self.gross_exposure,
                "net": self.net_exposure, "realized": self.realized_pnl, "unrealized": self.unrealized_pnl,
                "positions": {s: {"qty": p.quantity, "avg": p.avg_price, "last": p.last_price,
                                  "upl": p.unrealized_pnl} for s, p in self.open_positions().items()}}

    # ---- reconciliation ------------------------------------------------------------------------
    def reconcile(self, broker_positions: Dict[str, float], tol: float = 1e-6) -> Dict[str, Tuple[float, float]]:
        """Return ``{symbol: (internal, broker)}`` for every disagreement."""
        syms = sorted(set(broker_positions) | set(self.open_positions()))
        return {s: (self.quantity(s), broker_positions.get(s, 0.0)) for s in syms
                if abs(self.quantity(s) - broker_positions.get(s, 0.0)) > tol}
