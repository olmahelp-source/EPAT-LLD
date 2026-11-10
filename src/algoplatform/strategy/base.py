"""Strategy SDK: the one contract every strategy (built-in, rule-based, user code) follows.

A strategy is a pure decision function: given a read-only context it returns ``Signal``
objects stating the *target exposure* per symbol. It never creates orders, never talks
to a broker and never sees future data. Sizing, risk and execution happen downstream.
"""
from __future__ import annotations

import abc
import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..core.exceptions import MissingDataError, StrategyError
from ..core.models import Bar, Fill, Signal
from ..core.registry import Registry


class HistoryView:
    """Append-only, causal OHLCV store: strategies can only see bars already delivered.

    Backed by doubling numpy buffers, so ``window`` is an O(1) read-only slice and the same
    object serves both backtests and live trading (bars are appended as they arrive)."""
    COLS = ("open", "high", "low", "close", "volume")

    def __init__(self, initial_capacity: int = 1024) -> None:
        self._buf: Dict[str, np.ndarray] = {}
        self._n: Dict[str, int] = {}
        self._cap = initial_capacity

    def append(self, bar: Bar) -> None:
        s = bar.symbol
        if s not in self._buf:
            self._buf[s] = np.empty((self._cap, 5))
            self._n[s] = 0
        n, buf = self._n[s], self._buf[s]
        if n == len(buf):
            buf = self._buf[s] = np.concatenate([buf, np.empty_like(buf)])
        buf[n] = (bar.open, bar.high, bar.low, bar.close, bar.volume)
        self._n[s] = n + 1

    def count(self, symbol: str) -> int:
        return self._n.get(symbol, 0)

    def window(self, symbol: str, n: Optional[int] = None) -> Dict[str, np.ndarray]:
        if symbol not in self._buf:
            raise MissingDataError(f"no history for {symbol}")
        k = self._n[symbol]
        lo = 0 if n is None else max(0, k - n)
        out = {}
        for j, c in enumerate(self.COLS):
            v = self._buf[symbol][lo:k, j]
            v.flags.writeable = False
            out[c] = v
        return out


@dataclass
class StrategyContext:
    ts: datetime
    bars: Dict[str, Bar]                       # bars that arrived at this timestamp
    history: HistoryView
    positions: Dict[str, float]                # signed quantity currently held
    equity: float
    cash: float
    prices: Dict[str, float] = field(default_factory=dict)

    def window(self, symbol: str, n: Optional[int] = None) -> Dict[str, np.ndarray]:
        return self.history.window(symbol, n)

    def closes(self, symbol: str, n: Optional[int] = None) -> np.ndarray:
        return self.history.window(symbol, n)["close"]

    def bars_seen(self, symbol: str) -> int:
        return self.history.count(symbol)

    def position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)


STRATEGIES: Registry["Strategy"] = Registry("strategy")


class Strategy(abc.ABC):
    """Subclass, declare ``PARAMS`` defaults, implement :meth:`on_bar`.

    ``PARAM_BOUNDS`` optionally maps parameter name -> (low, high) for optimisers.
    """
    name = "strategy"
    PARAMS: Dict[str, Any] = {}
    PARAM_BOUNDS: Dict[str, tuple] = {}

    def __init__(self, symbols: Sequence[str], strategy_id: Optional[str] = None, **params: Any) -> None:
        if not symbols:
            raise StrategyError("strategy needs at least one symbol")
        unknown = set(params) - set(self.PARAMS)
        if unknown:
            raise StrategyError(f"unknown parameters {sorted(unknown)} for {self.name}",
                                details={"valid": sorted(self.PARAMS)})
        self.symbols = list(symbols)
        self.params: Dict[str, Any] = {**copy.deepcopy(self.PARAMS), **params}
        self.id = strategy_id or self.name
        self.validate_params()

    # ---- hooks ------------------------------------------------------------------------------
    def validate_params(self) -> None:
        """Raise StrategyError for inconsistent parameters (e.g. fast >= slow)."""

    @property
    def warmup(self) -> int:
        """Bars of history needed before the strategy may emit signals."""
        return 1

    def on_start(self, ctx: StrategyContext) -> None:
        pass

    @abc.abstractmethod
    def on_bar(self, ctx: StrategyContext) -> List[Signal]:
        """Return target-weight signals. Symbols not mentioned keep their current target."""

    def on_fill(self, fill: Fill) -> None:
        pass

    def on_end(self) -> None:
        pass

    # ---- helpers ------------------------------------------------------------------------------
    def p(self, name: str) -> Any:
        return self.params[name]

    def ready(self, ctx: StrategyContext, symbol: str) -> bool:
        return ctx.bars_seen(symbol) >= self.warmup

    def clone(self, **overrides: Any) -> "Strategy":
        return self.__class__(self.symbols, self.id, **{**self.params, **overrides})

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "id": self.id, "symbols": self.symbols, "params": self.params}


def strategy_factory(name: str, symbols: Sequence[str], **params: Any) -> Callable[..., Strategy]:
    """A zero-arg-friendly factory ``f(**overrides) -> Strategy`` used by optimisers."""
    cls = STRATEGIES.get(name)

    def make(**overrides: Any) -> Strategy:
        return cls(symbols, **{**params, **overrides})
    return make
