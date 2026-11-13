"""The paper-trading session behind the web UI: a trading engine fed bar by bar, and the metrics alert rules read.

A session is fed by a sample random walk, or by replaying downloaded daily history. It never looks ahead: each bar is
generated or read only when it is played."""
from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.exceptions import ValidationError
from ..core.logging import get_logger
from ..core.models import Bar
from ..persistence import SqliteStore


log = get_logger(__name__)
ALERT_METRICS = ("drawdown", "daily_pnl_pct", "equity", "gross_leverage", "strategy_errors")


class LiveSession:
    """A paper-trading engine fed bar by bar: by a sample random walk (``feed="sample"``), or by replaying downloaded
    daily history (``feed="history"``, the frames in ``replay``), so a strategy can be paper-traded on real prices."""

    def __init__(self, engine, symbols: List[str], seed: int, strategy: Optional[str] = None,
                 params: Optional[Dict[str, Any]] = None, store: Optional[SqliteStore] = None, feed: str = "sample",
                 replay: Optional[Dict[str, pd.DataFrame]] = None, pos: int = 0) -> None:
        self.engine, self.symbols, self.seed = engine, symbols, seed
        self.strategy, self.params, self.store = strategy, params or {}, store
        self.rng = np.random.default_rng(seed)
        self.last = {s: float(self.rng.uniform(50, 300)) for s in symbols}
        self.next_ts = pd.Timestamp("2024-01-01")
        self.feed, self.pos = feed, int(pos)
        self.replay: Optional[Dict[str, pd.DataFrame]] = None
        self.replay_index: Optional[pd.DatetimeIndex] = None
        if feed == "history" and replay:
            common = None
            for f in replay.values():
                common = f.index if common is None else common.intersection(f.index)
            if common is None or len(common) < 2:
                raise ValidationError("the downloaded histories of these symbols share fewer than 2 dates")
            self.replay_index = common.sort_values()
            self.replay = {s: replay[s].loc[self.replay_index] for s in symbols}
            if self.pos < len(self.replay_index):
                self.next_ts = self.replay_index[self.pos]
                self.last = {s: float(self.replay[s]["open"].iloc[self.pos]) for s in symbols}
        self.lock = threading.RLock()
        self.history: Dict[str, List[float]] = {s: [] for s in symbols}
        self.cond = threading.Condition()
        self.version = 0
        self.on_step: Optional[Callable[["LiveSession"], Any]] = None      # set by the app: evaluates alert rules
        self.recovery: Optional[Dict[str, Any]] = None                    # set when the session was resumed after a restart

    def touch(self) -> None:
        """Wake stream listeners: the visible state changed."""
        with self.cond:
            self.version += 1
            self.cond.notify_all()

    def save_meta(self) -> None:
        """Feed state needed to resume after a restart (orders/fills/positions are in the journal)."""
        if self.store is not None:
            self.store.put("session", {"symbols": self.symbols, "seed": self.seed, "strategy": self.strategy,
                                       "params": self.params, "last": self.last, "history": self.history,
                                       "next_ts": self.next_ts.isoformat(), "bars": self.engine.bars_processed,
                                       "feed": self.feed, "pos": self.pos})

    def replay_status(self) -> Optional[Dict[str, Any]]:
        if self.feed != "history":
            return None
        idx = self.replay_index
        if idx is None or not len(idx):
            return {"pos": self.pos, "total": None, "available": False, "from": None, "to": None}
        return {"pos": self.pos, "total": len(idx), "available": self.replay is not None,
                "from": idx[0].date().isoformat(), "to": idx[-1].date().isoformat()}

    def _next_bars(self) -> Dict[str, Bar]:
        if self.feed == "history":
            if self.replay is None or self.replay_index is None:
                raise ValidationError("the downloaded history for this session is not available any more (removed from the library, "
                                      "or the platform is in sample-data-only mode): close the session")
            if self.pos >= len(self.replay_index):
                raise ValidationError(f"all {len(self.replay_index)} bars of downloaded history have been replayed: close the session "
                                      "or download a longer range")
            ts = self.replay_index[self.pos]
            bars = {}
            for s in self.symbols:
                row = self.replay[s].iloc[self.pos]
                bars[s] = Bar(ts, s, float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"]), float(row["volume"]))
            self.pos += 1
            self.next_ts = self.replay_index[self.pos] if self.pos < len(self.replay_index) else ts + pd.offsets.BDay(1)
            return bars
        bars = {}
        for s in self.symbols:
            o = self.last[s]
            c = o * float(np.exp(self.rng.normal(0.0003, 0.012)))
            bars[s] = Bar(self.next_ts, s, o, max(o, c) * 1.003, min(o, c) * 0.997, c, 1e6)
        self.next_ts = self.next_ts + pd.offsets.BDay(1)
        return bars

    def step(self, n: int) -> None:
        with self.lock:
            for i in range(n):
                try:
                    bars = self._next_bars()
                except ValidationError:
                    if i == 0:
                        raise
                    break                                              # history ran out part-way: keep the bars already played
                for s, b in bars.items():
                    self.last[s] = b.close
                    self.history[s].append(round(b.close, 4))
                    self.history[s] = self.history[s][-200:]
                self.engine.on_bars(bars)
                if self.on_step is not None:
                    try:
                        self.on_step(self)
                    except Exception as exc:  # noqa: BLE001 - alerting must never stop the feed
                        log.error("alert evaluation failed: %r", exc)
            self.save_meta()
        self.touch()


def session_metrics(engine: Any, prices: Dict[str, float]) -> Dict[str, Any]:
    """What alert rules are evaluated against after each paper-trading step."""
    pf = engine.portfolio
    equity = pf.equity
    out: Dict[str, Any] = {"drawdown": engine.risk.monitor.drawdown, "daily_pnl_pct": engine.risk.monitor.daily_pnl_pct, "equity": equity,
                           "gross_leverage": pf.gross_exposure / equity if equity > 0 else None, "strategy_errors": engine.strategy_errors}
    out.update({f"price.{s}": float(p) for s, p in prices.items()})
    return out


__all__: List[str] = ["LiveSession", "session_metrics", "ALERT_METRICS"]
