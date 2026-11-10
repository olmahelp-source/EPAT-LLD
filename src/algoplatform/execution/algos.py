"""Execution algorithms. Each decides, bar by bar, how much to send and at what price.

Algos are pure decision objects: they see a ``SliceContext`` and return a ``SliceDecision``. The
manager sends the resulting child orders through the normal risk-checked path, so an algo can
never bypass a limit. Quantities already working at the broker are counted as committed, so an
algo never over-schedules while earlier slices are still open.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from ..core.exceptions import ValidationError
from ..core.models import Bar
from ..core.registry import Registry
from ..quant.impact import ac_trajectory
from .pricing import limit_price
from .parent import ParentOrder


@dataclass
class SliceContext:
    parent: ParentOrder
    bar: Bar
    bar_index: int                  # bars elapsed since the parent started
    uncommitted: float              # total - filled - working
    working: float                  # open child quantity at the broker
    working_age: int                # bars the oldest working child has waited
    volume_estimate: float          # recent average bar volume
    lot: float = 1.0
    tick: float = 0.01


@dataclass
class SliceDecision:
    qty: float = 0.0
    limit_price: Optional[float] = None
    cancel_working: bool = False    # ask the manager to pull stale working children first


ALGOS: Registry["ExecutionAlgo"] = Registry("algo")


def floor_lot(qty: float, lot: float) -> float:
    return math.floor(qty / lot + 1e-9) * lot if qty > 0 else 0.0


class ExecutionAlgo:
    name = "algo"

    def horizon(self) -> int:
        """Maximum bars the parent may stay active."""
        return 200

    def slice(self, ctx: SliceContext) -> SliceDecision:
        raise NotImplementedError


@ALGOS.register("twap")
class TWAP(ExecutionAlgo):
    """Equal slices over ``bars`` bars."""
    name = "twap"

    def __init__(self, bars: int = 10) -> None:
        if bars < 1:
            raise ValidationError("bars must be >= 1")
        self.bars = bars

    def horizon(self) -> int:
        return self.bars + 2

    def slice(self, ctx: SliceContext) -> SliceDecision:
        left = max(self.bars - ctx.bar_index, 1)
        qty = ctx.uncommitted if left == 1 else ctx.uncommitted / left
        return SliceDecision(floor_lot(qty, ctx.lot))


def u_shape(n: int, depth: float = 1.0) -> np.ndarray:
    """Default intraday volume profile: heavy at the open and close, light mid-session."""
    if n == 1:
        return np.array([1.0])
    x = np.linspace(-1, 1, n)
    w = 1.0 + depth * x ** 2
    return w / w.sum()


def volume_curve_from_history(volumes: List[float], bars: int) -> np.ndarray:
    """Average a history of per-bar volumes (one session per ``bars`` samples) into a profile."""
    v = np.asarray(volumes, float)
    sessions = len(v) // bars
    if sessions < 1:
        raise ValidationError("need at least one full session of volumes")
    prof = v[: sessions * bars].reshape(sessions, bars).mean(axis=0)
    return prof / prof.sum()


@ALGOS.register("vwap")
class VWAP(ExecutionAlgo):
    """Follow a volume profile so the order's average price tracks the market VWAP."""
    name = "vwap"

    def __init__(self, bars: int = 10, curve: Optional[List[float]] = None) -> None:
        if bars < 1:
            raise ValidationError("bars must be >= 1")
        w = np.asarray(curve, float) if curve is not None else u_shape(bars)
        if len(w) != bars or (w < 0).any() or w.sum() <= 0:
            raise ValidationError("curve must have `bars` non-negative weights")
        self.bars, self.cum = bars, np.cumsum(w / w.sum())

    def horizon(self) -> int:
        return self.bars + 2

    def slice(self, ctx: SliceContext) -> SliceDecision:
        k = min(ctx.bar_index, self.bars - 1)
        total = ctx.parent.total
        committed = total - ctx.uncommitted
        want = total * self.cum[k] - committed
        if k == self.bars - 1:
            want = ctx.uncommitted
        return SliceDecision(floor_lot(min(want, ctx.uncommitted), ctx.lot))


@ALGOS.register("pov")
class POV(ExecutionAlgo):
    """Percentage of volume: never more than ``participation`` of recent bar volume per bar."""
    name = "pov"

    def __init__(self, participation: float = 0.1, max_bars: int = 200) -> None:
        if not 0 < participation <= 1:
            raise ValidationError("participation must be in (0, 1]")
        self.participation, self.max_bars = participation, max_bars

    def horizon(self) -> int:
        return self.max_bars

    def slice(self, ctx: SliceContext) -> SliceDecision:
        room = self.participation * ctx.volume_estimate - ctx.working
        return SliceDecision(floor_lot(min(max(room, 0.0), ctx.uncommitted), ctx.lot))


@ALGOS.register("iceberg")
class Iceberg(ExecutionAlgo):
    """Show a small randomised clip as a passive limit order; refresh when filled or stale."""
    name = "iceberg"

    def __init__(self, display: float = 100, variance: float = 0.2, max_wait: int = 3, spread_bps: float = 2.0,
                 mode: str = "touch", seed: int = 0, max_bars: int = 500) -> None:
        if display <= 0 or not 0 <= variance < 1:
            raise ValidationError("display must be positive and variance in [0, 1)")
        self.display, self.variance, self.max_wait = display, variance, max_wait
        self.spread_bps, self.mode, self.max_bars = spread_bps, mode, max_bars
        self._rng = np.random.default_rng(seed)               # seeded: backtests stay reproducible

    def horizon(self) -> int:
        return self.max_bars

    def slice(self, ctx: SliceContext) -> SliceDecision:
        if ctx.working > 0:
            return SliceDecision(0.0, cancel_working=ctx.working_age >= self.max_wait)
        clip = self.display * (1 + self._rng.uniform(-self.variance, self.variance))   # hide the true size
        qty = floor_lot(min(clip, ctx.uncommitted), ctx.lot)
        if qty <= 0 and ctx.uncommitted > 0:
            qty = floor_lot(ctx.uncommitted, ctx.lot)
        px = limit_price(ctx.parent.side, ctx.bar.close, ctx.tick, self.mode, self.spread_bps)
        return SliceDecision(qty, px)


@ALGOS.register("is")
class ImplementationShortfall(ExecutionAlgo):
    """Almgren-Chriss schedule: ``kappa`` trades urgency (impact) against waiting (price risk)."""
    name = "is"

    def __init__(self, bars: int = 10, kappa: float = 1.0) -> None:
        if bars < 1:
            raise ValidationError("bars must be >= 1")
        self.bars, self.kappa = bars, kappa
        self.remaining = ac_trajectory(bars, kappa)

    def horizon(self) -> int:
        return self.bars + 2

    def slice(self, ctx: SliceContext) -> SliceDecision:
        k = min(ctx.bar_index, self.bars - 1)
        total = ctx.parent.total
        committed = total - ctx.uncommitted
        want = total * (1.0 - self.remaining[k + 1]) - committed
        if k == self.bars - 1:
            want = ctx.uncommitted
        return SliceDecision(floor_lot(min(want, ctx.uncommitted), ctx.lot))


def make_algo(name: str, **params: Any) -> ExecutionAlgo:
    return ALGOS.create(name, **params)
