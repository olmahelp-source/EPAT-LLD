"""Position sizing: how many units to hold for a given signal."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..core.config import SizingConfig
from ..core.exceptions import ValidationError
from ..core.models import Instrument
from ..core.registry import Registry
from ..utils.helpers import clip, safe_div


@dataclass
class SizingContext:
    symbol: str
    equity: float
    price: float
    volatility: Optional[float] = None        # per-period return std (e.g. daily)
    atr: Optional[float] = None               # in price units
    stop_distance: Optional[float] = None     # price units; overrides atr when set
    win_rate: Optional[float] = None
    payoff_ratio: Optional[float] = None      # avg win / avg loss
    instrument: Optional[Instrument] = None
    adv: Optional[float] = None               # average daily volume in units
    periods_per_year: float = 252.0
    available_cash: Optional[float] = None


class PositionSizer:
    """Returns the unsigned quantity for a full-conviction (weight = 1) position."""
    name = "base"

    def base_quantity(self, ctx: SizingContext) -> float:
        raise NotImplementedError


SIZERS: Registry[PositionSizer] = Registry("sizer")


@SIZERS.register("fixed_quantity")
class FixedQuantity(PositionSizer):
    def __init__(self, quantity: float = 1.0) -> None:
        if quantity <= 0:
            raise ValidationError("quantity must be positive")
        self.quantity = quantity

    def base_quantity(self, ctx: SizingContext) -> float:
        return self.quantity


@SIZERS.register("fixed_fractional")
class FixedFractional(PositionSizer):
    """Allocate a fixed fraction of equity as notional."""

    def __init__(self, fraction: float = 0.1) -> None:
        if not 0 < fraction <= 1:
            raise ValidationError("fraction must be in (0, 1]")
        self.fraction = fraction

    def base_quantity(self, ctx: SizingContext) -> float:
        return ctx.equity * self.fraction / ctx.price


@SIZERS.register("volatility_target")
class VolatilityTarget(PositionSizer):
    """Scale notional so the position contributes ``target_vol`` annualised volatility."""

    def __init__(self, target_vol: float = 0.15, max_leverage: float = 2.0) -> None:
        self.target_vol, self.max_leverage = target_vol, max_leverage

    def base_quantity(self, ctx: SizingContext) -> float:
        if not ctx.volatility or ctx.volatility <= 0:
            return 0.0
        ann_vol = ctx.volatility * math.sqrt(ctx.periods_per_year)
        leverage = min(self.target_vol / ann_vol, self.max_leverage)
        return ctx.equity * leverage / ctx.price


@SIZERS.register("risk_per_trade")
class RiskPerTrade(PositionSizer):
    """Lose at most ``risk_pct`` of equity if the stop is hit."""

    def __init__(self, risk_pct: float = 0.01, atr_multiple: float = 2.0) -> None:
        if not 0 < risk_pct <= 0.1:
            raise ValidationError("risk_pct must be in (0, 0.1]")
        self.risk_pct, self.atr_multiple = risk_pct, atr_multiple

    def base_quantity(self, ctx: SizingContext) -> float:
        dist = ctx.stop_distance or (ctx.atr * self.atr_multiple if ctx.atr else None)
        if not dist or dist <= 0:
            return 0.0
        return ctx.equity * self.risk_pct / dist


def kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
    """Discrete Kelly: f* = p - (1 - p) / b. Negative edge returns 0."""
    if payoff_ratio <= 0 or not 0 <= win_rate <= 1:
        raise ValidationError("need 0<=win_rate<=1 and payoff_ratio>0")
    return max(win_rate - (1 - win_rate) / payoff_ratio, 0.0)


def kelly_gaussian(mean: float, variance: float) -> float:
    """Continuous Kelly leverage f* = mu / sigma^2 for a normally distributed return."""
    return safe_div(mean, variance)


def optimal_f(trade_returns: Sequence[float], grid: int = 200) -> float:
    """Vince's optimal f by maximising terminal wealth multiple over a grid.

    ``trade_returns`` are per-trade P&L as a fraction of risked capital (losses < 0).
    """
    r = np.asarray(trade_returns, dtype=float)
    if r.size == 0 or r.min() >= 0:
        return 0.0
    worst = -r.min()
    best_f, best_twr = 0.0, 1.0
    for f in np.linspace(1 / grid, 1.0, grid):
        hpr = 1.0 + f * r / worst
        if np.any(hpr <= 0):
            continue
        twr = float(np.prod(hpr))
        if twr > best_twr:
            best_f, best_twr = float(f), twr
    return best_f


@SIZERS.register("kelly")
class KellySizer(PositionSizer):
    def __init__(self, fraction: float = 0.5, max_leverage: float = 1.0) -> None:
        self.fraction, self.max_leverage = fraction, max_leverage

    def base_quantity(self, ctx: SizingContext) -> float:
        if ctx.win_rate is None or ctx.payoff_ratio is None:
            return 0.0
        f = min(kelly_fraction(ctx.win_rate, ctx.payoff_ratio) * self.fraction, self.max_leverage)
        return ctx.equity * f / ctx.price


def drawdown_scalar(drawdown: float, start: float = 0.05, floor_at: float = 0.20, min_scale: float = 0.0) -> float:
    """Linear de-risking multiplier: 1 until ``start`` drawdown, falling to ``min_scale`` at ``floor_at``."""
    dd = abs(drawdown)
    if dd <= start:
        return 1.0
    if dd >= floor_at:
        return min_scale
    return 1.0 - (1.0 - min_scale) * (dd - start) / (floor_at - start)


class PositionSizeManager:
    """Applies a sizer then the hard caps that apply to every strategy."""

    def __init__(self, sizer: PositionSizer, cfg: Optional[SizingConfig] = None,
                 allow_short: bool = True, max_adv_participation: float = 0.1) -> None:
        self.sizer = sizer
        self.cfg = cfg or SizingConfig()
        self.allow_short = allow_short
        self.max_adv_participation = max_adv_participation

    @classmethod
    def from_config(cls, cfg: SizingConfig, allow_short: bool = True) -> "PositionSizeManager":
        kind = cfg.method
        if kind == "fixed_fractional":
            sizer = SIZERS.create(kind, cfg.max_position_pct)
        elif kind == "volatility_target":
            sizer = SIZERS.create(kind, cfg.target_volatility)
        elif kind == "risk_per_trade":
            sizer = SIZERS.create(kind, cfg.risk_per_trade_pct)
        elif kind == "kelly":
            sizer = SIZERS.create(kind, cfg.kelly_fraction)
        else:
            sizer = SIZERS.create(kind)
        return cls(sizer, cfg, allow_short)

    def target_quantity(self, weight: float, ctx: SizingContext, scale: float = 1.0) -> float:
        """Signed target quantity for ``weight`` in [-1, 1], after all caps.

        ``scale`` carries portfolio-level multipliers such as drawdown de-risking.
        """
        if ctx.price <= 0 or ctx.equity <= 0 or weight == 0:
            return 0.0
        if weight < 0 and (not self.allow_short or (ctx.instrument and not ctx.instrument.shortable)):
            return 0.0
        qty = self.sizer.base_quantity(ctx) * abs(weight) * clip(scale, 0.0, 1.0)
        mult = ctx.instrument.multiplier if ctx.instrument else 1.0
        # cap 1: per-symbol share of equity
        cap_equity = ctx.equity * self.cfg.max_position_pct / (ctx.price * mult)
        # cap 2: keep a cash buffer for longs
        cap_cash = math.inf
        if weight > 0 and ctx.available_cash is not None:
            cap_cash = max(ctx.available_cash - ctx.equity * self.cfg.cash_buffer_pct, 0.0) / (ctx.price * mult)
        # cap 3: liquidity
        cap_adv = ctx.adv * self.max_adv_participation if ctx.adv else math.inf
        qty = min(qty, cap_equity, cap_cash, cap_adv)
        if ctx.instrument:
            qty = abs(ctx.instrument.round_qty(qty))
        else:
            qty = math.floor(qty + 1e-9)
        return math.copysign(qty, weight) if qty > 0 else 0.0
