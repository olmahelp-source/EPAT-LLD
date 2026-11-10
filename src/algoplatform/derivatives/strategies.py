"""Option strategy construction, payoff and aggregate Greeks, plus contract helpers for the portfolio ledger."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..core.enums import Side
from ..core.exceptions import ValidationError
from ..core.models import Fill, Instrument
from ..quant.models import bs_greeks, bs_price


@dataclass(frozen=True)
class OptionLeg:
    kind: str                 # call | put | stock
    strike: float
    qty: float                # signed: +long / -short (contracts, or shares for 'stock')
    premium: float = 0.0      # price paid (+) or received per unit when the leg was opened
    multiplier: float = 100.0

    def __post_init__(self) -> None:
        if self.kind not in ("call", "put", "stock"):
            raise ValidationError("kind must be call, put or stock")

    def intrinsic(self, ST: np.ndarray) -> np.ndarray:
        if self.kind == "stock":
            return ST - self.strike
        return np.maximum(ST - self.strike, 0.0) if self.kind == "call" else np.maximum(self.strike - ST, 0.0)


def payoff(legs: Sequence[OptionLeg], ST) -> np.ndarray:
    """Profit/loss at expiry per leg-sum, net of premiums (stock legs use ``strike`` as the entry price)."""
    ST = np.atleast_1d(np.asarray(ST, float))
    total = np.zeros_like(ST)
    for leg in legs:
        mult = 1.0 if leg.kind == "stock" else leg.multiplier
        prem = 0.0 if leg.kind == "stock" else leg.premium
        total += leg.qty * mult * (leg.intrinsic(ST) - prem)
    return total


def value(legs: Sequence[OptionLeg], S: float, T: float, r: float, sigma: float) -> float:
    """Mark-to-model value of the position before expiry (Black-Scholes)."""
    v = 0.0
    for leg in legs:
        if leg.kind == "stock":
            v += leg.qty * (S - leg.strike)
        else:
            v += leg.qty * leg.multiplier * (bs_price(S, leg.strike, T, r, sigma, leg.kind) - leg.premium)
    return v


def greeks(legs: Sequence[OptionLeg], S: float, T: float, r: float, sigma: float) -> Dict[str, float]:
    """Aggregate position Greeks (delta in shares, gamma in shares per $1, vega per vol point, theta per day, rho per 1%)."""
    tot = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    for leg in legs:
        if leg.kind == "stock":
            tot["delta"] += leg.qty
            continue
        g = bs_greeks(S, leg.strike, T, r, sigma, leg.kind)
        for k in tot:
            tot[k] += leg.qty * leg.multiplier * g[k]
    return tot


def profile(legs: Sequence[OptionLeg], lo: float, hi: float, n: int = 2001) -> Dict[str, object]:
    """Max profit / max loss over [lo, hi] and breakeven prices (sign changes of the expiry payoff)."""
    grid = np.linspace(lo, hi, n)
    pl = payoff(legs, grid)
    be = [float(grid[i] - pl[i] * (grid[i + 1] - grid[i]) / (pl[i + 1] - pl[i])) for i in range(n - 1)
          if pl[i] == 0 or pl[i] * pl[i + 1] < 0]
    unbounded_up, unbounded_down = pl[-1] > pl[-2] + 1e-9, pl[0] > pl[1] + 1e-9
    return {"max_profit": float("inf") if unbounded_up or unbounded_down else float(pl.max()),
            "max_loss": float(pl.min()), "breakevens": be, "grid": grid, "pnl": pl}


# ---- builders (premium defaults to model value so examples need no market data) ---------------------------------------
def _leg(kind, strike, qty, S, T, r, sigma, mult) -> OptionLeg:
    return OptionLeg(kind, strike, qty, round(bs_price(S, strike, T, r, sigma, kind), 4), mult)


def vertical_spread(kind: str, long_strike: float, short_strike: float, S: float, T: float, r: float, sigma: float, n: int = 1, mult: float = 100.0) -> List[OptionLeg]:
    return [_leg(kind, long_strike, n, S, T, r, sigma, mult), _leg(kind, short_strike, -n, S, T, r, sigma, mult)]


def straddle(strike: float, S: float, T: float, r: float, sigma: float, n: int = 1, mult: float = 100.0) -> List[OptionLeg]:
    return [_leg("call", strike, n, S, T, r, sigma, mult), _leg("put", strike, n, S, T, r, sigma, mult)]


def strangle(put_strike: float, call_strike: float, S: float, T: float, r: float, sigma: float, n: int = 1, mult: float = 100.0) -> List[OptionLeg]:
    return [_leg("put", put_strike, n, S, T, r, sigma, mult), _leg("call", call_strike, n, S, T, r, sigma, mult)]


def butterfly(kind: str, lo: float, mid: float, hi: float, S: float, T: float, r: float, sigma: float, n: int = 1, mult: float = 100.0) -> List[OptionLeg]:
    if not lo < mid < hi:
        raise ValidationError("need lo < mid < hi")
    return [_leg(kind, lo, n, S, T, r, sigma, mult), _leg(kind, mid, -2 * n, S, T, r, sigma, mult), _leg(kind, hi, n, S, T, r, sigma, mult)]


def iron_condor(put_lo: float, put_hi: float, call_lo: float, call_hi: float, S: float, T: float, r: float, sigma: float, n: int = 1, mult: float = 100.0) -> List[OptionLeg]:
    """Short the inner strikes, buy the wings: collects premium, risk capped at the wing width."""
    if not put_lo < put_hi < call_lo < call_hi:
        raise ValidationError("need put_lo < put_hi < call_lo < call_hi")
    return [_leg("put", put_lo, n, S, T, r, sigma, mult), _leg("put", put_hi, -n, S, T, r, sigma, mult),
            _leg("call", call_lo, -n, S, T, r, sigma, mult), _leg("call", call_hi, n, S, T, r, sigma, mult)]


def covered_call(stock_price: float, call_strike: float, S: float, T: float, r: float, sigma: float, n: int = 1, mult: float = 100.0) -> List[OptionLeg]:
    return [OptionLeg("stock", stock_price, n * mult), _leg("call", call_strike, -n, S, T, r, sigma, mult)]


# ---- contracts in the portfolio ledger -------------------------------------------------------------------------------
def option_symbol(underlying: str, expiry: date, kind: str, strike: float) -> str:
    """OCC-style readable symbol, e.g. AAPL240119C00150000."""
    return f"{underlying}{expiry.strftime('%y%m%d')}{kind[0].upper()}{int(round(strike * 1000)):08d}"


def option_instrument(underlying: str, expiry: date, kind: str, strike: float, multiplier: float = 100.0, tick: float = 0.01) -> Instrument:
    """Instrument whose ``multiplier`` makes the existing Portfolio value contracts correctly (premium x 100 x qty)."""
    return Instrument(option_symbol(underlying, expiry, kind, strike), tick_size=tick, lot_size=1.0, multiplier=multiplier,
                      asset_class="option")


def expiry_fill(symbol: str, position_qty: float, kind: str, strike: float, underlying_price: float, ts=None, fill_id: str = "") -> Optional[Fill]:
    """Cash-settle an expiring contract: a closing fill at its intrinsic value (a tiny positive price when worthless,
    because fills must have positive prices). Returns None for a flat position."""
    if abs(position_qty) < 1e-12:
        return None
    intrinsic = max(underlying_price - strike, 0.0) if kind == "call" else max(strike - underlying_price, 0.0)
    side = Side.SELL if position_qty > 0 else Side.BUY
    return Fill(fill_id or f"EXP-{symbol}", f"EXP-{symbol}", symbol, side, abs(position_qty), max(intrinsic, 1e-9), ts, 0.0, "expiry")
