"""Synthetic market data for tests, demos and Monte Carlo (seeded and reproducible)."""
from __future__ import annotations

import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...foundation.core.exceptions import ValidationError


def _ohlc_from_close(close: np.ndarray, rng: np.random.Generator, intraday_vol: float,
                     base_volume: float) -> pd.DataFrame:
    n = len(close)
    open_ = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, intraday_vol * 0.2, n))
    open_ = np.clip(open_, close.min() * 0.5, None)
    spread_up = np.abs(rng.normal(0, intraday_vol, n))
    spread_dn = np.abs(rng.normal(0, intraday_vol, n))
    high = np.maximum(open_, close) * (1 + spread_up)
    low = np.minimum(open_, close) * (1 - np.minimum(spread_dn, 0.5))
    volume = base_volume * np.exp(rng.normal(0, 0.3, n))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def _index(n: int, start: str, freq: str) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n) if freq == "B" else pd.date_range(start=start, periods=n, freq=freq)


def gbm(n: int = 756, mu: float = 0.08, sigma: float = 0.2, s0: float = 100.0, seed: Optional[int] = 0,
        start: str = "2020-01-01", freq: str = "B", periods_per_year: float = 252.0,
        base_volume: float = 1e6) -> pd.DataFrame:
    """Geometric Brownian motion OHLCV."""
    if n < 2 or sigma < 0 or s0 <= 0:
        raise ValidationError("bad gbm parameters")
    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year
    r = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    close = s0 * np.exp(np.cumsum(r))
    df = _ohlc_from_close(close, rng, sigma * np.sqrt(dt) * 0.5, base_volume)
    df.index = _index(n, start, freq)
    return df


def regime_switching(n: int = 1000, seed: Optional[int] = 0, start: str = "2020-01-01", s0: float = 100.0,
                     params: Sequence[Tuple[float, float]] = ((0.25, 0.10), (-0.10, 0.35)),
                     stay_prob: float = 0.98, base_volume: float = 1e6) -> Tuple[pd.DataFrame, np.ndarray]:
    """Two-regime (bull/calm vs bear/volatile) Markov chain returns. Returns (df, regime_labels)."""
    rng = np.random.default_rng(seed)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if rng.random() < stay_prob else 1 - state[t - 1]
    dt = 1 / 252
    mus = np.array([params[s][0] for s in state])
    sig = np.array([params[s][1] for s in state])
    r = (mus - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * rng.standard_normal(n)
    close = s0 * np.exp(np.cumsum(r))
    df = _ohlc_from_close(close, rng, 0.006, base_volume)
    df.index = _index(n, start, "B")
    return df, state


def cointegrated_pair(n: int = 1000, beta: float = 1.5, theta: float = 0.05, spread_sigma: float = 0.8,
                      seed: Optional[int] = 0, start: str = "2020-01-01") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Two series sharing a stochastic trend with a mean-reverting (OU) spread: y = beta*x + spread."""
    rng = np.random.default_rng(seed)
    x = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = spread[t - 1] * (1 - theta) + spread_sigma * rng.standard_normal()
    y = beta * x + spread + 20
    y = np.clip(y, 1.0, None)
    dfx = _ohlc_from_close(x, rng, 0.004, 1e6)
    dfy = _ohlc_from_close(y, rng, 0.004, 1e6)
    idx = _index(n, start, "B")
    dfx.index = dfy.index = idx
    return dfy, dfx


def universe(symbols: Sequence[str], n: int = 756, seed: int = 0, corr: float = 0.4, sigma: float = 0.22,
             mu_range: Tuple[float, float] = (0.0, 0.15), start: str = "2020-01-01") -> Dict[str, pd.DataFrame]:
    """Correlated multi-asset universe driven by one market factor plus idiosyncratic noise."""
    if not 0 <= corr < 1:
        raise ValidationError("corr must be in [0, 1)")
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    market = rng.standard_normal(n)
    out: Dict[str, pd.DataFrame] = {}
    for i, s in enumerate(symbols):
        mu = rng.uniform(*mu_range)
        z = np.sqrt(corr) * market + np.sqrt(1 - corr) * rng.standard_normal(n)
        r = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
        close = rng.uniform(30, 300) * np.exp(np.cumsum(r))
        df = _ohlc_from_close(close, rng, 0.006, 1e6 * rng.uniform(0.5, 3))
        df.index = _index(n, start, "B")
        out[s] = df
    return out


# ---- named scenarios for the sample-data generator -----------------------------------------------------------------------
SCENARIOS: Dict[str, str] = {
    "random_walk": "Random walk with a modest upward drift (about 8% a year, 22% volatility).",
    "bull_trend": "Steady bull market: strong drift (about 30% a year), lower volatility.",
    "bear_trend": "Bear market: falling prices (about -25% a year) with higher volatility.",
    "sideways": "Range-bound: prices mean-revert around a level, so trend-followers struggle and reversion works.",
    "high_volatility": "Little drift, very high volatility (about 60% a year), like a speculative stock.",
    "regime_switching": "Calm up-trends that switch, at random, to turbulent sell-offs and back (shared by all symbols).",
    "crash_and_recovery": "Normal market, a sudden crash of about 30% over two weeks, a volatile bottom, then recovery.",
    "pairs": "Every symbol after the first is cointegrated with the first: spreads mean-revert (for pairs trading).",
}

BARS_PER_DAY = {"1Min": 390, "5Min": 78, "15Min": 26, "1Hour": 7, "1Day": 1}


def _symbol_seed(seed: int, symbol: str) -> List[int]:
    """Seed for one symbol's own randomness: depends on the symbol, not on which other symbols are generated with it."""
    return [int(seed) & 0xFFFFFFFF, zlib.crc32(symbol.encode())]


def session_index(start: Any, n: int, timeframe: str = "1Day") -> pd.DatetimeIndex:
    """``n`` bar timestamps from ``start``: business days for daily bars, 09:30-16:00 sessions for intraday bars."""
    if timeframe not in BARS_PER_DAY:
        raise ValidationError(f"unknown timeframe '{timeframe}'", details={"valid": list(BARS_PER_DAY)})
    if timeframe == "1Day":
        return pd.bdate_range(start=pd.Timestamp(start).normalize(), periods=n)
    per_day = BARS_PER_DAY[timeframe]
    minutes = 390 // per_day
    days = pd.bdate_range(start=pd.Timestamp(start).normalize(), periods=n // per_day + 2)
    stamps = [d + pd.Timedelta(hours=9, minutes=30 + minutes * k) for d in days for k in range(per_day)]
    return pd.DatetimeIndex(stamps[:n])


def scenario(name: str, symbols: Sequence[str], n: int = 756, seed: int = 0, corr: float = 0.4, start: Any = "2020-01-01",
             timeframe: str = "1Day") -> Dict[str, pd.DataFrame]:
    """OHLCV for each symbol under a named market scenario (see :data:`SCENARIOS`). Seeded and reproducible.

    All symbols share one market factor (correlation ``corr``) and, for regime and crash scenarios, the same market
    events. Each symbol's own noise and starting price depend only on the seed and the symbol, so ``AAA`` is the same
    series whether it is generated alone or with others. This is sample data: it illustrates mechanics, never a market."""
    if name not in SCENARIOS:
        raise ValidationError(f"unknown scenario '{name}'", details={"valid": sorted(SCENARIOS)})
    if not symbols:
        raise ValidationError("give at least one symbol")
    if n < 30 or n > 200_000:
        raise ValidationError("n must be between 30 and 200,000 bars")
    if not 0 <= corr < 1:
        raise ValidationError("corr must be in [0, 1)")
    per_year = 252.0 * BARS_PER_DAY.get(timeframe, 1)
    dt = 1.0 / per_year
    market_rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 1])
    market = market_rng.standard_normal(n)
    market -= market.mean()                                             # the noise adds no drift of its own: a scenario
    regime = np.zeros(n, dtype=int)
    if name == "regime_switching":
        stay = 1.0 - 1.0 / (60 * BARS_PER_DAY.get(timeframe, 1))      # regimes last about 60 trading days
        for t in range(1, n):
            regime[t] = regime[t - 1] if market_rng.random() < stay else 1 - regime[t - 1]
    crash_at = int(n * 0.6)
    crash_len = max(2, min(10 * BARS_PER_DAY.get(timeframe, 1), n // 10))
    idx = session_index(start, n, timeframe)
    out: Dict[str, pd.DataFrame] = {}
    first_close: Optional[np.ndarray] = None
    for s in symbols:
        rng = np.random.default_rng(_symbol_seed(seed, s))
        tilt = rng.uniform(-0.04, 0.04)                                 # a little dispersion between symbols
        own = rng.standard_normal(n)
        z = np.sqrt(corr) * market + np.sqrt(1.0 - corr) * (own - own.mean())     # ... trends the way its name says
        s0 = float(rng.uniform(30, 300))
        if name == "sideways":
            theta, sig = np.log(2) / (20 * BARS_PER_DAY.get(timeframe, 1)), 0.20      # halfway back in about 20 trading days
            x = np.zeros(n)
            for t in range(1, n):
                x[t] = x[t - 1] * (1 - theta) + sig * np.sqrt(dt) * z[t]
            close = s0 * np.exp(x)
        elif name == "pairs" and first_close is not None:
            beta = float(rng.uniform(0.6, 1.6))
            spread = np.zeros(n)
            for t in range(1, n):
                spread[t] = spread[t - 1] * 0.95 + 0.005 * first_close[t] * z[t]      # stationary, about 1.6% of the price
            close = np.clip(beta * first_close + spread + 0.2 * s0, 1.0, None)
        else:
            mu, sig = {"random_walk": (0.08, 0.22), "bull_trend": (0.30, 0.18), "bear_trend": (-0.25, 0.28),
                       "high_volatility": (0.20, 0.60), "regime_switching": (0.0, 0.0), "crash_and_recovery": (0.10, 0.22),
                       "pairs": (0.06, 0.20)}[name]
            mus, sigs = np.full(n, mu + tilt), np.full(n, sig)
            if name == "regime_switching":
                mus = np.where(regime == 0, 0.20, -0.20) + tilt
                sigs = np.where(regime == 0, 0.12, 0.40)
            r = (mus - 0.5 * sigs ** 2) * dt + sigs * np.sqrt(dt) * z
            if name == "crash_and_recovery":
                beta = float(rng.uniform(0.7, 1.3))
                end = min(n, crash_at + crash_len)
                r[crash_at:end] += beta * np.log(0.70) / (end - crash_at)              # about -30% for a market-like stock
                after = slice(end, min(n, end + 3 * crash_len))
                r[after] = r[after] * 3.0 + 0.30 * dt * 3                          # volatile bottom with a rebound bias
            close = s0 * np.exp(np.cumsum(r))
        if first_close is None:
            first_close = close
        df = _ohlc_from_close(close, rng, float(np.std(np.diff(np.log(close)))) * 0.5 + 1e-4, 1e6 * rng.uniform(0.5, 3))
        df.index = idx
        out[s] = df
    return out
