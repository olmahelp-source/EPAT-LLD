"""Synthetic market data for tests, demos and Monte Carlo (seeded and reproducible)."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.exceptions import ValidationError


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
