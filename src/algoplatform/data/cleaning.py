"""Validation and cleaning of OHLCV data."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..core.exceptions import DataError, ValidationError

OHLCV = ["open", "high", "low", "close", "volume"]


def validate_ohlcv(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Strict check used at load time. Raises DataError with a precise reason."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise DataError("empty or non-DataFrame data", details={"symbol": symbol})
    missing = [c for c in ["open", "high", "low", "close"] if c not in df.columns]
    if missing:
        raise DataError(f"missing columns {missing}", details={"symbol": symbol})
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataError("index must be a DatetimeIndex", details={"symbol": symbol})
    if not df.index.is_monotonic_increasing:
        raise DataError("index not sorted ascending", details={"symbol": symbol})
    if df.index.has_duplicates:
        raise DataError("duplicate timestamps", details={"symbol": symbol})
    px = df[["open", "high", "low", "close"]]
    if px.isna().any().any():
        raise DataError("NaN prices", details={"symbol": symbol})
    if (px <= 0).any().any():
        raise DataError("non-positive prices", details={"symbol": symbol})
    bad = (df["high"] < df[["open", "close", "low"]].max(axis=1) - 1e-9) | \
          (df["low"] > df[["open", "close", "high"]].min(axis=1) + 1e-9)
    if bad.any():
        raise DataError(f"{int(bad.sum())} bars with inconsistent OHLC", details={"symbol": symbol})
    if "volume" not in df.columns:
        df = df.assign(volume=0.0)
    return df


def clean_ohlcv(df: pd.DataFrame, max_gap_fill: int = 3, outlier_mads: float = 12.0) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Best-effort repair; returns (clean_df, report of what was changed).

    Steps: sort, drop duplicate stamps, drop non-positive/NaN bars, repair high/low envelope,
    drop return outliers (bad ticks) beyond ``outlier_mads`` robust deviations, and report
    remaining gaps. Never invents prices beyond forward-filling ``max_gap_fill`` bars when the
    index has been reindexed by the caller."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValidationError("index must be a DatetimeIndex")
    report = {"duplicates": 0, "invalid": 0, "repaired_envelope": 0, "outliers": 0}
    out = df.sort_index()
    report["duplicates"] = int(out.index.duplicated().sum())
    out = out[~out.index.duplicated(keep="last")]
    px = ["open", "high", "low", "close"]
    invalid = out[px].isna().any(axis=1) | (out[px] <= 0).any(axis=1)
    report["invalid"] = int(invalid.sum())
    out = out[~invalid].copy()
    hi = out[["open", "high", "low", "close"]].max(axis=1)
    lo = out[["open", "high", "low", "close"]].min(axis=1)
    env = (out["high"] < hi) | (out["low"] > lo)
    report["repaired_envelope"] = int(env.sum())
    out["high"], out["low"] = hi, lo
    if len(out) > 10:
        r = np.log(out["close"]).diff()
        med = r.median()
        # robust scale, floored by the typical absolute move so near-constant series don't give mad ~ 0
        mad = max((r - med).abs().median() * 1.4826, 0.25 * r.abs().median())
        if mad > 0:
            bad = ((r - med).abs() > outlier_mads * mad) & ((r.shift(-1) + r).abs() < 2 * outlier_mads * mad)
            # an isolated spike reverses next bar (r_next ~ -r); a real jump does not
            report["outliers"] = int(bad.sum())
            out = out[~bad]
    if "volume" not in out.columns:
        out["volume"] = 0.0
    out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)
    return out, report


def adjust_for_split(df: pd.DataFrame, split_date, ratio: float) -> pd.DataFrame:
    """Back-adjust prices before ``split_date`` for an N-for-1 split (ratio=N)."""
    if ratio <= 0:
        raise ValidationError("split ratio must be positive")
    out = df.copy()
    before = out.index < pd.Timestamp(split_date)
    out.loc[before, ["open", "high", "low", "close"]] /= ratio
    if "volume" in out:
        out.loc[before, "volume"] *= ratio
    return out


def find_gaps(df: pd.DataFrame, max_gap: pd.Timedelta) -> pd.DataFrame:
    """Rows where the time since the previous bar exceeds ``max_gap``."""
    delta = df.index.to_series().diff()
    return delta[delta > max_gap].to_frame("gap")
