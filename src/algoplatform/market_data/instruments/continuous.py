"""Continuous futures: stitch dated contracts into one series a strategy can be tested on.

Each contract dies at expiry and the next one usually trades at a different price, so naive splicing produces a fake
jump at every roll. Back-adjustment removes it: ``difference`` shifts older history by the price gap (keeps point
moves, can go negative far back), ``ratio`` scales it (keeps percentage moves, always positive).
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

import pandas as pd

from ...foundation.core.exceptions import InstrumentError

_PRICE_COLS = ("open", "high", "low", "close")


def roll_schedule(contracts: Mapping[Any, pd.DataFrame], roll_days: int = 5) -> List[Dict[str, Any]]:
    """Which contract is held when. ``contracts`` maps expiry date -> OHLCV frame. A contract is held until
    ``roll_days`` calendar days before its expiry, then the next one takes over."""
    if not contracts:
        raise InstrumentError("no contracts given")
    if roll_days < 0:
        raise InstrumentError("roll_days must be >= 0")
    expiries = sorted(pd.Timestamp(e) for e in contracts)
    out, start = [], None
    for i, exp in enumerate(expiries):
        last = i == len(expiries) - 1
        end = exp if last else exp - pd.Timedelta(days=roll_days)
        out.append({"expiry": exp, "start": start, "end": end})
        start = end
    return out


def continuous_series(contracts: Mapping[Any, pd.DataFrame], roll_days: int = 5, method: str = "difference") -> pd.DataFrame:
    """Back-adjusted continuous OHLCV with a ``contract`` column (the expiry held on each bar) and ``adjustment``."""
    if method not in ("difference", "ratio", "none"):
        raise InstrumentError(f"unknown roll method '{method}'", details={"valid": ["difference", "ratio", "none"]})
    frames = {pd.Timestamp(e): df.sort_index() for e, df in contracts.items()}
    pieces: List[pd.DataFrame] = []
    for seg in roll_schedule(contracts, roll_days):
        df = frames[seg["expiry"]]
        part = df[(df.index > seg["start"]) if seg["start"] is not None else slice(None)]
        part = part[part.index <= seg["end"]]
        if len(part):
            pieces.append(part.assign(contract=seg["expiry"]))
    if not pieces:
        raise InstrumentError("no data falls inside the roll schedule")
    adj = 0.0 if method == "difference" else 1.0
    out: List[pd.DataFrame] = []
    for i in range(len(pieces) - 1, -1, -1):                        # walk backwards so the newest contract is unadjusted
        part = pieces[i].copy()
        if method == "difference":
            part[[c for c in _PRICE_COLS if c in part]] += adj
        elif method == "ratio":
            part[[c for c in _PRICE_COLS if c in part]] *= adj
        part["adjustment"] = adj
        out.append(part)
        if i > 0 and method != "none":
            roll_day = pieces[i - 1].index[-1]                      # last bar on the old contract
            new = frames[pieces[i]["contract"].iloc[0]]
            if roll_day not in new.index:
                raise InstrumentError(f"contract {pieces[i]['contract'].iloc[0].date()} has no bar on roll day {roll_day.date()}",
                                      details={"hint": "both contracts must trade on the roll day to measure the gap"})
            old_close, new_close = float(pieces[i - 1]["close"].iloc[-1]), float(new.loc[roll_day, "close"])
            if method == "difference":
                adj += new_close - old_close
            else:
                adj *= new_close / old_close
    return pd.concat(out[::-1])
