"""The history library: downloaded bars, kept per (symbol, timeframe), shared by every user of the platform.

Market history is not private (unlike uploaded files), so one download serves everyone. Re-downloading a symbol from the
same provider merges the new bars over the old (newer values win on the same timestamp); a download from a different
provider replaces the series, so one series never mixes two sources. Size is capped so the library always fits in the
state store.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.exceptions import MissingDataError, ValidationError
from .cleaning import validate_ohlcv
from .store import MarketData

COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass
class Series:
    symbol: str
    timeframe: str
    provider: str
    frame: pd.DataFrame
    fetched_at: str
    cleaning: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        f = self.frame
        return {"symbol": self.symbol, "timeframe": self.timeframe, "provider": self.provider, "rows": len(f),
                "start": f.index[0].isoformat(), "end": f.index[-1].isoformat(), "last_close": float(f["close"].iloc[-1]),
                "fetched_at": self.fetched_at, "cleaning": dict(self.cleaning)}


class HistoryLibrary:
    def __init__(self, max_rows_per_series: int = 60_000, max_total_rows: int = 400_000, max_series: int = 300) -> None:
        self.max_rows_per_series, self.max_total_rows, self.max_series = max_rows_per_series, max_total_rows, max_series
        self._series: Dict[Tuple[str, str], Series] = {}
        self._lock = threading.RLock()
        self.version = 0                      # changes on every write: lets the platform skip saving an unchanged library

    # ---- writing ----------------------------------------------------------------------------------------------------------
    def put(self, symbol: str, timeframe: str, frame: pd.DataFrame, provider: str, cleaning: Optional[Mapping[str, int]] = None) -> Series:
        frame = validate_ohlcv(frame, symbol)[list(COLUMNS)].astype(float)
        with self._lock:
            key = (symbol, timeframe)
            old = self._series.get(key)
            if old is not None and old.provider == provider:
                frame = pd.concat([old.frame, frame])
                frame = frame[~frame.index.duplicated(keep="last")].sort_index()
            if len(frame) > self.max_rows_per_series:
                frame = frame.iloc[-self.max_rows_per_series:]                    # keep the most recent bars
            if old is None and len(self._series) >= self.max_series:
                raise ValidationError(f"the history library holds {self.max_series} series already; remove some first")
            total = self.total_rows() - (len(old.frame) if old is not None else 0) + len(frame)
            if total > self.max_total_rows:
                raise ValidationError(f"the history library would hold {total:,} bars, over its limit of {self.max_total_rows:,}; "
                                      "remove some series or download a shorter range")
            series = Series(symbol, timeframe, provider, frame, pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), dict(cleaning or {}))
            self._series[key] = series
            self.version += 1
            return series

    def remove(self, symbol: str, timeframe: str = "1Day") -> bool:
        with self._lock:
            gone = self._series.pop((symbol, timeframe), None) is not None
            if gone:
                self.version += 1
            return gone

    # ---- reading ----------------------------------------------------------------------------------------------------------
    def __contains__(self, key: Any) -> bool:
        symbol, timeframe = key if isinstance(key, tuple) else (key, "1Day")
        return (symbol, timeframe) in self._series

    def get(self, symbol: str, timeframe: str = "1Day") -> pd.DataFrame:
        try:
            return self._series[(symbol, timeframe)].frame
        except KeyError:
            raise MissingDataError(f"no downloaded {timeframe} history for {symbol}: download it on the Price data page first",
                                   details={"symbol": symbol, "timeframe": timeframe}) from None

    def symbols(self, timeframe: str = "1Day") -> List[str]:
        return sorted(s for s, tf in self._series if tf == timeframe)

    def total_rows(self) -> int:
        return sum(len(x.frame) for x in self._series.values())

    def summary(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._series[k].summary() for k in sorted(self._series)]

    def market_data(self, symbols: Sequence[str], timeframe: str = "1Day", min_rows: int = 0) -> MarketData:
        """A :class:`MarketData` holding the requested series. Every missing or too-short series is named in one error."""
        problems = []
        for s in symbols:
            if (s, timeframe) not in self._series:
                problems.append(f"{s}: not downloaded")
            elif len(self._series[(s, timeframe)].frame) < min_rows:
                problems.append(f"{s}: only {len(self._series[(s, timeframe)].frame)} bars, need {min_rows}")
        if problems:
            raise MissingDataError("history not usable: " + "; ".join(problems), details={"problems": problems})
        md = MarketData()
        for s in symbols:
            md.add(s, self._series[(s, timeframe)].frame)
        return md

    # ---- persistence --------------------------------------------------------------------------------------------------------
    def export_state(self) -> Dict[str, Any]:
        with self._lock:
            return {"series": [{"symbol": x.symbol, "timeframe": x.timeframe, "provider": x.provider, "fetched_at": x.fetched_at,
                                "cleaning": x.cleaning, "index": [t.isoformat() for t in x.frame.index],
                                "values": np.round(x.frame[list(COLUMNS)].to_numpy(float), 6).tolist()}
                               for x in self._series.values()]}

    def import_state(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        """Load what :meth:`export_state` wrote. A series that no longer validates is skipped and reported."""
        loaded, skipped = 0, []
        for item in state.get("series", []) if isinstance(state, Mapping) else []:
            try:
                frame = pd.DataFrame(item["values"], columns=list(COLUMNS), index=pd.DatetimeIndex(pd.to_datetime(item["index"])))
                self.put(item["symbol"], item["timeframe"], frame, item["provider"], item.get("cleaning"))
                self._series[(item["symbol"], item["timeframe"])].fetched_at = item.get("fetched_at", "")
                loaded += 1
            except Exception as exc:  # noqa: BLE001 - one damaged series must not stop the rest loading
                skipped.append({"series": f"{item.get('symbol', '?')} {item.get('timeframe', '?')}", "error": repr(exc)[:200]})
        return {"loaded": loaded, "skipped": skipped}

    def series(self) -> Iterable[Series]:
        return list(self._series.values())


__all__ = ["HistoryLibrary", "Series", "COLUMNS"]
