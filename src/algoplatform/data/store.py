"""In-memory market-data store with CSV I/O and a bar iterator for the engine."""
from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

from ..core.exceptions import DataError, MissingDataError
from ..core.models import Bar, Instrument
from .cleaning import validate_ohlcv


class MarketData:
    """Holds one validated OHLCV frame per symbol.

    Frames are converted once to numpy arrays so the engine can slice history in O(1)
    without pandas overhead on every bar.
    """

    def __init__(self, instruments: Optional[Dict[str, Instrument]] = None) -> None:
        self._frames: Dict[str, pd.DataFrame] = {}
        self._arrays: Dict[str, Dict[str, np.ndarray]] = {}
        self.instruments: Dict[str, Instrument] = dict(instruments or {})
        self._cache: Optional[List[Dict[str, Bar]]] = None

    # ---- loading ------------------------------------------------------------------------
    def add(self, symbol: str, df: pd.DataFrame, instrument: Optional[Instrument] = None) -> "MarketData":
        df = validate_ohlcv(df, symbol)
        if "volume" not in df:
            df = df.assign(volume=0.0)
        self._frames[symbol] = df[["open", "high", "low", "close", "volume"]].astype(float)
        self._arrays[symbol] = {c: self._frames[symbol][c].to_numpy() for c in self._frames[symbol].columns}
        self.instruments.setdefault(symbol, instrument or Instrument(symbol))
        self._cache = None
        return self

    def add_csv(self, symbol: str, path: str, date_col: str = "date", **read_kwargs) -> "MarketData":
        if not os.path.exists(path):
            raise MissingDataError(f"file not found: {path}")
        try:
            df = pd.read_csv(path, parse_dates=[date_col], index_col=date_col, **read_kwargs)
        except (ValueError, pd.errors.ParserError) as exc:
            raise DataError(f"cannot parse {path}: {exc}") from exc
        df.columns = [c.lower() for c in df.columns]
        df.index.name = None
        return self.add(symbol, df.sort_index())

    def to_csv(self, symbol: str, path: str) -> None:
        self.frame(symbol).rename_axis("date").to_csv(path)

    # ---- access ---------------------------------------------------------------------------
    @property
    def symbols(self) -> List[str]:
        return list(self._frames)

    def frame(self, symbol: str) -> pd.DataFrame:
        try:
            return self._frames[symbol]
        except KeyError:
            raise MissingDataError(f"no data for {symbol}") from None

    def arrays(self, symbol: str) -> Dict[str, np.ndarray]:
        self.frame(symbol)
        return self._arrays[symbol]

    def closes(self) -> pd.DataFrame:
        return pd.DataFrame({s: f["close"] for s, f in self._frames.items()})

    def returns(self) -> pd.DataFrame:
        return self.closes().pct_change().dropna(how="all")

    def timeline(self) -> pd.DatetimeIndex:
        idx = None
        for f in self._frames.values():
            idx = f.index if idx is None else idx.union(f.index)
        if idx is None:
            raise MissingDataError("market data is empty")
        return idx

    def slice(self, start=None, end=None) -> "MarketData":
        out = MarketData(self.instruments)
        for s, f in self._frames.items():
            sub = f.loc[start:end]
            if len(sub):
                out.add(s, sub)
        return out

    def bar_batches(self) -> List[Dict[str, Bar]]:
        """``[{symbol: Bar}, ...]`` in chronological order, built once and cached (the optimiser
        replays the same data hundreds of times)."""
        if self._cache is not None:
            return self._cache
        tl = self.timeline()
        pos = {s: f.index.get_indexer(tl) for s, f in self._frames.items()}
        batches: List[Dict[str, Bar]] = []
        for i, ts in enumerate(tl):
            row: Dict[str, Bar] = {}
            for s, a in self._arrays.items():
                j = pos[s][i]
                if j >= 0:
                    row[s] = Bar(ts, s, a["open"][j], a["high"][j], a["low"][j], a["close"][j], a["volume"][j])
            batches.append(row)
        self._cache = batches
        return batches

    def iter_bars(self) -> Iterator[Dict[str, Bar]]:
        yield from self.bar_batches()
