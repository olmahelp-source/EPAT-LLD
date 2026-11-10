"""Clocks and trading-calendar helpers.

Components take a ``Clock`` rather than calling ``datetime.now`` so a backtest can
drive time deterministically (``SimClock``) while live code uses ``SystemClock``.
"""
from __future__ import annotations

import threading
from datetime import datetime, time, timedelta, timezone
from typing import Iterable, Optional, Set


class Clock:
    def now(self) -> datetime:
        raise NotImplementedError


class SystemClock(Clock):
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class SimClock(Clock):
    """Deterministic clock driven by the data. Naive or aware datetimes both work as long as
    the same kind is used throughout one run."""

    def __init__(self, start: Optional[datetime] = None) -> None:
        self._now = start
        self._lock = threading.Lock()

    def now(self) -> datetime:
        return self._now if self._now is not None else datetime(2000, 1, 1)

    def set(self, ts: datetime) -> None:
        with self._lock:
            if self._now is not None and ts < self._now:
                raise ValueError("SimClock cannot go backwards")
            self._now = ts


class TradingCalendar:
    """Weekday calendar with explicit holidays and a daily session window (UTC)."""

    def __init__(self, open_time: time = time(14, 30), close_time: time = time(21, 0),
                 holidays: Iterable[str] = (), weekend: Set[int] = frozenset({5, 6})) -> None:
        self.open_time, self.close_time = open_time, close_time
        self.holidays = {datetime.fromisoformat(h).date() for h in holidays}
        self.weekend = set(weekend)

    def is_trading_day(self, d) -> bool:
        d = d.date() if isinstance(d, datetime) else d
        return d.weekday() not in self.weekend and d not in self.holidays

    def is_open(self, ts: datetime) -> bool:
        return self.is_trading_day(ts) and self.open_time <= ts.timetz().replace(tzinfo=None) < self.close_time

    def next_trading_day(self, d):
        d = d.date() if isinstance(d, datetime) else d
        d += timedelta(days=1)
        while not self.is_trading_day(d):
            d += timedelta(days=1)
        return d


def periods_per_year(freq: str) -> float:
    """Annualisation factor for a pandas-style frequency string."""
    table = {"D": 252, "B": 252, "W": 52, "M": 12, "ME": 12, "Q": 4, "Y": 1, "A": 1,
             "H": 252 * 6.5, "h": 252 * 6.5, "30min": 252 * 13, "15min": 252 * 26,
             "5min": 252 * 78, "1min": 252 * 390, "min": 252 * 390}
    try:
        return float(table[freq])
    except KeyError:
        raise ValueError(f"unknown frequency {freq!r}; known: {sorted(table)}") from None
