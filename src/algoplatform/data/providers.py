"""Historical market data from a provider: sample (generated), Alpaca, or Interactive Brokers.

Every provider returns the same thing: one OHLCV frame per symbol with a DatetimeIndex (UTC, timezone removed; daily
bars are stamped at midnight of their trading date), cleaned with :func:`clean_ohlcv` so it loads into
:class:`MarketData` like any other data. ``fetch_many`` attempts every symbol: one bad symbol is reported and the
others still load. Bad credentials stop the batch, because every further request would fail the same way.

Honest limits:

* :class:`AlpacaDataProvider` follows Alpaca's public market-data API (``/v2/stocks/{symbol}/bars`` and the crypto
  bars endpoint, page tokens, 429 rate limits). It is tested against a scripted fake transport, not the live service.
* :class:`IBDataProvider` needs Trader Workstation or IB Gateway running and the optional ``ib_async`` (or older
  ``ib_insync``) package. Its chunking, pacing and parsing are tested against a fake client; the thin
  :class:`IBClient` that talks to TWS has not been run against a real gateway.
* :class:`SampleDataProvider` generates data: it says nothing about any real market.
"""
from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.exceptions import (ConfigError, DataProviderError, MissingDataError, PlatformError, ProviderAuthError,
                               ProviderUnavailableError, ValidationError)
from ..core.logging import get_logger
from . import synthetic
from .cleaning import clean_ohlcv, validate_ohlcv

log = get_logger(__name__)

TIMEFRAMES: Tuple[str, ...] = ("1Min", "5Min", "15Min", "1Hour", "1Day")
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9./\-]{0,14}$")

Transport = Callable[[str, Dict[str, str]], Tuple[int, Any]]          # (url, headers) -> (status, parsed JSON body)


def check_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if not _SYMBOL.match(s):
        raise ValidationError(f"'{symbol}' is not a valid symbol (letters, digits, '.', '/', '-'; up to 15 characters)")
    return s


def check_timeframe(timeframe: str) -> str:
    if timeframe not in TIMEFRAMES:
        raise ValidationError(f"unknown timeframe '{timeframe}'", details={"valid": list(TIMEFRAMES)})
    return timeframe


def date_range(start: Any = None, end: Any = None, default_years: float = 3.0) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Parse ``start`` / ``end`` (anything pandas understands); the end defaults to today, the start to ``default_years``
    before it. Both are returned timezone-naive, in UTC."""
    def parse(x: Any) -> pd.Timestamp:
        ts = pd.Timestamp(x)
        return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts
    try:
        e = parse(end) if end not in (None, "") else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        s = parse(start) if start not in (None, "") else e - pd.Timedelta(days=int(365.25 * default_years))
    except (ValueError, TypeError) as exc:
        raise ValidationError(f"cannot read the date: {exc}") from None
    if not s < e:
        raise ValidationError("the start date must be before the end date", details={"start": str(s), "end": str(e)})
    return s, e


@dataclass
class FetchResult:
    """What a batch fetch produced: frames that loaded, and why each other symbol did not."""
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    cleaning: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {"loaded": {s: {"rows": len(f), "start": str(f.index[0]), "end": str(f.index[-1])} for s, f in self.frames.items()},
                "errors": dict(self.errors), "cleaning": dict(self.cleaning)}


class DataProvider:
    """Base class. Subclasses implement :meth:`_fetch` returning raw OHLCV; the base class cleans and checks it."""

    name = "provider"
    label = "Provider"
    external = True                       # False only for generated data

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "label": self.label, "external": self.external}

    def _fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> pd.DataFrame:
        raise NotImplementedError

    def fetch(self, symbol: str, start: Any = None, end: Any = None, timeframe: str = "1Day") -> pd.DataFrame:
        return self.fetch_with_report(symbol, start, end, timeframe)[0]

    def fetch_with_report(self, symbol: str, start: Any = None, end: Any = None, timeframe: str = "1Day") -> Tuple[pd.DataFrame, Dict[str, int]]:
        sym, tf = check_symbol(symbol), check_timeframe(timeframe)
        s, e = date_range(start, end)
        raw = self._fetch(sym, s, e, tf)
        if raw is None or raw.empty:
            raise MissingDataError(f"{self.label} returned no bars for {sym} between {s.date()} and {e.date()}",
                                   details={"symbol": sym, "timeframe": tf})
        clean, report = clean_ohlcv(raw[[c for c in ("open", "high", "low", "close", "volume") if c in raw.columns]].astype(float))
        return validate_ohlcv(clean, sym), report

    def fetch_many(self, symbols: Sequence[str], start: Any = None, end: Any = None, timeframe: str = "1Day") -> FetchResult:
        out = FetchResult()
        for i, symbol in enumerate(symbols):
            try:
                df, report = self.fetch_with_report(symbol, start, end, timeframe)
                out.frames[check_symbol(symbol)] = df
                out.cleaning[check_symbol(symbol)] = report
            except ProviderAuthError as exc:
                out.errors[str(symbol)] = exc.message
                for rest in symbols[i + 1:]:
                    out.errors[str(rest)] = "not attempted: the provider rejected the credentials"
                break
            except PlatformError as exc:                 # one symbol's problem must not stop the others
                out.errors[str(symbol)] = exc.message
        return out


# ---- generated data -----------------------------------------------------------------------------------------------------
class SampleDataProvider(DataProvider):
    """Generated OHLCV for any symbol and date range, under a named scenario (see ``synthetic.SCENARIOS``).

    The series for a symbol depends only on (scenario, seed, symbol, timeframe, start), so the same request always gives
    the same data. ``fetch_many`` shares one market factor across the batch."""

    name, label, external = "sample", "Sample data", False

    def __init__(self, scenario: str = "random_walk", seed: int = 7, corr: float = 0.4) -> None:
        if scenario not in synthetic.SCENARIOS:
            raise ValidationError(f"unknown scenario '{scenario}'", details={"valid": sorted(synthetic.SCENARIOS)})
        self.scenario, self.seed, self.corr = scenario, int(seed), float(corr)

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "scenario": self.scenario, "seed": self.seed}

    @staticmethod
    def bars_between(start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> int:
        days = len(pd.bdate_range(start.normalize(), end.normalize()))
        return max(30, min(200_000, days * synthetic.BARS_PER_DAY[timeframe]))

    def _fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> pd.DataFrame:
        n = self.bars_between(start, end, timeframe)
        return synthetic.scenario(self.scenario, [symbol], n=n, seed=self.seed, corr=self.corr, start=start, timeframe=timeframe)[symbol]

    def generate(self, symbols: Sequence[str], bars: int = 756, start: Any = "2020-01-01", timeframe: str = "1Day") -> Dict[str, pd.DataFrame]:
        """A whole universe at once (shared market factor, and the first symbol anchors the ``pairs`` scenario)."""
        return synthetic.scenario(self.scenario, [check_symbol(s) for s in symbols], n=int(bars), seed=self.seed, corr=self.corr,
                                  start=start, timeframe=check_timeframe(timeframe))


# ---- HTTPS transport ------------------------------------------------------------------------------------------------------
def urllib_transport(timeout: float = 15.0) -> Transport:
    """Default transport: stdlib urllib, HTTPS only, TLS verification on, JSON body parsed."""
    def get(url: str, headers: Dict[str, str]) -> Tuple[int, Any]:
        if not url.startswith("https://"):
            raise ConfigError("refusing a non-HTTPS data URL")
        req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:                      # noqa: S310 (https enforced)
                return r.status, _json(r.read())
        except urllib.error.HTTPError as exc:
            return exc.code, _json(exc.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ProviderUnavailableError(f"network error: {type(exc).__name__}") from None   # no URL or key in the message
    return get


def _json(raw: bytes) -> Any:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        return {"message": raw[:200].decode("utf-8", "replace")}


# ---- Alpaca ---------------------------------------------------------------------------------------------------------------
class AlpacaDataProvider(DataProvider):
    """Historical bars from Alpaca's market-data API, with the same API keys as an Alpaca trading account.

    Stocks use ``/v2/stocks/{symbol}/bars`` (``feed`` "iex" works on free plans; "sip" needs a subscription);
    symbols with a slash (``BTC/USD``) use the crypto endpoint. Prices are split- and dividend-adjusted by default
    (``adjustment="all"``). Pages are followed until the range is complete; 429 and 5xx answers are retried with
    exponential back-off."""

    name, label = "alpaca", "Alpaca"
    BASE_URL = "https://data.alpaca.markets"
    FEEDS = ("iex", "sip", "delayed_sip", "otc")
    ADJUSTMENTS = ("raw", "split", "dividend", "all")

    def __init__(self, key_id: str, secret_key: str, feed: str = "iex", adjustment: str = "all", base_url: str = BASE_URL,
                 transport: Optional[Transport] = None, max_pages: int = 100, retries: int = 3,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not key_id or not secret_key:
            raise ConfigError("Alpaca needs an API key ID and a secret key")
        if feed not in self.FEEDS or adjustment not in self.ADJUSTMENTS:
            raise ValidationError(f"feed must be one of {self.FEEDS} and adjustment one of {self.ADJUSTMENTS}")
        if not base_url.startswith("https://"):
            raise ConfigError("the Alpaca data URL must use HTTPS")
        self._key_id, self._secret = str(key_id), str(secret_key)
        self.feed, self.adjustment, self.base_url = feed, adjustment, base_url.rstrip("/")
        self.transport = transport or urllib_transport()
        self.max_pages, self.retries, self.sleep = max_pages, retries, sleep
        self.requests = 0

    def __repr__(self) -> str:                    # never show the keys
        return f"AlpacaDataProvider(feed={self.feed!r}, key_id='{self._key_id[:4]}…')"

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "feed": self.feed, "adjustment": self.adjustment, "key_id": self._key_id[:4] + "…"}

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}?{urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})}"
        headers = {"APCA-API-KEY-ID": self._key_id, "APCA-API-SECRET-KEY": self._secret}
        for attempt in range(self.retries + 1):
            self.requests += 1
            status, body = self.transport(url, headers)
            message = str(body.get("message", "")) if isinstance(body, dict) else ""
            if status == 200:
                return body if isinstance(body, dict) else {}
            if status in (401, 403):
                raise ProviderAuthError(f"Alpaca rejected the API keys (HTTP {status}){': ' + message if message else ''}",
                                        details={"status": status})
            if status == 429 or status >= 500:
                if attempt < self.retries:
                    self.sleep(min(30.0, 2.0 ** attempt))
                    continue
                raise ProviderUnavailableError(f"Alpaca is unavailable or rate-limiting (HTTP {status}) after {self.retries + 1} tries",
                                               details={"status": status})
            raise DataProviderError(f"Alpaca refused the request (HTTP {status}){': ' + message if message else ''}",
                                    details={"status": status})
        raise ProviderUnavailableError("Alpaca did not answer")                       # pragma: no cover - loop always returns

    @staticmethod
    def _iso(ts: pd.Timestamp) -> str:
        return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> pd.DataFrame:
        crypto = "/" in symbol
        path = "/v1beta3/crypto/us/bars" if crypto else f"/v2/stocks/{urllib.parse.quote(symbol, safe='')}/bars"
        params: Dict[str, Any] = {"timeframe": timeframe, "start": self._iso(start), "end": self._iso(end), "limit": 10000,
                                  "symbols": symbol if crypto else None, "feed": None if crypto else self.feed,
                                  "adjustment": None if crypto else self.adjustment, "sort": "asc"}
        rows: List[Dict[str, Any]] = []
        for _ in range(self.max_pages):
            body = self._get(path, params)
            bars = body.get("bars") or []
            if crypto and isinstance(bars, dict):
                bars = bars.get(symbol) or []
            rows.extend(bars)
            token = body.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        else:
            raise DataProviderError(f"{symbol}: more than {self.max_pages} pages of bars; shorten the range or use larger bars")
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        missing = [k for k in ("t", "o", "h", "l", "c") if k not in df.columns]
        if missing:
            raise DataProviderError(f"Alpaca bars are missing fields {missing}", details={"symbol": symbol})
        idx = pd.to_datetime(df["t"], utc=True).dt.tz_localize(None)
        if timeframe == "1Day":
            # a stock's daily bar is stamped at midnight New York time (04:00 or 05:00 UTC), a crypto one at midnight UTC:
            # keep only the trading date
            local = pd.to_datetime(df["t"], utc=True).dt.tz_convert("UTC" if crypto else "America/New_York")
            idx = local.dt.tz_localize(None).dt.normalize()
        out = pd.DataFrame({"open": df["o"].astype(float).to_numpy(), "high": df["h"].astype(float).to_numpy(),
                            "low": df["l"].astype(float).to_numpy(), "close": df["c"].astype(float).to_numpy(),
                            "volume": df["v"].astype(float).to_numpy() if "v" in df.columns else 0.0}, index=pd.DatetimeIndex(idx.to_numpy()))
        return out.sort_index()


# ---- Interactive Brokers ------------------------------------------------------------------------------------------------
class IBClient:
    """The thin part that talks to TWS / IB Gateway through ``ib_async`` (or ``ib_insync``), imported only when used.

    Read-only connection. Not run against a real gateway in this repository's tests (see the module docstring)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 17, timeout: float = 30.0) -> None:
        self.host, self.port, self.client_id, self.timeout = host, int(port), int(client_id), float(timeout)
        self._ib: Any = None
        self._mod: Any = None

    def _connect(self) -> Any:
        if self._ib is not None and self._ib.isConnected():
            return self._ib
        try:
            import ib_async as mod                                     # type: ignore[import-not-found]
        except ImportError:
            try:
                import ib_insync as mod                                # type: ignore[import-not-found,no-redef]
            except ImportError:
                raise ConfigError("Interactive Brokers needs the optional package 'ib_async' (pip install ib_async) "
                                  "and Trader Workstation or IB Gateway running with the API enabled") from None
        self._mod = mod
        ib = mod.IB()
        try:
            ib.connect(self.host, self.port, clientId=self.client_id, timeout=self.timeout, readonly=True)
        except (ConnectionError, OSError, TimeoutError) as exc:
            raise ProviderUnavailableError(f"cannot reach TWS / IB Gateway at {self.host}:{self.port} ({type(exc).__name__}); "
                                           "is it running with the API enabled?") from None
        self._ib = ib
        return ib

    def historical(self, symbol: str, sec_type: str, exchange: str, currency: str, end: datetime, duration: str,
                   bar_size: str, what_to_show: str, use_rth: bool) -> List[Dict[str, Any]]:
        ib = self._connect()
        mod = self._mod
        contract = mod.Forex(symbol.replace(".", "")) if sec_type == "CASH" else mod.Stock(symbol, exchange, currency)
        bars = ib.reqHistoricalData(contract, endDateTime=end, durationStr=duration, barSizeSetting=bar_size,
                                    whatToShow=what_to_show, useRTH=use_rth, formatDate=2, timeout=self.timeout)
        return [{"date": b.date, "open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume} for b in bars or []]

    def close(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()
            self._ib = None


class IBDataProvider(DataProvider):
    """Historical bars from Interactive Brokers through TWS or IB Gateway.

    Long ranges are split into requests IB accepts for the bar size, walked backwards from the end date. IB allows
    about 60 historical requests per ten minutes, so a range needing more than ``max_requests`` is refused up front with
    a clear message instead of being throttled half-way. ``EUR.USD``-style symbols are forex (midpoint prices)."""

    name, label = "ib", "Interactive Brokers"
    BAR_SIZE = {"1Min": "1 min", "5Min": "5 mins", "15Min": "15 mins", "1Hour": "1 hour", "1Day": "1 day"}
    CHUNK_DAYS = {"1Min": 1, "5Min": 7, "15Min": 14, "1Hour": 30, "1Day": 365}

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 17, client: Any = None, exchange: str = "SMART",
                 currency: str = "USD", use_rth: bool = True, max_requests: int = 50, pause: float = 0.0,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not 0 < int(port) < 65536:
            raise ValidationError("port must be 1-65535 (TWS paper 7497, live 7496; IB Gateway paper 4002, live 4001)")
        self.host, self.port, self.client_id = str(host), int(port), int(client_id)
        self.client = client if client is not None else IBClient(self.host, self.port, self.client_id)
        self.exchange, self.currency, self.use_rth = exchange, currency, use_rth
        self.max_requests, self.pause, self.sleep = max_requests, pause, sleep
        self.requests = 0

    def describe(self) -> Dict[str, Any]:
        return {**super().describe(), "host": self.host, "port": self.port, "client_id": self.client_id}

    @staticmethod
    def _duration(days: int) -> str:
        return f"{max(1, math.ceil(days / 365))} Y" if days > 365 else f"{max(1, days)} D"

    def plan(self, start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> List[Tuple[pd.Timestamp, int]]:
        """The (end, days) windows needed to cover the range, newest first."""
        step = self.CHUNK_DAYS[timeframe]
        windows: List[Tuple[pd.Timestamp, int]] = []
        cursor = end
        while cursor > start:
            days = min(step, max(1, math.ceil((cursor - start) / pd.Timedelta(days=1))))
            windows.append((cursor, days))
            cursor = cursor - pd.Timedelta(days=days)
        return windows

    def _fetch(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> pd.DataFrame:
        windows = self.plan(start, end, timeframe)
        if len(windows) > self.max_requests:
            raise DataProviderError(f"{symbol}: this range needs {len(windows)} requests to IB, more than the {self.max_requests} its pacing "
                                    "rules allow in one go; shorten the range or use larger bars", details={"requests": len(windows)})
        forex = bool(re.fullmatch(r"[A-Z]{3}\.[A-Z]{3}", symbol))
        rows: List[Dict[str, Any]] = []
        for i, (w_end, days) in enumerate(windows):
            if i and self.pause:
                self.sleep(self.pause)
            self.requests += 1
            rows.extend(self.client.historical(symbol, "CASH" if forex else "STK", "IDEALPRO" if forex else self.exchange,
                                               symbol.split(".")[1] if forex else self.currency,
                                               w_end.to_pydatetime().replace(tzinfo=timezone.utc), self._duration(days),
                                               self.BAR_SIZE[timeframe], "MIDPOINT" if forex else "TRADES", self.use_rth))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        stamps = pd.to_datetime(df["date"].astype(str) if timeframe == "1Day" else df["date"], utc=timeframe != "1Day")
        idx = stamps.dt.tz_localize(None) if timeframe != "1Day" else stamps.dt.normalize()
        out = pd.DataFrame({k: df[k].astype(float).to_numpy() for k in ("open", "high", "low", "close")}, index=pd.DatetimeIndex(idx.to_numpy()))
        out["volume"] = np.clip(df["volume"].astype(float).fillna(0.0).to_numpy(), 0.0, None)          # IB reports -1 for "no volume"
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out.loc[start:end]


__all__ = ["TIMEFRAMES", "FetchResult", "DataProvider", "SampleDataProvider", "AlpacaDataProvider", "IBDataProvider", "IBClient",
           "urllib_transport", "check_symbol", "check_timeframe", "date_range"]
