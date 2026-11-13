"""Market and reference data: uploads, the sample-data generator, data providers, downloaded history, the
sample-data-only lock, and the instrument master.

Every route checks the caller's permission on the server (through the role desks or ``UserManager.authorize``); hiding a
page in the browser is a convenience, not the control."""
from __future__ import annotations

import io
from typing import Any, Dict

import numpy as np
import pandas as pd

from ...core.enums import Permission
from ...core.exceptions import DataError, ValidationError
from ...data import synthetic
from ...data.cleaning import validate_ohlcv
from ...users.manager import User, UserManager

from ..common import series_points


MAX_UPLOAD_ROWS, MAX_UPLOAD_SYMBOLS = 5000, 10

MAX_INSTRUMENT_CSV = 200_000          # bytes

MAX_PREVIEW_SYMBOLS = 6


def _stats(close) -> Dict[str, float]:
    c = np.asarray(close, float)
    r = np.diff(np.log(c))
    peak = np.maximum.accumulate(c)
    return {"total_return": float(c[-1] / c[0] - 1), "volatility": float(r.std() * np.sqrt(252)) if len(r) > 1 else 0.0,
            "max_drawdown": float(np.min(c / peak - 1.0)), "first": float(c[0]), "last": float(c[-1])}


def register(app: Any) -> None:
    r, P = app.route, app.platform

    def public_status(u: User) -> Dict[str, Any]:
        st = P.data.status()
        ops = u.has(Permission.OPERATE_SYSTEM)
        if not ops:                                               # gateway addresses and start-up details are for operations
            st["providers"] = [{k: v for k, v in p.items() if k in ("name", "label", "help", "connected")} for p in st["providers"]]
            st.pop("environment", None)
        st.update({"can_configure": ops, "can_fetch": u.has(Permission.RUN_BACKTEST),
                   "uploaded": sorted(app.datasets.get(u.user_id, {}))})
        return st

    @r("POST", "/api/data/upload")
    def upload(u, b):
        UserManager.authorize(u, Permission.RUN_BACKTEST)
        sym = str(b.get("symbol", "")).strip().upper()
        if not sym.isalnum() or len(sym) > 8:
            raise ValidationError("symbol must be 1-8 letters/digits")
        text = b.get("csv")
        if not isinstance(text, str) or not text.strip():
            raise ValidationError("csv text is required")
        mine = app.datasets.setdefault(u.user_id, {})
        if sym not in mine and len(mine) >= MAX_UPLOAD_SYMBOLS:
            raise ValidationError(f"at most {MAX_UPLOAD_SYMBOLS} uploaded symbols per user")
        try:
            df = pd.read_csv(io.StringIO(text), nrows=MAX_UPLOAD_ROWS + 1)
        except Exception as exc:  # noqa: BLE001 - any parser failure is the user's data problem
            raise DataError(f"cannot parse CSV: {type(exc).__name__}") from None
        if len(df) > MAX_UPLOAD_ROWS:
            raise DataError(f"too many rows (max {MAX_UPLOAD_ROWS})")
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "date" not in df.columns:
            raise DataError("CSV needs a 'date' column plus open, high, low, close (volume optional)")
        try:
            df["date"] = pd.to_datetime(df["date"], format="mixed")             # each row parsed on its own, no guessing from the first
        except Exception:  # noqa: BLE001
            raise DataError("could not parse the 'date' column") from None
        df = df.set_index("date").sort_index()
        df.index.name = None
        validate_ohlcv(df, sym)                                           # raises DataError with the exact problem
        mine[sym] = df[[c for c in ("open", "high", "low", "close", "volume") if c in df.columns]].astype(float)
        P.audit.record(u.username, "data.upload", symbol=sym, rows=len(df))
        return {"symbol": sym, "rows": len(df), "start": df.index[0], "end": df.index[-1]}

    @r("GET", "/api/data")
    def list_data(u, _b):
        return [{"symbol": k, "rows": len(v), "start": v.index[0], "end": v.index[-1]}
                for k, v in sorted(app.datasets.get(u.user_id, {}).items())]

    @r("POST", "/api/data/clear")
    def clear_data(u, _b):
        app.datasets.pop(u.user_id, None)
        return {"ok": True}

    @r("GET", "/api/instruments")
    def instruments(u, _b):
        UserManager.authorize(u, Permission.VIEW_PORTFOLIO)
        return {"instruments": P.instruments.export_state()[:1000], "count": len(P.instruments.export_state()),
                "can_load": u.has(Permission.MANAGE_INSTRUMENTS) and u.has(Permission.OPERATE_SYSTEM)}

    @r("POST", "/api/instruments/load")
    def load_instruments(u, b):
        text = str(b.get("csv", ""))
        if not text.strip():
            raise ValidationError("paste CSV text with a header row (symbol, asset_class, name, exchange, currency, ...)")
        if len(text.encode()) > MAX_INSTRUMENT_CSV:
            raise ValidationError(f"CSV too large (limit {MAX_INSTRUMENT_CSV // 1000} KB)")
        try:
            frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValidationError(f"cannot read the CSV: {exc}") from None
        if "symbol" not in frame.columns:
            raise ValidationError("the CSV needs a 'symbol' column", details={"columns": list(frame.columns)})
        report = P.desk(u, "operations").load_instruments(frame.to_dict("records"), on_error="skip", overwrite=bool(b.get("overwrite")))
        return {**report, "count": len(P.instruments.export_state())}

    @r("GET", "/api/data/sources")
    def sources(u, _b):
        UserManager.authorize(u, Permission.VIEW_PORTFOLIO)
        return public_status(u)

    @r("GET", "/api/data/providers")
    def providers(u, _b):
        P.desk(u, "operations")                                   # the full view, with settings, is for operations
        return public_status(u)

    @r("POST", "/api/data/sample/preview")
    def preview(u, b):
        P.desk(u, "research")
        scenario = str(b.get("scenario") or "random_walk")
        symbols = [str(s).strip().upper() for s in b.get("symbols") or ["AAA", "BBB", "CCC"] if str(s).strip()]
        if not 1 <= len(symbols) <= MAX_PREVIEW_SYMBOLS:
            raise ValidationError(f"preview 1 to {MAX_PREVIEW_SYMBOLS} symbols")
        bars = int(b.get("bars", 756))
        if not 60 <= bars <= 3000:
            raise ValidationError("bars must be between 60 and 3000")
        frames = P.data.sample(scenario, symbols, bars=bars, seed=int(b.get("seed", 7)))
        return {"scenario": scenario, "description": synthetic.SCENARIOS[scenario], "bars": bars,
                "stats": [{"symbol": s, **_stats(f["close"])} for s, f in frames.items()],
                "series": {s: series_points(f["close"] / f["close"].iloc[0] * 100.0) for s, f in frames.items()}}

    @r("POST", "/api/data/fetch")
    def fetch(u, b):
        symbols = b.get("symbols") or []
        if isinstance(symbols, str):
            symbols = symbols.split(",")
        return P.desk(u, "research").fetch_history(str(b.get("provider", "")), symbols, b.get("start") or None, b.get("end") or None,
                                                   str(b.get("timeframe") or "1Day"))

    @r("POST", "/api/data/history/remove")
    def remove(u, b):
        if not P.desk(u, "research").remove_history(str(b.get("symbol", "")), str(b.get("timeframe") or "1Day")):
            raise ValidationError("that series is not in the library")
        return {"ok": True, "library": P.data.library.summary()}

    @r("POST", "/api/data/history/series")
    def series(u, b):
        P.desk(u, "research")
        P.data.require_real_data_allowed("using downloaded market data")
        frame = P.data.library.get(str(b.get("symbol", "")).upper(), str(b.get("timeframe") or "1Day"))
        return {"symbol": str(b.get("symbol", "")).upper(), "close": series_points(frame["close"]), "stats": _stats(frame["close"])}

    @r("POST", "/api/data/providers/connect")
    def connect(u, b):
        info = P.desk(u, "operations").connect_data_provider(str(b.get("name", "")), b.get("settings") or {}, b.get("secrets") or {})
        return {"provider": info, **public_status(u)}

    @r("POST", "/api/data/providers/disconnect")
    def disconnect(u, b):
        P.desk(u, "operations").disconnect_data_provider(str(b.get("name", "")))
        return public_status(u)

    @r("POST", "/api/data/providers/test")
    def test(u, b):
        result = P.desk(u, "operations").test_data_provider(str(b.get("name", "")), b.get("symbol") or None)
        return {"result": result, **public_status(u)}

    @r("POST", "/api/data/sample_only")
    def sample_only(u, b):
        P.desk(u, "operations").set_sample_data_only(bool(b.get("on")))
        return public_status(u)
