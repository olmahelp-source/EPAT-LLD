"""API routes for market data: the sample-data generator, data providers (Alpaca, Interactive Brokers), the shared
history library and the sample-data-only switch.

Kept apart from the other route files so each serves one job group (here: market and reference data). Every route
goes through ``TradingPlatform.data`` and the role desks, so permissions are checked on the server. API keys arrive
in a request body, go straight to the credential vault and are never returned, logged or saved to disk.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..core.enums import Permission
from ..core.exceptions import ValidationError
from ..data import synthetic
from ..users.manager import User, UserManager

MAX_PREVIEW_SYMBOLS = 6


def _stats(close) -> Dict[str, float]:
    c = np.asarray(close, float)
    r = np.diff(np.log(c))
    peak = np.maximum.accumulate(c)
    return {"total_return": float(c[-1] / c[0] - 1), "volatility": float(r.std() * np.sqrt(252)) if len(r) > 1 else 0.0,
            "max_drawdown": float(np.min(c / peak - 1.0)), "first": float(c[0]), "last": float(c[-1])}


def register(app: Any) -> None:
    from .server import series_points                             # deferred: server imports this module
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

    @r("GET", "/api/data/sources")
    def sources(u, _b):
        UserManager.authorize(u, Permission.VIEW_PORTFOLIO)
        return public_status(u)

    @r("GET", "/api/data/providers")
    def providers(u, _b):
        P.desk(u, "operations")                                   # the full view, with settings, is for operations
        return public_status(u)

    # ---- sample data --------------------------------------------------------------------------------------------------------
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

    # ---- history library ----------------------------------------------------------------------------------------------------
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

    # ---- providers (operations) -------------------------------------------------------------------------------------------
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


__all__: List[str] = ["register"]
