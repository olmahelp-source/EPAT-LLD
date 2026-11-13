"""Quantitative research: volatility estimates and forecasts, option-chain analytics.

Every route checks the caller's permission on the server (through the role desks or ``UserManager.authorize``); hiding a
page in the browser is a convenience, not the control."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ....foundation.core.exceptions import ValidationError
from ....quantitative.derivatives import synthetic_chain
from ....access.users.manager import User
from ....quantitative.volatility import (compare_vol_models, estimators, EWMAVol, GarchVol, HARVol, HistoricalVol, volatility_cone,
                                         VolatilityManager)

from ..common import bounded, last_value, series_points


VOL_MODELS = {"ewma": EWMAVol, "hist": HistoricalVol, "har": HARVol, "garch": GarchVol}


def register(app: Any) -> None:
    r, P = app.route, app.platform

    def research(u: User):
        return P.desk(u, "research")                              # raises AuthorizationError for roles without research

    def first_symbol(u: User, b: Dict[str, Any]):
        md, symbols = app._data(b, u)
        return md.frame(symbols[0]), symbols[0], md, symbols

    @r("POST", "/api/lab/volatility")
    def volatility(u, b):
        research(u)
        df, sym, _, _ = first_symbol(u, b)
        o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
        rets = c.pct_change().dropna()
        window = int(b.get("window", 20))
        est = {"close_to_close": estimators.close_to_close(c, window), "parkinson": estimators.parkinson(h, l, window),
               "garman_klass": estimators.garman_klass(o, h, l, c, window), "rogers_satchell": estimators.rogers_satchell(o, h, l, c, window),
               "yang_zhang": estimators.yang_zhang(o, h, l, c, window), "ewma": estimators.ewma(c)}
        model = b.get("model", "ewma")
        if model not in VOL_MODELS:
            raise ValidationError(f"unknown volatility model '{model}'", details={"valid": sorted(VOL_MODELS)})
        target = float(b.get("target_vol", 0.10))
        view = VolatilityManager(target_vol=target, model=model).assess(rets.to_numpy())
        n = len(rets)
        train = max(250, n // 2)
        league = None
        if n >= train + 60:
            step = max(20, (n - train) // 6)                         # at most about six refits: keeps the request fast
            league = compare_vol_models(rets.to_numpy(), {k: v for k, v in VOL_MODELS.items() if b.get("league_garch", True) or k != "garch"},
                                        train=train, step=step, horizon=5).to_dict("records")
        cone = volatility_cone(c).reset_index().to_dict("records") if n > 300 else []
        return {"symbol": sym, "estimators": [{"name": k, "value": last_value(v)} for k, v in est.items()], "assessment": view, "league": league,
                "cone": cone, "series": {"yang_zhang": series_points(pd.Series(est["yang_zhang"], index=df.index).dropna()),
                                         "ewma": series_points(pd.Series(est["ewma"], index=df.index).dropna())}}

    @r("POST", "/api/lab/option_chain")
    def option_chain(u, b):
        research(u)
        days = [int(d) for d in (b["days"] if b.get("days") is not None else [30, 60, 90])]      # an empty list is an error, not "default"
        if not 1 <= len(days) <= 6 or not all(1 <= d <= 730 for d in days) or len(set(days)) != len(days):
            raise ValidationError("days: 1 to 6 different expiries, each 1 to 730 days out")
        chain = synthetic_chain(spot=bounded(b, "spot", 100.0, 1.0, 100_000.0), days=sorted(days),
                                atm_vol=bounded(b, "atm_vol", 0.20, 0.03, 2.0), skew=bounded(b, "skew", -0.10, -1.0, 1.0),
                                smile=bounded(b, "smile", 0.30, 0.0, 5.0), seed=int(b.get("seed", 0)), underlying="SYN")
        expiries = chain.expiries()
        exp = expiries[bounded(b, "expiry", 0, 0, len(expiries) - 1, int)]
        frame = chain.frame(exp)
        rows: Dict[float, Dict[str, Any]] = {}
        for _, q in frame.iterrows():
            row = rows.setdefault(float(q["strike"]), {"strike": float(q["strike"])})
            for k in ("bid", "ask", "iv", "delta", "gamma", "theta", "vega", "open_interest"):
                row[f"{q['kind']}_{k}"] = q[k]
        gex = chain.gamma_exposure(exp)
        return {"underlying": chain.underlying, "spot": chain.spot, "expiry": str(exp), "summary": chain.summary(exp),
                "expiries": [{"index": i, "expiry": str(t["expiry"]), "days": int(t["days"]), "atm_iv": t["atm_iv"]}
                             for i, t in enumerate(chain.term_structure().to_dict("records"))],
                "skew": chain.skew(exp), "expected_move": chain.expected_move(exp), "walls": chain.oi_walls(exp),
                "smile": chain.smile(exp)[["strike", "moneyness", "iv", "kind"]].to_dict("records"),
                "gamma": [{"strike": float(k), "gex": float(v)} for k, v in gex.items()], "gamma_total": float(gex.sum()),
                "quality": chain.quality_report(), "chain": [rows[k] for k in sorted(rows)]}
