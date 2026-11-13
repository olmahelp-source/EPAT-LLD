"""API routes for the research job groups that had no screen of their own.

* Quantitative research: an option chain with implied volatility, Greeks, smile, skew, term structure and positioning.
* Signal libraries: news sentiment, scored per headline and per instrument.
* Learning and AI: walk-forward prediction with the model zoo, market regimes, change points and anomalies.

Kept apart from ``quant_api.py`` so each file serves one set of job groups. Permissions are checked on the server:
research needs the research desk, and fitting a prediction model also needs ``model:train`` (quant and admin only).
Everything here runs on synthetic data or text the user types in; nothing calls an outside service.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from ..ai import RegimeDetector, bocpd, changepoints_from_run_length, rolling_robust_zscore
from ..core.enums import Permission
from ..core.exceptions import ValidationError
from ..derivatives import synthetic_chain
from ..dl.models import MLP
from ..ml import (KNN, DecisionTree, GaussianNB, GradientBoosting, RandomForest, forward_returns, information_coefficient, roc_auc,
                  signal_from_probability, walk_forward_predict)
from ..research import FEATURE_NAMES, LogisticModel, build_features
from ..sentiment import EntityMapper, classify_event, score_text
from ..users.manager import User, UserManager

# Small, fast settings: each request refits a model up to MAX_FITS times on a few hundred rows.
MODELS: Dict[str, Callable[[], Any]] = {
    "logistic": lambda: LogisticModel(l2=1.0),
    "naive_bayes": lambda: GaussianNB(),
    "knn": lambda: KNN(k=25),
    "tree": lambda: DecisionTree(max_depth=3, min_samples_leaf=25),
    "forest": lambda: RandomForest(n_trees=30, max_depth=4, min_samples_leaf=20, seed=0),
    "boosting": lambda: GradientBoosting(n_estimators=40, max_depth=2, min_samples_leaf=20, seed=0),
    "mlp": lambda: MLP(hidden=(16,), epochs=60, patience=8, seed=0),
}
MAX_FITS = 15
MAX_HEADLINES = 100


def _bounded(b: Dict[str, Any], key: str, default: float, lo: float, hi: float, kind: Callable[[Any], Any] = float) -> Any:
    value = kind(b.get(key, default))
    if not lo <= value <= hi:
        raise ValidationError(f"{key} must be between {lo:g} and {hi:g}", details={key: value})
    return value


def _date(index: pd.Index, i: int) -> str:
    return pd.Timestamp(index[i]).strftime("%Y-%m-%d")


def register(app: Any) -> None:
    from .server import series_points                             # deferred: server imports this module
    r = app.route

    def research(u: User):
        return app.platform.desk(u, "research")                   # raises AuthorizationError for roles without research

    def first_frame(u: User, b: Dict[str, Any]):
        md, symbols = app._data(b, u)
        return md.frame(symbols[0]), symbols[0]

    # ---- quantitative research: options ------------------------------------------------------------------------------------
    @r("POST", "/api/lab/option_chain")
    def option_chain(u, b):
        research(u)
        days = [int(d) for d in (b["days"] if b.get("days") is not None else [30, 60, 90])]      # an empty list is an error, not "default"
        if not 1 <= len(days) <= 6 or not all(1 <= d <= 730 for d in days) or len(set(days)) != len(days):
            raise ValidationError("days: 1 to 6 different expiries, each 1 to 730 days out")
        chain = synthetic_chain(spot=_bounded(b, "spot", 100.0, 1.0, 100_000.0), days=sorted(days),
                                atm_vol=_bounded(b, "atm_vol", 0.20, 0.03, 2.0), skew=_bounded(b, "skew", -0.10, -1.0, 1.0),
                                smile=_bounded(b, "smile", 0.30, 0.0, 5.0), seed=int(b.get("seed", 0)), underlying="SYN")
        expiries = chain.expiries()
        exp = expiries[_bounded(b, "expiry", 0, 0, len(expiries) - 1, int)]
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

    # ---- signal libraries: sentiment ---------------------------------------------------------------------------------------
    @r("POST", "/api/lab/sentiment")
    def sentiment(u, b):
        research(u)
        lines = [x.strip()[:300] for x in str(b.get("headlines", "")).splitlines() if x.strip()]
        if not lines:
            raise ValidationError("enter at least one headline, one per line")
        if len(lines) > MAX_HEADLINES:
            raise ValidationError(f"at most {MAX_HEADLINES} headlines per request", details={"given": len(lines)})
        aliases: Dict[str, List[str]] = {}
        for line in str(b.get("aliases", "")).splitlines():
            if not line.strip():
                continue
            sym, _, names = line.partition("=")
            sym = sym.strip().upper()
            if not sym.isalnum() or len(sym) > 8:
                raise ValidationError(f"alias line '{line.strip()[:40]}': write SYMBOL = name, other name")
            aliases[sym] = [n.strip() for n in names.split(",") if n.strip()]
        mapper = EntityMapper(aliases) if aliases else None
        rows = []
        for text in lines:
            lex = score_text(text)
            events = classify_event(text)
            event_score = max(-1.0, min(1.0, sum(w for _, w in events)))
            # an identified event is stronger evidence than word counts, so it gets the larger share when present
            combined = 0.6 * event_score + 0.4 * lex["score"] if events else lex["score"]
            rows.append({"headline": text, "score": lex["score"], "terms": lex["n_terms"], "events": [name for name, _ in events],
                         "event_score": event_score, "combined": combined, "symbols": sorted(mapper.symbols(text)) if mapper else []})
        by_symbol: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            for sym in row["symbols"]:
                agg = by_symbol.setdefault(sym, {"symbol": sym, "headlines": 0, "total": 0.0, "events": []})
                agg["headlines"] += 1
                agg["total"] += row["combined"]
                agg["events"] += row["events"]
        for agg in by_symbol.values():
            agg["average"] = agg.pop("total") / agg["headlines"]
            agg["events"] = sorted(set(agg["events"]))
        scored = [x["combined"] for x in rows if x["terms"] or x["events"]]
        return {"headlines": rows, "by_symbol": sorted(by_symbol.values(), key=lambda a: -abs(a["average"])),
                "average": float(np.mean(scored)) if scored else 0.0, "neutral": len(rows) - len(scored)}

    # ---- learning and AI: walk-forward prediction --------------------------------------------------------------------------
    @r("GET", "/api/lab/models")
    def model_options(u, _b):
        UserManager.authorize(u, Permission.TRAIN_MODEL)
        return {"models": sorted(MODELS), "features": list(FEATURE_NAMES)}

    @r("POST", "/api/lab/models")
    def models(u, b):
        UserManager.authorize(u, Permission.TRAIN_MODEL)
        research(u)
        name = str(b.get("model", "logistic"))
        if name not in MODELS:
            raise ValidationError(f"unknown model '{name}'", details={"valid": sorted(MODELS)})
        df, sym = first_frame(u, b)
        horizon = _bounded(b, "horizon", 5, 1, 20, int)
        train = _bounded(b, "train_size", 250, 100, max(100, len(df) - 60), int)
        close = df["close"].to_numpy(float)
        X = build_features(df).to_numpy(float)
        fwd = forward_returns(close, horizon)
        y = np.where(np.isfinite(fwd), (fwd > 0).astype(float), np.nan)
        step = max(_bounded(b, "step", 20, 5, 250, int), math.ceil((len(y) - train - horizon) / MAX_FITS))
        wf = walk_forward_predict(MODELS[name], X, y, train_size=train, step=step, embargo=horizon)
        pred = wf["pred"]
        known = np.isfinite(pred) & np.isfinite(fwd)
        if known.sum() < 30:
            raise ValidationError("too few out-of-sample predictions to judge the model: use more bars or a shorter training window",
                                  details={"predictions": int(known.sum())})
        hit = float(np.mean((pred[known] > 0.5) == (fwd[known] > 0)))
        # trade the next bar on each prediction: decided at the close of t, held to the close of t+1 (no costs)
        signal = signal_from_probability(pred, edge=float(b.get("edge", 0.02)))
        nxt = forward_returns(close, 1)
        live = np.flatnonzero(np.isfinite(pred) & np.isfinite(nxt))
        idx = df.index[live]
        strat = pd.Series(1.0 + signal[live] * nxt[live], index=idx).cumprod()
        hold = pd.Series(1.0 + nxt[live], index=idx).cumprod()
        last = int(np.flatnonzero(np.isfinite(pred))[-1])
        return {"symbol": sym, "model": name, "horizon": horizon, "rows": len(y), "train_size": train, "step": step,
                "fits": wf["fits"], "failed_fits": wf["failed_fits"], "predictions": int(known.sum()),
                "accuracy": hit, "base_rate": float(np.mean(fwd[known] > 0)), "auc": roc_auc(y[known], pred[known]),
                "ic": information_coefficient(pred[known], fwd[known]), "exposure": float(np.mean(np.abs(signal[live]))),
                "strategy_return": float(strat.iloc[-1] - 1.0), "buy_hold_return": float(hold.iloc[-1] - 1.0),
                "equity": series_points(strat), "buy_hold": series_points(hold),
                "probability": series_points(pd.Series(pred, index=df.index).dropna()),
                "latest": {"date": _date(df.index, last), "probability": float(pred[last]), "signal": float(signal[last])},
                "features": list(FEATURE_NAMES)}

    # ---- learning and AI: regimes, change points, anomalies ---------------------------------------------------------------
    @r("POST", "/api/lab/regimes")
    def regimes(u, b):
        research(u)
        df, sym = first_frame(u, b)
        rets = df["close"].pct_change().dropna()
        if len(rets) < 200:
            raise ValidationError("regime detection needs at least 200 bars")
        states = _bounded(b, "states", 2, 2, 4, int)
        train = max(150, len(rets) // 2)
        det = RegimeDetector(n_states=states, seed=0).fit(rets.iloc[:train].to_numpy())
        probs = det.probabilities(rets.to_numpy())
        labels = det.labels
        regime = probs.argmax(axis=1)
        mu, sd = float(rets.iloc[:train].mean()), float(rets.iloc[:train].std())
        bo = bocpd((rets.to_numpy() - mu) / (sd or 1.0), hazard=1.0 / _bounded(b, "expected_run", 250, 20, 2000))
        points = changepoints_from_run_length(bo["run_length"], min_drop=_bounded(b, "min_drop", 20, 5, 200, int))
        z = rolling_robust_zscore(rets.to_numpy(), 50)
        limit = _bounded(b, "z_limit", 5.0, 2.0, 20.0)
        outliers = np.flatnonzero(np.isfinite(z) & (np.abs(z) > limit))
        segments: List[Dict[str, Any]] = []
        start = 0
        for i in range(1, len(regime) + 1):
            if i == len(regime) or regime[i] != regime[start]:
                segments.append({"label": labels[int(regime[start])], "start": _date(rets.index, start), "end": _date(rets.index, i - 1),
                                 "bars": i - start})
                start = i
        return {"symbol": sym, "labels": labels, "trained_on": train, "describe": det.describe(), "current": det.current(rets.to_numpy()),
                "turbulent": series_points(pd.Series(probs[:, -1], index=rets.index)),
                "close": series_points(df["close"]), "segments": segments[-20:][::-1],
                "time_in_regime": {labels[k]: float(np.mean(regime == k)) for k in range(states)},
                "changepoints": [{"date": _date(rets.index, max(0, p)), "close": float(df["close"].loc[rets.index[max(0, p)]])} for p in points][-20:][::-1],
                "change_prob": series_points(pd.Series(bo["change_prob"], index=rets.index)),
                "anomalies": [{"date": _date(rets.index, int(i)), "return": float(rets.iloc[int(i)]), "z": float(z[int(i)])} for i in outliers][-20:][::-1]}


__all__: List[str] = ["register", "MODELS"]
