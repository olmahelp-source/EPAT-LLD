"""Learning and AI: walk-forward prediction models, market regimes, change points and anomalies.

Every route checks the caller's permission on the server (through the role desks or ``UserManager.authorize``); hiding a
page in the browser is a convenience, not the control."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from ...ai import bocpd, changepoints_from_run_length, RegimeDetector, rolling_robust_zscore
from ...core.enums import Permission
from ...core.exceptions import ValidationError
from ...dl.models import MLP
from ...ml import (DecisionTree, forward_returns, GaussianNB, GradientBoosting, information_coefficient, KNN, RandomForest, roc_auc,
                   signal_from_probability, walk_forward_predict)
from ...research import build_features, FEATURE_NAMES, LogisticModel
from ...users.manager import User, UserManager

from ..common import bounded, date_at, series_points


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


def register(app: Any) -> None:
    r, P = app.route, app.platform

    def research(u: User):
        return P.desk(u, "research")                              # raises AuthorizationError for roles without research

    def first_frame(u: User, b: Dict[str, Any]):
        md, symbols = app._data(b, u)
        return md.frame(symbols[0]), symbols[0]

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
        horizon = bounded(b, "horizon", 5, 1, 20, int)
        train = bounded(b, "train_size", 250, 100, max(100, len(df) - 60), int)
        close = df["close"].to_numpy(float)
        X = build_features(df).to_numpy(float)
        fwd = forward_returns(close, horizon)
        y = np.where(np.isfinite(fwd), (fwd > 0).astype(float), np.nan)
        step = max(bounded(b, "step", 20, 5, 250, int), math.ceil((len(y) - train - horizon) / MAX_FITS))
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
                "latest": {"date": date_at(df.index, last), "probability": float(pred[last]), "signal": float(signal[last])},
                "features": list(FEATURE_NAMES)}

    @r("POST", "/api/lab/regimes")
    def regimes(u, b):
        research(u)
        df, sym = first_frame(u, b)
        rets = df["close"].pct_change().dropna()
        if len(rets) < 200:
            raise ValidationError("regime detection needs at least 200 bars")
        states = bounded(b, "states", 2, 2, 4, int)
        train = max(150, len(rets) // 2)
        det = RegimeDetector(n_states=states, seed=0).fit(rets.iloc[:train].to_numpy())
        probs = det.probabilities(rets.to_numpy())
        labels = det.labels
        regime = probs.argmax(axis=1)
        mu, sd = float(rets.iloc[:train].mean()), float(rets.iloc[:train].std())
        bo = bocpd((rets.to_numpy() - mu) / (sd or 1.0), hazard=1.0 / bounded(b, "expected_run", 250, 20, 2000))
        points = changepoints_from_run_length(bo["run_length"], min_drop=bounded(b, "min_drop", 20, 5, 200, int))
        z = rolling_robust_zscore(rets.to_numpy(), 50)
        limit = bounded(b, "z_limit", 5.0, 2.0, 20.0)
        outliers = np.flatnonzero(np.isfinite(z) & (np.abs(z) > limit))
        segments: List[Dict[str, Any]] = []
        start = 0
        for i in range(1, len(regime) + 1):
            if i == len(regime) or regime[i] != regime[start]:
                segments.append({"label": labels[int(regime[start])], "start": date_at(rets.index, start), "end": date_at(rets.index, i - 1),
                                 "bars": i - start})
                start = i
        return {"symbol": sym, "labels": labels, "trained_on": train, "describe": det.describe(), "current": det.current(rets.to_numpy()),
                "turbulent": series_points(pd.Series(probs[:, -1], index=rets.index)),
                "close": series_points(df["close"]), "segments": segments[-20:][::-1],
                "time_in_regime": {labels[k]: float(np.mean(regime == k)) for k in range(states)},
                "changepoints": [{"date": date_at(rets.index, max(0, p)), "close": float(df["close"].loc[rets.index[max(0, p)]])} for p in points][-20:][::-1],
                "change_prob": series_points(pd.Series(bo["change_prob"], index=rets.index)),
                "anomalies": [{"date": date_at(rets.index, int(i)), "return": float(rets.iloc[int(i)]), "z": float(z[int(i)])} for i in outliers][-20:][::-1]}
