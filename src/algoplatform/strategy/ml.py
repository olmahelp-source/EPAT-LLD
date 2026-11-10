"""MLStrategy: a rolling-retrained logistic model on causal features with triple-barrier labels.

Why this design is leakage-safe:
* features at bar t use data up to t only (``research.features.build_features``);
* a training label for event t0 needs ``horizon`` future bars, so training uses only events whose barrier time is
  already in the past at the moment of retraining (``triple_barrier`` drops incomplete events);
* the scaler is fitted on training rows only;
* the model is retrained every ``retrain_every`` bars from the data seen so far, never from the full history;
* if recent features mostly fall outside the range seen in training (75th-percentile share over the last ``drift_bars`` bars above ``drift_limit``) the model
  is extrapolating, so it stands aside until the next retrain catches up.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.exceptions import StrategyError
from ..core.models import Signal
from ..research.features import Scaler, build_features
from ..research.labels import daily_vol, sample_weights, triple_barrier
from ..research.models import LogisticModel, out_of_range_share
from .base import STRATEGIES, Strategy, StrategyContext


@STRATEGIES.register("ml_logistic")
class MLStrategy(Strategy):
    name = "ml_logistic"
    PARAMS = {"train_window": 300, "retrain_every": 20, "horizon": 5, "barrier": 1.0, "vol_span": 50, "edge": 0.05,
              "allow_short": False, "l2": 1.0, "drift_limit": 0.5, "drift_bars": 10, "min_samples": 80, "step": 0.25}
    PARAM_BOUNDS = {"train_window": (150, 500), "horizon": (3, 15), "edge": (0.0, 0.2)}

    def validate_params(self) -> None:
        p = self.params
        if p["train_window"] < 100 or p["horizon"] < 1 or p["retrain_every"] < 1:
            raise StrategyError("need train_window >= 100, horizon >= 1, retrain_every >= 1", details=p)
        if not 0 <= p["edge"] < 0.5 or p["step"] <= 0 or not 0 <= p["drift_limit"] <= 1:
            raise StrategyError("edge must be in [0, 0.5), step positive, drift_limit in [0, 1]", details=p)
        self._state: Dict[str, dict] = {}
        self.drift_events = 0
        self.trainings = 0

    @property
    def warmup(self) -> int:
        return int(self.p("train_window")) + int(self.p("horizon")) + 60

    # ---- training ---------------------------------------------------------------------------------------
    def _train(self, df: pd.DataFrame, feats: pd.DataFrame) -> Optional[dict]:
        h, p = int(self.p("horizon")), self.params
        close = df["close"]
        window = feats.iloc[-(int(p["train_window"]) + h):]
        events = window.dropna().index[:-h]                                         # last h rows have no complete label yet
        events = events[events.isin(close.index)]
        tgt = daily_vol(close, int(p["vol_span"])) * float(p["barrier"])
        lab = triple_barrier(close, events, tgt, h, (1.0, 1.0))
        if len(lab) < int(p["min_samples"]) or lab["label"].gt(0).nunique() < 2:
            return None
        y = (lab["label"] > 0).astype(float).to_numpy()
        X = feats.loc[lab.index].to_numpy()
        scaler = Scaler().fit(X)
        w = sample_weights(lab["t1"], close.index).to_numpy()
        model = LogisticModel(l2=float(p["l2"])).fit(scaler.transform(X), y, w)
        self.trainings += 1
        return {"model": model, "scaler": scaler, "train_X": X, "age": 0}

    # ---- decisions --------------------------------------------------------------------------------------
    def on_bar(self, ctx: StrategyContext) -> List[Signal]:
        out: List[Signal] = []
        n = self.warmup + 40
        for s in self.symbols:
            if s not in ctx.bars or not self.ready(ctx, s):
                continue
            w = ctx.window(s, n)
            df = pd.DataFrame({k: v for k, v in w.items()}, index=pd.RangeIndex(len(w["close"])))
            feats = build_features(df)
            st = self._state.get(s)
            if st is None or st["age"] >= int(self.p("retrain_every")):
                trained = self._train(df, feats)
                if trained is not None:
                    st = self._state[s] = trained
            if st is None:
                continue
            st["age"] += 1
            x = feats.iloc[-1].to_numpy()
            if not np.isfinite(x).all():
                continue
            recent = feats.iloc[-int(self.p("drift_bars")):].to_numpy()
            drift = float(np.quantile(out_of_range_share(st["train_X"], recent), 0.75))   # several features at once
            if drift > float(self.p("drift_limit")):
                self.drift_events += 1
                out.append(Signal(s, 0.0, f"drift outside-range={drift:.0%}"))
                continue
            prob = float(st["model"].predict_proba(st["scaler"].transform(x))[0])
            edge, step = float(self.p("edge")), float(self.p("step"))
            if prob > 0.5 + edge:
                wgt = min(1.0, (prob - 0.5) * 2)
            elif prob < 0.5 - edge and self.p("allow_short"):
                wgt = -min(1.0, (0.5 - prob) * 2)
            else:
                wgt = 0.0
            wgt = round(wgt / step) * step                           # coarse steps keep position churn (costs) down
            out.append(Signal(s, float(np.clip(wgt, -1, 1)), f"p={prob:.2f}"))
        return out
