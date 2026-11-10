"""Walk-forward optimisation: optimise on a train window, test untouched on the next."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from ..analytics.metrics import performance_metrics
from ..backtest.engine import BacktestEngine
from ..core.exceptions import OptimizationError, PlatformError
from ..core.logging import get_logger
from .optimizers import backtest_objective, optimize
from .space import ParameterSpace

log = get_logger(__name__)


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame
    oos_equity: pd.Series
    oos_metrics: Dict[str, float]
    efficiency: float                       # mean OOS score / mean IS score (1 = no degradation)
    param_stability: Dict[str, float]       # coefficient of variation of chosen params across folds

    def summary(self) -> Dict[str, Any]:
        return {"folds": len(self.folds), "walk_forward_efficiency": self.efficiency,
                "oos_sharpe": self.oos_metrics.get("sharpe"), "oos_total_return": self.oos_metrics.get("total_return"),
                "oos_max_drawdown": self.oos_metrics.get("max_drawdown"), "param_stability_cv": self.param_stability}


def walk_forward(engine: BacktestEngine, factory: Callable[..., Any], space: ParameterSpace, train_bars: int,
                 test_bars: int, metric: str = "sharpe", method: str = "random", anchored: bool = False,
                 min_trades: int = 3, seed: int = 0, **opt_kwargs: Any) -> WalkForwardResult:
    """Rolling (or anchored) walk-forward. Each fold: optimise on ``train_bars``, then run the best
    parameters on the following ``test_bars``. Test equity curves are chained into one
    out-of-sample curve. Earlier bars are available as warm-up history but never traded."""
    timeline = engine.data.timeline()
    n = len(timeline)
    if train_bars < 30 or test_bars < 5 or train_bars + test_bars > n:
        raise OptimizationError("bad window sizes for the available data",
                                details={"bars": n, "train": train_bars, "test": test_bars})
    if method in ("random",):
        opt_kwargs.setdefault("seed", seed)
    rows: List[Dict[str, Any]] = []
    oos_returns: List[pd.Series] = []
    i = 0
    while i + train_bars + test_bars <= n:
        tr_lo = 0 if anchored else i
        tr_start, tr_end = timeline[tr_lo], timeline[i + train_bars - 1]
        te_start, te_end = timeline[i + train_bars], timeline[i + train_bars + test_bars - 1]
        obj = backtest_objective(engine, factory, metric, tr_start, tr_end, min_trades)
        try:
            opt = optimize(obj, space, method, **opt_kwargs)
        except PlatformError as exc:
            log.warning("fold %s..%s skipped: %s", tr_start, tr_end, exc)
            i += test_bars
            continue
        res = engine.run(factory(**opt.best_params), start=te_start, end=te_end)
        m = res.metrics()
        oos_returns.append(res.returns)
        rows.append({"train_start": tr_start, "train_end": tr_end, "test_start": te_start, "test_end": te_end,
                     "train_score": opt.best_score, "test_score": float(m.get(metric, np.nan)),
                     "test_return": m["total_return"], "trades": len(res.trades), **opt.best_params})
        i += test_bars
    if not rows:
        raise OptimizationError("walk-forward produced no valid folds")
    folds = pd.DataFrame(rows)
    rets = pd.concat(oos_returns)
    rets = rets[~rets.index.duplicated(keep="first")].sort_index()
    equity = engine.config.initial_capital * (1 + rets).cumprod()
    equity = pd.concat([pd.Series([engine.config.initial_capital], index=[rets.index[0] - pd.Timedelta(days=1)]), equity])
    is_mean, oos_mean = folds["train_score"].mean(), folds["test_score"].mean()
    eff = float(oos_mean / is_mean) if is_mean and np.isfinite(is_mean) and is_mean != 0 else float("nan")
    stab = {}
    for name in space.names:
        col = folds[name].astype(float)
        stab[name] = float(col.std(ddof=0) / abs(col.mean())) if len(col) > 1 and col.mean() != 0 else 0.0
    return WalkForwardResult(folds, equity, performance_metrics(equity), eff, stab)
