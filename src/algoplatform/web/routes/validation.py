"""Validation: backtests, parameter optimisation, walk-forward analysis.

Every route checks the caller's permission on the server (through the role desks or ``UserManager.authorize``); hiding a
page in the browser is a convenience, not the control."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict

import numpy as np

from ...analytics.metrics import drawdown_series
from ...backtest import BacktestEngine
from ...core.enums import Permission
from ...core.exceptions import ValidationError
from ...optimize import ParameterSpace, walk_forward
from ...strategy.base import STRATEGIES
from ...users.manager import UserManager

from ..common import series_points, SLOW


def _pop(n: int, cap: int, minimum: int) -> int:
    return max(minimum, min(cap, n // 3))

OPTIMIZERS: Dict[str, Callable[[int, int], Dict[str, Any]]] = {
    "random": lambda n, d: {"n_iter": n},
    "genetic": lambda n, d: {"population": max(6, n // 4), "generations": 4},
    "bayesian": lambda n, d: {"n_init": _pop(n, 10, 2), "n_iter": max(n, _pop(n, 10, 2) + 1), "candidates": 300},
    "tpe": lambda n, d: {"n_init": _pop(n, 15, 4), "n_iter": max(n, _pop(n, 15, 4) + 1)},
    "pso": lambda n, d: {"particles": _pop(n, 12, 3), "iterations": max(1, n // _pop(n, 12, 3))},
    "de": lambda n, d: {"population": _pop(n, 15, 4), "generations": max(1, n // _pop(n, 15, 4) - 1)},
    "annealing": lambda n, d: {"n_iter": max(10, n)},
    "cmaes": lambda n, d: {"generations": max(1, n // (4 + int(3 * math.log(max(d, 1)))))},
}


def register(app: Any) -> None:
    r, P = app.route, app.platform

    @r("POST", "/api/backtest")
    def backtest(u, b):
        md, symbols = app._data(b, u)
        target = b["spec"] if b.get("spec") else b["strategy"]
        if not b.get("spec"):
            app._check_strategy(b["strategy"])
            symbols = app._symbols_for(b["strategy"], symbols)
        res = P.backtest(u, target, md, symbols, b.get("params") or {})
        rep = res.report(n_trials=int(b.get("n_trials", 1)))
        eq = res.equity
        bench = (1 + res.benchmark).cumprod() * eq.iloc[0] if res.benchmark is not None else None
        trades = res.trades_frame().tail(200)
        return {"metrics": rep["performance"], "trade_stats": rep["trades"],
                "significance": rep.get("significance"), "equity": series_points(eq),
                "drawdown": series_points(drawdown_series(eq)),
                "benchmark": series_points(bench) if bench is not None else None,
                "trades": trades.assign(entry_ts=trades["entry_ts"].astype(str),
                                        exit_ts=trades["exit_ts"].astype(str)).round(4).to_dict("records"),
                "orders": res.order_count, "rejected": res.rejected_orders,
                "kill_switch": res.kill_switch_triggered, "symbols": symbols, "risk": app._risk_measures(res.returns)}

    @r("POST", "/api/optimize")
    def optimize(u, b):
        method = b.get("method", "random")
        if method not in OPTIMIZERS:
            raise ValidationError(f"method must be one of {sorted(OPTIMIZERS)}")
        n = int(b.get("n_iter", 20))
        if not 5 <= n <= 60:
            raise ValidationError("n_iter must be between 5 and 60")
        if int(b.get("bars", 500)) > 1500:
            raise ValidationError("bars must be <= 1500 for optimisation")
        app._check_strategy(b["strategy"])
        if b["strategy"] in SLOW and n > 10:
            raise ValidationError("this strategy retrains a model at every step: use 10 trials or fewer")
        md, symbols = app._data({**b, "bars": b.get("bars", 500)}, u)
        symbols = app._symbols_for(b["strategy"], symbols)
        cls = STRATEGIES.get(b["strategy"])
        if not cls.PARAM_BOUNDS:
            raise ValidationError("strategy has no tunable parameters")
        space = ParameterSpace.from_bounds(cls.PARAM_BOUNDS)
        kw = OPTIMIZERS[method](n, len(space.names))
        res = P.optimize(u, b["strategy"], md, space, symbols, b.get("metric", "sharpe"), method, **kw)
        top = res.top(10)
        return {"summary": res.summary(), "top": top.to_dict("records"),
                "scores": [float(x) for x in res.trials["score"].replace([-np.inf], np.nan).dropna()],
                "plateau": res.plateau_score(space)}

    @r("POST", "/api/walkforward")
    def walkforward(u, b):
        UserManager.authorize(u, Permission.RUN_BACKTEST)
        name = b["strategy"]
        app._check_strategy(name)
        cls = STRATEGIES.get(name)
        if not cls.PARAM_BOUNDS:
            raise ValidationError("strategy has no tunable parameters")
        bars, train, test, n = (int(b.get("bars", 800)), int(b.get("train_bars", 300)), int(b.get("test_bars", 100)),
                                int(b.get("n_iter", 8)))
        if bars > 1500 or not 3 <= n <= (8 if name in SLOW else 20):
            raise ValidationError("bars must be <= 1500 and n_iter 3-20 (3-8 for model-based strategies)")
        md, symbols = app._data({**b, "bars": bars}, u)
        symbols = app._symbols_for(name, symbols)
        wf = walk_forward(BacktestEngine(md, P.config), P._factory(name, symbols), ParameterSpace.from_bounds(cls.PARAM_BOUNDS),
                          train, test, metric=b.get("metric", "sharpe"), method="random", n_iter=n, min_trades=1)
        P.audit.record(u.username, "walkforward.run", strategy=name, folds=len(wf.folds))
        folds = wf.folds.copy()
        for c in ("train_start", "train_end", "test_start", "test_end"):
            folds[c] = folds[c].astype(str).str[:10]
        return {"folds": folds.round(4).to_dict("records"), "oos_equity": series_points(wf.oos_equity),
                "oos_metrics": wf.oos_metrics, "efficiency": wf.efficiency, "stability": wf.param_stability,
                "params": list(cls.PARAM_BOUNDS)}
