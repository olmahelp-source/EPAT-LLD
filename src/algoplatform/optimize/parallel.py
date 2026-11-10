"""Picklable backtest objective for process-pool optimisation.

A closure over a live engine cannot be sent to worker processes. ``ParallelBacktestObjective`` instead carries plain data
(price frames, config dict, strategy name or rule spec) and rebuilds the engine inside each worker once.

    obj = make_parallel_objective(market_data, "sma_crossover", symbols, config)
    result = random_search(obj, space, n_iter=40, n_jobs=4)
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from ..backtest.engine import BacktestEngine
from ..core.config import PlatformConfig
from ..core.exceptions import OptimizationError
from ..data.store import MarketData
from ..strategy.base import STRATEGIES


class ParallelBacktestObjective:
    def __init__(self, frames: Dict[str, pd.DataFrame], strategy: Optional[str], symbols: Sequence[str], config: Dict[str, Any],
                 metric: str = "sharpe", start=None, end=None, min_trades: int = 5, max_drawdown: Optional[float] = None,
                 spec: Optional[Dict[str, Any]] = None, base_params: Optional[Dict[str, Any]] = None) -> None:
        if (strategy is None) == (spec is None):
            raise OptimizationError("give exactly one of strategy (registry name) or spec (rule-based strategy)")
        self.frames, self.strategy, self.symbols = frames, strategy, list(symbols)
        self.config, self.metric, self.start, self.end = config, metric, start, end
        self.min_trades, self.max_drawdown, self.spec = min_trades, max_drawdown, spec
        self.base_params = dict(base_params or {})
        self._engine: Optional[BacktestEngine] = None

    def __getstate__(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["_engine"] = None                               # engines are rebuilt in each process
        return d

    def _get_engine(self) -> BacktestEngine:
        if self._engine is None:
            md = MarketData()
            for s, df in self.frames.items():
                md.add(s, df)
            self._engine = BacktestEngine(md, PlatformConfig.from_dict(self.config))
        return self._engine

    def _strategy(self, params: Dict[str, Any]):
        merged = {**self.base_params, **params}
        if self.spec is not None:
            from ..strategy.builder import RuleBasedStrategy
            return RuleBasedStrategy(self.symbols, self.spec, **merged)
        return STRATEGIES.create(self.strategy, self.symbols, **merged)

    def __call__(self, params: Dict[str, Any]) -> float:
        res = self._get_engine().run(self._strategy(params), start=self.start, end=self.end)
        if len(res.trades) < self.min_trades:
            return -math.inf
        m = res.metrics()
        if self.max_drawdown is not None and m["max_drawdown"] < -abs(self.max_drawdown):
            return -math.inf
        if self.metric not in m:
            raise OptimizationError(f"unknown metric '{self.metric}'", details={"valid": sorted(m)})
        return float(m[self.metric])


def make_parallel_objective(data: MarketData, strategy: Optional[str], symbols: Sequence[str], config: Optional[PlatformConfig] = None,
                            **kw: Any) -> ParallelBacktestObjective:
    frames = {s: data.frame(s) for s in data.symbols}
    return ParallelBacktestObjective(frames, strategy, symbols, (config or PlatformConfig()).to_dict(), **kw)
