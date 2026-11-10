"""Fund allocation across strategies, with a capital ledger."""
from __future__ import annotations

import threading
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..core.exceptions import ValidationError
from . import optimizer as opt


def apply_weight_bounds(w: np.ndarray, lo: float, hi: float, iters: int = 100) -> np.ndarray:
    """Clip to [lo, hi] and redistribute the slack proportionally; sums to 1."""
    n = len(w)
    if lo * n > 1 + 1e-12 or hi * n < 1 - 1e-12:
        raise ValidationError("infeasible weight bounds", details={"lo": lo, "hi": hi, "n": n})
    w = np.asarray(w, float).copy()
    for _ in range(iters):
        w = np.clip(w, lo, hi)
        gap = 1.0 - w.sum()
        if abs(gap) < 1e-12:
            break
        free = (w < hi - 1e-12) if gap > 0 else (w > lo + 1e-12)
        if not free.any():
            break
        w[free] += gap * w[free] / w[free].sum() if w[free].sum() > 0 else gap / free.sum()
    return w / w.sum()


class FundAllocator:
    """Splits capital across strategies and tracks what has been handed out."""

    def __init__(self, total_capital: float, reserve_pct: float = 0.05) -> None:
        if total_capital <= 0 or not 0 <= reserve_pct < 1:
            raise ValidationError("bad capital or reserve")
        self.total = float(total_capital)
        self.reserve_pct = reserve_pct
        self._alloc: Dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def investable(self) -> float:
        return self.total * (1 - self.reserve_pct)

    def weights(self, strategy_returns: Optional[pd.DataFrame] = None, names: Optional[list] = None,
                method: str = "equal", min_weight: float = 0.0, max_weight: float = 1.0,
                lookback: Optional[int] = None) -> Dict[str, float]:
        """Weights per strategy by ``method``: equal | inverse_vol | risk_parity |
        min_variance | performance (softmax of Sharpe)."""
        if strategy_returns is None:
            if not names:
                raise ValidationError("need strategy_returns or names")
            w = np.full(len(names), 1.0 / len(names))
            cols = list(names)
        else:
            df = strategy_returns.dropna()
            if lookback:
                df = df.tail(lookback)
            if len(df) < 20:
                raise ValidationError("need at least 20 observations to allocate by risk")
            cols, cov = list(df.columns), df.cov().values
            if method == "equal":
                w = opt.equal_weight(len(cols))
            elif method == "inverse_vol":
                w = opt.inverse_volatility(cov)
            elif method == "risk_parity":
                w = opt.risk_parity(cov)
            elif method == "min_variance":
                w = opt.min_variance(cov)
            elif method == "performance":
                sharpe = (df.mean() / df.std(ddof=1).replace(0, np.nan)).fillna(0.0).values
                e = np.exp(np.clip(sharpe - sharpe.max(), -30, 0))
                w = e / e.sum()
            else:
                raise ValidationError(f"unknown allocation method '{method}'")
        w = apply_weight_bounds(w, min_weight, max_weight)
        return dict(zip(cols, map(float, w)))

    def allocate(self, weights: Dict[str, float]) -> Dict[str, float]:
        if not weights or abs(sum(weights.values()) - 1) > 1e-6 or min(weights.values()) < 0:
            raise ValidationError("weights must be non-negative and sum to 1")
        with self._lock:
            self._alloc = {k: v * self.investable for k, v in weights.items()}
            return dict(self._alloc)

    def capital_for(self, strategy_id: str) -> float:
        return self._alloc.get(strategy_id, 0.0)

    def update_total(self, new_total: float) -> None:
        """Mark capital to the latest NAV, keeping weights unchanged."""
        if new_total <= 0:
            raise ValidationError("total capital must be positive")
        with self._lock:
            old_invest = self.investable
            self.total = float(new_total)
            if old_invest > 0 and self._alloc:
                scale = self.investable / old_invest
                self._alloc = {k: v * scale for k, v in self._alloc.items()}

    def needs_rebalance(self, current_values: Dict[str, float], threshold: float = 0.05) -> bool:
        """True if any strategy's share drifts more than ``threshold`` from its target share."""
        tot = sum(current_values.values())
        tgt_tot = sum(self._alloc.values())
        if tot <= 0 or tgt_tot <= 0:
            return False
        return any(abs(current_values.get(k, 0.0) / tot - v / tgt_tot) > threshold for k, v in self._alloc.items())
