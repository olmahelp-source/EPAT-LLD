"""Long-run price relationships: hedge ratios, spreads and a scan for cointegrated pairs."""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from ..core.exceptions import PlatformError, StatisticsError
from ..stats.tests import engle_granger
from .correlation import _aligned


def hedge_ratio(y, x, method: str = "ols") -> float:
    """Units of x to short per unit of y. ``tls`` (total least squares) treats both series as noisy, so the answer does
    not depend on which one is called y; OLS does."""
    y, x = _aligned(y, x, 30)
    if method == "ols":
        return float(np.cov(y, x)[0, 1] / x.var(ddof=1))
    if method == "tls":
        c = np.cov(np.vstack([x, y]))
        vals, vecs = np.linalg.eigh(c)
        v = vecs[:, int(np.argmax(vals))]                       # principal axis of the (x, y) cloud
        if abs(v[0]) < 1e-12:
            raise StatisticsError("x has no variation along the principal axis")
        return float(v[1] / v[0])
    raise StatisticsError(f"unknown hedge ratio method '{method}'")


def spread(y, x, ratio: Optional[float] = None) -> np.ndarray:
    y, x = np.asarray(y, float), np.asarray(x, float)
    return y - (hedge_ratio(y, x) if ratio is None else ratio) * x


def spread_zscore(y, x, window: int = 60, ratio: Optional[float] = None) -> np.ndarray:
    """Rolling z-score of the spread (causal: the mean and deviation at t use the window ending at t)."""
    s = pd.Series(spread(y, x, ratio))
    sd = s.rolling(window).std()
    return ((s - s.rolling(window).mean()) / sd.where(sd > 0)).to_numpy()


def find_cointegrated_pairs(prices: pd.DataFrame, level: str = "5%", min_half_life: float = 1.0, max_half_life: float = 60.0,
                            use_log: bool = True) -> pd.DataFrame:
    """Engle-Granger test on every pair (both directions, the stronger kept), filtered to spreads that mean-revert at a
    tradable speed. ``attrs["expected_false_positives"]`` reminds you that testing many pairs at 5% finds 5% by luck:
    treat the output as candidates to validate out of sample, not as discoveries. Pairs that could not be tested
    (a constant series, a degenerate regression) are listed in ``attrs["skipped"]`` instead of stopping the scan."""
    if level not in ("1%", "5%", "10%"):
        raise StatisticsError("level must be '1%', '5%' or '10%'")
    data: pd.DataFrame = np.log(prices) if use_log else prices          # type: ignore[assignment]
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    tested = 0
    for a, b in combinations(prices.columns, 2):
        pair = data[[a, b]].dropna()
        if len(pair) < 60:
            continue
        if pair[a].std() == 0 or pair[b].std() == 0:               # a series that never moves cannot be tested
            skipped.append(f"{a}/{b}: constant series")
            continue
        tested += 1
        best: Optional[Tuple[str, str, Dict[str, Any]]] = None
        for y, x in ((a, b), (b, a)):
            try:
                res = cast(Dict[str, Any], engle_granger(pair[y].to_numpy(), pair[x].to_numpy()))
            except (PlatformError, ArithmeticError, np.linalg.LinAlgError) as exc:     # one degenerate pair must not end the scan
                skipped.append(f"{y}/{x}: {type(exc).__name__}")
                continue
            if best is None or res["stat"] < best[2]["stat"]:
                best = (y, x, res)
        if best is None:
            continue
        y, x, res = best
        hl, stat, critical = float(res["half_life"]), float(res["stat"]), float(res["critical_values"][level])
        if stat < critical and min_half_life <= hl <= max_half_life:
            resid = np.asarray(res["residuals"])
            rows.append({"y": y, "x": x, "stat": stat, "critical": critical,
                         "hedge_ratio": float(res["hedge_ratio"]), "half_life": float(hl),
                         "zscore": float((resid[-1] - resid.mean()) / resid.std(ddof=1)) if resid.std(ddof=1) > 0 else 0.0, "n": len(pair)})
    out = pd.DataFrame(rows, columns=["y", "x", "stat", "critical", "hedge_ratio", "half_life", "zscore", "n"]).sort_values("stat").reset_index(drop=True)
    out.attrs.update(tested=tested, expected_false_positives=tested * float(level.rstrip("%")) / 100.0, skipped=skipped)
    return out
