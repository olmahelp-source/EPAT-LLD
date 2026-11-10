"""OLS regression with classical and Newey-West (HAC) standard errors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..core.exceptions import StatisticsError
from .special import t_two_sided_p


@dataclass
class OLSResult:
    coef: np.ndarray
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    r2: float
    resid: np.ndarray
    n: int
    names: Sequence[str]

    def __getitem__(self, name: str) -> float:
        return float(self.coef[list(self.names).index(name)])

    def summary(self) -> dict:
        return {n: {"coef": float(c), "se": float(s), "t": float(t), "p": float(p)}
                for n, c, s, t, p in zip(self.names, self.coef, self.se, self.t, self.p)} | {"r2": self.r2, "n": self.n}


def ols(y, X, add_const: bool = True, hac_lags: Optional[int] = None,
        names: Optional[Sequence[str]] = None) -> OLSResult:
    y = np.asarray(y, float).ravel()
    X = np.asarray(X, float)
    X = X.reshape(len(y), -1) if X.ndim == 1 else X
    if len(X) != len(y):
        raise StatisticsError("X and y length mismatch")
    cols = list(names) if names else [f"x{i}" for i in range(X.shape[1])]
    if add_const:
        X = np.column_stack([np.ones(len(y)), X])
        cols = ["const"] + cols
    n, k = X.shape
    if n <= k:
        raise StatisticsError("not enough observations for regression", details={"n": n, "k": k})
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[mask], X[mask]
    n = len(y)
    if n <= k:
        raise StatisticsError("not enough finite observations")
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    if hac_lags is None:
        s2 = resid @ resid / (n - k)
        cov = s2 * xtx_inv
    else:                                           # Newey-West sandwich
        Xu = X * resid[:, None]
        S = Xu.T @ Xu
        for lag in range(1, hac_lags + 1):
            w = 1 - lag / (hac_lags + 1)
            G = Xu[lag:].T @ Xu[:-lag]
            S += w * (G + G.T)
        cov = xtx_inv @ S @ xtx_inv * n / (n - k)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    p = np.array([t_two_sided_p(float(v), n - k) for v in t])
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid @ resid) / tss if tss > 0 else 0.0
    return OLSResult(beta, se, t, p, float(r2), resid, n, cols)
