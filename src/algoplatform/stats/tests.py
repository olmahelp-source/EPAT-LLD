"""Statistical tests and estimators used in strategy research and validation."""
from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from ..core.exceptions import StatisticsError
from ..utils.helpers import drop_nan, to_array
from .regression import ols
from .special import chi2_sf, norm_cdf, norm_ppf, t_two_sided_p


# ---- descriptive ---------------------------------------------------------------------
def skewness(x) -> float:
    x = drop_nan(to_array(x, min_len=3))
    s = x.std()
    return float(np.mean(((x - x.mean()) / s) ** 3)) if s > 0 else 0.0


def excess_kurtosis(x) -> float:
    x = drop_nan(to_array(x, min_len=4))
    s = x.std()
    return float(np.mean(((x - x.mean()) / s) ** 4) - 3.0) if s > 0 else 0.0


def autocorrelation(x, nlags: int = 20) -> np.ndarray:
    x = drop_nan(to_array(x, min_len=nlags + 2))
    x = x - x.mean()
    denom = float(x @ x)
    if denom == 0:
        return np.zeros(nlags + 1)
    return np.array([1.0] + [float(x[k:] @ x[:-k]) / denom for k in range(1, nlags + 1)])


def ljung_box(x, lags: int = 10) -> Dict[str, float]:
    """H0: no autocorrelation up to ``lags``. Small p => serial dependence."""
    x = drop_nan(to_array(x, min_len=lags + 2))
    n = len(x)
    ac = autocorrelation(x, lags)[1:]
    q = n * (n + 2) * float(np.sum(ac ** 2 / (n - np.arange(1, lags + 1))))
    return {"stat": q, "p_value": chi2_sf(q, lags), "lags": lags}


def jarque_bera(x) -> Dict[str, float]:
    """H0: normally distributed returns."""
    x = drop_nan(to_array(x, min_len=8))
    n = len(x)
    jb = n / 6.0 * (skewness(x) ** 2 + excess_kurtosis(x) ** 2 / 4.0)
    return {"stat": jb, "p_value": chi2_sf(jb, 2), "skew": skewness(x), "excess_kurtosis": excess_kurtosis(x)}


def t_test_mean(x, mu0: float = 0.0) -> Dict[str, float]:
    x = drop_nan(to_array(x, min_len=3))
    n = len(x)
    se = x.std(ddof=1) / math.sqrt(n)
    t = (x.mean() - mu0) / se if se > 0 else 0.0
    return {"t": float(t), "p_value": t_two_sided_p(t, n - 1), "mean": float(x.mean()), "n": n}


# ---- stationarity / mean reversion ----------------------------------------------------
# MacKinnon (2010) response-surface coefficients: crit = b0 + b1/T + b2/T^2 + b3/T^3
_ADF_CRIT = {
    "n":  {"1%": (-2.56574, -2.2358, -3.627, 0.0), "5%": (-1.94100, -0.2686, -3.365, 31.223),
           "10%": (-1.61682, 0.2656, -2.714, 25.364)},
    "c":  {"1%": (-3.43035, -6.5393, -16.786, -79.433), "5%": (-2.86154, -2.8903, -4.234, -40.040),
           "10%": (-2.56677, -1.5384, -2.809, 0.0)},
    "ct": {"1%": (-3.95877, -9.0531, -28.428, -134.155), "5%": (-3.41049, -4.3904, -9.036, -45.374),
           "10%": (-3.12705, -2.5856, -3.925, -22.380)},
}
_EG_CRIT = {"1%": (-3.89644, -10.9519, -22.527, 0.0), "5%": (-3.33613, -6.1101, -6.823, 0.0),
            "10%": (-3.04445, -4.2412, -2.720, 0.0)}


def _crit(table: Dict[str, Tuple[float, ...]], n: int) -> Dict[str, float]:
    return {k: c[0] + c[1] / n + c[2] / n ** 2 + c[3] / n ** 3 for k, c in table.items()}


def adf_test(x, maxlag: Optional[int] = None, regression: str = "c") -> Dict[str, object]:
    """Augmented Dickey-Fuller. H0: unit root. ``reject_5pct`` True => stationary evidence.
    Lag order picked by AIC up to ``maxlag`` (Schwert rule by default)."""
    if regression not in _ADF_CRIT:
        raise StatisticsError("regression must be 'n', 'c' or 'ct'")
    y = drop_nan(to_array(x, min_len=20))
    n = len(y)
    maxlag = maxlag if maxlag is not None else int(math.ceil(12 * (n / 100.0) ** 0.25))
    maxlag = max(0, min(maxlag, n // 3 - 2))
    dy = np.diff(y)

    def fit(lag: int):
        start = lag
        Y = dy[start:]
        cols = [y[start:-1]]
        for i in range(1, lag + 1):
            cols.append(dy[start - i:len(dy) - i])
        X = np.column_stack(cols)
        if regression in ("c", "ct"):
            X = np.column_stack([X, np.ones(len(Y))])
        if regression == "ct":
            X = np.column_stack([X, np.arange(len(Y), dtype=float)])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ beta
        sse = float(res @ res)
        k = X.shape[1]
        nobs = len(Y)
        aic = nobs * math.log(max(sse / nobs, 1e-300)) + 2 * k
        cov = sse / (nobs - k) * np.linalg.pinv(X.T @ X)
        return beta[0] / math.sqrt(max(cov[0, 0], 1e-300)), aic, nobs, beta[0]

    best = min((fit(l) + (l,) for l in range(maxlag + 1)), key=lambda r: (r[1] / r[2], r[4]))
    stat, _, nobs, gamma, lag = best
    crit = _crit(_ADF_CRIT[regression], nobs)
    return {"stat": float(stat), "critical_values": crit, "lags": lag, "nobs": nobs,
            "reject_5pct": bool(stat < crit["5%"]), "reject_1pct": bool(stat < crit["1%"])}


def kpss_test(x, lags: Optional[int] = None) -> Dict[str, object]:
    """KPSS level test. H0: stationary. ``reject_5pct`` True => NOT stationary."""
    y = drop_nan(to_array(x, min_len=20))
    n = len(y)
    e = y - y.mean()
    s = np.cumsum(e)
    lags = lags if lags is not None else int(math.ceil(12 * (n / 100.0) ** 0.25))
    s2 = float(e @ e) / n
    for k in range(1, lags + 1):
        s2 += 2 * (1 - k / (lags + 1)) * float(e[k:] @ e[:-k]) / n
    stat = float(s @ s) / (n ** 2 * s2) if s2 > 0 else 0.0
    crit = {"10%": 0.347, "5%": 0.463, "1%": 0.739}
    return {"stat": stat, "critical_values": crit, "reject_5pct": stat > crit["5%"], "lags": lags}


def engle_granger(y, x) -> Dict[str, object]:
    """Two-step cointegration test of y on x. Returns hedge ratio, ADF on residuals and half-life."""
    y, x = to_array(y, min_len=30), to_array(x, min_len=30)
    if len(x) != len(y):
        raise StatisticsError("series must have equal length")
    fit = ols(y, x, names=["x"])
    resid = fit.resid
    adf = adf_test(resid, regression="n")
    crit = _crit(_EG_CRIT, adf["nobs"])
    return {"hedge_ratio": float(fit["x"]), "intercept": float(fit["const"]), "stat": adf["stat"],
            "critical_values": crit, "cointegrated_5pct": bool(adf["stat"] < crit["5%"]),
            "half_life": ou_half_life(resid)["half_life"], "residuals": resid}


def ou_half_life(x) -> Dict[str, float]:
    """Fit dx = theta (mu - x) dt + noise by OLS; returns half-life in bars (inf if not mean reverting)."""
    x = drop_nan(to_array(x, min_len=10))
    fit = ols(np.diff(x), x[:-1], names=["lag"])
    b, a = fit["lag"], fit["const"]
    if b >= 0:
        return {"half_life": math.inf, "theta": 0.0, "mu": float(x.mean()), "sigma": float(np.std(fit.resid))}
    theta = -b
    return {"half_life": math.log(2) / -math.log(1 + b) if b > -1 else 0.0, "theta": float(theta),
            "mu": float(a / theta), "sigma": float(np.std(fit.resid, ddof=1))}


def hurst_exponent(x, max_lag: int = 100) -> float:
    """~0.5 random walk, <0.5 mean reverting, >0.5 trending (variance-of-differences method)."""
    x = drop_nan(to_array(x, min_len=30))
    max_lag = min(max_lag, len(x) // 2)
    lags = np.arange(2, max_lag)
    tau = np.array([np.std(x[l:] - x[:-l]) for l in lags])
    ok = tau > 0
    if ok.sum() < 5:
        raise StatisticsError("series too short or constant for Hurst estimate")
    return float(np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)[0])


def variance_ratio(returns, q: int = 5) -> Dict[str, float]:
    """Lo-MacKinlay variance ratio with heteroscedasticity-robust z. VR<1 mean reversion."""
    r = drop_nan(to_array(returns, min_len=2 * q + 2))
    n = len(r)
    mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / (n - 1)
    rq = np.convolve(r, np.ones(q), mode="valid")
    varq = np.sum((rq - q * mu) ** 2) / (len(rq) * q) if len(rq) else 0.0
    vr = varq / var1 if var1 > 0 else 1.0
    d = (r - mu) ** 2
    theta = 0.0
    for j in range(1, q):
        delta = np.sum(d[j:] * d[:-j]) / np.sum(d) ** 2 * n
        theta += (2 * (q - j) / q) ** 2 * delta
    z = (vr - 1) / math.sqrt(theta / n) if theta > 0 else 0.0
    return {"vr": float(vr), "z": float(z), "p_value": float(2 * (1 - norm_cdf(abs(z)))), "q": q}


# ---- resampling ----------------------------------------------------------------------------
def bootstrap_ci(x, stat: Callable[[np.ndarray], float], n_boot: int = 2000, alpha: float = 0.05,
                 block: Optional[int] = None, seed: Optional[int] = 0) -> Dict[str, float]:
    """Percentile bootstrap CI. ``block`` > 1 uses the stationary bootstrap (dependent data)."""
    x = drop_nan(to_array(x, min_len=5))
    n = len(x)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        if block and block > 1:
            idx = np.empty(n, dtype=int)
            idx[0] = rng.integers(n)
            jumps = rng.random(n) < 1.0 / block
            new = rng.integers(n, size=n)
            for i in range(1, n):
                idx[i] = new[i] if jumps[i] else (idx[i - 1] + 1) % n
        else:
            idx = rng.integers(n, size=n)
        out[b] = stat(x[idx])
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(stat(x)), "low": float(lo), "high": float(hi), "se": float(out.std(ddof=1))}


def monte_carlo_trades(trade_pnls: Sequence[float], initial_equity: float = 100_000.0, n_sims: int = 2000,
                       seed: Optional[int] = 0) -> Dict[str, object]:
    """Reshuffle trade order to see the spread of max drawdown and final equity (path dependence)."""
    pnl = to_array(trade_pnls, "trade_pnls", min_len=5)
    rng = np.random.default_rng(seed)
    finals, dds = np.empty(n_sims), np.empty(n_sims)
    for i in range(n_sims):
        eq = initial_equity + np.cumsum(rng.permutation(pnl))
        peak = np.maximum.accumulate(np.concatenate([[initial_equity], eq]))[1:]
        dds[i] = float(np.max((peak - eq) / peak))
        finals[i] = eq[-1]
    return {"max_dd_p50": float(np.percentile(dds, 50)), "max_dd_p95": float(np.percentile(dds, 95)),
            "max_dd_p99": float(np.percentile(dds, 99)), "final_p05": float(np.percentile(finals, 5)),
            "final_p50": float(np.percentile(finals, 50)), "prob_loss": float(np.mean(finals < initial_equity)),
            "drawdowns": dds, "finals": finals}


# ---- Sharpe ratio inference -------------------------------------------------------------------
def sharpe_ratio(returns, periods_per_year: float = 252.0, rf: float = 0.0) -> float:
    r = drop_nan(to_array(returns, min_len=2)) - rf / periods_per_year
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0


def sharpe_standard_error(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Mertens (2002): SE of a per-period Sharpe under non-normal returns (kurt is NOT excess)."""
    v = (1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2) / (n - 1)
    return math.sqrt(max(v, 0.0))


def probabilistic_sharpe(returns, sr_benchmark: float = 0.0) -> float:
    """PSR: probability that the true (per-period) Sharpe exceeds ``sr_benchmark`` (per-period)."""
    r = drop_nan(to_array(returns, min_len=10))
    sr = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else 0.0
    se = sharpe_standard_error(sr, len(r), skewness(r), excess_kurtosis(r) + 3.0)
    return float(norm_cdf((sr - sr_benchmark) / se)) if se > 0 else float(sr > sr_benchmark)


def expected_max_sharpe(n_trials: int, var_trials: float) -> float:
    """Expected maximum Sharpe among ``n_trials`` independent zero-skill trials."""
    if n_trials < 2:
        return 0.0
    g = 0.5772156649015329
    return math.sqrt(var_trials) * ((1 - g) * norm_ppf(1 - 1.0 / n_trials) + g * norm_ppf(1 - 1.0 / (n_trials * math.e)))


def deflated_sharpe(returns, n_trials: int, var_trials: Optional[float] = None) -> Dict[str, float]:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado): PSR against the best-of-N benchmark.

    ``var_trials`` is the variance of per-period Sharpe ratios across all tried configurations
    (defaults to the estimator's own variance if unknown). A low value means the "best" backtest is
    plausibly just the luckiest of many."""
    r = drop_nan(to_array(returns, min_len=10))
    n = len(r)
    sr = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else 0.0
    if var_trials is None:
        var_trials = sharpe_standard_error(0.0, n) ** 2
    bench = expected_max_sharpe(n_trials, var_trials)
    return {"deflated_sharpe": probabilistic_sharpe(r, bench), "benchmark_sharpe": bench,
            "sharpe_per_period": float(sr), "n_trials": n_trials}


def min_track_record_length(sr: float, sr_benchmark: float = 0.0, skew: float = 0.0, kurt: float = 3.0,
                            confidence: float = 0.95) -> float:
    """Observations needed for the Sharpe (per-period) to beat ``sr_benchmark`` at ``confidence``."""
    if sr <= sr_benchmark:
        return math.inf
    z = norm_ppf(confidence)
    return 1 + (1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2) * (z / (sr - sr_benchmark)) ** 2


def lo_adjusted_sharpe(returns, periods_per_year: float = 252.0, q: int = 10) -> float:
    """Annualised Sharpe with Lo (2002) autocorrelation correction instead of sqrt(T) scaling."""
    r = drop_nan(to_array(returns, min_len=q + 2))
    sr = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else 0.0
    rho = autocorrelation(r, q)[1:]
    eta = q + 2 * sum((q - k) * rho[k - 1] for k in range(1, q))
    return float(sr * q / math.sqrt(eta)) * math.sqrt(periods_per_year / q) if eta > 0 else 0.0


# ---- multiple testing ----------------------------------------------------------------------------
def bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> Dict[str, np.ndarray]:
    p = np.asarray(p_values, float)
    adj = np.minimum(p * len(p), 1.0)
    return {"adjusted": adj, "reject": adj <= alpha + 1e-12}


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """Controls the false discovery rate across many strategy/parameter tests."""
    p = np.asarray(p_values, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.minimum(adj_sorted, 1.0)
    return {"adjusted": adj, "reject": adj <= alpha + 1e-12}
