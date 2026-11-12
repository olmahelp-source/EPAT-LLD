"""Strategy result analysis: from "here are the numbers" to "should this be trusted with capital".

``metrics`` and ``trades`` describe a result. This module interrogates it:

* :func:`bootstrap_metrics`      how uncertain are Sharpe, CAGR and drawdown, given this one sample path?
* :func:`drawdown_probability`   how likely is a drawdown of X over the next year if the return process is as measured?
* :func:`rolling_stability`      did it earn steadily, or in one lucky stretch?
* :func:`regime_breakdown`       does it depend on one kind of market (trend, calm, crisis)?
* :func:`trade_concentration`    does the whole profit come from a handful of trades?
* :func:`compare_strategies`     side-by-side table plus correlations (are the "different" strategies the same bet?)
* :func:`evaluate_strategy`      all of the above folded into a pass/fail scorecard with reasons and a grade.

The scorecard is deliberately conservative: it is the gate the governance layer uses before a strategy may be promoted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.exceptions import ValidationError
from ..stats.tests import deflated_sharpe, probabilistic_sharpe
from .metrics import infer_periods_per_year, max_drawdown


def _returns(returns: Any, min_len: int = 20) -> pd.Series:
    r = returns if isinstance(returns, pd.Series) else pd.Series(np.asarray(returns, float))
    r = r[np.isfinite(r.to_numpy(float))]
    if len(r) < min_len:
        raise ValidationError(f"need at least {min_len} returns to analyse a result", details={"n": len(r)})
    return r.astype(float)


def _periods(r: pd.Series, periods: Optional[float]) -> float:
    if periods:
        return float(periods)
    return infer_periods_per_year(r.index) if isinstance(r.index, pd.DatetimeIndex) else 252.0


def _stats(r: np.ndarray, ppy: float) -> Dict[str, float]:
    eq = np.cumprod(1.0 + r)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))
    years = len(r) / ppy
    return {"sharpe": float(r.mean() / sd * math.sqrt(ppy)) if sd > 0 else 0.0,
            "cagr": float(eq[-1] ** (1.0 / years) - 1.0) if eq[-1] > 0 and years > 0 else -1.0,
            "max_drawdown": float((np.concatenate([[1.0], eq]) / peak - 1.0).min()), "volatility": float(sd * math.sqrt(ppy))}


def _block_indices(n: int, length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap indices: resampling whole blocks keeps short-term autocorrelation and volatility
    clustering, which an i.i.d. shuffle would destroy (and with them, realistic drawdowns)."""
    starts = rng.integers(0, n, size=int(math.ceil(length / block)))
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:length]


def bootstrap_metrics(returns: Any, n_boot: int = 1000, block: int = 10, confidence: float = 0.90, periods: Optional[float] = None,
                      seed: Optional[int] = 0) -> Dict[str, Dict[str, float]]:
    """Confidence intervals for Sharpe, CAGR, max drawdown and volatility from a block bootstrap of the return series.
    ``{"sharpe": {"low", "median", "high", "observed"}, ...}``. If the Sharpe interval includes zero, the backtest has
    not shown skill, whatever the point estimate."""
    r = _returns(returns)
    if n_boot < 50 or block < 1 or not 0.5 < confidence < 1:
        raise ValidationError("need n_boot >= 50, block >= 1, 0.5 < confidence < 1")
    ppy, x, rng = _periods(r, periods), r.to_numpy(float), np.random.default_rng(seed)
    draws = [_stats(x[_block_indices(len(x), len(x), block, rng)], ppy) for _ in range(n_boot)]
    observed = _stats(x, ppy)
    lo, hi = (1 - confidence) / 2 * 100, (1 + confidence) / 2 * 100
    out = {}
    for k in observed:
        v = np.array([d[k] for d in draws])
        out[k] = {"low": float(np.percentile(v, lo)), "median": float(np.median(v)), "high": float(np.percentile(v, hi)), "observed": observed[k]}
    return out


def drawdown_probability(returns: Any, threshold: float = 0.20, horizon: Optional[int] = None, n_sims: int = 2000, block: int = 10,
                         periods: Optional[float] = None, seed: Optional[int] = 0) -> Dict[str, float]:
    """Chance of a peak-to-trough loss of at least ``threshold`` within ``horizon`` bars (default: one year), simulated
    by block-resampling the observed returns. Also returns the median and 95th-percentile worst drawdown. The backtest's
    own max drawdown is one draw from this distribution, and usually an optimistic one."""
    r = _returns(returns)
    if not 0 < threshold < 1 or n_sims < 100:
        raise ValidationError("need 0 < threshold < 1 and n_sims >= 100")
    ppy = _periods(r, periods)
    horizon = int(horizon or round(ppy))
    x, rng = r.to_numpy(float), np.random.default_rng(seed)
    worst = np.empty(n_sims)
    for i in range(n_sims):
        eq = np.concatenate([[1.0], np.cumprod(1.0 + x[_block_indices(len(x), horizon, block, rng)])])
        worst[i] = (eq / np.maximum.accumulate(eq) - 1.0).min()
    return {"probability": float((worst <= -threshold).mean()), "median_drawdown": float(np.median(worst)),
            "p95_drawdown": float(np.percentile(worst, 5)), "threshold": threshold, "horizon": horizon}


def rolling_stability(returns: Any, window: int = 126, periods: Optional[float] = None) -> Dict[str, float]:
    """Consistency through time. ``positive_windows``: share of rolling windows that made money. ``equity_r2``: how
    close log equity is to a straight line (1 = perfectly steady compounding). ``worst_window`` / ``best_window``:
    extremes of the rolling return. ``sharpe_dispersion``: standard deviation of the rolling Sharpe."""
    r = _returns(returns)
    if window < 10 or len(r) < 2 * window:
        raise ValidationError("need window >= 10 and at least two windows of returns", details={"n": len(r), "window": window})
    ppy = _periods(r, periods)
    log_r = np.log1p(r.clip(lower=-0.999999))
    roll = np.expm1(log_r.rolling(window).sum()).dropna()
    sd = r.rolling(window).std()
    rs = (r.rolling(window).mean() / sd.where(sd > 0) * math.sqrt(ppy)).dropna()
    y = log_r.cumsum().to_numpy()
    t = np.arange(len(y), dtype=float)
    corr = np.corrcoef(t, y)[0, 1] if y.std() > 0 else 0.0
    return {"positive_windows": float((roll > 0).mean()), "worst_window": float(roll.min()), "best_window": float(roll.max()),
            "equity_r2": float(corr ** 2 if corr > 0 else 0.0), "sharpe_dispersion": float(rs.std()) if len(rs) > 1 else 0.0,
            "window": window}


def regime_breakdown(returns: Any, regimes: Any, periods: Optional[float] = None) -> pd.DataFrame:
    """Performance within each market regime. ``regimes`` holds one label per return (e.g. from
    ``ai.RegimeDetector.regimes`` or ``volatility.VolRegime.classify``; use causal labels). ``pnl_share`` shows how much
    of the total (log) return each regime contributed: a strategy with 90% in one regime is a bet on that regime."""
    r = _returns(returns, 10)
    if not isinstance(regimes, pd.Series):
        values = np.asarray(regimes)
        if len(values) != len(r):
            raise ValidationError("regimes must provide one label per return", details={"labels": len(values), "returns": len(r)})
        lab = pd.Series(values, index=r.index)
    else:
        lab = regimes.reindex(r.index)
    if lab.isna().all():
        raise ValidationError("no regime label matches the returns' index")
    ppy = _periods(r, periods)
    total = float(np.log1p(r.clip(lower=-0.999999)).sum())
    rows = []
    for name, grp in r.groupby(lab):
        sd = grp.std(ddof=1) if len(grp) > 1 else 0.0
        rows.append({"regime": name, "bars": len(grp), "share_of_time": len(grp) / len(r), "ann_return": float(grp.mean() * ppy),
                     "ann_vol": float(sd * math.sqrt(ppy)), "sharpe": float(grp.mean() / sd * math.sqrt(ppy)) if sd > 0 else 0.0,
                     "hit_rate": float((grp > 0).mean()),
                     "pnl_share": float(np.log1p(grp.clip(lower=-0.999999)).sum() / total) if abs(total) > 1e-12 else float("nan")})
    return pd.DataFrame(rows).set_index("regime")


def trade_concentration(trade_pnls: Sequence[float], top: int = 5) -> Dict[str, float]:
    """How much the result leans on its best trades. ``top_share``: the ``top`` best trades' profit as a share of net
    profit (above 1 means everything else lost money). ``pnl_without_top``: net profit with those trades removed."""
    p = np.asarray([x for x in trade_pnls if np.isfinite(x)], float)
    if len(p) == 0:
        return {"trades": 0, "net": 0.0, "top_share": float("nan"), "pnl_without_top": 0.0, "profit_factor": float("nan")}
    net = float(p.sum())
    best = np.sort(p)[::-1][:top]
    gains, losses = float(p[p > 0].sum()), float(-p[p < 0].sum())
    return {"trades": int(len(p)), "net": net, "top_share": float(best.sum() / net) if net > 0 else float("inf"),
            "pnl_without_top": float(net - best.sum()), "profit_factor": gains / losses if losses > 0 else float("inf")}


def compare_strategies(returns: Mapping[str, Any], periods: Optional[float] = None) -> pd.DataFrame:
    """League table (best Sharpe first) with each strategy's deflated Sharpe given that this many candidates were
    compared. ``attrs["correlation"]`` holds the return correlation matrix; ``avg_correlation`` per strategy shows which
    ones are near-duplicates of the rest."""
    if len(returns) < 2:
        raise ValidationError("need at least two strategies to compare")
    series = {k: _returns(v) for k, v in returns.items()}
    frame = pd.DataFrame(series)
    corr = frame.corr()
    rows = []
    for name, r in series.items():
        s = _stats(r.to_numpy(float), _periods(r, periods))
        others = corr[name].drop(name)
        rows.append({"strategy": name, **s, "calmar": s["cagr"] / abs(s["max_drawdown"]) if s["max_drawdown"] < 0 else float("inf"),
                     "deflated_sharpe": deflated_sharpe(r.to_numpy(float), len(series))["deflated_sharpe"],
                     "avg_correlation": float(others.mean()) if len(others) else float("nan")})
    out = pd.DataFrame(rows).set_index("strategy").sort_values("sharpe", ascending=False)
    out.attrs["correlation"] = corr
    return out


# ---- scorecard -----------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluationLimits:
    """What a result must show before it is believed. Defaults are a sensible floor, not a guarantee of anything."""
    min_years: float = 1.0
    min_trades: int = 30
    min_sharpe: float = 0.5
    min_skill_probability: float = 0.95      # PSR, or deflated Sharpe when several configurations were tried
    max_drawdown: float = 0.25
    max_drawdown_probability: float = 0.25   # chance of breaching max_drawdown over the next year
    min_positive_windows: float = 0.60
    max_top_trade_share: float = 0.60
    min_profit_factor: float = 1.2


@dataclass
class Check:
    name: str
    passed: bool
    value: float
    limit: float
    detail: str


@dataclass
class StrategyVerdict:
    checks: List[Check]
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def score(self) -> float:
        return sum(c.passed for c in self.checks) / len(self.checks) if self.checks else 0.0

    @property
    def grade(self) -> str:
        s = self.score
        return "A" if s == 1 else "B" if s >= 0.85 else "C" if s >= 0.7 else "D" if s >= 0.5 else "F"

    @property
    def failures(self) -> List[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if not c.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {"passed": self.passed, "grade": self.grade, "score": round(self.score, 3), "failures": self.failures,
                "checks": [{"name": c.name, "passed": c.passed, "value": c.value, "limit": c.limit, "detail": c.detail} for c in self.checks],
                "metrics": self.metrics}


def evaluate_strategy(returns: Any, trade_pnls: Optional[Sequence[float]] = None, n_trials: int = 1,
                      limits: Optional[EvaluationLimits] = None, periods: Optional[float] = None, seed: Optional[int] = 0) -> StrategyVerdict:
    """Scorecard for a backtest or paper-trading track record.

    ``n_trials`` is the number of parameter sets / variants that were tried to arrive at this one. It matters more than
    any other input: with it, skill is judged by the deflated Sharpe ratio (the bar rises with every extra thing tried).
    Being honest about it is the user's job; the platform's optimisers report it as ``OptimizationResult.n_trials``.
    A check that cannot be computed (too little data) counts as failed, never as passed."""
    lim = limits or EvaluationLimits()
    if n_trials < 1:
        raise ValidationError("n_trials must be >= 1")
    r = _returns(returns)
    ppy = _periods(r, periods)
    x = r.to_numpy(float)
    s = _stats(x, ppy)
    years = len(x) / ppy
    checks: List[Check] = []

    def add(name: str, ok: bool, value: float, limit: float, detail: str) -> None:
        checks.append(Check(name, bool(ok), float(value), float(limit), detail))
    add("track_record", years >= lim.min_years, years, lim.min_years, f"{years:.1f} years of data (need {lim.min_years:g})")
    add("sharpe", s["sharpe"] >= lim.min_sharpe, s["sharpe"], lim.min_sharpe, f"Sharpe {s['sharpe']:.2f} (need {lim.min_sharpe:g})")
    skill = deflated_sharpe(x, n_trials)["deflated_sharpe"] if n_trials > 1 else probabilistic_sharpe(x)
    label = f"deflated Sharpe over {n_trials} trials" if n_trials > 1 else "probabilistic Sharpe"
    add("skill_probability", skill >= lim.min_skill_probability, skill, lim.min_skill_probability,
        f"{label} {skill:.2f} (need {lim.min_skill_probability:g})")
    add("max_drawdown", abs(s["max_drawdown"]) <= lim.max_drawdown, abs(s["max_drawdown"]), lim.max_drawdown,
        f"max drawdown {abs(s['max_drawdown']):.1%} (limit {lim.max_drawdown:.0%})")
    ddp = drawdown_probability(r, lim.max_drawdown, periods=ppy, n_sims=1000, seed=seed)
    add("drawdown_risk", ddp["probability"] <= lim.max_drawdown_probability, ddp["probability"], lim.max_drawdown_probability,
        f"{ddp['probability']:.0%} chance of a {lim.max_drawdown:.0%} drawdown within a year (limit {lim.max_drawdown_probability:.0%})")
    window = int(min(max(ppy / 2, 20), len(x) // 2))
    stability: Dict[str, float] = {}
    try:
        stability = rolling_stability(r, window, ppy)
        add("consistency", stability["positive_windows"] >= lim.min_positive_windows, stability["positive_windows"], lim.min_positive_windows,
            f"{stability['positive_windows']:.0%} of {window}-bar windows profitable (need {lim.min_positive_windows:.0%})")
    except ValidationError:
        add("consistency", False, float("nan"), lim.min_positive_windows, "not enough data to measure consistency")
    conc: Dict[str, float] = {}
    if trade_pnls is not None:
        conc = trade_concentration(trade_pnls)
        add("trade_count", conc["trades"] >= lim.min_trades, conc["trades"], lim.min_trades, f"{conc['trades']} trades (need {lim.min_trades})")
        pf = conc["profit_factor"]
        add("profit_factor", bool(np.isfinite(pf) and pf >= lim.min_profit_factor) or pf == float("inf"), pf if np.isfinite(pf) else 0.0,
            lim.min_profit_factor, f"profit factor {pf:.2f} (need {lim.min_profit_factor:g})")
        top = conc["top_share"]
        add("trade_concentration", bool(np.isfinite(top) and top <= lim.max_top_trade_share), top if np.isfinite(top) else 0.0,
            lim.max_top_trade_share, f"best 5 trades are {top:.0%} of net profit (limit {lim.max_top_trade_share:.0%})"
            if np.isfinite(top) else "no net profit to attribute")
    return StrategyVerdict(checks, {**s, "years": years, "skill_probability": skill, "n_trials": n_trials, "drawdown_risk": ddp,
                                    "stability": stability, "trades": conc})


def evaluate_backtest(result: Any, n_trials: int = 1, limits: Optional[EvaluationLimits] = None) -> StrategyVerdict:
    """Scorecard for a ``BacktestResult`` (anything with ``returns`` and ``trades`` whose items have ``pnl``)."""
    return evaluate_strategy(result.returns, [t.pnl for t in result.trades], n_trials, limits)


__all__ = ["Check", "EvaluationLimits", "StrategyVerdict", "bootstrap_metrics", "compare_strategies", "drawdown_probability",
           "evaluate_backtest", "evaluate_strategy", "max_drawdown", "regime_breakdown", "rolling_stability", "trade_concentration"]
