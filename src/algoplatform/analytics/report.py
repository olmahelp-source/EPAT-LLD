"""Tear sheets: text, JSON and a self-contained HTML report (inline SVG, no dependencies)."""
from __future__ import annotations

import html
import json
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..portfolio.portfolio import Trade
from ..stats.tests import deflated_sharpe, probabilistic_sharpe
from .metrics import drawdown_series, monthly_returns, performance_metrics, to_returns
from .trades import contribution_by, trade_statistics

_PCT = {"total_return", "cagr", "volatility", "max_drawdown", "var_95", "cvar_95", "best_bar", "worst_bar",
        "pct_positive_bars", "alpha", "tracking_error", "win_rate", "up_capture", "down_capture"}


def build_report(equity: pd.Series, trades: Sequence[Trade] = (), periods_per_year: Optional[float] = None,
                 benchmark: Optional[pd.Series] = None, n_trials: int = 1) -> Dict[str, Any]:
    """Collect every analytic into one dict (JSON-serialisable via ``to_json``)."""
    perf = performance_metrics(equity, periods_per_year, benchmark=benchmark)
    r = to_returns(equity)
    rep: Dict[str, Any] = {"performance": perf, "trades": trade_statistics(trades)}
    if len(r) >= 30:
        rep["significance"] = {
            "probabilistic_sharpe_vs_0": probabilistic_sharpe(r.values, 0.0),
            "deflated_sharpe": deflated_sharpe(r.values, max(n_trials, 1))["deflated_sharpe"],
            "n_trials": n_trials}
    if trades:
        rep["by_symbol"] = contribution_by(trades, "symbol").round(4).to_dict("index")
    return rep


def to_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))


def _fmt(k: str, v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.2%}" if k in _PCT else f"{v:,.3f}"
    return str(v)


def text_report(report: Dict[str, Any], title: str = "Strategy report") -> str:
    lines = [title, "=" * len(title)]
    for section in ("performance", "trades", "significance"):
        if section in report:
            lines += ["", section.upper()]
            lines += [f"  {k:<26}{_fmt(k, v):>14}" for k, v in report[section].items()]
    return "\n".join(lines)


def _svg_line(values: pd.Series, width: int = 760, height: int = 220, color: str = "#2563eb",
              fill: bool = False, baseline: Optional[float] = None) -> str:
    v = values.dropna()
    if len(v) < 2:
        return ""
    lo, hi = float(v.min()), float(v.max())
    if baseline is not None:
        lo, hi = min(lo, baseline), max(hi, baseline)
    span = (hi - lo) or 1.0
    xs = np.linspace(8, width - 8, len(v))
    ys = height - 8 - (v.values - lo) / span * (height - 16)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f'<polygon points="8,{height-8} {pts} {width-8},{height-8}" fill="{color}" opacity="0.15"/>' if fill else ""
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" role="img">{area}'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"/></svg>')


def html_report(equity: pd.Series, report: Dict[str, Any], title: str = "Strategy report") -> str:
    def table(d: Dict[str, Any]) -> str:
        rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(_fmt(k, v))}</td></tr>" for k, v in d.items())
        return f"<table>{rows}</table>"
    dd = drawdown_series(equity)
    mr = monthly_returns(equity)
    mr_html = mr.map(lambda x: "" if pd.isna(x) else f"{x:.1%}").to_html(border=0) if not mr.empty else ""
    parts = [f"<h1>{html.escape(title)}</h1>", "<h2>Equity</h2>", _svg_line(equity, fill=True),
             "<h2>Drawdown</h2>", _svg_line(dd, color="#dc2626", fill=True, baseline=0.0),
             "<h2>Performance</h2>", table(report["performance"]), "<h2>Trades</h2>", table(report["trades"])]
    if "significance" in report:
        parts += ["<h2>Statistical significance</h2>", table(report["significance"])]
    if mr_html:
        parts += ["<h2>Monthly returns</h2>", mr_html]
    style = ("body{font:14px system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#111}"
             "table{border-collapse:collapse;width:100%}td{padding:4px 8px;border-bottom:1px solid #e5e7eb}"
             "td:last-child{text-align:right;font-variant-numeric:tabular-nums}h2{margin-top:1.6rem}")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{style}</style></head><body>{''.join(parts)}</body></html>"
