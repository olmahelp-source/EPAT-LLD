"""Portfolio and risk: the multi-strategy portfolio, margin, multi-currency exposure.

Every route checks the caller's permission on the server (through the role desks or ``UserManager.authorize``); hiding a
page in the browser is a convenience, not the control."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import pandas as pd

from ....validation.analytics.metrics import drawdown_series
from ....validation.analytics.trades import contribution_by
from ....validation.backtest import BacktestEngine
from ....foundation.core.enums import Permission
from ....foundation.core.exceptions import PlatformError, ValidationError
from ....capital.portfolio.allocation import FundAllocator
from ....capital.portfolio.currency import exposure_by_currency, fx_pnl_attribution, FxRates, hedge_notionals
from ....capital.risk.margin import margin_report, MarginPosition, position_margin
from ....strategies.strategy.base import STRATEGIES
from ....access.users.manager import UserManager

from ..common import series_points, SLOW


MAX_POSITIONS = 30


def _num(row: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    v = row.get(key, default)
    if v is None or (isinstance(v, str) and not v.strip()):
        return default
    return float(v)


def _margin_position(row: Dict[str, Any]) -> MarginPosition:
    kind = str(row.get("asset_class") or "equity").lower()
    days = _num(row, "days")
    return MarginPosition(symbol=str(row.get("symbol") or "?").strip().upper()[:16], quantity=float(row["quantity"]), price=float(row["price"]),
                          asset_class=kind, multiplier=float(_num(row, "multiplier", 1.0) or 1.0),
                          margin_pct=float(_num(row, "margin_pct", 1.0) or 1.0), underlying=(str(row.get("underlying") or "").strip().upper() or None),
                          underlying_price=_num(row, "underlying_price"), strike=_num(row, "strike"),
                          right=(str(row.get("right") or "").lower() or None), years_to_expiry=None if days is None else days / 365.0,
                          volatility=_num(row, "volatility"))


def register(app: Any) -> None:
    r, P = app.route, app.platform

    @r("POST", "/api/portfolio")
    def portfolio(u, b):
        UserManager.authorize(u, Permission.RUN_BACKTEST)
        items = b.get("strategies") or []
        if not 1 <= len(items) <= 5:
            raise ValidationError("choose 1-5 strategies")
        pcts = [float(it.get("pct", 0)) for it in items]
        if min(pcts) <= 0 or sum(pcts) > 100.0 + 1e-9:
            raise ValidationError("each allocation must be > 0% and the total at most 100%")
        if any(it["name"] in SLOW for it in items) and len(items) > 2:
            raise ValidationError("model-based strategies are slow: combine at most 2 strategies")
        md, symbols = app._data(b, u)

        def build(i: int, it: Dict[str, Any]):
            app._check_strategy(it["name"])
            return STRATEGIES.create(it["name"], app._symbols_for(it["name"], symbols), strategy_id=f"{it['name']}#{i + 1}",
                                     **(it.get("params") or {}))
        combined = BacktestEngine(md, P.config).run([(build(i, it), pct / 100.0) for i, (it, pct) in enumerate(zip(items, pcts))])
        alone = {f"{it['name']}#{i + 1}": BacktestEngine(md, P.config).run(build(i, it)).returns for i, it in enumerate(items)}
        df = pd.DataFrame(alone).dropna()
        suggested, corr = {}, None
        if len(items) >= 2 and len(df) >= 20:
            corr = df.corr().round(3).to_dict()
            for m in ("equal", "inverse_vol", "risk_parity", "min_variance"):
                try:
                    suggested[m] = {k: round(v, 4) for k, v in FundAllocator(1.0).weights(df, method=m).items()}
                except PlatformError:
                    pass
        trades = combined.trades_frame()
        contrib = contribution_by(combined.trades, "strategy_id").reset_index().round(4).to_dict("records") if len(trades) else []
        P.audit.record(u.username, "portfolio.run", strategies=[it["name"] for it in items])
        return {"metrics": combined.metrics(), "equity": series_points(combined.equity),
                "drawdown": series_points(drawdown_series(combined.equity)), "contributions": contrib,
                "standalone": {k: {"total_return": float((1 + v).prod() - 1), "sharpe": float(v.mean() / v.std() * math.sqrt(252)) if v.std() > 0 else 0.0}
                               for k, v in alone.items()},
                "correlation": corr, "suggested_weights": suggested, "risk": app._risk_measures(combined.returns),
                "allocation": {f"{it['name']}#{i + 1}": pct for i, (it, pct) in enumerate(zip(items, pcts))}}

    @r("POST", "/api/risk/margin")
    def margin(u, b):
        UserManager.authorize(u, Permission.VIEW_RISK)
        if b.get("source") == "session":
            sess = app.sessions.get(u.user_id)
            if sess is None:
                raise ValidationError("no open paper-trading session: open one on the Paper trading page, or enter positions")
            with sess.lock:
                pf = sess.engine.portfolio
                positions = [MarginPosition(s, p.quantity, p.last_price, "equity", p.multiplier) for s, p in pf.open_positions().items()]
                equity = float(pf.equity)
        else:
            rows = b.get("positions") or []
            if not 1 <= len(rows) <= MAX_POSITIONS:
                raise ValidationError(f"enter 1 to {MAX_POSITIONS} positions")
            positions, problems = [], []
            for i, row in enumerate(rows):                          # check every row, then refuse with all problems listed:
                try:                                                # margin on a partial book would understate the requirement
                    positions.append(_margin_position(row))
                except (ValidationError, KeyError, TypeError, ValueError) as exc:
                    problems.append(f"row {i + 1} ({row.get('symbol', '?')}): {getattr(exc, 'message', exc)}")
            if problems:
                raise ValidationError("; ".join(problems[:5]), details={"rows": problems})
            equity = float(b.get("equity", 100_000))
        if not positions:
            raise ValidationError("the session has no open positions")
        rules, scenario = margin_report(positions, equity, "rules"), margin_report(positions, equity, "scenario")
        rows_out = []
        for p in positions:
            initial, maintenance = position_margin(p)
            rows_out.append({"symbol": p.symbol, "asset_class": p.asset_class, "quantity": p.quantity, "price": p.price, "value": p.value,
                             "group": p.group, "initial": initial, "maintenance": maintenance})
        return {"equity": equity, "positions": rows_out, "rules": rules, "scenario": scenario}

    @r("POST", "/api/risk/currency")
    def currency(u, b):
        UserManager.authorize(u, Permission.VIEW_PORTFOLIO)
        base = str(b.get("base", "USD"))
        fx = FxRates(pivot=base.strip().upper())
        for row in b.get("rates") or []:
            fx.set(row["pair"], float(row["rate"]))
        values: Dict[str, float] = {}
        for row in b.get("holdings") or []:
            ccy = str(row["currency"]).strip().upper()
            values[ccy] = values.get(ccy, 0.0) + float(row["amount"])
        if not values:
            raise ValidationError("enter at least one holding")
        exposure = exposure_by_currency(values, base, fx)
        ratio = float(b.get("hedge_ratio", 1.0))
        out: Dict[str, Any] = {"exposure": exposure, "hedges": hedge_notionals(exposure, ratio), "hedge_ratio": ratio, "pairs": fx.pairs()}
        att = b.get("attribution")
        if att:
            out["attribution"] = fx_pnl_attribution(float(att["value_start"]), float(att["value_end"]), float(att["rate_start"]),
                                                    float(att["rate_end"]))
        return out
