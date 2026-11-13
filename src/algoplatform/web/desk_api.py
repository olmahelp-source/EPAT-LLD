"""API routes for the operational job groups that had no screen of their own.

* Market and reference data: the instrument master (list; operations can load contract specs from CSV).
* Portfolio and risk: margin (rule-based and scenario) and multi-currency exposure with hedge amounts.
* Trading: the order-book simulator, to see what latency, order type and size do to a fill.
* Operations and control: health checks, scheduled jobs, undelivered notifications and the restart report.

Kept apart from ``quant_api.py`` and ``research_api.py`` so each file serves one set of job groups. Every route checks
the caller's permission on the server; the browser hiding a page is a convenience, not the control.
"""
from __future__ import annotations

import io
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..brokers.orderbook import MarketSimulator
from ..core.enums import Permission, Side
from ..core.exceptions import ValidationError
from ..portfolio.currency import FxRates, exposure_by_currency, fx_pnl_attribution, hedge_notionals
from ..risk.margin import MarginPosition, margin_report, position_margin
from ..users.manager import UserManager

MAX_POSITIONS = 30
MAX_INSTRUMENT_CSV = 200_000          # bytes
TICK = 0.01


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


def _run_book(side: Side, quantity: float, limit_offset: Optional[int], latency: float, volatility: float, event_rate: float,
              warmup: int, events: int, seed: int) -> Dict[str, Any]:
    """One simulated order: warm the book up, send the order, let ``events`` background events happen, measure."""
    sim = MarketSimulator(mid=100.0, tick_size=TICK, event_rate=event_rate, latency=latency, volatility=volatility, seed=seed)
    sim.run(warmup)
    book = sim.book
    mid0 = book.mid()
    sent_at = sim.now
    if mid0 is None:                                               # cannot happen after a warm-up (an emptied side is re-seeded)
        raise ValidationError("the simulated book has no two-sided quote to measure against")
    arrival = float(mid0)
    depth, estimate = book.depth(5), book.cost_to_trade(side, quantity)
    price = None
    if limit_offset is not None:                                   # a limit order this many ticks behind the touch on its own side
        touch = book.best_bid() if side is Side.BUY else book.best_ask()
        price = round((arrival if touch is None else touch) - side.sign * limit_offset * TICK, 2)   # both sides exist: mid0 was found
    oid = sim.submit(side, quantity, price)
    sim.run(events)
    filled = sim.filled(oid)
    avg = sim.average_fill_price(oid) if filled > 0 else float("nan")
    end = book.mid() or arrival
    q = sim.quotes()
    mid = (q["bid"] + q["ask"]) / 2.0
    keep = max(1, len(q) // 300)
    trades_after = [t for t in sim.fills if oid in (t.maker_id, t.taker_id)]
    return {"limit_price": price, "arrival_mid": arrival, "sent_at": sent_at, "filled": filled, "unfilled": quantity - filled,
            "fill_rate": filled / quantity, "average_price": avg,
            "slippage_bps": side.sign * (avg - arrival) / arrival * 1e4 if filled > 0 else None,
            # the part not filled is still wanted: the market's move since arrival is what it now costs (opportunity cost)
            "opportunity_bps": side.sign * (end - arrival) / arrival * 1e4 * (quantity - filled) / quantity,
            "end_mid": end, "fills": len(trades_after), "rejections": [m for o, m in sim.rejections if o == oid],
            "spread_bps": float(np.nanmean((q["ask"] - q["bid"]) / mid) * 1e4),
            "mid": [[f"{t:.1f}s", float(m)] for t, m in zip(q["time"].iloc[::keep], mid.iloc[::keep]) if math.isfinite(m)],
            "depth": depth, "estimate": estimate}


def register(app: Any) -> None:
    r, P = app.route, app.platform

    def store_check():
        if P.store is None:
            return True, "not configured: shared state is kept in memory only and is lost on restart"
        P.store.put("health.ping", {"ok": True})                  # raises if the database cannot be written
        return True, "writable"

    def sessions_check():
        halted = [s for s in list(app.sessions.values()) if s.engine.risk.kill_switch.state == "TRIGGERED"]
        return not halted, f"{len(app.sessions)} open, {len(halted)} halted by the kill switch"

    # health checks and a housekeeping job for the operations page; none of them is critical to serving requests
    P.health.register("state store", store_check, critical=False)
    P.health.register("notification delivery", lambda: (len(P.notifier.dead_letters) == 0, f"{len(P.notifier.dead_letters)} undelivered"),
                      critical=False)
    P.health.register("paper sessions", sessions_check, critical=False)

    def data_check():
        st = P.data.status()
        if st["sample_only"]:
            return True, "sample-data-only mode: no data provider is used"
        parts, ok = [], True
        for prov in st["providers"]:
            last = prov["status"]
            if not prov["connected"]:
                parts.append(f"{prov['label']}: not connected")
                continue
            if last.get("ok") is False:
                ok = False
            parts.append(f"{prov['label']}: " + ("never tested" if not last else "last call ok" if last.get("ok") else "last call failed"))
        return ok, "; ".join(parts) + f"; {len(st['library'])} downloaded series"

    P.health.register("market data", data_check, critical=False)
    if P.store is not None:
        P.scheduler.every(300, P.autosave, name="save platform state")

    # ---- market and reference data: instruments ---------------------------------------------------------------------------
    @r("GET", "/api/instruments")
    def instruments(u, _b):
        UserManager.authorize(u, Permission.VIEW_PORTFOLIO)
        return {"instruments": P.instruments.export_state()[:1000], "count": len(P.instruments.export_state()),
                "can_load": u.has(Permission.MANAGE_INSTRUMENTS) and u.has(Permission.OPERATE_SYSTEM)}

    @r("POST", "/api/instruments/load")
    def load_instruments(u, b):
        text = str(b.get("csv", ""))
        if not text.strip():
            raise ValidationError("paste CSV text with a header row (symbol, asset_class, name, exchange, currency, ...)")
        if len(text.encode()) > MAX_INSTRUMENT_CSV:
            raise ValidationError(f"CSV too large (limit {MAX_INSTRUMENT_CSV // 1000} KB)")
        try:
            frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValidationError(f"cannot read the CSV: {exc}") from None
        if "symbol" not in frame.columns:
            raise ValidationError("the CSV needs a 'symbol' column", details={"columns": list(frame.columns)})
        report = P.desk(u, "operations").load_instruments(frame.to_dict("records"), on_error="skip", overwrite=bool(b.get("overwrite")))
        return {**report, "count": len(P.instruments.export_state())}

    # ---- portfolio and risk: margin and currency ---------------------------------------------------------------------------
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

    # ---- trading: order-book simulator -------------------------------------------------------------------------------------
    @r("POST", "/api/trading/orderbook")
    def orderbook(u, b):
        UserManager.authorize(u, Permission.RUN_BACKTEST)
        side = Side(str(b.get("side", "BUY")).upper())
        quantity = float(b.get("quantity", 500))
        if not 1 <= quantity <= 100_000:
            raise ValidationError("quantity must be between 1 and 100,000")
        order = b.get("order", "market")
        if order not in ("market", "limit"):
            raise ValidationError("order must be 'market' or 'limit'")
        offset = int(b.get("offset_ticks", 0)) if order == "limit" else None
        if offset is not None and not -20 <= offset <= 50:
            raise ValidationError("offset_ticks must be between -20 and 50")
        latency = float(b.get("latency", 0.5))
        events = int(b.get("events", 1500))
        if not 0 <= latency <= 10 or not 100 <= events <= 5000:
            raise ValidationError("latency must be 0-10 seconds and events 100-5000")
        kw = dict(volatility=float(b.get("volatility", 2.0)), event_rate=float(b.get("event_rate", 50.0)), warmup=300, events=events,
                  seed=int(b.get("seed", 1)))
        if not 0 <= kw["volatility"] <= 20 or not 5 <= kw["event_rate"] <= 500:
            raise ValidationError("volatility must be 0-20 ticks per square-root second and event_rate 5-500 per second")
        chosen = _run_book(side, quantity, offset, latency, **kw)
        instant = _run_book(side, quantity, offset, 0.0, **kw) if latency > 0 else chosen
        return {"side": side.value, "quantity": quantity, "order": order, "latency": latency, "result": chosen, "no_latency": instant}

    # ---- operations and control: health ------------------------------------------------------------------------------------
    @r("GET", "/api/ops/health")
    def health(u, _b):
        desk = P.desk(u, "operations")
        return {"health": desk.health(), "jobs": desk.jobs(), "undelivered": desk.undelivered(), "restore": P.restore_report,
                "persistent": P.store is not None, "sessions": len(app.sessions), "users": P.users.user_count(),
                "routes": [{"scope": "personal" if rt.user_id else "platform", "channel": rt.channel.name, "min_severity": rt.min_severity.value,
                            "topics": list(rt.topics or [])} for rt in P.notifier.routes]}

    @r("POST", "/api/ops/replay")
    def replay(u, _b):
        return P.desk(u, "operations").replay_undelivered()

    @r("POST", "/api/ops/run_jobs")
    def run_jobs(u, _b):
        return {"ran": P.desk(u, "operations").run_scheduled()}


__all__: List[str] = ["register"]
