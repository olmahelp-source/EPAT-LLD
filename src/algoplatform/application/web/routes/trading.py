"""Trading: the paper-trading session (orders, execution algos, kill switch) and the order-book simulator.

Every route checks the caller's permission on the server (through the role desks or ``UserManager.authorize``); hiding a
page in the browser is a convenience, not the control."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import numpy as np

from ....validation.analytics.tca import parent_tca
from ....trading.brokers.orderbook import MarketSimulator
from ....foundation.core.enums import OrderType, Permission, Side
from ....foundation.core.exceptions import ValidationError
from ....market_data.data.providers import check_symbol
from ....operations.persistence import SqliteStore
from ....access.users.manager import UserManager
from ....foundation.utils.timeutils import SimClock

from ..session import LiveSession


ALGO_NAMES = {"twap", "vwap", "pov", "iceberg", "is"}

TICK = 0.01


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

    @r("POST", "/api/session/open")
    def open_session(u, b):
        feed = str(b.get("feed") or "sample")
        if feed not in ("sample", "history"):
            raise ValidationError("feed must be sample or history")
        symbols = [check_symbol(s) for s in b.get("symbols", ["AAA", "BBB"]) if str(s).strip()]
        if not 1 <= len(symbols) <= 6 or (feed == "sample" and not all(s.isalnum() and len(s) <= 8 for s in symbols)):
            raise ValidationError("symbols: 1-6 tickers (letters and digits for sample data)")
        replay = None
        if feed == "history":
            md = P.data.history(symbols, "1Day", min_rows=30)
            replay = {s: md.frame(s) for s in symbols}
        with app.lock:
            if u.user_id in app.sessions:
                raise ValidationError("a session is already open; close it first")
            strategies = []
            if b.get("strategy"):
                strategies = [(P.create_strategy(u, b["strategy"], symbols, **(b.get("params") or {})), 1.0)]
            if app._session_path(u) and os.path.exists(app._session_path(u)):
                raise ValidationError("a saved session exists; resume it from the Paper trading tab or close it first")
            store = SqliteStore(app._session_path(u)) if app._session_path(u) else None
            acc = P.accounts.link(u, "paper", {}, "ui-session")
            # simulated time driven by the synthetic feed (like a backtest), so risk windows such as the
            # runaway-order guard see bar-to-bar time, not the microseconds between clicks
            engine = P.open_session(u, acc.account_id, strategies, clock=SimClock(), store=store)
            engine.start()
            sess = app._watch(LiveSession(engine, symbols, int(b.get("seed", 1)), b.get("strategy") or None,
                                           b.get("params") or {}, store, feed=feed, replay=replay))
            sess.step(1)
            app.sessions[u.user_id] = sess
        return app._snapshot(sess)

    @r("GET", "/api/session")
    def session(u, _b):
        UserManager.authorize(u, Permission.VIEW_PORTFOLIO)
        s = app.sessions.get(u.user_id)
        if s is None:
            with app.lock:
                s = app.sessions.get(u.user_id) or app._restore_session(u)
        return {"open": False} if s is None else app._snapshot(s)

    @r("POST", "/api/session/step")
    def step(u, b):
        UserManager.authorize(u, Permission.PLACE_ORDER)
        s = app._session(u)
        n = int(b.get("n", 1))
        if not 1 <= n <= 100:
            raise ValidationError("n must be 1-100")
        if s.feed == "history":
            P.data.require_real_data_allowed("replaying downloaded history")
        s.step(n)
        return app._snapshot(s)

    @r("POST", "/api/session/order")
    def order(u, b):
        s = app._session(u)
        algo = b.get("algo")
        with s.lock:
            if algo:                                                  # work the order as many risk-checked child orders
                UserManager.authorize(u, Permission.PLACE_ORDER)
                name = str(algo.get("name", "")).lower()
                if name not in ALGO_NAMES:
                    raise ValidationError(f"algo must be one of {sorted(ALGO_NAMES)}")
                params: Dict[str, Any] = {}
                if "bars" in algo and name in ("twap", "vwap", "is"):
                    params["bars"] = int(algo["bars"])
                    if not 2 <= params["bars"] <= 50:
                        raise ValidationError("bars must be 2-50")
                if name == "pov":
                    params["participation"] = float(algo.get("participation", 0.1))
                    if not 0.01 <= params["participation"] <= 0.5:
                        raise ValidationError("participation must be 1%-50%")
                if name == "iceberg":
                    params["display"] = float(algo.get("display", 10))
                    if params["display"] <= 0:
                        raise ValidationError("display size must be positive")
                if name == "is":
                    params["kappa"] = float(algo.get("kappa", 1.0))
                sym, qty = b["symbol"].upper(), float(b["quantity"])
                if sym not in s.symbols or qty <= 0:
                    raise ValidationError("unknown symbol or non-positive quantity")
                s.engine.risk.kill_switch.check()
                parent = s.engine.algos.start(sym, Side(b["side"]), qty, name, s.last[sym], strategy_id=f"manual:{u.username}", **params)
                bar = s.engine._bars.get(sym)
                if bar is not None:
                    s.engine.algos.work_now(parent, bar)
                P.audit.record(u.username, "algo.start", parent=parent.parent_id, algo=name, symbol=sym, qty=qty)
                s.touch()
                return {"parent": parent_tca(parent), **app._snapshot(s)}
            o = P.place_manual_order(u, s.engine, b["symbol"].upper(), Side(b["side"]), float(b["quantity"]),
                                     OrderType(b.get("type", "MARKET")),
                                     float(b["limit_price"]) if b.get("limit_price") else None)
        s.touch()
        return {"order": app._order(o), **app._snapshot(s)}

    @r("POST", "/api/session/cancel")
    def cancel_one(u, b):
        UserManager.authorize(u, Permission.CANCEL_ORDER)
        s = app._session(u)
        with s.lock:
            ok = s.engine.execution.cancel(s.engine.oms.get(str(b["order_id"])))
        s.touch()
        return {"cancelled": ok, **app._snapshot(s)}

    @r("POST", "/api/session/cancel_all")
    def cancel_all(u, _b):
        s = app._session(u)
        n = P.cancel_all(u, s.engine)
        s.engine.algos.cancel_all(reason="cancel all")
        s.touch()
        return {"cancelled": n, **app._snapshot(s)}

    @r("POST", "/api/session/killswitch")
    def killswitch(u, b):
        s = app._session(u)
        with s.lock:
            if b.get("action") == "trip":
                P.trip_kill_switch(u, s.engine, b.get("reason") or "manual halt from UI")
            elif b.get("action") == "reset":
                P.reset_kill_switch(u, s.engine, b.get("reason", ""))
            else:
                raise ValidationError("action must be trip or reset")
        s.touch()
        return app._snapshot(s)

    @r("POST", "/api/session/close")
    def close(u, _b):
        with app.lock:
            s = app.sessions.pop(u.user_id, None)
        if s:
            s.engine.stop()
            if s.store is not None:
                s.store.close()
        path = app._session_path(u)
        if path:                                                  # an ended session must not resurrect on restart
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.remove(path + suffix)
                except OSError:
                    pass
        return {"open": False}

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
