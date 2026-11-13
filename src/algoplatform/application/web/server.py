"""The web application: the route table, request dispatch and error mapping, and the state the routes share.

``App`` owns one ``TradingPlatform`` and, with a ``db_dir``, the databases that let users, paper sessions and the
platform's shared services survive a restart. The routes live in ``web/routes`` (one module per job group), the HTTP
middleware in ``web/http.py`` and the paper-trading feed in ``web/session.py``.

Security posture (read before exposing beyond localhost):
* Bearer tokens from ``UserManager`` (HMAC-signed, expiring); every route except login/register
  needs one, and every action goes through ``TradingPlatform`` which enforces RBAC.
* Request bodies are capped, static paths are confined to ``static/``, and responses carry
  CSP / nosniff / no-referrer headers. There is NO TLS here: bind to 127.0.0.1 or put a TLS
  reverse proxy in front. This is a paper-trading research UI, not a hardened live gateway.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...signals import fundamentals as _fundamentals, sentiment as _sentiment  # noqa: F401  (registers factor_ls / sentiment)
from ...validation.analytics.tca import parent_tca, tca_summary
from ..app.platform import TradingPlatform
from ...foundation.core.config import PlatformConfig
from ...foundation.core.enums import OrderStatus
from ...foundation.core.exceptions import (AuthenticationError, AuthorizationError, BacktestError, ConfigError, DataError, GovernanceError,
                                           ModelError, OptimizationError, OrderError, PlatformError, RiskError, StatisticsError,
                                           StrategyError, UserError, ValidationError)
from ...foundation.core.logging import get_logger
from ...market_data.data import MarketData, synthetic
from ...market_data.data.providers import check_symbol
from ...operations.persistence import SqliteStore
from ...capital.risk.measures import (HISTORICAL_SCENARIOS, expected_shortfall, historical_var, parametric_var, reverse_stress_loss,
                                      stress_test)
from ...access.users.manager import User
from ...foundation.utils.timeutils import SimClock
from . import routes
from .common import NEEDS_DATA, clean, series_points
from .http import MAX_BODY, STATIC, create_server, make_handler, serve  # noqa: F401  (re-exported: the public entry points)
from .session import LiveSession, session_metrics

log = get_logger(__name__)
_STATUS = ((AuthenticationError, 401), (AuthorizationError, 403), (UserError, 409), (RiskError, 409), (OrderError, 409),
           (GovernanceError, 409), (ValidationError, 400), (ConfigError, 400), (StrategyError, 400), (BacktestError, 400),
           (OptimizationError, 400), (DataError, 400), (ModelError, 400), (StatisticsError, 400))


class App:
    def __init__(self, config: Optional[PlatformConfig] = None, fast_hash: bool = False,
                 db_dir: Optional[str] = None) -> None:
        self.db_dir = db_dir
        self._meta: Optional[SqliteStore] = None
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        # with a db_dir, instruments, strategy approvals, models, alert rules and notification routes survive a restart
        self.platform = TradingPlatform(config or PlatformConfig(log_level="WARNING"),
                                        store=SqliteStore(os.path.join(db_dir, "platform.db")) if db_dir else None)
        if fast_hash:                                    # tests only: cheap scrypt
            from ...access.users.security import PasswordHasher
            self.platform.users.hasher = PasswordHasher(n=2 ** 4)
        if db_dir:
            self._meta = SqliteStore(os.path.join(db_dir, "app.db"))
            n = self.platform.users.import_users(self._meta.get("users", []))
            log.info("restored %d users from %s", n, db_dir)
        self.sessions: Dict[str, LiveSession] = {}
        self.datasets: Dict[str, Dict[str, pd.DataFrame]] = {}      # user id -> uploaded OHLCV by symbol
        self.shutting_down = False
        self.lock = threading.Lock()
        self.routes: Dict[Tuple[str, str], Tuple[Callable, bool]] = {}
        routes.register_all(self)

    def close(self) -> None:
        """Stop serving streams and close every database this app opened (platform, users, paper sessions)."""
        self.shutting_down = True
        stores = [s.store for s in self.sessions.values() if s.store is not None] + [self._meta, self.platform.store]
        for st in stores:
            if st is not None:
                try:
                    st.close()
                except Exception as exc:  # noqa: BLE001 - closing one database must not stop the others
                    log.warning("closing a store failed: %r", exc)

    def _watch(self, sess: "LiveSession") -> "LiveSession":
        """Evaluate the platform's alert rules against this session after every simulated bar."""
        sess.on_step = lambda s: self.platform.alerts.evaluate(session_metrics(s.engine, s.last))
        return sess

    # ---- routing --------------------------------------------------------------------------
    def route(self, method: str, path: str, public: bool = False):
        def deco(fn):
            self.routes[(method, path)] = (fn, public)
            return fn
        return deco

    def dispatch(self, method: str, path: str, body: Dict[str, Any], token: Optional[str]) -> Tuple[int, Any]:
        entry = self.routes.get((method, path))
        if entry is None:
            return 404, {"error": "NOT_FOUND", "message": f"no route {method} {path}"}
        fn, public = entry
        try:
            user = None
            if not public:
                if not token:
                    raise AuthenticationError("missing bearer token")
                user = self.platform.users.user_from_token(token)
            return 200, clean(fn(user, body))
        except PlatformError as exc:
            for cls, code in _STATUS:
                if isinstance(exc, cls):
                    return code, {"error": exc.code, "message": exc.message}
            return 500, {"error": "INTERNAL", "message": "internal error"}
        except (KeyError, TypeError, ValueError) as exc:     # malformed client input
            return 400, {"error": "BAD_REQUEST", "message": f"invalid request: {exc}"}
        except Exception:  # noqa: BLE001
            log.exception("unhandled error on %s %s", method, path)
            return 500, {"error": "INTERNAL", "message": "internal error"}

    # ---- persistence helpers ---------------------------------------------------------------
    def _save_users(self) -> None:
        if self._meta is not None:
            self._meta.put("users", self.platform.users.export_users())

    def _session_path(self, user: User) -> Optional[str]:
        if not self.db_dir:
            return None
        safe = "".join(c for c in user.username if c.isalnum() or c in "_-@.")
        return os.path.join(self.db_dir, f"session-{safe}.db")

    def _restore_session(self, user: User) -> Optional[LiveSession]:
        """Resume a session that was alive when the server stopped: rebuild the simulated broker from the journal,
        recover the engine, and reconcile. A mismatch halts trading rather than guessing."""
        path = self._session_path(user)
        if not path or not os.path.exists(path):
            return None
        store = SqliteStore(path)
        meta = store.get("session")
        if not meta:
            store.close()
            return None
        P = self.platform
        strategies = []
        if meta.get("strategy"):
            strategies = [(P.create_strategy(user, meta["strategy"], meta["symbols"], **(meta.get("params") or {})), 1.0)]
        acc = P.accounts.link(user, "paper", {}, "ui-session")
        clock = SimClock()
        engine = P.open_session(user, acc.account_id, strategies, clock=clock, store=store)
        orders = {o.client_order_id: o for o in store.load_orders()}
        fills = store.load_fills()
        for f in fills:
            if f.client_order_id in orders:
                orders[f.client_order_id].fills.append(f)
        live = [o for o in orders.values() if o.status in (OrderStatus.ACKNOWLEDGED, OrderStatus.PARTIALLY_FILLED)]
        broker: Any = engine.broker                                  # always the paper broker in the UI: it can be rebuilt
        broker.restore(fills, live, meta.get("last"))
        feed, replay = meta.get("feed", "sample"), None
        if feed == "history" and not P.data.sample_only:
            try:
                replay = {s_: P.data.library.get(s_, "1Day") for s_ in meta["symbols"]}
            except PlatformError:
                replay = None                                       # removed since: the session comes back, stepping explains why it cannot
        sess = self._watch(LiveSession(engine, meta["symbols"], meta["seed"], meta.get("strategy"), meta.get("params"), store,
                                       feed=feed, replay=replay, pos=int(meta.get("pos", 0))))
        sess.last, sess.history = {k: float(v) for k, v in meta["last"].items()}, meta["history"]
        sess.next_ts = pd.Timestamp(meta["next_ts"])
        clock.set((sess.next_ts - pd.offsets.BDay(1)).to_pydatetime())      # simulated time resumes at the last bar
        sess.rng = np.random.default_rng([meta["seed"], int(meta.get("bars", 0))])
        engine.start()
        rep = engine.recover()
        engine.bars_processed = int(meta.get("bars", 0))
        sess.recovery = {"clean": rep.clean, "halted": rep.halted, "reason": rep.halt_reason,
                         "orders": rep.orders, "fills": rep.fills}
        self.sessions[user.user_id] = sess
        return sess

    # ---- helpers --------------------------------------------------------------------------
    def _data(self, body: Dict[str, Any], user: Optional[User] = None) -> Tuple[MarketData, List[str]]:
        """The data a research request runs on, chosen by ``source``:

        * ``"sample"`` (the default): generated data; ``scenario`` picks a market scenario, otherwise a mixed universe;
        * ``"history"``: daily bars downloaded from a data provider into the shared library;
        * ``"uploaded"`` (or the older ``dataset="uploaded"``): the user's own CSV files.

        In sample-data-only mode the last two are refused."""
        source = str(body.get("source") or ("uploaded" if body.get("dataset") == "uploaded" else "sample"))
        if source not in ("sample", "history", "uploaded"):
            raise ValidationError("source must be sample, history or uploaded")
        if source == "history":
            wanted = body.get("symbols") or self.platform.data.library.symbols("1Day")[:10]
            symbols = list(dict.fromkeys(check_symbol(s_) for s_ in wanted if str(s_).strip()))
            if not 1 <= len(symbols) <= 10:
                raise ValidationError("downloaded history: choose 1-10 symbols (download them on the Price data page first)")
            md = self.platform.data.history(symbols, "1Day", min_rows=int(body.get("min_rows", 150)))
            return md, symbols
        if source == "uploaded":
            self.platform.data.require_real_data_allowed("using uploaded files")
            frames = self.datasets.get(user.user_id, {}) if user else {}
            if not frames:
                raise ValidationError("no uploaded data: upload at least one CSV on the Price data page")
            wanted = [s.strip().upper() for s in body.get("symbols") or frames if s.strip()]
            missing = [s for s in wanted if s not in frames]
            if missing:
                raise ValidationError(f"not uploaded: {missing}")
            md = MarketData()
            for s_ in wanted:
                if len(frames[s_]) < 150:
                    raise ValidationError(f"{s_}: need at least 150 rows, have {len(frames[s_])}")
                md.add(s_, frames[s_])
            return md, wanted
        symbols = [s.strip().upper() for s in body.get("symbols", ["AAA", "BBB", "CCC"]) if s.strip()]
        if not 1 <= len(symbols) <= 10 or not all(s.isalnum() and len(s) <= 8 for s in symbols):
            raise ValidationError("symbols: 1-10 alphanumeric tickers (max 8 chars)")
        bars = int(body.get("bars", 756))
        if not 150 <= bars <= 3000:
            raise ValidationError("bars must be between 150 and 3000")
        md = MarketData()
        scenario = body.get("scenario")
        frames = (self.platform.data.sample(str(scenario), symbols, bars=bars, seed=int(body.get("seed", 7))) if scenario
                  else synthetic.universe(symbols, n=bars, seed=int(body.get("seed", 7))))
        for s_, df in frames.items():
            md.add(s_, df)
        return md, symbols

    @staticmethod
    def _check_strategy(name: str) -> None:
        if name in NEEDS_DATA:
            raise ValidationError(f"'{name}' needs attached fundamentals/news data and cannot run from the UI")

    @staticmethod
    def _symbols_for(strategy: str, symbols: List[str]) -> List[str]:
        return symbols[:2] if strategy == "pairs_trading" else symbols

    @staticmethod
    def _risk_measures(returns: pd.Series) -> Optional[Dict[str, float]]:
        """Historical and parametric VaR / expected shortfall of per-bar returns (as positive loss fractions)."""
        r = returns.dropna().to_numpy()
        if len(r) < 30:
            return None
        return {"var_95": historical_var(r, 0.95), "var_99": historical_var(r, 0.99), "es_95": expected_shortfall(r, 0.95),
                "es_99": expected_shortfall(r, 0.99), "parametric_var_99": parametric_var(r, 0.99), "bars": len(r)}

    def _session(self, user: User) -> LiveSession:
        s = self.sessions.get(user.user_id)
        if s is None:
            raise ValidationError("no open session; open one first")
        return s

    # ---- endpoints --------------------------------------------------------------------------

    # ---- serialisation of live state ---------------------------------------------------------
    @staticmethod
    def _order(o) -> Dict[str, Any]:
        return {"id": o.client_order_id, "symbol": o.symbol, "side": o.side, "qty": o.quantity, "type": o.order_type,
                "limit": o.limit_price, "status": o.status, "filled": o.filled_qty, "avg": o.avg_fill_price,
                "reason": o.reject_reason, "strategy": o.strategy_id}

    def _snapshot(self, s: LiveSession) -> Dict[str, Any]:
        with s.lock:
            e = s.engine
            snap = e.portfolio.snapshot()
            ks = e.risk.kill_switch
            return {"open": True, "resumed": s.recovery is not None, "recovery": s.recovery,
                    "symbols": s.symbols, "prices": s.last, "history": s.history, "feed": s.feed, "replay": s.replay_status(),
                    "portfolio": snap, "bars": e.bars_processed,
                    "equity_curve": series_points(e.portfolio.equity_curve().tail(300)) if e.bars_processed else [],
                    "orders": [self._order(o) for o in reversed(e.oms.all_orders()[-60:])],
                    "kill_switch": {"state": ks.state, "reason": ks.reason, "source": ks.source},
                    "risk": {"drawdown": e.risk.monitor.drawdown, "daily_pnl_pct": e.risk.monitor.daily_pnl_pct,
                             "max_daily_loss_pct": e.cfg.risk.max_daily_loss_pct,
                             "max_drawdown_pct": e.cfg.risk.max_drawdown_pct,
                             "rejections": e.risk.rejections, "exposure_scalar": e.risk.exposure_scalar()},
                    "strategy_errors": e.strategy_errors, "anomalies": len(e.oms.anomalies),
                    "parents": [parent_tca(p) for p in list(e.algos.parents.values())[-8:]][::-1],
                    "tca": tca_summary(e.algos.parents.values()), "measures": self._live_measures(e), "version": s.version}

    @staticmethod
    def _live_measures(e) -> Dict[str, Any]:
        """VaR/ES from the session's own equity returns, stress scenarios on current positions, reverse stress."""
        eq = e.portfolio.equity_curve()
        out: Dict[str, Any] = {"var": App._risk_measures(eq.pct_change().dropna()) if len(eq) > 31 else None}
        values = {sy: p.market_value for sy, p in e.portfolio.open_positions().items()}
        equity, gross = e.portfolio.equity, e.portfolio.gross_exposure
        out["stress"] = [{"scenario": name, "pnl": stress_test(values, shocks)["total"], "pct_equity": stress_test(values, shocks)["total"] / equity}
                         for name, shocks in HISTORICAL_SCENARIOS.items()] if values else []
        out["reverse_stress_move"] = reverse_stress_loss(equity, gross, e.cfg.risk.max_daily_loss_pct) if gross > 0 else None
        return out


__all__: List[str] = ["App", "LiveSession", "create_server", "serve", "make_handler", "clean", "series_points", "session_metrics"]
