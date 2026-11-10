"""JSON API + static file server for the UI.

Security posture (read before exposing beyond localhost):
* Bearer tokens from ``UserManager`` (HMAC-signed, expiring); every route except login/register
  needs one, and every action goes through ``TradingPlatform`` which enforces RBAC.
* Request bodies are capped, static paths are confined to ``static/``, and responses carry
  CSP / nosniff / no-referrer headers. There is NO TLS here: bind to 127.0.0.1 or put a TLS
  reverse proxy in front. This is a paper-trading research UI, not a hardened live gateway.
"""
from __future__ import annotations

import io
import json
import math
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from ..analytics.metrics import drawdown_series
from ..analytics.tca import parent_tca, tca_summary
from ..analytics.trades import contribution_by
from ..backtest import BacktestEngine
from ..app.platform import TradingPlatform
from ..core.config import PlatformConfig
from ..core.enums import OrderStatus, OrderType, Permission, Role, Side
from ..core.exceptions import (AuthenticationError, AuthorizationError, BacktestError, ConfigError, DataError,
                               OptimizationError, OrderError, PlatformError, RiskError, StrategyError, UserError,
                               ValidationError)
from ..core.logging import get_logger
from ..core.models import Bar
from ..data import MarketData, synthetic
from ..data.cleaning import validate_ohlcv
from ..portfolio.allocation import FundAllocator
from ..risk.measures import (HISTORICAL_SCENARIOS, expected_shortfall, historical_var, parametric_var,
                             reverse_stress_loss, stress_test)
from ..persistence import SqliteStore
from ..optimize import ParameterSpace, walk_forward
from .. import fundamentals as _fundamentals, sentiment as _sentiment  # noqa: F401  (registers factor_ls / sentiment)
from ..strategy.base import STRATEGIES
from ..utils.timeutils import SimClock
from ..users.manager import User, UserManager

log = get_logger(__name__)
STATIC = os.path.join(os.path.dirname(__file__), "static")
MAX_BODY = 1_000_000
NEEDS_DATA = {"factor_ls", "sentiment"}          # need a fundamentals store / news feed: not runnable from the UI
SLOW = {"ml_logistic"}                            # retrains repeatedly: keep optimisation small
ALGO_NAMES = {"twap", "vwap", "pov", "iceberg", "is"}
MAX_UPLOAD_ROWS, MAX_UPLOAD_SYMBOLS = 5000, 10
_STATUS = ((AuthenticationError, 401), (AuthorizationError, 403), (UserError, 409), (RiskError, 409), (OrderError, 409),
           (ValidationError, 400), (ConfigError, 400), (StrategyError, 400), (BacktestError, 400),
           (OptimizationError, 400), (DataError, 400))


def clean(o: Any) -> Any:
    """Make anything JSON-safe: numpy -> python, NaN/inf -> null, timestamps -> ISO strings."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)) or hasattr(o, "isoformat"):
        return o.isoformat()
    if hasattr(o, "value") and not isinstance(o, (str, int)):          # Enum
        return o.value
    return o


def series_points(s: pd.Series, max_points: int = 400) -> List[list]:
    if len(s) > max_points:
        s = s.iloc[np.linspace(0, len(s) - 1, max_points).astype(int)]
    return [[pd.Timestamp(i).strftime("%Y-%m-%d"), float(v)] for i, v in s.items()]


class LiveSession:
    """A paper-trading engine fed by a synthetic random-walk price stream."""

    def __init__(self, engine, symbols: List[str], seed: int, strategy: Optional[str] = None,
                 params: Optional[Dict[str, Any]] = None, store: Optional[SqliteStore] = None) -> None:
        self.engine, self.symbols, self.seed = engine, symbols, seed
        self.strategy, self.params, self.store = strategy, params or {}, store
        self.rng = np.random.default_rng(seed)
        self.last = {s: float(self.rng.uniform(50, 300)) for s in symbols}
        self.next_ts = pd.Timestamp("2024-01-01")
        self.lock = threading.RLock()
        self.history: Dict[str, List[float]] = {s: [] for s in symbols}
        self.cond = threading.Condition()
        self.version = 0

    def touch(self) -> None:
        """Wake stream listeners: the visible state changed."""
        with self.cond:
            self.version += 1
            self.cond.notify_all()

    def save_meta(self) -> None:
        """Feed state needed to resume after a restart (orders/fills/positions are in the journal)."""
        if self.store is not None:
            self.store.put("session", {"symbols": self.symbols, "seed": self.seed, "strategy": self.strategy,
                                       "params": self.params, "last": self.last, "history": self.history,
                                       "next_ts": self.next_ts.isoformat(), "bars": self.engine.bars_processed})

    def step(self, n: int) -> None:
        with self.lock:
            for _ in range(n):
                bars = {}
                for s in self.symbols:
                    o = self.last[s]
                    c = o * float(np.exp(self.rng.normal(0.0003, 0.012)))
                    bars[s] = Bar(self.next_ts, s, o, max(o, c) * 1.003, min(o, c) * 0.997, c, 1e6)
                    self.last[s] = c
                    self.history[s].append(round(c, 4))
                    self.history[s] = self.history[s][-200:]
                self.engine.on_bars(bars)
                self.next_ts = self.next_ts + pd.offsets.BDay(1)
            self.save_meta()
        self.touch()


class App:
    def __init__(self, config: Optional[PlatformConfig] = None, fast_hash: bool = False,
                 db_dir: Optional[str] = None) -> None:
        self.platform = TradingPlatform(config or PlatformConfig(log_level="WARNING"))
        if fast_hash:                                    # tests only: cheap scrypt
            from ..users.security import PasswordHasher
            self.platform.users.hasher = PasswordHasher(n=2 ** 4)
        self.db_dir = db_dir
        self._meta: Optional[SqliteStore] = None
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            self._meta = SqliteStore(os.path.join(db_dir, "app.db"))
            n = self.platform.users.import_users(self._meta.get("users", []))
            log.info("restored %d users from %s", n, db_dir)
        self.sessions: Dict[str, LiveSession] = {}
        self.datasets: Dict[str, Dict[str, pd.DataFrame]] = {}      # user id -> uploaded OHLCV by symbol
        self.shutting_down = False
        self.lock = threading.Lock()
        self.routes: Dict[Tuple[str, str], Tuple[Callable, bool]] = {}
        self._register()

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
        engine.broker.restore(fills, live, meta.get("last"))
        sess = LiveSession(engine, meta["symbols"], meta["seed"], meta.get("strategy"), meta.get("params"), store)
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
        """Synthetic universe, or the user's uploaded CSVs when ``dataset == "uploaded"``."""
        if body.get("dataset") == "uploaded":
            frames = self.datasets.get(user.user_id, {}) if user else {}
            if not frames:
                raise ValidationError("no uploaded data: upload at least one CSV in the Data tab")
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
        for s_, df in synthetic.universe(symbols, n=bars, seed=int(body.get("seed", 7))).items():
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
    def _register(self) -> None:
        r, P = self.route, self.platform

        @r("POST", "/api/register", public=True)
        def register(_u, b):
            # bootstrap: on a brand-new install the first account is the administrator. Everyone after
            # that self-registers as a plain trader; only an admin can raise roles.
            with self.lock:
                role = Role.ADMIN if P.users.user_count() == 0 else Role.TRADER
                user = P.users.register(b["username"], b["password"], role)
                self._save_users()
            return {"username": user.username, "role": user.role}

        @r("POST", "/api/login", public=True)
        def login(_u, b):
            token = P.users.login(b["username"], b["password"])
            user = P.users.user_from_token(token)
            return {"token": token, "username": user.username, "role": user.role}

        @r("GET", "/api/me")
        def me(u, _b):
            return {"username": u.username, "role": u.role, "permissions": sorted(p.value for p in Permission if u.has(p))}

        @r("GET", "/api/strategies")
        def strategies(_u, _b):
            return [{"name": n, "params": STRATEGIES.get(n).PARAMS, "bounds": STRATEGIES.get(n).PARAM_BOUNDS,
                     "doc": (STRATEGIES.get(n).__doc__ or "").strip().split("\n")[0], "runnable": n not in NEEDS_DATA,
                     "slow": n in SLOW} for n in STRATEGIES.names()]

        @r("POST", "/api/backtest")
        def backtest(u, b):
            md, symbols = self._data(b, u)
            target = b["spec"] if b.get("spec") else b["strategy"]
            if not b.get("spec"):
                self._check_strategy(b["strategy"])
                symbols = self._symbols_for(b["strategy"], symbols)
            res = P.backtest(u, target, md, symbols, b.get("params") or {})
            rep = res.report(n_trials=int(b.get("n_trials", 1)))
            eq = res.equity
            bench = (1 + res.benchmark).cumprod() * eq.iloc[0] if res.benchmark is not None else None
            trades = res.trades_frame().tail(200)
            return {"metrics": rep["performance"], "trade_stats": rep["trades"],
                    "significance": rep.get("significance"), "equity": series_points(eq),
                    "drawdown": series_points(drawdown_series(eq)),
                    "benchmark": series_points(bench) if bench is not None else None,
                    "trades": trades.assign(entry_ts=trades["entry_ts"].astype(str),
                                            exit_ts=trades["exit_ts"].astype(str)).round(4).to_dict("records"),
                    "orders": res.order_count, "rejected": res.rejected_orders,
                    "kill_switch": res.kill_switch_triggered, "symbols": symbols, "risk": self._risk_measures(res.returns)}

        @r("POST", "/api/optimize")
        def optimize(u, b):
            method = b.get("method", "random")
            if method not in ("random", "genetic"):
                raise ValidationError("method must be random or genetic")
            n = int(b.get("n_iter", 20))
            if not 5 <= n <= 60:
                raise ValidationError("n_iter must be between 5 and 60")
            if int(b.get("bars", 500)) > 1500:
                raise ValidationError("bars must be <= 1500 for optimisation")
            self._check_strategy(b["strategy"])
            if b["strategy"] in SLOW and n > 10:
                raise ValidationError("this strategy retrains a model at every step: use 10 trials or fewer")
            md, symbols = self._data({**b, "bars": b.get("bars", 500)}, u)
            symbols = self._symbols_for(b["strategy"], symbols)
            cls = STRATEGIES.get(b["strategy"])
            if not cls.PARAM_BOUNDS:
                raise ValidationError("strategy has no tunable parameters")
            space = ParameterSpace.from_bounds(cls.PARAM_BOUNDS)
            kw = {"n_iter": n} if method == "random" else {"population": max(6, n // 4), "generations": 4}
            res = P.optimize(u, b["strategy"], md, space, symbols, b.get("metric", "sharpe"), method, **kw)
            top = res.top(10)
            return {"summary": res.summary(), "top": top.to_dict("records"),
                    "scores": [float(x) for x in res.trials["score"].replace([-np.inf], np.nan).dropna()],
                    "plateau": res.plateau_score(space)}

        # ---- uploaded data -----------------------------------------------------------------------
        @r("POST", "/api/data/upload")
        def upload(u, b):
            UserManager.authorize(u, Permission.RUN_BACKTEST)
            sym = str(b.get("symbol", "")).strip().upper()
            if not sym.isalnum() or len(sym) > 8:
                raise ValidationError("symbol must be 1-8 letters/digits")
            text = b.get("csv")
            if not isinstance(text, str) or not text.strip():
                raise ValidationError("csv text is required")
            mine = self.datasets.setdefault(u.user_id, {})
            if sym not in mine and len(mine) >= MAX_UPLOAD_SYMBOLS:
                raise ValidationError(f"at most {MAX_UPLOAD_SYMBOLS} uploaded symbols per user")
            try:
                df = pd.read_csv(io.StringIO(text), nrows=MAX_UPLOAD_ROWS + 1)
            except Exception as exc:  # noqa: BLE001 - any parser failure is the user's data problem
                raise DataError(f"cannot parse CSV: {type(exc).__name__}") from None
            if len(df) > MAX_UPLOAD_ROWS:
                raise DataError(f"too many rows (max {MAX_UPLOAD_ROWS})")
            df.columns = [str(c).strip().lower() for c in df.columns]
            if "date" not in df.columns:
                raise DataError("CSV needs a 'date' column plus open, high, low, close (volume optional)")
            try:
                df["date"] = pd.to_datetime(df["date"])
            except Exception:  # noqa: BLE001
                raise DataError("could not parse the 'date' column") from None
            df = df.set_index("date").sort_index()
            df.index.name = None
            validate_ohlcv(df, sym)                                           # raises DataError with the exact problem
            mine[sym] = df[[c for c in ("open", "high", "low", "close", "volume") if c in df.columns]].astype(float)
            P.audit.record(u.username, "data.upload", symbol=sym, rows=len(df))
            return {"symbol": sym, "rows": len(df), "start": df.index[0], "end": df.index[-1]}

        @r("GET", "/api/data")
        def list_data(u, _b):
            return [{"symbol": k, "rows": len(v), "start": v.index[0], "end": v.index[-1]}
                    for k, v in sorted(self.datasets.get(u.user_id, {}).items())]

        @r("POST", "/api/data/clear")
        def clear_data(u, _b):
            self.datasets.pop(u.user_id, None)
            return {"ok": True}

        # ---- walk-forward and multi-strategy portfolio ---------------------------------------------
        @r("POST", "/api/walkforward")
        def walkforward(u, b):
            UserManager.authorize(u, Permission.RUN_BACKTEST)
            name = b["strategy"]
            self._check_strategy(name)
            cls = STRATEGIES.get(name)
            if not cls.PARAM_BOUNDS:
                raise ValidationError("strategy has no tunable parameters")
            bars, train, test, n = (int(b.get("bars", 800)), int(b.get("train_bars", 300)), int(b.get("test_bars", 100)),
                                    int(b.get("n_iter", 8)))
            if bars > 1500 or not 3 <= n <= (8 if name in SLOW else 20):
                raise ValidationError("bars must be <= 1500 and n_iter 3-20 (3-8 for model-based strategies)")
            md, symbols = self._data({**b, "bars": bars}, u)
            symbols = self._symbols_for(name, symbols)
            wf = walk_forward(BacktestEngine(md, P.config), P._factory(name, symbols), ParameterSpace.from_bounds(cls.PARAM_BOUNDS),
                              train, test, metric=b.get("metric", "sharpe"), method="random", n_iter=n, min_trades=1)
            P.audit.record(u.username, "walkforward.run", strategy=name, folds=len(wf.folds))
            folds = wf.folds.copy()
            for c in ("train_start", "train_end", "test_start", "test_end"):
                folds[c] = folds[c].astype(str).str[:10]
            return {"folds": folds.round(4).to_dict("records"), "oos_equity": series_points(wf.oos_equity),
                    "oos_metrics": wf.oos_metrics, "efficiency": wf.efficiency, "stability": wf.param_stability,
                    "params": list(cls.PARAM_BOUNDS)}

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
            md, symbols = self._data(b, u)

            def build(i: int, it: Dict[str, Any]):
                self._check_strategy(it["name"])
                return STRATEGIES.create(it["name"], self._symbols_for(it["name"], symbols), strategy_id=f"{it['name']}#{i + 1}",
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
                    "correlation": corr, "suggested_weights": suggested, "risk": self._risk_measures(combined.returns),
                    "allocation": {f"{it['name']}#{i + 1}": pct for i, (it, pct) in enumerate(zip(items, pcts))}}

        # ---- live (paper) session -----------------------------------------------------------
        @r("POST", "/api/session/open")
        def open_session(u, b):
            symbols = [s.strip().upper() for s in b.get("symbols", ["AAA", "BBB"]) if s.strip()]
            if not 1 <= len(symbols) <= 6 or not all(s.isalnum() and len(s) <= 8 for s in symbols):
                raise ValidationError("symbols: 1-6 alphanumeric tickers")
            with self.lock:
                if u.user_id in self.sessions:
                    raise ValidationError("a session is already open; close it first")
                strategies = []
                if b.get("strategy"):
                    strategies = [(P.create_strategy(u, b["strategy"], symbols, **(b.get("params") or {})), 1.0)]
                if self._session_path(u) and os.path.exists(self._session_path(u)):
                    raise ValidationError("a saved session exists; resume it from the Paper trading tab or close it first")
                store = SqliteStore(self._session_path(u)) if self._session_path(u) else None
                acc = P.accounts.link(u, "paper", {}, "ui-session")
                # simulated time driven by the synthetic feed (like a backtest), so risk windows such as the
                # runaway-order guard see bar-to-bar time, not the microseconds between clicks
                engine = P.open_session(u, acc.account_id, strategies, clock=SimClock(), store=store)
                engine.start()
                sess = LiveSession(engine, symbols, int(b.get("seed", 1)), b.get("strategy") or None,
                                   b.get("params") or {}, store)
                sess.step(1)
                self.sessions[u.user_id] = sess
            return self._snapshot(sess)

        @r("GET", "/api/session")
        def session(u, _b):
            UserManager.authorize(u, Permission.VIEW_PORTFOLIO)
            s = self.sessions.get(u.user_id)
            if s is None:
                with self.lock:
                    s = self.sessions.get(u.user_id) or self._restore_session(u)
            return {"open": False} if s is None else self._snapshot(s)

        @r("POST", "/api/session/step")
        def step(u, b):
            UserManager.authorize(u, Permission.PLACE_ORDER)
            s = self._session(u)
            n = int(b.get("n", 1))
            if not 1 <= n <= 100:
                raise ValidationError("n must be 1-100")
            s.step(n)
            return self._snapshot(s)

        @r("POST", "/api/session/order")
        def order(u, b):
            s = self._session(u)
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
                    return {"parent": parent_tca(parent), **self._snapshot(s)}
                o = P.place_manual_order(u, s.engine, b["symbol"].upper(), Side(b["side"]), float(b["quantity"]),
                                         OrderType(b.get("type", "MARKET")),
                                         float(b["limit_price"]) if b.get("limit_price") else None)
            s.touch()
            return {"order": self._order(o), **self._snapshot(s)}

        @r("POST", "/api/session/cancel")
        def cancel_one(u, b):
            UserManager.authorize(u, Permission.CANCEL_ORDER)
            s = self._session(u)
            with s.lock:
                ok = s.engine.execution.cancel(s.engine.oms.get(str(b["order_id"])))
            s.touch()
            return {"cancelled": ok, **self._snapshot(s)}

        @r("POST", "/api/session/cancel_all")
        def cancel_all(u, _b):
            s = self._session(u)
            n = P.cancel_all(u, s.engine)
            s.engine.algos.cancel_all(reason="cancel all")
            s.touch()
            return {"cancelled": n, **self._snapshot(s)}

        @r("POST", "/api/session/killswitch")
        def killswitch(u, b):
            s = self._session(u)
            with s.lock:
                if b.get("action") == "trip":
                    P.trip_kill_switch(u, s.engine, b.get("reason") or "manual halt from UI")
                elif b.get("action") == "reset":
                    P.reset_kill_switch(u, s.engine, b.get("reason", ""))
                else:
                    raise ValidationError("action must be trip or reset")
            s.touch()
            return self._snapshot(s)

        @r("POST", "/api/session/close")
        def close(u, _b):
            with self.lock:
                s = self.sessions.pop(u.user_id, None)
            if s:
                s.engine.stop()
                if s.store is not None:
                    s.store.close()
            path = self._session_path(u)
            if path:                                                  # an ended session must not resurrect on restart
                for suffix in ("", "-wal", "-shm"):
                    try:
                        os.remove(path + suffix)
                    except OSError:
                        pass
            return {"open": False}

        # ---- admin ---------------------------------------------------------------------------
        @r("GET", "/api/audit")
        def audit(u, _b):
            if not (u.has(Permission.MANAGE_USERS) or u.has(Permission.MANAGE_RISK)):
                raise AuthorizationError("audit log needs admin or risk-manager role")
            return list(reversed(P.audit.entries()[-150:]))

        @r("GET", "/api/users")
        def users(u, _b):
            return [{"id": x.user_id, "username": x.username, "role": x.role, "active": x.active}
                    for x in P.users.list_users(u)]

        @r("POST", "/api/users/role")
        def set_role(u, b):
            P.users.set_role(u, b["user_id"], Role(b["role"]))
            self._save_users()
            return {"ok": True}

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
            return {"open": True, "resumed": getattr(s, "recovery", None) is not None, "recovery": getattr(s, "recovery", None),
                    "symbols": s.symbols, "prices": s.last, "history": s.history,
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


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "algoplatform"

        def log_message(self, fmt, *args):          # quiet by default
            log.debug("%s " + fmt, self.address_string(), *args)

        def _headers(self, code: int, ctype: str, length: int) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def _json(self, code: int, obj: Any) -> None:
            raw = json.dumps(obj).encode()
            self._headers(code, "application/json; charset=utf-8", len(raw))
            self.wfile.write(raw)

        def _static(self, path: str) -> None:
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            full = os.path.realpath(os.path.join(STATIC, rel))
            if not full.startswith(os.path.realpath(STATIC) + os.sep) or not os.path.isfile(full):
                return self._json(404, {"error": "NOT_FOUND", "message": "not found"})
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            with open(full, "rb") as fh:
                data = fh.read()
            self._headers(200, ctype + ("; charset=utf-8" if ctype.startswith("text") or "javascript" in ctype else ""), len(data))
            self.wfile.write(data)

        def _stream(self) -> None:
            """Server-Sent Events: push a full snapshot whenever the session changes (auth via the Authorization header,
            so the token never appears in a URL)."""
            auth = self.headers.get("Authorization", "")
            try:
                user = app.platform.users.user_from_token(auth[7:] if auth.startswith("Bearer ") else "")
                UserManager.authorize(user, Permission.VIEW_PORTFOLIO)
            except PlatformError as exc:
                return self._json(401 if isinstance(exc, AuthenticationError) else 403, {"error": exc.code, "message": exc.message})
            sess = app.sessions.get(user.user_id)
            if sess is None:
                return self._json(404, {"error": "NO_SESSION", "message": "no open session"})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'")
            self.end_headers()
            self.close_connection = True
            last, idle = -1, 0
            try:
                while not app.shutting_down:
                    with sess.cond:
                        if sess.version == last:
                            sess.cond.wait(timeout=1.0)
                        ver = sess.version
                    if app.sessions.get(user.user_id) is not sess:
                        self.wfile.write(b"event: closed\ndata: {}\n\n")
                        self.wfile.flush()
                        return
                    if ver != last:
                        payload = json.dumps(clean(app._snapshot(sess)))
                        self.wfile.write(f"event: snapshot\ndata: {payload}\n\n".encode())
                        self.wfile.flush()
                        last, idle = ver, 0
                    else:
                        idle += 1
                        if idle >= 15:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            idle = 0
            except (BrokenPipeError, ConnectionResetError, OSError):
                return                                                    # the browser went away

        def _handle(self, method: str) -> None:
            path = urlparse(self.path).path
            if method == "GET" and not path.startswith("/api/"):
                return self._static(path)
            if method == "GET" and path == "/api/session/stream":
                return self._stream()
            body: Dict[str, Any] = {}
            if method == "POST":
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    return self._json(400, {"error": "BAD_REQUEST", "message": "bad Content-Length"})
                if n > MAX_BODY:
                    return self._json(413, {"error": "TOO_LARGE", "message": "body too large"})
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError
                except ValueError:
                    return self._json(400, {"error": "BAD_REQUEST", "message": "body must be a JSON object"})
            auth = self.headers.get("Authorization", "")
            token = auth[7:] if auth.startswith("Bearer ") else None
            code, obj = app.dispatch(method, path, body, token)
            self._json(code, obj)

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")
    return Handler


def create_server(app: App, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    """Threaded server whose worker threads never block shutdown (open event streams would otherwise)."""
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    httpd.block_on_close = False
    return httpd


def serve(host: str = "127.0.0.1", port: int = 8080, app: Optional[App] = None) -> None:
    app = app or App()
    httpd = create_server(app, host, port)
    print(f"algoplatform UI on http://{host}:{port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutting_down = True
        httpd.server_close()
