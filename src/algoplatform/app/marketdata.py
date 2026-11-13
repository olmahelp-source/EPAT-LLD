"""The market-data service: connected providers, the shared history library and the sample-data-only lock.

* **Providers.** Operations connects Alpaca (API key ID and secret) or Interactive Brokers (host, port, client id of a
  running TWS / IB Gateway). Keys go into the platform's credential vault (encrypted when ``cryptography`` is
  installed) and are never written to the state store or shown again; after a restart they come back only from the
  environment (``ALPACA_API_KEY_ID`` / ``ALPACA_API_SECRET_KEY``, or the ``APCA_*`` names Alpaca's own tools use) or by
  connecting again. IB settings hold no secret and are restored.
* **History.** Research downloads bars into one library shared by everyone (market history is not private).
* **Sample-data-only lock.** When on, nothing real is used: downloading, using downloaded history and using uploaded
  files are refused, so every page runs on generated data (training, demos, a fresh install). Operations switches it.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from ..core.enums import Permission
from ..core.exceptions import DataProviderError, PlatformError, ValidationError
from ..core.logging import AuditLog, get_logger
from ..data import synthetic
from ..data.library import HistoryLibrary
from ..data.providers import (TIMEFRAMES, AlpacaDataProvider, DataProvider, IBDataProvider, SampleDataProvider, check_symbol,
                              check_timeframe)
from ..data.store import MarketData
from ..users.manager import User, UserManager
from ..users.security import CredentialVault

log = get_logger(__name__)

PROVIDERS: Dict[str, Dict[str, Any]] = {
    "alpaca": {"label": "Alpaca", "settings": ("feed", "adjustment"), "secrets": ("key_id", "secret_key"),
               "test_symbol": "SPY", "help": "US stocks and crypto. Use the API keys of an Alpaca account; the free plan gives the IEX feed."},
    "ib": {"label": "Interactive Brokers", "settings": ("host", "port", "client_id"), "secrets": (),
           "test_symbol": "SPY", "help": "Needs Trader Workstation or IB Gateway running with the API enabled, and the ib_async package."},
}
SAMPLE_ONLY_CODE = "SAMPLE_DATA_ONLY"
MAX_SYMBOLS_PER_FETCH = 20


def _alpaca(settings: Mapping[str, Any], secrets: Mapping[str, str]) -> DataProvider:
    return AlpacaDataProvider(secrets.get("key_id", ""), secrets.get("secret_key", ""), feed=str(settings.get("feed") or "iex"),
                              adjustment=str(settings.get("adjustment") or "all"))


def _ib(settings: Mapping[str, Any], secrets: Mapping[str, str]) -> DataProvider:
    return IBDataProvider(host=str(settings.get("host") or "127.0.0.1"), port=int(settings.get("port") or 7497),
                          client_id=int(settings.get("client_id") or 17))


class MarketDataService:
    def __init__(self, vault: Optional[CredentialVault] = None, audit: Optional[AuditLog] = None, env: Optional[Mapping[str, str]] = None,
                 factories: Optional[Dict[str, Callable[[Mapping[str, Any], Mapping[str, str]], DataProvider]]] = None) -> None:
        """``factories`` replaces how a provider is built from its settings and secrets (tests pass fakes here)."""
        self.vault = vault or CredentialVault()
        self.audit = audit or AuditLog()
        self.library = HistoryLibrary()
        self.sample_only = False
        self._factories = {"alpaca": _alpaca, "ib": _ib, **(factories or {})}
        self._providers: Dict[str, DataProvider] = {}
        self._settings: Dict[str, Dict[str, Any]] = {}
        self._status: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.state_version = 0
        self.environment_report = self._from_environment(os.environ if env is None else env)

    # ---- connections ----------------------------------------------------------------------------------------------------
    def use_factory(self, name: str, factory: Callable[[Mapping[str, Any], Mapping[str, str]], DataProvider]) -> None:
        """Change how provider ``name`` is built from (settings, secrets): for tests with a fake transport or client, or a
        firm's own variant of a provider. Takes effect at the next connect."""
        if name not in PROVIDERS:
            raise ValidationError(f"unknown data provider '{name}'", details={"valid": sorted(PROVIDERS)})
        self._factories[name] = factory

    def _install(self, name: str, settings: Mapping[str, Any], secrets: Mapping[str, str], by: str) -> DataProvider:
        if name not in PROVIDERS:
            raise ValidationError(f"unknown data provider '{name}'", details={"valid": sorted(PROVIDERS)})
        spec = PROVIDERS[name]
        clean = {k: settings[k] for k in spec["settings"] if settings.get(k) not in (None, "")}
        missing = [k for k in spec["secrets"] if not str(secrets.get(k) or "").strip()]
        if missing:
            raise ValidationError(f"{spec['label']} needs {', '.join(missing)}")
        provider = self._factories[name](clean, {k: str(secrets[k]).strip() for k in spec["secrets"]})
        with self._lock:
            if spec["secrets"]:
                self.vault.put(f"data:{name}", {k: str(secrets[k]).strip() for k in spec["secrets"]})
            self._providers[name] = provider
            self._settings[name] = {**clean, "connected_by": by, "connected_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
            self._status.pop(name, None)
            self.state_version += 1
        return provider

    def _from_environment(self, env: Mapping[str, str]) -> Dict[str, str]:
        """Connect what the server's environment configures. A bad setting is reported, never fatal at start-up."""
        report: Dict[str, str] = {}
        key = env.get("ALPACA_API_KEY_ID") or env.get("APCA_API_KEY_ID")
        secret = env.get("ALPACA_API_SECRET_KEY") or env.get("APCA_API_SECRET_KEY")
        wanted = {"alpaca": ({"feed": env.get("ALPACA_DATA_FEED", "iex")}, {"key_id": key or "", "secret_key": secret or ""}) if key and secret else None,
                  "ib": ({"host": env.get("IB_HOST", "127.0.0.1"), "port": env.get("IB_PORT", "7497"), "client_id": env.get("IB_CLIENT_ID", "17")}, {})
                  if env.get("IB_PORT") or env.get("IB_HOST") else None}
        for name, conf in wanted.items():
            if conf is None:
                continue
            try:
                self._install(name, conf[0], conf[1], "environment")
                report[name] = "connected from the environment"
            except (PlatformError, ValueError) as exc:
                report[name] = f"not connected: {getattr(exc, 'message', exc)}"
                log.error("data provider %s from the environment: %s", name, report[name])
        return report

    def connect(self, user: User, name: str, settings: Optional[Mapping[str, Any]] = None, secrets: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        UserManager.authorize(user, Permission.OPERATE_SYSTEM)
        self._install(name, settings or {}, secrets or {}, user.username)
        self.audit.record(user.username, "data.provider.connect", provider=name,
                          settings={k: v for k, v in self._settings[name].items() if k not in ("connected_by", "connected_at")})
        return self.describe(name)

    def disconnect(self, user: User, name: str) -> bool:
        UserManager.authorize(user, Permission.OPERATE_SYSTEM)
        with self._lock:
            gone = self._providers.pop(name, None) is not None
            self._settings.pop(name, None)
            self._status.pop(name, None)
            self.vault.delete(f"data:{name}")
            self.state_version += 1
        if gone:
            self.audit.record(user.username, "data.provider.disconnect", provider=name)
        return gone

    def provider(self, name: str) -> DataProvider:
        if name in self._providers:
            return self._providers[name]
        label = PROVIDERS.get(name, {}).get("label", name)
        raise ValidationError(f"{label} is not connected: operations connects it on the Data providers page", code="PROVIDER_NOT_CONNECTED")

    def test(self, user: User, name: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Fetch the last two weeks of one liquid symbol, to prove the connection and the keys work."""
        UserManager.authorize(user, Permission.OPERATE_SYSTEM)
        sym = check_symbol(symbol or PROVIDERS.get(name, {}).get("test_symbol", "SPY"))
        end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
        try:
            df = self.provider(name).fetch(sym, end - pd.Timedelta(days=14), end, "1Day")
            result = {"ok": True, "symbol": sym, "bars": len(df), "last_date": df.index[-1].date().isoformat(), "last_close": float(df["close"].iloc[-1])}
        except PlatformError as exc:
            result = {"ok": False, "symbol": sym, "error": exc.message, "code": exc.code}
        self._status[name] = {**result, "at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
        self.audit.record(user.username, "data.provider.test", provider=name, ok=result["ok"])
        return result

    # ---- the lock -------------------------------------------------------------------------------------------------------------
    def set_sample_only(self, user: User, on: bool) -> bool:
        UserManager.authorize(user, Permission.OPERATE_SYSTEM)
        with self._lock:
            self.sample_only = bool(on)
            self.state_version += 1
        self.audit.record(user.username, "data.sample_only", on=self.sample_only)
        return self.sample_only

    def require_real_data_allowed(self, what: str) -> None:
        if self.sample_only:
            raise DataProviderError(f"{what} is switched off: the platform is in sample-data-only mode (operations can change "
                                    "this on the Data providers page)", code=SAMPLE_ONLY_CODE)

    # ---- research ---------------------------------------------------------------------------------------------------------
    def fetch(self, user: User, name: str, symbols: Sequence[str], start: Any = None, end: Any = None, timeframe: str = "1Day") -> Dict[str, Any]:
        """Download history into the shared library. Every symbol is attempted; the report says what loaded and why the
        rest did not."""
        UserManager.authorize(user, Permission.RUN_BACKTEST)
        self.require_real_data_allowed("downloading market data")
        tf = check_timeframe(timeframe)
        symbols = list(dict.fromkeys(str(s).strip().upper() for s in symbols if str(s).strip()))
        if not 1 <= len(symbols) <= MAX_SYMBOLS_PER_FETCH:
            raise ValidationError(f"download 1 to {MAX_SYMBOLS_PER_FETCH} symbols at a time")
        provider = self.provider(name)
        result = provider.fetch_many(symbols, start, end, tf)
        stored: Dict[str, Any] = {}
        for sym, frame in result.frames.items():
            try:
                stored[sym] = self.library.put(sym, tf, frame, name, result.cleaning.get(sym)).summary()
            except PlatformError as exc:                       # e.g. the library is full: report it for this symbol, keep going
                result.errors[sym] = exc.message
        self._status[name] = {"ok": not result.errors, "last_fetch": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
                              "loaded": len(stored), "failed": len(result.errors)}
        self.audit.record(user.username, "data.fetch", provider=name, timeframe=tf, loaded=sorted(stored), failed=sorted(result.errors))
        return {"provider": name, "timeframe": tf, "stored": stored, "errors": result.errors, "cleaning": result.cleaning}

    def history(self, symbols: Sequence[str], timeframe: str = "1Day", min_rows: int = 0) -> MarketData:
        self.require_real_data_allowed("using downloaded market data")
        return self.library.market_data(list(symbols), check_timeframe(timeframe), min_rows)

    def remove_history(self, user: User, symbol: str, timeframe: str = "1Day") -> bool:
        UserManager.authorize(user, Permission.RUN_BACKTEST)
        gone = self.library.remove(check_symbol(symbol), check_timeframe(timeframe))
        if gone:
            self.audit.record(user.username, "data.history.remove", symbol=symbol, timeframe=timeframe)
        return gone

    @staticmethod
    def sample(scenario: str, symbols: Sequence[str], bars: int = 756, seed: int = 7, start: Any = "2020-01-01",
               timeframe: str = "1Day") -> Dict[str, pd.DataFrame]:
        return SampleDataProvider(scenario, seed).generate(symbols, bars, start, timeframe)

    # ---- views ----------------------------------------------------------------------------------------------------------------
    def describe(self, name: str) -> Dict[str, Any]:
        spec = PROVIDERS[name]
        connected = name in self._providers
        settings = {k: v for k, v in self._settings.get(name, {}).items()}
        return {"name": name, "label": spec["label"], "help": spec["help"], "connected": connected, "settings": settings,
                "needs": list(spec["settings"]) + list(spec["secrets"]), "secret_fields": list(spec["secrets"]),
                "status": dict(self._status.get(name, {}))}

    def status(self) -> Dict[str, Any]:
        return {"sample_only": self.sample_only, "providers": [self.describe(n) for n in PROVIDERS],
                "connected": sorted(self._providers), "scenarios": dict(synthetic.SCENARIOS), "timeframes": list(TIMEFRAMES),
                "library": self.library.summary(), "environment": dict(self.environment_report)}

    # ---- persistence: settings only, never secrets ---------------------------------------------------------------------------
    def export_state(self) -> Dict[str, Any]:
        return {"sample_only": self.sample_only,
                "providers": {n: {k: v for k, v in s.items() if k in PROVIDERS[n]["settings"]} for n, s in self._settings.items()}}

    def import_state(self, state: Mapping[str, Any]) -> Dict[str, Any]:
        """Restore the lock and the provider settings. A provider that needs keys is listed as ``needs_keys`` unless the
        environment already connected it."""
        self.sample_only = bool(state.get("sample_only", False))
        restored, needs_keys = [], []
        for name, settings in dict(state.get("providers") or {}).items():
            if name not in PROVIDERS or name in self._providers:
                continue
            if PROVIDERS[name]["secrets"]:
                needs_keys.append(name)
                continue
            try:
                self._install(name, settings, {}, "restored")
                restored.append(name)
            except (PlatformError, ValueError) as exc:
                needs_keys.append(f"{name} ({getattr(exc, 'message', exc)})")
        return {"sample_only": self.sample_only, "restored": restored, "needs_keys": needs_keys}


__all__: List[str] = ["MarketDataService", "PROVIDERS", "SAMPLE_ONLY_CODE"]
