"""The market-data service: provider connections and their keys, downloads into the shared library, the
sample-data-only lock, start-up from the environment, and what survives a restart."""
import os
import tempfile
import unittest

import helpers
from algoplatform.app import TradingPlatform
from algoplatform.app.marketdata import SAMPLE_ONLY_CODE, MarketDataService
from algoplatform.core import AuthenticationError, AuthorizationError, DataProviderError, MissingDataError, PlatformConfig, Role, ValidationError
from algoplatform.data import AlpacaDataProvider, IBDataProvider
from algoplatform.persistence import SqliteStore
from algoplatform.users import PasswordHasher
from test_data_providers import FakeHttp, FakeIB, bar

KEYS = {"key_id": "AKFAKE01", "secret_key": "very-secret-123"}


def daily_bars(n=200):
    import pandas as pd
    return [bar(d.strftime("%Y-%m-%d"), c=100 + i * 0.1, h=101 + i * 0.1) for i, d in enumerate(pd.bdate_range("2023-01-02", periods=n))]


def fake_alpaca(http):
    return lambda settings, secrets: AlpacaDataProvider(secrets["key_id"], secrets["secret_key"], feed=settings.get("feed", "iex"),
                                                        transport=http, sleep=lambda s: None)


class PlatformCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "platform.db")

    def boot(self, store=True):
        st = SqliteStore(self.path) if store else None
        if st is not None:
            self.addCleanup(st.close)
        p = TradingPlatform(PlatformConfig(log_level="CRITICAL"), store=st)
        p.users.hasher = PasswordHasher(n=2 ** 4)
        admin = p.users.register("admin", helpers.STRONG_PW, Role.ADMIN)
        users = {r: p.users.register(r.value.lower(), helpers.STRONG_PW, r, actor=admin) for r in Role if r is not Role.ADMIN}
        users[Role.ADMIN] = admin
        return p, users


class ConnectionTests(PlatformCase):
    def test_only_operations_connects_and_keys_never_leave_the_vault(self):
        p, u = self.boot(store=False)
        http = FakeHttp((200, {"bars": daily_bars(10)}))
        p.data.use_factory("alpaca", fake_alpaca(http))
        for role in (Role.TRADER, Role.QUANT, Role.RISK_MANAGER, Role.VIEWER):
            with self.assertRaises(AuthorizationError, msg=role):
                p.data.connect(u[role], "alpaca", {"feed": "iex"}, KEYS)
        ops = p.desk(u[Role.OPERATIONS], "operations")
        info = ops.connect_data_provider("alpaca", {"feed": "sip", "ignored": "x"}, KEYS)
        self.assertTrue(info["connected"])
        self.assertEqual(info["settings"]["feed"], "sip")
        self.assertNotIn("ignored", info["settings"])
        visible = repr(p.data.status()) + repr(p.data.export_state()) + repr(p.audit.entries())
        self.assertNotIn("very-secret-123", visible)
        self.assertEqual(p.data.vault.get("data:alpaca"), KEYS)
        test = ops.test_data_provider("alpaca")
        self.assertEqual((test["ok"], test["symbol"], test["bars"]), (True, "SPY", 10))
        self.assertEqual(http.calls[0][1]["APCA-API-SECRET-KEY"], "very-secret-123")
        self.assertTrue(ops.disconnect_data_provider("alpaca"))
        with self.assertRaises(AuthenticationError):                                    # the keys are gone from the vault
            p.data.vault.get("data:alpaca")
        with self.assertRaises(ValidationError):
            ops.connect_data_provider("alpaca", {}, {"key_id": "only-half"})
        with self.assertRaises(ValidationError):
            ops.connect_data_provider("bloomberg", {}, {})

    def test_a_failing_connection_test_is_a_result_not_a_crash(self):
        p, u = self.boot(store=False)
        p.data.use_factory("alpaca", fake_alpaca(FakeHttp((403, {"message": "forbidden"}))))
        ops = p.desk(u[Role.OPERATIONS], "operations")
        ops.connect_data_provider("alpaca", {}, KEYS)
        r = ops.test_data_provider("alpaca")
        self.assertFalse(r["ok"])
        self.assertIn("rejected the API keys", r["error"])
        self.assertFalse(p.data.describe("alpaca")["status"]["ok"])


class FetchAndLockTests(PlatformCase):
    def setUp(self):
        super().setUp()
        self.p, self.u = self.boot(store=False)
        self.http = FakeHttp((200, {"bars": daily_bars(200)}), (422, {"message": "invalid symbol: ZZZZ"}))
        self.p.data.use_factory("alpaca", fake_alpaca(self.http))
        self.ops = self.p.desk(self.u[Role.OPERATIONS], "operations")
        self.ops.connect_data_provider("alpaca", {}, KEYS)

    def test_research_downloads_into_the_shared_library(self):
        research = self.p.desk(self.u[Role.QUANT], "research")
        r = research.fetch_history("alpaca", ["aapl", "ZZZZ", "aapl"], "2023-01-01", "2023-12-31")
        self.assertEqual(sorted(r["stored"]), ["AAPL"])                           # duplicates collapsed, bad symbol reported
        self.assertIn("invalid symbol", r["errors"]["ZZZZ"])
        self.assertTrue(self.p.data.describe("alpaca")["status"]["ok"])           # one unknown ticker is not a provider failure
        md = self.p.data.history(["AAPL"])
        self.assertEqual(len(md.frame("AAPL")), 200)
        self.assertEqual(self.p.data.library.summary()[0]["provider"], "alpaca")
        self.assertTrue(any(e["action"] == "data.fetch" for e in self.p.audit.entries()))
        backtest = research.backtest("sma_crossover", md, ["AAPL"])                  # downloaded history runs like any other data
        self.assertGreater(len(backtest.equity), 100)
        with self.assertRaises(AuthorizationError):
            self.p.data.fetch(self.u[Role.VIEWER], "alpaca", ["AAPL"])
        with self.assertRaises(ValidationError):
            research.fetch_history("ib", ["AAPL"])                                  # not connected
        with self.assertRaises(ValidationError):
            research.fetch_history("alpaca", [f"S{i}" for i in range(21)])
        research.fetch_history("alpaca", ["ZZZZ"])                                  # nothing loaded at all: that is a failure
        self.assertFalse(self.p.data.describe("alpaca")["status"]["ok"])
        self.assertTrue(research.remove_history("AAPL"))
        with self.assertRaises(MissingDataError):
            self.p.data.history(["AAPL"])

    def test_sample_only_lock_blocks_every_use_of_real_data(self):
        research = self.p.desk(self.u[Role.QUANT], "research")
        research.fetch_history("alpaca", ["AAPL"], "2023-01-01", "2023-12-31")
        with self.assertRaises(AuthorizationError):
            self.p.data.set_sample_only(self.u[Role.TRADER], True)
        self.assertTrue(self.ops.set_sample_data_only(True))
        calls = len(self.http.calls)
        for attempt in (lambda: research.fetch_history("alpaca", ["MSFT"]), lambda: self.p.data.history(["AAPL"])):
            with self.assertRaises(DataProviderError) as ctx:
                attempt()
            self.assertEqual(ctx.exception.code, SAMPLE_ONLY_CODE)
        self.assertEqual(len(self.http.calls), calls)                              # nothing was sent
        self.assertEqual(len(self.p.data.sample("bull_trend", ["AAA", "BBB"], bars=100)), 2)   # sample data still works
        self.assertFalse(self.ops.set_sample_data_only(False))
        self.assertEqual(len(self.p.data.history(["AAPL"]).frame("AAPL")), 200)


class EnvironmentTests(unittest.TestCase):
    def test_keys_and_gateway_from_the_environment(self):
        svc = MarketDataService(env={"APCA_API_KEY_ID": "AK1", "APCA_API_SECRET_KEY": "SK1", "ALPACA_DATA_FEED": "sip",
                                     "IB_HOST": "10.0.0.5", "IB_PORT": "4002"})
        self.assertEqual(sorted(svc.status()["connected"]), ["alpaca", "ib"])
        self.assertEqual(svc.describe("alpaca")["settings"]["feed"], "sip")
        self.assertEqual(svc.describe("alpaca")["settings"]["connected_by"], "environment")
        ib = svc.provider("ib")
        self.assertIsInstance(ib, IBDataProvider)
        self.assertEqual((ib.host, ib.port), ("10.0.0.5", 4002))

    def test_a_bad_setting_is_reported_not_fatal(self):
        svc = MarketDataService(env={"IB_PORT": "seventy"})
        self.assertIn("not connected", svc.environment_report["ib"])
        self.assertEqual(svc.status()["connected"], [])
        self.assertEqual(MarketDataService(env={}).status()["connected"], [])


class RestartTests(PlatformCase):
    def test_history_lock_and_gateway_settings_survive_keys_do_not(self):
        p, u = self.boot()
        p.data.use_factory("alpaca", fake_alpaca(FakeHttp((200, {"bars": daily_bars(150)}))))
        p.data.use_factory("ib", lambda s, k: IBDataProvider(host=s.get("host", "127.0.0.1"), port=int(s.get("port", 7497)), client=FakeIB()))
        ops = p.desk(u[Role.OPERATIONS], "operations")
        ops.connect_data_provider("alpaca", {"feed": "iex"}, KEYS)
        ops.connect_data_provider("ib", {"host": "127.0.0.1", "port": 4002, "client_id": 9}, {})
        p.desk(u[Role.QUANT], "research").fetch_history("alpaca", ["AAPL"], "2023-01-01", "2023-12-31")
        ops.set_sample_data_only(True)
        p.store.close()

        q, _ = self.boot()
        self.assertEqual(q.restore_report["data"], {"sample_only": True, "restored": ["ib"], "needs_keys": ["alpaca"]})
        self.assertEqual(q.restore_report["history"], {"loaded": 1, "skipped": []})
        self.assertEqual(q.data.library.summary()[0]["rows"], 150)
        self.assertEqual(q.data.describe("ib")["settings"]["port"], 4002)
        self.assertFalse(q.data.describe("alpaca")["connected"])
        with open(self.path, "rb") as fh:
            self.assertNotIn(b"very-secret-123", fh.read())

    def test_unchanged_history_is_not_rewritten_on_every_save(self):
        p, u = self.boot()
        p.data.use_factory("alpaca", fake_alpaca(FakeHttp((200, {"bars": daily_bars(50)}))))
        p.desk(u[Role.OPERATIONS], "operations").connect_data_provider("alpaca", {}, KEYS)
        p.desk(u[Role.QUANT], "research").fetch_history("alpaca", ["AAPL"], "2023-01-01", "2023-06-30")
        writes = []
        real_put = p.store.put
        p.store.put = lambda key, value: (writes.append(key), real_put(key, value))[1]
        p.desk(u[Role.QUANT], "research").register_strategy("trend")                    # an unrelated change saves the platform
        self.assertIn("platform.lifecycle", writes)
        self.assertNotIn("platform.history", writes)
        writes.clear()
        p.desk(u[Role.QUANT], "research").remove_history("AAPL")
        self.assertIn("platform.history", writes)


if __name__ == "__main__":
    unittest.main()
