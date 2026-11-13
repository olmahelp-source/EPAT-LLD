"""Web routes for market data: sample preview, provider connections, downloads, the data source of every research
request, the sample-data-only lock, and paper trading on replayed history (including after a restart)."""
import tempfile
import unittest

import pandas as pd

import helpers  # noqa: F401
from algoplatform.market_data.data import AlpacaDataProvider
from test_data_providers import FakeHttp, bar
from test_web_quant import Client, WebQuantTestCase, make_app

KEYS = {"key_id": "AKWEB001", "secret_key": "web-secret-xyz"}


def bars_for(n, start="2022-01-03", drift=0.002):
    days = pd.bdate_range(start, periods=n)
    out, c = [], 100.0
    for i, d in enumerate(days):
        o = c
        c = c * (1 + drift * (1 if i % 3 else -1.5))
        out.append(bar(d.strftime("%Y-%m-%d"), o=o, h=max(o, c) * 1.01, l=min(o, c) * 0.99, c=c, v=1000 + i))
    return out


def install_fake_alpaca(app, http):
    app.platform.data.use_factory("alpaca", lambda s, k: AlpacaDataProvider(k["key_id"], k["secret_key"], transport=http, sleep=lambda x: None))


class DataWebCase(WebQuantTestCase):
    def connect_and_fetch(self, symbols=("AAPL", "MSFT"), n=300):
        install_fake_alpaca(self.app, FakeHttp((200, {"bars": bars_for(n)})))
        self.ok("operations", "POST", "/api/data/providers/connect", {"name": "alpaca", "settings": {"feed": "iex"}, "secrets": KEYS})
        return self.ok("quant", "POST", "/api/data/fetch", {"provider": "alpaca", "symbols": ",".join(symbols), "start": "2022-01-01", "end": "2023-12-31"})


class SourcesAndProvidersTests(DataWebCase):
    def test_everyone_sees_the_mode_only_operations_sees_settings(self):
        viewer = self.ok("viewer", "GET", "/api/data/sources")
        self.assertFalse(viewer["sample_only"])
        self.assertEqual(set(viewer["providers"][0]), {"name", "label", "help", "connected"})
        self.assertNotIn("environment", viewer)
        self.assertEqual((viewer["can_configure"], viewer["can_fetch"]), (False, False))
        ops = self.ok("operations", "GET", "/api/data/providers")
        self.assertIn("settings", ops["providers"][0])
        self.assertIn("pairs", ops["scenarios"])
        for who in ("quant", "trader", "viewer", "risk_manager"):
            self.assertEqual(self.call(who, "GET", "/api/data/providers")[0], 403, who)

    def test_connect_test_disconnect_and_keys_never_come_back(self):
        install_fake_alpaca(self.app, FakeHttp((200, {"bars": bars_for(10)})))
        r = self.ok("operations", "POST", "/api/data/providers/connect", {"name": "alpaca", "settings": {"feed": "iex"}, "secrets": KEYS})
        self.assertTrue(r["provider"]["connected"])
        test = self.ok("operations", "POST", "/api/data/providers/test", {"name": "alpaca"})
        self.assertTrue(test["result"]["ok"])
        everything = repr(r) + repr(test) + repr(self.ok("operations", "GET", "/api/data/providers")) + repr(self.ok("auditor", "GET", "/api/audit"))
        self.assertNotIn("web-secret-xyz", everything)
        self.assertEqual(self.call("trader", "POST", "/api/data/providers/connect", {"name": "alpaca", "secrets": KEYS})[0], 403)
        code, obj = self.call("operations", "POST", "/api/data/providers/connect", {"name": "alpaca", "secrets": {"key_id": "x"}})
        self.assertEqual(code, 400)
        self.assertIn("secret_key", obj["message"])
        self.assertFalse(self.ok("operations", "POST", "/api/data/providers/disconnect", {"name": "alpaca"})["providers"][0]["connected"])

    def test_sample_preview(self):
        r = self.ok("quant", "POST", "/api/data/sample/preview", {"scenario": "bear_trend", "symbols": ["AAA", "BBB"], "bars": 500, "seed": 3})
        self.assertTrue(all(x["total_return"] < 0 for x in r["stats"]))
        self.assertEqual(r["series"]["AAA"][0][1], 100.0)                                     # rebased
        self.assertEqual(r, self.ok("trader", "POST", "/api/data/sample/preview", {"scenario": "bear_trend", "symbols": ["AAA", "BBB"], "bars": 500, "seed": 3}))
        for bad in ({"scenario": "moon"}, {"bars": 10}, {"symbols": [f"S{i}" for i in range(7)]}):
            self.assertEqual(self.call("quant", "POST", "/api/data/sample/preview", bad)[0], 400, bad)
        self.assertEqual(self.call("viewer", "POST", "/api/data/sample/preview", {})[0], 403)


class DownloadAndSourceTests(DataWebCase):
    def test_download_then_research_on_real_history(self):
        r = self.connect_and_fetch()
        self.assertEqual(sorted(r["stored"]), ["AAPL", "MSFT"])
        lib = self.ok("viewer", "GET", "/api/data/sources")["library"]
        self.assertEqual([(x["symbol"], x["rows"]) for x in lib], [("AAPL", 300), ("MSFT", 300)])
        bt = self.ok("quant", "POST", "/api/backtest", {"strategy": "sma_crossover", "source": "history", "symbols": ["AAPL", "MSFT"]})
        self.assertEqual(bt["equity"][0][0][:10], "2022-01-03")                                 # it ran on the downloaded dates
        tech = self.ok("quant", "POST", "/api/lab/technical", {"source": "history", "symbols": ["AAPL"]})
        self.assertEqual(tech["symbol"], "AAPL")
        series = self.ok("quant", "POST", "/api/data/history/series", {"symbol": "AAPL"})
        self.assertEqual(len(series["close"]), 300)
        code, obj = self.call("quant", "POST", "/api/backtest", {"strategy": "sma_crossover", "source": "history", "symbols": ["TSLA"]})
        self.assertEqual(code, 400)
        self.assertIn("not downloaded", obj["message"])
        self.assertEqual(self.call("viewer", "POST", "/api/data/fetch", {"provider": "alpaca", "symbols": "AAPL"})[0], 403)
        self.ok("quant", "POST", "/api/data/history/remove", {"symbol": "MSFT"})
        self.assertEqual(self.call("quant", "POST", "/api/data/history/remove", {"symbol": "MSFT"})[0], 400)

    def test_sample_scenarios_reach_every_research_route(self):
        mixed = self.ok("quant", "POST", "/api/backtest", {"strategy": "sma_crossover", "symbols": ["AAA"], "bars": 400})
        legacy = self.ok("quant", "POST", "/api/backtest", {"strategy": "sma_crossover", "symbols": ["AAA"], "bars": 400, "source": "sample"})
        bull = self.ok("quant", "POST", "/api/backtest", {"strategy": "sma_crossover", "symbols": ["AAA"], "bars": 400, "source": "sample",
                                                          "scenario": "bull_trend"})
        self.assertEqual(mixed["equity"], legacy["equity"])                                     # the default is unchanged
        self.assertNotEqual(mixed["equity"], bull["equity"])
        self.assertEqual(self.call("quant", "POST", "/api/backtest", {"strategy": "sma_crossover", "source": "tape"})[0], 400)

    def test_the_sample_only_lock_is_enforced_on_the_server(self):
        self.connect_and_fetch(("AAPL",))
        self.ok("trader", "POST", "/api/data/upload", {"symbol": "UPL", "csv": "date,open,high,low,close\n" + "\n".join(
            f"{d.date()},10,11,9,10.5" for d in pd.bdate_range("2023-01-02", periods=160))})
        self.assertEqual(self.call("trader", "POST", "/api/data/sample_only", {"on": True})[0], 403)
        self.assertTrue(self.ok("operations", "POST", "/api/data/sample_only", {"on": True})["sample_only"])
        self.assertTrue(self.ok("viewer", "GET", "/api/data/sources")["sample_only"])
        refused = [("quant", "POST", "/api/data/fetch", {"provider": "alpaca", "symbols": "MSFT"}),
                   ("quant", "POST", "/api/backtest", {"strategy": "sma_crossover", "source": "history", "symbols": ["AAPL"]}),
                   ("trader", "POST", "/api/backtest", {"strategy": "sma_crossover", "dataset": "uploaded"}),
                   ("quant", "POST", "/api/data/history/series", {"symbol": "AAPL"}),
                   ("trader", "POST", "/api/session/open", {"symbols": ["AAPL"], "feed": "history"})]
        for who, method, path, body in refused:
            code, obj = self.call(who, method, path, body)
            self.assertEqual(code, 400, (path, obj))
            self.assertEqual(obj["error"], "SAMPLE_DATA_ONLY", path)
        self.ok("quant", "POST", "/api/backtest", {"strategy": "sma_crossover", "symbols": ["AAA"], "bars": 300, "scenario": "sideways"})
        self.ok("operations", "POST", "/api/data/sample_only", {"on": False})
        self.ok("quant", "POST", "/api/backtest", {"strategy": "sma_crossover", "source": "history", "symbols": ["AAPL"]})


class ReplayTests(DataWebCase):
    def test_paper_trading_replays_downloaded_bars_in_order(self):
        self.connect_and_fetch(("AAPL",), n=40)
        snap = self.ok("trader", "POST", "/api/session/open", {"symbols": ["AAPL"], "feed": "history"})
        closes = [b["c"] for b in bars_for(40)]
        self.assertEqual(snap["feed"], "history")
        self.assertEqual(snap["replay"]["pos"], 1)
        self.assertAlmostEqual(snap["prices"]["AAPL"], closes[0], places=4)
        snap = self.ok("trader", "POST", "/api/session/step", {"n": 9})
        self.assertAlmostEqual(snap["prices"]["AAPL"], closes[9], places=4)
        self.assertEqual(snap["replay"], {"pos": 10, "total": 40, "available": True, "from": "2022-01-03", "to": snap["replay"]["to"]})
        snap = self.ok("trader", "POST", "/api/session/step", {"n": 100})                      # runs to the end, keeps what played
        self.assertEqual(snap["replay"]["pos"], 40)
        self.assertAlmostEqual(snap["prices"]["AAPL"], closes[-1], places=4)
        code, obj = self.call("trader", "POST", "/api/session/step", {"n": 1})
        self.assertEqual(code, 400)
        self.assertIn("have been replayed", obj["message"])
        code, obj = self.call("trader", "POST", "/api/session/open", {"symbols": ["AAPL"], "feed": "tape"})
        self.assertEqual(code, 400)

    def test_lock_turned_on_mid_session_stops_the_replay(self):
        self.connect_and_fetch(("AAPL",), n=40)
        self.ok("trader", "POST", "/api/session/open", {"symbols": ["AAPL"], "feed": "history"})
        self.ok("operations", "POST", "/api/data/sample_only", {"on": True})
        code, obj = self.call("trader", "POST", "/api/session/step", {"n": 1})
        self.assertEqual((code, obj["error"]), (400, "SAMPLE_DATA_ONLY"))

    def test_a_replay_session_resumes_after_a_restart(self):
        with tempfile.TemporaryDirectory() as d:
            app1 = make_app(d)
            self.addCleanup(app1.close)
            self.app, self.call = app1, Client(app1)
            self.connect_and_fetch(("AAPL",), n=60)
            self.ok("trader", "POST", "/api/session/open", {"symbols": ["AAPL"], "feed": "history"})
            self.ok("trader", "POST", "/api/session/step", {"n": 4})
            before = self.ok("trader", "GET", "/api/session")
            app2 = make_app(d)                                                                    # new process, same directory
            self.addCleanup(app2.close)
            call2 = Client(app2)
            code, after = call2("trader", "GET", "/api/session")
            self.assertEqual(code, 200)
            self.assertTrue(after["resumed"])
            self.assertEqual((after["feed"], after["replay"]["pos"], after["replay"]["available"]), ("history", 5, True))
            self.assertEqual(after["prices"], before["prices"])
            code, nxt = call2("trader", "POST", "/api/session/step", {"n": 1})
            self.assertEqual(code, 200)
            self.assertAlmostEqual(nxt["prices"]["AAPL"], [b["c"] for b in bars_for(60)][5], places=4)
            self.assertEqual(app2.platform.restore_report["history"], {"loaded": 1, "skipped": []})
            self.assertEqual(app2.platform.restore_report["data"]["needs_keys"], ["alpaca"])       # keys are not kept on disk
            call2("trader", "POST", "/api/session/close", {})


if __name__ == "__main__":
    unittest.main()
