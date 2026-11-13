"""The command line: data sources for backtests, the data commands (with a fake Alpaca behind the real code path),
the sample-data-only lock, and the web command."""
import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

import helpers  # noqa: F401
from algoplatform.__main__ import run
from algoplatform.application.app import marketdata
from algoplatform.application.app.cli import main
from algoplatform.market_data.data import AlpacaDataProvider
from test_data_providers import FakeHttp, bar


def bars_for(n):
    return [bar(d.strftime("%Y-%m-%d"), c=100 + i * 0.2, h=101 + i * 0.2) for i, d in enumerate(pd.bdate_range("2023-01-02", periods=n))]


class CliCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name
        self.state = os.path.join(self.dir, "state")

    def cli(self, *argv, runner=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv), web_runner=runner) if runner is not None else main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def with_fake_alpaca(self, http):
        """Connect Alpaca from the environment, as a real deployment would, but answer from ``http``."""
        factory = lambda s, k: AlpacaDataProvider(k["key_id"], k["secret_key"], transport=http, sleep=lambda x: None)   # noqa: E731
        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(marketdata, "_alpaca", factory))
        stack.enter_context(mock.patch.dict(os.environ, {"ALPACA_API_KEY_ID": "AKCLI", "ALPACA_API_SECRET_KEY": "cli-secret-1"}))
        self.addCleanup(stack.close)


class SampleAndBacktestTests(CliCase):
    def test_scenarios_and_sample_files(self):
        code, out, _ = self.cli("data", "scenarios")
        self.assertEqual(code, 0)
        self.assertIn("crash_and_recovery", out)
        code, out, _ = self.cli("data", "sample", "--scenario", "bear_trend", "--symbols", "AAA,BBB", "--bars", "300", "--out", self.dir)
        self.assertEqual(code, 0)
        self.assertEqual(sorted(f for f in os.listdir(self.dir) if f.endswith(".csv")), ["AAA.csv", "BBB.csv"])
        df = pd.read_csv(os.path.join(self.dir, "AAA.csv"), index_col="date", parse_dates=True)
        self.assertEqual(len(df), 300)
        self.assertLess(df["close"].iloc[-1], df["close"].iloc[0])
        code, out, _ = self.cli("backtest", "--csv", f"AAA={os.path.join(self.dir, 'AAA.csv')}", "--json")
        self.assertEqual(code, 0)
        self.assertIn('"total_return"', out)

    def test_backtest_on_a_scenario_differs_from_the_mixed_universe(self):
        _, mixed, _ = self.cli("backtest", "--symbols", "AAA", "--bars", "400", "--json")
        _, bull, _ = self.cli("backtest", "--symbols", "AAA", "--bars", "400", "--scenario", "bull_trend", "--json")
        _, again, _ = self.cli("backtest", "--symbols", "AAA", "--bars", "400", "--scenario", "bull_trend", "--json")
        self.assertNotEqual(mixed, bull)
        self.assertEqual(bull, again)                                                   # reproducible

    def test_history_source_needs_downloads(self):
        with self.assertRaises(SystemExit) as ctx:
            self.cli("backtest", "--source", "history", "--db-dir", self.state)
        self.assertIn("data fetch", str(ctx.exception))


class DataCommandTests(CliCase):
    def test_fetch_list_export_backtest_and_remove(self):
        http = FakeHttp((200, {"bars": bars_for(250)}))
        self.with_fake_alpaca(http)
        code, out, err = self.cli("data", "providers", "--db-dir", self.state, "--test")
        self.assertEqual(code, 0)
        self.assertIn("ok: 250 bars of SPY", out)
        self.assertNotIn("cli-secret-1", out + err)
        code, out, err = self.cli("data", "fetch", "--provider", "alpaca", "--symbols", "aapl,msft", "--start", "2023-01-01",
                                  "--end", "2023-12-31", "--db-dir", self.state)
        self.assertEqual(code, 0, err)
        self.assertIn("AAPL", out)
        self.assertEqual(http.calls[-1][1]["APCA-API-SECRET-KEY"], "cli-secret-1")
        code, out, _ = self.cli("data", "list", "--db-dir", self.state, "--json")
        self.assertEqual([(r["symbol"], r["rows"]) for r in __import__("json").loads(out)], [("AAPL", 250), ("MSFT", 250)])
        target = os.path.join(self.dir, "aapl.csv")
        self.assertEqual(self.cli("data", "export", "AAPL", "--out", target, "--db-dir", self.state)[0], 0)
        self.assertEqual(len(pd.read_csv(target)), 250)
        code, out, _ = self.cli("backtest", "--source", "history", "--symbols", "AAPL", "--db-dir", self.state)
        self.assertEqual(code, 0)
        self.assertIn("sma_crossover on AAPL", out)
        self.assertEqual(self.cli("data", "remove", "MSFT", "--db-dir", self.state)[0], 0)
        self.assertEqual(self.cli("data", "remove", "MSFT", "--db-dir", self.state)[0], 1)

    def test_the_lock_is_shared_with_the_web_server_and_refuses_real_data(self):
        self.with_fake_alpaca(FakeHttp((200, {"bars": bars_for(100)})))
        self.cli("data", "fetch", "--provider", "alpaca", "--symbols", "AAPL", "--db-dir", self.state)
        code, out, _ = self.cli("data", "sample-only", "on", "--db-dir", self.state)
        self.assertIn("ON", out)
        code, _, err = self.cli("data", "fetch", "--provider", "alpaca", "--symbols", "MSFT", "--db-dir", self.state)
        self.assertEqual(code, 2)
        self.assertIn("SAMPLE_DATA_ONLY", err)
        self.assertEqual(self.cli("backtest", "--source", "history", "--db-dir", self.state)[0], 2)
        from algoplatform.application.web.server import App                                        # the web server reads the same lock
        app = App(db_dir=self.state)
        self.addCleanup(app.close)
        self.assertTrue(app.platform.data.sample_only)
        self.assertEqual(app.platform.data.library.symbols(), ["AAPL"])
        self.assertIn("off", self.cli("data", "sample-only", "off", "--db-dir", self.state)[1])

    def test_without_a_provider_the_error_says_how_to_connect(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "IB_HOST", "IB_PORT"):
                os.environ.pop(k, None)
            code, _, err = self.cli("data", "fetch", "--provider", "alpaca", "--symbols", "AAPL", "--db-dir", self.state)
        self.assertEqual(code, 2)
        self.assertIn("ALPACA_API_KEY_ID", err)


class WebCommandTests(CliCase):
    def test_web_starts_through_the_launcher(self):
        seen = []
        code, _, _ = self.cli("web", "--port", "9123", "--db-dir", self.state, runner=lambda a: seen.append((a.host, a.port, a.db_dir)) or 0)
        self.assertEqual((code, seen), (0, [("127.0.0.1", 9123, self.state)]))
        code, _, err = self.cli("web")                                                  # app.cli alone may not import the web layer
        self.assertEqual(code, 2)
        self.assertIn("python -m algoplatform web", err)
        with mock.patch("algoplatform.__main__.start_web", return_value=0) as started, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run(["structure", "--check"]), 0)                      # the launcher runs every other command as before
            started.assert_not_called()
            self.assertEqual(run(["web", "--port", "9124"]), 0)
            self.assertEqual(started.call_args[0][0].port, 9124)


if __name__ == "__main__":
    unittest.main()
