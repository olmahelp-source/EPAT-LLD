"""Sample-data scenarios, the Alpaca and Interactive Brokers providers (against fakes), and the history library."""
import unittest
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import helpers  # noqa: F401
from algoplatform.core import (ConfigError, DataProviderError, MissingDataError, ProviderAuthError, ProviderUnavailableError,
                               ValidationError)
from algoplatform.data import (AlpacaDataProvider, HistoryLibrary, IBClient, IBDataProvider, MarketData, SampleDataProvider,
                               synthetic)


def bar(day, o=100.0, h=101.0, l=99.0, c=100.5, v=1000):
    return {"t": f"{day}T05:00:00Z", "o": o, "h": h, "l": l, "c": c, "v": v}


class FakeHttp:
    """Scripted (status, body) answers, one per call; records every URL and header set."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, url, headers):
        self.calls.append((url, dict(headers)))
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


class ScenarioTests(unittest.TestCase):
    def closes(self, name, seed=7, symbols=("AAA", "BBB", "CCC")):
        md = MarketData()
        for s, df in synthetic.scenario(name, list(symbols), n=756, seed=seed).items():
            md.add(s, df)                                          # every scenario's bars pass the strict OHLCV check
        return md.closes()

    def test_each_scenario_looks_like_its_name_for_several_seeds(self):
        for seed in (1, 7, 42):
            total = lambda c: (c.iloc[-1] / c.iloc[0] - 1)        # noqa: E731
            self.assertTrue((total(self.closes("bull_trend", seed)) > 0.3).all(), seed)
            self.assertTrue((total(self.closes("bear_trend", seed)) < -0.3).all(), seed)
            sideways = self.closes("sideways", seed)
            self.assertTrue(((sideways.max() / sideways.min()) < 1.6).all(), seed)
            crash = self.closes("crash_and_recovery", seed)
            self.assertTrue((((crash / crash.cummax()) - 1).min() < -0.2).all(), seed)
            vol = lambda c: np.log(c).diff().std().mean() * np.sqrt(252)   # noqa: E731
            self.assertGreater(vol(self.closes("high_volatility", seed)), 2 * vol(self.closes("bull_trend", seed)))

    def test_pairs_spreads_mean_revert(self):
        c = self.closes("pairs", 3, ("AAA", "BBB"))
        beta = np.polyfit(c["AAA"], c["BBB"], 1)[0]
        spread = c["BBB"] - beta * c["AAA"]
        lag = spread.autocorr(1)
        self.assertLess(lag, 0.99)                                 # a random walk would sit at about 1
        self.assertGreater(np.log(c).diff().corr().iloc[0, 1], 0.8)

    def test_reproducible_and_symbol_stable(self):
        a = synthetic.scenario("regime_switching", ["AAA", "BBB"], n=300, seed=5)
        b = synthetic.scenario("regime_switching", ["AAA", "BBB"], n=300, seed=5)
        alone = synthetic.scenario("regime_switching", ["BBB"], n=300, seed=5)
        pd.testing.assert_frame_equal(a["AAA"], b["AAA"])
        pd.testing.assert_frame_equal(a["BBB"], alone["BBB"])
        self.assertFalse(a["AAA"].equals(synthetic.scenario("regime_switching", ["AAA"], n=300, seed=6)["AAA"]))

    def test_intraday_sessions_and_bad_input(self):
        df = synthetic.scenario("random_walk", ["AAA"], n=200, seed=1, timeframe="15Min", start="2024-01-05")["AAA"]
        times = df.index
        self.assertEqual((times[0].hour, times[0].minute), (9, 30))
        self.assertTrue(((times.hour * 60 + times.minute) < 16 * 60).all())
        self.assertTrue((times.dayofweek < 5).all())
        for bad in ({"name": "moon"}, {"symbols": []}, {"n": 5}, {"corr": 1.0}, {"timeframe": "2Day"}):
            args = {"name": "random_walk", "symbols": ["AAA"], "n": 100, **bad}
            with self.assertRaises(ValidationError, msg=bad):
                synthetic.scenario(args.pop("name"), args.pop("symbols"), **args)


class SampleProviderTests(unittest.TestCase):
    def test_fetch_covers_the_dates_and_is_repeatable(self):
        p = SampleDataProvider("random_walk", seed=11)
        df = p.fetch("AAA", "2023-01-02", "2023-06-30")
        self.assertEqual(df.index[0], pd.Timestamp("2023-01-02"))
        self.assertEqual(len(df), len(pd.bdate_range("2023-01-02", "2023-06-30")))
        pd.testing.assert_frame_equal(df, SampleDataProvider("random_walk", seed=11).fetch("AAA", "2023-01-02", "2023-06-30"))
        self.assertFalse(p.external)

    def test_one_bad_symbol_does_not_stop_the_batch(self):
        r = SampleDataProvider().fetch_many(["AAA", "no good!", "BBB"], "2023-01-01", "2023-03-01")
        self.assertEqual(sorted(r.frames), ["AAA", "BBB"])
        self.assertIn("no good!", r.errors)
        with self.assertRaises(ValidationError):
            SampleDataProvider().fetch("AAA", "2023-03-01", "2023-01-01")          # start after end
        with self.assertRaises(ValidationError):
            SampleDataProvider("nope")


class AlpacaTests(unittest.TestCase):
    def provider(self, http, **kw):
        sleeps = []
        p = AlpacaDataProvider("AKID1234", "s3cret-value", transport=http, sleep=sleeps.append, **kw)
        return p, sleeps

    def test_pages_are_followed_and_daily_bars_keep_their_trading_date(self):
        http = FakeHttp((200, {"bars": [bar("2024-01-02"), bar("2024-01-03")], "next_page_token": "abc"}),
                        (200, {"bars": [bar("2024-01-04", c=101.0, h=102.0)], "next_page_token": None}))
        p, _ = self.provider(http, feed="sip")
        df = p.fetch("AAPL", "2024-01-01", "2024-01-10")
        self.assertEqual(list(df.index), [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")])
        self.assertEqual(df["close"].iloc[-1], 101.0)
        first, second = http.calls
        self.assertIn("/v2/stocks/AAPL/bars?", first[0])
        for part in ("timeframe=1Day", "feed=sip", "adjustment=all", "start=2024-01-01T00%3A00%3A00Z"):
            self.assertIn(part, first[0])
        self.assertIn("page_token=abc", second[0])
        self.assertEqual(first[1]["APCA-API-KEY-ID"], "AKID1234")
        self.assertNotIn("s3cret", repr(p) + str(p.describe()))

    def test_crypto_uses_the_crypto_endpoint_and_utc_dates(self):
        http = FakeHttp((200, {"bars": {"BTC/USD": [{"t": "2024-01-02T00:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 3}]}}))
        p, _ = self.provider(http)
        df = p.fetch("btc/usd", "2024-01-01", "2024-01-05")
        self.assertIn("/v1beta3/crypto/us/bars?", http.calls[0][0])
        self.assertIn("symbols=BTC%2FUSD", http.calls[0][0])
        self.assertNotIn("feed=", http.calls[0][0])
        self.assertEqual(df.index[0], pd.Timestamp("2024-01-02"))

    def test_rate_limits_are_retried_with_back_off(self):
        http = FakeHttp((429, {"message": "too many requests"}), (503, {}), (200, {"bars": [bar("2024-01-02")]}))
        p, sleeps = self.provider(http)
        self.assertEqual(len(p.fetch("MSFT", "2024-01-01", "2024-01-05")), 1)
        self.assertEqual(sleeps, [1.0, 2.0])
        p, sleeps = self.provider(FakeHttp((503, {})), retries=2)
        with self.assertRaises(ProviderUnavailableError) as ctx:
            p.fetch("MSFT", "2024-01-01", "2024-01-05")
        self.assertTrue(ctx.exception.retryable)
        self.assertEqual(sleeps, [1.0, 2.0])

    def test_bad_keys_stop_the_batch_and_never_leak(self):
        http = FakeHttp((401, {"message": "unauthorized"}))
        p, _ = self.provider(http)
        r = p.fetch_many(["AAPL", "MSFT", "GOOG"], "2024-01-01", "2024-01-05")
        self.assertEqual(len(http.calls), 1)                                         # no point hammering with bad keys
        self.assertIn("rejected the API keys", r.errors["AAPL"])
        self.assertIn("not attempted", r.errors["GOOG"])
        self.assertFalse(any("s3cret" in m for m in r.errors.values()))
        with self.assertRaises(ProviderAuthError):
            p.fetch("AAPL", "2024-01-01", "2024-01-05")

    def test_other_refusals_empty_ranges_and_dirty_bars(self):
        p, _ = self.provider(FakeHttp((422, {"message": "invalid timeframe"})))
        with self.assertRaises(DataProviderError) as ctx:
            p.fetch("AAPL", "2024-01-01", "2024-01-05")
        self.assertIn("invalid timeframe", ctx.exception.message)
        p, _ = self.provider(FakeHttp((200, {"bars": None})))
        with self.assertRaises(MissingDataError):
            p.fetch("AAPL", "2024-01-01", "2024-01-05")
        p, _ = self.provider(FakeHttp((200, {"bars": [bar("2024-01-02", h=99.5, c=100.5), bar("2024-01-03")]})))   # high below close
        df, report = p.fetch_with_report("AAPL", "2024-01-01", "2024-01-05")
        self.assertEqual(report["repaired_envelope"], 1)
        self.assertGreaterEqual(df["high"].iloc[0], df["close"].iloc[0])

    def test_configuration_is_checked(self):
        with self.assertRaises(ConfigError):
            AlpacaDataProvider("", "x")
        with self.assertRaises(ConfigError):
            AlpacaDataProvider("a", "b", base_url="http://data.alpaca.markets")
        with self.assertRaises(ValidationError):
            AlpacaDataProvider("a", "b", feed="nasdaq")


class FakeIB:
    """Answers each historical request with daily bars inside its window (including one duplicate across windows)."""

    def __init__(self):
        self.calls = []

    def historical(self, symbol, sec_type, exchange, currency, end, duration, bar_size, what_to_show, use_rth):
        self.calls.append(dict(symbol=symbol, sec_type=sec_type, exchange=exchange, currency=currency, end=end, duration=duration,
                               bar_size=bar_size, what=what_to_show))
        days = int(duration.split()[0]) * (365 if duration.endswith("Y") else 1)
        stamps = pd.bdate_range(end=pd.Timestamp(end).tz_localize(None).normalize(), periods=max(1, days * 5 // 7))
        return [{"date": d.date(), "open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": -1} for d in stamps]


class IBTests(unittest.TestCase):
    def test_long_daily_ranges_are_split_and_merged(self):
        fake = FakeIB()
        p = IBDataProvider(client=fake)
        df = p.fetch("AAPL", "2021-01-01", "2023-12-29")
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual({c["bar_size"] for c in fake.calls}, {"1 day"})
        self.assertEqual({(c["sec_type"], c["exchange"], c["what"]) for c in fake.calls}, {("STK", "SMART", "TRADES")})
        self.assertEqual(fake.calls[0]["end"].tzinfo, timezone.utc)
        self.assertTrue(df.index.is_unique and df.index.is_monotonic_increasing)
        self.assertTrue((df["volume"] == 0).all())                                  # IB's -1 "no volume" becomes 0
        self.assertGreaterEqual(df.index[0], pd.Timestamp("2021-01-01"))

    def test_forex_and_pacing_limit(self):
        fake = FakeIB()
        IBDataProvider(client=fake).fetch("EUR.USD", "2024-01-01", "2024-03-01")
        self.assertEqual((fake.calls[0]["sec_type"], fake.calls[0]["exchange"], fake.calls[0]["currency"], fake.calls[0]["what"]),
                         ("CASH", "IDEALPRO", "USD", "MIDPOINT"))
        fake = FakeIB()
        with self.assertRaises(DataProviderError) as ctx:
            IBDataProvider(client=fake, max_requests=50).fetch("AAPL", "2023-01-01", "2023-12-31", "1Min")
        self.assertIn("pacing", ctx.exception.message)
        self.assertEqual(fake.calls, [])                                             # refused before sending anything

    def test_plan_covers_the_range_without_gaps(self):
        p = IBDataProvider(client=FakeIB())
        start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-15")
        windows = p.plan(start, end, "1Hour")
        self.assertEqual(sum(d for _, d in windows), (end - start).days)
        self.assertEqual(windows[0][0], end)

    def test_real_client_explains_what_is_missing(self):
        with self.assertRaises(ConfigError) as ctx:
            IBClient().historical("AAPL", "STK", "SMART", "USD", datetime(2024, 1, 2), "1 D", "1 day", "TRADES", True)
        self.assertIn("ib_async", ctx.exception.message)
        with self.assertRaises(ValidationError):
            IBDataProvider(port=0, client=FakeIB())


class LibraryTests(unittest.TestCase):
    def frame(self, start, n, close=100.0):
        idx = pd.bdate_range(start, periods=n)
        c = np.full(n, close)
        return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": np.ones(n)}, index=idx)

    def test_merge_replace_and_round_trip(self):
        lib = HistoryLibrary()
        lib.put("AAA", "1Day", self.frame("2024-01-01", 10), "alpaca")
        lib.put("AAA", "1Day", self.frame("2024-01-08", 10, close=200.0), "alpaca")       # overlaps by a week: newer values win
        f = lib.get("AAA")
        self.assertEqual(len(f), len(pd.bdate_range("2024-01-01", periods=10).union(pd.bdate_range("2024-01-08", periods=10))))
        self.assertEqual(f.loc["2024-01-08", "close"], 200.0)
        lib.put("AAA", "1Day", self.frame("2024-01-01", 5, close=50.0), "ib")              # another provider replaces the series
        self.assertEqual((len(lib.get("AAA")), lib.summary()[0]["provider"]), (5, "ib"))
        copy = HistoryLibrary()
        self.assertEqual(copy.import_state(lib.export_state()), {"loaded": 1, "skipped": []})
        pd.testing.assert_frame_equal(copy.get("AAA"), lib.get("AAA"), check_freq=False)
        broken = lib.export_state()
        broken["series"].append({"symbol": "BAD", "timeframe": "1Day", "provider": "x", "index": ["2024-01-01"], "values": [[1, 0.5, 2, 1, 1]]})
        report = HistoryLibrary().import_state(broken)
        self.assertEqual((report["loaded"], len(report["skipped"])), (1, 1))

    def test_limits_and_missing_series_are_explained(self):
        lib = HistoryLibrary(max_rows_per_series=8, max_total_rows=12)
        lib.put("AAA", "1Day", self.frame("2024-01-01", 10), "sample")
        self.assertEqual(len(lib.get("AAA")), 8)                                           # the most recent bars are kept
        with self.assertRaises(ValidationError):
            lib.put("BBB", "1Day", self.frame("2024-01-01", 6), "sample")                  # 8 + 6 > 12
        with self.assertRaises(MissingDataError) as ctx:
            lib.market_data(["AAA", "ZZZ", "YYY"], min_rows=10)
        self.assertEqual(len(ctx.exception.details["problems"]), 3)
        v = lib.version
        self.assertTrue(lib.remove("AAA"))
        self.assertGreater(lib.version, v)
        self.assertFalse(lib.remove("AAA"))


if __name__ == "__main__":
    unittest.main()
