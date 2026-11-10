import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from helpers import free_config
from algoplatform.backtest import BacktestEngine
from algoplatform.core.exceptions import StrategyError, ValidationError
from algoplatform.data import MarketData
from algoplatform.sentiment import (EntityMapper, NewsStore, SentimentSignal, SentimentStrategy, classify_event, score_text)

T0 = datetime(2024, 3, 1, 10, 0)


class LexiconTests(unittest.TestCase):
    def test_polarity(self):
        self.assertGreater(score_text("Company beats estimates on strong growth")["score"], 0.8)
        self.assertLess(score_text("Profit warning as sales plunge and losses widen")["score"], -0.8)
        neutral = score_text("The meeting is scheduled for Tuesday")
        self.assertEqual((neutral["score"], neutral["n_terms"]), (0.0, 0))
        self.assertAlmostEqual(score_text("strong growth but weak margins")["score"], 1 / 3)       # 2 positive vs 1 negative
        self.assertLess(score_text("Profit warning from Acme")["score"], -0.9)                       # phrase beats the word 'profit'
        self.assertGreater(score_text("Acme hits all-time high on record profit")["score"], 0.9)

    def test_negation_and_intensifiers(self):
        self.assertLess(score_text("results were not strong")["score"], 0)                        # 'not strong' flips
        self.assertGreater(score_text("no loss this quarter")["score"], 0)                          # 'no loss' flips
        self.assertEqual(score_text("strong")["pos"], 1.0)
        self.assertEqual(score_text("very strong")["pos"], 1.5)
        self.assertEqual(score_text("slightly strong")["pos"], 0.5)
        self.assertLess(score_text("strong results", negation_window=3)["score"] * -1, 0)
        far = score_text("not that anyone expected such a strong quarter", negation_window=3)["score"]
        self.assertGreater(far, 0)                                                                   # negator too far away to count


class EventTests(unittest.TestCase):
    def test_event_types_and_direction(self):
        ev = dict(classify_event("Acme beats earnings estimates"))
        self.assertGreater(ev["earnings_beat"], 0)
        self.assertLess(dict(classify_event("Acme cuts full-year guidance"))["guidance_cut"], 0)
        self.assertIn("approval", dict(classify_event("FDA approval for Acme drug")))
        self.assertEqual(dict(classify_event("Acme files for Chapter 11 bankruptcy"))["bankruptcy"], -1.0)
        self.assertIn("downgrade", dict(classify_event("Analyst downgraded Acme to sell")))
        self.assertEqual(classify_event("Acme opens a new office"), [])
        self.assertIn("dividend_cut", dict(classify_event("Acme suspends dividend")))

    def test_entity_mapping(self):
        m = EntityMapper({"AAPL": ["Apple"], "ON": ["ON Semiconductor"], "A": ["Agilent"]})
        self.assertEqual(m.symbols("Apple unveils new phone"), {"AAPL"})
        self.assertEqual(m.symbols("AAPL shares rise"), {"AAPL"})
        self.assertEqual(m.symbols("ON Semiconductor wins design"), {"ON"})
        self.assertEqual(m.symbols("turn on the lights"), set())                                    # short tickers are case-sensitive
        self.assertEqual(m.symbols("A new report from Agilent"), {"A"})
        self.assertEqual(m.symbols("A new report"), set())                                          # 1-letter ticker never matches the article
        self.assertEqual(m.symbols("Pineapple harvest"), set())                                     # whole words only


def store(**kw):
    return NewsStore(EntityMapper({"ACME": ["Acme"], "BETA": ["Beta Corp"]}), **kw)


class NewsStoreTests(unittest.TestCase):
    def test_dedupe_window_and_validation(self):
        s = store()
        self.assertIsNotNone(s.add(T0, "wire", "Acme beats estimates!"))
        self.assertIsNone(s.add(T0 + timedelta(hours=1), "blog", "acme BEATS estimates"))           # same story, other outlet
        self.assertEqual(s.duplicates, 1)
        self.assertIsNotNone(s.add(T0 + timedelta(hours=13), "wire", "Acme beats estimates"))       # old enough to count again
        self.assertIsNotNone(s.add(T0, "wire", "Acme raises guidance"))
        self.assertEqual(len(s), 3)
        with self.assertRaises(ValidationError):
            s.add(T0, "wire", "   ")

    def test_point_in_time_with_latency(self):
        s = store(default_latency=timedelta(minutes=5))
        s.add(T0, "wire", "Acme beats estimates")
        w = timedelta(days=1)
        self.assertEqual(s.asof("ACME", T0 + timedelta(minutes=3), w), [])                          # published but not yet actionable
        self.assertEqual(len(s.asof("ACME", T0 + timedelta(minutes=6), w)), 1)
        self.assertEqual(s.asof("ACME", T0 + timedelta(days=3), w), [])                              # outside the window
        self.assertEqual(s.asof("BETA", T0 + timedelta(minutes=6), w), [])                           # other symbol

    def test_annotation(self):
        it = store().add(T0, "wire", "Acme cuts guidance after weak quarter")
        self.assertEqual(it.symbols, {"ACME"})
        self.assertLess(it.score, 0)
        self.assertLess(it.event_weight, 0)
        self.assertIn("guidance_cut", it.events)


class SignalTests(unittest.TestCase):
    def test_decay_source_weight_and_accumulation(self):
        s = store(source_weights={"reuters": 1.0}, unknown_source_weight=0.5, default_latency=timedelta(0))
        s.add(T0, "reuters", "Acme strong growth")
        sig = SentimentSignal(s, half_life=timedelta(hours=24), scale=0.01, event_weight=0.0)
        now, later = sig.score("ACME", T0), sig.score("ACME", T0 + timedelta(hours=24))
        self.assertAlmostEqual(later / now, 0.5, places=3)                                           # one half-life: half the impact
        s2 = store(source_weights={"reuters": 1.0}, unknown_source_weight=0.5, default_latency=timedelta(0))
        s2.add(T0, "random_blog", "Acme strong growth")
        self.assertAlmostEqual(SentimentSignal(s2, scale=0.01, event_weight=0.0).score("ACME", T0) / now, 0.5, places=3)
        s.add(T0 + timedelta(minutes=1), "reuters", "Acme record profit")
        self.assertGreater(sig.score("ACME", T0 + timedelta(minutes=2)), now)
        self.assertLess(abs(sig.score("ACME", T0)), 1.0)                                              # bounded
        self.assertEqual(sig.score("BETA", T0), 0.0)

    def test_events_add_to_text_sentiment(self):
        s = store(default_latency=timedelta(0), source_weights={"wire": 1.0})
        s.add(T0, "wire", "Acme files for bankruptcy")
        self.assertLess(SentimentSignal(s, scale=1.0).score("ACME", T0), -0.9)


def market(n=120):
    idx = pd.bdate_range("2024-01-01", periods=n)
    c = np.full(n, 100.0)
    md = MarketData()
    md.add("X", pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1e6}, index=idx))
    return md, idx


class SentimentStrategyTests(unittest.TestCase):
    def build(self, idx, news):
        s = NewsStore(EntityMapper({"X": ["Xeno"]}), default_latency=timedelta(minutes=1))
        for day, text in news:
            s.add(idx[day].to_pydatetime() + timedelta(hours=16), "wire", text)
        return SentimentStrategy.with_signal(SentimentSignal(s, half_life=timedelta(days=3), scale=1.0), ["X"], threshold=0.3)

    def test_trades_after_news_and_not_before(self):
        md, idx = market()
        res = BacktestEngine(md, free_config()).run(self.build(idx, [(40, "Xeno beats estimates on strong growth")]))
        self.assertTrue(res.fills)
        self.assertGreater(min(f.ts for f in res.fills), idx[40])                                    # news at 16:00 acts on a later bar
        self.assertEqual(res.summary["positions"], {})                                                # decayed away, position closed

    def test_future_news_cannot_change_the_past(self):
        md, idx = market()
        base = [(40, "Xeno beats estimates on strong growth")]
        a = BacktestEngine(md, free_config()).run(self.build(idx, base))
        b = BacktestEngine(md, free_config()).run(self.build(idx, base + [(90, "Xeno shares plunge on fraud probe")]))
        pd.testing.assert_series_equal(a.equity.loc[:idx[89]], b.equity.loc[:idx[89]])

    def test_short_side_and_validation(self):
        md, idx = market()
        s = self.build(idx, [(40, "Xeno files for bankruptcy amid fraud probe")])
        s.params["allow_short"] = True
        res = BacktestEngine(md, free_config()).run(s)
        self.assertTrue(any(f.side.value == "SELL" for f in res.fills))
        with self.assertRaises(StrategyError):
            SentimentStrategy(["X"], threshold=2.0)


if __name__ == "__main__":
    unittest.main()
