import math
import os
import tempfile
import unittest
from datetime import date

import numpy as np
import pandas as pd

import helpers  # noqa: F401
from algoplatform.core.exceptions import AggregateError, DataError, InstrumentError, MissingDataError, ValidationError
from algoplatform.derivatives import OptionChain, iv_percentile, iv_rank, option_symbol, synthetic_chain
from algoplatform.instruments import (AssetClass, ContractSpec, CorporateAction, InstrumentMaster, actions_from_records, adjust_ohlcv,
                                      adjustment_factors, continuous_series, dividend_cash, future_spec, future_symbol, option_spec,
                                      parse_option_symbol, roll_schedule, unadjusted_position)
from algoplatform.quant import bs_price


def flat_frame(n, price=100.0, start="2024-01-01", volume=1000.0):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"open": price, "high": price * 1.01, "low": price * 0.99, "close": price, "volume": volume}, index=idx)


class ContractSpecTests(unittest.TestCase):
    def test_validation(self):
        for kwargs in ({"symbol": ""}, {"symbol": " X"}, {"symbol": "X", "tick_size": 0}, {"symbol": "X", "asset_class": "widget"},
                       {"symbol": "X", "margin_pct": 0}, {"symbol": "F", "asset_class": "future"},
                       {"symbol": "O", "asset_class": "option", "underlying": "X", "expiry": "2025-01-17", "right": "call"},
                       {"symbol": "O", "asset_class": "option", "underlying": "X", "expiry": "2025-01-17", "right": "both", "strike": 10},
                       {"symbol": "F", "asset_class": "future", "underlying": "ES", "expiry": "not-a-date"}):
            with self.assertRaises(InstrumentError, msg=str(kwargs)):
                ContractSpec(**kwargs)

    def test_views_and_expiry(self):
        fut = future_spec("ES", "2025-03-21", multiplier=50, tick_size=0.25, broker_symbols={"IBKR": "ESH5"})
        self.assertEqual((fut.symbol, fut.asset_class, fut.expiry), ("ESH25", AssetClass.FUTURE, date(2025, 3, 21)))
        self.assertEqual(fut.days_to_expiry("2025-03-01"), 20)
        self.assertFalse(fut.is_expired("2025-03-21"))
        self.assertTrue(fut.is_expired("2025-03-22"))
        self.assertEqual(fut.notional(5000, -2), 500_000)
        self.assertEqual((fut.broker_symbol("ibkr"), fut.broker_symbol("other")), ("ESH5", "ESH25"))
        inst = fut.to_instrument()
        self.assertEqual((inst.multiplier, inst.tick_size, inst.asset_class), (50, 0.25, "future"))
        self.assertIsNone(ContractSpec("AAA").days_to_expiry("2025-01-01"))
        self.assertEqual(future_symbol("CL", date(2026, 12, 1)), "CLZ26")

    def test_option_symbol_round_trip_with_the_derivatives_package(self):
        spec = option_spec("AAPL", date(2025, 1, 17), "put", 152.5)
        self.assertEqual(spec.symbol, option_symbol("AAPL", date(2025, 1, 17), "put", 152.5))
        self.assertEqual(parse_option_symbol(spec.symbol), {"underlying": "AAPL", "expiry": date(2025, 1, 17), "right": "put", "strike": 152.5})
        for bad in ("AAPL", "AAPL250117X00150000", "AAPL251340C00150000"):
            with self.assertRaises(InstrumentError):
                parse_option_symbol(bad)

    def test_from_dict_handles_loose_rows(self):
        spec = ContractSpec.from_dict({"symbol": "NIFTY25JAN22000CE", "asset_class": "OPTION", "underlying": "NIFTY", "expiry": "2025-01-30",
                                       "strike": "22000", "right": "CE", "lot_size": "50", "shortable": "yes", "sector": "",
                                       "broker:Zerodha": "NIFTY25JAN22000CE", "broker:other": "", "unknown_column": "ignored"})
        self.assertEqual((spec.right, spec.strike, spec.lot_size, spec.shortable), ("call", 22000.0, 50.0, True))
        self.assertEqual(dict(spec.broker_symbols), {"zerodha": "NIFTY25JAN22000CE"})
        self.assertEqual(ContractSpec.from_dict(spec.to_dict()), spec)
        with self.assertRaises(InstrumentError):
            ContractSpec.from_dict({"name": "no symbol"})
        with self.assertRaises(InstrumentError):
            ContractSpec.from_dict({"symbol": "X", "tick_size": "abc"})


class InstrumentMasterTests(unittest.TestCase):
    def setUp(self):
        self.m = InstrumentMaster([ContractSpec("AAPL", name="Apple Inc", sector="Tech", isin="US0378331005", broker_symbols={"zerodha": "AAPL-EQ"}),
                                   ContractSpec("XOM", name="Exxon", sector="Energy", exchange="NYSE"),
                                   future_spec("ES", "2025-03-21", 50, 0.25), future_spec("ES", "2025-06-20", 50, 0.25),
                                   option_spec("AAPL", "2025-01-17", "call", 150), option_spec("AAPL", "2025-01-17", "put", 150),
                                   option_spec("AAPL", "2025-02-21", "call", 155)])

    def test_symbol_translation_both_ways(self):
        self.assertEqual(self.m.resolve("Zerodha", "AAPL-EQ").symbol, "AAPL")
        self.assertEqual(self.m.broker_symbol("AAPL", "zerodha"), "AAPL-EQ")
        self.assertEqual(self.m.resolve("alpaca", "XOM").symbol, "XOM")              # no mapping recorded: same symbol
        with self.assertRaises(InstrumentError) as cm:
            self.m.resolve("zerodha", "AAPL")                                         # zerodha has its own name for it: ours is not valid there
        self.assertEqual(cm.exception.code, "UNMAPPED_SYMBOL")
        with self.assertRaises(InstrumentError):
            self.m.get("NOPE")

    def test_duplicates_and_broker_symbol_clashes_are_refused(self):
        with self.assertRaises(InstrumentError) as cm:
            self.m.add(ContractSpec("AAPL"))
        self.assertEqual(cm.exception.code, "DUPLICATE_INSTRUMENT")
        with self.assertRaises(InstrumentError) as cm:
            self.m.add(ContractSpec("AAPL2", broker_symbols={"zerodha": "AAPL-EQ"}))
        self.assertEqual(cm.exception.code, "BROKER_SYMBOL_CLASH")
        self.m.add(ContractSpec("AAPL", broker_symbols={"zerodha": "AAPL-NEW"}), overwrite=True)
        self.assertEqual(self.m.resolve("zerodha", "AAPL-NEW").symbol, "AAPL")
        with self.assertRaises(InstrumentError):
            self.m.resolve("zerodha", "AAPL-EQ")                                      # the old mapping is gone

    def test_search_and_deactivate(self):
        self.assertEqual([s.symbol for s in self.m.search("apple")], ["AAPL"])
        self.assertEqual([s.symbol for s in self.m.search("us0378331005")], ["AAPL"])
        self.assertEqual([s.symbol for s in self.m.search(sector="energy")], ["XOM"])
        self.assertEqual(len(self.m.search(asset_class="option")), 3)
        self.assertEqual(len(self.m.search(asset_class=AssetClass.OPTION, asof="2025-02-01")), 1)     # expired contracts drop out
        self.m.deactivate("XOM")
        self.assertEqual(self.m.search(sector="Energy"), [])
        self.assertEqual(len(self.m.search(sector="Energy", active_only=False)), 1)

    def test_derivative_queries(self):
        self.assertEqual(self.m.expiries("AAPL"), [date(2025, 1, 17), date(2025, 2, 21)])
        self.assertEqual([(s.strike, s.right) for s in self.m.option_chain("AAPL", "2025-01-17")], [(150.0, "call"), (150.0, "put")])
        self.assertEqual(self.m.front_future("ES", "2025-01-01").symbol, "ESH25")
        self.assertEqual(self.m.front_future("ES", "2025-03-18", roll_days=5).symbol, "ESM25")       # inside the roll window: next contract
        with self.assertRaises(InstrumentError):
            self.m.front_future("ES", "2026-01-01")

    def test_bulk_load_reports_every_bad_row(self):
        rows = [{"symbol": "GOOD1"}, {"symbol": "BAD", "tick_size": "-1"}, {"symbol": "GOOD2", "sector": "Tech"}, {"name": "no symbol"},
                {"symbol": "AAPL"}]
        with self.assertRaises(AggregateError) as cm:
            InstrumentMaster().load_records(rows[:4])
        self.assertEqual(len(cm.exception.errors), 2)
        report = self.m.load_records(rows, on_error="skip")
        self.assertEqual(report["loaded"], 2)
        self.assertEqual(len(report["rejected"]), 3)                                  # two malformed rows and one duplicate
        self.assertIn("GOOD2", self.m)
        self.assertEqual(self.m.sectors()["GOOD2"], "Tech")

    def test_csv_and_exports(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "master.csv")
            pd.DataFrame([{"symbol": "TCS", "exchange": "NSE", "tick_size": 0.05, "broker:zerodha": "TCS"},
                          {"symbol": "INFY", "exchange": "NSE", "tick_size": 0.05, "broker:zerodha": "INFY-EQ"}]).to_csv(path, index=False)
            m = InstrumentMaster()
            self.assertEqual(m.load_csv(path)["loaded"], 2)
            self.assertEqual(m.get("TCS").tick_size, 0.05)
            self.assertEqual(m.resolve("zerodha", "INFY-EQ").symbol, "INFY")
            with self.assertRaises(MissingDataError):
                m.load_csv(os.path.join(d, "missing.csv"))
        self.assertEqual(set(self.m.instruments(["AAPL", "XOM"])), {"AAPL", "XOM"})
        self.assertEqual(len(self.m.to_frame()), len(self.m))


class CorporateActionTests(unittest.TestCase):
    def test_split_removes_the_artificial_jump(self):
        df = flat_frame(10)
        ex = df.index[5]
        df.loc[ex:, ["open", "high", "low", "close"]] /= 2.0                          # 2-for-1: raw price halves
        df.loc[ex:, "volume"] *= 2.0
        adj = adjust_ohlcv(df, [CorporateAction("X", ex, "split", 2.0)])
        np.testing.assert_allclose(adj["close"], 50.0)
        np.testing.assert_allclose(adj["volume"], 2000.0)
        self.assertEqual(df["close"].iloc[0], 100.0)                                  # the input is not modified
        np.testing.assert_allclose(adj["close"] * adj["volume"], df["close"] * df["volume"])   # traded value is unchanged by a split

    def test_dividend_makes_return_equal_total_return(self):
        df = flat_frame(6)
        ex = df.index[3]
        df.loc[ex:, ["open", "high", "low", "close"]] -= 2.0                          # falls by the 2.0 dividend on the ex-date
        adj = adjust_ohlcv(df, [CorporateAction("X", ex, "dividend", 2.0)])
        self.assertAlmostEqual(adj["close"].iloc[3] / adj["close"].iloc[2] - 1, 0.0)  # economically nothing happened
        f = adjustment_factors(df, [CorporateAction("X", ex, "dividend", 2.0)])
        self.assertAlmostEqual(f["price"].iloc[0], 0.98)
        self.assertEqual(f["price"].iloc[-1], 1.0)

    def test_actions_outside_the_data_and_bad_actions(self):
        df = flat_frame(5)
        same = adjust_ohlcv(df, [CorporateAction("X", "2030-01-01", "split", 3.0), CorporateAction("X", "2000-01-01", "split", 3.0)])
        pd.testing.assert_frame_equal(same, df)
        for kind, value in (("merger", 1.0), ("split", 0.0), ("dividend", float("nan"))):
            with self.assertRaises(InstrumentError):
                CorporateAction("X", "2024-01-01", kind, value)
        with self.assertRaises(InstrumentError):
            adjust_ohlcv(df, [CorporateAction("X", df.index[2], "dividend", 500.0)])   # larger than the price

    def test_position_effects(self):
        self.assertEqual(unadjusted_position(100, 50.0, CorporateAction("X", "2024-01-01", "split", 2.0)), (200.0, 25.0))
        self.assertEqual(unadjusted_position(100, 50.0, CorporateAction("X", "2024-01-01", "dividend", 1.0)), (100, 50.0))
        acts = actions_from_records([{"symbol": "X", "ex_date": "2024-01-05", "kind": "dividend", "value": "1.5"},
                                     {"symbol": "X", "ex_date": "2024-02-05", "kind": "split", "value": 2}])
        self.assertEqual(dividend_cash(-10, acts), -15.0)                             # a short position pays the dividend


class ContinuousFuturesTests(unittest.TestCase):
    def setUp(self):
        i1, i2 = pd.bdate_range("2024-01-01", periods=40), pd.bdate_range("2024-01-15", periods=60)
        c1, c2 = np.linspace(100, 110, 40), np.linspace(115, 130, 60)                 # the second contract trades ~12 higher
        self.c1 = pd.DataFrame({"open": c1, "high": c1 + 1, "low": c1 - 1, "close": c1, "volume": 1.0}, index=i1)
        self.c2 = pd.DataFrame({"open": c2, "high": c2 + 1, "low": c2 - 1, "close": c2, "volume": 1.0}, index=i2)
        self.contracts = {"2024-02-23": self.c1, "2024-04-05": self.c2}

    def test_schedule(self):
        sched = roll_schedule(self.contracts, roll_days=5)
        self.assertEqual(sched[0]["end"], pd.Timestamp("2024-02-18"))
        self.assertEqual(sched[1]["start"], sched[0]["end"])
        self.assertEqual(sched[1]["end"], pd.Timestamp("2024-04-05"))
        with self.assertRaises(InstrumentError):
            roll_schedule({})

    def test_back_adjustment_removes_the_roll_gap(self):
        raw = continuous_series(self.contracts, 5, "none")
        self.assertGreater(raw["close"].diff().abs().max(), 10)                       # spliced naively there is a ~12 point jump
        for method in ("difference", "ratio"):
            cs = continuous_series(self.contracts, 5, method)
            self.assertLess(cs["close"].diff().abs().max(), 1.0, method)
            self.assertTrue(cs.index.is_monotonic_increasing and cs.index.is_unique)
            self.assertEqual(cs["contract"].nunique(), 2)
            pd.testing.assert_series_equal(cs["close"].iloc[-20:], self.c2["close"].iloc[-20:], check_names=False)   # newest contract untouched
        ratio = continuous_series(self.contracts, 5, "ratio")
        old = ratio[ratio["contract"] == pd.Timestamp("2024-02-23")]
        np.testing.assert_allclose(old["close"].pct_change().dropna(), self.c1["close"].loc[old.index].pct_change().dropna())   # % moves kept

    def test_errors(self):
        with self.assertRaises(InstrumentError):
            continuous_series(self.contracts, 5, "magic")
        late = {"2024-02-23": self.c1, "2024-04-05": self.c2.iloc[30:]}               # new contract has no bar on the roll day
        with self.assertRaises(InstrumentError):
            continuous_series(late, 5)


class OptionChainTests(unittest.TestCase):
    def setUp(self):
        self.chain = synthetic_chain(spot=100.0, atm_vol=0.2, skew=-0.15, smile=0.3, spread=0.0, seed=1)

    def test_implied_vol_recovers_the_vol_that_priced_each_quote(self):
        df = self.chain.frame()
        self.assertEqual(self.chain.iv_failures, 0)
        row = df[(df["strike"] == 100.0) & (df["kind"] == "call")].iloc[0]
        self.assertAlmostEqual(row["iv"], 0.2 + 0.02 * math.sqrt(row["T"]), places=3)
        repriced = [bs_price(100.0, r.strike, r.T, 0.02, r.iv, r.kind) for r in df.itertuples()]
        np.testing.assert_allclose(repriced, df["mid"], atol=2e-4)
        calls, puts = df[df["kind"] == "call"], df[df["kind"] == "put"]
        self.assertTrue(((calls["delta"] > 0) & (calls["delta"] < 1)).all() and ((puts["delta"] < 0) & (puts["delta"] > -1)).all())
        self.assertTrue((df["gamma"] > 0).all() and (df["vega"] > 0).all())

    def test_skew_and_term_structure_have_the_planted_shape(self):
        sk = self.chain.skew()
        self.assertLess(sk["risk_reversal"], 0)                                       # negative skew: puts carry more vol than calls
        self.assertGreater(sk["butterfly"], 0)                                        # positive smile: wings above the money
        self.assertGreater(sk["put_iv"], sk["atm_iv"])
        ts = self.chain.term_structure()
        self.assertEqual(list(ts["days"]), [30, 60, 90])
        self.assertTrue(ts["atm_iv"].is_monotonic_increasing)
        smile = self.chain.smile()
        self.assertTrue((smile.loc[smile["strike"] < 100, "kind"] == "put").all())    # the out-of-the-money side only
        self.assertEqual(self.chain.iv_surface().shape[1], 3)
        self.assertTrue(math.isnan(self.chain.iv_at_delta(None, "call", 0.001)))      # beyond the listed strikes: no extrapolation

    def test_strike_selection_and_expected_move(self):
        self.assertEqual(self.chain.atm_strike(), 100.0)
        self.assertEqual(len(self.chain.expiries()), 3)
        k = self.chain.strike_by_delta(None, "put", 0.25)
        self.assertLess(k, 100.0)
        df = self.chain.frame(self.chain.nearest_expiry())
        self.assertAlmostEqual(abs(df[(df["strike"] == k) & (df["kind"] == "put")]["delta"].iloc[0]), 0.25, delta=0.08)
        em = self.chain.expected_move()
        self.assertAlmostEqual(em["iv_move_pct"], self.chain.atm_iv() * math.sqrt(30 / 365), places=6)
        self.assertAlmostEqual(em["straddle"] / em["iv_move"], 0.8, delta=0.05)       # ATM straddle ~ 0.8 x one-sigma move
        self.assertEqual(self.chain.nearest_expiry(min_days=45), self.chain.expiries()[1])
        with self.assertRaises(MissingDataError):
            self.chain.nearest_expiry(min_days=500)

    def test_open_interest_analytics_on_a_hand_built_chain(self):
        rows = []
        oi = {90: (10, 500), 95: (20, 300), 100: (100, 100), 105: (400, 20), 110: (600, 10)}       # strike -> (call OI, put OI)
        for k, (co, po) in oi.items():
            rows.append({"expiry": "2024-02-01", "strike": k, "kind": "C", "last": bs_price(100, k, 30 / 365, 0.0, 0.2, "call"), "open_interest": co, "volume": co})
            rows.append({"expiry": "2024-02-01", "strike": k, "kind": "PE", "last": bs_price(100, k, 30 / 365, 0.0, 0.2, "put"), "open_interest": po, "volume": 2 * po})
        ch = OptionChain.from_frame(pd.DataFrame(rows), "X", 100.0, "2024-01-02")
        self.assertAlmostEqual(ch.put_call_ratio(), 930 / 1130)
        self.assertAlmostEqual(ch.put_call_ratio(by="volume"), 2 * 930 / 1130)
        self.assertEqual(ch.max_pain(), 100.0)
        walls = ch.oi_walls(n=1)
        self.assertEqual((walls["resistance"], walls["support"]), ([(110.0, 600.0)], [(90.0, 500.0)]))
        gex = ch.gamma_exposure()
        self.assertGreater(gex.loc[110.0], 0)                                         # call-heavy strike: positive dealer gamma
        self.assertLess(gex.loc[90.0], 0)
        np.testing.assert_allclose(ch.frame()["iv"], 0.2, atol=1e-4)                  # falls back to last when there is no bid/ask
        with self.assertRaises(ValidationError):
            ch.put_call_ratio(by="notional")

    def test_bad_quotes_are_isolated_and_reported(self):
        df = self.chain.frame()[["expiry", "strike", "kind", "bid", "ask", "last", "volume", "open_interest"]].copy()
        i_arb, i_cross = df.index[(df["strike"] == 90.0) & (df["kind"] == "call")][0], df.index[(df["strike"] == 105.0) & (df["kind"] == "put")][0]
        df.loc[i_arb, ["bid", "ask", "last"]] = [1.0, 1.0, 1.0]                       # a 90-strike call cannot be worth 1 with spot at 100
        df.loc[i_cross, ["bid", "ask"]] = [9.0, 8.0]                                  # crossed market
        ch = OptionChain.from_frame(df, "SYN", 100.0, "2024-01-02", r=0.02)
        self.assertEqual(ch.iv_failures, 1)                                           # that quote gets NaN; the other 101 are fine
        self.assertEqual(int(ch.frame()["iv"].isna().sum()), 1)
        q = ch.quality_report()
        self.assertFalse(q["ok"])
        self.assertEqual((q["crossed"], q["iv_failures"]), (1, 1))
        self.assertTrue(q["arbitrage"])
        self.assertTrue(self.chain.quality_report()["ok"])
        self.assertEqual(set(self.chain.summary()) >= {"atm_iv", "put_call_ratio", "max_pain", "expected_move_pct"}, True)

    def test_input_validation(self):
        base = pd.DataFrame([{"expiry": "2024-03-01", "strike": 100, "kind": "call", "last": 3.0}])
        for mutate, exc in ((lambda d: d.drop(columns="kind"), DataError), (lambda d: d.drop(columns="last"), DataError),
                            (lambda d: d.assign(kind="straddle"), DataError), (lambda d: d.assign(strike=-5), DataError),
                            (lambda d: pd.concat([d, d]), DataError), (lambda d: d.assign(expiry="2023-01-01"), MissingDataError)):
            with self.assertRaises(exc):
                OptionChain.from_frame(mutate(base), "X", 100.0, "2024-01-02")
        with self.assertRaises(ValidationError):
            OptionChain.from_frame(base, "X", -1.0, "2024-01-02")
        with self.assertRaises(MissingDataError):
            OptionChain.from_frame(base, "X", 100.0, "2024-01-02").frame("2030-01-01")

    def test_iv_rank_and_percentile(self):
        hist = [0.10, 0.20, 0.30, 0.90, 0.25]
        self.assertAlmostEqual(iv_rank(hist), (0.25 - 0.10) / 0.80)
        self.assertAlmostEqual(iv_percentile(hist), 0.4)                              # less moved by the single 0.90 spike than rank is
        self.assertAlmostEqual(iv_rank(hist, 0.5), 0.5)
        self.assertTrue(math.isnan(iv_rank([0.2])))


if __name__ == "__main__":
    unittest.main()
