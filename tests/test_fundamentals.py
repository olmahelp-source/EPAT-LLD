import unittest
from datetime import date

import numpy as np
import pandas as pd

from helpers import free_config
from algoplatform.backtest import BacktestEngine
from algoplatform.core.exceptions import StrategyError, ValidationError
from algoplatform.data import MarketData
from algoplatform.fundamentals import (FactorStrategy, FundamentalsStore, Statement, altman_z, beneish_m, capm, composite_score,
                                       dcf, factor_frame, gordon_ddm, margin_of_safety, multiples, piotroski_f, ratios,
                                       residual_income, reverse_dcf, sector_neutralize, sloan_accruals, terminal_sensitivity,
                                       wacc, winsorize, zscore)

PREV = dict(revenue=1000, cogs=600, sga=150, depreciation=50, operating_income=200, interest_expense=20, pretax_income=180,
            tax_expense=45, net_income=135, cash=100, receivables=120, inventory=150, payables=80, current_assets=400, ppe=500,
            total_assets=1000, current_liabilities=250, long_term_debt=200, short_term_debt=50, total_liabilities=600,
            equity=400, retained_earnings=250, shares=100, cfo=150, capex=60, dividends=20)
CUR = dict(revenue=1200, cogs=690, sga=170, depreciation=55, operating_income=280, interest_expense=18, pretax_income=262,
           tax_expense=65, net_income=197, cash=160, receivables=130, inventory=160, payables=90, current_assets=480, ppe=540,
           total_assets=1150, current_liabilities=260, long_term_debt=150, short_term_debt=40, total_liabilities=600,
           equity=550, retained_earnings=400, shares=100, cfo=250, capex=70, dividends=25)


def stmt(data, end="2023-12-31", ann="2024-02-20", sym="ACME", version=1):
    return Statement(sym, date.fromisoformat(end), date.fromisoformat(ann), dict(data), version=version)


class StatementTests(unittest.TestCase):
    def test_validation_and_derived_lines(self):
        s = stmt(CUR)
        self.assertAlmostEqual(s.gross_profit, 510)
        self.assertAlmostEqual(s.ebitda, 335)
        self.assertAlmostEqual(s.total_debt, 190)
        self.assertAlmostEqual(s.free_cash_flow, 180)
        self.assertAlmostEqual(s.eps, 1.97)
        self.assertAlmostEqual(s.balance_gap(), 0.0)                                   # 1150 = 600 + 550
        with self.assertRaises(ValidationError):
            stmt(CUR, end="2024-03-31", ann="2024-02-20")                              # announced before the period ended
        with self.assertRaises(ValidationError):
            Statement("X", date(2023, 12, 31), date(2024, 2, 1), {"nonsense": 1})
        with self.assertRaises(ValidationError):
            stmt({"revenue": 1})["cogs"]

    def test_point_in_time_and_restatements(self):
        store = FundamentalsStore()
        store.add(stmt(PREV, "2022-12-31", "2023-02-15"))
        store.add(stmt(CUR, "2023-12-31", "2024-02-20"))
        restated = dict(CUR, net_income=150)
        store.add(stmt(restated, "2023-12-31", "2024-08-01", version=2))
        on = date.fromisoformat
        self.assertEqual(store.asof("ACME", on("2024-01-31")).period_end, date(2022, 12, 31))   # FY23 not public yet
        v1 = store.asof("ACME", on("2024-03-01"))
        self.assertEqual((v1.period_end, v1["net_income"]), (date(2023, 12, 31), 197))
        v2 = store.asof("ACME", on("2024-09-01"))
        self.assertEqual(v2["net_income"], 150)                                                  # restated figures only after the restatement
        self.assertEqual(store.asof("ACME", on("2024-03-01"), lag_days=30).period_end, date(2022, 12, 31))   # safety lag
        self.assertEqual(store.previous("ACME", v1, on("2024-03-01")).period_end, date(2022, 12, 31))
        self.assertIsNone(store.asof("ACME", on("2022-01-01")))
        self.assertIsNone(store.asof("OTHER", on("2024-03-01")))


class RatioTests(unittest.TestCase):
    def setUp(self):
        self.cur, self.prev = stmt(CUR), stmt(PREV, "2022-12-31", "2023-02-15")

    def test_profitability_and_leverage(self):
        p = ratios.profitability(self.cur, self.prev)
        self.assertAlmostEqual(p["gross_margin"], 0.425)
        self.assertAlmostEqual(p["operating_margin"], 280 / 1200)
        self.assertAlmostEqual(p["roe"], 197 / 475)                                      # average equity
        self.assertAlmostEqual(p["roa"], 197 / 1075)
        lev = ratios.leverage(self.cur)
        self.assertAlmostEqual(lev["debt_to_equity"], 190 / 550)
        self.assertAlmostEqual(lev["net_debt_to_ebitda"], 30 / 335)
        self.assertAlmostEqual(lev["interest_coverage"], 280 / 18)
        self.assertAlmostEqual(lev["current_ratio"], 480 / 260)
        self.assertAlmostEqual(lev["quick_ratio"], 320 / 260)

    def test_efficiency_and_growth(self):
        e = ratios.efficiency(self.cur)
        self.assertAlmostEqual(e["inventory_days"], 160 / 690 * 365)
        self.assertAlmostEqual(e["cash_conversion_cycle"], e["inventory_days"] + e["receivable_days"] - e["payable_days"])
        g = ratios.growth(self.cur, self.prev)
        self.assertAlmostEqual(g["revenue_growth"], 0.2)
        self.assertAlmostEqual(g["eps_growth"], 1.97 / 1.35 - 1)
        self.assertAlmostEqual(g["fcf_growth"], 1.0)                                      # FCF 90 -> 180
        self.assertIsNone(ratios.growth(self.cur, None)["revenue_growth"])
        loss = stmt(dict(PREV, net_income=-10), "2022-12-31", "2023-02-15")
        self.assertIsNone(ratios.growth(self.cur, loss)["eps_growth"])                    # growth from a loss is undefined

    def test_dupont_identities(self):
        d3, d5 = ratios.dupont3(self.cur), ratios.dupont5(self.cur)
        self.assertAlmostEqual(d3["roe"], 197 / 550)
        self.assertAlmostEqual(d5["roe"], 197 / 550)                                      # both decompositions reproduce ROE
        self.assertAlmostEqual(d5["tax_burden"], 197 / 262)
        self.assertIn("roic", ratios.all_ratios(self.cur, self.prev))


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.cur, self.prev = stmt(CUR), stmt(PREV, "2022-12-31", "2023-02-15")

    def test_piotroski_worked_example(self):
        r = piotroski_f(self.cur, self.prev)
        self.assertEqual(r["score"], 9)                                                   # every one of the nine tests passes
        worse = stmt(dict(CUR, shares=120, cfo=100, long_term_debt=300), "2023-12-31", "2024-02-20")
        r2 = piotroski_f(worse, self.prev)
        self.assertLess(r2["score"], 9)
        self.assertFalse(r2["checks"]["no_dilution"])
        self.assertFalse(r2["checks"]["cfo_exceeds_net_income"])

    def test_altman_z_hand_computed(self):
        z = altman_z(self.cur, market_cap=2000)
        expected = (1.2 * 220 / 1150 + 1.4 * 400 / 1150 + 3.3 * 280 / 1150 + 0.6 * 2000 / 600 + 1200 / 1150)
        self.assertAlmostEqual(z["z"], expected)
        self.assertEqual(z["zone"], "safe")
        weak = stmt(dict(CUR, retained_earnings=-300, operating_income=-50, current_assets=200), "2023-12-31", "2024-02-20")
        self.assertEqual(altman_z(weak, market_cap=100)["zone"], "distress")

    def test_beneish_flags_inflated_revenue_quality(self):
        self.assertFalse(beneish_m(self.cur, self.prev)["flag"])
        cooked = stmt(dict(CUR, receivables=450, cfo=20, revenue=1500, cogs=900), "2023-12-31", "2024-02-20")
        m = beneish_m(cooked, self.prev)
        self.assertTrue(m["flag"])
        self.assertGreater(m["components"]["DSRI"], 2.0)
        self.assertGreater(m["m"], beneish_m(self.cur, self.prev)["m"])

    def test_sloan_accruals(self):
        self.assertAlmostEqual(sloan_accruals(self.cur, self.prev), (197 - 250) / 1075)
        self.assertGreater(sloan_accruals(stmt(dict(CUR, cfo=20)), self.prev), 0)           # profits without cash: positive accruals


class ValuationTests(unittest.TestCase):
    def test_dcf_matches_closed_forms(self):
        flat = dcf(100, 0.0, 0.10, 0.0, years=10, fade=False)
        self.assertAlmostEqual(flat["enterprise_value"], 1000.0, places=6)                 # perpetuity 100 / 0.10
        self.assertAlmostEqual(flat["terminal_share"], 1 / 1.1 ** 10, places=6)
        growing = dcf(100, 0.03, 0.10, 0.03, years=10, fade=False)
        self.assertAlmostEqual(growing["enterprise_value"], 100 * 1.03 / 0.07, places=6)    # Gordon growth
        eq = dcf(100, 0.0, 0.10, 0.0, years=10, net_debt=200, shares=50, fade=False)
        self.assertAlmostEqual(eq["per_share"], (1000 - 200) / 50)
        with self.assertRaises(ValidationError):
            dcf(100, 0.05, 0.03, 0.03)                                                     # discount <= terminal growth

    def test_dcf_is_fragile_in_the_terminal_assumptions(self):
        grid = terminal_sensitivity(100, 0.08, [0.08, 0.09, 0.10, 0.12], [0.01, 0.02, 0.03], shares=1)
        self.assertGreater(grid.loc[0.03, "r=8.0%"], grid.loc[0.01, "r=8.0%"])
        self.assertTrue((grid["r=8.0%"].diff().dropna() > 0).all())
        self.assertGreater(grid.max().max() / grid.min().min(), 1.5)                       # same company, very different values
        lo, hi = dcf(100, 0.05, 0.12, 0.02), dcf(100, 0.05, 0.08, 0.02)
        self.assertGreater(hi["terminal_share"], lo["terminal_share"])

    def test_reverse_dcf_recovers_the_growth_it_was_priced_with(self):
        price = dcf(100, 0.08, 0.10, 0.025, 10, net_debt=0, shares=1)["per_share"]
        self.assertAlmostEqual(reverse_dcf(price, 100, 0.10, 0.025), 0.08, places=4)
        self.assertIsNone(reverse_dcf(1e9, 100, 0.10, 0.025))                              # no plausible growth justifies it

    def test_cost_of_capital_and_simple_models(self):
        self.assertAlmostEqual(capm(0.03, 1.2, 0.05), 0.09)
        self.assertAlmostEqual(wacc(70, 30, 0.10, 0.05, 0.25), 0.7 * 0.10 + 0.3 * 0.05 * 0.75)
        self.assertAlmostEqual(gordon_ddm(2, 0.03, 0.08), 2 * 1.03 / 0.05)
        with self.assertRaises(ValidationError):
            gordon_ddm(2, 0.08, 0.08)
        self.assertAlmostEqual(residual_income(100, [0.10] * 5, 0.10), 100.0)              # earning exactly the cost of equity adds nothing
        self.assertGreater(residual_income(100, [0.15] * 5, 0.10), 100.0)
        self.assertAlmostEqual(margin_of_safety(100, 70), 0.30)

    def test_multiples(self):
        m = multiples(stmt(CUR), 20.0)
        self.assertAlmostEqual(m["pe"], 20 / 1.97)
        self.assertAlmostEqual(m["pb"], 2000 / 550)
        self.assertAlmostEqual(m["ev_ebitda"], (2000 + 190 - 160) / 335)
        self.assertAlmostEqual(m["fcf_yield"], 180 / 2000)
        loss = multiples(stmt(dict(CUR, net_income=-5)), 20.0)
        self.assertIsNone(loss["pe"])                                                      # a P/E on losses is meaningless


class FactorTests(unittest.TestCase):
    def test_winsorize_zscore_and_sector_neutralize(self):
        s = pd.Series([1, 2, 3, 4, 1000.0], index=list("abcde"))
        w = winsorize(s, 0.2)
        self.assertLess(w.max(), 1000)
        z = zscore(pd.Series([1.0, 2, 3, 4, 5]))
        self.assertAlmostEqual(z.mean(), 0.0)
        self.assertAlmostEqual(z.std(ddof=1), 1.0)
        n = sector_neutralize(pd.Series([10.0, 12, 100, 104], index=list("abcd")), {"a": "x", "b": "x", "c": "y", "d": "y"})
        self.assertAlmostEqual(n["a"], -1.0)
        self.assertAlmostEqual(n["c"], -2.0)                                                # ranked against own sector

    def test_composite_direction_and_missing_data(self):
        f = pd.DataFrame({"earnings_yield": [0.10, 0.05, 0.02, 0.07], "debt_to_equity": [0.2, 0.5, 2.0, 0.4],
                          "roe": [0.20, 0.10, np.nan, 0.15]}, index=list("ABCD"))
        sc = composite_score(f, winsor=0.0)
        self.assertEqual(sc.idxmax(), "A")
        self.assertEqual(sc.idxmin(), "C")                                                  # cheap AND low debt wins; expensive AND levered loses
        self.assertFalse(sc.isna().any())                                                   # C still scored with a missing factor
        with self.assertRaises(ValidationError):
            composite_score(pd.DataFrame({"unknown": [1, 2, 3]}))

    def build_store(self, with_future=False):
        store = FundamentalsStore()
        quality = {"A": 1500, "B": 1000, "C": 500, "D": 100}
        for sym, ni in quality.items():
            old = {"A": 100, "B": 500, "C": 1000, "D": 1500}[sym]                          # FY21: the OPPOSITE ranking to FY22
            store.add(Statement(sym, date(2021, 12, 31), date(2022, 3, 1), dict(PREV, net_income=old, cfo=old * 1.2, equity=5000, shares=100)))
            store.add(Statement(sym, date(2022, 12, 31), date(2023, 4, 3), dict(CUR, net_income=ni, cfo=ni * 1.2, equity=5000, shares=100)))
            if with_future:                                                                  # FY2023: the ranking flips
                inv = {"A": 100, "B": 500, "C": 1000, "D": 1500}[sym]
                store.add(Statement(sym, date(2023, 12, 31), date(2024, 2, 1), dict(CUR, net_income=inv, cfo=inv * 1.2, equity=5000, shares=100)))
        return store

    def market(self):
        idx = pd.bdate_range("2023-01-02", periods=300)
        md = MarketData()
        for s in "ABCD":
            c = np.full(len(idx), 100.0)
            md.add(s, pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1e6}, index=idx))
        return md

    def test_factor_frame_is_point_in_time(self):
        store = self.build_store(with_future=True)
        prices = {s: 100.0 for s in "ABCD"}
        self.assertTrue(factor_frame(store, prices, date(2022, 1, 15)).empty)               # nothing announced yet
        stale = factor_frame(store, prices, date(2023, 3, 1))                                # FY22 exists but is not public until 3 Apr
        self.assertEqual(composite_score(stale).idxmax(), "D")
        mid = factor_frame(store, prices, date(2023, 6, 1))
        self.assertEqual(composite_score(mid).idxmax(), "A")
        late = factor_frame(store, prices, date(2024, 3, 1))
        self.assertEqual(composite_score(late).idxmax(), "D")

    def test_strategy_trades_only_after_announcement_and_future_data_is_invisible(self):
        md = self.market()
        cfg = free_config()
        def run(with_future):
            s = FactorStrategy.with_store(self.build_store(with_future), list("ABCD"), top_frac=0.25, rebalance_every=21, lag_days=1)
            return BacktestEngine(md, cfg).run(s)
        full, early = run(True), run(False)
        ann = pd.Timestamp("2023-04-03")
        self.assertEqual({f.symbol for f in early.fills if f.ts <= ann}, {"D"})             # before the filing: the stale FY21 winner
        after = {f.symbol for f in early.fills if f.ts > ann + pd.Timedelta(days=30)}
        self.assertIn("A", after)                                                           # the FY22 winner once it is public
        self.assertFalse([f for f in early.fills if f.symbol == "A" and f.ts <= ann])       # never bought on unannounced numbers
        cut = pd.Timestamp("2024-01-31")
        pd.testing.assert_series_equal(full.equity.loc[:cut], early.equity.loc[:cut])      # FY23 numbers cannot leak backwards
        self.assertIn("D", {f.symbol for f in full.fills})                                  # but are used once public

    def test_strategy_needs_a_store_and_validates(self):
        with self.assertRaises(StrategyError):
            FactorStrategy(["A", "B"], top_frac=0.9)
        res_err = None
        try:
            BacktestEngine(self.market(), free_config()).run(FactorStrategy(list("ABCD")))
        except Exception as exc:                                                            # a strategy that errors is contained by the engine
            res_err = exc
        self.assertIsNone(res_err)


if __name__ == "__main__":
    unittest.main()
