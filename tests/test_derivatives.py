import math
import unittest
from datetime import date

import numpy as np

from algoplatform.core import Side
from algoplatform.core.exceptions import StatisticsError, ValidationError
from algoplatform.derivatives import (OptionLeg, binomial_price, butterfly, check_no_arbitrage, covered_call, expiry_fill, fit_svi,
                                      greeks, hedge_trade, implied_vol_slice, iron_condor, mc_price, option_instrument, option_symbol,
                                      payoff, profile, simulate_delta_hedge, straddle, svi_total_variance, value,
                                      vertical_spread, whalley_wilmott_band)
from algoplatform.portfolio import Portfolio
from algoplatform.quant import bs_price


class PricingTests(unittest.TestCase):
    def test_binomial_converges_to_black_scholes(self):
        bs_c, bs_p = bs_price(100, 100, 1, 0.05, 0.2, "call"), bs_price(100, 100, 1, 0.05, 0.2, "put")
        for kind, ref in (("call", bs_c), ("put", bs_p)):
            e50 = abs(binomial_price(100, 100, 1, 0.05, 0.2, kind, steps=50) - ref)
            e400 = abs(binomial_price(100, 100, 1, 0.05, 0.2, kind, steps=400) - ref)
            self.assertLess(e400, 0.02)
            self.assertLess(e400, e50 + 1e-9)
        self.assertAlmostEqual(bs_c, 10.4506, places=3)

    def test_american_early_exercise(self):
        eu_p = binomial_price(100, 110, 1, 0.05, 0.2, "put", steps=300)
        am_p = binomial_price(100, 110, 1, 0.05, 0.2, "put", american=True, steps=300)
        self.assertGreater(am_p, eu_p + 0.05)                                                  # the early-exercise premium
        self.assertGreaterEqual(am_p, 10.0)                                                     # never below intrinsic value
        eu_c = binomial_price(100, 100, 1, 0.05, 0.2, "call", steps=300)
        am_c = binomial_price(100, 100, 1, 0.05, 0.2, "call", american=True, steps=300)
        self.assertAlmostEqual(am_c, eu_c, places=9)                                             # no dividend: calls are never exercised early
        self.assertGreater(binomial_price(100, 100, 1, 0.05, 0.2, "call", american=True, steps=300, q=0.06),
                           binomial_price(100, 100, 1, 0.05, 0.2, "call", steps=300, q=0.06))   # with dividends they can be
        with self.assertRaises(ValidationError):
            binomial_price(100, 100, 1, 0.05, -0.2)
        with self.assertRaises(ValidationError):
            binomial_price(100, 100, 1, 3.0, 0.01, steps=2)                                      # r too high for the lattice: arbitrage

    def test_monte_carlo(self):
        ref = bs_price(100, 100, 1, 0.05, 0.2, "call")
        mc = mc_price(100, 100, 1, 0.05, 0.2, "call", n=60_000, seed=1)
        self.assertLess(abs(mc["price"] - ref), 4 * mc["stderr"] + 0.05)                        # within sampling error (+ discretisation)
        self.assertEqual(mc_price(100, 100, 1, 0.05, 0.2, seed=5), mc_price(100, 100, 1, 0.05, 0.2, seed=5))
        asian = mc_price(100, 100, 1, 0.05, 0.2, payoff="asian", seed=1)["price"]
        self.assertTrue(0 < asian < mc["price"])                                                 # averaging lowers volatility and value
        ko = mc_price(100, 100, 1, 0.05, 0.2, payoff="barrier_out", barrier=85, seed=1)["price"]
        self.assertTrue(0 < ko < mc["price"])                                                    # a knock-out is cheaper than the vanilla
        tight = mc_price(100, 100, 1, 0.05, 0.2, payoff="barrier_out", barrier=99.9, seed=1)["price"]
        self.assertLess(tight, ko)
        with self.assertRaises(ValidationError):
            mc_price(100, 100, 1, 0.05, 0.2, payoff="barrier_out")
        with self.assertRaises(ValidationError):
            mc_price(100, 100, 1, 0.05, 0.2, payoff="lookback")


class SurfaceTests(unittest.TestCase):
    def test_svi_fit_recovers_the_curve(self):
        k = np.linspace(-0.4, 0.4, 21)
        true = dict(a=0.02, b=0.10, rho=-0.4, m=0.05, sigma=0.15)
        w = svi_total_variance(k, **true) + np.random.default_rng(0).normal(0, 2e-5, len(k))
        fit = fit_svi(k, w)
        self.assertLess(fit["rmse"], 1e-4)
        grid = np.linspace(-0.35, 0.35, 15)
        self.assertLess(np.max(np.abs(fit["w"](grid) - svi_total_variance(grid, **true))), 5e-4)
        self.assertGreater(fit["b"], 0)
        self.assertLess(abs(fit["rho"]), 1)
        with self.assertRaises(StatisticsError):
            fit_svi(k[:3], w[:3])

    def test_no_arbitrage_checks(self):
        K = np.array([80, 90, 100, 110, 120.0])
        C = np.array([bs_price(100, k, 0.5, 0.03, 0.25, "call") for k in K])
        self.assertEqual(check_no_arbitrage(K, C, 100, 0.03, 0.5), [])
        bump = C.copy()
        bump[2] += 3.0                                                                            # middle strike far too expensive
        self.assertTrue(any("butterfly" in v for v in check_no_arbitrage(K, bump, 100, 0.03, 0.5)))
        rising = C.copy()
        rising[3:] = rising[2] + 0.5
        self.assertTrue(any("rises with strike" in v for v in check_no_arbitrage(K, rising, 100, 0.03, 0.5)))
        below = C.copy()
        below[0] = 5.0                                                                            # a deep ITM call below intrinsic
        self.assertTrue(any("intrinsic" in v for v in check_no_arbitrage(K, below, 100, 0.03, 0.5)))
        with self.assertRaises(ValidationError):
            check_no_arbitrage([100, 90, 110], [1, 2, 3], 100, 0.03, 0.5)

    def test_implied_vol_slice(self):
        K = [90, 100, 110]
        prices = [bs_price(100, k, 0.5, 0.03, 0.25, "call") for k in K]
        np.testing.assert_allclose(implied_vol_slice(K, prices, 100, 0.5, 0.03), 0.25, atol=1e-5)
        out = implied_vol_slice(K, [prices[0], -1.0, prices[2]], 100, 0.5, 0.03)
        self.assertTrue(np.isnan(out[1]))                                                          # a bad quote becomes NaN, not an exception


class StrategyTests(unittest.TestCase):
    def test_bull_call_spread_hand_computed(self):
        legs = [OptionLeg("call", 100, 1, 5.0), OptionLeg("call", 110, -1, 2.0)]                 # net debit 3 per share
        np.testing.assert_allclose(payoff(legs, [90, 103, 120]), [-300, 0, 700])
        p = profile(legs, 80, 130)
        self.assertAlmostEqual(p["max_loss"], -300)
        self.assertAlmostEqual(p["max_profit"], 700)
        self.assertAlmostEqual(p["breakevens"][0], 103, places=1)

    def test_straddle_unbounded_and_condor_capped(self):
        st = straddle(100, 100, 0.25, 0.03, 0.2)
        p = profile(st, 40, 160)
        self.assertEqual(p["max_profit"], float("inf"))
        self.assertEqual(len(p["breakevens"]), 2)
        paid = sum(l.premium for l in st)
        self.assertAlmostEqual(p["breakevens"][0], 100 - paid, delta=0.2)
        ic = iron_condor(85, 90, 110, 115, 100, 0.25, 0.03, 0.2)
        pi = profile(ic, 60, 140)
        self.assertGreater(pi["max_profit"], 0)
        self.assertGreaterEqual(pi["max_loss"], -5 * 100)                                          # risk capped at the wing width
        self.assertEqual(len(pi["breakevens"]), 2)
        with self.assertRaises(ValidationError):
            iron_condor(90, 85, 110, 115, 100, 0.25, 0.03, 0.2)
        fly = butterfly("call", 90, 100, 110, 100, 0.25, 0.03, 0.2)
        self.assertEqual(sum(l.qty for l in fly), 0)                                                 # 1 : -2 : 1
        self.assertLess(profile(fly, 50, 150)["max_loss"], 0)

    def test_covered_call_and_greeks(self):
        cc = covered_call(100, 105, 100, 0.25, 0.03, 0.2)
        g = greeks(cc, 100, 0.25, 0.03, 0.2)
        self.assertTrue(0 < g["delta"] < 100)                                                       # long stock hedged by the short call
        self.assertLess(g["gamma"], 0)
        long_straddle = greeks(straddle(100, 100, 0.25, 0.03, 0.2), 100, 0.25, 0.03, 0.2)
        self.assertLess(abs(long_straddle["delta"]), 15)                                            # near delta-neutral
        self.assertGreater(long_straddle["gamma"], 0)
        self.assertGreater(long_straddle["vega"], 0)
        self.assertLess(long_straddle["theta"], 0)
        short = greeks([OptionLeg(l.kind, l.strike, -l.qty, l.premium) for l in straddle(100, 100, 0.25, 0.03, 0.2)], 100, 0.25, 0.03, 0.2)
        self.assertAlmostEqual(short["gamma"], -long_straddle["gamma"])

    def test_value_at_expiry_equals_payoff(self):
        legs = vertical_spread("put", 110, 100, 100, 0.25, 0.03, 0.2)
        for S in (85.0, 105.0, 120.0):
            self.assertAlmostEqual(value(legs, S, 0.0, 0.03, 0.2), float(payoff(legs, S)[0]), places=6)
        with self.assertRaises(ValidationError):
            OptionLeg("future", 100, 1)

    def test_contracts_in_the_portfolio_ledger(self):
        self.assertEqual(option_symbol("AAPL", date(2024, 1, 19), "call", 150), "AAPL240119C00150000")
        inst = option_instrument("AAPL", date(2024, 1, 19), "call", 150)
        self.assertEqual((inst.multiplier, inst.asset_class), (100.0, "option"))
        sym = inst.symbol
        from algoplatform.core import Fill
        pf = Portfolio(100_000, {sym: inst})
        pf.apply_fill(Fill("f1", "o1", sym, Side.BUY, 2, 5.0, None))
        self.assertAlmostEqual(pf.cash, 100_000 - 2 * 5.0 * 100)                                      # premium x multiplier x contracts
        pf.mark({sym: 8.0})
        self.assertAlmostEqual(pf.unrealized_pnl, 600.0)
        pf.apply_fill(expiry_fill(sym, 2, "call", 150, underlying_price=158.0))                      # intrinsic 8
        self.assertAlmostEqual(pf.realized_pnl, 600.0)
        pf2 = Portfolio(100_000, {sym: inst})
        pf2.apply_fill(Fill("f1", "o1", sym, Side.BUY, 2, 5.0, None))
        pf2.apply_fill(expiry_fill(sym, 2, "call", 150, underlying_price=140.0))                     # expires worthless
        self.assertAlmostEqual(pf2.realized_pnl, -1000.0, places=3)
        self.assertIsNone(expiry_fill(sym, 0, "call", 150, 140.0))


def gbm_paths(n_paths=200, steps=40, T=0.25, mu=0.03, sigma=0.2, S0=100.0, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / steps
    z = rng.standard_normal((n_paths, steps))
    inc = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    return S0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), np.cumsum(inc, axis=1)], axis=1))


class HedgeTests(unittest.TestCase):
    def test_hedging_frequency_trades_risk_for_cost(self):
        paths = gbm_paths()
        daily = simulate_delta_hedge(paths, 100, 0.25, 0.03, 0.2, rebalance_every=1)
        weekly = simulate_delta_hedge(paths, 100, 0.25, 0.03, 0.2, rebalance_every=10)
        once = simulate_delta_hedge(paths, 100, 0.25, 0.03, 0.2, rebalance_every=10_000)
        self.assertLess(daily["pnl"].std(), weekly["pnl"].std())
        self.assertLess(weekly["pnl"].std(), once["pnl"].std())                                       # less frequent hedging = more risk
        self.assertLess(abs(daily["pnl"].mean()), 0.25 * once["pnl"].std())                          # fair price: mean P&L ~ 0
        self.assertGreater(daily["trades"].mean(), weekly["trades"].mean())
        c_daily = simulate_delta_hedge(paths, 100, 0.25, 0.03, 0.2, rebalance_every=1, cost_bps=10)
        c_weekly = simulate_delta_hedge(paths, 100, 0.25, 0.03, 0.2, rebalance_every=10, cost_bps=10)
        self.assertGreater(c_daily["cost"].mean(), c_weekly["cost"].mean())                           # more trading, more cost
        self.assertLess(c_daily["pnl"].mean(), daily["pnl"].mean())

    def test_no_trade_band_cuts_trades_and_keeps_most_of_the_risk_reduction(self):
        paths = gbm_paths()
        daily = simulate_delta_hedge(paths, 100, 0.25, 0.03, 0.2, rebalance_every=1)
        banded = simulate_delta_hedge(paths, 100, 0.25, 0.03, 0.2, band=6.0)
        once = simulate_delta_hedge(paths, 100, 0.25, 0.03, 0.2, rebalance_every=10_000)
        self.assertLess(banded["trades"].mean(), daily["trades"].mean())
        self.assertLess(banded["pnl"].std(), once["pnl"].std())

    def test_band_formula_and_trade_rule(self):
        base = whalley_wilmott_band(100, 0.05, 0.001)
        self.assertGreater(whalley_wilmott_band(100, 0.05, 0.004), base)                              # costlier trading: wider band
        self.assertLess(whalley_wilmott_band(100, 0.05, 0.001, risk_aversion=5), base)               # more risk-averse: tighter
        self.assertGreater(whalley_wilmott_band(100, 0.10, 0.001), base)                              # more gamma: wider (more to hedge)
        self.assertEqual(hedge_trade(50, -48, band=5), 0.0)                                           # |net| = 2 inside the band
        self.assertEqual(hedge_trade(50, -30, band=5), -20.0)                                          # |net| = 20: trade back to flat
        with self.assertRaises(ValidationError):
            whalley_wilmott_band(100, 0.05, -1)


if __name__ == "__main__":
    unittest.main()
