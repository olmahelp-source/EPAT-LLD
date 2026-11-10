import math
import unittest

import numpy as np

import helpers  # noqa: F401
from algoplatform.core.exceptions import StatisticsError, ValidationError
from algoplatform.data import synthetic
from algoplatform.quant import (KalmanHedge, bs_greeks, bs_price, ema, ewma_volatility, garch11_fit, garch11_filter,
                                implied_vol, kalman_smooth_1d, ledoit_wolf, nearest_psd, parkinson_vol, put_call_parity_gap,
                                realized_vol, rsi, sma, atr, bollinger, macd, donchian, adx, stochastic, zscore,
                                crossed_above, crossed_below, true_range, vwap, obv)
from algoplatform.stats import (adf_test, autocorrelation, benjamini_hochberg, bonferroni, bootstrap_ci, chi2_sf,
                                deflated_sharpe, engle_granger, excess_kurtosis, expected_max_sharpe, hurst_exponent,
                                jarque_bera, kpss_test, ljung_box, lo_adjusted_sharpe, min_track_record_length,
                                monte_carlo_trades, norm_cdf, norm_ppf, ols, ou_half_life, probabilistic_sharpe,
                                sharpe_ratio, skewness, t_ppf, t_sf, t_test_mean, variance_ratio)


def ar1(n, phi, seed=0, sigma=1.0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, sigma)
    return x


class SpecialFunctionTests(unittest.TestCase):
    def test_normal(self):
        self.assertAlmostEqual(norm_cdf(0), 0.5)
        self.assertAlmostEqual(norm_ppf(0.975), 1.959964, places=5)
        self.assertAlmostEqual(norm_cdf(norm_ppf(0.123)), 0.123, places=9)
        self.assertEqual(norm_ppf(1.0), math.inf)

    def test_chi2_and_t(self):
        self.assertAlmostEqual(chi2_sf(3.841459, 1), 0.05, places=4)
        self.assertAlmostEqual(chi2_sf(18.307, 10), 0.05, places=3)
        self.assertAlmostEqual(t_sf(2.228, 10), 0.025, places=3)
        self.assertAlmostEqual(t_ppf(0.975, 10), 2.228, places=2)
        self.assertAlmostEqual(t_sf(0.0, 5), 0.5)


class RegressionTests(unittest.TestCase):
    def test_recovers_coefficients_and_significance(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=500)
        y = 1.5 + 2.0 * x + rng.normal(0, 0.5, 500)
        r = ols(y, x, names=["x"])
        self.assertAlmostEqual(r["x"], 2.0, delta=0.1)
        self.assertAlmostEqual(r["const"], 1.5, delta=0.1)
        self.assertLess(r.p[1], 1e-10)
        self.assertGreater(r.r2, 0.9)

    def test_newey_west_wider_with_autocorrelated_errors(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=400)
        e = ar1(400, 0.8, seed=5)
        y = 0.5 * x + e
        plain, hac = ols(y, x), ols(y, x, hac_lags=10)
        self.assertGreater(hac.se[0], plain.se[0])

    def test_too_few_observations(self):
        with self.assertRaises(StatisticsError):
            ols([1, 2], [1, 2])


class TestsOfStationarity(unittest.TestCase):
    def test_adf_distinguishes_stationary_from_random_walk(self):
        stationary = adf_test(ar1(600, 0.5, seed=1))
        walk = adf_test(np.cumsum(np.random.default_rng(2).normal(size=600)))
        self.assertTrue(stationary["reject_5pct"])
        self.assertFalse(walk["reject_5pct"])
        self.assertLess(stationary["critical_values"]["1%"], stationary["critical_values"]["10%"])

    def test_kpss_opposite_null(self):
        self.assertFalse(kpss_test(ar1(500, 0.3, seed=4))["reject_5pct"])
        self.assertTrue(kpss_test(np.cumsum(np.random.default_rng(0).normal(size=500)))["reject_5pct"])

    def test_engle_granger_detects_cointegration(self):
        y, x = synthetic.cointegrated_pair(n=800, beta=1.5, theta=0.1, seed=1)
        res = engle_granger(y["close"].values, x["close"].values)
        self.assertTrue(res["cointegrated_5pct"])
        self.assertAlmostEqual(res["hedge_ratio"], 1.5, delta=0.1)
        self.assertLess(res["half_life"], 30)

    def test_engle_granger_rejects_independent_walks(self):
        rng = np.random.default_rng(11)
        a, b = np.cumsum(rng.normal(size=800)) + 100, np.cumsum(rng.normal(size=800)) + 100
        self.assertFalse(engle_granger(a, b)["cointegrated_5pct"])

    def test_half_life_matches_theory(self):
        hl = ou_half_life(ar1(3000, 0.9, seed=7))["half_life"]
        self.assertAlmostEqual(hl, math.log(2) / -math.log(0.9), delta=1.5)          # ~6.6 bars
        self.assertEqual(ou_half_life(np.cumsum(np.random.default_rng(1).normal(size=500)))["half_life"] > 50, True)

    def test_hurst_and_variance_ratio(self):
        rw = np.cumsum(np.random.default_rng(8).normal(size=3000))
        self.assertAlmostEqual(hurst_exponent(rw), 0.5, delta=0.1)
        self.assertLess(hurst_exponent(ar1(3000, 0.7, seed=9)), 0.4)
        r_mr = np.diff(ar1(3000, 0.9, seed=3))
        self.assertLess(variance_ratio(np.random.default_rng(1).normal(size=3000), 5)["vr"] - 1, 0.2)
        self.assertLess(variance_ratio(-0.4 * np.roll(np.random.default_rng(2).normal(size=3000), 1) +
                                       np.random.default_rng(2).normal(size=3000), 5)["vr"], 1.0)
        self.assertGreater(len(r_mr), 0)


class DescriptiveTests(unittest.TestCase):
    def test_moments_and_normality(self):
        x = np.random.default_rng(0).normal(size=5000)
        self.assertAlmostEqual(skewness(x), 0, delta=0.1)
        self.assertAlmostEqual(excess_kurtosis(x), 0, delta=0.2)
        self.assertGreater(jarque_bera(x)["p_value"], 0.01)
        fat = np.random.default_rng(0).standard_t(3, 5000)
        self.assertLess(jarque_bera(fat)["p_value"], 1e-6)

    def test_autocorrelation_and_ljung_box(self):
        white = np.random.default_rng(0).normal(size=1000)
        self.assertGreater(ljung_box(white, 10)["p_value"], 0.01)
        self.assertLess(ljung_box(ar1(1000, 0.6), 10)["p_value"], 1e-6)
        self.assertAlmostEqual(autocorrelation(ar1(5000, 0.6, seed=2), 1)[1], 0.6, delta=0.05)

    def test_t_test(self):
        x = np.random.default_rng(0).normal(0.5, 1, 200)
        self.assertLess(t_test_mean(x)["p_value"], 1e-5)
        self.assertGreater(t_test_mean(x - 0.5)["p_value"], 0.01)


class InferenceTests(unittest.TestCase):
    def test_bootstrap_ci_contains_truth(self):
        x = np.random.default_rng(0).normal(1.0, 2.0, 400)
        ci = bootstrap_ci(x, np.mean, n_boot=500)
        self.assertTrue(ci["low"] < 1.0 < ci["high"])
        ci2 = bootstrap_ci(x, np.mean, n_boot=200, block=5)                         # stationary bootstrap path
        self.assertLess(ci2["low"], ci2["high"])

    def test_psr_and_dsr_behaviour(self):
        r = np.random.default_rng(1).normal(0.001, 0.01, 750)
        psr0 = probabilistic_sharpe(r, 0.0)
        self.assertGreater(psr0, 0.5)
        d1 = deflated_sharpe(r, n_trials=1, var_trials=0.0001)["deflated_sharpe"]
        d100 = deflated_sharpe(r, n_trials=100, var_trials=0.0001)["deflated_sharpe"]
        d1000 = deflated_sharpe(r, n_trials=1000, var_trials=0.0001)["deflated_sharpe"]
        self.assertGreater(d1, d100)
        self.assertGreater(d100, d1000)                                              # more trials => more skepticism
        self.assertGreater(expected_max_sharpe(1000, 0.01), expected_max_sharpe(10, 0.01))

    def test_min_track_record(self):
        self.assertTrue(math.isinf(min_track_record_length(0.0)))
        self.assertGreater(min_track_record_length(0.05), min_track_record_length(0.15))

    def test_sharpe_helpers(self):
        r = np.random.default_rng(0).normal(0.001, 0.01, 2000)
        self.assertAlmostEqual(sharpe_ratio(r), r.mean() / r.std(ddof=1) * math.sqrt(252), places=9)
        self.assertTrue(math.isfinite(lo_adjusted_sharpe(r)))

    def test_multiple_testing(self):
        p = [0.001, 0.01, 0.03, 0.04, 0.5]
        bf, bh = bonferroni(p), benjamini_hochberg(p)
        self.assertEqual(bf["reject"].tolist(), [True, True, False, False, False])
        self.assertEqual(bh["reject"].tolist(), [True, True, True, True, False])
        self.assertTrue((bh["adjusted"] <= bf["adjusted"] + 1e-12).all())

    def test_monte_carlo_trade_reshuffle(self):
        rng = np.random.default_rng(0)
        pnl = rng.normal(50, 500, 100)
        mc = monte_carlo_trades(pnl, 100_000, n_sims=300)
        self.assertGreater(mc["max_dd_p95"], mc["max_dd_p50"])
        self.assertAlmostEqual(np.median(mc["finals"]), 100_000 + pnl.sum(), delta=1)   # order does not change the end


class QuantModelTests(unittest.TestCase):
    def test_black_scholes_reference_values(self):
        c = bs_price(100, 100, 1.0, 0.05, 0.2, "call")
        p = bs_price(100, 100, 1.0, 0.05, 0.2, "put")
        self.assertAlmostEqual(c, 10.4506, places=3)
        self.assertAlmostEqual(p, 5.5735, places=3)
        self.assertAlmostEqual(put_call_parity_gap(c, p, 100, 100, 1.0, 0.05), 0.0, places=9)
        self.assertEqual(bs_price(110, 100, 0.0, 0.05, 0.2, "call"), 10.0)           # expiry => intrinsic

    def test_greeks(self):
        g = bs_greeks(100, 100, 1.0, 0.05, 0.2, "call")
        self.assertAlmostEqual(g["delta"], 0.6368, places=3)
        self.assertAlmostEqual(g["gamma"], 0.01876, places=4)
        self.assertAlmostEqual(g["vega"], 0.3752, places=3)
        gp = bs_greeks(100, 100, 1.0, 0.05, 0.2, "put")
        self.assertAlmostEqual(g["delta"] - gp["delta"], 1.0, places=9)             # call delta - put delta = 1
        h = 0.01
        num_delta = (bs_price(100 + h, 100, 1, .05, .2) - bs_price(100 - h, 100, 1, .05, .2)) / (2 * h)
        self.assertAlmostEqual(num_delta, g["delta"], places=5)

    def test_implied_vol_roundtrip_and_bounds(self):
        price = bs_price(100, 110, 0.5, 0.03, 0.35, "put")
        self.assertAlmostEqual(implied_vol(price, 100, 110, 0.5, 0.03, "put"), 0.35, places=5)
        with self.assertRaises(StatisticsError):
            implied_vol(0.0001 - 5, 100, 100, 1, 0.05)
        with self.assertRaises(ValidationError):
            bs_price(100, 100, 1, 0.05, -0.1)

    def test_volatility_estimators(self):
        r = np.random.default_rng(0).normal(0, 0.01, 2000)
        self.assertAlmostEqual(ewma_volatility(r)[-1], 0.01, delta=0.004)
        self.assertAlmostEqual(realized_vol(r), 0.01 * math.sqrt(252), delta=0.01)
        df = synthetic.gbm(500, sigma=0.3, seed=2)
        self.assertGreater(parkinson_vol(df.high, df.low), 0)

    def test_garch_fit_recovers_persistence(self):
        rng = np.random.default_rng(0)
        omega, a, b, n = 2e-6, 0.08, 0.90, 4000
        r, v = np.zeros(n), np.full(n, omega / (1 - a - b))
        for t in range(1, n):
            v[t] = omega + a * r[t - 1] ** 2 + b * v[t - 1]
            r[t] = math.sqrt(v[t]) * rng.normal()
        fit = garch11_fit(r)
        self.assertAlmostEqual(fit["persistence"], 0.98, delta=0.06)
        self.assertEqual(len(garch11_filter(r, fit["omega"], fit["alpha"], fit["beta"])), n)
        with self.assertRaises(ValidationError):
            garch11_filter(r, 1e-6, 0.5, 0.6)

    def test_ledoit_wolf_and_psd(self):
        X = np.random.default_rng(0).normal(size=(30, 20))
        lw = ledoit_wolf(X)
        self.assertTrue(0 < lw["shrinkage"] <= 1)
        self.assertGreater(np.linalg.eigvalsh(lw["cov"]).min(), 0)                  # well conditioned even when n ~ p
        bad = np.array([[1, 2], [2, 1.0]])
        self.assertGreater(np.linalg.eigvalsh(nearest_psd(bad)).min(), 0)

    def test_kalman_hedge_tracks_beta(self):
        rng = np.random.default_rng(0)
        x = 100 + np.cumsum(rng.normal(0, 1, 800))
        y = 2.0 * x + 5 + rng.normal(0, 0.5, 800)
        kf = KalmanHedge(delta=1e-4, obs_var=0.25)
        for xi, yi in zip(x, y):
            out = kf.update(xi, yi)
        self.assertAlmostEqual(out["beta"], 2.0, delta=0.1)
        sm = kalman_smooth_1d(x + rng.normal(0, 3, 800), q=1.0, r=9.0)     # q, r matched to the data scale
        self.assertLess(np.std(sm - x), 3.0)


class IndicatorTests(unittest.TestCase):
    def setUp(self):
        df = synthetic.gbm(300, seed=1)
        self.o, self.h, self.l, self.c, self.v = (df[k].values for k in ("open", "high", "low", "close", "volume"))

    def test_warmup_nans_and_lengths(self):
        for arr, warm in ((sma(self.c, 20), 19), (ema(self.c, 20), 19), (rsi(self.c, 14), 14), (atr(self.h, self.l, self.c, 14), 13)):
            self.assertEqual(len(arr), 300)
            self.assertTrue(np.isnan(arr[:warm]).all())
            self.assertFalse(np.isnan(arr[warm + 1:]).any())

    def test_sma_and_ema_values(self):
        self.assertAlmostEqual(sma([1, 2, 3, 4, 5], 3)[-1], 4.0)
        e = ema([1, 2, 3, 4, 5, 6], 3)
        self.assertAlmostEqual(e[2], 2.0)                                           # seeded with SMA
        self.assertAlmostEqual(e[3], 0.5 * 4 + 0.5 * 2.0)

    def test_rsi_bounds_and_extremes(self):
        r = rsi(self.c, 14)
        self.assertTrue(((r[14:] >= 0) & (r[14:] <= 100)).all())
        self.assertEqual(rsi(np.arange(1, 40, dtype=float), 14)[-1], 100.0)

    def test_indicators_are_causal(self):
        """Changing future data must not change past indicator values."""
        c2 = self.c.copy()
        c2[200:] *= 1.5
        h2, l2 = self.h.copy(), self.l.copy()
        h2[200:] *= 1.5
        l2[200:] *= 1.5
        for f in (lambda c, h, l: sma(c, 20), lambda c, h, l: ema(c, 20), lambda c, h, l: rsi(c, 14),
                  lambda c, h, l: atr(h, l, c, 14), lambda c, h, l: bollinger(c, 20)["upper"],
                  lambda c, h, l: macd(c)["hist"], lambda c, h, l: zscore(c, 20),
                  lambda c, h, l: donchian(h, l, 20)["upper"], lambda c, h, l: adx(h, l, c, 14)["adx"],
                  lambda c, h, l: stochastic(h, l, c)["k"]):
            a, b = f(self.c, self.h, self.l), f(c2, h2, l2)
            np.testing.assert_allclose(a[:200], b[:200], rtol=1e-12, equal_nan=True)

    def test_bollinger_macd_donchian_shapes(self):
        bb = bollinger(self.c, 20, 2)
        self.assertTrue((bb["upper"][19:] >= bb["mid"][19:]).all())
        m = macd(self.c)
        np.testing.assert_allclose(m["hist"][40:], (m["macd"] - m["signal"])[40:])
        d = donchian(self.h, self.l, 10)
        self.assertTrue((d["upper"][9:] >= d["lower"][9:]).all())
        a = adx(self.h, self.l, self.c, 14)["adx"]
        self.assertTrue(((a[~np.isnan(a)] >= 0) & (a[~np.isnan(a)] <= 100)).all())

    def test_crosses_vwap_obv_true_range(self):
        self.assertTrue(crossed_above([1, 3], [2, 2]))
        self.assertFalse(crossed_above([3, 4], [2, 2]))
        self.assertTrue(crossed_below([3, 1], [2, 2]))
        self.assertAlmostEqual(vwap([10, 20], [1, 3])[-1], 17.5)
        self.assertEqual(obv([1, 2, 1, 1], [10, 10, 5, 7]).tolist(), [0, 10, 5, 5])
        self.assertEqual(true_range([10, 12], [9, 11], [9.5, 11.5])[1], 2.5)         # |12 - 9.5| dominates
        with self.assertRaises(ValidationError):
            sma([1, 2, 3], 0)


if __name__ == "__main__":
    unittest.main()
