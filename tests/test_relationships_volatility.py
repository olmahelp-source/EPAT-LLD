import math
import unittest

import numpy as np
import pandas as pd

import helpers  # noqa: F401
from algoplatform.foundation.core.exceptions import ModelError, StatisticsError, ValidationError
from algoplatform.market_data.data import synthetic
from algoplatform.quantitative.relationships import (RelationshipManager, absorption_ratio, average_correlation, beta, correlation_distance,
                                        correlation_matrix, cross_correlation, ewma_correlation, find_cointegrated_pairs,
                                        granger_causality, hedge_ratio, hierarchical_clusters, lead_lag, lead_lag_matrix,
                                        minimum_spanning_tree, partial_correlation, pca_factors, rolling_beta, spread_zscore,
                                        stress_correlation, tail_dependence)
from algoplatform.quantitative.volatility import (MODELS, EWMAVol, GarchVol, HARVol, HistoricalVol, VolatilityManager, VolRegime, close_to_close,
                                     compare_vol_models, estimators, forecast_volatility, garman_klass, make_model, mincer_zarnowitz,
                                     parkinson, qlike, rogers_satchell, variance_proxy, variance_risk_premium, vol_of_vol,
                                     vol_target_scalar, vol_target_weights, volatility_cone, walk_forward_forecasts, yang_zhang)


def block_returns(n=800, seed=0):
    """Two groups of three series; within a group they share a factor, across groups they do not."""
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(0, 0.01, n), rng.normal(0, 0.01, n)
    cols = {f"A{i}": f1 + rng.normal(0, 0.004, n) for i in range(3)}
    cols.update({f"B{i}": f2 + rng.normal(0, 0.004, n) for i in range(3)})
    return pd.DataFrame(cols, index=pd.bdate_range("2020-01-01", periods=n))


def garch_path(n=3000, omega=2e-6, alpha=0.08, beta_=0.90, seed=0):
    rng = np.random.default_rng(seed)
    h, r = np.empty(n), np.empty(n)
    h[0] = omega / (1 - alpha - beta_)
    for t in range(n):
        r[t] = math.sqrt(h[t]) * rng.normal()
        if t + 1 < n:
            h[t + 1] = omega + alpha * r[t] ** 2 + beta_ * h[t]
    return r, h


class CorrelationTests(unittest.TestCase):
    def setUp(self):
        self.r = block_returns()

    def test_matrix_and_summaries(self):
        c = correlation_matrix(self.r)
        self.assertGreater(c.loc["A0", "A1"], 0.7)
        self.assertLess(abs(c.loc["A0", "B0"]), 0.15)
        self.assertGreater(correlation_matrix(self.r, "spearman").loc["B0", "B2"], 0.7)
        d = correlation_distance(c)
        self.assertEqual(d.loc["A0", "A0"], 0.0)
        self.assertLess(d.loc["A0", "A1"], d.loc["A0", "B0"])
        self.assertAlmostEqual(average_correlation(pd.DataFrame(np.eye(3))), 0.0)
        with self.assertRaises(StatisticsError):
            correlation_matrix(self.r, "cosine")
        with self.assertRaises(StatisticsError):
            correlation_matrix(self.r[["A0"]])

    def test_partial_correlation_removes_the_common_driver(self):
        rng = np.random.default_rng(1)
        m = rng.normal(size=2000)
        df = pd.DataFrame({"market": m, "x": m + rng.normal(0, 0.5, 2000), "y": m + rng.normal(0, 0.5, 2000)})
        self.assertGreater(correlation_matrix(df).loc["x", "y"], 0.7)                 # correlated only through the market
        self.assertLess(abs(partial_correlation(df).loc["x", "y"]), 0.08)

    def test_beta_recovers_the_market_model(self):
        rng = np.random.default_rng(2)
        m = rng.normal(0, 0.01, 1500)
        a = 0.0002 + 1.4 * m + rng.normal(0, 0.003, 1500)
        b = beta(a, m)
        self.assertAlmostEqual(b["beta"], 1.4, delta=0.05)
        self.assertAlmostEqual(b["alpha"], 0.0002, delta=0.0003)
        self.assertGreater(b["r2"], 0.9)
        self.assertAlmostEqual(b["beta_up"], 1.4, delta=0.2)
        rb = rolling_beta(pd.Series(a), pd.Series(m), 250).dropna()
        self.assertAlmostEqual(rb.mean(), 1.4, delta=0.05)
        with self.assertRaises(StatisticsError):
            beta(a[:5], m[:5])
        with self.assertRaises(StatisticsError):
            beta(a, m[:-1])

    def test_ewma_correlation_tracks_a_change_faster_than_the_full_sample(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=600)
        y = np.concatenate([x[:300] + rng.normal(0, 0.3, 300), -x[300:] + rng.normal(0, 0.3, 300)])   # relationship flips sign
        e = ewma_correlation(x, y, 0.94)
        self.assertGreater(e[290], 0.8)
        self.assertLess(e[-1], -0.8)
        self.assertLess(abs(np.corrcoef(x, y)[0, 1]), 0.2)                            # the full-sample number hides both regimes

    def test_tail_dependence_and_stress_correlation(self):
        rng = np.random.default_rng(4)
        n = 6000
        shock = np.where(rng.random(n) < 0.03, rng.normal(-0.08, 0.01, n), 0.0)       # rare common crashes
        a, b = rng.normal(0, 0.01, n) + shock, rng.normal(0, 0.01, n) + shock
        td = tail_dependence(a, b, 0.03)
        self.assertGreater(td["lower"], 0.6)                                          # they crash together ...
        self.assertLess(td["upper"], 0.25)                                            # ... but do not rally together
        sc = stress_correlation(a, b)
        self.assertGreater(sc["stress"], sc["calm"] + 0.3)
        self.assertAlmostEqual(tail_dependence(rng.normal(size=20000), rng.normal(size=20000), 0.05)["lower"], 0.05, delta=0.03)


class LeadLagTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.a = rng.normal(size=1500)
        self.b = np.concatenate([[0.0, 0.0], self.a[:-2]]) * 0.6 + rng.normal(size=1500) * 0.5   # b follows a by two bars
        self.noise = rng.normal(size=1500)

    def test_detects_who_leads_and_by_how_much(self):
        cc = cross_correlation(self.a, self.b, 5)
        self.assertEqual(int(cc.idxmax()), 2)
        res = lead_lag(self.a, self.b, 5)
        self.assertEqual((res["best_lag"], res["leader"], res["significant"]), (2, "a", True))
        swapped = lead_lag(self.b, self.a, 5)
        self.assertEqual((swapped["best_lag"], swapped["leader"]), (-2, "b"))

    def test_noise_is_not_reported_as_a_relationship(self):
        hits = sum(lead_lag(np.random.default_rng(s).normal(size=800), np.random.default_rng(s + 100).normal(size=800), 5)["significant"]
                   for s in range(40))
        self.assertLessEqual(hits, 5)                                                 # ~5% false positives after the multiple-lag correction
        self.assertEqual(lead_lag(self.a, self.noise, 5)["leader"] in ("none", "a", "b"), True)

    def test_granger_direction(self):
        self.assertLess(granger_causality(self.b, self.a, 3)["p_value"], 1e-6)        # a's past predicts b
        self.assertGreater(granger_causality(self.a, self.b, 3)["p_value"], 0.01)     # b's past does not predict a
        self.assertFalse(granger_causality(self.a, self.noise, 2)["causes_5pct"] and granger_causality(self.noise, self.a, 2)["causes_5pct"])
        with self.assertRaises(StatisticsError):
            granger_causality(self.a[:10], self.b[:10], 3)

    def test_matrix_is_antisymmetric(self):
        m = lead_lag_matrix(pd.DataFrame({"a": self.a, "b": self.b, "n": self.noise}), 5)
        self.assertEqual((m.loc["a", "b"], m.loc["b", "a"]), (2, -2))
        self.assertEqual(m.loc["a", "a"], 0)


class CointegrationTests(unittest.TestCase):
    def setUp(self):
        y, x = synthetic.cointegrated_pair(900, beta=1.5, seed=2)
        rng = np.random.default_rng(9)
        walks = {f"W{i}": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 900))) for i in range(3)}
        self.prices = pd.DataFrame({"Y": y["close"].to_numpy(), "X": x["close"].to_numpy(), **walks})

    def test_scan_finds_the_planted_pair(self):
        pairs = find_cointegrated_pairs(self.prices, level="1%", use_log=False)
        self.assertEqual({pairs.iloc[0]["y"], pairs.iloc[0]["x"]}, {"X", "Y"})
        self.assertLess(pairs.iloc[0]["stat"], pairs.iloc[0]["critical"])
        self.assertTrue(1 <= pairs.iloc[0]["half_life"] <= 60)
        self.assertEqual(pairs.attrs["tested"], 10)
        self.assertAlmostEqual(pairs.attrs["expected_false_positives"], 0.1)          # the caveat travels with the result
        self.assertEqual(len(find_cointegrated_pairs(self.prices, max_half_life=0.5)), 0)
        with self.assertRaises(StatisticsError):
            find_cointegrated_pairs(self.prices, level="7%")

    def test_hedge_ratio_and_spread(self):
        self.assertAlmostEqual(hedge_ratio(self.prices["Y"], self.prices["X"]), 1.5, delta=0.1)
        tls_yx, tls_xy = hedge_ratio(self.prices["Y"], self.prices["X"], "tls"), hedge_ratio(self.prices["X"], self.prices["Y"], "tls")
        self.assertAlmostEqual(tls_yx * tls_xy, 1.0, places=6)                        # total least squares is symmetric; OLS is not
        z = spread_zscore(self.prices["Y"], self.prices["X"], 60)
        self.assertTrue(np.isnan(z[:59]).all())
        self.assertLess(np.nanmax(np.abs(z)), 6)
        with self.assertRaises(StatisticsError):
            hedge_ratio(self.prices["Y"], self.prices["X"], "magic")


class StructureTests(unittest.TestCase):
    def setUp(self):
        self.r = block_returns()
        self.c = correlation_matrix(self.r)

    def test_clusters_recover_the_two_groups(self):
        for linkage in ("single", "complete", "average"):
            h = hierarchical_clusters(self.c, n_clusters=2, linkage=linkage)
            lab = h["labels"]
            self.assertEqual(lab[["A0", "A1", "A2"]].nunique(), 1, linkage)
            self.assertEqual(lab[["B0", "B1", "B2"]].nunique(), 1, linkage)
            self.assertNotEqual(lab["A0"], lab["B0"], linkage)
            order = h["order"]
            self.assertEqual({o[0] for o in order[:3]}.__len__(), 1)                  # leaf order keeps each group contiguous
            self.assertEqual(sorted(order), sorted(self.c.index))
        self.assertEqual(hierarchical_clusters(self.c, threshold=0.5)["n_clusters"], 2)
        self.assertEqual(hierarchical_clusters(self.c, n_clusters=6)["n_clusters"], 6)
        for kwargs in ({}, {"n_clusters": 2, "threshold": 0.5}, {"n_clusters": 9}, {"n_clusters": 2, "linkage": "ward"}):
            with self.assertRaises(StatisticsError):
                hierarchical_clusters(self.c, **kwargs)

    def test_spanning_tree(self):
        edges = minimum_spanning_tree(self.c)
        self.assertEqual(len(edges), 5)
        self.assertEqual(sum(1 for a, b, _ in edges if a[0] != b[0]), 1)              # exactly one bridge between the two groups
        self.assertEqual({n for a, b, _ in edges for n in (a, b)}, set(self.c.index))

    def test_pca_and_absorption(self):
        p = pca_factors(self.r, 2)
        self.assertGreater(p["explained_ratio"].sum(), 0.85)                          # two factors generated the data
        self.assertEqual(p["factors"].shape, (len(self.r), 2))
        self.assertGreater(p["loadings"]["PC1"].sum(), 0)
        self.assertAlmostEqual(absorption_ratio(self.r, 6), 1.0)
        one_factor = pd.DataFrame({f"S{i}": self.r["A0"] + np.random.default_rng(i).normal(0, 0.001, len(self.r)) for i in range(5)})
        self.assertGreater(absorption_ratio(one_factor, 1), 0.95)                     # everything moves together
        with self.assertRaises(StatisticsError):
            pca_factors(self.r, 9)


class RelationshipManagerTests(unittest.TestCase):
    def setUp(self):
        r = block_returns(900, seed=3)
        self.prices = (1 + r).cumprod() * 100
        y, x = synthetic.cointegrated_pair(900, seed=2)
        self.prices["Y"], self.prices["X"] = y["close"].to_numpy(), x["close"].to_numpy()
        self.m = RelationshipManager({"A0": "tech", "A1": "tech", "B0": "energy"})
        self.summary = self.m.build(self.prices, corr_threshold=0.6)

    def test_build_and_questions(self):
        self.assertEqual(self.summary["sector"], 1)
        self.assertGreaterEqual(self.summary["cointegrated"], 1)
        self.assertEqual(set(self.m.peers("A0", 2)), {"A1", "A2"})
        self.assertIn({"A0", "A1", "A2"}, self.m.groups())
        self.assertIn({"B0", "B1", "B2"}, self.m.groups())
        pair = self.m.pairs()[0]
        self.assertEqual({pair.a, pair.b}, {"X", "Y"})
        self.assertIn("hedge_ratio", pair.meta)
        self.assertEqual(self.m.related("A0", "sector")[0].other("A0"), "A1")
        self.assertEqual(len(self.m.to_frame()), sum(self.summary.values()))

    def test_concentration_counts_related_positions_as_one_bet(self):
        same = self.m.concentration({"A0": 0.25, "A1": 0.25, "A2": 0.25, "B0": 0.25})
        self.assertEqual(same["positions"], 4)
        self.assertAlmostEqual(same["largest_share"], 0.75)
        self.assertAlmostEqual(same["effective_bets"], 1 / (0.75 ** 2 + 0.25 ** 2))   # 1.6 bets, not 4
        spread = self.m.concentration({"A0": 0.5, "B0": 0.5})
        self.assertAlmostEqual(spread["effective_bets"], 2.0)
        self.assertEqual(self.m.concentration({})["effective_bets"], 0.0)

    def test_manual_relations_survive_a_rebuild_and_errors_are_recorded(self):
        self.m.add("A0", "B0", "supplier", 0.9, directed=True, note="A0 sells to B0")
        self.m.build(self.prices, corr_threshold=0.6)
        self.assertEqual(len(self.m.relations("supplier")), 1)
        with self.assertRaises(ValidationError):
            self.m.add("A0", "A0", "self")
        flat = self.prices.assign(FLAT=100.0)
        m2 = RelationshipManager()
        m2.build(flat, corr_threshold=0.6)                                            # a constant series must not stop the build
        self.assertTrue(any("FLAT" in e for e in m2.errors))
        self.assertIn({"A0", "A1", "A2"}, m2.groups())


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        self.df = synthetic.gbm(2500, sigma=0.2, seed=4)

    def test_all_estimators_recover_the_true_volatility(self):
        o, h, l, c = (self.df[k] for k in ("open", "high", "low", "close"))
        est = {"close_to_close": close_to_close(c, 60), "parkinson": parkinson(h, l, 60), "garman_klass": garman_klass(o, h, l, c, 60),
               "rogers_satchell": rogers_satchell(o, h, l, c, 60), "yang_zhang": yang_zhang(o, h, l, c, 60), "ewma": estimators.ewma(c)}
        for name, v in est.items():
            self.assertEqual(len(v), len(c), name)
            self.assertAlmostEqual(float(np.nanmean(v)), 0.2, delta=0.03, msg=name)
        self.assertTrue(np.isnan(est["yang_zhang"][:60]).all())
        with self.assertRaises(ValidationError):
            yang_zhang(o, h, l, c, 2)
        with self.assertRaises(ValidationError):
            estimators.ewma(c, 1.5)

    def test_range_estimators_are_less_noisy_than_close_to_close(self):
        rng = np.random.default_rng(0)
        steps = rng.normal(0, 0.2 / math.sqrt(252 * 78), (1500, 78))                  # true intraday paths, no gaps
        path = np.cumsum(steps, axis=1)
        prev = np.concatenate([[0.0], np.cumsum(path[:, -1])[:-1]])
        o, c = 100 * np.exp(prev), 100 * np.exp(prev + path[:, -1])
        h, l = 100 * np.exp(prev + np.maximum(path.max(axis=1), 0)), 100 * np.exp(prev + np.minimum(path.min(axis=1), 0))
        cc, gk = close_to_close(c, 20), garman_klass(o, h, l, c, 20)
        self.assertLess(np.nanstd(gk), 0.7 * np.nanstd(cc))                           # same target, tighter estimate

    def test_cone_proxy_and_vol_of_vol(self):
        cone = volatility_cone(self.df["close"])
        self.assertEqual(list(cone.index), [10, 21, 63, 126, 252])
        self.assertTrue((cone["min"] <= cone["q50"]).all() and (cone["q50"] <= cone["max"]).all())
        self.assertGreater(cone.loc[10, "max"] - cone.loc[10, "min"], cone.loc[252, "max"] - cone.loc[252, "min"])   # short windows vary more
        self.assertTrue(((cone["percentile"] >= 0) & (cone["percentile"] <= 1)).all())
        o, h, l, c = (self.df[k] for k in ("open", "high", "low", "close"))
        for kind in ("squared_return", "parkinson", "garman_klass"):
            v = variance_proxy(o, h, l, c, kind)
            self.assertAlmostEqual(math.sqrt(np.nanmean(v) * 252), 0.2, delta=0.035, msg=kind)
        with self.assertRaises(ValidationError):
            variance_proxy(o, h, l, c, "magic")
        self.assertTrue((vol_of_vol(close_to_close(c, 20), 20)[60:] > 0).all())
        with self.assertRaises(ValidationError):
            volatility_cone(c.iloc[:5])


class VolModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r, cls.h = garch_path()
        cls.garch = GarchVol().fit(cls.r)

    def test_garch_recovers_simulated_parameters(self):
        p = self.garch.params
        self.assertAlmostEqual(p["alpha"], 0.08, delta=0.03)
        self.assertAlmostEqual(p["beta"], 0.90, delta=0.04)
        self.assertAlmostEqual(p["persistence"], 0.98, delta=0.015)
        self.assertAlmostEqual(p["long_run_vol"], math.sqrt(2e-6 / 0.02 * 252), delta=0.02)
        fitted, truth = self.garch.conditional_vol()[200:], np.sqrt(self.h[200:] * 252)
        self.assertGreater(np.corrcoef(fitted, truth)[0, 1], 0.95)                    # tracks the true conditional volatility

    def test_forecast_mean_reverts_to_the_long_run_level(self):
        short, long = self.garch.forecast(1), self.garch.forecast(2000)
        lr = self.garch.params["long_run_vol"]
        self.assertAlmostEqual(math.sqrt(long.variance_path[-1] * 252), lr, delta=0.002)
        self.assertLessEqual(abs(long.vol - lr), abs(short.next_vol - lr) + 1e-12)    # the longer the horizon, the closer to the long run
        self.assertEqual(len(self.garch.forecast(7).variance_path), 7)
        self.assertEqual(self.garch.forecast(1).vol, short.next_vol)

    def test_gjr_detects_the_leverage_effect(self):
        rng = np.random.default_rng(1)
        n, r, h = 4000, np.empty(4000), 1e-4
        for t in range(n):
            r[t] = math.sqrt(h) * rng.normal()
            h = 2e-6 + (0.02 + (0.12 if r[t] < 0 else 0.0)) * r[t] ** 2 + 0.88 * h    # falls raise volatility more than rises
        gjr = GarchVol(asymmetric=True).fit(r)
        self.assertGreater(gjr.params["gamma"], 0.05)
        self.assertEqual(gjr.name, "gjr")
        self.assertLess(GarchVol(asymmetric=True).fit(self.r).params["gamma"], 0.04)  # symmetric data: no leverage term found

    def test_simple_models_and_har(self):
        hist, ew = HistoricalVol(20).fit(self.r), EWMAVol(0.94).fit(self.r)
        self.assertAlmostEqual(hist.forecast(5).next_vol, math.sqrt(np.mean(self.r[-20:] ** 2) * 252), places=10)
        np.testing.assert_allclose(ew.forecast(10).variance_path, ew.forecast(1).variance_path[0])       # EWMA forecast is flat
        self.assertAlmostEqual(hist.conditional_vol()[100], math.sqrt(np.mean(self.r[80:100] ** 2) * 252), places=10)   # uses bars before t only
        har = HARVol().fit(self.r)
        self.assertEqual(set(har.params), {"const", "b1", "b5", "b22"})
        self.assertGreater(har.forecast(22).vol, 0.05)
        proxy = HARVol().fit(self.r, variance=self.r ** 2 * 1.0)
        self.assertAlmostEqual(proxy.forecast(1).vol, har.forecast(1).vol, places=10)

    def test_conditional_vol_is_causal(self):
        for name in ("hist", "ewma"):
            a, b = make_model(name).fit(self.r[:1000]).conditional_vol(), make_model(name).fit(self.r[:600]).conditional_vol()
            np.testing.assert_allclose(a[:600], b, err_msg=name)

    def test_errors(self):
        with self.assertRaises(ModelError):
            GarchVol().forecast(1)
        with self.assertRaises(ModelError):
            EWMAVol().conditional_vol()
        with self.assertRaises(StatisticsError):
            GarchVol().fit(self.r[:50])
        with self.assertRaises(ValidationError):
            self.garch.forecast(0)
        with self.assertRaises(ValidationError):
            make_model("lstm")
        with self.assertRaises(ValidationError):
            HARVol(lags=(5, 1))
        self.assertEqual(set(MODELS), {"hist", "ewma", "garch", "gjr", "har"})
        self.assertAlmostEqual(forecast_volatility(self.r, "ewma", 3).vol, EWMAVol().fit(self.r).forecast(3).vol)


class VolEvaluationTests(unittest.TestCase):
    def test_qlike_ranks_the_true_variance_first(self):
        r, h = garch_path(4000, seed=3)
        realized = r ** 2
        self.assertLess(qlike(realized, h), qlike(realized, np.full(len(h), h.mean())))      # truth beats a constant
        self.assertLess(qlike(realized, h), qlike(realized, h * 2))                           # ... and a biased forecast
        self.assertLess(qlike(realized, h * 2), qlike(realized, h * 0.5))                     # under-prediction costs more than over-prediction
        self.assertAlmostEqual(qlike(h, h), 0.0)
        mz = mincer_zarnowitz(realized, h)
        self.assertAlmostEqual(mz["slope"], 1.0, delta=0.3)                                   # unbiased: slope near 1
        with self.assertRaises(StatisticsError):
            qlike([1.0], [1.0])

    def test_walk_forward_and_league_table(self):
        r, _ = garch_path(1600, seed=5)
        wf = walk_forward_forecasts(r, EWMAVol, train=500, step=50, horizon=5)
        self.assertEqual(list(wf.index[:3]), [500, 550, 600])
        self.assertAlmostEqual(wf["realized_var"].iloc[0], float(np.mean(r[500:505] ** 2)))
        self.assertAlmostEqual(wf["forecast_var"].iloc[0], float(EWMAVol().fit(r[:500]).forecast(5).variance_path.mean()))   # fitted on the past only

        class Broken(EWMAVol):
            def _fit(self, r):
                raise StatisticsError("no convergence")
        table = compare_vol_models(r, {"ewma": EWMAVol, "hist": HistoricalVol, "broken": Broken}, train=500, step=100)
        self.assertEqual(list(table["model"])[-1], "broken")                          # listed with its error, not dropped
        self.assertEqual(table.iloc[-1]["error"], "no convergence")
        self.assertTrue(table["qlike"].iloc[:2].is_monotonic_increasing)
        with self.assertRaises(StatisticsError):
            walk_forward_forecasts(r[:100], EWMAVol, train=500)


class VolManagementTests(unittest.TestCase):
    def test_target_scalar(self):
        self.assertAlmostEqual(vol_target_scalar(0.20, 0.10), 0.5)
        self.assertEqual(vol_target_scalar(0.01, 0.10, max_leverage=2.0), 2.0)
        self.assertEqual(vol_target_scalar(float("nan"), 0.10, min_scalar=0.1), 0.1)  # unknown volatility: be small
        self.assertEqual(vol_target_scalar(0.0, 0.10), 0.0)
        with self.assertRaises(ValidationError):
            vol_target_scalar(0.2, -1)
        self.assertAlmostEqual(variance_risk_premium(0.25, 0.20), 0.0225)

    def test_target_weights_stabilise_risk_without_look_ahead(self):
        rng = np.random.default_rng(0)
        r = pd.Series(np.concatenate([rng.normal(0, 0.005, 600), rng.normal(0, 0.03, 600)]))    # calm then wild
        w = vol_target_weights(r, 0.10, window=20, max_leverage=3.0)
        managed = (r * w).dropna()
        calm, wild = managed.iloc[100:550].std() * math.sqrt(252), managed.iloc[700:].std() * math.sqrt(252)
        self.assertAlmostEqual(calm, 0.10, delta=0.03)
        self.assertAlmostEqual(wild, 0.10, delta=0.03)
        r2 = r.copy()
        r2.iloc[900:] *= 5
        pd.testing.assert_series_equal(vol_target_weights(r2, 0.10, 20, 3.0).iloc[:901], w.iloc[:901])   # weight at t uses returns before t

    def test_regime_classifier_with_hysteresis(self):
        vol = np.concatenate([np.full(200, 0.10), np.full(60, 0.30), np.full(100, 0.10)]) + np.random.default_rng(0).normal(0, 0.002, 360)
        reg = VolRegime(lookback=150, min_history=60)
        labels = reg.classify(vol)
        self.assertEqual(labels[10], "normal")                                        # not enough history yet
        self.assertEqual(labels[205], "high")
        self.assertTrue(set(labels[330:]) <= {"low", "normal"})
        pct = reg.percentiles(vol)
        self.assertTrue(np.isnan(pct[:60]).all())
        np.testing.assert_array_equal(reg.classify(vol[:250]), labels[:250])          # causal
        flips = lambda lab: int((lab[1:] != lab[:-1]).sum())                          # noqa: E731
        wobble = 0.10 + 0.02 * np.sin(np.arange(600) / 3.0) + np.random.default_rng(1).normal(0, 0.004, 600)
        self.assertLessEqual(flips(VolRegime(150, buffer=0.15, min_history=60).classify(wobble)),
                             flips(VolRegime(150, buffer=0.0, min_history=60).classify(wobble)))
        with self.assertRaises(ValidationError):
            VolRegime(low_cut=0.9, high_cut=0.5)

    def test_manager_cuts_exposure_when_volatility_is_high(self):
        rng = np.random.default_rng(2)
        calm = rng.normal(0, 0.006, 600)
        wild = np.concatenate([calm, rng.normal(0, 0.035, 40)])
        vm = VolatilityManager(target_vol=0.10, model="ewma")
        a, b = vm.assess(calm), vm.assess(wild)
        self.assertEqual(b["regime"], "high")
        self.assertLess(b["scalar"], a["scalar"] * 0.5)
        self.assertAlmostEqual(b["scalar"], b["base_scalar"] * 0.5)                   # the regime haircut on top of volatility targeting
        self.assertAlmostEqual(a["base_scalar"], min(0.10 / a["forecast_vol"], 2.0))
        with self.assertRaises(ValidationError):
            VolatilityManager(model="nope")


if __name__ == "__main__":
    unittest.main()
