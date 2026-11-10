import math
import unittest

import numpy as np
import pandas as pd

from helpers import free_config, market_from_closes
from algoplatform.analytics.signals import (forward_returns, ic_decay, ic_summary, information_coefficient,
                                            quantile_returns, signal_autocorrelation, signal_correlation, signal_turnover)
from algoplatform.backtest import BacktestEngine
from algoplatform.core.exceptions import StrategyError, ValidationError
from algoplatform.research import (CombinatorialPurgedCV, LogisticModel, PurgedKFold, RidgeModel, Scaler, SklearnAdapter,
                                   average_uniqueness, brier_score, build_features, calibration_table, cv_scores,
                                   daily_vol, ffd_weights, frac_diff, log_loss, min_stationary_d, num_concurrent,
                                   out_of_range_share, permutation_importance, probability_of_backtest_overfitting, psi, sample_weights,
                                   sequential_bootstrap, triple_barrier)
from algoplatform.research.models import accuracy
from algoplatform.strategy import STRATEGIES, MLStrategy


def series(vals, start="2024-01-01"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)), dtype=float)


class TripleBarrierTests(unittest.TestCase):
    def test_profit_stop_and_time_barriers(self):
        up = series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        tgt = pd.Series(0.02, index=up.index)
        r = triple_barrier(up, up.index[:1], tgt, horizon=5)
        self.assertEqual((r.iloc[0]["barrier"], r.iloc[0]["label"]), ("pt", 1.0))
        self.assertEqual(r.iloc[0]["t1"], up.index[2])                                   # 102 is +2%: touched on bar 2
        self.assertAlmostEqual(r.iloc[0]["ret"], 0.02)
        dn = series([100, 99, 98, 97, 96, 95, 94, 93, 92, 91])
        r = triple_barrier(dn, dn.index[:1], tgt, horizon=5)
        self.assertEqual((r.iloc[0]["barrier"], r.iloc[0]["label"]), ("sl", -1.0))
        flat = series([100, 100.2, 100.1, 100.3, 100.2, 100.4, 100, 100, 100, 100])
        r = triple_barrier(flat, flat.index[:1], tgt, horizon=5)
        self.assertEqual((r.iloc[0]["barrier"], r.iloc[0]["label"]), ("time", 1.0))      # sign of the final return
        self.assertEqual(r.iloc[0]["t1"], flat.index[5])

    def test_stop_wins_a_same_bar_tie_and_barriers_can_be_disabled(self):
        c = series([100, 90, 100, 100, 100, 100, 100])                                   # -10% bar
        tgt = pd.Series(0.05, index=c.index)
        r = triple_barrier(c, c.index[:1], tgt, 3)
        self.assertEqual(r.iloc[0]["barrier"], "sl")
        r2 = triple_barrier(c, c.index[:1], tgt, 3, pt_sl=(1.0, 0.0))                    # no stop: falls through to time
        self.assertEqual(r2.iloc[0]["barrier"], "time")

    def test_events_without_full_horizon_are_dropped_not_truncated(self):
        c = series(np.linspace(100, 110, 8))
        tgt = pd.Series(0.5, index=c.index)
        r = triple_barrier(c, c.index, tgt, horizon=3)
        self.assertEqual(len(r), 5)                                                      # events 0..4: each needs bars p0+1..p0+3 to exist
        self.assertTrue((r["t1"] <= c.index[-1]).all())

    def test_meta_labels_follow_the_primary_side(self):
        up = series([100, 101, 102, 103, 104, 105, 106, 107])
        tgt = pd.Series(0.02, index=up.index)
        long = triple_barrier(up, up.index[:1], tgt, 5, side=pd.Series(1.0, index=up.index))
        short = triple_barrier(up, up.index[:1], tgt, 5, side=pd.Series(-1.0, index=up.index))
        self.assertEqual((long.iloc[0]["label"], short.iloc[0]["label"]), (1.0, 0.0))    # same path, opposite bet
        self.assertLess(short.iloc[0]["ret"], 0)

    def test_validation(self):
        c = series(np.arange(100, 120.0))
        with self.assertRaises(ValidationError):
            triple_barrier(c, c.index[:1], pd.Series(0.01, index=c.index), 0)
        with self.assertRaises(ValidationError):
            triple_barrier(c, pd.DatetimeIndex(["2030-01-01"]), pd.Series(0.01, index=c.index), 3)

    def test_daily_vol_tracks_volatility(self):
        rng = np.random.default_rng(0)
        calm = series(100 * np.exp(np.cumsum(rng.normal(0, 0.005, 300))))
        wild = series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
        self.assertLess(daily_vol(calm).iloc[-1], daily_vol(wild).iloc[-1])


class UniquenessTests(unittest.TestCase):
    def idx(self, n=30):
        return pd.bdate_range("2024-01-01", periods=n)

    def test_overlap_lowers_uniqueness(self):
        ix = self.idx()
        disjoint = pd.Series([ix[4], ix[9]], index=[ix[0], ix[5]])
        np.testing.assert_allclose(average_uniqueness(disjoint, ix).values, [1.0, 1.0])
        same = pd.Series([ix[4], ix[4]], index=[ix[0], ix[0]])
        same = pd.Series([ix[4], ix[4]], index=pd.DatetimeIndex([ix[0], ix[0]]))
        np.testing.assert_allclose(average_uniqueness(same, ix).values, [0.5, 0.5])
        self.assertEqual(num_concurrent(same, ix).iloc[2], 2)

    def test_weights_average_one_and_favour_unique_events(self):
        ix = self.idx(40)
        t0 = [ix[0], ix[1], ix[2], ix[20]]
        t1 = pd.Series([ix[10], ix[11], ix[12], ix[25]], index=t0)
        w = sample_weights(t1, ix)
        self.assertAlmostEqual(w.mean(), 1.0)
        self.assertGreater(w.iloc[3], w.iloc[0])                                         # the isolated event counts more

    def test_sequential_bootstrap_is_more_independent_than_plain(self):
        ix = self.idx(120)
        t0 = ix[:60]
        t1 = pd.Series([ix[min(i + 15, 119)] for i in range(60)], index=t0)
        uniq = average_uniqueness(t1, ix).to_numpy()
        seq = [uniq[sequential_bootstrap(t1, ix, 60, seed=s)].mean() for s in range(5)]
        plain = [uniq[np.random.default_rng(s).integers(0, 60, 60)].mean() for s in range(5)]
        self.assertGreater(np.mean(seq), np.mean(plain))
        self.assertEqual(len(sequential_bootstrap(t1, ix, 10)), 10)


def events(n=100, span=5):
    ix = pd.bdate_range("2023-01-02", periods=n + span)
    t0 = ix[:n]
    return pd.Series([ix[i + span] for i in range(n)], index=t0)


class CrossValidationTests(unittest.TestCase):
    def test_purged_folds_have_no_label_overlap_with_test(self):
        t1 = events()
        cv = PurgedKFold(5, t1, embargo_pct=0.02)
        tested = []
        for tr, te in cv.split():
            tested.extend(te)
            lo, hi = t1.index[te[0]], t1.iloc[te].max()
            overlap = [(t1.index[i] <= hi) and (t1.iloc[i] >= lo) for i in tr]
            self.assertFalse(any(overlap))                                               # nothing in train touches the test period
            after = [i for i in tr if i > te[-1]]
            if after:
                self.assertGreater(after[0] - te[-1], cv.embargo)                         # embargo gap after the test fold
        self.assertEqual(sorted(tested), list(range(len(t1))))                            # every sample tested exactly once

    def test_plain_kfold_would_leak_and_purging_removes_it(self):
        t1 = events()
        cv = PurgedKFold(5, t1, embargo_pct=0.0)
        n_purged = sum(len(range(te[0], te[-1] + 1)) + len(tr) < len(t1) for tr, te in cv.split())
        self.assertGreater(n_purged, 0)                                                   # samples were removed to stop leakage

    def test_combinatorial_cv_counts_and_disjointness(self):
        t1 = events(120)
        cv = CombinatorialPurgedCV(6, 2, t1, embargo_pct=0.01)
        self.assertEqual((cv.n_combinations, cv.n_paths), (15, 5))
        splits = list(cv.split())
        self.assertEqual(len(splits), 15)
        for tr, te in splits:
            self.assertEqual(len(set(tr) & set(te)), 0)
            self.assertEqual(len(te), 40)
        with self.assertRaises(ValidationError):
            CombinatorialPurgedCV(4, 4, t1)

    def test_cv_scores_and_validation(self):
        t1 = events()
        y = np.arange(len(t1)) % 2
        s = cv_scores(lambda tr, te: np.full(len(te), 0.5), y.astype(float), PurgedKFold(4, t1), brier_score)
        self.assertTrue(np.allclose(s, 0.25))
        with self.assertRaises(ValidationError):
            PurgedKFold(1, t1)
        with self.assertRaises(ValidationError):
            PurgedKFold(3, t1.iloc[::-1])

    def test_pbo_flags_pure_noise_but_not_a_real_edge(self):
        rng = np.random.default_rng(0)
        T, N = 400, 20
        noise = pd.DataFrame(rng.normal(0, 0.01, (T, N)))
        pbo_noise = probability_of_backtest_overfitting(noise, 8)
        self.assertTrue(0.3 <= pbo_noise["pbo"] <= 0.7, pbo_noise["pbo"])                  # picking the best of noise is a coin flip
        real = noise.copy()
        real[0] += 0.004                                                                  # one genuinely better strategy
        pbo_real = probability_of_backtest_overfitting(real, 8)
        self.assertLess(pbo_real["pbo"], 0.2)
        self.assertGreater(pbo_real["mean_oos_sharpe_of_selected"], pbo_noise["mean_oos_sharpe_of_selected"])
        self.assertEqual(pbo_noise["n_splits"], 70)
        with self.assertRaises(ValidationError):
            probability_of_backtest_overfitting(noise, 7)


class FeatureTests(unittest.TestCase):
    def test_frac_diff_endpoints_and_memory(self):
        s = series(np.cumsum(np.random.default_rng(1).normal(size=500)) + 100)
        np.testing.assert_allclose(frac_diff(s, 0.0).dropna().values, s.values)
        d1 = frac_diff(s, 1.0).dropna()
        np.testing.assert_allclose(d1.values, s.diff().dropna().values, atol=1e-9)
        self.assertEqual(list(ffd_weights(1.0)), [-1.0, 1.0])
        out = min_stationary_d(np.log(series(100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0.0003, 0.01, 800))))))
        self.assertTrue(0 < out["d"] <= 1.0)
        diff_corr = np.corrcoef(np.log(series(100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0.0003, 0.01, 800))))).diff().dropna(),
                                np.log(series(100 * np.exp(np.cumsum(np.random.default_rng(2).normal(0.0003, 0.01, 800))))).iloc[1:])[0, 1]
        self.assertGreater(abs(out["corr_with_original"]), abs(diff_corr))                  # fractional diff keeps more memory
        with self.assertRaises(ValidationError):
            frac_diff(s, -0.1)

    def test_features_are_causal_and_complete(self):
        rng = np.random.default_rng(3)
        c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
        df = pd.DataFrame({"open": c * 0.999, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": rng.uniform(1e5, 2e5, 300)})
        f = build_features(df)
        self.assertEqual(f.shape[1], 12)
        self.assertFalse(f.iloc[100:].isna().any().any())
        alt = df.copy()
        alt.iloc[200:] *= 1.7
        f2 = build_features(alt)
        pd.testing.assert_frame_equal(f.iloc[:200], f2.iloc[:200])                          # the future cannot change the past
        with self.assertRaises(ValidationError):
            build_features(df.drop(columns=["volume"]))

    def test_scaler_uses_train_statistics_only(self):
        train, test = np.random.default_rng(0).normal(0, 1, (200, 3)), np.random.default_rng(1).normal(10, 1, (50, 3))
        sc = Scaler().fit(train)
        out = sc.transform(test)
        self.assertGreater(out.mean(), 4)                                                    # test shift is visible, not normalised away
        self.assertLessEqual(np.abs(out).max(), 5.0)                                         # winsorised
        with self.assertRaises(ValidationError):
            Scaler().transform(test)


class ModelTests(unittest.TestCase):
    def data(self, n=600, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 4))
        p = 1 / (1 + np.exp(-(2.0 * X[:, 0] - 1.0 * X[:, 1])))
        return X, (rng.random(n) < p).astype(float)

    def test_logistic_learns_calibrated_probabilities(self):
        X, y = self.data()
        Xt, yt = self.data(seed=1)
        m = LogisticModel(l2=1.0).fit(X, y)
        p = m.predict_proba(Xt)
        self.assertGreater(accuracy(yt, p), 0.7)
        self.assertLess(log_loss(yt, p), log_loss(yt, np.full(len(yt), yt.mean())))         # beats the base rate
        self.assertGreater(m.coef_[1], 1.0)
        self.assertLess(m.coef_[2], -0.4)
        cal = calibration_table(yt, p, 5)
        self.assertTrue((cal["mean_pred"] - cal["observed"]).abs().mean() < 0.12)

    def test_sample_weights_move_the_decision(self):
        X, y = self.data(400)
        w_pos = np.where(y == 1, 5.0, 1.0)
        base = LogisticModel().fit(X, y).predict_proba(X).mean()
        tilted = LogisticModel().fit(X, y, w_pos).predict_proba(X).mean()
        self.assertGreater(tilted, base)
        with self.assertRaises(ValidationError):
            LogisticModel().fit(X, y + 2)
        with self.assertRaises(ValidationError):
            LogisticModel().fit(np.where(X > 3, np.nan, X), y)
        with self.assertRaises(ValidationError):
            LogisticModel().predict_proba(X)

    def test_ridge_recovers_coefficients_and_adapter_wraps_estimators(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 2))
        y = 1.0 + 2.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 0.1, 300)
        r = RidgeModel(0.1).fit(X, y)
        np.testing.assert_allclose(r.coef_, [1.0, 2.0, -0.5], atol=0.05)

        class Dummy:                                                                          # sklearn-style, no sample_weight support
            def fit(self, X, y):
                self.p = float(np.mean(y))
                return self

            def predict_proba(self, X):
                return np.column_stack([1 - np.full(len(X), self.p), np.full(len(X), self.p)])
        a = SklearnAdapter(Dummy()).fit(X, (y > 1).astype(float), sample_weight=np.ones(300))
        self.assertEqual(a.predict_proba(X).shape, (300,))

    def test_permutation_importance_finds_the_real_features(self):
        X, y = self.data(800)
        Xt, yt = self.data(800, seed=3)
        m = LogisticModel().fit(X, y)
        imp = permutation_importance(m.predict_proba, Xt, yt, names=list("abcd"))
        self.assertEqual(set(imp.index[:2]), {"a", "b"})
        self.assertGreater(imp["a"], 10 * max(imp["c"], imp["d"], 1e-6))

    def test_out_of_range_share(self):
        rng = np.random.default_rng(0)
        train = rng.normal(0, 1, (1000, 3))
        self.assertLess(out_of_range_share(train, rng.normal(0, 1, (200, 3))).max(), 0.10)
        self.assertGreater(out_of_range_share(train, rng.normal(0, 4, (200, 3))).min(), 0.5)

    def test_psi_detects_shift(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 2000)
        self.assertLess(psi(a, rng.normal(0, 1, 2000)), 0.1)
        self.assertGreater(psi(a, rng.normal(1.5, 1, 2000)), 0.25)
        self.assertTrue(math.isnan(psi([1, 2], [1, 2])))


def panel(n=300, assets=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    rets = pd.DataFrame(rng.normal(0.0004, 0.01, (n, assets)), index=idx, columns=[f"A{i}" for i in range(assets)])
    return rets, 100 * (1 + rets).cumprod()


class SignalTests(unittest.TestCase):
    def test_forward_returns_alignment(self):
        rets, close = panel()
        f = forward_returns(close, 3)
        self.assertTrue(f.iloc[-3:].isna().all().all())
        self.assertAlmostEqual(f.iloc[0, 0], close.iloc[3, 0] / close.iloc[0, 0] - 1)
        with self.assertRaises(ValidationError):
            forward_returns(close, 0)

    def test_ic_of_a_real_signal_vs_noise(self):
        rets, close = panel(500, 30)
        fwd1 = forward_returns(close, 1)
        rng = np.random.default_rng(1)
        good = fwd1 + pd.DataFrame(rng.normal(0, 0.02, fwd1.shape), index=fwd1.index, columns=fwd1.columns)    # informative + noise
        noise = pd.DataFrame(rng.normal(size=fwd1.shape), index=fwd1.index, columns=fwd1.columns)
        s_good, s_noise = ic_summary(information_coefficient(good, fwd1)), ic_summary(information_coefficient(noise, fwd1))
        self.assertGreater(s_good["ic_mean"], 0.2)
        self.assertLess(s_good["p_value"], 1e-6)
        self.assertAlmostEqual(s_noise["ic_mean"], 0.0, delta=0.05)
        self.assertGreater(s_noise["p_value"], 0.001)
        self.assertGreater(s_good["hit_rate"], 0.8)
        with self.assertRaises(ValidationError):
            information_coefficient(good, fwd1, "kendall")

    def test_ic_decay_and_quantile_monotonicity(self):
        rets, close = panel(500, 30)
        sig = forward_returns(close, 1).fillna(0) + pd.DataFrame(np.random.default_rng(2).normal(0, 0.01, close.shape),
                                                              index=close.index, columns=close.columns)
        dec = ic_decay(sig, close, (1, 5, 20))
        self.assertGreater(dec[1], dec[20])                                                  # predicts tomorrow, not next month
        q = quantile_returns(sig, forward_returns(close, 1), 5)
        self.assertGreater(q["monotonicity"], 0.8)
        self.assertGreater(q["spread"], 0)

    def test_turnover_and_signal_correlation(self):
        rets, close = panel(300, 20)
        slow = close.pct_change(60)
        fast = pd.DataFrame(np.random.default_rng(3).normal(size=close.shape), index=close.index, columns=close.columns)
        self.assertGreater(signal_autocorrelation(slow.dropna()), 0.8)
        self.assertGreater(signal_turnover(fast), 0.8)
        corr = signal_correlation({"slow": slow.dropna(), "fast": fast, "slow2": slow.dropna() * 2 + 1})
        self.assertAlmostEqual(corr.loc["slow", "slow2"], 1.0, places=6)
        self.assertLess(abs(corr.loc["slow", "fast"]), 0.2)


def ar_closes(n=700, phi=0.15, seed=0):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for t in range(1, n):
        r[t] = phi * r[t - 1] + rng.normal(0.0003, 0.01)
    return 100 * np.exp(np.cumsum(r))


class MLStrategyTests(unittest.TestCase):
    def test_registered_and_validated(self):
        self.assertIn("ml_logistic", STRATEGIES.names())
        with self.assertRaises(StrategyError):
            MLStrategy(["X"], train_window=50)
        with self.assertRaises(StrategyError):
            MLStrategy(["X"], edge=0.9)

    def go(self, closes, **params):
        md = market_from_closes(X=closes)
        s = MLStrategy(["X"], **params)
        return BacktestEngine(md, free_config()).run(s), s, md

    def test_trades_after_warmup_only_and_is_deterministic(self):
        res, strat, md = self.go(ar_closes())
        res2, _, _ = self.go(ar_closes())
        pd.testing.assert_series_equal(res.equity, res2.equity)
        first_trade = min(f.ts for f in res.fills) if res.fills else None
        self.assertIsNotNone(first_trade)
        self.assertGreaterEqual(md.frame("X").index.get_loc(first_trade), strat.warmup)      # nothing before the model could exist
        self.assertGreater(strat.trainings, 3)                                               # retrained on a rolling basis
        self.assertTrue(np.isfinite(res.metrics()["sharpe"]))

    def test_no_look_ahead_future_changes_do_not_change_the_past(self):
        base = ar_closes(600, seed=4)
        alt = base.copy()
        alt[500:] *= 2.5
        r1, _, md = self.go(base)
        r2, _, _ = self.go(alt)
        cut = md.frame("X").index[498]
        pd.testing.assert_series_equal(r1.equity.loc[:cut], r2.equity.loc[:cut])

    def test_drift_guard_fires_on_a_regime_break_far_more_than_in_calm_data(self):
        calm = ar_closes(seed=0)
        r = np.diff(np.log(calm), prepend=np.log(calm[0]))
        r[500:] *= 5                                                                       # volatility jumps fivefold
        broken = 100 * np.exp(np.cumsum(r))
        _, s_calm, _ = self.go(calm)
        _, s_break, _ = self.go(broken)
        self.assertLess(s_calm.drift_events, 0.1 * (len(calm) - s_calm.warmup))             # rarely cries wolf
        self.assertGreater(s_break.drift_events, 2 * s_calm.drift_events + 5)

    def test_drift_parameters_are_validated(self):
        with self.assertRaises(StrategyError):
            MLStrategy(["X"], drift_limit=1.5)

    def test_short_side_optional(self):
        res, strat, _ = self.go(ar_closes(seed=2), allow_short=True, edge=0.02)
        self.assertGreaterEqual(res.trade_stats().get("trades", 0), 1)

    def test_builtin_name_is_creatable(self):
        s = STRATEGIES.create("ml_logistic", ["X"], horizon=3)
        self.assertEqual(s.params["horizon"], 3)


if __name__ == "__main__":
    unittest.main()
