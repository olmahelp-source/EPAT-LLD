import math
import unittest

import numpy as np

import helpers  # noqa: F401
from algoplatform.ai import (FuzzySystem, FuzzyVariable, GaussianHMM, IsolationForest, RegimeDetector, SymbolicAlpha, bocpd,
                             changepoints_from_run_length, cusum, evaluate_tree, gaussian, left_shoulder, mahalanobis, right_shoulder,
                             robust_zscore, rolling_robust_zscore, to_string, trapezoid, triangle)
from algoplatform.core.exceptions import ModelError, ValidationError
from algoplatform.dl import (LSTM, MLP, Activation, Adam, Autoencoder, Dense, Dropout, LSTMForecaster, Sequential, gradient_check,
                             make_sequences)
from algoplatform.ml import (KNN, PCA, DecisionTree, GaussianNB, GradientBoosting, KMeans, LassoModel, RandomForest, StackingEnsemble,
                             VotingEnsemble, confusion, directional_accuracy, forward_returns, information_coefficient,
                             precision_recall_f1, r2_score, roc_auc, signal_from_probability, walk_forward_predict)
from algoplatform.research import LogisticModel
from algoplatform.rl import (DQNAgent, EpsilonGreedyBandit, QLearningAgent, ReinforceAgent, ThompsonBandit, TradingEnv, UCB1Bandit,
                             as_policy, evaluate, standardize, train)


def classification_data(n=1500, seed=0):
    """Label depends on an interaction (x0 high AND x1 low) plus a linear term: trees can find it, a line cannot fully."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    logit = 1.5 * (X[:, 0] > 0.3) * (X[:, 1] < 0) - 0.8 * X[:, 2]
    y = (logit + rng.logistic(size=n) * 0.5 > 0.2).astype(float)
    return X, y


class TreeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = classification_data()
        cls.tr, cls.te = slice(0, 1000), slice(1000, 1500)

    def _auc(self, model):
        return roc_auc(self.y[self.te], model.fit(self.X[self.tr], self.y[self.tr]).predict_proba(self.X[self.te]))

    def test_models_learn_and_trees_beat_the_linear_baseline(self):
        base = self._auc(LogisticModel())
        scores = {"tree": self._auc(DecisionTree(max_depth=4)), "forest": self._auc(RandomForest(n_trees=40, max_depth=6)),
                  "boosting": self._auc(GradientBoosting(n_estimators=120))}
        for name, auc in scores.items():
            self.assertGreater(auc, 0.80, name)
        self.assertGreater(scores["forest"], base + 0.02)                             # the interaction is worth something
        self.assertGreater(self._auc(KNN(25)), 0.75)
        self.assertGreater(self._auc(GaussianNB()), 0.75)

    def test_tree_structure_and_importance(self):
        t = DecisionTree(max_depth=3, min_samples_leaf=25).fit(self.X[self.tr], self.y[self.tr])
        self.assertLessEqual(t.n_leaves, 8)
        leaves = t.apply(self.X[self.tr])
        self.assertGreaterEqual(np.bincount(leaves)[np.unique(leaves)].min(), 25)     # the leaf-size floor is respected
        self.assertAlmostEqual(t.feature_importances_.sum(), 1.0)
        self.assertEqual(set(np.argsort(t.feature_importances_)[-3:]), {0, 1, 2})     # the three features that matter
        self.assertTrue(set(np.unique(t.predict(self.X[self.te]))) <= {0.0, 1.0})
        stump = DecisionTree(max_depth=1).fit(np.array([[0.0], [1.0], [2.0], [3.0]] * 10), np.array([0.0, 0.0, 1.0, 1.0] * 10))
        self.assertAlmostEqual(stump.threshold_[0], 1.5)                              # splits exactly between the classes

    def test_sample_weights_change_the_fit(self):
        X = np.array([[0.0], [1.0]] * 50)
        y = np.array([0.0, 1.0] * 50)
        w = np.where(X[:, 0] == 1, 9.0, 1.0)
        t = DecisionTree(max_depth=1, min_samples_leaf=200).fit(X, y, w)              # cannot split: predicts the weighted mean
        self.assertAlmostEqual(t.predict_proba(X[:1])[0], 0.9)

    def test_regression_trees(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(1200, 4))
        y = np.where(X[:, 0] > 0, 2.0, -1.0) + 0.5 * X[:, 1] + rng.normal(0, 0.2, 1200)
        gb = GradientBoosting("regression", 150, 0.1).fit(X[:900], y[:900])
        self.assertGreater(r2_score(y[900:], gb.predict(X[900:])), 0.9)
        rf = RandomForest("regression", 40, max_depth=6, max_features=None).fit(X[:900], y[:900])
        self.assertGreater(rf.oob_score_, 0.8)
        self.assertGreater(r2_score(y[900:], rf.predict(X[900:])), 0.85)
        with self.assertRaises(ModelError):
            gb.predict_proba(X)

    def test_boosting_early_stopping_uses_the_latest_data(self):
        gb = GradientBoosting(n_estimators=300, learning_rate=0.2, validation_fraction=0.25, patience=5).fit(self.X[self.tr], self.y[self.tr])
        self.assertLess(gb.best_iteration_, 300)                                      # stopped before overfitting
        self.assertEqual(len(gb.trees_), gb.best_iteration_)
        self.assertAlmostEqual(min(gb.validation_loss_), gb.validation_loss_[gb.best_iteration_ - 1])

    def test_determinism_and_validation(self):
        a = RandomForest(n_trees=10, seed=7).fit(self.X[self.tr], self.y[self.tr]).predict_proba(self.X[self.te])
        b = RandomForest(n_trees=10, seed=7).fit(self.X[self.tr], self.y[self.tr]).predict_proba(self.X[self.te])
        np.testing.assert_array_equal(a, b)
        t = DecisionTree()
        with self.assertRaises(ModelError):
            t.predict(self.X)
        for X, y in ((self.X[:10], self.y[:9]), (np.full((10, 2), np.nan), self.y[:10]), (self.X[:10], np.arange(10.0))):
            with self.assertRaises(ValidationError):
                DecisionTree().fit(X, y)
        with self.assertRaises(ValidationError):
            DecisionTree().fit(self.X[:50], self.y[:50]).predict(self.X[:5, :3])       # wrong number of features
        for bad in (lambda: DecisionTree(task="ranking"), lambda: RandomForest(n_trees=0), lambda: GradientBoosting(learning_rate=0)):
            with self.assertRaises(ValidationError):
                bad()


class ClassicModelTests(unittest.TestCase):
    def test_lasso_selects_the_true_features(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(800, 8)) * np.array([1, 10, 0.1, 1, 1, 1, 1, 1])        # mixed scales
        y = 3.0 + 2.0 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(0, 0.3, 800)
        m = LassoModel(alpha=0.1).fit(X, y)
        self.assertEqual(list(m.selected_), [0, 1])
        self.assertAlmostEqual(m.coef_[0], 3.0, delta=0.1)
        self.assertGreater(r2_score(y, m.predict(X)), 0.97)
        self.assertEqual(len(LassoModel(alpha=100.0).fit(X, y).selected_), 0)         # a huge penalty removes everything
        np.testing.assert_allclose(LassoModel(alpha=0.0).fit(X, y).coef_[1], 2.0, atol=0.05)
        with self.assertRaises(ModelError):
            LassoModel().predict(X)

    def test_knn_and_naive_bayes(self):
        rng = np.random.default_rng(3)
        X = np.vstack([rng.normal(-2, 0.5, (100, 2)), rng.normal(2, 0.5, (100, 2))])
        y = np.repeat([0.0, 1.0], 100)
        for model in (KNN(5), KNN(5, weights="distance"), GaussianNB()):
            p = model.fit(X, y).predict_proba(np.array([[-2.0, -2.0], [2.0, 2.0]]))
            self.assertLess(p[0], 0.05, type(model).__name__)
            self.assertGreater(p[1], 0.95, type(model).__name__)
        reg = KNN(3, task="regression").fit(np.arange(10.0).reshape(-1, 1), np.arange(10.0) * 2)
        self.assertAlmostEqual(reg.predict(np.array([[5.0]]))[0], 10.0)
        with self.assertRaises(ValidationError):
            KNN(50).fit(X[:10], y[:10])
        with self.assertRaises(ValidationError):
            GaussianNB().fit(X[:50], y[:50])                                          # a single class

    def test_pca_and_kmeans(self):
        rng = np.random.default_rng(4)
        f = rng.normal(size=500)
        X = np.column_stack([f, 2 * f + rng.normal(0, 0.05, 500), rng.normal(size=500)])
        pca = PCA(2).fit(X)
        self.assertGreater(pca.explained_variance_ratio_[0], 0.6)
        full = PCA(3).fit(X)
        np.testing.assert_allclose(full.inverse_transform(full.transform(X)), X, atol=1e-9)
        np.testing.assert_allclose(np.corrcoef(pca.transform(X), rowvar=False)[0, 1], 0.0, atol=1e-8)   # components are uncorrelated
        with self.assertRaises(ModelError):
            PCA(1).transform(X)
        blobs = np.vstack([rng.normal(c, 0.3, (80, 2)) for c in ((0, 0), (4, 0), (0, 4))])
        km = KMeans(3, seed=1).fit(blobs)
        self.assertEqual(sorted(np.bincount(km.labels_)), [80, 80, 80])
        self.assertEqual(len(set(km.predict(np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])))), 3)
        with self.assertRaises(ValidationError):
            KMeans(5).fit(blobs[:3])


class EnsembleAndPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = classification_data()

    def test_voting_survives_a_failing_member(self):
        class Broken:
            def fit(self, X, y, sample_weight=None):
                raise RuntimeError("cannot fit")
        v = VotingEnsemble([LogisticModel(), GaussianNB(), Broken()]).fit(self.X[:1000], self.y[:1000])
        self.assertEqual(len(v.fitted_), 2)
        self.assertIn("Broken", v.failed_[0])
        self.assertGreater(roc_auc(self.y[1000:], v.predict_proba(self.X[1000:])), 0.78)
        with self.assertRaises(ModelError):
            VotingEnsemble([Broken(), Broken()]).fit(self.X[:100], self.y[:100])
        with self.assertRaises(ValidationError):
            VotingEnsemble([LogisticModel()], weights=[0.0])

    def test_stacking(self):
        s = StackingEnsemble([GaussianNB(), DecisionTree(max_depth=4), KNN(25)], LogisticModel()).fit(self.X[:1000], self.y[:1000])
        self.assertGreater(roc_auc(self.y[1000:], s.predict_proba(self.X[1000:])), 0.80)
        with self.assertRaises(ModelError):
            StackingEnsemble([GaussianNB()], LogisticModel()).predict(self.X)

    def test_walk_forward_never_trains_on_what_it_predicts(self):
        seen = []

        class Spy:
            def fit(self, X, y, sample_weight=None):
                seen.append(len(y))
                self.mean = float(y.mean())
                return self

            def predict_proba(self, X):
                return np.full(len(X), self.mean)
        out = walk_forward_predict(Spy, self.X[:600], self.y[:600], train_size=200, step=100, embargo=10, scale=False)
        self.assertTrue(np.isnan(out["pred"][:210]).all())                            # nothing before train_size + embargo
        self.assertFalse(np.isnan(out["pred"][210:]).any())
        self.assertEqual(seen, [200, 300, 400, 500])                                  # expanding window, each ending 10 rows before its block
        self.assertAlmostEqual(out["pred"][210], self.y[:200].mean())
        rolling = walk_forward_predict(Spy, self.X[:600], self.y[:600], train_size=200, step=100, expanding=False, scale=False)
        self.assertEqual(rolling["fits"], 4)
        self.assertEqual(seen[-4:], [200] * 4)

    def test_walk_forward_reports_failed_fits_and_real_skill(self):
        y = self.y.copy()
        y[:320] = 1.0                                                                 # the first training windows contain a single class
        out = walk_forward_predict(lambda: GaussianNB(), self.X[:900], y[:900], train_size=300, step=100)
        self.assertGreater(out["failed_fits"], 0)
        self.assertTrue(np.isnan(out["pred"][300:400]).all())                         # a failed block stays NaN, no stale model reused
        good = walk_forward_predict(lambda: GradientBoosting(n_estimators=40), self.X, self.y, 500, 250, embargo=5)
        ok = ~np.isnan(good["pred"])
        self.assertGreater(roc_auc(self.y[ok], good["pred"][ok]), 0.8)
        with self.assertRaises(ValidationError):
            walk_forward_predict(GaussianNB, self.X[:100], self.y[:100], train_size=200)

    def test_labels_metrics_and_signal_mapping(self):
        close = np.array([100.0, 110.0, 99.0, 108.9])
        np.testing.assert_allclose(forward_returns(close, 1), [0.1, -0.1, 0.1, np.nan], equal_nan=True)
        self.assertTrue(np.isnan(forward_returns(close, 2)[-2:]).all())               # the future of the last bars is unknown
        np.testing.assert_allclose(signal_from_probability([0.9, 0.52, 0.1, np.nan]), [0.8, 0.0, -0.8, 0.0])
        np.testing.assert_allclose(signal_from_probability([0.1], allow_short=False), [0.0])
        y, p = np.array([1, 1, 0, 0, 1.0]), np.array([0.9, 0.4, 0.6, 0.1, 0.8])
        self.assertEqual(confusion(y, p), {"tp": 2, "fp": 1, "tn": 1, "fn": 1})
        prf = precision_recall_f1(y, p)
        self.assertAlmostEqual(prf["precision"], 2 / 3)
        self.assertAlmostEqual(roc_auc(y, p), 5 / 6)
        self.assertAlmostEqual(roc_auc(y, -p), 1 / 6)
        self.assertTrue(math.isnan(roc_auc(np.ones(4), p[:4])))
        self.assertAlmostEqual(information_coefficient([1, 2, 3, 4], [10, 20, 25, 90]), 1.0)
        self.assertAlmostEqual(directional_accuracy([1, -1, 2, 0], [0.5, 0.2, 1.0, 3.0]), 2 / 3)


class NeuralNetworkTests(unittest.TestCase):
    def test_backward_passes_match_numerical_gradients(self):
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(6, 5)), rng.normal(size=(6, 1))
        for loss, target in (("mse", y), ("huber", y * 3), ("bce", (y > 0).astype(float))):
            net = Sequential([Dense(5, 7, rng), Activation("tanh"), Dense(7, 4, rng), Activation("sigmoid"), Dense(4, 3, rng),
                              Activation("leaky_relu"), Dense(3, 1, rng)], loss)
            self.assertLess(gradient_check(net, X, target, n_checks=40), 1e-5, loss)
        seq, ys = rng.normal(size=(4, 6, 3)), rng.normal(size=(4, 1))
        lstm = Sequential([LSTM(3, 5, rng), Dense(5, 1, rng)], "mse")
        self.assertLess(gradient_check(lstm, seq, ys, n_checks=60), 1e-5)
        stacked = Sequential([LSTM(3, 4, rng, return_sequences=True), LSTM(4, 3, rng), Dense(3, 1, rng)], "mse")
        self.assertLess(gradient_check(stacked, seq, ys, n_checks=60), 1e-5)          # back-propagation through time, two layers deep

    def test_shapes_dropout_and_weights(self):
        rng = np.random.default_rng(1)
        self.assertEqual(LSTM(3, 4, rng, return_sequences=True).forward(np.zeros((2, 5, 3))).shape, (2, 5, 4))
        self.assertEqual(Dense(3, 2, rng).forward(np.zeros((2, 5, 3))).shape, (2, 5, 2))         # Dense works on the last axis
        with self.assertRaises(ValidationError):
            LSTM(3, 4, rng).forward(np.zeros((2, 3)))
        drop = Dropout(0.5, rng)
        x = np.ones((2000, 10))
        np.testing.assert_array_equal(drop.forward(x), x)                             # inactive when predicting
        out = drop.forward(x, training=True)
        self.assertAlmostEqual(out.mean(), 1.0, delta=0.05)                           # inverted dropout keeps the expected value
        self.assertAlmostEqual((out == 0).mean(), 0.5, delta=0.05)
        net = Sequential([Dense(3, 4, rng), Activation("relu"), Dense(4, 1, rng)])
        self.assertEqual(net.n_params, 3 * 4 + 4 + 4 + 1)
        w = net.get_weights()
        net.params[0][...] = 0.0
        net.set_weights(w)
        np.testing.assert_array_equal(net.params[0], w[0])
        with self.assertRaises(ModelError):
            net.set_weights(w[:-1])
        for bad in (lambda: Sequential([]), lambda: Sequential([Dense(1, 1)], "hinge"), lambda: Activation("swish"), lambda: Dropout(1.0),
                    lambda: Adam(lr=0), lambda: Dense(0, 1)):
            with self.assertRaises(ValidationError):
                bad()

    def test_training_stops_early_restores_best_and_reports_divergence(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(300, 3))
        y = X @ np.array([1.0, -2.0, 0.5])
        net = Sequential([Dense(3, 8, rng), Activation("tanh"), Dense(8, 1, rng)], "mse", Adam(0.01), seed=0)
        net.fit(X, y, epochs=400, batch_size=32, patience=10)
        self.assertLess(len(net.history["loss"]), 400)                                # early stopping fired
        self.assertAlmostEqual(net.loss(X[240:], y[240:]), net.best_val_loss_, places=10)   # the best weights are the ones kept
        self.assertLess(net.best_val_loss_, 0.05)
        wild = Sequential([Dense(3, 1, rng)], "mse", Adam(1e200, clip_norm=None), seed=0)       # an absurd step size
        with np.errstate(over="ignore", invalid="ignore"), self.assertRaises(ModelError):
            wild.fit(X, y, epochs=50, validation_fraction=0.0)
        self.assertTrue(np.isfinite(wild.params[0]).all())                            # left with its last good weights, not NaN
        with self.assertRaises(ValidationError):
            net.fit(np.full((10, 3), np.nan), np.zeros(10))

    def test_same_seed_same_model(self):
        X, y = classification_data(400, seed=5)
        a = MLP(hidden=(8,), epochs=20, seed=3).fit(X, y).predict_proba(X[:20])
        b = MLP(hidden=(8,), epochs=20, seed=3).fit(X, y).predict_proba(X[:20])
        np.testing.assert_array_equal(a, b)


class DeepModelTests(unittest.TestCase):
    def test_mlp_learns_a_non_linear_boundary(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1500, 4))
        y = (X[:, 0] * X[:, 1] > 0).astype(float)                                     # XOR-like: no linear model can do it
        self.assertLess(roc_auc(y[1000:], LogisticModel().fit(X[:1000], y[:1000]).predict_proba(X[1000:])), 0.6)
        m = MLP(hidden=(16, 8), epochs=150).fit(X[:1000], y[:1000])
        self.assertGreater(roc_auc(y[1000:], m.predict_proba(X[1000:])), 0.95)
        self.assertTrue(set(np.unique(m.predict(X[1000:]))) <= {0.0, 1.0})

    def test_mlp_regression_and_errors(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(1200, 3))
        y = 50 + 10 * np.sin(2 * X[:, 0]) + 5 * X[:, 1] ** 2 + rng.normal(0, 0.5, 1200)   # target far from zero: scaling is internal
        m = MLP("regression", hidden=(32, 16), epochs=200, dropout=0.0).fit(X[:900], y[:900])
        self.assertGreater(r2_score(y[900:], m.predict(X[900:])), 0.85)
        with self.assertRaises(ModelError):
            m.predict_proba(X)
        with self.assertRaises(ModelError):
            MLP().predict(X)
        with self.assertRaises(ValidationError):
            MLP().fit(X, y)                                                           # classification needs 0/1 labels

    def test_sequences_end_at_the_row_they_predict(self):
        X = np.arange(20.0).reshape(10, 2)
        X[4, 0] = np.nan
        seq, y, last = make_sequences(X, np.arange(10.0), lookback=3)
        self.assertEqual(list(last), [2, 3, 7, 8, 9])                                 # windows touching the NaN row are dropped
        np.testing.assert_array_equal(seq[0], X[0:3])
        np.testing.assert_array_equal(y, last.astype(float))                          # label of the window's last row: nothing later
        self.assertEqual(make_sequences(X[:2], None, 3)[0].shape, (0, 3, 2))

    def test_lstm_uses_memory_and_predicts_causally(self):
        rng = np.random.default_rng(2)
        F = rng.normal(size=(900, 2))
        target = np.convolve(F[:, 0], np.ones(5) / 5, mode="full")[:900]              # mean of the last five values: needs memory
        model = LSTMForecaster(lookback=8, hidden=12, epochs=60).fit(F[:650], target[:650])
        pred = model.predict(F)
        self.assertTrue(np.isnan(pred[:7]).all())
        self.assertGreater(r2_score(target[650:], pred[650:]), 0.85)
        changed = F.copy()
        changed[800:] += 10.0
        np.testing.assert_allclose(model.predict(changed)[:800], pred[:800])          # predictions never depend on later rows
        with self.assertRaises(ValidationError):
            LSTMForecaster(lookback=8).fit(F[:20], target[:20])

    def test_autoencoder_flags_what_it_has_not_seen(self):
        rng = np.random.default_rng(3)
        f = rng.normal(size=(1200, 2))
        X = np.column_stack([f[:, 0], 2 * f[:, 0] + 0.01 * rng.normal(size=1200), f[:, 1], -f[:, 1]])   # 4 columns, 2 real dimensions
        ae = Autoencoder(2, 8, epochs=120).fit(X)
        normal = ae.reconstruction_error(X[:200]).mean()
        broken = ae.reconstruction_error(np.array([[2.0, -4.0, 1.0, 1.0]]))[0]        # violates both relationships
        self.assertGreater(broken, 20 * normal)
        self.assertTrue(ae.is_anomaly(np.array([[2.0, -4.0, 1.0, 1.0]]))[0])
        self.assertLess(ae.is_anomaly(X).mean(), 0.05)
        self.assertEqual(ae.encode(X[:5]).shape, (5, 2))


def planted_market(n=3000, seed=1):
    """Next return = sign(signal) * drift + noise, so a learnable edge exists; the second feature is pure noise."""
    rng = np.random.default_rng(seed)
    sig = rng.normal(size=n)
    ret = np.zeros(n)
    ret[1:] = 0.004 * np.sign(sig[:-1]) + 0.004 * rng.normal(size=n - 1)
    return ret, np.column_stack([sig, rng.normal(size=n)])


class TradingEnvTests(unittest.TestCase):
    def test_reward_timing_and_costs(self):
        ret = np.array([0.0, 0.01, -0.02, 0.03])
        env = TradingEnv(ret, np.arange(8.0).reshape(4, 2), cost_bps=10, reward_scale=1.0)
        obs = env.reset()
        np.testing.assert_array_equal(obs, [0.0, 1.0, 0.0])                           # features at t plus the current position
        obs, reward, done, info = env.step(2)                                         # go long: paid the NEXT bar's return, minus cost
        self.assertAlmostEqual(reward, 0.01 - 0.001)
        np.testing.assert_array_equal(obs, [2.0, 3.0, 1.0])
        _, reward, _, _ = env.step(2)                                                 # stay long: no cost
        self.assertAlmostEqual(reward, -0.02)
        _, reward, done, info = env.step(0)                                           # flip to short: cost on two units
        self.assertAlmostEqual(reward, -0.03 - 0.002)
        self.assertTrue(done)
        with self.assertRaises(ValidationError):
            env.step(1)                                                               # the episode is over
        risk = TradingEnv(ret, np.zeros((4, 1)), cost_bps=0, reward_scale=1.0, risk_aversion=10.0)
        risk.reset()
        self.assertAlmostEqual(risk.step(2)[1], 0.01 - 10 * 0.01 ** 2)

    def test_construction_and_validation(self):
        close = np.array([100.0, 101.0, 102.0, 101.0, 103.0])
        feats = np.array([[np.nan], [np.nan], [1.0], [2.0], [3.0]])                   # indicator warm-up rows
        env = TradingEnv.from_prices(close, feats)
        self.assertEqual(len(env.returns), 3)
        self.assertAlmostEqual(env.returns[0], 102 / 101 - 1)
        self.assertEqual((env.n_actions, env.obs_dim), (3, 2))
        for kwargs in ({"returns": [0.1, np.nan, 0.1], "features": np.zeros((3, 1))}, {"returns": [0.1] * 3, "features": np.zeros((4, 1))},
                       {"returns": [0.1] * 3, "features": np.zeros((3, 1)), "actions": (2.0, 0.0)},
                       {"returns": [0.1] * 3, "features": np.zeros((3, 1)), "cost_bps": -1}):
            with self.assertRaises(ValidationError):
                TradingEnv(**kwargs)
        with self.assertRaises(ValidationError):
            env.step(7) if env.reset() is not None else None
        with self.assertRaises(ValidationError):
            TradingEnv.from_prices(close, np.array([[1.0], [np.nan], [1.0], [1.0], [np.nan]]))
        np.testing.assert_allclose(standardize(np.array([[1.0], [3.0]]), np.array([[2.0], [4.0]])), [[0.0], [2.0]])

    def test_random_episode_windows(self):
        ret, feat = planted_market(500)
        env = TradingEnv(ret, feat, episode_length=50, seed=0)
        starts = set()
        for _ in range(5):
            env.reset()
            starts.add(env._t)
            steps = 0
            done = False
            while not done:
                _, _, done, _ = env.step(1)
                steps += 1
            self.assertEqual(steps, 50)
        self.assertGreater(len(starts), 1)


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ret, cls.feat = planted_market()
        cls.train_env = lambda self=None: TradingEnv(cls.ret[:2000], cls.feat[:2000], cost_bps=1, episode_length=250, seed=0)
        cls.test_env = lambda self=None: TradingEnv(cls.ret[2000:], cls.feat[2000:], cost_bps=1)

    def test_agents_learn_the_planted_edge_out_of_sample(self):
        tr = self.train_env()
        obs = np.column_stack([self.feat[:2000], np.random.default_rng(0).choice([-1, 0, 1], 2000)])
        agents = {"q": (QLearningAgent(3, bins=4).prepare(obs), 40), "dqn": (DQNAgent(tr.obs_dim, 3, hidden=(16,), epsilon_decay=0.9), 15),
                  "reinforce": (ReinforceAgent(tr.obs_dim, 3, hidden=(16,)), 60)}
        for name, (agent, episodes) in agents.items():
            history = train(agent, self.train_env(), episodes)
            self.assertGreater(np.mean(history[-5:]), np.mean(history[:5]), name)     # reward improved with training
            result = evaluate(agent, self.test_env())
            self.assertGreater(result["sharpe"], 3.0, name)                           # the edge is large by construction
            self.assertEqual(result["steps"], 999)
            self.assertEqual(len(result["equity"]), 999)

    def test_no_skill_is_found_where_there_is_none(self):
        rng = np.random.default_rng(7)
        noise = 0.01 * rng.normal(size=3000)
        env = TradingEnv(noise[:2000], self.feat[:2000], episode_length=250, seed=0)
        agent = DQNAgent(env.obs_dim, 3, hidden=(16,), epsilon_decay=0.9)
        train(agent, env, 12)
        self.assertLess(abs(evaluate(agent, TradingEnv(noise[2000:], self.feat[2000:]))["sharpe"]), 2.5)

    def test_agent_mechanics(self):
        q = QLearningAgent(3, bins=3, alpha=0.5, gamma=0.0, epsilon=0.0, epsilon_min=0.0).prepare(np.array([[0.0], [1.0], [2.0]]))
        q.observe([0.0], 2, 1.0, [1.0], False)
        self.assertEqual(q.act([0.0], greedy=True), 2)
        self.assertAlmostEqual(q.q_values([0.0])[2], 0.5)                             # alpha * (reward - 0)
        with self.assertRaises(ModelError):
            QLearningAgent(3).act([0.0])                                              # prepare() first
        e = QLearningAgent(3, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5)
        e.end_episode()
        e.end_episode()
        e.end_episode()
        e.end_episode()
        self.assertAlmostEqual(e.epsilon, 0.1)                                        # decays to the floor, not below
        d = DQNAgent(2, 3, warmup=8, batch_size=8, target_sync=2, seed=0)
        before = d.target.get_weights()[0].copy()
        for i in range(12):
            d.observe([0.1 * i, 0.0], i % 3, 1.0, [0.1 * i, 1.0], False)
        self.assertEqual(d.steps, 5)
        self.assertFalse(np.array_equal(d.target.get_weights()[0], before))           # the target network was synchronised
        r = ReinforceAgent(2, 3, seed=0)
        self.assertAlmostEqual(r.probabilities([0.0, 0.0]).sum(), 1.0)
        for bad in (lambda: QLearningAgent(1), lambda: DQNAgent(2, 3, warmup=1, batch_size=8), lambda: ReinforceAgent(2, 3, gamma=2)):
            with self.assertRaises(ValidationError):
                bad()
        with self.assertRaises(ValidationError):
            train(q, TradingEnv([0.0, 0.1, 0.1], np.zeros((3, 1))), 0)

    def test_policy_adapter_remembers_its_position(self):
        class Fixed:
            def __init__(self):
                self.seen = []

            def act(self, obs, greedy=False):
                self.seen.append(obs.copy())
                return 2
        agent = Fixed()
        policy = as_policy(agent, lambda w: np.array([w["close"][-1]]))
        self.assertEqual(policy({"close": np.array([1.0, 5.0])}), 1.0)
        policy({"close": np.array([1.0, 6.0])})
        np.testing.assert_array_equal(agent.seen[0], [5.0, 0.0])
        np.testing.assert_array_equal(agent.seen[1], [6.0, 1.0])                      # the previous target is part of the next observation
        self.assertEqual(policy({"close": np.array([np.nan])}), 1.0)                  # unusable features: hold what you have


class BanditTests(unittest.TestCase):
    def test_all_bandits_find_the_best_arm_and_keep_exploring(self):
        means = {"a": 0.0, "b": 0.05, "c": 0.25}
        for cls in (EpsilonGreedyBandit, UCB1Bandit, ThompsonBandit):
            rng = np.random.default_rng(0)
            bandit = cls(list(means), seed=0)
            for _ in range(800):
                arm = bandit.select()
                bandit.update(arm, means[arm] + rng.normal() * 0.5)
            pulls = {s["arm"]: s["n"] for s in bandit.summary()}
            self.assertEqual(bandit.best(), "c", cls.__name__)
            self.assertGreater(pulls["c"], 400, cls.__name__)
            self.assertGreaterEqual(min(pulls.values()), 2, cls.__name__)             # every arm was tried before being set aside
        self.assertGreater(bandit.weights()["c"], 0.6)
        self.assertAlmostEqual(sum(bandit.weights().values()), 1.0)

    def test_discount_lets_a_bandit_follow_a_change(self):
        rng = np.random.default_rng(1)
        fixed, forgetful = ThompsonBandit(["x", "y"], seed=0), ThompsonBandit(["x", "y"], discount=0.97, seed=0)
        for step in range(1200):
            best = "x" if step < 600 else "y"                                         # the edge moves half way through
            for b in (fixed, forgetful):
                arm = b.select()
                b.update(arm, (0.3 if arm == best else 0.0) + rng.normal() * 0.3)
        self.assertEqual(forgetful.best(), "y")
        self.assertGreater(forgetful.means[1] - forgetful.means[0], fixed.means[1] - fixed.means[0])

    def test_validation(self):
        with self.assertRaises(ValidationError):
            UCB1Bandit(["only"])
        with self.assertRaises(ValidationError):
            UCB1Bandit(["a", "a"])
        b = EpsilonGreedyBandit(["a", "b"])
        with self.assertRaises(ValidationError):
            b.update("z", 1.0)
        with self.assertRaises(ValidationError):
            b.update("a", float("inf"))
        self.assertEqual(b.select(), "a")                                             # untried arms first


def switching_returns(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    states = np.zeros(n, int)
    s = 0
    for t in range(1, n):
        if rng.random() < (0.02 if s == 0 else 0.05):
            s = 1 - s
        states[t] = s
    return np.where(states == 0, rng.normal(0.0006, 0.006, n), rng.normal(-0.001, 0.02, n)), states


class RegimeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r, cls.states = switching_returns()
        cls.det = RegimeDetector(2).fit(cls.r[:2000])

    def test_recovers_regimes_and_their_statistics(self):
        calm, wild = self.det.describe()
        self.assertEqual((calm["label"], wild["label"]), ("calm", "turbulent"))
        self.assertAlmostEqual(calm["ann_vol"], 0.006 * math.sqrt(252), delta=0.02)
        self.assertAlmostEqual(wild["ann_vol"], 0.02 * math.sqrt(252), delta=0.05)
        self.assertAlmostEqual(calm["expected_bars"], 50, delta=25)                   # 1 / 0.02
        self.assertGreater((self.det.regimes(self.r)[2000:] == self.states[2000:]).mean(), 0.88)   # out of sample, causally
        cur = self.det.current(self.r)
        self.assertEqual(cur["label"], self.det.labels[cur["state"]])
        self.assertAlmostEqual(sum(cur["probabilities"].values()), 1.0)

    def test_filter_is_causal_and_smoothing_is_not(self):
        full, part = self.det.probabilities(self.r), self.det.probabilities(self.r[:2500])
        np.testing.assert_allclose(full[:2500], part, atol=1e-12)                     # filtered probabilities ignore the future
        hmm = self.det.hmm
        self.assertFalse(np.allclose(hmm.predict_proba(self.r)[:2500], hmm.predict_proba(self.r[:2500]), atol=1e-6))   # smoothed ones do not
        np.testing.assert_allclose(hmm.filter(self.r).sum(axis=1), 1.0)
        np.testing.assert_allclose(hmm.transition_.sum(axis=1), 1.0)
        self.assertEqual(len(hmm.predict(self.r)), len(self.r))
        self.assertAlmostEqual(hmm.next_state_proba(self.r).sum(), 1.0)

    def test_likelihood_prefers_the_right_model_and_errors(self):
        self.assertGreater(self.det.hmm.score(self.r[2000:]), GaussianHMM(2, n_iter=1, n_init=1).fit(self.r[:2000]).score(self.r[2000:]) - 50)
        three = RegimeDetector(3).fit(self.r[:2000])
        self.assertEqual(three.labels, ["calm", "regime_1", "turbulent"])
        vols = [d["ann_vol"] for d in three.describe()]
        self.assertEqual(vols, sorted(vols))
        with self.assertRaises(ModelError):
            RegimeDetector().regimes(self.r)
        with self.assertRaises(ValidationError):
            RegimeDetector().fit(self.r[:10])
        with self.assertRaises(ValidationError):
            self.det.regimes(np.array([0.01, np.nan]))
        with self.assertRaises(ValidationError):
            self.det.hmm.filter(np.zeros((5, 2)))


class FuzzyTests(unittest.TestCase):
    def setUp(self):
        self.sys = FuzzySystem([FuzzyVariable("rsi", {"oversold": left_shoulder(25, 40), "overbought": right_shoulder(60, 75)}),
                                FuzzyVariable("trend", {"down": left_shoulder(-1, 0), "up": right_shoulder(0, 1)})])
        self.sys.add_rule("IF rsi IS oversold AND trend IS up THEN 1.0")
        self.sys.add_rule("if rsi is overbought and trend is down then -1.0")
        self.sys.add_rule("IF rsi IS NOT oversold AND rsi IS NOT overbought THEN 0 WEIGHT 0.5")

    def test_membership_functions(self):
        self.assertEqual([triangle(0, 5, 10)(x) for x in (-1, 0, 2.5, 5, 10)], [0.0, 0.0, 0.5, 1.0, 0.0])
        self.assertEqual([trapezoid(0, 2, 4, 6)(x) for x in (1, 3, 5, 7)], [0.5, 1.0, 0.5, 0.0])
        self.assertEqual([left_shoulder(25, 40)(x) for x in (10, 25, 32.5, 40, 90)], [1.0, 1.0, 0.5, 0.0, 0.0])
        self.assertAlmostEqual(gaussian(0, 1)(1.0), math.exp(-0.5))
        for bad in (lambda: triangle(5, 1, 3), lambda: left_shoulder(4, 4), lambda: gaussian(0, 0)):
            with self.assertRaises(ValidationError):
                bad()

    def test_output_is_graded_not_stepped(self):
        scores = [self.sys.evaluate(rsi=r, trend=0.8) for r in (20, 30, 35, 39, 50)]
        self.assertEqual(scores[0], 1.0)
        self.assertEqual(scores, sorted(scores, reverse=True))                        # conviction fades smoothly as RSI rises
        self.assertTrue(0 < scores[2] < 1)
        self.assertEqual(self.sys.evaluate(rsi=80, trend=-1), -1.0)
        self.assertEqual(self.sys.evaluate(rsi=50, trend=0), 0.0)
        explained = self.sys.explain(rsi=35, trend=0.5)
        self.assertEqual(len(explained), 3)
        self.assertAlmostEqual(explained[0]["strength"], 1 / 3)                       # min(oversold 1/3, up 1/2)

    def test_or_rules_defaults_and_errors(self):
        s = FuzzySystem([FuzzyVariable("x", {"low": left_shoulder(0, 1), "high": right_shoulder(0, 1)})], default=-9.0)
        s.add_rule("IF x IS low OR x IS high THEN 1")
        self.assertEqual(s.evaluate(x=0.5), 1.0)
        only = FuzzySystem([FuzzyVariable("x", {"high": right_shoulder(0, 1)})], default=-9.0).add_rule("IF x IS high THEN 1")
        self.assertEqual(only.evaluate(x=-5), -9.0)                                   # nothing fired: the declared default
        for text in ("x IS low THEN 1", "IF y IS low THEN 1", "IF x IS medium THEN 1", "IF x IS low AND x IS high OR x IS low THEN 1",
                     "IF x IS low THEN lots"):
            with self.assertRaises(ValidationError, msg=text):
                s.add_rule(text)
        with self.assertRaises(ValidationError):
            s.evaluate()
        with self.assertRaises(ValidationError):
            s.evaluate(x=float("nan"))
        with self.assertRaises(ValidationError):
            FuzzySystem([FuzzyVariable("x", {"a": triangle(0, 1, 2)})]).evaluate(x=1)  # no rules


class AnomalyAndChangeTests(unittest.TestCase):
    def test_isolation_forest_ranks_outliers_first(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1000, 3))
        X[:5] += 8.0
        forest = IsolationForest(seed=1).fit(X)
        score = forest.score(X)
        self.assertEqual(set(np.argsort(score)[-5:]), {0, 1, 2, 3, 4})
        self.assertGreater(score[:5].min(), 0.7)
        self.assertAlmostEqual(np.median(score[5:]), 0.45, delta=0.1)
        self.assertTrue(forest.is_anomaly(X)[:5].all())
        self.assertAlmostEqual(forest.is_anomaly(X).mean(), 0.01, delta=0.005)        # the contamination level
        np.testing.assert_array_equal(IsolationForest(seed=1).fit(X).score(X[:10]), score[:10])
        with self.assertRaises(ModelError):
            IsolationForest().score(X)
        with self.assertRaises(ValidationError):
            IsolationForest().fit(X[:3])

    def test_robust_scores(self):
        rng = np.random.default_rng(1)
        x = np.concatenate([rng.normal(size=200), [15.0], rng.normal(size=20)])
        self.assertGreater(robust_zscore(x)[200], 10)
        self.assertLess(abs((x[200] - x.mean()) / x.std()), robust_zscore(x)[200])    # the outlier inflates the ordinary z-score's own scale
        z = rolling_robust_zscore(x, 50)
        self.assertTrue(np.isnan(z[:50]).all())
        self.assertGreater(z[200], 10)
        np.testing.assert_allclose(rolling_robust_zscore(x[:150], 50), z[:150], equal_nan=True)   # causal
        X = rng.normal(size=(500, 2)) @ np.array([[1.0, 0.9], [0.0, 0.3]])            # strongly correlated columns
        d = mahalanobis(np.array([[1.0, 1.0], [1.0, -1.0]]), X)
        self.assertGreater(d[1], 3 * d[0])                                            # same distance from the centre, against the grain
        with self.assertRaises(ValidationError):
            mahalanobis(X[:2], X[:2])

    def test_change_points(self):
        rng = np.random.default_rng(2)
        x = np.concatenate([rng.normal(0, 1, 400), rng.normal(2.5, 1, 300), rng.normal(0, 3, 300)])
        cps = changepoints_from_run_length(bocpd(x, hazard=1 / 200)["run_length"])
        self.assertTrue(any(abs(c - 400) < 15 for c in cps))                          # the mean shift
        self.assertTrue(any(abs(c - 700) < 40 for c in cps))                          # the variance shift
        self.assertLessEqual(len(cps), 5)
        quiet = bocpd(rng.normal(0, 1, 600), hazard=1 / 200)
        self.assertEqual(changepoints_from_run_length(quiet["run_length"], 100), [])
        prefix = bocpd(x[:500], hazard=1 / 200)
        np.testing.assert_array_equal(prefix["run_length"], bocpd(x, hazard=1 / 200)["run_length"][:500])   # online: the past never changes
        cs = cusum(x)
        self.assertTrue(any(400 <= a <= 420 for a in cs["alarms"]))
        self.assertEqual(cs["direction"][[i for i, a in enumerate(cs["alarms"]) if 400 <= a <= 420][0]], 1)
        self.assertEqual(len(cusum(rng.normal(0, 1, 300), threshold=12)["alarms"]), 0)
        with self.assertRaises(ValidationError):
            bocpd(x, hazard=1.5)
        with self.assertRaises(ValidationError):
            cusum(x, threshold=0)


class GeneticProgrammingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        n = 2500
        self.close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        self.volume = rng.lognormal(10, 0.3, n)
        dev = (self.close - np.convolve(self.close, np.ones(10) / 10, mode="full")[:n]) / self.close
        self.target = np.concatenate([(-0.8 * dev + rng.normal(0, 0.01, n))[:-1], [np.nan]])    # mean reversion to the 10-bar average
        self.noise = np.concatenate([rng.normal(0, 0.01, n - 1), [np.nan]])

    def test_finds_a_planted_relationship_and_it_holds_out_of_sample(self):
        res = SymbolicAlpha(population=120, generations=8, seed=1).fit({"close": self.close, "volume": self.volume}, self.target)
        self.assertGreater(abs(res.train_ic), 0.5)
        self.assertGreater(res.test_ic, 0.5)                                          # on data evolution never saw
        self.assertIn("close", res.formula)
        self.assertGreater(res.n_evaluated, 100)                                      # the number of trials, for deflated Sharpe
        self.assertEqual(res.history, sorted(res.history))                            # elitism: the best never gets worse
        values = res.values({"close": self.close, "volume": self.volume})
        self.assertGreater(information_coefficient(values[1800:], self.target[1800:]), 0.5)   # oriented: higher means higher return

    def test_on_noise_the_in_sample_fit_does_not_survive(self):
        res = SymbolicAlpha(population=120, generations=8, seed=1).fit({"close": self.close, "volume": self.volume}, self.noise)
        self.assertLess(abs(res.test_ic), 0.08)
        self.assertLess(abs(res.test_ic), abs(res.train_ic) + 0.05)

    def test_trees_are_causal_and_printable(self):
        tree = ("sub", ("ts_mean", ("x", "close"), 5), ("delay", ("x", "close"), 3))
        self.assertEqual(to_string(tree), "sub(ts_mean(close, 5), delay(close, 3))")
        full, part = evaluate_tree(tree, {"close": self.close}), evaluate_tree(tree, {"close": self.close[:300]})
        np.testing.assert_allclose(full[:300], part, equal_nan=True)
        self.assertTrue(np.isnan(evaluate_tree(("div", ("x", "close"), ("const", 0.0)), {"close": self.close})).all())   # protected division
        with self.assertRaises(ValidationError):
            evaluate_tree(("future_peek", ("x", "close")), {"close": self.close})
        with self.assertRaises(ValidationError):
            SymbolicAlpha().fit({"close": self.close[:100]}, self.target[:100])
        with self.assertRaises(ValidationError):
            SymbolicAlpha(population=2)


if __name__ == "__main__":
    unittest.main()
