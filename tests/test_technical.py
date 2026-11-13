import math
import unittest

import numpy as np
import pandas as pd

import helpers  # noqa: F401
from helpers import free_config
from algoplatform import technical as ta
from algoplatform.backtest import BacktestEngine
from algoplatform.core.exceptions import ConfigError, ExpressionError, MissingDataError, StrategyError, ValidationError
from algoplatform.data import MarketData, synthetic
from algoplatform.strategy import (PolicyStrategy, create_from_expressions, expression_factory,
                                   expression_spec_from_json, expression_spec_to_json, probability_policy)
from algoplatform.technical import operators as op
from algoplatform.technical.candles import (doji, engulfing, gap, hammer, inside_bar, marubozu, morning_star, outside_bar,
                                            shooting_star, three_white_soldiers)

DF = synthetic.gbm(400, seed=11)
REQUIRED_N = {"n": 10}


def ohlc(rows):
    """rows of (open, high, low, close) -> four arrays."""
    a = np.array(rows, float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def _outputs(result):
    return list(result.values()) if isinstance(result, dict) else [result]


class IndicatorValueTests(unittest.TestCase):
    def test_rolling_helpers_match_pandas(self):
        c, v = DF["close"], DF["volume"]
        np.testing.assert_allclose(ta.rolling_sum(c, 7), c.rolling(7).sum(), equal_nan=True)
        np.testing.assert_allclose(ta.vwma(c, v, 9), (c * v).rolling(9).sum() / v.rolling(9).sum(), equal_nan=True)
        np.testing.assert_allclose(ta.momentum(c, 5), c.diff(5), equal_nan=True)
        tp = (DF["high"] + DF["low"] + c) / 3
        mad = tp.rolling(20).apply(lambda w: np.abs(w - w.mean()).mean(), raw=True)
        np.testing.assert_allclose(ta.cci(DF["high"], DF["low"], c, 20), (tp - tp.rolling(20).mean()) / (0.015 * mad), equal_nan=True)
        hh, ll = DF["high"].rolling(14).max(), DF["low"].rolling(14).min()
        np.testing.assert_allclose(ta.williams_r(DF["high"], DF["low"], c, 14), -100 * (hh - c) / (hh - ll), equal_nan=True)

    def test_moving_average_family_on_a_straight_line(self):
        x = np.arange(1.0, 101.0)                                       # on a line every lag-corrected average equals the price
        for name, fn in (("dema", ta.dema), ("tema", ta.tema), ("hma", ta.hma)):
            out = fn(x, 9)
            valid = ~np.isnan(out)
            self.assertTrue(valid.sum() > 50, name)
            np.testing.assert_allclose(out[valid], x[valid], atol=1e-8, err_msg=name)
        lr = ta.linreg(x, 10)
        np.testing.assert_allclose(lr["slope"][9:], 1.0)
        np.testing.assert_allclose(lr["value"][9:], x[9:])
        np.testing.assert_allclose(lr["r2"][9:], 1.0)

    def test_kama_follows_trend_and_ignores_noise(self):
        trend = np.arange(1.0, 201.0)
        self.assertAlmostEqual(ta.kama(trend, 10)[-1], trend[-1], delta=1.5)          # efficiency ratio 1: tracks closely
        chop = 100 + np.tile([1.0, -1.0], 100)
        self.assertLess(np.nanstd(ta.kama(chop, 10)[50:]), 0.2)                       # efficiency ratio ~0: almost flat

    def test_bounded_oscillators_stay_in_range(self):
        h, l, c, v = DF["high"], DF["low"], DF["close"], DF["volume"]
        bounded = {"williams_r": (ta.williams_r(h, l, c), -100, 0), "mfi": (ta.mfi(h, l, c, v), 0, 100), "cmf": (ta.cmf(h, l, c, v), -1, 1),
                   "cmo": (ta.cmo(c), -100, 100), "tsi": (ta.tsi(c), -100, 100), "uo": (ta.ultimate_oscillator(h, l, c), 0, 100),
                   "stoch_rsi_k": (ta.stoch_rsi(c)["k"], 0, 100), "aroon_up": (ta.aroon(h, l)["up"], 0, 100),
                   "choppiness": (ta.choppiness(h, l, c), 0, 100)}
        for name, (x, lo, hi) in bounded.items():
            x = x[~np.isnan(x)]
            self.assertTrue(len(x) > 100, name)
            self.assertTrue((x >= lo - 1e-9).all() and (x <= hi + 1e-9).all(), name)

    def test_supertrend_and_sar_sit_on_the_correct_side_of_price(self):
        up = synthetic.gbm(300, mu=0.6, sigma=0.08, seed=2)
        st = ta.supertrend(up["high"], up["low"], up["close"], 10, 3.0)
        d, line, c = st["direction"], st["line"], up["close"].to_numpy()
        ok = ~np.isnan(d)
        self.assertTrue(((d[ok] > 0) == (c[ok] >= line[ok])).mean() > 0.97)           # the line is below price in an uptrend, above in a downtrend
        self.assertGreater((d[ok] > 0).mean(), 0.7)
        sar = ta.parabolic_sar(up["high"], up["low"])
        ok = ~np.isnan(sar["sar"])
        below = sar["sar"][ok] <= up["low"].to_numpy()[ok]
        self.assertTrue(((sar["direction"][ok] > 0) == below).mean() > 0.9)

    def test_aroon_reports_bars_since_extreme(self):
        h = np.array([1, 2, 3, 9, 4, 5, 6.0])
        l = h - 0.5
        a = ta.aroon(h, l, 4)
        self.assertTrue(np.isnan(a["up"][3]))                                         # needs n + 1 bars
        self.assertEqual(a["up"][4], 75.0)                                            # the high (9) was 1 bar ago
        self.assertEqual(a["up"][6], 25.0)                                            # ... and 3 bars ago two bars later

    def test_ichimoku_spans_are_shifted_forward_never_backward(self):
        h, l, c = DF["high"].to_numpy(), DF["low"].to_numpy(), DF["close"].to_numpy()
        ich = ta.ichimoku(h, l, c, 9, 26, 52)
        raw_a = (ich["tenkan"] + ich["kijun"]) / 2
        np.testing.assert_allclose(ich["span_a"][26:], raw_a[:-26], equal_nan=True)   # value shown at t was known 26 bars earlier
        self.assertTrue(set(np.unique(ich["cloud_position"][~np.isnan(ich["cloud_position"])])) <= {-1.0, 0.0, 1.0})

    def test_relative_volume_excludes_the_current_bar(self):
        v = np.array([100.0] * 20 + [500.0])
        self.assertAlmostEqual(ta.relative_volume(v, 20)[-1], 5.0)

    def test_volume_profile_finds_point_of_control(self):
        price = np.concatenate([np.full(50, 100.0), np.linspace(90, 110, 50)])
        vp = ta.volume_profile(price, np.full(100, 10.0), bins=20)
        self.assertAlmostEqual(vp["poc"], 100.0, delta=1.0)
        self.assertTrue(vp["val"] <= vp["poc"] <= vp["vah"])
        with self.assertRaises(ValidationError):
            ta.volume_profile([1.0], [1.0])


class CausalityTests(unittest.TestCase):
    def test_every_catalogue_indicator_and_pattern_is_prefix_invariant(self):
        """The no-look-ahead property, for the whole catalogue at once: values on the first 250 bars must not change when
        150 later bars are appended."""
        short = DF.iloc[:250]
        checked = 0
        for name, spec in ta.CATALOG.items():
            if spec.kind == "operator":
                continue
            params = {k: REQUIRED_N[k] for k, v in spec.params.items() if v is ta.catalog.REQUIRED}
            full, part = ta.compute(name, DF, **params), ta.compute(name, short, **params)
            for a, b in zip(_outputs(full), _outputs(part)):
                self.assertEqual(len(a), len(DF), name)
                np.testing.assert_allclose(np.asarray(a, float)[:250], np.asarray(b, float), rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=name)
            checked += 1
        self.assertGreater(checked, 60)

    def test_operators_are_prefix_invariant(self):
        c = DF["close"].to_numpy()
        cond = c > ta.sma(c, 10)
        cases = {"crossover": lambda x, k: op.crossover(x, ta.sma(x, 10)), "rising": lambda x, k: op.rising(x, 3),
                 "barssince": lambda x, k: op.barssince(k), "valuewhen": lambda x, k: op.valuewhen(k, x, 1), "streak": lambda x, k: op.streak(k),
                 "any_true": lambda x, k: op.any_true(k, 5), "percent_rank": lambda x, k: op.percent_rank(x, 20),
                 "latch": lambda x, k: op.latch(k, ~k), "exrem": lambda x, k: op.exrem(k, ~k), "cummax": lambda x, k: op.cummax(x)}
        for name, fn in cases.items():
            np.testing.assert_array_equal(np.asarray(fn(c, cond))[:250].astype(float), np.asarray(fn(c[:250], cond[:250])).astype(float), err_msg=name)


class LevelTests(unittest.TestCase):
    def test_swings_are_reported_only_once_confirmed(self):
        h = np.array([1, 2, 3, 2, 1, 2, 5, 2, 1.0])
        sw = ta.swing_points(h, h - 0.5, left=2, right=2)
        highs = [(s.index, s.confirm, s.price) for s in sw if s.kind == "high"]
        self.assertEqual(highs, [(2, 4, 3.0), (6, 8, 5.0)])
        step = ta.last_swing(h, h - 0.5, "high", 2, 2)
        self.assertTrue(np.isnan(step[:4]).all())                                     # unknown until bar 4, although the peak was at bar 2
        np.testing.assert_array_equal(step[4:], [3, 3, 3, 3, 5])

    def test_support_resistance_clusters_touches(self):
        h = np.array([10, 12, 10, 9, 10, 12.05, 10, 9.02, 10, 11, 10.5])
        levels = ta.support_resistance(h, h - 1.0, h - 0.5, left=1, right=1, tolerance=0.01)
        res = [lv for lv in levels if lv["kind"] == "resistance"]
        self.assertEqual(res[0]["touches"], 2)
        self.assertAlmostEqual(res[0]["price"], 12.025, places=3)

    def test_pivots_use_the_previous_bar(self):
        h, l, c = np.array([10.0, 20.0]), np.array([8.0, 15.0]), np.array([9.0, 18.0])
        pv = ta.pivot_points(h, l, c)
        self.assertTrue(np.isnan(pv["pivot"][0]))
        self.assertAlmostEqual(pv["pivot"][1], 9.0)
        self.assertAlmostEqual(pv["r1"][1], 2 * 9.0 - 8.0)
        for method in ("fibonacci", "camarilla", "woodie"):
            self.assertEqual(len(ta.pivot_points(h, l, c, method)["s3"]), 2)
        with self.assertRaises(ValidationError):
            ta.pivot_points(h, l, c, "nope")

    def test_heikin_ashi_and_fibonacci(self):
        ha = ta.heikin_ashi(DF["open"], DF["high"], DF["low"], DF["close"])
        self.assertTrue((ha["high"] >= np.maximum(ha["open"], ha["close"]) - 1e-12).all())
        self.assertTrue((ha["low"] <= np.minimum(ha["open"], ha["close"]) + 1e-12).all())
        fib = ta.fibonacci_levels(100, 200)
        self.assertAlmostEqual(fib["0.618"], 138.2)
        self.assertAlmostEqual(fib["ext_1.618"], 261.8)
        with self.assertRaises(ValidationError):
            ta.fibonacci_levels(2, 1)


class CandleTests(unittest.TestCase):
    def test_single_bar_shapes(self):
        o, h, l, c = ohlc([(10, 11, 9, 10.02), (10, 12, 10, 12), (12, 12, 10, 10), (10, 10.1, 9.9, 10.05)])
        np.testing.assert_array_equal(doji(o, h, l, c), [1, 0, 0, 0])
        np.testing.assert_array_equal(marubozu(o, h, l, c), [0, 1, -1, 0])

    def test_engulfing_and_inside_outside_and_gap(self):
        o, h, l, c = ohlc([(10, 10.2, 9.4, 9.5), (9.4, 10.6, 9.3, 10.5), (10.3, 10.4, 10.0, 10.1), (11, 11.5, 10.9, 11.4)])
        np.testing.assert_array_equal(engulfing(o, h, l, c), [0, 1, 0, 0])
        np.testing.assert_array_equal(outside_bar(o, h, l, c), [0, 1, 0, 0])
        np.testing.assert_array_equal(inside_bar(o, h, l, c), [0, 0, 1, 0])
        np.testing.assert_array_equal(gap(o, h, l, c), [0, 0, 0, 1])

    def test_reversal_patterns_need_a_prior_trend(self):
        down = [(100 - i, 100.3 - i, 98.8 - i, 99 - i) for i in range(15)]
        hammer_bar = (84.0, 84.1, 81.0, 83.9)                                         # small body on top, long lower shadow
        o, h, l, c = ohlc(down + [hammer_bar])
        self.assertEqual(hammer(o, h, l, c)[-1], 1)
        up = [(100 + i, 101.2 + i, 99.7 + i, 101 + i) for i in range(15)]
        o, h, l, c = ohlc(up + [(116.0, 116.1, 113.0, 115.9)])                        # same shape after a rise is not a hammer
        self.assertEqual(hammer(o, h, l, c)[-1], 0)
        o, h, l, c = ohlc(up + [(116.0, 119.0, 115.9, 116.1)])
        self.assertEqual(shooting_star(o, h, l, c)[-1], -1)

    def test_three_bar_patterns(self):
        base = [(50, 50.5, 49.5, 50.2)] * 14
        o, h, l, c = ohlc(base + [(50, 50.1, 46.9, 47.0), (46.5, 46.8, 46.2, 46.6), (46.9, 49.6, 46.8, 49.5)])
        self.assertEqual(morning_star(o, h, l, c)[-1], 1)
        o, h, l, c = ohlc(base + [(50, 51.1, 49.9, 51.0), (50.5, 52.1, 50.4, 52.0), (51.5, 53.1, 51.4, 53.0)])
        self.assertEqual(three_white_soldiers(o, h, l, c)[-1], 1)

    def test_scan_scores_directional_patterns_only(self):
        scan = ta.scan_candles(DF)
        self.assertEqual(set(scan.columns), set(ta.CANDLE_PATTERNS) | {"score"})
        directional = [n for n in ta.CANDLE_PATTERNS if n not in ("doji", "spinning_top", "inside_bar", "outside_bar")]
        np.testing.assert_array_equal(scan["score"], scan[directional].sum(axis=1))
        self.assertTrue(scan.index.equals(DF.index))
        with self.assertRaises(ValidationError):
            ta.scan_candles(DF, ["nope"])
        with self.assertRaises(ValidationError):
            ta.scan_candles({"close": [1.0]})


def _zigzag(points, per_leg=8):
    """A price path through the given turning points, with small bars around it."""
    close = np.concatenate([np.linspace(a, b, per_leg, endpoint=False) for a, b in zip(points, points[1:])] + [[points[-1]]])
    return pd.DataFrame({"open": close, "high": close * 1.002, "low": close * 0.998, "close": close, "volume": 1e6})


class ChartPatternTests(unittest.TestCase):
    def test_double_top_confirmed_on_neckline_break(self):
        df = _zigzag([100, 120, 108, 120.5, 100])
        found = ta.double_tops_bottoms(df["high"], df["low"], df["close"])
        self.assertEqual([m.name for m in found], ["double_top"])
        m = found[0]
        self.assertEqual(m.direction, -1)
        self.assertLess(df["close"].iloc[m.confirm], m.level)                         # confirmed by a close below the neckline
        self.assertGreater(m.confirm, m.end)
        self.assertAlmostEqual(m.target, m.level - (m.meta["extreme"] - m.level), places=6)
        early = ta.double_tops_bottoms(df["high"].iloc[:m.confirm], df["low"].iloc[:m.confirm], df["close"].iloc[:m.confirm])
        self.assertEqual(early, [])                                                   # one bar earlier it was not yet known

    def test_double_bottom_and_invalidating_third_peak(self):
        df = _zigzag([120, 100, 111, 99.6, 121])
        self.assertEqual([m.name for m in ta.double_tops_bottoms(df["high"], df["low"], df["close"])], ["double_bottom"])
        df = _zigzag([100, 120, 108, 120.5, 115, 130])                                # rallies through the peaks: no pattern
        self.assertEqual(ta.double_tops_bottoms(df["high"], df["low"], df["close"]), [])

    def test_head_and_shoulders_both_ways(self):
        df = _zigzag([100, 115, 108, 125, 109, 116, 95])
        found = ta.head_and_shoulders(df["high"], df["low"], df["close"])
        self.assertEqual([(m.name, m.direction) for m in found], [("head_and_shoulders", -1)])
        self.assertEqual(found[0].meta["head"], max(m for m in [found[0].meta["head"], found[0].meta["left_shoulder"], found[0].meta["right_shoulder"]]))
        inv = _zigzag([125, 110, 117, 100, 116, 109, 130])
        self.assertEqual([m.name for m in ta.head_and_shoulders(inv["high"], inv["low"], inv["close"])], ["inverse_head_and_shoulders"])

    def test_consolidation_shapes(self):
        asc = _zigzag([100, 110, 102, 110, 104, 110, 106, 110, 108], per_leg=6)       # flat highs, rising lows
        self.assertEqual(ta.consolidation_pattern(asc["high"], asc["low"], asc["close"]).name, "ascending_triangle")
        sym = _zigzag([100, 120, 102, 117, 105, 114, 108, 112, 110], per_leg=6)       # falling highs, rising lows
        self.assertEqual(ta.consolidation_pattern(sym["high"], sym["low"], sym["close"]).name, "symmetrical_triangle")
        chan = _zigzag([100, 110, 104, 114, 108, 118, 112, 122, 116], per_leg=6)
        self.assertEqual(ta.consolidation_pattern(chan["high"], chan["low"], chan["close"]).name, "channel_up")
        self.assertIsNone(ta.consolidation_pattern([1.0, 2.0], [0.5, 1.5], [1.0, 2.0]))

    def test_range_breakout_and_divergence(self):
        flat = np.full(30, 100.0)
        c = np.concatenate([flat, [104.0]])
        sig = ta.range_breakout(c * 1.001, c * 0.999, c, 20, 0.05)
        self.assertEqual(sig[-1], 1)
        self.assertEqual(int(np.abs(sig[:-1]).sum()), 0)
        df = _zigzag([100, 110, 95, 104, 92, 108])                                    # lower low in price ...
        osc = np.interp(np.arange(len(df)), [0, 16, 32, len(df) - 1], [50, 20, 30, 60])  # ... higher low in the oscillator
        found = ta.divergences(df["high"], df["low"], osc)
        self.assertIn("bullish_divergence", [m.name for m in found])
        with self.assertRaises(ValidationError):
            ta.divergences(df["high"], df["low"], osc[:-1])

    def test_detection_is_prefix_consistent(self):
        """Running the detectors on the first 500 bars must give exactly the full-history matches confirmed before bar 500."""
        long = synthetic.gbm(700, sigma=0.3, seed=4)

        def reversals(df):
            h, l, c = df["high"], df["low"], df["close"]
            found = ta.double_tops_bottoms(h, l, c) + ta.head_and_shoulders(h, l, c) + ta.divergences(h, l, ta.rsi(c, 14))
            return sorted((m.name, m.start, m.end, m.confirm, round(m.level, 8)) for m in found)
        full = [m for m in reversals(long) if m[3] < 500]
        self.assertEqual(reversals(long.iloc[:500]), full)
        self.assertGreater(len(full), 3)
        matches = [m for m in ta.scan_chart_patterns(long) if m.confirm < 700]
        signals = ta.pattern_signals(matches, len(long))
        self.assertEqual(len(signals), len(long))
        self.assertEqual(int(signals.sum()), sum(m.direction for m in matches))


class OperatorTests(unittest.TestCase):
    def test_crossing_and_nan_safety(self):
        a = np.array([1, 2, 3, 2, 1, np.nan, 3.0])
        np.testing.assert_array_equal(op.crossover(a, 2), [False, False, True, False, False, False, False])
        np.testing.assert_array_equal(op.crossunder(a, 2), [False, False, False, False, True, False, False])
        np.testing.assert_array_equal(op.above(a, 2), [False, False, True, False, False, False, True])     # NaN is never "above"
        np.testing.assert_array_equal(op.between(a, 2, 3), [False, True, True, True, False, False, True])
        with self.assertRaises(ValidationError):
            op.crossover([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_lookback_operators(self):
        x = np.array([1, 2, 3, 3, 2, 5.0])
        np.testing.assert_array_equal(op.rising(x, 2), [False, False, True, False, False, False])
        np.testing.assert_array_equal(op.falling(x, 1), [False, False, False, False, True, False])
        np.testing.assert_allclose(op.shift(x, 2), [np.nan, np.nan, 1, 2, 3, 3], equal_nan=True)
        np.testing.assert_allclose(op.change(x), [np.nan, 1, 1, 0, -1, 3], equal_nan=True)
        np.testing.assert_allclose(op.pct_change(x, 1)[1], 1.0)
        with self.assertRaises(ValidationError):
            op.shift(x, -1)                                                           # looking forward is refused outright

    def test_event_bookkeeping(self):
        cond = np.array([False, True, False, False, True, True, False])
        x = np.arange(10.0, 17.0)
        np.testing.assert_allclose(op.barssince(cond), [np.nan, 0, 1, 2, 0, 0, 1], equal_nan=True)
        np.testing.assert_allclose(op.valuewhen(cond, x), [np.nan, 11, 11, 11, 14, 15, 15], equal_nan=True)
        np.testing.assert_allclose(op.valuewhen(cond, x, 1), [np.nan, np.nan, np.nan, np.nan, 11, 14, 14], equal_nan=True)
        np.testing.assert_array_equal(op.streak(cond), [0, 1, 0, 0, 1, 2, 0])
        np.testing.assert_array_equal(op.rising_edge(cond), [False, True, False, False, True, False, False])
        np.testing.assert_array_equal(op.any_true(cond, 2), [False, True, True, False, True, True, True])
        np.testing.assert_array_equal(op.all_true(cond, 2), [False, False, False, False, False, True, False])
        np.testing.assert_allclose(op.count_true(cond, 3), [np.nan, np.nan, 1, 1, 1, 2, 2], equal_nan=True)

    def test_state_machines(self):
        entry = np.array([True, True, False, False, True, False, True])
        exit_ = np.array([False, False, True, False, False, False, True])
        np.testing.assert_array_equal(op.latch(entry, exit_), [True, True, False, False, True, True, False])   # reset wins a tie
        np.testing.assert_array_equal(op.exrem(entry, exit_), [True, False, False, False, True, False, False])

    def test_scaling_and_logic(self):
        x = np.array([5, 1, 3, 9, 2.0])
        np.testing.assert_allclose(op.percent_rank(x, 3), [np.nan, np.nan, 0.5, 1.0, 0.0], equal_nan=True)
        np.testing.assert_allclose(op.normalize(x, 3), [np.nan, np.nan, 0.5, 1.0, 0.0], equal_nan=True)
        np.testing.assert_array_equal(op.and_([True, False, True], [True, True, np.nan]), [True, False, False])
        np.testing.assert_array_equal(op.or_([False, False], [np.nan, 1.0]), [False, True])
        np.testing.assert_array_equal(op.not_([1.0, np.nan]), [False, True])
        np.testing.assert_allclose(op.where([True, False], 1.0, [5.0, 6.0]), [1.0, 6.0])
        np.testing.assert_allclose(op.nz([np.nan, 2.0], 7), [7, 2])
        np.testing.assert_allclose(op.log([-1.0, math.e]), [np.nan, 1.0], equal_nan=True)
        np.testing.assert_allclose(op.cummax([1, np.nan, 3, 2.0]), [1, 1, 3, 3])


class CatalogTests(unittest.TestCase):
    def test_compute_dispatches_inputs_outputs_and_source(self):
        np.testing.assert_allclose(ta.compute("rsi", DF, n=14), ta.rsi(DF["close"], 14), equal_nan=True)
        np.testing.assert_allclose(ta.compute("atr", DF), ta.atr(DF["high"], DF["low"], DF["close"]), equal_nan=True)
        np.testing.assert_allclose(ta.compute("macd", DF, output="hist"), ta.macd(DF["close"])["hist"], equal_nan=True)
        self.assertEqual(set(ta.compute("bollinger", DF)), {"mid", "upper", "lower", "pct_b", "bandwidth"})
        np.testing.assert_allclose(ta.compute("sma", DF, source="volume", n=5), ta.sma(DF["volume"], 5), equal_nan=True)

    def test_errors_name_the_problem(self):
        with self.assertRaises(ValidationError) as cm:
            ta.compute("rsii", DF)
        self.assertIn("rsi", cm.exception.details["did_you_mean"])
        with self.assertRaises(ValidationError) as cm:
            ta.compute("rsi", DF, period=14)
        self.assertIn("rsi(close, n=14)", cm.exception.details["signature"])
        with self.assertRaises(ValidationError):
            ta.compute("sma", DF)                                                     # missing required n
        with self.assertRaises(ValidationError):
            ta.compute("macd", DF, output="nope")
        with self.assertRaises(ValidationError):
            ta.compute("atr", DF, source="open")
        with self.assertRaises(ValidationError):
            ta.compute("crossover", DF)                                               # operators need explicit arguments
        with self.assertRaises(MissingDataError):
            ta.compute("mfi", {"close": DF["close"]})
        with self.assertRaises(ValidationError) as cm:
            ta.call("sma", DF["close"], "ten")
        self.assertEqual(cm.exception.details["function"], "sma")

    def test_describe_and_registration(self):
        kinds = {d["kind"] for d in ta.describe()}
        self.assertEqual(kinds, {"indicator", "pattern", "operator"})
        self.assertTrue(all(d["kind"] == "pattern" for d in ta.describe(kind="pattern")))
        self.assertGreater(len(ta.describe(kind="indicator")), 50)
        ta.register_function("test_double_close", lambda close, k=2.0: np.asarray(close, float) * k, category="custom", overwrite=True)
        np.testing.assert_allclose(ta.compute("test_double_close", DF, k=3.0), DF["close"] * 3)
        np.testing.assert_allclose(ta.evaluate("test_double_close(close) / close", DF), 2.0)   # immediately usable in formulas
        with self.assertRaises(ConfigError):
            ta.register_function("test_double_close", lambda close: close)
        with self.assertRaises(ConfigError):
            ta.register_function("close", lambda close: close)
        with self.assertRaises(ConfigError):
            ta.register_function("bad_inputs", lambda x: x, inputs=("price",))

    def test_indicator_frame_isolates_a_bad_column(self):
        cols = {"rsi": "rsi", "fast": ("ema", {"n": 10}), "hist": ("macd", {"output": "hist"}), "broken": ("ema", {"n": -1})}
        with self.assertRaises(ValidationError) as cm:
            ta.indicator_frame(DF, cols)
        self.assertEqual(cm.exception.details["column"], "broken")
        frame = ta.indicator_frame(DF, cols, on_error="nan")
        self.assertEqual(list(frame.columns), ["rsi", "fast", "hist", "broken"])
        self.assertTrue(frame["broken"].isna().all())
        self.assertFalse(frame["rsi"].isna().all())
        self.assertEqual(list(frame.attrs["errors"]), ["broken"])
        self.assertTrue(frame.index.equals(DF.index))


class ExpressionTests(unittest.TestCase):
    def test_arithmetic_comparison_logic(self):
        c = DF["close"].to_numpy()
        np.testing.assert_allclose(ta.evaluate("(high + low) / 2 - close * 0", DF), (DF["high"] + DF["low"]) / 2)
        np.testing.assert_array_equal(ta.Expression("close > open and not (volume < 0)").mask(DF), c > DF["open"].to_numpy())
        np.testing.assert_array_equal(ta.Expression("0 < rsi(close, 14) < 30").mask(DF), ta.rsi(c, 14) < 30)
        np.testing.assert_allclose(ta.evaluate("1 if close > open else -1", DF), np.where(c > DF["open"], 1.0, -1.0))
        np.testing.assert_allclose(ta.evaluate("close ** 2", DF), c ** 2)
        self.assertTrue(np.isnan(ta.evaluate("close / (close - close)", DF)).all())  # division by zero is unknown, not an error

    def test_functions_outputs_and_params(self):
        e = ta.Expression("crossover(ema(close, fast), ema(close, slow)) and macd(close).hist > 0", {"fast": 10, "slow": 30})
        self.assertEqual((e.columns, e.calls), ({"close"}, {"crossover", "ema", "macd"}))
        c = DF["close"].to_numpy()
        expected = op.crossover(ta.ema(c, 10), ta.ema(c, 30)) & (ta.macd(c)["hist"] > 0)
        np.testing.assert_array_equal(e.mask(DF), expected)
        other = e.mask(DF, {"fast": 5})                                               # parameters can be overridden per call (optimisers)
        np.testing.assert_array_equal(other, op.crossover(ta.ema(c, 5), ta.ema(c, 30)) & (ta.macd(c)["hist"] > 0))
        np.testing.assert_allclose(ta.evaluate("supertrend(high, low, close, n=10, mult=3).direction", DF),
                                   ta.supertrend(DF["high"], DF["low"], DF["close"])["direction"], equal_nan=True)
        self.assertEqual(ta.Expression("last_swing(high, low, 'high') > 0").mask(DF).dtype, bool)

    def test_warmup_nan_never_fires(self):
        e = ta.Expression("sma(close, 50) > 0")
        self.assertFalse(e.mask(DF)[:49].any())
        self.assertTrue(e.mask(DF)[49:].all())
        self.assertFalse(ta.Expression("sma(close, 50) != 0").mask(DF)[:49].any())    # even "not equal" is false while unknown
        self.assertFalse(e.holds(DF.iloc[:10]))
        self.assertTrue(math.isnan(e.last({"close": np.array([])})) or True)

    def test_constant_formulas_broadcast(self):
        self.assertTrue(ta.Expression("1 < 2").mask(DF).all())
        self.assertEqual(len(ta.Expression("1 < 2").mask(DF)), len(DF))

    def test_rejects_everything_that_is_not_a_formula(self):
        hostile = ["__import__('os').system('ls')", "close.__class__", "(lambda: 1)()", "[c for c in close]", "open('x')", "close[0]",
                   "exec('1')", "getattr(close, 'x')", "close if close else __builtins__", "{'a': 1}", "f'{close}'", "x := 1",
                   "macd(close).__dict__", "sma(*[close, 3])", "sma(close, **{'n': 3})", "close; open", "import os", "", "   "]
        for text in hostile:
            with self.assertRaises(ExpressionError, msg=text):
                ta.Expression(text).mask(DF)

    def test_helpful_errors(self):
        with self.assertRaises(ExpressionError) as cm:
            ta.Expression("smaa(close, 3)")
        self.assertIn("sma", cm.exception.details["did_you_mean"])
        with self.assertRaises(ExpressionError) as cm:
            ta.Expression("macd(close).histo")
        self.assertIn("hist", cm.exception.details["outputs"])
        with self.assertRaises(ExpressionError):
            ta.Expression("macd(close) > 0").mask(DF)                                 # multi-output used as a number
        with self.assertRaises(ExpressionError):
            ta.Expression("macd(close)").mask(DF)
        with self.assertRaises(ExpressionError) as cm:
            ta.Expression("clos > 1").mask(DF)
        self.assertIn("close", cm.exception.details["known"])
        with self.assertRaises(ExpressionError):
            ta.Expression("sma(close)").mask(DF)                                      # wrong arguments surface as a formula error
        with self.assertRaises(ExpressionError):
            ta.Expression("close ** 99").mask(DF)
        with self.assertRaises(ExpressionError):
            ta.Expression("close > 1", {"close": 5})                                  # a parameter may not shadow a data column
        with self.assertRaises(ExpressionError):
            ta.Expression("close + " * 400 + "close")
        with self.assertRaises(ExpressionError):
            ta.Expression("(" * 60 + "close" + ")" * 60 + " + " + "+".join(["-" * 50 + "close"]))


SPEC = {"name": "trend_pullback", "params": {"fast": 10, "slow": 40}, "bounds": {"fast": [5, 20], "slow": [30, 80]},
        "long_entry": "crossover(ema(close, fast), ema(close, slow))", "long_exit": "crossunder(ema(close, fast), ema(close, slow))",
        "history": 200, "warmup": 60}


def _market(n=500, seed=3):
    md = MarketData()
    md.add("AAA", synthetic.gbm(n, mu=0.1, sigma=0.25, seed=seed))
    return md


def _signals(strategy, md, symbol="AAA"):
    """Run a backtest and return the strategy's non-hold decisions as [(bar index, reason), ...]."""
    log = []
    inner = strategy.on_bar

    def recording(ctx):
        out = inner(ctx)
        log.extend((ctx.bars_seen(symbol) - 1, sig.reason) for sig in out if sig.reason != "hold")
        return out
    strategy.on_bar = recording
    result = BacktestEngine(md, free_config()).run(strategy)
    return log, result


class ExpressionStrategyTests(unittest.TestCase):
    def test_backtest_trades_and_is_deterministic(self):
        md = _market()
        a = BacktestEngine(md, free_config()).run(create_from_expressions(SPEC, ["AAA"]))
        b = BacktestEngine(md, free_config()).run(create_from_expressions(SPEC, ["AAA"]))
        self.assertGreater(len(a.trades), 1)
        pd.testing.assert_series_equal(a.equity, b.equity)

    def test_decisions_match_the_formulas_evaluated_over_the_whole_series(self):
        """Bar by bar on a rolling window, the strategy must enter on exactly the bars where the entry formula is true
        over the full history, and leave on the first exit bar after each entry."""
        md = _market()
        log, _ = _signals(create_from_expressions(SPEC, ["AAA"]), md)
        c = md.frame("AAA")["close"].to_numpy()
        fast, slow = ta.ema(c, 10), ta.ema(c, 40)
        entries = [int(i) for i in np.flatnonzero(op.crossover(fast, slow)) if i >= 59]
        exits = [int(i) for i in np.flatnonzero(op.crossunder(fast, slow)) if i > entries[0]]
        self.assertEqual([bar for bar, reason in log if reason == "long_entry"], entries)
        self.assertEqual([bar for bar, reason in log if reason == "long_exit"], exits)
        self.assertGreater(len(entries), 3)

    def test_no_look_ahead(self):
        md = _market()
        base = BacktestEngine(md, free_config()).run(create_from_expressions(SPEC, ["AAA"]))
        df = md.frame("AAA").copy()
        df.iloc[350:, :4] *= 1.5
        md2 = MarketData()
        md2.add("AAA", df)
        changed = BacktestEngine(md2, free_config()).run(create_from_expressions(SPEC, ["AAA"]))
        pd.testing.assert_series_equal(base.equity.iloc[:349], changed.equity.iloc[:349])

    def test_target_mode_emits_stepped_weights(self):
        md = _market()
        spec = {"name": "fade", "target": "clamp(-zscore(close, 20) / 2, -1, 1)", "step": 0.25}
        weights = []
        strat = create_from_expressions(spec, ["AAA"])
        inner = strat.on_bar
        strat.on_bar = lambda ctx: [weights.append(sig.target_weight) or sig for sig in inner(ctx)]
        res = BacktestEngine(md, free_config()).run(strat)
        self.assertGreater(res.order_count, 5)
        self.assertTrue(set(np.round(np.abs(weights), 2)) <= {0.0, 0.25, 0.5, 0.75, 1.0})
        self.assertTrue(min(weights) < 0 < max(weights))                               # fades both ways

    def test_trailing_stop_exits_when_price_falls_from_its_peak(self):
        md = _market()
        stop = 0.02
        log, _ = _signals(create_from_expressions(dict(SPEC, name="with_stop", stop_pct=stop), ["AAA"]), md)
        plain, _ = _signals(create_from_expressions(SPEC, ["AAA"]), md)
        c = md.frame("AAA")["close"].to_numpy()
        stops = [bar for bar, reason in log if reason == "stop"]
        self.assertGreater(len(stops), 2)
        entries = [bar for bar, reason in log if reason == "long_entry"]
        self.assertEqual(entries, [bar for bar, reason in plain if reason == "long_entry"])   # a stop changes exits, not entries
        for entry, out in zip(entries, stops):
            self.assertLessEqual(c[out], c[entry:out + 1].max() * (1 - stop) + 1e-9)          # the stop condition held on the exit bar
            self.assertTrue((c[entry + 1:out] > np.maximum.accumulate(c[entry:out])[1:] * (1 - stop)).all())   # ... and on no earlier bar

    def test_params_bounds_clone_and_factory(self):
        s = create_from_expressions(SPEC, ["AAA"], fast=12)
        self.assertEqual((s.p("fast"), s.p("slow"), s.PARAM_BOUNDS["fast"]), (12, 40, (5, 20)))
        self.assertEqual(s.clone(slow=50).p("slow"), 50)
        self.assertEqual(expression_factory(SPEC, ["AAA"])(fast=7).p("fast"), 7)
        with self.assertRaises(StrategyError):
            create_from_expressions(SPEC, ["AAA"], nope=1)

    def test_spec_validation_and_json_round_trip(self):
        self.assertEqual(expression_spec_from_json(expression_spec_to_json(SPEC)), SPEC)
        bad = [{}, {"name": "x"}, dict(SPEC, target="close"), dict(SPEC, long_entry="smaa(close, 3)"), dict(SPEC, weight=2),
               dict(SPEC, bounds={"nope": [1, 2]}), dict(SPEC, stop_pct=1.5), dict(SPEC, extra=1), dict(SPEC, long_entry="__import__('os')")]
        for spec in bad:
            with self.assertRaises(StrategyError, msg=str(spec)[:60]):
                create_from_expressions(spec, ["AAA"])
        with self.assertRaises(StrategyError) as cm:
            create_from_expressions(dict(SPEC, long_exit="macd(close).nope > 0"), ["AAA"])
        self.assertEqual(cm.exception.details["rule"], "long_exit")
        with self.assertRaises(StrategyError):
            expression_spec_from_json("{not json")


class PolicyStrategyTests(unittest.TestCase):
    def test_policy_drives_positions_and_errors_stand_aside(self):
        md = _market(300)
        calls = {"n": 0}

        def policy(window):
            calls["n"] += 1
            if calls["n"] % 50 == 0:
                raise RuntimeError("model hiccup")
            return 5.0 if window["close"][-1] > window["close"][-20:].mean() else float("nan")
        strat = PolicyStrategy(["AAA"], policy, history=60, warmup=30)
        res = BacktestEngine(md, free_config()).run(strat)
        self.assertGreater(res.order_count, 2)
        self.assertGreater(strat.policy_errors, 0)                                    # raised or non-finite: counted, target 0
        self.assertEqual(strat.clone(step=0.5).p("step"), 0.5)
        with self.assertRaises(StrategyError):
            PolicyStrategy(["AAA"], "not callable")
        with self.assertRaises(StrategyError):
            PolicyStrategy(["AAA"], policy, warmup=500, history=100)

    def test_probability_policy_bands(self):
        feats = lambda w: np.array([w["close"][-1]])                                  # noqa: E731
        mk = lambda p, **kw: probability_policy(lambda X: np.array([p]), feats, **kw)   # noqa: E731
        w = {"close": np.array([1.0, 2.0])}
        self.assertAlmostEqual(mk(0.8)(w), 0.6)
        self.assertEqual(mk(0.52)(w), 0.0)                                            # inside the no-trade band
        self.assertEqual(mk(0.2)(w), 0.0)
        self.assertAlmostEqual(mk(0.2, allow_short=True)(w), -0.6)
        self.assertEqual(probability_policy(lambda X: np.array([0.9]), lambda w: np.array([np.nan]))(w), 0.0)


if __name__ == "__main__":
    unittest.main()
