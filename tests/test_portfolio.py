import unittest
from datetime import datetime

import numpy as np
import pandas as pd

import helpers  # noqa: F401
from algoplatform.core import Fill, Instrument, Side
from algoplatform.core.exceptions import OptimizationError, ValidationError
from algoplatform.portfolio import (FixedFractional, FixedQuantity, FundAllocator, KellySizer, Portfolio,
                                    PositionSizeManager, RiskPerTrade, SizingContext, VolatilityTarget,
                                    apply_weight_bounds, black_litterman, drawdown_scalar, inverse_volatility,
                                    kelly_fraction, kelly_gaussian, max_sharpe, mean_variance, min_variance,
                                    optimal_f, project_capped_simplex, risk_contributions, risk_parity)
from algoplatform.core.config import SizingConfig

_n = [0]


def fill(side, qty, price, comm=0.0, sym="X", ts=None, strat="s"):
    _n[0] += 1
    return Fill(f"f{_n[0]}", "o", sym, side, qty, price, ts or datetime(2024, 1, _n[0] % 27 + 1), comm, strat)


class PortfolioTests(unittest.TestCase):
    def test_round_trip_pnl_with_commissions(self):
        p = Portfolio(100_000)
        p.apply_fill(fill(Side.BUY, 100, 10.0, 1.0))
        p.apply_fill(fill(Side.SELL, 100, 12.0, 1.0))
        self.assertAlmostEqual(p.cash, 100_000 - 1000 - 1 + 1200 - 1)
        self.assertAlmostEqual(p.realized_pnl, 198.0)
        self.assertEqual(len(p.trades), 1)
        self.assertAlmostEqual(p.trades[0].pnl, 198.0)
        self.assertTrue(p.position("X").is_flat)
        self.assertAlmostEqual(p.total_pnl, 198.0)

    def test_average_cost_and_unrealized(self):
        p = Portfolio(100_000)
        p.apply_fill(fill(Side.BUY, 100, 10.0))
        p.apply_fill(fill(Side.BUY, 100, 20.0))
        self.assertAlmostEqual(p.position("X").avg_price, 15.0)
        p.mark({"X": 18.0})
        self.assertAlmostEqual(p.unrealized_pnl, 600.0)
        self.assertAlmostEqual(p.equity, 100_000 + 600.0)

    def test_partial_close_keeps_average(self):
        p = Portfolio(100_000)
        p.apply_fill(fill(Side.BUY, 100, 10.0))
        p.apply_fill(fill(Side.SELL, 40, 11.0))
        pos = p.position("X")
        self.assertEqual(pos.quantity, 60)
        self.assertAlmostEqual(pos.avg_price, 10.0)
        self.assertAlmostEqual(p.trades[0].pnl, 40.0)

    def test_short_position_pnl(self):
        p = Portfolio(100_000)
        p.apply_fill(fill(Side.SELL, 50, 20.0))
        p.mark({"X": 18.0})
        self.assertAlmostEqual(p.unrealized_pnl, 100.0)
        p.apply_fill(fill(Side.BUY, 50, 18.0))
        self.assertAlmostEqual(p.trades[0].pnl, 100.0)
        self.assertEqual(p.trades[0].direction, -1)

    def test_flip_long_to_short(self):
        p = Portfolio(100_000)
        p.apply_fill(fill(Side.BUY, 100, 10.0))
        p.apply_fill(fill(Side.SELL, 150, 12.0))
        pos = p.position("X")
        self.assertEqual(pos.quantity, -50)
        self.assertAlmostEqual(pos.avg_price, 12.0)                # new short opens at the flip price
        self.assertAlmostEqual(p.trades[0].pnl, 200.0)

    def test_equity_always_equals_initial_plus_realized_plus_unrealized(self):
        p = Portfolio(50_000)
        rng = np.random.default_rng(4)
        for i in range(60):
            side = Side.BUY if rng.random() < 0.5 else Side.SELL
            px = 100 + rng.normal(0, 3)
            p.apply_fill(fill(side, int(rng.integers(1, 50)), px, 0.5))
            p.mark({"X": 100 + rng.normal(0, 3)})
            self.assertAlmostEqual(p.equity, 50_000 + p.realized_pnl + p.unrealized_pnl, places=6)

    def test_multiplier_and_mae_mfe(self):
        p = Portfolio(1_000_000, {"ES": Instrument("ES", multiplier=50.0)})
        p.apply_fill(fill(Side.BUY, 2, 4000.0, sym="ES"))
        p.mark({"ES": 3950.0})
        p.mark({"ES": 4100.0})
        p.apply_fill(fill(Side.SELL, 2, 4050.0, sym="ES"))
        t = p.trades[0]
        self.assertAlmostEqual(t.pnl, 2 * 50 * 50.0)
        self.assertAlmostEqual(t.mae, 50.0)
        self.assertAlmostEqual(t.mfe, 100.0)

    def test_reconcile_and_equity_curve(self):
        p = Portfolio(10_000)
        p.apply_fill(fill(Side.BUY, 10, 5.0))
        self.assertEqual(p.reconcile({"X": 10}), {})
        self.assertEqual(p.reconcile({"X": 9, "Y": 1}), {"X": (10, 9), "Y": (0, 1)})
        p.mark({"X": 5}, ts=pd.Timestamp("2024-01-02"))
        p.mark({"X": 6}, ts=pd.Timestamp("2024-01-03"))
        self.assertEqual(len(p.equity_curve()), 2)

    def test_strategy_attribution(self):
        p = Portfolio(10_000)
        p.apply_fill(fill(Side.BUY, 10, 5.0, strat="a"))
        p.apply_fill(fill(Side.SELL, 10, 6.0, strat="a"))
        self.assertAlmostEqual(p.realized_by_strategy["a"], 10.0)

    def test_invalid_capital(self):
        with self.assertRaises(ValidationError):
            Portfolio(0)


class SizingTests(unittest.TestCase):
    def ctx(self, **kw):
        base = dict(symbol="X", equity=100_000, price=50.0, volatility=0.01, atr=1.0)
        base.update(kw)
        return SizingContext(**base)

    def test_fixed_fractional_and_quantity(self):
        self.assertAlmostEqual(FixedFractional(0.1).base_quantity(self.ctx()), 200)
        self.assertEqual(FixedQuantity(7).base_quantity(self.ctx()), 7)
        with self.assertRaises(ValidationError):
            FixedFractional(1.5)

    def test_volatility_target_scales_inversely(self):
        lo = VolatilityTarget(0.10).base_quantity(self.ctx(volatility=0.005))
        hi = VolatilityTarget(0.10).base_quantity(self.ctx(volatility=0.02))
        self.assertGreater(lo, hi)
        self.assertEqual(VolatilityTarget().base_quantity(self.ctx(volatility=None)), 0.0)

    def test_risk_per_trade(self):
        q = RiskPerTrade(0.01, atr_multiple=2).base_quantity(self.ctx(atr=2.0))     # risk $1000 / $4 stop
        self.assertAlmostEqual(q, 250)
        self.assertEqual(RiskPerTrade().base_quantity(self.ctx(atr=None)), 0.0)

    def test_kelly(self):
        self.assertAlmostEqual(kelly_fraction(0.6, 1.0), 0.2)
        self.assertEqual(kelly_fraction(0.4, 1.0), 0.0)                              # negative edge => no bet
        self.assertAlmostEqual(kelly_gaussian(0.001, 0.0004), 2.5)
        q = KellySizer(0.5).base_quantity(self.ctx(win_rate=0.6, payoff_ratio=1.0))
        self.assertAlmostEqual(q, 100_000 * 0.1 / 50)

    def test_optimal_f_positive_for_positive_edge(self):
        rng = np.random.default_rng(0)
        trades = np.where(rng.random(300) < 0.65, 1.0, -1.0) * rng.uniform(0.5, 1.5, 300)
        f = optimal_f(trades)
        self.assertTrue(0.0 < f < 1.0)
        self.assertEqual(optimal_f([1, 2, 3]), 0.0)

    def test_manager_caps_and_rounding(self):
        cfg = SizingConfig(max_position_pct=0.10, cash_buffer_pct=0.0)
        m = PositionSizeManager(FixedFractional(0.5), cfg)
        q = m.target_quantity(1.0, self.ctx())                                       # wants 1000, cap 200
        self.assertEqual(q, 200)
        self.assertEqual(m.target_quantity(-1.0, self.ctx()), -200)
        self.assertEqual(m.target_quantity(0.0, self.ctx()), 0.0)
        lot = self.ctx(instrument=Instrument("X", lot_size=30))
        self.assertEqual(m.target_quantity(1.0, lot), 180)                           # rounded down to lots

    def test_manager_respects_short_ban_cash_and_adv(self):
        cfg = SizingConfig(max_position_pct=1.0, cash_buffer_pct=0.0)
        m = PositionSizeManager(FixedFractional(1.0), cfg, allow_short=False)
        self.assertEqual(m.target_quantity(-1.0, self.ctx()), 0.0)
        self.assertEqual(m.target_quantity(1.0, self.ctx(available_cash=1000)), 20)
        self.assertEqual(PositionSizeManager(FixedFractional(1.0), cfg).target_quantity(1.0, self.ctx(adv=1000)), 100)
        self.assertEqual(m.target_quantity(1.0, self.ctx(), scale=0.5), 1000)

    def test_drawdown_scalar(self):
        self.assertEqual(drawdown_scalar(-0.02), 1.0)
        self.assertEqual(drawdown_scalar(-0.30), 0.0)
        self.assertAlmostEqual(drawdown_scalar(-0.125, start=0.05, floor_at=0.20), 0.5)


def make_cov():
    vols = np.array([0.10, 0.20, 0.30])
    corr = np.array([[1, 0.3, 0.1], [0.3, 1, 0.4], [0.1, 0.4, 1]])
    return corr * np.outer(vols, vols)


class OptimizerTests(unittest.TestCase):
    def test_projection_properties(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            w = project_capped_simplex(rng.normal(size=6), cap=0.3)
            self.assertAlmostEqual(w.sum(), 1.0, places=9)
            self.assertTrue((w >= -1e-12).all() and (w <= 0.3 + 1e-9).all())
        with self.assertRaises(OptimizationError):
            project_capped_simplex(np.ones(3), cap=0.2)

    def test_min_variance_beats_equal_weight(self):
        cov = make_cov()
        w = min_variance(cov)
        ew = np.ones(3) / 3
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertLess(w @ cov @ w, ew @ cov @ ew)
        self.assertGreater(w[0], w[2])                                               # least volatile asset dominates

    def test_cap_constraint_binds(self):
        w = min_variance(make_cov(), cap=0.5)
        self.assertLessEqual(w.max(), 0.5 + 1e-9)

    def test_risk_parity_equalises_contributions(self):
        cov = make_cov()
        w = risk_parity(cov)
        rc = risk_contributions(w, cov)
        self.assertTrue(np.allclose(rc, 1 / 3, atol=1e-4))
        iv = inverse_volatility(cov)
        self.assertAlmostEqual(iv.sum(), 1.0)

    def test_risk_budget(self):
        cov = make_cov()
        rc = risk_contributions(risk_parity(cov, budget=np.array([0.5, 0.25, 0.25])), cov)
        self.assertTrue(np.allclose(rc, [0.5, 0.25, 0.25], atol=1e-4))

    def test_mean_variance_and_max_sharpe_prefer_high_return(self):
        cov = make_cov()
        mu = np.array([0.02, 0.10, 0.03])
        w = mean_variance(mu, cov, risk_aversion=2.0)
        self.assertEqual(int(np.argmax(w)), 1)
        ms = max_sharpe(mu, cov)
        sr = lambda x: (x @ mu) / np.sqrt(x @ cov @ x)
        self.assertGreaterEqual(sr(ms), sr(np.ones(3) / 3) - 1e-9)

    def test_black_litterman_moves_towards_view(self):
        cov = make_cov()
        mkt = np.array([0.4, 0.4, 0.2])
        base = 2.5 * cov @ mkt
        post = black_litterman(cov, mkt, np.array([[1, 0, 0]]), np.array([base[0] + 0.05]))
        self.assertGreater(post[0], base[0])


class AllocatorTests(unittest.TestCase):
    def returns(self):
        rng = np.random.default_rng(1)
        return pd.DataFrame({"a": rng.normal(0.001, 0.005, 300), "b": rng.normal(0.001, 0.02, 300),
                             "c": rng.normal(0.0, 0.01, 300)})

    def test_methods_sum_to_one_and_respect_bounds(self):
        fa = FundAllocator(100_000)
        for m in ("equal", "inverse_vol", "risk_parity", "min_variance", "performance"):
            w = fa.weights(self.returns(), method=m, min_weight=0.1, max_weight=0.6)
            self.assertAlmostEqual(sum(w.values()), 1.0)
            self.assertTrue(all(0.1 - 1e-9 <= v <= 0.6 + 1e-9 for v in w.values()), m)

    def test_inverse_vol_favours_calm_strategy(self):
        w = FundAllocator(1).weights(self.returns(), method="inverse_vol")
        self.assertGreater(w["a"], w["b"])

    def test_allocate_ledger_and_reserve(self):
        fa = FundAllocator(100_000, reserve_pct=0.1)
        out = fa.allocate({"a": 0.5, "b": 0.5})
        self.assertAlmostEqual(out["a"], 45_000)
        fa.update_total(200_000)
        self.assertAlmostEqual(fa.capital_for("a"), 90_000)
        self.assertTrue(fa.needs_rebalance({"a": 80, "b": 20}))
        self.assertFalse(fa.needs_rebalance({"a": 51, "b": 49}))
        with self.assertRaises(ValidationError):
            fa.allocate({"a": 0.7, "b": 0.7})

    def test_infeasible_bounds_and_short_history(self):
        with self.assertRaises(ValidationError):
            apply_weight_bounds(np.array([0.5, 0.5]), 0.0, 0.4)
        with self.assertRaises(ValidationError):
            FundAllocator(1).weights(self.returns().head(5), method="equal")


if __name__ == "__main__":
    unittest.main()
