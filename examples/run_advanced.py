"""Tour of the advanced modules on synthetic data. Run:  python examples/run_advanced.py"""
import os
import sys
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from algoplatform.analytics.tca import tca_summary
from algoplatform.app.engine import TradingEngine
from algoplatform.backtest import BacktestEngine
from algoplatform.brokers import CostModel, PaperBroker
from algoplatform.brokers.fix import FixGateway
from algoplatform.brokers.fix.testing import FixAcceptor, LoopbackChannel
from algoplatform.core import EventBus, PlatformConfig, Side
from algoplatform.core.config import CostConfig
from algoplatform.data import MarketData, synthetic
from algoplatform.derivatives import binomial_price, iron_condor, profile, simulate_delta_hedge
from algoplatform.fundamentals import dcf, reverse_dcf, terminal_sensitivity
from algoplatform.optimize import Param, ParameterSpace, make_parallel_objective, random_search
from algoplatform.orders import OrderManager
from algoplatform.persistence import SqliteStore
from algoplatform.research import (PurgedKFold, daily_vol, probability_of_backtest_overfitting, sample_weights,
                                   triple_barrier)
from algoplatform.sentiment import EntityMapper, NewsStore, SentimentSignal
from algoplatform.strategy import MLStrategy, SMACrossover
from algoplatform.utils import SimClock


def banner(t):
    print(f"\n======== {t} ========")


def main():
    md = MarketData()
    for s, df in synthetic.universe(["AAA", "BBB"], n=700, seed=5).items():
        md.add(s, df)
    cfg = PlatformConfig(costs=CostConfig(impact_coef=2.0, commission_per_share=0.0, min_commission=0, slippage_bps=0, spread_bps=0))

    banner("1. Same strategy: single market orders vs a 5-bar TWAP algo")
    plain = BacktestEngine(md, cfg).run(SMACrossover(md.symbols, fast=10, slow=40))
    algo = BacktestEngine(md, cfg).run(SMACrossover(md.symbols, fast=10, slow=40), algo=("twap", {"bars": 5}))
    print("final equity   single orders: %.0f | TWAP: %.0f" % (plain.equity.iloc[-1], algo.equity.iloc[-1]))
    print("Note: these orders are tiny next to bar volume, so impact is negligible and slicing mostly just DELAYS entry (a cost for a")
    print("      trend strategy). Algos pay off when order size is a meaningful share of volume; they are a tool, not an upgrade.")
    print("TCA:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in tca_summary(algo.parents).items() if k != "by_algo"})

    banner("2. FIX 4.4 session (in-memory exchange)")
    ch = LoopbackChannel()
    gw = FixGateway(sender="CLIENT", target="EXCH", channel=ch)
    exch = FixAcceptor("EXCH", "CLIENT")
    exch.link(gw.session, ch)
    oms = OrderManager(EventBus())
    gw.set_event_handler(oms.on_broker_event)
    gw.connect()
    o = oms.create("AAPL", Side.BUY, 100)
    oms.mark_sent(o)
    gw.submit_order(o)
    exch.fill(o.client_order_id, 100, 187.25)
    print("order status:", o.status.value, "| avg fill:", o.avg_fill_price, "| session seq:", gw.session.store.next_send, "/", gw.session.store.next_expected)

    banner("3. Crash and recover from the SQLite journal")
    path = os.path.join(tempfile.mkdtemp(), "state.db")
    broker = PaperBroker(100_000, CostModel(cfg.costs))
    e1 = TradingEngine(cfg, broker, dict(md.instruments), SimClock(), store=SqliteStore(path))
    e1.add_strategy(SMACrossover(md.symbols, fast=10, slow=40))
    e1.start()
    for bars in md.bar_batches()[:200]:
        e1.on_bars(bars)
    e2 = TradingEngine(cfg, broker, dict(md.instruments), SimClock(), store=SqliteStore(path))       # a fresh process
    rep = e2.recover()
    print("recovered orders/fills:", rep.orders, rep.fills, "| clean:", rep.clean,
          "| cash matches:", abs(e2.portfolio.cash - e1.portfolio.cash) < 1e-6)

    banner("4. Research hygiene: purged CV, sample weights, probability of backtest overfitting")
    close = md.frame("AAA")["close"]
    lab = triple_barrier(close, close.index[60:-10], daily_vol(close) * 1.0, horizon=5)
    w = sample_weights(lab["t1"], close.index)
    cv = PurgedKFold(5, lab["t1"], embargo_pct=0.01)
    print("labels:", len(lab), "| avg uniqueness weight range: %.2f-%.2f" % (w.min(), w.max()), "| folds:", sum(1 for _ in cv.split()))
    rets = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (400, 20)))                         # 20 tried configs, pure noise
    print("PBO of 'best of 20 noise strategies': %.2f (about 0.5 = a coin flip)" % probability_of_backtest_overfitting(rets)["pbo"])
    ml = BacktestEngine(md, PlatformConfig()).run(MLStrategy(["AAA"]))
    print("MLStrategy trades:", len(ml.trades), "| sharpe %.2f" % ml.metrics()["sharpe"])

    banner("5. Valuation: why a DCF answer is fragile")
    base = dcf(100, 0.06, 0.09, 0.025)
    print("value/share %.0f, %.0f%% of it is the terminal assumption" % (base["per_share"], 100 * base["terminal_share"]))
    grid = terminal_sensitivity(100, 0.06, [0.08, 0.09, 0.10], [0.015, 0.025, 0.035])
    print("range across plausible assumptions: %.0f to %.0f" % (grid.min().min(), grid.max().max()))
    print("growth implied by a price of 2000:", round(reverse_dcf(2000, 100, 0.09, 0.025), 3))

    banner("6. News sentiment (point-in-time, de-duplicated)")
    store = NewsStore(EntityMapper({"ACME": ["Acme"]}), default_latency=timedelta(minutes=1), source_weights={"wire": 1.0})
    t0 = datetime(2024, 3, 1, 9, 0)
    store.add(t0, "wire", "Acme beats earnings estimates, raises guidance")
    store.add(t0 + timedelta(minutes=5), "blog", "ACME beats earnings estimates, raises guidance")     # same story again
    sig = SentimentSignal(store, half_life=timedelta(hours=12))
    print("duplicates dropped:", store.duplicates, "| score now: %+.2f | after 24h: %+.2f" % (sig.score("ACME", t0 + timedelta(minutes=2)), sig.score("ACME", t0 + timedelta(days=1))))

    banner("7. Options: a defined-risk structure and what hedging frequency costs")
    ic = iron_condor(85, 90, 110, 115, 100, 0.25, 0.03, 0.2)
    p = profile(ic, 60, 140)
    print("iron condor: max profit $%.0f, max loss $%.0f, breakevens %s" % (p["max_profit"], p["max_loss"], [round(b, 1) for b in p["breakevens"]]))
    print("American vs European put:", round(binomial_price(100, 110, 1, 0.05, 0.2, "put", american=True), 3), ">", round(binomial_price(100, 110, 1, 0.05, 0.2, "put"), 3))
    z = np.random.default_rng(0).standard_normal((150, 40))
    paths = 100 * np.exp(np.concatenate([np.zeros((150, 1)), np.cumsum(-0.0002 + 0.02 * z * 0.5, axis=1)], axis=1))
    for every in (1, 10):
        h = simulate_delta_hedge(paths, 100, 0.25, 0.03, 0.2, rebalance_every=every, cost_bps=5)
        print("hedge every %2d bars: P&L std %.0f, mean cost %.0f" % (every, h["pnl"].std(), h["cost"].mean()))

    banner("8. Parallel optimisation (identical results to serial)")
    obj = make_parallel_objective(md, "sma_crossover", md.symbols, PlatformConfig(), min_trades=3)
    space = ParameterSpace([Param("fast", "int", 3, 15), Param("slow", "int", 16, 60)])
    serial, par = random_search(obj, space, 8, seed=1), random_search(obj, space, 8, seed=1, n_jobs=2)
    print("best:", par.best_params, "| trial tables identical:", serial.trials.equals(par.trials))


if __name__ == "__main__":
    main()
