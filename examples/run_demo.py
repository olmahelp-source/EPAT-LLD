"""End-to-end tour of the platform on synthetic data. Run:  python examples/run_demo.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from algoplatform.analytics import html_report, text_report
from algoplatform.app import TradingPlatform
from algoplatform.core import PlatformConfig, Role, Side
from algoplatform.core.models import Bar
from algoplatform.data import MarketData, synthetic
from algoplatform.optimize import Param, ParameterSpace, walk_forward, backtest_objective, random_search
from algoplatform.backtest import BacktestEngine
from algoplatform.portfolio import FundAllocator, risk_parity, risk_contributions
from algoplatform.stats import adf_test, deflated_sharpe, engle_granger, hurst_exponent
from algoplatform.strategy import SMACrossover, StrategyBuilder

OUT = os.path.join(os.path.dirname(__file__), "demo_report.html")


def banner(t):
    print(f"\n{'=' * 8} {t} {'=' * 8}")


def main():
    plat = TradingPlatform(PlatformConfig(log_level="ERROR"))
    admin = plat.users.register("admin_user", "Str0ng-Passw0rd!", Role.ADMIN)
    trader = plat.users.register("trader_one", "Str0ng-Passw0rd!")
    risk = plat.users.register("risk_desk", "Str0ng-Passw0rd!", Role.RISK_MANAGER, actor=admin)

    md = MarketData()
    for sym, df in synthetic.universe(["AAA", "BBB", "CCC", "DDD"], n=1000, seed=11).items():
        md.add(sym, df)

    banner("1. Backtest a built-in strategy")
    res = plat.backtest(trader, "sma_crossover", md, params={"fast": 10, "slow": 50})
    rep = res.report()
    print(text_report(rep, "sma_crossover 10/50")[:900], "...")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html_report(res.equity, rep, "sma_crossover 10/50"))
    print(f"\nHTML tear sheet: {OUT}")

    banner("2. Custom strategy from a JSON-able spec (no code)")
    spec = (StrategyBuilder("rsi_trend").param("n", 14).param("lvl", 35)
            .indicator("rsi", "rsi", period="$n").indicator("trend", "sma", period=100).indicator("atr", "atr", period=14)
            .long_entry(("rsi", "<", "$lvl"), ("close", ">", "trend")).long_exit(("rsi", ">", 60))
            .stop_atr("atr", 2.5).build())
    r2 = plat.backtest(trader, spec, md)
    print("trades:", len(r2.trades), "| return: %.1f%%" % (100 * r2.metrics()["total_return"]),
          "| sharpe: %.2f" % r2.metrics()["sharpe"])

    banner("3. Optimise + guard against overfitting")
    space = ParameterSpace([Param("fast", "int", 3, 30), Param("slow", "int", 31, 150)],
                           constraint=lambda p: p["fast"] < p["slow"])
    eng = BacktestEngine(md, plat.config)
    obj = backtest_objective(eng, lambda **p: SMACrossover(md.symbols, **p), "sharpe", min_trades=5)
    opt = random_search(obj, space, n_iter=25, seed=1)
    print("best:", opt.best_params, "sharpe %.2f" % opt.best_score, "| plateau score %.2f (1 = robust)" % opt.plateau_score(space))
    best = BacktestEngine(md, plat.config).run(SMACrossover(md.symbols, **opt.best_params))
    dsr = deflated_sharpe(best.returns.values, opt.n_trials, opt.summary()["score_variance_for_dsr"] / 252)
    print("deflated Sharpe prob after %d trials: %.2f" % (opt.n_trials, dsr["deflated_sharpe"]))

    banner("4. Walk-forward (out-of-sample)")
    wf = walk_forward(eng, lambda **p: SMACrossover(md.symbols, **p), space, train_bars=400, test_bars=150, n_iter=10)
    print(wf.folds[["test_start", "train_score", "test_score", "fast", "slow"]].to_string(index=False))
    print("walk-forward efficiency: %.2f | OOS sharpe: %.2f" % (wf.efficiency, wf.oos_metrics["sharpe"]))

    banner("5. Portfolio construction and fund allocation")
    rets = md.returns().dropna()
    cov = rets.cov().values
    w = risk_parity(cov)
    print("risk-parity weights:", {k: round(float(v), 3) for k, v in zip(md.symbols, w)}, "| contributions:", [round(float(v), 3) for v in risk_contributions(w, cov)])
    print("capital by strategy:", FundAllocator(100_000).allocate({"trend": 0.6, "reversion": 0.4}))

    banner("6. Statistical tests")
    y, x = synthetic.cointegrated_pair(n=800, seed=3)
    eg = engle_granger(y.close.values, x.close.values)
    print("Engle-Granger cointegrated:", eg["cointegrated_5pct"], "| hedge ratio %.2f | half-life %.1f bars" % (eg["hedge_ratio"], eg["half_life"]))
    print("ADF on a price series rejects unit root:", adf_test(md.frame("AAA").close.values)["reject_5pct"],
          "| Hurst %.2f" % hurst_exponent(md.frame("AAA").close.values))

    banner("7. Paper trading session with risk gate and kill switch")
    acc = plat.accounts.link(trader, "paper", {})
    session = plat.open_session(trader, acc.account_id, [])
    session.start()
    ts = md.timeline()
    session.on_bars({"AAA": Bar(ts[0], "AAA", 100, 101, 99, 100, 1e6)})
    ok = plat.place_manual_order(trader, session, "AAA", Side.BUY, 100)
    bad = plat.place_manual_order(trader, session, "AAA", Side.BUY, 10**7)
    print("normal order ->", ok.status.value, "| oversized order ->", bad.status.value, "|", bad.reject_reason[:60])
    plat.trip_kill_switch(trader, session, "demo halt")
    blocked = plat.place_manual_order(trader, session, "AAA", Side.BUY, 1)
    print("after kill switch ->", blocked.status.value, "|", blocked.reject_reason[:50])
    plat.reset_kill_switch(risk, session, "demo finished, cleared by risk desk")
    print("audit entries recorded:", len(plat.audit))


if __name__ == "__main__":
    main()
