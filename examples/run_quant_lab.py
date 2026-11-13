"""Tour of the quant-research, learning and control layers on synthetic data. Run:  python examples/run_quant_lab.py

Everything here runs on generated data, so the numbers show that the machinery works, not that any strategy makes money.
"""
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from algoplatform import technical as ta
from algoplatform.ai import RegimeDetector, SymbolicAlpha, bocpd, changepoints_from_run_length
from algoplatform.analytics import bootstrap_metrics, evaluate_backtest, evaluate_strategy
from algoplatform.app import TradingPlatform
from algoplatform.backtest import BacktestEngine
from algoplatform.core import PlatformConfig, Role, Severity
from algoplatform.core.exceptions import AuthorizationError, GovernanceError, KillSwitchActive, PlatformError
from algoplatform.data import MarketData, synthetic
from algoplatform.derivatives import synthetic_chain
from algoplatform.dl import LSTMForecaster
from algoplatform.instruments import ContractSpec, CorporateAction, InstrumentMaster, adjust_ohlcv, future_spec, option_spec
from algoplatform.ml import GradientBoosting, forward_returns, information_coefficient, roc_auc, walk_forward_predict
from algoplatform.notify import MemoryChannel, drawdown_alert
from algoplatform.optimize import Param, ParameterSpace, backtest_objective, optimize
from algoplatform.relationships import RelationshipManager, find_cointegrated_pairs
from algoplatform.rl import DQNAgent, TradingEnv, evaluate, standardize, train
from algoplatform.selection import UniverseSelector
from algoplatform.strategy import SMACrossover, create_from_expressions
from algoplatform.users import PasswordHasher
from algoplatform.utils import ErrorBoundary, ErrorBudget, attempt, classify, first_success, partition
from algoplatform.volatility import EWMAVol, GarchVol, HARVol, VolatilityManager, compare_vol_models, yang_zhang

logging.disable(logging.CRITICAL)                   # the tour triggers failures on purpose; keep the output readable
T0 = time.time()


def section(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(4, 96 - len(title)))


# ---------------------------------------------------------------------------------------------------------------------
section("1. Error handling: classify, contain, fall back")
for exc in (ConnectionError("feed down"), KillSwitchActive("daily loss"), ValueError("bad row")):
    info = classify(exc)
    print(f"  {type(exc).__name__:<18} -> category={info.category:<8} severity={info.severity.value:<8} action={info.action.value}")
price = first_success(lambda: 1 / 0, lambda: {"AAA": 101.2}["AAA"])
print(f"  fallback chain: primary source failed, backup returned {price}")
values, errors = partition([attempt(float, raw) for raw in ("1.5", "n/a", "2.5")])
print(f"  batch of 3 parsed without stopping: {values} ok, {len(errors)} failed")
tripped = []
boundary = ErrorBoundary("strategy", budget=ErrorBudget(3, 60), on_exhausted=lambda name, info: tripped.append(name))
for _ in range(5):
    boundary.run(lambda: 1 / 0, default=0.0)
print(f"  boundary contained {boundary.errors} errors and escalated {len(tripped)} time (after the third)")

# ---------------------------------------------------------------------------------------------------------------------
section("2. Technical analysis: catalogue, patterns, formulas")
df = synthetic.gbm(700, mu=0.10, sigma=0.25, seed=11)
kinds = pd.Series([s.kind for s in ta.CATALOG.values()]).value_counts().to_dict()
print(f"  function catalogue: {kinds}")
st = ta.compute("supertrend", df, n=10, mult=3.0)
print(f"  supertrend direction on the last bar: {st['direction'][-1]:+.0f};  RSI(14) = {ta.compute('rsi', df)[-1]:.1f}")
candles = ta.scan_candles(df)
print(f"  candlestick patterns found: {int((candles.drop(columns='score') != 0).sum().sum())} flags on {len(df)} bars")
patterns = ta.scan_chart_patterns(df)
print("  chart patterns:", ", ".join(sorted({m.name for m in patterns})) or "none")
levels = ta.support_resistance(df["high"], df["low"], df["close"])[:2]
print("  strongest levels:", [(round(lv["price"], 2), lv["kind"], lv["touches"]) for lv in levels])
formula = ta.Expression("crossover(ema(close, fast), ema(close, slow)) and rsi(close, 14) < 70", {"fast": 10, "slow": 40})
print(f"  formula true on {int(formula.mask(df).sum())} bars; reads columns {sorted(formula.columns)}")
try:
    ta.Expression("__import__('os').system('rm -rf /')")
except PlatformError as exc:
    print(f"  a hostile formula is refused: {exc.message}")

# ---------------------------------------------------------------------------------------------------------------------
section("3. Formula strategy -> backtest -> scorecard")
md = MarketData()
md.add("AAA", df)
spec = {"name": "trend_pullback", "params": {"fast": 10, "slow": 40}, "bounds": {"fast": [5, 20], "slow": [30, 90]},
        "long_entry": "crossover(ema(close, fast), ema(close, slow))", "long_exit": "crossunder(ema(close, fast), ema(close, slow))",
        "stop_pct": 0.08}
result = BacktestEngine(md, PlatformConfig()).run(create_from_expressions(spec, ["AAA"]))
print(f"  backtest: Sharpe {result.metrics()['sharpe']:.2f}, {len(result.trades)} trades")
card = evaluate_backtest(result, n_trials=1)
print(f"  scorecard: grade {card.grade}, passed={card.passed}")
for failure in card.failures[:3]:
    print(f"    - {failure}")

# ---------------------------------------------------------------------------------------------------------------------
section("4. Parameter search, and why the number of trials matters")
space = ParameterSpace([Param("fast", "int", 3, 20), Param("slow", "int", 25, 90)])
objective = backtest_objective(BacktestEngine(md, PlatformConfig()), lambda **p: SMACrossover(["AAA"], **p), min_trades=2)
for method, kwargs in (("random", {"n_iter": 20}), ("bayesian", {"n_iter": 20}), ("tpe", {"n_iter": 20})):
    res = optimize(objective, space, method, **kwargs)
    print(f"  {method:<9} best Sharpe {res.best_score:5.2f} with {res.best_params} after {res.n_trials} backtests")
best = BacktestEngine(md, PlatformConfig()).run(SMACrossover(["AAA"], **res.best_params))
naive, honest = evaluate_backtest(best, n_trials=1), evaluate_backtest(best, n_trials=60)
print(f"  skill probability of the best result: {naive.metrics['skill_probability']:.2f} if it were the only thing tried, "
      f"{honest.metrics['skill_probability']:.2f} after counting the 60 backtests")
ci = bootstrap_metrics(best.returns, n_boot=300)["sharpe"]
print(f"  90% interval for its Sharpe: {ci['low']:.2f} to {ci['high']:.2f}")

# ---------------------------------------------------------------------------------------------------------------------
section("5. Instruments, corporate actions, option chain")
master = InstrumentMaster([ContractSpec("AAPL", name="Apple", sector="Tech", broker_symbols={"zerodha": "AAPL-EQ"}),
                           future_spec("ES", "2025-03-21", multiplier=50, tick_size=0.25), option_spec("AAPL", "2025-01-17", "call", 150)])
report = master.load_records([{"symbol": "MSFT", "sector": "Tech"}, {"symbol": "BAD", "tick_size": "-1"}], on_error="skip")
print(f"  master holds {len(master)} instruments; bulk load: {report['loaded']} loaded, {len(report['rejected'])} rejected with a reason")
print(f"  zerodha calls AAPL '{master.broker_symbol('AAPL', 'zerodha')}'; front ES future on 2025-01-02: {master.front_future('ES', '2025-01-02').symbol}")
raw = df.iloc[:10].copy()
raw.loc[raw.index[5]:, ["open", "high", "low", "close"]] /= 2
adjusted = adjust_ohlcv(raw, [CorporateAction("AAA", raw.index[5], "split", 2.0)])
print(f"  2-for-1 split: raw one-bar return {raw['close'].pct_change().iloc[5]:+.1%}, adjusted {adjusted['close'].pct_change().iloc[5]:+.1%}")
chain = synthetic_chain(spot=100.0, seed=1)
s = chain.summary()
print(f"  option chain: ATM IV {s['atm_iv']:.1%}, put/call ratio {s['put_call_ratio']:.2f}, max pain {s['max_pain']:.0f}, "
      f"25-delta risk reversal {s['skew_risk_reversal']:+.3f}, expected move {s['expected_move_pct']:.1%}")

# ---------------------------------------------------------------------------------------------------------------------
section("6. Selection and relationships")
universe = MarketData()
frames = synthetic.universe([f"S{i}" for i in range(10)], n=600, seed=5)
for sym, frame in frames.items():
    universe.add(sym, frame)
asof = frames["S0"].index[450]
picked = (UniverseSelector().require_history(252).min_dollar_volume(1e6).rank_by("momentum", lookback=126)
          .rank_by("low_volatility", weight=0.5).diversify(0.9).top(4).select(universe, asof=asof))
print(f"  selected as of {asof.date()}: {picked.symbols}  ({len(picked.rejected)} rejected, each with a reason)")
pair_y, pair_x = synthetic.cointegrated_pair(600, seed=2)
prices = universe.closes().assign(Y=pair_y["close"].to_numpy(), X=pair_x["close"].to_numpy())
relations = RelationshipManager()
print(f"  relationships found: {relations.build(prices, corr_threshold=0.5)}")
scan = find_cointegrated_pairs(prices)
print(f"    (of {scan.attrs['tested']} pairs tested for cointegration, about {scan.attrs['expected_false_positives']:.0f} would pass by chance: "
      f"candidates to validate, not discoveries)")
book = {sym: 0.25 for sym in picked.symbols}
conc = relations.concentration(book)
print(f"  {conc['positions']} equal positions are {conc['effective_bets']:.1f} independent bets once correlated names are grouped")

# ---------------------------------------------------------------------------------------------------------------------
section("7. Volatility: estimate, forecast, manage")
rets = df["close"].pct_change().dropna().to_numpy()
yz = yang_zhang(df["open"], df["high"], df["low"], df["close"], 20)[-1]
print(f"  Yang-Zhang 20-bar volatility: {yz:.1%}")
league = compare_vol_models(rets, {"ewma": EWMAVol, "har": HARVol, "garch": GarchVol}, train=400, step=60, horizon=5)
print("  out-of-sample league (lower QLIKE is better): " + ", ".join(f"{r.model} {r.qlike:.3f}" for r in league.itertuples()))
view = VolatilityManager(target_vol=0.10, model="ewma").assess(rets)
print(f"  forecast {view['forecast_vol']:.1%}, regime '{view['regime']}' -> run {view['scalar']:.2f}x exposure for a 10% volatility target")

# ---------------------------------------------------------------------------------------------------------------------
section("8. Learning: walk-forward ML, LSTM, reinforcement learning, regimes, formula search")
rng = np.random.default_rng(1)
n = 2400
signal = rng.normal(size=n)
ret = np.zeros(n)
ret[1:] = 0.004 * np.sign(signal[:-1]) + 0.004 * rng.normal(size=n - 1)               # a planted, learnable edge
close = 100 * np.exp(np.cumsum(ret))
features = np.column_stack([signal, rng.normal(size=n)])
label = (forward_returns(close, 1) > 0).astype(float)
label[-1] = np.nan
wf = walk_forward_predict(lambda: GradientBoosting(n_estimators=40), features, label, train_size=600, step=300, embargo=1)
ok = ~np.isnan(wf["pred"]) & ~np.isnan(label)
print(f"  gradient boosting, walk-forward ({wf['fits']} refits): out-of-sample AUC {roc_auc(label[ok], wf['pred'][ok]):.2f}")
lstm = LSTMForecaster(lookback=5, hidden=8, epochs=25).fit(features[:1500], forward_returns(close, 1)[:1500])
pred = lstm.predict(features)
print(f"  LSTM forecaster: out-of-sample information coefficient {information_coefficient(pred[1500:-1], forward_returns(close, 1)[1500:-1]):.2f}")
scaled = standardize(features[:1500], features)
agent = DQNAgent(3, 3, hidden=(16,), epsilon_decay=0.9)
train(agent, TradingEnv(ret[:1500], scaled[:1500], cost_bps=1, episode_length=250, seed=0), 12)
oos = evaluate(agent, TradingEnv(ret[1500:], scaled[1500:], cost_bps=1))
print(f"  deep Q-network after 12 episodes: out-of-sample Sharpe {oos['sharpe']:.1f} over {oos['steps']} bars, {oos['trades']} trades (edge is planted)")
regime_data, true_states = synthetic.regime_switching(1500, seed=3)
r = regime_data["close"].pct_change().dropna().to_numpy()
detector = RegimeDetector(2).fit(r[:1000])
calm, wild = detector.describe()
hit = (detector.regimes(r)[1000:] == true_states[1:][1000:]).mean()
print(f"  hidden Markov regimes: calm {calm['ann_vol']:.0%} vol, turbulent {wild['ann_vol']:.0%} vol; causal out-of-sample accuracy {hit:.0%}")
shifted = np.concatenate([rng.normal(0, 1, 300), rng.normal(2.5, 1, 300)])
print(f"  change-point detection on a series that shifts at bar 300: {changepoints_from_run_length(bocpd(shifted, hazard=1 / 200)['run_length'])}")
dev = (close - np.convolve(close, np.ones(10) / 10, mode="full")[:n]) / close
target = np.concatenate([(-0.8 * dev + rng.normal(0, 0.01, n))[:-1], [np.nan]])
found = SymbolicAlpha(population=100, generations=6, seed=1).fit({"close": close, "volume": np.full(n, 1e6)}, target)
print(f"  genetic programming tried {found.n_evaluated} formulas, kept '{found.formula}': train IC {found.train_ic:+.2f}, held-out IC {found.test_ic:+.2f}")

# ---------------------------------------------------------------------------------------------------------------------
section("9. One platform, different roles")
platform = TradingPlatform(PlatformConfig(log_level="CRITICAL"))
platform.users.hasher = PasswordHasher(n=2 ** 4)                                      # fast hashing, for the demo only
password = "Correct-Horse-9"
admin = platform.users.register("admin", password, Role.ADMIN)
people = {role: platform.users.register(role.value.lower(), password, role, actor=admin) for role in Role if role is not Role.ADMIN}
for role in Role:
    user = admin if role is Role.ADMIN else people[role]
    print(f"  {role.value:<18} desks: {', '.join(sorted(platform.desks(user)))}")
quant, trader, pm, risk = (people[r] for r in (Role.QUANT, Role.TRADER, Role.PORTFOLIO_MANAGER, Role.RISK_MANAGER))
try:
    platform.desk(quant, "trading")
except AuthorizationError as exc:
    print(f"  research cannot trade: {exc.message}")

section("10. Governance, alerts and notifications")
research = platform.desk(quant, "research")
research.register_strategy("trend_pullback", {"spec": spec})
try:
    research.propose("trend_pullback", card)
except GovernanceError as exc:
    print(f"  proposing with a failing scorecard is refused: {exc.message}")
idx = pd.bdate_range("2019-01-01", periods=1260)
evidence = evaluate_strategy(pd.Series(np.random.default_rng(0).normal(0.0008, 0.008, 1260), index=idx), [100.0] * 100 + [-80.0] * 60)
research.propose("trend_pullback", evidence, note="illustrative evidence")
print(f"  proposed with a passing scorecard (grade {evidence.grade}); pending approval: {[r.strategy_id for r in platform.lifecycle.pending()]}")
platform.desk(pm, "portfolio").approve("trend_pullback", "sound")
print(f"  approved by a second person -> stage {platform.lifecycle.stage('trend_pullback').value}")
pager = MemoryChannel("pager")
platform.notifier.add_channel(pager, min_severity=Severity.ERROR)
platform.desk(risk, "risk").add_alert(drawdown_alert(0.10, for_count=2))
for drawdown in (0.12, 0.13, 0.14, 0.04):
    platform.alerts.evaluate({"drawdown": drawdown})
print(f"  drawdown alert over four evaluations sent: {[n.title for n in platform.inbox.items]}")
account = platform.accounts.link(trader, "paper", {})
engine = platform.desk(trader, "trading").open_session(account.account_id, [])
platform.desk(trader, "trading").halt(engine, "fat-finger order")
print(f"  trader tripped the kill switch -> pager received: {[(n.severity.value, n.title) for n in pager.items if n.title == 'Kill switch']}")
try:
    platform.reset_kill_switch(trader, engine, "let me trade again")
except AuthorizationError:
    print("  the trader cannot reset it; ", end="")
platform.desk(risk, "risk").resume(engine, "positions reconciled with the broker")
print(f"the risk desk can -> {engine.risk.kill_switch.state.value}")
print(f"  audit trail holds {len(platform.audit.entries())} entries")

print(f"\nDone in {time.time() - T0:.1f} s.")
