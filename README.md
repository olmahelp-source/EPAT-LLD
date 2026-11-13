# algoplatform: a modular algorithmic trading platform in Python

One codebase for **research -> backtest -> paper -> live**. The same `TradingEngine` runs a backtest and a live session, so what you test is what you trade. Built from the 850-topic master list kept in the EPAT-Executive repo (`TRADING_PLATFORM_MASTER_TOPICS.md`); this repository holds the platform's low-level design and implementation.

Pure Python 3.10+ with `numpy` and `pandas`. No scipy: the statistical functions are implemented in `stats/special.py`.

```
pip install -e .                    # from this folder (optional: pip install -e ".[security]" for encrypted credentials)
python examples/run_demo.py         # core tour, ~15 s
python examples/run_advanced.py     # algos, FIX, recovery, research, valuation, sentiment, options, parallel search (~5 s)
python examples/run_quant_lab.py    # patterns, formulas, selection, volatility, ML / DL / RL, roles, governance, alerts (~10 s)
python -m algoplatform backtest --strategy zscore_reversion --html report.html
python -m algoplatform optimize --strategy sma_crossover --method bayesian     # also: grid random genetic tpe pso de annealing cmaes
python -m algoplatform structure    # the job groups, their packages and roles; --check verifies the dependency rules
python -m algoplatform functions    # every indicator, pattern and operator usable in a formula
cd tests && python -m unittest discover -p "test_*.py"     # 750+ tests (CI runs Python 3.10-3.13)
```

Further reading: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (structure by job, dependency rules, roles and desks) and
[`PROGRESS.md`](PROGRESS.md) (what is built, what is verified, what is pending).

## Web UI

```
PYTHONPATH=src python -m algoplatform.web --port 8080                    # then open http://127.0.0.1:8080
PYTHONPATH=src python -m algoplatform.web --port 8080 --db-dir ./state   # users, sessions, approvals and alerts survive a restart
```
Standard-library server plus a no-build single-page UI (`src/algoplatform/web/`). Tabs:

- **Backtest**: any runnable strategy (including the ML one) on synthetic data or your own CSV; equity vs buy-and-hold, drawdown, metrics, trades, VaR / expected shortfall.
- **Strategy builder**: indicators and rules as a form, validated server-side.
- **Optimise** and **Walk-forward**: random/genetic search with a plateau score; rolling out-of-sample testing with stitched equity, walk-forward efficiency and parameter stability.
- **Portfolio**: several strategies on shared capital (opposing views net out), standalone vs combined results, contribution, correlation, and suggested allocations you can apply with one click.
- **Data**: upload OHLCV CSVs (validated, capped, private to you).
- **Paper trading**: simulated broker on a synthetic feed with simulated time. Order ticket with optional execution algos (TWAP, VWAP, POV, iceberg, implementation shortfall), per-order cancel, parent-order cost analysis, positions, VaR and stress tests, risk meters, kill switch. Updates are pushed over Server-Sent Events, and the controls are not redrawn, so typing is never interrupted.
- **Technical lab**: run a formula over the whole history (any of the 118 catalogue functions), see the chart patterns and candlesticks found, support and resistance levels, and a searchable function reference.
- **Volatility**: five estimators, a volatility cone, an out-of-sample league of forecasting models, and the exposure multiplier for a volatility target.
- **Selection**: rank a universe by any blend of rankers with a correlation limit; scores, the reason each instrument was left out, cointegrated pairs, lead-lag and the effective number of bets.
- **Governance**: register a strategy, score it, propose it for paper or live; approvers see pending requests with the scorecard and approve, reject or demote. The submitter can never approve.
- **Alerts & inbox**: personal and firm-wide alert rules checked after every paper-trading bar, your own notifications, platform notifications, and unacknowledged critical alerts that a risk manager or operations user must acknowledge.
- **Admin & audit**: audit log, role management.
- **Platform map**: your role and desks, the job groups and packages, and whether the dependency rules hold.

Each role sees only the tabs it can use: research tabs need `backtest:run`, paper trading needs `order:place`, so a viewer or auditor sees governance, alerts and the map only.

- The first account created becomes the administrator; later sign-ups are traders. Raise roles in the Admin tab (a Risk Manager or Admin is needed to reset a tripped kill switch).
- Every call goes through `TradingPlatform`, so permissions are enforced server-side, not just hidden in the UI. Server text is rendered with `textContent` only, responses carry CSP and no-sniff headers, and the stream authenticates with a header (never a URL token).
- **Localhost only.** There is no TLS. Do not expose it to a network without a TLS reverse proxy and your own auth in front. Uploaded data lives in server memory.
- API: see `web/server.py` (`/api/backtest|optimize|walkforward|portfolio|data/*|session/*|audit|users/*`) and `web/quant_api.py` (`/api/lab/*|governance/*|inbox/*|alerts/*|structure|functions`).

## Architecture

```
            ┌──────────────┐   Signal(target weight)    ┌─────────────────────────────┐
 data ────► │  Strategy    │ ─────────────────────────► │ TradingEngine (app/engine)  │
 (store,    │ (SDK / rules)│                            │  sizing -> net targets      │
 feeds)     └──────────────┘                            └──────────────┬──────────────┘
                                                                       │ orders
                  ┌────────────────────────────────────────────────────▼───────────────┐
                  │ ExecutionService (orders/execution)                                │
                  │   OMS (state machine, idempotency) -> RiskManager gate -> Broker   │
                  └───────┬───────────────────────────────┬────────────────────┬───────┘
                          │ fills/events                  │ halt               │
                    ┌─────▼─────┐                  ┌──────▼──────┐      ┌──────▼──────┐
                    │ Portfolio │                  │ KillSwitch  │      │ BrokerGateway│ paper | your adapter
                    │ (ledger)  │                  │ + Monitor   │      │ (+ failover) │
                    └───────────┘                  └─────────────┘      └─────────────┘
```

Layers only depend downward; strategies never see orders or brokers, risk never knows about strategies, brokers only emit events. These rules are written down as data in `architecture.py` and checked by a test, so an import that mixes concerns fails the build.

## Structure, grouped by job

| # | Job group | Packages | Main roles |
|---|---|---|---|
| 1 | Foundation | `core`, `utils` | everyone |
| 2 | Access control | `users` | Admin |
| 3 | Market and reference data | `data`, `instruments` | Operations, Quant |
| 4 | Signal libraries | `technical`, `fundamentals`, `sentiment` | Quant |
| 5 | Quantitative research | `stats`, `quant`, `derivatives`, `relationships`, `volatility` | Quant |
| 6 | Learning and AI | `research`, `ml`, `dl`, `rl`, `ai` | Quant |
| 7 | Strategy | `strategy`, `selection` | Quant, Trader |
| 8 | Portfolio and risk | `portfolio`, `risk` | Portfolio manager, Risk manager |
| 9 | Trading | `brokers`, `orders`, `execution` | Trader |
| 10 | Validation | `backtest`, `optimize`, `analytics` | Quant, Portfolio manager |
| 11 | Operations and control | `persistence`, `notify`, `ops`, `governance` | Operations, Risk manager, Auditor |
| 12 | Application | `app`, `web` | everyone, through role desks |

## Module map

| Package | What it does | Key classes / functions |
|---|---|---|
| `core` | Shared contracts | typed exception tree (`retryable` flag), enums, `Order/Fill/Position/Signal/Bar`, `PlatformConfig` (validated), `EventBus`, `AuditLog`, `Registry` |
| `utils` | Reusable helpers and resilience | `retry`, `CircuitBreaker`, `RateLimiter`, `SimClock/SystemClock`, `TradingCalendar`, `IdGenerator`, validators; `resilience`: `classify` / `ErrorPolicy`, `Result`, `first_success`, `call_with_timeout`, `Deadline`, `Bulkhead`, `ErrorBudget`, `ErrorBoundary`, `DeadLetterQueue`, `ErrorCollector`, `error_context` |
| `users` | Users and broker accounts | scrypt passwords, lockout, signed tokens, RBAC (8 roles, 20 permissions), `CredentialVault`, `BrokerAccountManager` |
| `instruments` | Reference data | `ContractSpec`, `InstrumentMaster` (search, broker symbol mapping, bulk load with a per-row error report), corporate actions, continuous futures |
| `technical` | Technical analysis | 55 indicators, 19 candlestick patterns, chart patterns (double top/bottom, head and shoulders, triangles, wedges, channels, breakouts, divergences), swings and support/resistance, 43 operators, function catalogue (`compute`, `indicator_frame`), safe formula engine (`Expression`) |
| `relationships` | How instruments relate | correlation (Pearson, rank, partial, EWMA), beta, tail dependence, lead-lag, Granger causality, cointegration scan, hierarchical clusters, spanning tree, PCA, `RelationshipManager` (peers, pairs, effective number of bets) |
| `volatility` | Volatility | estimators (close-to-close, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang), cone, forecasters (EWMA, GARCH, GJR, HAR), QLIKE / Mincer-Zarnowitz evaluation, volatility targeting, `VolRegime`, `VolatilityManager` |
| `ml` | Machine learning | `DecisionTree`, `RandomForest`, `GradientBoosting`, `KNN`, `GaussianNB`, `LassoModel`, `PCA`, `KMeans`, voting and stacking, `walk_forward_predict`, metrics |
| `dl` | Deep learning (numpy) | `Dense`, `LSTM`, `Dropout`, `Adam`, `Sequential` with early stopping; `MLP`, `LSTMForecaster`, `Autoencoder`; gradients verified numerically |
| `rl` | Reinforcement learning | `TradingEnv`, `QLearningAgent`, `DQNAgent`, `ReinforceAgent`, epsilon-greedy / UCB / Thompson bandits |
| `ai` | Other AI methods | `GaussianHMM` / `RegimeDetector`, fuzzy rule systems, genetic programming (`SymbolicAlpha`), `IsolationForest`, CUSUM and Bayesian online change-point detection |
| `selection` | Instrument selection | `UniverseSelector` (filters, 9 rankers, external scores, correlation-aware diversification, group caps), one-per-cluster, pair selection; point-in-time |
| `notify` | Notifications and alerts | log / memory / file / webhook / Slack / Telegram / e-mail channels, `Notifier` (retry, dedupe, private routing, rate limit, dead letters, escalation), `AlertManager` and rule factories |
| `ops` | Operations | `HealthMonitor` (checks, heartbeats), `Scheduler` (interval and daily jobs on the platform clock) |
| `governance` | Control | `StrategyLifecycle` (research, paper, live; four-eyes approval), `ModelRegistry` (versions, data fingerprint, rollback) |
| `brokers` | Venue connectivity | `BrokerGateway` ABC + registry, `PaperBroker` (matching, bracket/OCO/OTO, trailing stops, impact costs), `CostModel`, `ResilientGateway` (retry/breaker/failover), REST adapters `AlpacaGateway` / `ZerodhaGateway` / `BinanceGateway`, `brokers.fix` (FIX 4.4 wire format, session layer, `FixGateway`, test exchange), `OrderBook` (price-time matching, IOC/FOK, post-only, self-trade prevention, queue position, cost to trade) and `MarketSimulator` (seeded tick flow with latency, `ticks_to_bars`) |
| `orders` | Order management | `OrderManager` (state machine, idempotent ids, duplicate/late fills, reconcile, restore), `ExecutionService` (+ bracket/OCO) |
| `execution` | Execution algos | TWAP, VWAP, POV, iceberg, implementation shortfall, `AlgoExecutionManager` (children go through risk), limit-price logic, pre-trade cost / go-no-go |
| `persistence` | Durability | `SqliteStore` (WAL, migrations, event journal), `Journal`, `recover` (rebuild OMS/portfolio/kill switch, reconcile with the broker, halt on any break); `TradingPlatform(store=...)` also saves and restores instruments, strategy approvals, the model registry, alert rules and notification routes |
| `risk` | Separate risk layer | pre-trade checks, `KillSwitch` (two-person reset), `RiskMonitor`, `RiskManager`, VaR/ES/Kupiec/stress, margin (rule-based and scenario / risk-array, account report, liquidation price, `MarginCheck` that never blocks risk reduction) |
| `portfolio` | Positions, sizing, allocation | `Portfolio` ledger, sizers (fixed-fractional, vol-target, risk-per-trade, Kelly, optimal-f), `FundAllocator`, optimisers (min-var, risk-parity, max-Sharpe, Black-Litterman), multi-currency (`FxRates` with point-in-time and cross rates, `CashLedger` per currency, exposure by currency, FX P&L attribution, hedge amounts) |
| `strategy` | Strategy SDK + creators | `Strategy` ABC, built-ins (trend, reversion, breakout, momentum, pairs, `MLStrategy`), `StrategyBuilder` (JSON rules, no `eval`), `ExpressionStrategy` (formulas over the whole catalogue), `PolicyStrategy` (adapter for any model) |
| `research` | ML and signal research | triple-barrier and meta labels, sample weights, sequential bootstrap, purged / combinatorial CV, PBO, fractional differentiation, causal features, models, drift |
| `fundamentals` | Fundamental analysis | point-in-time statements, ratios, DuPont, Piotroski / Altman / Beneish, DCF + sensitivity + reverse DCF, factor scoring, `FactorStrategy` |
| `sentiment` | News and sentiment | lexicon (negation, phrases), event classifier, entity mapping, point-in-time de-duplicated `NewsStore`, `SentimentSignal`, `SentimentStrategy`, `LLMSentimentScorer` (language-model scoring with lexicon fallback) |
| `derivatives` | Options | binomial (American) and Monte Carlo pricing, SVI fit, no-arbitrage checks, strategy builders with payoff and Greeks, contracts in the ledger, delta-hedge simulation, `OptionChain` (implied vol, Greeks, put/call ratio, max pain, skew, term structure, quality report) |
| `data` | Market data | `MarketData` store/CSV, `clean_ohlcv`, split adjustment, synthetic generators |
| `backtest` | Event-driven backtester | `BacktestEngine`, `BacktestResult` |
| `optimize` | Parameter search | grid / random / genetic (optionally in a process pool, identical results), Bayesian, TPE, particle swarm, differential evolution, simulated annealing, CMA-ES, successive halving, Pareto front, `walk_forward`, plateau score |
| `analytics` | Measurement | metrics, drawdown periods, trade stats, TCA (shortfall, markouts), signal IC / decay / quantiles, tear sheet (text/JSON/HTML), result analysis (bootstrap intervals, drawdown probability, stability, regime breakdown, trade concentration, `evaluate_strategy` scorecard) |
| `stats` | Statistical methods | ADF, KPSS, Engle-Granger, Hurst, variance ratio, bootstrap, PSR/DSR, FDR, OLS+Newey-West |
| `quant` | Quantitative methods | indicators, Black-Scholes + Greeks + IV, GARCH/EWMA, Ledoit-Wolf, Kalman, impact models (square-root law, Almgren-Chriss) |
| `app` | Wiring | `TradingEngine`, `TradingPlatform` facade (permission-checked), role desks (`workspaces`), CLI |

## Safety properties (each has tests)

- **No look-ahead.** Strategies read a causal `HistoryView`; a signal on bar *t* fills no earlier than the open of bar *t+1*. Altering future data leaves past results unchanged.
- **Deterministic.** Same inputs give identical results, independent of Python's hash seed.
- **Idempotent orders.** Reusing an `idempotency_key` returns the original order; duplicate fills are ignored; late fills after a cancel are recorded, not lost.
- **Fail closed.** A broken risk check rejects the order. A crashing strategy is contained, and repeated failures trip the kill switch.
- **Kill switch.** Trips on daily loss, drawdown, consecutive errors or stale data (or by hand). It cancels resting orders, flattens positions with reduce-only orders, blocks everything else, and resets only with a permitted user, a written reason and (optionally) a second approver.
- **Netting.** Opposing views from different strategies cancel internally before any order is sent.
- **Overfitting guards.** Walk-forward analysis, parameter-plateau score, deflated Sharpe ratio that accounts for the number of trials, probability of backtest overfitting, purged CV, a `min_trades` filter in objectives.
- **Point-in-time data.** Filings are used only from their announcement date (restatements by the version known then); news only after publication plus latency. Tests change the future and assert the past is untouched.
- **Algo children cannot bypass risk.** Every child order of an execution algo passes the same pre-trade checks; repeated rejects fail the parent instead of retrying forever.
- **Restart safety.** State is journaled to SQLite; after a crash the engine rebuilds itself, reconciles with the broker and halts trading on any mismatch. A failed journal write trips the kill switch. A tripped kill switch stays tripped across restarts.
- **Secrets stay secret.** Broker credentials are held in a `SecretStr`, never logged, and tests assert they do not appear in logs, `repr` or errors. The same holds for webhook URLs and bot tokens in notification channels.
- **Concerns stay separate.** `architecture.py` lists what each package may import; a test parses the source tree and fails on anything else. Libraries cannot reach orders or brokers; strategies cannot reach risk.
- **Four eyes.** A strategy reaches a live account only with a passing scorecard and the approval of a second person; the submitter and the owner cannot approve, an administrator included. Research roles cannot place orders; whoever trades cannot reset the kill switch.
- **Formulas cannot execute code.** Strategy formulas are parsed and then interpreted by a small evaluator that knows arithmetic, comparisons and catalogue functions only; there is no `eval`, attribute access or import. Tests feed it hostile input.
- **Every indicator is checked for look-ahead.** One test runs all catalogue indicators and patterns on a price history and on a prefix of it and requires identical values on the shared bars.
- **One failure stays local.** A bad symbol in a scan, a crashing optimiser trial, a dead notification channel, a broken alert rule or a failing scheduled job is recorded and skipped; the rest carries on. Undelivered notifications go to a dead-letter queue and can be replayed.

## Quick start

```python
from algoplatform.data import MarketData, synthetic
from algoplatform.backtest import BacktestEngine
from algoplatform.strategy import SMACrossover
from algoplatform.core import PlatformConfig

md = MarketData()
for sym, df in synthetic.universe(["AAA", "BBB"], n=750).items():
    md.add(sym, df)                      # or md.add_csv("AAA", "aaa.csv")  (date,open,high,low,close,volume)

result = BacktestEngine(md, PlatformConfig()).run(SMACrossover(md.symbols, fast=10, slow=50))
print(result.metrics()["sharpe"], result.trade_stats()["win_rate"])
```

### Define a strategy without code
```python
from algoplatform.strategy import StrategyBuilder, create_from_spec
spec = (StrategyBuilder("rsi_trend").param("lvl", 35)
        .indicator("rsi", "rsi", period=14).indicator("trend", "sma", period=100).indicator("atr", "atr", period=14)
        .long_entry(("rsi", "<", "$lvl"), ("close", ">", "trend")).long_exit(("rsi", ">", 60))
        .stop_atr("atr", 2.5).build())          # validated; JSON-serialisable via spec_to_json
strategy = create_from_spec(spec, ["AAA"])
```

### Write a strategy in code
```python
from algoplatform.strategy import Strategy, STRATEGIES
from algoplatform.core import Signal
from algoplatform.quant import sma

@STRATEGIES.register("my_trend")
class MyTrend(Strategy):
    name = "my_trend"
    PARAMS = {"n": 50}
    PARAM_BOUNDS = {"n": (10, 200)}          # lets the optimiser and CLI tune it
    @property
    def warmup(self): return self.p("n") + 1
    def on_bar(self, ctx):
        out = []
        for s in self.symbols:
            if s in ctx.bars and self.ready(ctx, s):
                c = ctx.closes(s, self.p("n") + 1)
                out.append(Signal(s, 1.0 if c[-1] > sma(c, self.p("n"))[-1] else 0.0))
        return out
```

### Define a strategy as formulas
```python
from algoplatform.strategy import create_from_expressions
spec = {"name": "trend_pullback", "params": {"fast": 20, "slow": 100}, "bounds": {"fast": [5, 60], "slow": [60, 250]},
        "long_entry": "ema(close, fast) > ema(close, slow) and crossover(rsi(close, 14), 40)",
        "long_exit": "crossunder(close, ema(close, slow)) or rsi(close, 14) > 80", "stop_pct": 0.08}
strategy = create_from_expressions(spec, ["AAA"])     # any catalogue function works: python -m algoplatform functions
# or a target weight directly:  {"name": "fade", "target": "clamp(-zscore(close, 20) / 2, -1, 1)"}
```

### Choose what to trade, point-in-time
```python
from algoplatform.selection import UniverseSelector
picked = (UniverseSelector().require_history(252).min_dollar_volume(2e6)
          .rank_by("momentum", lookback=126).rank_by("low_volatility", weight=0.5)
          .diversify(max_correlation=0.8).per_group(sectors, 2).top(10)
          .select(market_data, asof="2023-06-30"))     # picked.symbols, picked.scores, picked.rejected (with reasons)
```

### Score a result before believing it
```python
from algoplatform.analytics import evaluate_backtest
verdict = evaluate_backtest(result, n_trials=opt.n_trials)   # the bar rises with the number of things tried
verdict.passed, verdict.grade, verdict.failures
```

### Size by volatility and regime
```python
from algoplatform.volatility import VolatilityManager
view = VolatilityManager(target_vol=0.10, model="garch").assess(returns)   # forecast_vol, regime, scalar (exposure multiplier)
```

### Alerts and notifications
```python
from algoplatform.notify import SlackChannel, drawdown_alert
platform.notifier.add_channel(SlackChannel(webhook_url), min_severity=Severity.ERROR)   # kill switch and risk breaches arrive by themselves
platform.alerts.add(drawdown_alert(0.10, for_count=2))
platform.alerts.evaluate({"drawdown": 0.12})          # call every bar; fires once, repeats after the cooldown, reports when resolved
```

### Work through role desks
```python
desks = platform.desks(user)                          # exactly the desks this user's role may open
research = platform.desk(quant, "research")           # backtest, optimize, evaluate, select_universe, propose
research.register_strategy("trend_pullback")
research.propose("trend_pullback", verdict)           # refused unless the scorecard passes
platform.desk(pm, "portfolio").approve("trend_pullback")   # must be a different person
```

### Add a broker
Subclass `BrokerGateway` (or `HttpBrokerGateway` for REST APIs: it supplies signing hooks, rate limiting, retries, error classification and a polling `sync()`), implement the request mapping, emit `BrokerEvent`s for ack/fill/reject/cancel, and register with `@BROKERS.register("name")`. Nothing else changes. Test it first against a `FakeTransport`, then the broker's sandbox.

### Work a large order with an execution algo
```python
engine = TradingEngine(cfg, broker, instruments, clock, algo=("twap", {"bars": 5}), algo_min_notional=10_000)
# or explicitly:
parent = engine.algos.start("AAPL", Side.BUY, 5_000, "pov", arrival_price=187.2, participation=0.1)
# BacktestEngine(...).run(strategy, algo=("vwap", {"bars": 8})) ; analytics.tca.tca_summary(result.parents)
```
Algos pay off when your order is a meaningful share of volume. For small orders slicing mostly delays entry (see `examples/run_advanced.py`).

### Survive a restart
```python
engine = TradingEngine(cfg, broker, instruments, clock, store=SqliteStore("state.db"))
report = engine.recover()        # rebuilds orders, fills, positions and the kill switch; reconciles with the broker
if not report.clean: ...         # trading is already halted; a human reviews report.reconcile / report.unresolved
```

### Optimise in parallel
```python
obj = make_parallel_objective(market_data, "sma_crossover", symbols, cfg, min_trades=5)   # picklable: data + config, not closures
result = random_search(obj, space, n_iter=60, n_jobs=4)       # same trial table as n_jobs=1
```

### Run a permission-checked session
```python
from algoplatform.app import TradingPlatform
from algoplatform.core import Role
from algoplatform.strategy import SMACrossover
plat = TradingPlatform()
admin = plat.users.register("admin_user", "Str0ng-Passw0rd!", Role.ADMIN)
trader = plat.users.register("trader_one", "Str0ng-Passw0rd!")
acc = plat.accounts.link(trader, "paper", {})
engine = plat.open_session(trader, acc.account_id, [(SMACrossover(["AAA"]), 1.0)])
# engine.start(); feed bars with engine.on_bars({...}); plat.trip_kill_switch(trader, engine, "reason")
```

## Honest limits

- **Simulated fills are bar-based** (open/limit/stop touch, volume participation, square-root impact scaled by recent volatility). The trading engine and backtester run on bars. `brokers.OrderBook` and `MarketSimulator` add a real matching engine with queue position and latency for execution research, but the background order flow is random ("zero intelligence"), so it shows mechanics, not real market behaviour, and the engine does not yet run tick by tick.
- **Broker connectivity is verified only against fakes.** The Alpaca, Zerodha and Binance adapters follow the brokers' public API shapes and are tested against a scripted fake transport; the FIX gateway against an in-memory exchange and a localhost socket. None has been run against a live or sandbox broker. FIX is a working order-entry subset, not a certified implementation. Zerodha has no sandbox, so the adapter refuses `paper=True`. Test each adapter in the broker's sandbox, with tiny size, before anything else.
- **Credentials** are encrypted at rest only if the optional `cryptography` package is installed; otherwise they live in memory only and are never logged.
- **Persistence** is a single-process SQLite file: no replication or failover. The simulated paper broker is rebuilt from the journal; a real broker keeps its own state, which recovery reconciles against.
- **Learning models are numpy implementations sized for daily data**: trees, forests, boosting, a small neural-network library (dense and LSTM layers, gradients verified numerically), Q-learning, DQN and policy-gradient agents, hidden Markov regimes, genetic programming. They train in seconds on thousands of rows and are not a substitute for a GPU framework; a model trained elsewhere plugs in through `SklearnAdapter` or `PolicyStrategy`. The reinforcement-learning and genetic-programming tests use synthetic data with a planted edge: they show the machinery learns, not that a real market edge exists. Both are strong overfitting machines; judge them only out of sample and count the trials.
- **The language-model sentiment scorer** is provider-neutral (it takes a `complete(system, user)` function) and is tested with fakes. `anthropic_completer` was checked against the SDK's method signature but has not been run against the live API from this repository. Using it sends headlines and article text to an external service.
- **Notification channels** (Slack, Telegram, webhook, e-mail) are tested against injected fake transports, never a real service.
- **Chart patterns** are rule-based detections on confirmed swings. The thresholds are parameters, not calibrated truths, and a detected pattern is a description, not a forecast.
- **Cointegration, lead-lag and formula searches test many candidates.** Results carry the number tested and the false positives expected by chance; treat them as candidates to validate.
- **Fundamentals and sentiment** ship the models and the point-in-time handling, not data feeds. The sentiment lexicon is deliberately simple and auditable, not a trained language model.
- **Options** are priced with Black-Scholes, binomial trees, Monte Carlo and SVI, with Greeks and expiry settlement helpers. `OptionChain` analyses quotes you supply; there is no option-chain data feed, margin model, or live assignment/exercise management.
- **Margin** follows the two standard methods (fixed percentages, and a scenario grid that revalues each underlying's positions together), but its parameters are typical defaults, not any exchange's or broker's schedule. Load the real numbers before using it for live sizing.
- **Multi-currency** accounting sits beside the position ledger: `Portfolio` keeps one cash figure and each position in its instrument's own currency; `CashLedger` and `portfolio_currency_exposure` give the per-currency view. The engine does not convert P&L automatically.
- **Parallel search** works for registry or rule-spec strategies through `make_parallel_objective`; walk-forward is still serial.
- **The web UI is for localhost.** No TLS; uploaded data lives in server memory. Alert rules are evaluated against the paper-trading session that is stepping, so with several users trading at once a firm-wide metric alert sees whichever session moved last.
- **What a restart keeps.** With a store, notification channels that hold a secret (webhook URL, bot token, e-mail password) are deliberately not written to disk and must be registered again; the restore report lists them. Fitted model objects are not pickled: restored registry versions must have their model re-attached, which is checked against the training-data fingerprint. Alert rules defined in code (rather than as thresholds) must be added again.
- **Backtests use the data you supply.** Synthetic data is for tests and demos and says nothing about real markets. Measured here: about 0.1 s per 1,000 bars per symbol for a simple strategy; model-based and rule-heavy strategies are slower.
- Research code, not investment advice. Paper trade for an extended period before risking capital.

## Topic coverage (from the master list)

Strong: order lifecycle and idempotency (13.2), pre-trade controls (13.3), execution algos, order types and TCA (13.4-13.5, 13.12, 15.8), connectivity incl. FIX and broker abstraction (13.7), kill switch and restart recovery (12.1, 13.2), position sizing (11.4), portfolio optimisation (11.1), VaR/ES/stress (12.2), backtesting and bias control (10.x), statistical validation incl. deflated Sharpe, PBO and purged CV (2.1-2.2, 8.1), signal analytics (15.10), fundamentals and valuation (Part 4), news sentiment (6.1), options pricing and hedging (9.1), performance analytics (15.x), strategy SDK (7.3), user/RBAC and audit (16.6, 17.x), technical indicators and pattern recognition, volatility estimation and forecasting, relationship analysis, instrument selection, parameter optimisation, notifications and alerting, strategy and model governance.
Partial: derivatives beyond options (futures roll is covered; fixed income and FX specifics are not), data engineering (in-memory store, SQLite journal), machine, deep and reinforcement learning (numpy implementations for daily data), language models (sentiment scoring only), smart order routing across venues, margin (both methods, default parameters), multi-currency accounting (beside the ledger, not inside the engine), order-book simulation (a laboratory, not a tick-driven backtester).
Not included: alternative-data feeds, crypto DeFi, exchange-certified SPAN files, HFT latency engineering on real infrastructure.
