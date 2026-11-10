# algoplatform: a modular algorithmic trading platform in Python

One codebase for **research -> backtest -> paper -> live**. The same `TradingEngine` runs a backtest and a live session, so what you test is what you trade. Built from the 850-topic master list kept in the EPAT-Executive repo (`TRADING_PLATFORM_MASTER_TOPICS.md`); this repository holds the platform's low-level design and implementation.

Pure Python 3.10+ with `numpy` and `pandas`. No scipy: the statistical functions are implemented in `stats/special.py`.

```
pip install -e .                    # from this folder (optional: pip install -e ".[security]" for encrypted credentials)
python examples/run_demo.py         # core tour, ~15 s
python examples/run_advanced.py     # algos, FIX, recovery, research, valuation, sentiment, options, parallel search (~5 s)
python -m algoplatform backtest --strategy zscore_reversion --html report.html
python -m algoplatform optimize --strategy sma_crossover --method genetic
cd tests && python -m unittest discover -p "test_*.py"     # 400+ tests (CI runs Python 3.10-3.13)
```

## Web UI

```
PYTHONPATH=src python -m algoplatform.web --port 8080                    # then open http://127.0.0.1:8080
PYTHONPATH=src python -m algoplatform.web --port 8080 --db-dir ./state   # users and paper sessions survive a restart
```
Standard-library server plus a no-build single-page UI (`src/algoplatform/web/`). Tabs:

- **Backtest**: any runnable strategy (including the ML one) on synthetic data or your own CSV; equity vs buy-and-hold, drawdown, metrics, trades, VaR / expected shortfall.
- **Strategy builder**: indicators and rules as a form, validated server-side.
- **Optimise** and **Walk-forward**: random/genetic search with a plateau score; rolling out-of-sample testing with stitched equity, walk-forward efficiency and parameter stability.
- **Portfolio**: several strategies on shared capital (opposing views net out), standalone vs combined results, contribution, correlation, and suggested allocations you can apply with one click.
- **Data**: upload OHLCV CSVs (validated, capped, private to you).
- **Paper trading**: simulated broker on a synthetic feed with simulated time. Order ticket with optional execution algos (TWAP, VWAP, POV, iceberg, implementation shortfall), per-order cancel, parent-order cost analysis, positions, VaR and stress tests, risk meters, kill switch. Updates are pushed over Server-Sent Events, and the controls are not redrawn, so typing is never interrupted.
- **Admin & audit**: audit log, role management.

- The first account created becomes the administrator; later sign-ups are traders. Raise roles in the Admin tab (a Risk Manager or Admin is needed to reset a tripped kill switch).
- Every call goes through `TradingPlatform`, so permissions are enforced server-side, not just hidden in the UI. Server text is rendered with `textContent` only, responses carry CSP and no-sniff headers, and the stream authenticates with a header (never a URL token).
- **Localhost only.** There is no TLS. Do not expose it to a network without a TLS reverse proxy and your own auth in front. Uploaded data lives in server memory.
- API: see `web/server.py` (`/api/backtest|optimize|walkforward|portfolio|data/*|session/*|audit|users/*`).

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

Layers only depend downward; strategies never see orders or brokers, risk never knows about strategies, brokers only emit events.

## Module map

| Package | What it does | Key classes / functions |
|---|---|---|
| `core` | Shared contracts | typed exception tree (`retryable` flag), enums, `Order/Fill/Position/Signal/Bar`, `PlatformConfig` (validated), `EventBus`, `AuditLog`, `Registry` |
| `utils` | Reusable helpers | `retry`, `CircuitBreaker`, `RateLimiter`, `SimClock/SystemClock`, `TradingCalendar`, `IdGenerator`, validators |
| `users` | Users and broker accounts | scrypt passwords, lockout, signed tokens, RBAC (`Role`/`Permission`), `CredentialVault`, `BrokerAccountManager` |
| `brokers` | Venue connectivity | `BrokerGateway` ABC + registry, `PaperBroker` (matching, bracket/OCO/OTO, trailing stops, impact costs), `CostModel`, `ResilientGateway` (retry/breaker/failover), REST adapters `AlpacaGateway` / `ZerodhaGateway` / `BinanceGateway`, `brokers.fix` (FIX 4.4 wire format, session layer, `FixGateway`, test exchange) |
| `orders` | Order management | `OrderManager` (state machine, idempotent ids, duplicate/late fills, reconcile, restore), `ExecutionService` (+ bracket/OCO) |
| `execution` | Execution algos | TWAP, VWAP, POV, iceberg, implementation shortfall, `AlgoExecutionManager` (children go through risk), limit-price logic, pre-trade cost / go-no-go |
| `persistence` | Durability | `SqliteStore` (WAL, migrations, event journal), `Journal`, `recover` (rebuild OMS/portfolio/kill switch, reconcile with the broker, halt on any break) |
| `risk` | Separate risk layer | pre-trade checks, `KillSwitch` (two-person reset), `RiskMonitor`, `RiskManager`, VaR/ES/Kupiec/stress |
| `portfolio` | Positions, sizing, allocation | `Portfolio` ledger, sizers (fixed-fractional, vol-target, risk-per-trade, Kelly, optimal-f), `FundAllocator`, optimisers (min-var, risk-parity, max-Sharpe, Black-Litterman) |
| `strategy` | Strategy SDK + creator | `Strategy` ABC, built-ins (trend, reversion, breakout, momentum, pairs, `MLStrategy`), `StrategyBuilder` (JSON rules, no `eval`) |
| `research` | ML and signal research | triple-barrier and meta labels, sample weights, sequential bootstrap, purged / combinatorial CV, PBO, fractional differentiation, causal features, models, drift |
| `fundamentals` | Fundamental analysis | point-in-time statements, ratios, DuPont, Piotroski / Altman / Beneish, DCF + sensitivity + reverse DCF, factor scoring, `FactorStrategy` |
| `sentiment` | News and sentiment | lexicon (negation, phrases), event classifier, entity mapping, point-in-time de-duplicated `NewsStore`, `SentimentSignal`, `SentimentStrategy` |
| `derivatives` | Options | binomial (American) and Monte Carlo pricing, SVI fit, no-arbitrage checks, strategy builders with payoff and Greeks, contracts in the ledger, delta-hedge simulation |
| `data` | Market data | `MarketData` store/CSV, `clean_ohlcv`, split adjustment, synthetic generators |
| `backtest` | Event-driven backtester | `BacktestEngine`, `BacktestResult` |
| `optimize` | Parameter search | grid / random / genetic (optionally in a process pool, identical results), `walk_forward`, plateau score |
| `analytics` | Measurement | metrics, drawdown periods, trade stats, TCA (shortfall, markouts), signal IC / decay / quantiles, tear sheet (text/JSON/HTML) |
| `stats` | Statistical methods | ADF, KPSS, Engle-Granger, Hurst, variance ratio, bootstrap, PSR/DSR, FDR, OLS+Newey-West |
| `quant` | Quantitative methods | indicators, Black-Scholes + Greeks + IV, GARCH/EWMA, Ledoit-Wolf, Kalman, impact models (square-root law, Almgren-Chriss) |
| `app` | Wiring | `TradingEngine`, `TradingPlatform` facade (permission-checked), CLI |

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
- **Secrets stay secret.** Broker credentials are held in a `SecretStr`, never logged, and tests assert they do not appear in logs, `repr` or errors.

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

- **Simulated fills are bar-based** (open/limit/stop touch, volume participation, square-root impact scaled by recent volatility). There is no order-book queue or latency simulation, so high-frequency strategies are out of scope. Execution algos are tested at bar granularity.
- **Broker connectivity is verified only against fakes.** The Alpaca, Zerodha and Binance adapters follow the brokers' public API shapes and are tested against a scripted fake transport; the FIX gateway against an in-memory exchange and a localhost socket. None has been run against a live or sandbox broker. FIX is a working order-entry subset, not a certified implementation. Zerodha has no sandbox, so the adapter refuses `paper=True`. Test each adapter in the broker's sandbox, with tiny size, before anything else.
- **Credentials** are encrypted at rest only if the optional `cryptography` package is installed; otherwise they live in memory only and are never logged.
- **Persistence** is a single-process SQLite file: no replication or failover. The simulated paper broker is rebuilt from the journal; a real broker keeps its own state, which recovery reconciles against.
- **ML** is a numpy logistic model plus the leakage-safe research framework (labels, purged CV, PBO, drift). There is no deep learning, RL or LLM tooling; any scikit-learn-style estimator can be plugged in through `SklearnAdapter`.
- **Fundamentals and sentiment** ship the models and the point-in-time handling, not data feeds. The sentiment lexicon is deliberately simple and auditable, not a trained language model.
- **Options** are priced with Black-Scholes, binomial trees, Monte Carlo and SVI, with Greeks and expiry settlement helpers. There is no option-chain data feed, margin model, or live assignment/exercise management.
- **Parallel search** works for registry or rule-spec strategies through `make_parallel_objective`; walk-forward is still serial.
- **The web UI is for localhost.** No TLS; uploaded data lives in server memory.
- **Backtests use the data you supply.** Synthetic data is for tests and demos and says nothing about real markets. Measured here: about 0.1 s per 1,000 bars per symbol for a simple strategy; model-based and rule-heavy strategies are slower.
- Research code, not investment advice. Paper trade for an extended period before risking capital.

## Topic coverage (from the master list)

Strong: order lifecycle and idempotency (13.2), pre-trade controls (13.3), execution algos, order types and TCA (13.4-13.5, 13.12, 15.8), connectivity incl. FIX and broker abstraction (13.7), kill switch and restart recovery (12.1, 13.2), position sizing (11.4), portfolio optimisation (11.1), VaR/ES/stress (12.2), backtesting and bias control (10.x), statistical validation incl. deflated Sharpe, PBO and purged CV (2.1-2.2, 8.1), signal analytics (15.10), fundamentals and valuation (Part 4), news sentiment (6.1), options pricing and hedging (9.1), performance analytics (15.x), strategy SDK (7.3), user/RBAC and audit (16.6, 17.x).
Partial: derivatives beyond options (futures roll, fixed income, FX specifics), data engineering (in-memory store, SQLite journal), ML (classical models only), smart order routing across venues.
Not included: deep learning, RL and LLM agents (8.2-8.4), alternative-data feeds, crypto DeFi, SPAN/portfolio margin, HFT latency engineering.
