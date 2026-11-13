# Architecture

`algoplatform` is one platform built from parts that each do one job. This document says what the jobs are, who does
them, and which rules keep the parts from growing into each other. The tables below are generated from the code
(`algoplatform/architecture.py`, `core/enums.py`, `users/security.py`, `app/workspaces.py`), which is the authority if
the two ever disagree. Print the live version with `python -m algoplatform structure`.

## 1. Structure, grouped by job

| # | Job group | The question it answers | Packages | Who mainly works here |
|---|---|---|---|---|
| 1 | **Foundation** | Shared contracts, error handling and resilience that everything else relies on. | `core`, `utils` | everyone, through the desks |
| 2 | **Access control** | Who is this user, what may they do, and which broker accounts are theirs? | `users` | Admin |
| 3 | **Market and reference data** | What can be traded, on what terms, and what has it cost? | `data`, `instruments` | Ops, Quant |
| 4 | **Signal libraries** | Turn raw data into numbers a rule can use: technical, fundamental, sentiment. | `technical`, `fundamentals`, `sentiment` | Quant |
| 5 | **Quantitative research** | The mathematics: statistics, pricing and options, relationships, volatility. | `stats`, `quant`, `derivatives`, `relationships`, `volatility` | Quant |
| 6 | **Learning and AI** | Models that learn from data: leakage-safe research, ML, deep learning, RL, other AI. | `research`, `ml`, `dl`, `rl`, `ai` | Quant |
| 7 | **Strategy** | Decide what to hold: the strategy SDK, strategy creators and instrument selection. | `strategy`, `selection` | Quant, Trader |
| 8 | **Portfolio and risk** | How much to hold, how capital is shared out, and when to stop. | `portfolio`, `risk` | PM, Risk |
| 9 | **Trading** | Get orders to the market and fills back: brokers, order management, execution algorithms. | `brokers`, `orders`, `execution` | Trader |
| 10 | **Validation** | Evidence that an idea works: backtests, parameter optimisation, result analysis. | `backtest`, `optimize`, `analytics` | Quant, PM |
| 11 | **Operations and control** | Keep it running and accountable: durability, alerts, health, schedules, approvals. | `persistence`, `notify`, `ops`, `governance` | Ops, Risk, Auditor |
| 12 | **Application** | Wire the parts together behind permission-checked, role-specific desks; CLI and web UI. | `app`, `web` | everyone, through the desks |

Each package has a single reason to change. A few examples of where a concern lives, and where it deliberately does not:

| Concern | Lives in | Never in |
|---|---|---|
| "What is the RSI?" | `technical` (a pure function of prices) | `strategy`, `risk` |
| "Should I be long?" | `strategy` (returns a target weight) | anything that knows about orders or brokers |
| "How many shares is that?" | `portfolio` (sizing) | `strategy` |
| "Is this order allowed?" | `risk` (does not know which strategy asked) | `strategy`, `brokers` |
| "How does it get to the exchange?" | `orders`, `execution`, `brokers` | `strategy`, `risk` |
| "Is this user allowed to?" | `users` and the desks in `app` | any library |
| "Who must be told?" | `notify` | the code that detected the problem (it publishes an event) |
| "May this strategy go live?" | `governance` | the strategy, or the person who wrote it |

## 2. Data flow

```
 instruments + data ──► technical / fundamentals / sentiment ──► strategy ──► Signal (target weight)
        ▲                     ▲            ▲                        │
        │            stats, quant, volatility, relationships        ▼
   selection ◄──────── research, ml, dl, rl, ai ─────────►   portfolio (sizing, allocation)
                                                                    │ order
                                              risk (pre-trade checks, kill switch) ──► orders ──► execution ──► brokers
                                                                    │ events                          │ fills
                                   notify ◄── EventBus ◄────────────┴──────────────────────────────────┘
                                      │
     backtest / optimize / analytics (evidence) ──► governance (approve) ──► live session
```

The same `TradingEngine` runs a backtest and a live session, so the path a signal takes is identical in both.

## 3. Dependency rules

Every package has an allow-list of packages it may import. `architecture.check_architecture()` parses the source tree
and reports any import that is not on the list; a unit test (`tests/test_architecture_desks.py`) fails the build if it
finds one. Adding a dependency therefore means editing the allow-list on purpose.

The invariants behind the list:

1. `core` imports nothing. `utils` imports only `core`.
2. **Libraries stay pure computation.** `stats`, `quant`, `derivatives`, `relationships`, `volatility`, `technical`,
   `research`, `ml`, `dl`, `rl` and `ai` never import users, brokers, orders, execution, risk, portfolio, persistence,
   notifications, governance or the application.
3. **`strategy` never sees an order, a broker, a risk check or a user.** It returns target weights.
4. **`risk` never knows which strategy produced an order. `brokers` only translate and emit events.**
5. Only `app` and `web` see everything. Nothing imports `web`.

Deliberate exceptions, each documented in the allow-list: `fundamentals` and `sentiment` each ship a `Strategy`
adapter, so they import `strategy`; `backtest` imports `app` because it reuses the live engine rather than keeping a
second, slightly different one; `risk` imports `quant` because scenario margin reprices options.

| Package | May import |
|---|---|
| `core` | *nothing* |
| `utils` | `core` |
| `users` | `core`, `utils` |
| `data` | `core` |
| `instruments` | `core`, `utils` |
| `technical` | `core`, `quant` |
| `fundamentals` | `core`, `strategy`, `utils` |
| `sentiment` | `core`, `strategy`, `utils` |
| `stats` | `core`, `utils` |
| `quant` | `core`, `stats` |
| `derivatives` | `core`, `quant` |
| `relationships` | `core`, `stats` |
| `volatility` | `core`, `quant`, `stats`, `utils` |
| `research` | `core`, `quant`, `stats` |
| `ml` | `core`, `research`, `utils` |
| `dl` | `core`, `utils` |
| `rl` | `core`, `dl`, `utils` |
| `ai` | `core`, `quant`, `technical`, `utils` |
| `strategy` | `core`, `quant`, `research`, `stats`, `technical` |
| `selection` | `core`, `relationships` |
| `portfolio` | `core`, `utils` |
| `risk` | `core`, `portfolio`, `quant`, `stats`, `utils` |
| `brokers` | `core`, `users`, `utils` |
| `orders` | `brokers`, `core`, `portfolio`, `risk`, `utils` |
| `execution` | `brokers`, `core`, `orders`, `quant`, `utils` |
| `backtest` | `analytics`, `app`, `brokers`, `core`, `data`, `portfolio`, `strategy`, `utils` |
| `optimize` | `analytics`, `backtest`, `core`, `data`, `stats`, `strategy`, `utils` |
| `analytics` | `core`, `portfolio`, `stats`, `utils` |
| `persistence` | `brokers`, `core` |
| `notify` | `core`, `utils` |
| `ops` | `core`, `utils` |
| `governance` | `analytics`, `core`, `users`, `utils` |
| `app` | `analytics`, `backtest`, `brokers`, `core`, `data`, `execution`, `fundamentals`, `governance`, `instruments`, `notify`, `ops`, `optimize`, `orders`, `persistence`, `portfolio`, `quant`, `relationships`, `risk`, `selection`, `sentiment`, `strategy`, `technical`, `users`, `utils` |
| `web` | `ai`, `analytics`, `app`, `backtest`, `brokers`, `core`, `data`, `derivatives`, `dl`, `fundamentals`, `ml`, `notify`, `optimize`, `persistence`, `portfolio`, `relationships`, `research`, `risk`, `selection`, `sentiment`, `strategy`, `technical`, `users`, `utils`, `volatility` |

## 4. One platform, different roles

A user has exactly one role. A role is a set of permissions. Every operation checks a permission on the server side,
so hiding a button in a UI is never the control.

### Permissions by role

| Permission | Admin | Quant | Trader | PM | Risk | Ops | Auditor | Viewer |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `order:place` | yes |  | yes |  |  |  |  |  |
| `order:cancel` | yes |  | yes | yes | yes | yes |  |  |
| `order:view` | yes | yes | yes | yes | yes | yes | yes | yes |
| `strategy:manage` | yes | yes | yes |  |  |  |  |  |
| `backtest:run` | yes | yes | yes | yes |  |  |  |  |
| `portfolio:view` | yes | yes | yes | yes | yes | yes | yes | yes |
| `risk:manage` | yes |  |  |  | yes |  |  |  |
| `risk:reset_kill_switch` | yes |  |  |  | yes |  |  |  |
| `risk:trigger_kill_switch` | yes |  | yes | yes | yes | yes |  |  |
| `users:manage` | yes |  |  |  |  |  |  |  |
| `brokers:manage` | yes |  | yes |  |  |  |  |  |
| `model:train` | yes | yes |  |  |  |  |  |  |
| `strategy:submit` | yes | yes | yes |  |  |  |  |  |
| `strategy:approve` | yes |  |  | yes | yes |  |  |  |
| `allocation:manage` | yes |  |  | yes |  |  |  |  |
| `risk:view` | yes | yes | yes | yes | yes | yes | yes | yes |
| `audit:view` | yes |  |  |  | yes |  | yes |  |
| `alerts:manage` | yes |  | yes | yes | yes | yes |  |  |
| `instruments:manage` | yes |  |  |  |  | yes |  |  |
| `system:operate` | yes |  |  |  |  | yes |  |  |

### Segregation of duties (enforced by tests)

- **Research cannot trade.** Only Trader (and Admin) hold `order:place`.
- **Nobody approves their own work.** No role except Admin holds both `strategy:submit` and `strategy:approve`, and the
  lifecycle additionally refuses an approver who is the submitter or the owner, Admin included.
- **Whoever trades cannot restart trading.** Any trader may trip the kill switch; only Risk (and Admin) may reset it.
- **Read-only means read-only.** Auditor and Viewer hold view permissions only; the compliance desk exposes no method
  that changes state.

### Desks

A desk (`app/workspaces.py`) is the slice of the platform one job works with. `platform.desks(user)` returns exactly
the desks a user may open.

| Desk | Needs permission | Admin | Quant | Trader | PM | Risk | Ops | Auditor | Viewer |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `research` | `backtest:run` | yes | yes | yes | yes |  |  |  |  |
| `trading` | `order:place` | yes |  | yes |  |  |  |  |  |
| `portfolio` | `allocation:manage` | yes |  |  | yes |  |  |  |  |
| `risk` | `risk:view` | yes | yes | yes | yes | yes | yes | yes | yes |
| `operations` | `system:operate` | yes |  |  |  |  | yes |  |  |
| `compliance` | `audit:view` | yes |  |  |  | yes |  | yes |  |
| `admin` | `users:manage` | yes |  |  |  |  |  |  |  |

| Desk | What it does |
|---|---|
| `research` | backtest, optimise, score results, select instruments, map relationships, register models, propose a strategy, download market history |
| `trading` | open a session on the user's own broker account, place and cancel orders, halt, personal alerts |
| `portfolio` | allocate capital between strategies, approve or reject promotions |
| `risk` | risk snapshot and concentration, halt and resume, demote a live strategy, firm-wide alerts |
| `operations` | health, notification channels, undelivered notifications, instrument master, scheduled jobs, data providers, the sample-data-only lock |
| `compliance` | audit trail, strategy history, open critical alerts (read-only) |
| `admin` | users and roles |

### The path to live trading

```
RESEARCH ──submit + approve──► PAPER ──(minimum days) submit + approve──► LIVE ──► RETIRED
                                  ▲──────────────── demote (one person, a reason) ──────┘
```

A promotion request must carry a passing scorecard (`analytics.evaluate_strategy`), and a different person must approve
it. A live broker account refuses any strategy that is not in the LIVE stage. Taking risk off (demote, retire, halt) is
deliberately a one-person action.

## 5. Error handling

| Layer | Mechanism | Where |
|---|---|---|
| Say what went wrong | typed exception tree with `code`, `details` and a `retryable` flag | `core/exceptions.py` |
| Decide what to do | `classify()` / `ErrorPolicy` map an exception to a category, a severity and an action (retry, fall back, skip, escalate, halt) | `utils/resilience.py` |
| Survive transient faults | `retry`, `CircuitBreaker`, `RateLimiter`, `call_with_timeout`, `Deadline`, `Bulkhead` | `utils` |
| Keep one failure local | `ErrorBoundary` with an `ErrorBudget`; `Result` and `ErrorCollector` for batches; `first_success` for fallbacks | `utils/resilience.py` |
| Lose nothing | `DeadLetterQueue` with replay | `utils/resilience.py`, `notify` |
| Tell a person | events on the bus become notifications; unacknowledged critical ones escalate | `notify` |

Two rules hold everywhere: a failure in one item of a batch (a symbol, a trial, a channel, an alert rule, a scheduled
job) never stops the others, and an error whose action is HALT (kill switch, bad configuration) is never swallowed.

## 6. Extending the platform

| To add | Do this |
|---|---|
| an indicator, pattern or operator | write a causal function; register it with `technical.register_function`. It is then usable by name, in formulas and in the catalogue. The prefix-invariance test will check it for look-ahead. |
| a strategy without code | a formula spec for `strategy.create_from_expressions`, or rules for `StrategyBuilder` |
| a strategy in code | subclass `strategy.Strategy`, register with `@STRATEGIES.register` |
| a model-driven strategy | wrap the model in a policy function and pass it to `strategy.PolicyStrategy` |
| a broker | subclass `BrokerGateway`, register with `@BROKERS.register`, map its symbols in the `InstrumentMaster` |
| a ranking method | decorate a function with `@selection.ranker("name")` |
| a volatility model | subclass `volatility.VolModel` and add it to `volatility.MODELS` |
| a notification channel | subclass `notify.Channel`; raise `DeliveryError` for transient failures, `NotificationError` for permanent ones |
| an alert | `notify.threshold(...)` or an `AlertRule` with any condition over the metrics dictionary |
| a package | create it, assign it to a job group and give it an allow-list in `architecture.py` |

## 7. State that survives a restart

| What | Where it is kept | Notes |
|---|---|---|
| Orders, fills, positions, kill-switch state | engine journal (`persistence`), one SQLite file per session | rebuilt and reconciled with the broker on start-up; any mismatch halts trading |
| Users and roles | web server's `app.db` | password hashes only |
| Instruments, strategy approvals and their history, model registry metadata, alert rules, notification routes, open critical alerts, undelivered notifications | `TradingPlatform(store=...)`, key `platform.*` | loaded at start-up, saved after every change made through a desk; each part loads independently |
| Notification channels holding a secret | **not stored** | listed as unresolved after a restart; register them again |
| Fitted model objects | **not stored** (no pickling) | re-attach with `ModelRegistry.attach`, checked against the training-data fingerprint |
| Alert rules written as code | **not stored** | only threshold-style rules can be written down |
| Downloaded history (the shared library) | `TradingPlatform(store=...)`, tables `bar_series` and `bars` (schema v3, one row per bar) | written as it changes, before memory; capped at 400,000 bars; history saved by an earlier version under `platform.history` is moved once |
| Data-provider settings and the sample-data-only lock | key `platform.data` | IB settings come back connected; Alpaca keys are **not stored** (listed as `needs_keys`) unless the environment supplies them |

## 8. The web UI follows the same structure

The menu of the web UI is grouped by the job groups of section 1, in the same order, and takes each group's title and
question from `GET /api/structure`, so it cannot drift from `architecture.py`. Every group has at least one page except
Foundation, which is shared code with nothing to show. A role sees a page only if it holds the permission the page
needs; the server checks that permission again on every request, and `tests/test_web_groups.py` fails if the menu and
the server ever disagree for any role.

| Job group | Pages | Needs |
|---|---|---|
| Access control | Users and roles | `users:manage` |
| Market and reference data | Price data (sample generator, downloads, history library, uploads) · Data providers · Instruments (operations loads contract specs) | `backtest:run` · `system:operate` · `portfolio:view` |
| Signal libraries | Technical lab · News sentiment | `backtest:run` |
| Quantitative research | Volatility · Option chain | `backtest:run` |
| Learning and AI | Prediction models · Regimes and change points | `model:train` · `backtest:run` |
| Strategy | Strategy builder · Instrument selection | `backtest:run` |
| Portfolio and risk | Multi-strategy portfolio · Margin · Currency exposure | `backtest:run` · `risk:view` · `portfolio:view` |
| Trading | Paper trading · Order-book simulator | `order:place` · `backtest:run` |
| Validation | Backtest · Optimise (8 search methods) · Walk-forward | `backtest:run` |
| Operations and control | Strategy approvals · Alerts and inbox · Health and jobs · Audit log | `order:view` · `risk:view` · `system:operate` · `audit:view` |
| Application | Home: your role, desks and groups, and the whole map | everyone |

The API is split the same way: `web/routes/<group>.py`, one module per job group (`access`, `market_data`, `signals`,
`quant`, `learning`, `strategy`, `capital`, `trading`, `validation`, `operations`, `application`). Around them:

| File | Layer | Job |
|---|---|---|
| `web/http.py` | middleware | request parsing and size limit, JSON-only bodies, static files confined to `static/`, security headers, the authenticated event stream, the threaded server |
| `web/server.py` | application | `App`: the route table, token check and dispatch, mapping platform errors to HTTP codes, the data source of a request, shared state (sessions, uploads), closing every database |
| `web/session.py` | trading | the paper-trading feed (sample random walk or replayed history) and the metrics alert rules read |
| `web/common.py` | shared | JSON-safe conversion, chart series, small parsing helpers |
| `web/routes/*.py` | API | the routes of one job group each |

`tests/test_web_groups.py` checks that the route modules are exactly the job groups, that no route is registered twice,
and that each page of the UI calls routes of its own group. To let the web
layer show each group, its allow-list includes the libraries it displays (`derivatives`, `research`, `ml`, `dl`, `ai`,
`brokers` for the order-book simulator); the libraries themselves still import nothing from the web or the platform.

## 9. Market data: sample, downloaded, uploaded

| Source | Where it comes from | Who | Used by |
|---|---|---|---|
| Sample | `data/synthetic.py` scenarios through `SampleDataProvider`: seeded, any symbol and date range | everyone with research | every research page, paper trading (random walk) |
| Downloaded history | `AlpacaDataProvider` or `IBDataProvider` (`data/providers.py`) into `HistoryLibrary` (`data/library.py`), shared by all users | operations connects; research downloads | every research page (`source="history"`), paper trading replay |
| Uploaded | the user's own CSV files, in server memory | the uploading user only | every research page (`source="uploaded"`) |

`app/marketdata.py` (`TradingPlatform.data`) holds the connections, the library and the sample-data-only lock; the desks
expose it (research: `fetch_history`, `remove_history`; operations: `connect_data_provider`, `test_data_provider`,
`disconnect_data_provider`, `set_sample_data_only`). Providers live in `data`, which imports only `core`: a provider is
given its keys, it never looks them up, so the credential vault stays in the access-control layer.
