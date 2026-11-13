# Progress record

This file is the single source of truth for what has been built, what has been verified, and what is still pending.
It is updated at the end of every unit of work. Rules for keeping it honest are in [`CLAUDE.md`](CLAUDE.md).

**Last updated:** 2026-11-13, after round six (packages moved into one folder per job group)
**Working branch:** `ccr-b7d3820c-a4gug7`
**Default branch:** `claude/trading-platform-system-y9ke4s`

**Merge state.** Pull request #2 merged the first part of this work, and pull request #3 merged round two (at 08:44
UTC on 2026-11-13, head `e3d8918`). Round three (the web UI regrouped by job group, commit `a1b58dd`) was pushed after
that merge. Rounds three to five went in together through **pull request #4**
(https://github.com/olmahelp-source/EPAT-LLD/pull/4), merged by the owner at 14:00 UTC on 2026-11-13 (head `568faae`,
merge commit `1f1708b`). Round six (every package moved into its job group's folder, commit `df21697`) went in
through **pull request #5** (https://github.com/olmahelp-source/EPAT-LLD/pull/5), merged at the owner's request at
14:46 UTC on 2026-11-13 (merge commit `b369b69`).

**`main`** had had nothing new since #1 (2026-11-10). The owner merged **pull request #7** (the default branch into
`main`, rounds two to six) at 15:08:59 UTC, and **pull request #6** (progress notes, into the default branch) at 15:09:15.
Because #7 went first, **pull request #8** brought #6 into `main` as well (merge commit `2223a8a`, 15:11:30 UTC). `main` and
the default branch now hold the same files, and the working branch was fast-forwarded to the default branch (`683ba6d`).
Only this note is waiting to be merged; it goes in with the next pull request.

---

## 1. Where things stand

| Area | State |
|---|---|
| Every item of the original request | **Done.** Code and passing tests for all 27 items. |
| Items found missing in the gap analysis | **Done** except live-service verification (cannot be done here, see section 6). Margin, multi-currency accounting and an order-book simulator were built in round two. |
| Size of the work | 77 new source files (two of them added in round three) plus the modified files listed in section 5: about 13,400 lines of source and UI and 5,350 lines of tests, on top of the original platform. Round three alone: +1,090 lines of source and UI, +242 lines of tests (`git diff --numstat`). |
| Whole test-suite | **823 tests pass** (407 original + 416 new), with deprecation warnings treated as errors. |
| Dependency rules ("do not mix concerns") | **Verified.** `python -m algoplatform structure --check` reports no violation; the same check is a unit test. |
| Linter (`ruff`) | **Clean.** |
| Type checker (`mypy`) | **No findings in the files added by this work.** 109 findings remain in files that predate it (the CI job stays informational). |
| Web UI | **Done, grouped by job.** The menu follows the 12 job groups of `architecture.py` (titles come from the server); 25 pages; every group except Foundation has a page. A Sample / Real data switch in the header. Checked in a headless browser as all eight roles, on desktop and phone width. |
| Restart safety of the shared services | **Done.** Including downloaded history and data-provider settings (round four); API keys are never written to disk. |
| Market data (round four) | **Done, tested against fakes.** Sample generator with 8 scenarios; Alpaca and Interactive Brokers history providers; shared history library; sample-data-only lock; paper trading on replayed history. No call made with real Alpaca keys or a real IB gateway (see section 6). |
| Example scripts | All three run. |
| CI on GitHub | Round three (run 24, commit `a1b58dd`): lint, tests on Python 3.10 to 3.13 and the optional-dependencies job all passed. The type-check job fails as before (informational): 112 findings, all in files that predate this work. Every earlier commit since the lint fixes passed; see section 7. |

---

## 2. What the original request missed (gap analysis)

The request listed 27 areas. A production trading platform also needs the following. Each was checked against the
repository.

| Missing from the request | Status in the repository |
|---|---|
| Market-data quality (validation, cleaning, gaps, split adjustment) | Already present: `data` |
| Corporate actions and continuous futures | **Added:** `market_data/instruments/corporate_actions.py`, `market_data/instruments/continuous.py` |
| Execution algorithms and transaction-cost analysis | Already present: `execution`, `validation/analytics/tca.py` |
| Durable state, crash recovery, broker reconciliation | Already present: `persistence` |
| Kill switch and live risk monitor | Already present: `risk` |
| Audit trail | Already present: `foundation/core/logging.py` (`AuditLog`) |
| Segregation of duties between roles | **Added:** 4 new roles, 9 new permissions, role desks (`application/app/workspaces.py`) |
| Strategy lifecycle with approval (research, paper, live) | **Added:** `operations/governance/lifecycle.py` (four-eyes promotion) |
| Model registry (versions, training-data fingerprint, rollback) | **Added:** `operations/governance/models.py` |
| Health monitoring and heartbeats | **Added:** `operations/ops/health.py` |
| Job scheduler | **Added:** `operations/ops/scheduler.py` |
| Market-regime detection | **Added:** `learning/ai/hmm.py`, `quantitative/volatility/management.py` |
| Overfitting control (walk-forward, deflated Sharpe, purged CV) | Already present: `optimize`, `stats`, `research`; scorecard **added** in `validation/analytics/evaluation.py` |
| Enforced separation of concerns | **Added:** `architecture.py` (allow-list of dependencies, checked by a test) |
| Margin model (scenario / portfolio margin) | **Added (round two):** `capital/risk/margin.py`; default parameters, not an exchange schedule |
| Multi-currency accounting and FX conversion | **Added (round two):** `capital/portfolio/currency.py` |
| Tick / order-book data and latency simulation | **Added (round two):** `trading/brokers/orderbook.py`; a laboratory beside the bar-based engine |
| Restart safety of approvals, alerts, routes, instruments, models | **Added (round two):** `TradingPlatform(store=...)` |
| Live broker certification | **Not possible here.** Adapters are tested against fakes only. |

---

## 3. Requirement coverage

| # | Requested item | Where it lives | Code | Tests |
|---|---|---|---|---|
| 1 | Library with error handling | `foundation/core/exceptions.py` (typed tree, `retryable` flag); 9 exception types added | done | passing (`test_core_utils`, `test_resilience`) |
| 2 | Advanced error handling | `foundation/utils/resilience.py`: classification and policy, `Result`, fallback chain, timeout, deadline, bulkhead, error budget, error boundary, dead-letter queue, collector, context | done | passing (`test_resilience`) |
| 3 | Multi-broker and multi-user | `users`, `brokers` (existing); 8 roles and 20 permissions in `foundation/core/enums.py`, `access/users/security.py` | done | passing (existing, `test_architecture_desks`) |
| 4 | Order management | `orders`, `execution` (existing) | done | passing (existing) |
| 5 | Risk management | `risk` (existing) | done | passing (existing) |
| 6, 17 | Position sizing and fund allocation | `portfolio` (existing) | done | passing (existing) |
| 7 | Instrument data helpers | `instruments`: contract specs, instrument master with broker symbol mapping, corporate actions, continuous futures | done | passing (`test_instruments_options`) |
| 8 | Option data handling | `quantitative/derivatives/chain.py`: option chain, implied volatility, Greeks, put/call ratio, max pain, skew, term structure, quality report | done | passing (`test_instruments_options`) |
| 9 | Technical indicators | `technical`: 55 indicators (core set re-exported from `quantitative/quant/indicators.py`) | done | passing (`test_technical`) |
| 10 | Fundamental indicators | `fundamentals` (existing) | done | passing (existing) |
| 11 | Pattern identifiers | `signals/technical/candles.py` (19 candlestick patterns), `signals/technical/chart_patterns.py` (double top/bottom, head and shoulders, triangles, wedges, channels, range breakout, divergence), `signals/technical/levels.py` (swings, support/resistance, pivots) | done | passing (`test_technical`) |
| 12 | Functions for operations and indicators | `signals/technical/operators.py` (43 operators), `signals/technical/catalog.py` (run any function by name), `signals/technical/expression.py` (safe formula engine) | done | passing (`test_technical`) |
| 13 | Strategy creator | `strategies/strategy/builder.py` (existing), `strategies/strategy/expression.py` (formula strategies), `strategies/strategy/policy.py` (model adapter) | done | passing (`test_technical`, existing) |
| 14 | Backtesting and optimisation | `backtest`, `optimize` (existing) | done | passing (existing) |
| 15 | Instrument selection | `selection`: filters, 9 rankers, external scores, diversification, group caps, one-per-cluster, pair selection | done | passing (`test_search_selection_evaluation`) |
| 16 | Relationship management | `relationships`: correlation, beta, tail dependence, lead-lag, Granger causality, cointegration scan, clustering, spanning tree, PCA, `RelationshipManager` | done | passing (`test_relationships_volatility`) |
| 16b | Sentiment and news analysis | `sentiment` (existing); `signals/sentiment/llm.py` (language-model scorer with lexicon fallback) | done | passing (existing, `test_notify_ops_governance`); see limits |
| 17b | Statistical methods | `stats` (existing) | done | passing (existing) |
| 18 | Quantitative methods | `quant` (existing) | done | passing (existing) |
| 19 | Volatility management and prediction | `volatility`: 5 estimators, volatility cone, EWMA / GARCH / GJR / HAR forecasters, forecast evaluation, volatility targeting, regime classifier | done | passing (`test_relationships_volatility`) |
| 20 | Machine learning | `ml`: decision tree, random forest, gradient boosting, k-nearest neighbours, naive Bayes, lasso, PCA, k-means, voting and stacking, walk-forward prediction; `research` (existing) | done | passing (`test_learning`) |
| 21 | Deep learning | `dl`: Dense, LSTM, Dropout, Adam, training loop, MLP, LSTM forecaster, autoencoder (numpy; gradients checked numerically) | done | passing (`test_learning`) |
| 22 | Reinforcement learning | `rl`: trading environment, Q-learning, deep Q-network, policy gradient, three bandits | done | passing (`test_learning`) |
| 23 | Advanced parameter optimisation | `validation/optimize/advanced.py`: Bayesian, TPE, particle swarm, differential evolution, simulated annealing, CMA-ES, successive halving, Pareto front | done | passing (`test_search_selection_evaluation`) |
| 24 | Other AI methods | `ai`: hidden Markov regimes, fuzzy logic, genetic programming, isolation forest, change-point detection | done | passing (`test_learning`) |
| 25 | Notification and alert | `notify`: 7 channels, notifier (retry, dedupe, private routing, dead letters, escalation), alert rules | done | passing (`test_notify_ops_governance`); see limits |
| 26 | Analytics | `analytics` (existing) | done | passing (existing) |
| 27 | Strategy result analysis | `validation/analytics/evaluation.py`: bootstrap confidence intervals, drawdown probability, stability, regime breakdown, trade concentration, comparison, pass/fail scorecard | done | passing (`test_search_selection_evaluation`) |
| - | One platform, different roles | `application/app/workspaces.py`: research, trading, portfolio, risk, operations, compliance, admin desks | done | passing (`test_architecture_desks`) |
| - | Structure grouped by job | `architecture.py`: 12 job groups, dependency allow-list, `check_architecture()` | done | passing (`test_architecture_desks`) |
| - | Operations and governance (from the gap analysis) | `ops`, `governance` | done | passing (`test_notify_ops_governance`) |
| - | Margin (from the gap analysis) | `capital/risk/margin.py`: rule-based and scenario margin, account report, liquidation price, pre-trade `MarginCheck` | done | passing (`test_margin_currency_orderbook`) |
| - | Multi-currency accounting (from the gap analysis) | `capital/portfolio/currency.py`: point-in-time FX rates with cross rates, cash per currency, exposure, FX P&L attribution, hedge amounts | done | passing (`test_margin_currency_orderbook`) |
| - | Order book and tick simulation (from the gap analysis) | `trading/brokers/orderbook.py`: matching engine, queue position, cost to trade, seeded simulator with latency, tick-to-bar | done | passing (`test_margin_currency_orderbook`) |
| - | Restart safety of the shared services | `save_state` / `load_state` / autosave in `application/app/platform.py`; `export_state` / `import_state` on each service | done | passing (`test_state_persistence`, `test_web_quant`) |
| - | Web UI for the new features | `application/web/quant_api.py`, six pages in `application/web/static/app.js` | done | passing (`test_web_quant`); browser walkthrough |
| - | Sample data generator, historical data from a broker or data provider, sample/real switch (round four) | `market_data/data/synthetic.py` (scenarios), `market_data/data/providers.py`, `market_data/data/library.py`, `application/app/marketdata.py`, `application/web/data_api.py`, Price data and Data providers pages, header switch, paper-trading replay | done | passing (`test_data_providers`, `test_market_data_service`, `test_web_data`); browser walkthrough with a fake Alpaca |
| - | Web UI grouped by job (round three) | `application/web/static/app.js` (menu by job group, hash routing, Home), `application/web/research_api.py` (option chain, sentiment, prediction models, regimes), `application/web/desk_api.py` (instruments, margin, currency, order book, health), 8 optimiser methods, audit log on the compliance permission | done | passing (`test_web_groups`); browser walkthrough as 8 roles |

---

## 4. Structure (grouped by job)

Print it at any time with `PYTHONPATH=src python -m algoplatform structure`. Full description: `docs/ARCHITECTURE.md`.

| # | Job group | Packages |
|---|---|---|
| 1 | Foundation | `core`, `utils` |
| 2 | Access control | `users` |
| 3 | Market and reference data | `data`, `instruments` |
| 4 | Signal libraries | `technical`, `fundamentals`, `sentiment` |
| 5 | Quantitative research | `stats`, `quant`, `derivatives`, `relationships`, `volatility` |
| 6 | Learning and AI | `research`, `ml`, `dl`, `rl`, `ai` |
| 7 | Strategy | `strategy`, `selection` |
| 8 | Portfolio and risk | `portfolio`, `risk` |
| 9 | Trading | `brokers`, `orders`, `execution` |
| 10 | Validation | `backtest`, `optimize`, `analytics` |
| 11 | Operations and control | `persistence`, `notify`, `ops`, `governance` |
| 12 | Application | `app`, `web` |

Since round six every package lives in its job group's folder: `src/algoplatform/<group>/<package>`, for example
`capital/risk`, `signals/technical`, `application/web` (folders: `foundation`, `access`, `market_data`, `signals`,
`quantitative`, `learning`, `strategies`, `capital`, `trading`, `validation`, `operations`, `application`). The two
groups whose key matched a package inside them were renamed (`quant` → `quantitative`, `strategy` → `strategies`), and
so were their route module and page script. `check_architecture()` fails if a package sits at the top level or in
another group's folder. File paths elsewhere in this record are relative to `src/algoplatform`.

---

## 5. File inventory

### New packages (all files new)

| Package | Files |
|---|---|
| `signals/technical/` | `trend.py`, `momentum.py`, `bands.py`, `volume.py`, `levels.py`, `candles.py`, `chart_patterns.py`, `operators.py`, `catalog.py`, `expression.py` |
| `market_data/instruments/` | `contracts.py`, `master.py`, `corporate_actions.py`, `continuous.py` |
| `quantitative/relationships/` | `correlation.py`, `leadlag.py`, `cointegration.py`, `structure.py`, `manager.py` |
| `quantitative/volatility/` | `estimators.py`, `models.py`, `evaluation.py`, `management.py` |
| `learning/ml/` | `tree.py`, `classic.py`, `unsupervised.py`, `ensemble.py`, `metrics.py`, `pipeline.py` |
| `learning/dl/` | `nn.py`, `models.py` |
| `learning/rl/` | `env.py`, `agents.py`, `bandits.py` |
| `learning/ai/` | `hmm.py`, `fuzzy.py`, `genetic_programming.py`, `anomaly.py`, `changepoint.py` |
| `strategies/selection/` | `rankers.py`, `selector.py` |
| `operations/notify/` | `message.py`, `channels.py`, `notifier.py`, `alerts.py` |
| `operations/ops/` | `health.py`, `scheduler.py` |
| `operations/governance/` | `lifecycle.py`, `models.py` |

Each package also has an `__init__.py` that exports its public names.

### New files in existing packages

`market_data/data/providers.py`, `market_data/data/library.py`, `application/app/marketdata.py`, `application/web/data_api.py` (round four); `application/web/research_api.py`, `application/web/desk_api.py` (round three); `capital/risk/margin.py`, `capital/portfolio/currency.py`, `trading/brokers/orderbook.py`, `application/web/quant_api.py` (round two);
`foundation/utils/resilience.py`, `foundation/utils/numeric.py`, `quantitative/derivatives/chain.py`, `validation/optimize/advanced.py`, `validation/analytics/evaluation.py`,
`signals/sentiment/llm.py`, `strategies/strategy/expression.py`, `strategies/strategy/policy.py`, `application/app/workspaces.py`, and the top-level
`architecture.py`.

### Modified source files

`foundation/core/exceptions.py` (9 exception types), `foundation/core/enums.py` (4 roles, 9 permissions, `Severity`), `foundation/core/__init__.py`,
`access/users/security.py` (role-to-permission table), `foundation/utils/__init__.py`, `strategies/strategy/__init__.py`, `quantitative/derivatives/__init__.py`,
`signals/sentiment/__init__.py`, `validation/analytics/__init__.py`, `validation/optimize/__init__.py`, `validation/optimize/optimizers.py` (dispatcher knows the
new methods), `quantitative/stats/tests.py` (half-life fix, see section 8), `application/app/platform.py` (new shared services, `desk()` /
`desks()`), `application/app/__init__.py`, `application/app/cli.py` (`structure` and `functions` commands, new optimiser names),
`application/web/static/app.js` and `application/web/static/styles.css` (rebuilt around the job groups in round three), `application/web/server.py` (registers the new route files, optimiser methods, audit permission), `architecture.py` (web allow-list), `application/app/workspaces.py` (`load_instruments(overwrite=...)`), `algoplatform/__init__.py` (version 0.2.0). Round four: `foundation/core/exceptions.py` (three data-provider errors), `market_data/data/synthetic.py` (scenarios), `market_data/data/__init__.py`, `application/app/platform.py` (`TradingPlatform.data`, parts saved only when changed), `application/app/workspaces.py` (data desk methods), `application/web/server.py` (data source of every request, history replay), `application/web/desk_api.py` (market-data health check), `application/web/static/*` (switch, Price data and Data providers pages), `pyproject.toml` (`ib` extra).

### Other new or changed files

`PROGRESS.md`, `CLAUDE.md`, `docs/ARCHITECTURE.md`, `examples/run_quant_lab.py`, `README.md` (updated),
`pyproject.toml` (version, optional `llm` extra).

### Test files

| File | Tests | State |
|---|---|---|
| 20 original `tests/test_*.py` files | 407 | passing |
| `tests/test_resilience.py` | 24 | passing |
| `tests/test_technical.py` | 50 | passing |
| `tests/test_instruments_options.py` | 24 | passing |
| `tests/test_relationships_volatility.py` | 32 | passing |
| `tests/test_learning.py` (ml, dl, rl, ai) | 45 | passing |
| `tests/test_search_selection_evaluation.py` | 30 | passing |
| `tests/test_notify_ops_governance.py` (incl. `signals/sentiment/llm.py`) | 46 | passing |
| `tests/test_architecture_desks.py` | 27 | passing |
| `tests/test_margin_currency_orderbook.py` | 39 | passing |
| `tests/test_state_persistence.py` | 19 | passing |
| `tests/test_web_quant.py` | 15 | passing |
| `tests/test_web_groups.py` (menu, scripts and API follow the structure; the menu matches the server; routes) | 20 | passing |
| `tests/test_data_providers.py` (scenarios, Alpaca and IB against fakes, history library) | 18 | passing |
| `tests/test_market_data_service.py` (connections, keys, downloads, sample-only lock, environment, restart) | 8 | passing |
| `tests/test_web_data.py` (data routes, source of every request, lock on the server, history replay and its restart) | 9 | passing |
| `tests/test_cli.py` (data sources, data commands with a fake Alpaca, the shared lock, the web command) | 7 | passing |
| **Total** | **823** | **passing** |

---

## 6. Pending tasks

**Round six, done** (requested: "move packages into one folder per job group"):

| Part | State |
|---|---|
| 34 packages moved with `git mv` into 12 group folders (history kept: git records 200 renames); a group `__init__.py` names each group's job | done |
| Every relative import recomputed from the file's new location (190 files); absolute `algoplatform.<package>` names updated in tests, examples and documents (45 files) | done |
| Group keys `quant` → `quantitative`, `strategy` → `strategies` (they clashed with packages inside them); route modules and page scripts renamed to match | done |
| `check_architecture()` for the new layout: packages found inside group folders; a package at the top level, in another group's folder, or unassigned is a violation; `architecture.package_dir()` | done, tested |
| Packaging: the wheel shipped only `static/*` (missing the 13 scripts added in round five); now ships `static/js/**` too; console script checked from an installed wheel | done, verified in a fresh virtual environment |


**Round five, done** (requested: audit the whole file structure; remove messy, duplicate and unnecessary files;
check frontend, backend, middleware, database and API against the platform's rules; a switch for which sample data to use).

| Step | State |
|---|---|
| Audit: inventory of all 238 tracked files; nothing generated is tracked; every module is imported; no duplicated function bodies; two unused names | **done** |
| Small fixes: the order manager refuses a reused order id (`DuplicateOrderError`) instead of overwriting; unused `make_loopback` removed; ignore rules for tool caches and logs; 39 local cache folders and a generated report deleted | **done, tested** |
| Database: downloaded history moved from one JSON document to `bar_series` / `bars` tables (schema v3, upgrade in place), written as it changes; old-format history moved once; `App.close()` closes every database (no more unclosed-connection warnings in the web tests) | **done, tested** |
| CLI updated (requested during round five): `backtest` / `optimize` take `--source sample|history|csv` and `--scenario`; new `data` commands (scenarios, sample, providers, fetch, list, export, remove, sample-only) on the same state directory and service as the web server; `web` command through the top-level launcher (the installed `algoplatform` command now points there) | **done, tested** (`test_cli`) |
| API: the 60 routes moved unchanged from five mixed files into `application/web/routes/<group>.py`, one module per job group; `application/web/http.py` (middleware), `application/web/session.py` (paper feed), `application/web/common.py` (shared helpers); `application/web/server.py` down from 912 to 289 lines (the app only); `quant_api.py`, `research_api.py`, `desk_api.py`, `data_api.py` deleted; tests check route modules = job groups and each page calls its own group's routes; 3 old type findings fixed on the way | **done, tested** (full suite, browser walkthrough) |
| Frontend: `app.js` (1,744 lines) split into `js/core.js`, `js/pages/<group>.js` (one per job group, same names as the route modules) and `js/shell.js`; a sample-scenario dropdown beside the Sample / Real switch sets the scenario every data form and the generator start with; tests check each page lives in its group's script and the load order | **done, tested** (full suite, three browser walkthroughs) |


**Round four, done** (requested: "add a sample data generator, or historical data from a broker (Alpaca or IB), or
connect a data provider; add a switch for sample data"). All three were built:

| Part | Where | State |
|---|---|---|
| Sample data generator: 8 named scenarios, daily or intraday, seeded, symbol-stable | `market_data/data/synthetic.py`, `SampleDataProvider`; Price data page; every data form | done, tested |
| Historical data from Alpaca (stocks and crypto) and Interactive Brokers (TWS / IB Gateway) | `market_data/data/providers.py`; Data providers page (operations) and Price data page (download) | done, **tested against fakes only** |
| A data provider connection: keys in the credential vault, or environment variables at start-up | `application/app/marketdata.py` | done, tested |
| Shared history library, saved with the platform state | `market_data/data/library.py` | done, tested |
| Switch for sample data: per browser in the header (Sample / Real), and a platform-wide sample-data-only lock for operations that the server enforces | `application/web/static/app.js`, `application/app/marketdata.py` | done, tested |
| Paper trading on replayed downloaded history (survives a restart) | `application/web/server.py` `LiveSession` | done, tested |

Rounds two to four completed every pending item that can be done from this environment. What is left:

| # | Task | State | Why it is not done |
|---|---|---|---|
| 1 | (none waiting) | - | Pull requests #6, #7 and #8 were merged; `main` and the default branch hold the same files. |
| 2 | Run the integrations against real services: Claude sentiment client, Slack, Telegram, webhook, e-mail, broker sandboxes, **Alpaca market data, Interactive Brokers history** | **blocked** | Needs the owner's credentials and accounts (and, for IB, a running TWS or IB Gateway); nothing here should send messages or orders to real services. All are tested against fakes. Alpaca's data server answered from this environment (HTTP 401 without keys), so the Alpaca check can be run here once keys are supplied: set them on the Data providers page and press Test. |
| 3 | Type findings in the 109 places in files that predate this work | open, optional | Older code, outside the request; the CI type-check job is informational. |
| 4 | (done in round six) Move packages into one folder per job group | **done** | See section 4. |

Done in round two (previously pending): web UI for the new features; persistence of the new shared services; type
findings in new files; margin model; multi-currency accounting; order-book and tick simulator; documentation.

Done in round three (requested: "update the frontend as per the new structure"): the menu is grouped by job group;
pages added for the groups and features that had none (instruments, news sentiment, option chain, prediction models,
regimes and change points, margin, currency exposure, order-book simulator, health and jobs, audit log, users and
roles, Home); all eight optimiser methods on the Optimise page.

---

## 7. Verification log

| When | Command | Result |
|---|---|---|
| 2026-11-12 | `cd tests && python -m unittest discover -p "test_*.py"` (before any change) | 407 tests, OK |
| 2026-11-12 | same, after all source changes, with `-W error::DeprecationWarning:algoplatform` | 407 tests, OK |
| 2026-11-12 | each new test file on its own | 24 + 50 + 24 + 32 + 45 + 30 + 46 + 26 = 277 tests, OK |
| 2026-11-12 | `cd tests && python -W error::DeprecationWarning:algoplatform -m unittest discover -p "test_*.py"` (final) | **684 tests, OK** (about 60 s) |
| 2026-11-12 | `python -m ruff check .` | 6 unused imports found and removed; then "All checks passed!" |
| 2026-11-12 | `PYTHONPATH=src python -m algoplatform structure --check` | 12 groups, 34 packages, "dependency rules: OK" |
| 2026-11-12 | `python examples/run_demo.py`, `run_advanced.py`, `run_quant_lab.py` (deprecation warnings as errors) | all exit 0; `run_quant_lab` takes about 9 s |
| 2026-11-12 | static scan of 211 `.py` files for syntax and library features newer than Python 3.10 | none found |
| 2026-11-12 | import of `algoplatform.web.server` | OK |
| 2026-11-12 | GitHub Actions, workflow "CI", run 11 (all code, tests, documentation and lint fixes) | lint: success; test 3.10, 3.11, 3.12, 3.13: success; optional-dependencies: success; typecheck (informational): failure |
| 2026-11-12 | GitHub Actions, runs 6 to 9 (commits before the lint fixes) | failed in the lint job only: six unused imports, since removed |
| 2026-11-12 | `python -m mypy src/algoplatform` | 178 findings in 50 files: 112 in pre-existing files, 66 in new files |
| 2026-11-13 | `cd tests && python -m unittest test_margin_currency_orderbook test_state_persistence test_web_quant` | 39 + 19 + 15 tests, OK |
| 2026-11-13 | full suite with deprecation warnings as errors (exit status checked) | **757 tests, OK** |
| 2026-11-13 | `python -m mypy src/algoplatform` | 0 findings in new files; 113 in pre-existing files |
| 2026-11-13 | `python -m ruff check .`; `python -m algoplatform structure --check` | clean; dependency rules OK |
| 2026-11-13 | headless browser (Playwright) against a live server: quant, trader, portfolio manager, risk manager, viewer | every new tab loads and works; tabs hidden from roles that cannot use them; a personal alert fired into the trader's inbox; the trader could not acknowledge a kill-switch alert, the risk manager could; no layout overflow at phone width; no JavaScript errors |
| 2026-11-13 | `cd tests && python -m unittest test_web_groups` | 16 tests, OK |
| 2026-11-13 | full suite with deprecation warnings as errors (exit status checked) | **773 tests, OK** |
| 2026-11-13 | `python -m ruff check .`; `python -m algoplatform structure --check`; `python -m mypy src/algoplatform` | clean; dependency rules OK; 0 findings in new files, 112 in pre-existing files |
| 2026-11-13 | headless browser against a live server with `--db-dir`, all eight roles, desktop and 390 px phone width | each role's menu holds exactly its pages; every page's main action works (option chain, sentiment, 7 models, regimes, margin from entered and from paper positions, currency, order book, instrument load, health, audit filter, Bayesian optimisation); a forbidden deep link falls back to Home; phone menu opens and closes; no page wider than the screen; no "null", "undefined" or "NaN" on any page; no JavaScript errors; loaded instruments still there after a server restart |
| 2026-11-13 | GitHub Actions, workflow "CI", run 24 (commit `a1b58dd`, round three) | lint: success; test 3.10, 3.11, 3.12, 3.13: success; optional-dependencies: success; run conclusion: success; typecheck (informational): failure, 112 findings, same files as locally, none in files added by this work |
| 2026-11-13 | GitHub Actions, runs 18 to 22 (round two, including pull-request runs 20 and 22) | all success |
| 2026-11-13 | `cd tests && python -m unittest test_data_providers`; full suite (exit status checked); ruff; structure check; mypy on `market_data/data/` | 18 tests OK; **791 tests, OK**; clean; dependency rules OK; no findings in the new data files |
| 2026-11-13 | `cd tests && python -m unittest test_market_data_service test_state_persistence`; full suite (exit status checked); ruff; structure; mypy | 27 tests OK; **799 tests, OK**; clean; OK; 112 findings, all in pre-existing files |
| 2026-11-13 | `cd tests && python -m unittest test_web_data test_web_groups` | 25 tests, OK |
| 2026-11-13 | full suite with deprecation warnings as errors (exit status checked); ruff; structure check; mypy | **808 tests, OK**; clean; dependency rules OK; 112 findings, all in pre-existing files |
| 2026-11-13 | headless browser against a live server with `--db-dir` and a fake Alpaca transport | operations connected Alpaca with a wrong key (Test reported the 401) then the right one (Test ok); the key never appeared in the page; quant previewed a crash scenario, downloaded SPY, AAPL, MSFT and an unknown symbol (reported, the rest loaded), charted a series; after switching the header to Real, Backtest and Technical lab started on downloaded history and ran; the trader replayed history in paper trading; after operations locked to sample data, the header showed the lock, history options were disabled, downloads were off and a history replay step was refused by the server; no page wider than a phone; no "null" on any page; no JavaScript errors |
| 2026-11-13 | `curl https://data.alpaca.markets/v2/stocks/SPY/bars` without keys | HTTP 401: the host is reachable from this environment; nothing else was sent |
| 2026-11-13 | GitHub Actions run 28 (round four, commit `bc4f137`); runs 25 to 27 | all success |
| 2026-11-13 | round five audit script (unimported modules, unreferenced names, identical function bodies) | 0 unimported modules, 2 unused names (fixed), 0 duplicate bodies |
| 2026-11-13 | full suite after the database change (exit status checked); ruff; structure; mypy; resource warnings in the web tests | **811 tests, OK**; clean; OK; 112 pre-existing findings; 0 unclosed connections |
| 2026-11-13 | `cd tests && python -m unittest test_cli`; full suite (exit status checked); ruff; structure; mypy | 7 tests OK; **818 tests, OK**; clean; OK; 112 pre-existing findings |
| 2026-11-13 | after the API restructure: full suite (exit status checked); ruff; structure; mypy; browser walkthrough with a fake Alpaca; every page as admin checked for stray text | **820 tests, OK**, no warnings in the log; clean; OK; 109 pre-existing findings; walkthrough passed (no JavaScript errors, no page wider than a phone, no "null") |
| 2026-11-13 | after the frontend split: `test_web_groups`, `test_web`; full suite (exit status checked); ruff; structure; mypy; browser: scenario dropdown, data walkthrough with a fake Alpaca, all-roles walkthrough on a fresh server, every page as admin | 29 tests OK; **822 tests, OK**; clean; OK; 109 pre-existing findings; all walkthroughs passed (13 scripts load, no JavaScript errors, no page wider than a phone, no "null") |
| 2026-11-13 | GitHub Actions runs 29 to 33 (round five, commits `b5c45e0` to `36a432f`) | all success |
| 2026-11-13 | pull request #4 opened (GitHub API) | https://github.com/olmahelp-source/EPAT-LLD/pull/4, base `claude/trading-platform-system-y9ke4s`, 11 commits, 63 files |
| 2026-11-13 | pull request #4 (GitHub API); CI runs 35 (push) and 36 (pull request) on `568faae` | merged at 14:00:57 UTC; lint, tests on Python 3.10 to 3.13 and optional-dependencies all success (typecheck informational) |
| 2026-11-13 | round six: full suite (exit status checked); ruff; structure check; mypy; examples; wheel built and installed in a fresh virtual environment (`algoplatform structure --check`, `algoplatform data scenarios`, static files present, API answers); browser: all-roles walkthrough, every page as admin, scenario check | **823 tests, OK**; clean; OK; 109 pre-existing findings; all 3 examples run; installed wheel works (15 static files, 12 group folders); walkthroughs passed, no JavaScript errors |
| 2026-11-13 | GitHub Actions run 39 (push, round six, `df21697`) | lint, tests on Python 3.10 to 3.13 and optional-dependencies all success (typecheck informational) |
| 2026-11-13 | pull request #5 (GitHub API) | opened and merged at the owner's request: base `claude/trading-platform-system-y9ke4s`, 2 commits, 283 files; merge commit `b369b69`; working branch fast-forwarded to it (`git merge --ff-only`) |
| 2026-11-13 | pull requests #6 and #7 opened (GitHub API); `git merge-tree --write-tree origin/main origin/claude/trading-platform-system-y9ke4s`, then `git diff --stat` of that tree against the default branch | https://github.com/olmahelp-source/EPAT-LLD/pull/6 and /pull/7; merge into `main` is clean and its tree equals the default branch |
| 2026-11-13 | pull requests #6 and #7 (GitHub API); CI run on #7 | #7 merged 15:08:59 UTC (`d172983`), #6 merged 15:09:15 (`683ba6d`); lint, tests on Python 3.10 to 3.13 and optional-dependencies all success (typecheck informational) |
| 2026-11-13 | pull request #8 opened and merged at the owner's request (GitHub API); `git diff --stat origin/main origin/claude/trading-platform-system-y9ke4s` | merge commit `2223a8a`; before: 1 file (`PROGRESS.md`), after: empty |
| 2026-11-13 | pull request #3 (GitHub API) | merged 08:44 UTC at `e3d8918`, before round three was pushed; default branch merged back into the working branch as `a1631ed` (`git diff a1b58dd a1631ed` is empty) |

---

## 8. Defects found and fixed during this work

Found by the new tests:

- `stats.ou_half_life` (existing code) divided by zero when the mean-reversion coefficient was a tiny negative
  number; a constant price series crashed the cointegration scan. Fixed with `log1p`; the scan now skips and reports
  pairs it cannot test.
- `ml.DecisionTree` raised `AttributeError` instead of `ModelError` when used before `fit`.
- `rl.QLearningAgent.act` did not fail until its first greedy call when `prepare` had not been called.
- `analytics.regime_breakdown` let a pandas `ValueError` escape for a label / return length mismatch; it now raises
  `ValidationError`.
- The VIEWER role could open no desk at all; it now has the read-only risk snapshot.

Found in round two:

- The governance form in the UI lost the typed strategy id after registering (the form was rebuilt on reload).
- The Paper trading and research tabs were shown to roles that cannot use them (a viewer saw Backtest).
- A web test assumed an exact catalogue size; it was committed once with that failure because a chained shell
  command did not stop on it. Fixed in the next commit; the full suite's exit status is now checked before committing.

Found in round three:

- The Paper trading page printed the word "null" under the algo-order table when there was no cost summary, and the
  Strategy approvals page did the same when no decision had been made yet (`replaceChildren(null)` writes "null").
- The audit-log route required the admin or risk-manager role, so an auditor (who has `audit:view` and the compliance
  desk) could not read it. It now follows `audit:view`.
- Re-loading a corrected instrument was always rejected as a duplicate, with no way to replace it. The operations desk
  now takes `overwrite`, recorded in the audit entry; without it a duplicate is still refused.
- The option-chain route treated an empty list of expiries as "use the defaults"; it is now refused.

Found in round four:

- The first version of the scenarios let the shared random noise swamp the trend: with one seed the "bull trend" ended
  lower than it started. The noise is now centred, so each scenario reliably looks like its name (checked for three
  seeds in `test_data_providers`).
- Skipping the save of an unchanged part was first keyed by part only, so an explicit save to a second store would
  have skipped unchanged history; it is now keyed by store and part (checked in `test_state_persistence`).
- The provider health check reported a provider as failing when one unknown ticker in a download failed; it now fails
  only when nothing in the download loaded.
- The Data providers page printed "null" when no provider came from the environment (`replaceChildren(null)`).

Found during development, before the tests were written:

- Bayesian search over-explored when scores were heavy-tailed (fixed with rank-based normal scores).
- TPE stalled at the centre of a cluster of mediocre points (fixed with a small "good" set, as in the reference method).
- The language-model scorer's circuit breaker did not count malformed answers as failures.

Wrong expectations in the tests themselves were corrected as tests, not by changing the source: about 20 across the
eight files (for example comparing against the engine's trade count, which includes rebalancing orders, or asserting
that a 90% interval always contains zero for one random sample).

---

## 9. Known limits (also in the README)

- Learning models are numpy implementations sized for daily data. They are not a substitute for a GPU framework.
- The reinforcement-learning and genetic-programming tests use synthetic data with a planted edge. They show that the
  machinery learns, not that any real market edge exists.
- `signals/sentiment/llm.py`: the Claude client wrapper was checked against the SDK's method signature but has **not** been
  run against the live API (no credentials in this environment). The scorer is tested with fake completers.
- Notification channels (Slack, Telegram, webhook, e-mail) are tested against injected fakes, never a real service.
- Chart-pattern detection is rule-based on confirmed swings; thresholds are parameters, not calibrated truths.
- Broker adapters are tested against fakes only (unchanged from before this work).
- Margin parameters are typical defaults, not an exchange's or broker's schedule.
- Multi-currency accounting sits beside the ledger; the engine does not convert P&L between currencies by itself.
- The order-book simulator's background flow is random; the engine and backtester still run on bars.
- After a restart, channels holding secrets, fitted model objects and alert rules written as code must be supplied
  again; the restore report lists them.
- The web UI's research pages run on synthetic prices (or uploaded CSVs) and a synthetic option chain; the
  news-sentiment page uses the word list and event rules, not the language-model scorer; prediction-model results are
  gross of costs.
