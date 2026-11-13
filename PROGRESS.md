# Progress record

This file is the single source of truth for what has been built, what has been verified, and what is still pending.
It is updated at the end of every unit of work. Rules for keeping it honest are in [`CLAUDE.md`](CLAUDE.md).

**Last updated:** 2026-11-13, after round three (web UI regrouped by job group)
**Working branch:** `ccr-b7d3820c-a4gug7`
**Default branch:** `claude/trading-platform-system-y9ke4s`

**Merge state.** Pull request #2 merged the first part of this work (all source and tests up to that point). Everything
after it is on the working branch, which already contains the default branch, and is proposed in pull request #3
(round three was pushed to the same branch, so it is part of that pull request too).

---

## 1. Where things stand

| Area | State |
|---|---|
| Every item of the original request | **Done.** Code and passing tests for all 27 items. |
| Items found missing in the gap analysis | **Done** except live-service verification (cannot be done here, see section 6). Margin, multi-currency accounting and an order-book simulator were built in round two. |
| Size of the work | 77 new source files (two of them added in round three) plus the modified files listed in section 5: about 13,400 lines of source and UI and 5,350 lines of tests, on top of the original platform. Round three alone: +1,090 lines of source and UI, +242 lines of tests (`git diff --numstat`). |
| Whole test-suite | **773 tests pass** (407 original + 366 new), with deprecation warnings treated as errors. |
| Dependency rules ("do not mix concerns") | **Verified.** `python -m algoplatform structure --check` reports no violation; the same check is a unit test. |
| Linter (`ruff`) | **Clean.** |
| Type checker (`mypy`) | **No findings in the files added by this work.** 112 findings remain in files that predate it (the CI job stays informational). |
| Web UI | **Done, grouped by job.** The menu follows the 12 job groups of `architecture.py` (titles come from the server); 24 pages, 9 of them new in round three; every group except Foundation has a page. Checked in a headless browser as all eight roles, on desktop and phone width. |
| Restart safety of the shared services | **Done.** |
| Example scripts | All three run. |
| CI on GitHub | Passed (lint and tests on Python 3.10 to 3.13) for every pushed commit since the lint fixes; see section 7. |

---

## 2. What the original request missed (gap analysis)

The request listed 27 areas. A production trading platform also needs the following. Each was checked against the
repository.

| Missing from the request | Status in the repository |
|---|---|
| Market-data quality (validation, cleaning, gaps, split adjustment) | Already present: `data` |
| Corporate actions and continuous futures | **Added:** `instruments/corporate_actions.py`, `instruments/continuous.py` |
| Execution algorithms and transaction-cost analysis | Already present: `execution`, `analytics/tca.py` |
| Durable state, crash recovery, broker reconciliation | Already present: `persistence` |
| Kill switch and live risk monitor | Already present: `risk` |
| Audit trail | Already present: `core/logging.py` (`AuditLog`) |
| Segregation of duties between roles | **Added:** 4 new roles, 9 new permissions, role desks (`app/workspaces.py`) |
| Strategy lifecycle with approval (research, paper, live) | **Added:** `governance/lifecycle.py` (four-eyes promotion) |
| Model registry (versions, training-data fingerprint, rollback) | **Added:** `governance/models.py` |
| Health monitoring and heartbeats | **Added:** `ops/health.py` |
| Job scheduler | **Added:** `ops/scheduler.py` |
| Market-regime detection | **Added:** `ai/hmm.py`, `volatility/management.py` |
| Overfitting control (walk-forward, deflated Sharpe, purged CV) | Already present: `optimize`, `stats`, `research`; scorecard **added** in `analytics/evaluation.py` |
| Enforced separation of concerns | **Added:** `architecture.py` (allow-list of dependencies, checked by a test) |
| Margin model (scenario / portfolio margin) | **Added (round two):** `risk/margin.py`; default parameters, not an exchange schedule |
| Multi-currency accounting and FX conversion | **Added (round two):** `portfolio/currency.py` |
| Tick / order-book data and latency simulation | **Added (round two):** `brokers/orderbook.py`; a laboratory beside the bar-based engine |
| Restart safety of approvals, alerts, routes, instruments, models | **Added (round two):** `TradingPlatform(store=...)` |
| Live broker certification | **Not possible here.** Adapters are tested against fakes only. |

---

## 3. Requirement coverage

| # | Requested item | Where it lives | Code | Tests |
|---|---|---|---|---|
| 1 | Library with error handling | `core/exceptions.py` (typed tree, `retryable` flag); 9 exception types added | done | passing (`test_core_utils`, `test_resilience`) |
| 2 | Advanced error handling | `utils/resilience.py`: classification and policy, `Result`, fallback chain, timeout, deadline, bulkhead, error budget, error boundary, dead-letter queue, collector, context | done | passing (`test_resilience`) |
| 3 | Multi-broker and multi-user | `users`, `brokers` (existing); 8 roles and 20 permissions in `core/enums.py`, `users/security.py` | done | passing (existing, `test_architecture_desks`) |
| 4 | Order management | `orders`, `execution` (existing) | done | passing (existing) |
| 5 | Risk management | `risk` (existing) | done | passing (existing) |
| 6, 17 | Position sizing and fund allocation | `portfolio` (existing) | done | passing (existing) |
| 7 | Instrument data helpers | `instruments`: contract specs, instrument master with broker symbol mapping, corporate actions, continuous futures | done | passing (`test_instruments_options`) |
| 8 | Option data handling | `derivatives/chain.py`: option chain, implied volatility, Greeks, put/call ratio, max pain, skew, term structure, quality report | done | passing (`test_instruments_options`) |
| 9 | Technical indicators | `technical`: 55 indicators (core set re-exported from `quant/indicators.py`) | done | passing (`test_technical`) |
| 10 | Fundamental indicators | `fundamentals` (existing) | done | passing (existing) |
| 11 | Pattern identifiers | `technical/candles.py` (19 candlestick patterns), `technical/chart_patterns.py` (double top/bottom, head and shoulders, triangles, wedges, channels, range breakout, divergence), `technical/levels.py` (swings, support/resistance, pivots) | done | passing (`test_technical`) |
| 12 | Functions for operations and indicators | `technical/operators.py` (43 operators), `technical/catalog.py` (run any function by name), `technical/expression.py` (safe formula engine) | done | passing (`test_technical`) |
| 13 | Strategy creator | `strategy/builder.py` (existing), `strategy/expression.py` (formula strategies), `strategy/policy.py` (model adapter) | done | passing (`test_technical`, existing) |
| 14 | Backtesting and optimisation | `backtest`, `optimize` (existing) | done | passing (existing) |
| 15 | Instrument selection | `selection`: filters, 9 rankers, external scores, diversification, group caps, one-per-cluster, pair selection | done | passing (`test_search_selection_evaluation`) |
| 16 | Relationship management | `relationships`: correlation, beta, tail dependence, lead-lag, Granger causality, cointegration scan, clustering, spanning tree, PCA, `RelationshipManager` | done | passing (`test_relationships_volatility`) |
| 16b | Sentiment and news analysis | `sentiment` (existing); `sentiment/llm.py` (language-model scorer with lexicon fallback) | done | passing (existing, `test_notify_ops_governance`); see limits |
| 17b | Statistical methods | `stats` (existing) | done | passing (existing) |
| 18 | Quantitative methods | `quant` (existing) | done | passing (existing) |
| 19 | Volatility management and prediction | `volatility`: 5 estimators, volatility cone, EWMA / GARCH / GJR / HAR forecasters, forecast evaluation, volatility targeting, regime classifier | done | passing (`test_relationships_volatility`) |
| 20 | Machine learning | `ml`: decision tree, random forest, gradient boosting, k-nearest neighbours, naive Bayes, lasso, PCA, k-means, voting and stacking, walk-forward prediction; `research` (existing) | done | passing (`test_learning`) |
| 21 | Deep learning | `dl`: Dense, LSTM, Dropout, Adam, training loop, MLP, LSTM forecaster, autoencoder (numpy; gradients checked numerically) | done | passing (`test_learning`) |
| 22 | Reinforcement learning | `rl`: trading environment, Q-learning, deep Q-network, policy gradient, three bandits | done | passing (`test_learning`) |
| 23 | Advanced parameter optimisation | `optimize/advanced.py`: Bayesian, TPE, particle swarm, differential evolution, simulated annealing, CMA-ES, successive halving, Pareto front | done | passing (`test_search_selection_evaluation`) |
| 24 | Other AI methods | `ai`: hidden Markov regimes, fuzzy logic, genetic programming, isolation forest, change-point detection | done | passing (`test_learning`) |
| 25 | Notification and alert | `notify`: 7 channels, notifier (retry, dedupe, private routing, dead letters, escalation), alert rules | done | passing (`test_notify_ops_governance`); see limits |
| 26 | Analytics | `analytics` (existing) | done | passing (existing) |
| 27 | Strategy result analysis | `analytics/evaluation.py`: bootstrap confidence intervals, drawdown probability, stability, regime breakdown, trade concentration, comparison, pass/fail scorecard | done | passing (`test_search_selection_evaluation`) |
| - | One platform, different roles | `app/workspaces.py`: research, trading, portfolio, risk, operations, compliance, admin desks | done | passing (`test_architecture_desks`) |
| - | Structure grouped by job | `architecture.py`: 12 job groups, dependency allow-list, `check_architecture()` | done | passing (`test_architecture_desks`) |
| - | Operations and governance (from the gap analysis) | `ops`, `governance` | done | passing (`test_notify_ops_governance`) |
| - | Margin (from the gap analysis) | `risk/margin.py`: rule-based and scenario margin, account report, liquidation price, pre-trade `MarginCheck` | done | passing (`test_margin_currency_orderbook`) |
| - | Multi-currency accounting (from the gap analysis) | `portfolio/currency.py`: point-in-time FX rates with cross rates, cash per currency, exposure, FX P&L attribution, hedge amounts | done | passing (`test_margin_currency_orderbook`) |
| - | Order book and tick simulation (from the gap analysis) | `brokers/orderbook.py`: matching engine, queue position, cost to trade, seeded simulator with latency, tick-to-bar | done | passing (`test_margin_currency_orderbook`) |
| - | Restart safety of the shared services | `save_state` / `load_state` / autosave in `app/platform.py`; `export_state` / `import_state` on each service | done | passing (`test_state_persistence`, `test_web_quant`) |
| - | Web UI for the new features | `web/quant_api.py`, six pages in `web/static/app.js` | done | passing (`test_web_quant`); browser walkthrough |
| - | Web UI grouped by job (round three) | `web/static/app.js` (menu by job group, hash routing, Home), `web/research_api.py` (option chain, sentiment, prediction models, regimes), `web/desk_api.py` (instruments, margin, currency, order book, health), 8 optimiser methods, audit log on the compliance permission | done | passing (`test_web_groups`); browser walkthrough as 8 roles |

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

Existing packages were kept at their original import paths (so the 407 original tests keep guarding them); the
grouping is expressed in `architecture.py` and enforced by its dependency check rather than by moving directories.
Moving the directories into group folders is possible later; it would change every import path and is a decision for
the owner of the repository.

---

## 5. File inventory

### New packages (all files new)

| Package | Files |
|---|---|
| `technical/` | `trend.py`, `momentum.py`, `bands.py`, `volume.py`, `levels.py`, `candles.py`, `chart_patterns.py`, `operators.py`, `catalog.py`, `expression.py` |
| `instruments/` | `contracts.py`, `master.py`, `corporate_actions.py`, `continuous.py` |
| `relationships/` | `correlation.py`, `leadlag.py`, `cointegration.py`, `structure.py`, `manager.py` |
| `volatility/` | `estimators.py`, `models.py`, `evaluation.py`, `management.py` |
| `ml/` | `tree.py`, `classic.py`, `unsupervised.py`, `ensemble.py`, `metrics.py`, `pipeline.py` |
| `dl/` | `nn.py`, `models.py` |
| `rl/` | `env.py`, `agents.py`, `bandits.py` |
| `ai/` | `hmm.py`, `fuzzy.py`, `genetic_programming.py`, `anomaly.py`, `changepoint.py` |
| `selection/` | `rankers.py`, `selector.py` |
| `notify/` | `message.py`, `channels.py`, `notifier.py`, `alerts.py` |
| `ops/` | `health.py`, `scheduler.py` |
| `governance/` | `lifecycle.py`, `models.py` |

Each package also has an `__init__.py` that exports its public names.

### New files in existing packages

`web/research_api.py`, `web/desk_api.py` (round three); `risk/margin.py`, `portfolio/currency.py`, `brokers/orderbook.py`, `web/quant_api.py` (round two);
`utils/resilience.py`, `utils/numeric.py`, `derivatives/chain.py`, `optimize/advanced.py`, `analytics/evaluation.py`,
`sentiment/llm.py`, `strategy/expression.py`, `strategy/policy.py`, `app/workspaces.py`, and the top-level
`architecture.py`.

### Modified source files

`core/exceptions.py` (9 exception types), `core/enums.py` (4 roles, 9 permissions, `Severity`), `core/__init__.py`,
`users/security.py` (role-to-permission table), `utils/__init__.py`, `strategy/__init__.py`, `derivatives/__init__.py`,
`sentiment/__init__.py`, `analytics/__init__.py`, `optimize/__init__.py`, `optimize/optimizers.py` (dispatcher knows the
new methods), `stats/tests.py` (half-life fix, see section 8), `app/platform.py` (new shared services, `desk()` /
`desks()`), `app/__init__.py`, `app/cli.py` (`structure` and `functions` commands, new optimiser names),
`web/static/app.js` and `web/static/styles.css` (rebuilt around the job groups in round three), `web/server.py` (registers the new route files, optimiser methods, audit permission), `architecture.py` (web allow-list), `app/workspaces.py` (`load_instruments(overwrite=...)`), `algoplatform/__init__.py` (version 0.2.0).

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
| `tests/test_notify_ops_governance.py` (incl. `sentiment/llm.py`) | 46 | passing |
| `tests/test_architecture_desks.py` | 26 | passing |
| `tests/test_margin_currency_orderbook.py` | 39 | passing |
| `tests/test_state_persistence.py` | 19 | passing |
| `tests/test_web_quant.py` | 15 | passing |
| `tests/test_web_groups.py` (menu follows the structure and the server; new routes) | 16 | passing |
| **Total** | **773** | **passing** |

---

## 6. Pending tasks

Round two completed every pending item that can be done from this environment. What is left:

| # | Task | State | Why it is not done |
|---|---|---|---|
| 1 | Merge pull request #3 into the default branch | **waiting for the owner** | Opened: https://github.com/olmahelp-source/EPAT-LLD/pull/3. Merging into the default branch is the repository owner's decision, as with pull request #2. |
| 2 | Run the integrations against real services: Claude sentiment client, Slack, Telegram, webhook, e-mail, broker sandboxes | **blocked** | Needs the owner's credentials and accounts; nothing here should send messages or orders to real services. All are tested against fakes. |
| 3 | Type findings in the 113 places in files that predate this work | open, optional | Older code, outside the request; the CI type-check job is informational. |
| 4 | Move packages into one folder per job group | not started, optional | Changes every import path; a decision for the owner. The grouping is already enforced by `architecture.py`. |

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
- `sentiment/llm.py`: the Claude client wrapper was checked against the SDK's method signature but has **not** been
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
