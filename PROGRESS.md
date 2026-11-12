# Progress record

This file is the single source of truth for what has been built, what has been verified, and what is still pending.
It is updated at the end of every unit of work. Rules for keeping it honest are in [`CLAUDE.md`](CLAUDE.md).

**Last updated:** 2026-11-12, after commit `0d40e28` and the final verification run
**Branch:** `ccr-b7d3820c-a4gug7`

---

## 1. Where things stand

| Area | State |
|---|---|
| Source code for every requested item | **Done.** 71 new source files (about 10,300 lines) and 16 modified source files. |
| Tests for the new code | **Done and passing.** 8 new test files, 277 tests (about 3,900 lines). |
| Whole test-suite | **684 tests pass** (407 original + 277 new), with deprecation warnings treated as errors. |
| Dependency rules ("do not mix concerns") | **Verified.** `python -m algoplatform structure --check` reports no violation; the same check is a unit test. |
| Linter (`ruff`, the project's configuration) | **Clean.** |
| Example scripts | **All three run** (`run_demo`, `run_advanced`, `run_quant_lab`). |
| Documentation | **Done.** `README.md` updated; `docs/ARCHITECTURE.md` added. |
| Committed and pushed | **Yes.** |
| Continuous integration on Python 3.10 to 3.12 | **Not confirmed from here.** This machine has Python 3.13 only. A static scan of all 211 Python files found no syntax or library feature newer than 3.10; the CI run on the pushed branch is the real check. |

Nothing in this file is marked done without a command in section 7 that proves it.

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
| Margin model (SPAN / portfolio margin) | **Not built.** Only `ContractSpec.margin_pct` exists. |
| Multi-currency accounting and FX conversion | **Not built.** |
| Tick / order-book data and latency simulation | **Not built** (bar-based platform; stated in the README limits). |
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

`utils/resilience.py`, `utils/numeric.py`, `derivatives/chain.py`, `optimize/advanced.py`, `analytics/evaluation.py`,
`sentiment/llm.py`, `strategy/expression.py`, `strategy/policy.py`, `app/workspaces.py`, and the top-level
`architecture.py`.

### Modified source files

`core/exceptions.py` (9 exception types), `core/enums.py` (4 roles, 9 permissions, `Severity`), `core/__init__.py`,
`users/security.py` (role-to-permission table), `utils/__init__.py`, `strategy/__init__.py`, `derivatives/__init__.py`,
`sentiment/__init__.py`, `analytics/__init__.py`, `optimize/__init__.py`, `optimize/optimizers.py` (dispatcher knows the
new methods), `stats/tests.py` (half-life fix, see section 8), `app/platform.py` (new shared services, `desk()` /
`desks()`), `app/__init__.py`, `app/cli.py` (`structure` and `functions` commands, new optimiser names),
`web/static/app.js` (role list), `algoplatform/__init__.py` (version 0.2.0).

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
| **Total** | **684** | **passing** |

---

## 6. Pending tasks

Nothing from the original request is pending. What remains is listed in order of value; none of it is started.

1. **Confirm the CI run** on the pushed branch (Python 3.10, 3.11, 3.12, 3.13 and the lint job). Could not be run here.
2. **Web UI for the new features.** The browser UI still shows the original tabs. The new packages are reachable from
   Python and the CLI only. Only the role list in the UI was updated.
3. **Persist the new shared services.** Notification routes, alert rules, strategy lifecycle records, the model
   registry and the instrument master live in memory; they are not yet written to the SQLite store, so they do not
   survive a restart (orders, fills, positions and the kill switch do).
4. **Run the integrations against real services** before relying on them: the Claude sentiment client, Slack,
   Telegram, webhook and e-mail channels. All are tested against fakes only.
5. **Not built, by decision:** margin model, multi-currency accounting, tick / order-book simulation.
6. Optional: move packages into one directory per job group (see the note in section 4).

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
- No margin model, no multi-currency accounting, no tick or order-book simulation.
