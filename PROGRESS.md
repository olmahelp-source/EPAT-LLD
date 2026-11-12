# Progress record

This file is the single source of truth for what has been built, what has been verified, and what is still pending.
It is updated at the end of every unit of work. Rules for keeping it honest are in [`CLAUDE.md`](CLAUDE.md).

**Last updated:** 2026-11-12, after commit `37c0422` plus the technical/resilience test fixes
**Branch:** `ccr-b7d3820c-a4gug7`

---

## 1. Where things stand

| Area | State |
|---|---|
| Source code for every requested item | **Written.** 71 new source files (about 10,100 lines) and 15 modified files, on top of the original platform. |
| All packages import, dependency rules hold | **Verified.** 34 packages import; `python -m algoplatform structure --check` reports no violation. |
| Original test-suite (407 tests) | **Verified passing** after the changes. |
| Tests for the new code | **In progress.** 2 of 8 planned test files are written and passing (74 tests). 6 are still to be written. |
| Documentation (`README.md`, `docs/ARCHITECTURE.md`) | **Pending.** The README still describes the platform as it was before this work. |
| Example script for the new features | **Pending.** |
| Committed and pushed | **Yes**, as work in progress. |

Nothing is finished until its row in section 3 says "tests: passing". Code that has only been smoke-tested by hand is
marked as such.

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

"Smoke-tested" means the module was run by hand during development and behaved correctly, but has no unit tests yet.

| # | Requested item | Where it lives | Code | Tests |
|---|---|---|---|---|
| 1 | Library with error handling | `core/exceptions.py` (typed tree, `retryable` flag); 9 exception types added | done | passing (`test_core_utils`, `test_resilience`) |
| 2 | Advanced error handling | `utils/resilience.py`: classification and policy, `Result`, fallback chain, timeout, deadline, bulkhead, error budget, error boundary, dead-letter queue, collector, context | done | passing (`test_resilience`, 24 tests) |
| 3 | Multi-broker and multi-user | `users`, `brokers` (existing); new roles and permissions in `core/enums.py`, `users/security.py` | done | existing tests passing; new roles **pending** |
| 4 | Order management | `orders`, `execution` (existing) | done | passing (existing) |
| 5 | Risk management | `risk` (existing) | done | passing (existing) |
| 6, 17 | Position sizing and fund allocation | `portfolio` (existing) | done | passing (existing) |
| 7 | Instrument data helpers | `instruments`: contract specs, instrument master with broker symbol mapping, corporate actions, continuous futures | done | **pending** (smoke-tested) |
| 8 | Option data handling | `derivatives/chain.py`: option chain, implied volatility, Greeks, put/call ratio, max pain, skew, term structure, quality report | done | **pending** (smoke-tested) |
| 9 | Technical indicators | `technical`: 55 indicators (core set re-exported from `quant/indicators.py`) | done | passing (`test_technical`) |
| 10 | Fundamental indicators | `fundamentals` (existing) | done | passing (existing) |
| 11 | Pattern identifiers | `technical/candles.py` (19 candlestick patterns), `technical/chart_patterns.py` (double top/bottom, head and shoulders, triangles, wedges, channels, range breakout, divergence), `technical/levels.py` (swings, support/resistance, pivots) | done | passing (`test_technical`) |
| 12 | Functions for operations and indicators | `technical/operators.py` (43 operators), `technical/catalog.py` (run any function by name), `technical/expression.py` (safe formula engine) | done | passing (`test_technical`) |
| 13 | Strategy creator | `strategy/builder.py` (existing), `strategy/expression.py` (formula strategies), `strategy/policy.py` (model adapter) | done | passing (`test_technical`, existing) |
| 14 | Backtesting and optimisation | `backtest`, `optimize` (existing) | done | passing (existing) |
| 15 | Instrument selection | `selection`: filters, 9 rankers, diversification, sector caps, one-per-cluster, pair selection | done | **pending** (smoke-tested) |
| 16 | Relationship management | `relationships`: correlation, beta, tail dependence, lead-lag, Granger causality, cointegration scan, clustering, spanning tree, PCA, `RelationshipManager` | done | **pending** (smoke-tested) |
| 16b | Sentiment and news analysis | `sentiment` (existing); `sentiment/llm.py` (language-model scorer with lexicon fallback) | done | existing passing; `llm.py` **pending** (smoke-tested with fakes) |
| 17b | Statistical methods | `stats` (existing) | done | passing (existing) |
| 18 | Quantitative methods | `quant` (existing) | done | passing (existing) |
| 19 | Volatility management and prediction | `volatility`: 5 estimators, volatility cone, EWMA / GARCH / GJR / HAR forecasters, forecast evaluation, volatility targeting, regime classifier | done | **pending** (smoke-tested) |
| 20 | Machine learning | `ml`: decision tree, random forest, gradient boosting, k-nearest neighbours, naive Bayes, lasso, PCA, k-means, voting and stacking, walk-forward prediction; `research` (existing) | done | **pending** (smoke-tested) |
| 21 | Deep learning | `dl`: Dense, LSTM, Dropout, Adam, training loop, MLP, LSTM forecaster, autoencoder (numpy; gradients checked numerically) | done | **pending** (smoke-tested) |
| 22 | Reinforcement learning | `rl`: trading environment, Q-learning, deep Q-network, policy gradient, three bandits | done | **pending** (smoke-tested) |
| 23 | Advanced parameter optimisation | `optimize/advanced.py`: Bayesian, TPE, particle swarm, differential evolution, simulated annealing, CMA-ES, successive halving, Pareto front | done | **pending** (smoke-tested) |
| 24 | Other AI methods | `ai`: hidden Markov regimes, fuzzy logic, genetic programming, isolation forest, change-point detection | done | **pending** (smoke-tested) |
| 25 | Notification and alert | `notify`: 7 channels, notifier (retry, dedupe, private routing, dead letters, escalation), alert rules | done | **pending** (smoke-tested) |
| 26 | Analytics | `analytics` (existing) | done | passing (existing) |
| 27 | Strategy result analysis | `analytics/evaluation.py`: bootstrap confidence intervals, drawdown probability, stability, regime breakdown, trade concentration, comparison, pass/fail scorecard | done | **pending** (smoke-tested) |
| - | One platform, different roles | `app/workspaces.py`: research, trading, portfolio, risk, operations, compliance, admin desks | done | **pending** (smoke-tested) |
| - | Structure grouped by job | `architecture.py`: 12 job groups, dependency allow-list, `check_architecture()` | done | **pending** (check passes by hand) |

---

## 4. Structure (grouped by job)

Print it at any time with `PYTHONPATH=src python -m algoplatform structure`.

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

### New files in existing packages

`utils/resilience.py`, `utils/numeric.py`, `derivatives/chain.py`, `optimize/advanced.py`, `analytics/evaluation.py`,
`sentiment/llm.py`, `strategy/expression.py`, `strategy/policy.py`, `app/workspaces.py`, and the top-level
`architecture.py`.

### Modified files

`core/exceptions.py` (9 exception types), `core/enums.py` (4 roles, 9 permissions, `Severity`), `core/__init__.py`,
`users/security.py` (role-to-permission table), `utils/__init__.py`, `strategy/__init__.py`, `derivatives/__init__.py`,
`sentiment/__init__.py`, `analytics/__init__.py`, `optimize/__init__.py`, `optimize/optimizers.py` (dispatcher knows the
new methods), `app/platform.py` (new shared services, `desk()` / `desks()`), `app/__init__.py`, `app/cli.py`
(`structure` and `functions` commands, new optimiser names), `web/static/app.js` (role list).

### Test files

| File | State |
|---|---|
| 20 original `tests/test_*.py` files | passing (407 tests) |
| `tests/test_resilience.py` | **new, passing** (24 tests) |
| `tests/test_technical.py` | **new, passing** (50 tests) |
| `tests/test_instruments_options.py` | pending |
| `tests/test_relationships_volatility.py` | pending |
| `tests/test_learning.py` (ml, dl, rl, ai) | pending |
| `tests/test_search_selection_evaluation.py` (advanced optimisers, selection, scorecard) | pending |
| `tests/test_notify_ops_governance.py` (incl. `sentiment/llm.py`) | pending |
| `tests/test_architecture_desks.py` (dependency rules, roles, desks, CLI) | pending |

---

## 6. Pending tasks, in order

1. Write the six pending test files listed above; fix whatever they uncover.
2. Run the complete suite with deprecation warnings as errors; confirm the count.
3. Update `README.md` (module map, quick starts, honest limits) and add `docs/ARCHITECTURE.md`.
4. Add `examples/run_quant_lab.py` showing the new features end to end.
5. Update `pyproject.toml` (optional `llm` extra, version) and check the CI workflow still fits.
6. Run the linter configuration from `pyproject.toml` if `ruff` is available.
7. Final commit and push; update this file.

---

## 7. Verification log

| When | Command | Result |
|---|---|---|
| 2026-11-12 | `cd tests && python -m unittest discover -p "test_*.py"` (before any change) | 407 tests, OK |
| 2026-11-12 | same, after all source changes, with `-W error::DeprecationWarning:algoplatform` | 407 tests, OK |
| 2026-11-12 | `PYTHONPATH=src python -m algoplatform structure --check` | 12 groups, 34 packages, "dependency rules: OK" |
| 2026-11-12 | import of every package | 34 of 34 |
| 2026-11-12 | `python -m unittest test_resilience test_technical` | 74 tests, OK |

Findings while testing so far: 8 initial failures in `test_technical.py`, all wrong expectations in the tests
themselves (for example, comparing against the engine's trade count, which includes rebalancing orders). No defect
found in the source by those tests. Defects found and fixed earlier, during development: Bayesian search over-explored
on heavy-tailed scores (fixed with rank-based normal scores); TPE stalled at the centre of a cluster (fixed with a
smaller "good" set); the language-model scorer's circuit breaker did not count malformed answers (fixed).

---

## 8. Known limits (to be carried into the README)

- Learning models are numpy implementations sized for daily data. They are not a substitute for a GPU framework.
- `sentiment/llm.py`: the Claude client wrapper was checked against the SDK's method signature but has **not** been
  run against the live API (no credentials in this environment). The scorer is tested with fake completers.
- Notification channels (Slack, Telegram, webhook, e-mail) are tested against injected fakes, never a real service.
- Chart-pattern detection is rule-based on confirmed swings; thresholds are parameters, not calibrated truths.
- No margin model, no multi-currency accounting, no tick or order-book simulation.
