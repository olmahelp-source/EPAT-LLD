# Working instructions for this repository

`algoplatform` is an algorithmic trading platform: research, backtest, paper and live trading on one engine.
Source is in `src/algoplatform/<job group>/<package>` (for example `capital/risk`), tests in `tests`, the status of the work
in [`PROGRESS.md`](PROGRESS.md).

## Start of every session

1. Read `PROGRESS.md`. It says what is done, what is verified and what is pending. Continue from section 6.
2. Run `git status` and `git log --oneline -5` to confirm the working tree matches what `PROGRESS.md` describes.
3. If they disagree, trust the files on disk, find out why, and correct `PROGRESS.md` before doing anything else.

## Recording progress

- Update `PROGRESS.md` at the end of each unit of work (a package, a test file, a document), not only at the end of a session.
- Mark something "done" or "passing" only after running the command that proves it, and add that command and its result to the verification log.
- "Written but not tested" is a real state. Record it as such. Never round it up to "done".
- Keep the pending list ordered and current: remove what is finished, add what was discovered.
- Commit and push after each verified unit of work. The container is temporary; uncommitted work can be lost.

## How to verify

```
cd tests && python -W error::DeprecationWarning:algoplatform -m unittest discover -p "test_*.py"   # full suite
cd tests && python -m unittest test_technical                                                     # one file
PYTHONPATH=src python -m algoplatform structure --check                                           # dependency rules
PYTHONPATH=src python -m algoplatform functions                                                   # indicator / operator catalogue
```

## Rules of the code base

- **Dependencies:** numpy and pandas only. No scipy, no scikit-learn, no deep-learning framework. Optional extras are imported lazily.
- **Python 3.10 or later.** Do not use syntax or library features newer than 3.10. Line length is 140.
- **No look-ahead.** Any function of market data must be causal: the value at bar t uses data up to t only. New indicators, patterns and operators are covered by the prefix-invariance test in `tests/test_technical.py`; keep it passing.
- **Errors:** raise a subclass of `foundation.core.exceptions.PlatformError` with a clear message and `details`. A failure in one item of a batch (one symbol, one trial, one channel, one rule) must not stop the others. An error whose action is HALT (kill switch, bad configuration) is never swallowed.
- **Separation of concerns:** every package belongs to one job group in `architecture.py` and lives in that group's folder; the same file lists what each package may import. A new dependency between packages requires editing that allow-list on purpose. Libraries (`stats`, `quant`, `technical`, `ml`, `dl`, `rl`, `ai`, ...) stay pure computation.
- **Roles:** access goes through the desks in `application/app/workspaces.py`. Research cannot trade; trading cannot approve its own strategy or reset its own kill switch; approval always needs a second person.
- **Determinism:** anything random takes a `seed`. The same inputs must give the same outputs.
- **Honesty about limits:** if something is tested only against fakes or synthetic data, say so in the docstring and in the README limits.
- **Tests** use `unittest`, live in `tests/test_*.py`, and import `helpers` first. Test behaviour and properties, not implementation details.

## Git

- Work on the branch named in `PROGRESS.md`. Do not open a pull request unless asked.
