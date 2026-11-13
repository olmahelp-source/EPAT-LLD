"""The platform's structure as data: which package does which job, and which package may depend on which.

Two things live here so that they cannot drift apart from the code:

* :data:`GROUPS`  the packages grouped by the job they do, with the roles that mainly work in each group. Each group is a
  folder (``src/algoplatform/<group key>/``) holding exactly its packages, so the import path says the job:
  ``algoplatform.capital.risk``, ``algoplatform.signals.technical``. This is the map printed by
  ``python -m algoplatform structure`` and shown in ``docs/ARCHITECTURE.md``.
* :data:`ALLOWED` for every package, the complete list of other packages it is allowed to import. The test-suite
  parses every source file and fails if an import is not on the list (:func:`check_architecture`).

"Do not mix concerns" is therefore not a convention but a failing test: a strategy that imports a broker, a risk check
that imports a strategy, or an indicator that reaches for the order manager cannot be merged. Adding a dependency means
editing this file, which is exactly the moment someone should ask whether it belongs.

The invariants the allow-list encodes:

1. ``core`` imports nothing; ``utils`` imports only ``core``.
2. Libraries (statistics, quant, technical, relationships, volatility, ml, dl, rl, ai, research) are pure computation:
   they never import users, brokers, orders, execution, risk, persistence, notifications or the application.
3. ``strategy`` decides *what to hold* and never sees an order, a broker, a risk check or a user.
4. ``risk`` judges orders and never knows which strategy produced them. ``brokers`` only translate and emit events.
5. Only ``app`` and ``web`` see everything; nothing imports ``web``.
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from .foundation.core.enums import Role


@dataclass(frozen=True)
class JobGroup:
    key: str
    title: str
    job: str                          # the one question this group answers
    packages: Tuple[str, ...]
    roles: Tuple[Role, ...] = ()      # who mainly works here (ADMIN can go everywhere)


GROUPS: Tuple[JobGroup, ...] = (
    JobGroup("foundation", "Foundation", "Shared contracts, error handling and resilience that everything else relies on.",
             ("core", "utils")),
    JobGroup("access", "Access control", "Who is this user, what may they do, and which broker accounts are theirs?",
             ("users",), (Role.ADMIN,)),
    JobGroup("market_data", "Market and reference data", "What can be traded, on what terms, and what has it cost?",
             ("data", "instruments"), (Role.OPERATIONS, Role.QUANT)),
    JobGroup("signals", "Signal libraries", "Turn raw data into numbers a rule can use: technical, fundamental, sentiment.",
             ("technical", "fundamentals", "sentiment"), (Role.QUANT,)),
    JobGroup("quantitative", "Quantitative research", "The mathematics: statistics, pricing and options, relationships, volatility.",
             ("stats", "quant", "derivatives", "relationships", "volatility"), (Role.QUANT,)),
    JobGroup("learning", "Learning and AI", "Models that learn from data: leakage-safe research, ML, deep learning, RL, other AI.",
             ("research", "ml", "dl", "rl", "ai"), (Role.QUANT,)),
    JobGroup("strategies", "Strategy", "Decide what to hold: the strategy SDK, strategy creators and instrument selection.",
             ("strategy", "selection"), (Role.QUANT, Role.TRADER)),
    JobGroup("capital", "Portfolio and risk", "How much to hold, how capital is shared out, and when to stop.",
             ("portfolio", "risk"), (Role.PORTFOLIO_MANAGER, Role.RISK_MANAGER)),
    JobGroup("trading", "Trading", "Get orders to the market and fills back: brokers, order management, execution algorithms.",
             ("brokers", "orders", "execution"), (Role.TRADER,)),
    JobGroup("validation", "Validation", "Evidence that an idea works: backtests, parameter optimisation, result analysis.",
             ("backtest", "optimize", "analytics"), (Role.QUANT, Role.PORTFOLIO_MANAGER)),
    JobGroup("operations", "Operations and control", "Keep it running and accountable: durability, alerts, health, schedules, approvals.",
             ("persistence", "notify", "ops", "governance"), (Role.OPERATIONS, Role.RISK_MANAGER, Role.AUDITOR)),
    JobGroup("application", "Application", "Wire the parts together behind permission-checked, role-specific desks; CLI and web UI.",
             ("app", "web")),
)

_LIBS = ("core", "utils")
ALLOWED: Dict[str, FrozenSet[str]] = {k: frozenset(v) for k, v in {
    # foundation
    "core": (),
    "utils": ("core",),
    # access
    "users": _LIBS,
    # market and reference data
    "data": ("core",),
    "instruments": _LIBS,
    # quantitative research (pure computation)
    "stats": _LIBS,
    "quant": ("core", "stats"),
    "derivatives": ("core", "quant"),
    "relationships": ("core", "stats"),
    "volatility": ("core", "utils", "stats", "quant"),
    # signal libraries
    "technical": ("core", "quant"),
    "fundamentals": ("core", "utils", "strategy"),          # ships a Strategy adapter (FactorStrategy)
    "sentiment": ("core", "utils", "strategy"),             # ships a Strategy adapter (SentimentStrategy)
    # learning and AI (pure computation)
    "research": ("core", "stats", "quant"),
    "ml": ("core", "utils", "research"),
    "dl": _LIBS,
    "rl": ("core", "utils", "dl"),
    "ai": ("core", "utils", "quant", "technical"),
    # strategy
    "strategy": ("core", "stats", "quant", "research", "technical"),
    "selection": ("core", "relationships"),
    # portfolio and risk
    "portfolio": _LIBS,
    "risk": ("core", "utils", "stats", "quant", "portfolio"),      # quant: option repricing for scenario margin
    # trading
    "brokers": ("core", "utils", "users"),
    "orders": ("core", "utils", "brokers", "portfolio", "risk"),
    "execution": ("core", "utils", "quant", "brokers", "orders"),
    # validation
    "analytics": ("core", "utils", "stats", "portfolio"),
    "backtest": ("core", "utils", "data", "portfolio", "brokers", "strategy", "analytics", "app"),   # reuses app.engine: one engine for test and live
    "optimize": ("core", "utils", "stats", "data", "strategy", "backtest", "analytics"),
    # operations and control
    "persistence": ("core", "brokers"),
    "notify": _LIBS,
    "ops": _LIBS,
    "governance": ("core", "utils", "users", "analytics"),
    # application
    "app": ("core", "utils", "users", "data", "instruments", "quant", "technical", "fundamentals", "sentiment", "relationships", "strategy", "selection",
            "portfolio", "risk", "brokers", "orders", "execution", "backtest", "optimize", "analytics", "persistence", "notify", "ops",
            "governance"),
    "web": ("core", "utils", "users", "data", "technical", "fundamentals", "sentiment", "relationships", "volatility", "derivatives",
            "research", "ml", "dl", "ai", "strategy", "selection", "portfolio", "risk", "brokers", "backtest", "optimize", "analytics",
            "persistence", "notify", "app"),     # one screen per job group: the web layer reaches each group's library to show it
}.items()}

# Packages that must stay pure computation: importing any of FORBIDDEN_FOR_LIBRARIES from them is always a violation,
# even if someone adds it to ALLOWED above.
LIBRARIES: FrozenSet[str] = frozenset({"stats", "quant", "derivatives", "relationships", "volatility", "technical", "research", "ml", "dl",
                                       "rl", "ai"})
FORBIDDEN_FOR_LIBRARIES: FrozenSet[str] = frozenset({"users", "brokers", "orders", "execution", "risk", "portfolio", "persistence", "notify",
                                                     "ops", "governance", "backtest", "optimize", "app", "web", "strategy"})
NEVER: Dict[str, FrozenSet[str]] = {
    "strategy": frozenset({"users", "brokers", "orders", "execution", "risk", "persistence", "notify", "app", "web"}),
    "risk": frozenset({"strategy", "brokers", "orders", "execution", "app", "web"}),
    "brokers": frozenset({"strategy", "risk", "portfolio", "orders", "execution", "app", "web"}),
}


def group_of(package: str) -> JobGroup:
    for g in GROUPS:
        if package in g.packages:
            return g
    raise KeyError(f"package '{package}' is not assigned to a job group")


def packages() -> List[str]:
    return [p for g in GROUPS for p in g.packages]


def _imports(path: str, parts: List[str], group_dirs: FrozenSet[str]) -> List[Tuple[str, int]]:
    """Packages imported by one source file, with line numbers. ``parts`` is the file's module path
    (``["algoplatform", group, package, ...]``); an imported module is named by the folder after its group folder."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), path)

    def package_of(full: List[str], names: List[str]) -> List[str]:
        if len(full) > 2 and full[1] in group_dirs:
            return [full[2]]
        if len(full) == 2 and full[1] in group_dirs:          # "from ...signals import fundamentals"
            return names
        return []

    out: List[Tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = [a.name for a in node.names]
            if node.level:
                base = parts[:-1]
                base = base[:len(base) - (node.level - 1)]
                full = base + (node.module.split(".") if node.module else [])
                out += [(p, node.lineno) for p in package_of(full, names)]
            elif node.module and node.module.split(".")[0] == "algoplatform":
                out += [(p, node.lineno) for p in package_of(node.module.split("."), names)]
        elif isinstance(node, ast.Import):
            out += [(p, node.lineno) for a in node.names if a.name.startswith("algoplatform.") for p in package_of(a.name.split("."), [])]
    return out


def package_dir(package: str) -> str:
    """Where a package lives, relative to ``src/algoplatform``: ``<job group>/<package>``."""
    return f"{group_of(package).key}/{package}"


def check_architecture(root: Optional[str] = None) -> List[str]:
    """Scan the source tree and return every violation as a readable line (an empty list means the structure holds).

    Checks: every package sits in the folder of its job group and has an allow-list; nothing but group folders sits at
    the top; every import between packages is on the importer's allow-list; libraries stay pure; the hard "never" rules
    hold."""
    root = root or os.path.dirname(os.path.abspath(__file__))
    known = set(packages())
    group_dirs = frozenset(g.key for g in GROUPS)
    problems: List[str] = []
    is_pkg = lambda *p: os.path.isdir(os.path.join(root, *p)) and os.path.exists(os.path.join(root, *p, "__init__.py"))   # noqa: E731
    on_disk: Dict[str, str] = {}
    for top in sorted(os.listdir(root)):
        if not is_pkg(top):
            continue
        if top not in group_dirs:
            problems.append(f"package '{top}' sits at the top level; every package belongs in its job group's folder"
                            + (f" ('{package_dir(top)}')" if top in known else ""))
            continue
        for sub in sorted(os.listdir(os.path.join(root, top))):
            if is_pkg(top, sub):
                on_disk[sub] = top
    problems += [f"group folders '{top}' and '{on_disk[p]}' both hold a package '{p}'" for p in on_disk for top in group_dirs
                 if top != on_disk[p] and is_pkg(top, p)]
    for p, top in sorted(on_disk.items()):
        if p not in known:
            problems.append(f"package '{top}/{p}' exists but is not assigned to a job group")
        elif group_of(p).key != top:
            problems.append(f"package '{p}' is in the folder of job group '{top}' but belongs to '{group_of(p).key}'")
    problems += [f"job group lists package '{p}' which does not exist" for p in sorted(known - set(on_disk))]
    problems += [f"package '{p}' has no allow-list" for p in sorted(known - set(ALLOWED))]
    seen = [p for g in GROUPS for p in g.packages]
    problems += [f"package '{p}' is in more than one job group" for p in sorted({p for p in seen if seen.count(p) > 1})]
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            parts = ["algoplatform"] + os.path.relpath(path, root)[:-3].split(os.sep)
            if len(parts) < 4 or parts[1] not in group_dirs:
                continue                                       # top-level modules (this file, __main__) and group folders' own files
            pkg = parts[2]
            for target, line in _imports(path, parts, group_dirs):
                if target == pkg or target not in on_disk:
                    continue
                where = f"{os.path.relpath(path, root)}:{line}"
                if target not in ALLOWED.get(pkg, frozenset()):
                    problems.append(f"{where}: '{pkg}' may not import '{target}' (allowed: {', '.join(sorted(ALLOWED.get(pkg, ()))) or 'nothing'})")
                if pkg in LIBRARIES and target in FORBIDDEN_FOR_LIBRARIES:
                    problems.append(f"{where}: library '{pkg}' must stay pure computation and cannot import '{target}'")
                if target in NEVER.get(pkg, frozenset()):
                    problems.append(f"{where}: '{pkg}' must never depend on '{target}'")
    return sorted(set(problems))


def structure_text() -> str:
    """The job-group map as plain text."""
    lines = ["algoplatform: one platform, grouped by job", ""]
    for i, g in enumerate(GROUPS, 1):
        roles = ", ".join(r.value for r in g.roles) or "all roles (through the application layer)"
        lines += [f"{i:2d}. {g.title}", f"    job:      {g.job}", f"    folder:   src/algoplatform/{g.key}/",
                  f"    packages: {', '.join(g.packages)}", f"    roles:    {roles}", ""]
    return "\n".join(lines)
