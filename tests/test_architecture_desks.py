import io
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import helpers
from helpers import free_config, market_from_closes
from algoplatform import architecture
from algoplatform.analytics import evaluate_strategy
from algoplatform.app import DESKS, TradingPlatform, desks_for, open_desk
from algoplatform.app.cli import main as cli_main
from algoplatform.core.enums import KillSwitchState, Permission, Role, Severity
from algoplatform.core.exceptions import AuthorizationError, ConfigError, GovernanceError
from algoplatform.governance import Stage
from algoplatform.notify import MemoryChannel, drawdown_alert, price_alert
from algoplatform.optimize import Param, ParameterSpace
from algoplatform.relationships import RelationshipManager
from algoplatform.selection import UniverseSelector
from algoplatform.strategy import SMACrossover
from algoplatform.users import ROLE_PERMISSIONS, PasswordHasher

SRC = os.path.dirname(os.path.abspath(architecture.__file__))
P = Permission


class ArchitectureTests(unittest.TestCase):
    def test_the_source_tree_obeys_its_own_rules(self):
        self.assertEqual(architecture.check_architecture(), [])

    def test_every_package_has_exactly_one_job(self):
        on_disk = {d for d in os.listdir(SRC) if os.path.exists(os.path.join(SRC, d, "__init__.py"))}
        listed = architecture.packages()
        self.assertEqual(set(listed), on_disk)
        self.assertEqual(len(listed), len(set(listed)))
        self.assertEqual(set(architecture.ALLOWED), on_disk)
        self.assertEqual(architecture.group_of("technical").key, "signals")
        self.assertEqual(architecture.group_of("governance").key, "operations")
        with self.assertRaises(KeyError):
            architecture.group_of("nope")
        text = architecture.structure_text()
        for pkg in listed:
            self.assertIn(pkg, text)

    def test_the_stated_invariants_hold_in_the_allow_list(self):
        allowed = architecture.ALLOWED
        self.assertEqual(allowed["core"], frozenset())
        self.assertEqual(allowed["utils"], frozenset({"core"}))
        for lib in architecture.LIBRARIES:
            self.assertEqual(allowed[lib] & architecture.FORBIDDEN_FOR_LIBRARIES, frozenset(), lib)   # libraries stay pure computation
        for pkg, never in architecture.NEVER.items():
            self.assertEqual(allowed[pkg] & never, frozenset(), pkg)
        self.assertFalse(any("web" in deps for deps in allowed.values()))                             # nothing depends on the UI
        self.assertEqual({p for p, deps in allowed.items() if "app" in deps}, {"backtest", "web"})

    def test_violations_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "algoplatform")
            shutil.copytree(SRC, root, ignore=shutil.ignore_patterns("__pycache__"))
            with open(os.path.join(root, "strategy", "sneaky.py"), "w") as fh:
                fh.write("from ..brokers.base import BrokerGateway\n")                                 # a strategy reaching for a broker
            with open(os.path.join(root, "technical", "impure.py"), "w") as fh:
                fh.write("from algoplatform.orders import oms\nimport algoplatform.risk.manager\n")    # a library reaching for the order manager
            os.mkdir(os.path.join(root, "mystery"))
            open(os.path.join(root, "mystery", "__init__.py"), "w").close()
            problems = architecture.check_architecture(root)
        text = "\n".join(problems)
        self.assertIn("strategy/sneaky.py:1: 'strategy' may not import 'brokers'", text)
        self.assertIn("'strategy' must never depend on 'brokers'", text)
        self.assertIn("technical/impure.py:1: library 'technical' must stay pure computation and cannot import 'orders'", text)
        self.assertIn("technical/impure.py:2: 'technical' may not import 'risk'", text)
        self.assertIn("package 'mystery' exists but is not assigned to a job group", text)
        self.assertEqual(len([p for p in problems if "sneaky" in p]), 2)


class RolePermissionTests(unittest.TestCase):
    def test_every_role_is_defined_and_only_admin_holds_everything(self):
        self.assertEqual(set(ROLE_PERMISSIONS), set(Role))
        self.assertEqual(ROLE_PERMISSIONS[Role.ADMIN], frozenset(Permission))
        for role, perms in ROLE_PERMISSIONS.items():
            if role is not Role.ADMIN:
                self.assertNotIn(P.MANAGE_USERS, perms, role)

    def test_segregation_of_duties(self):
        for role, perms in ROLE_PERMISSIONS.items():
            if role is Role.ADMIN:
                continue
            self.assertFalse({P.SUBMIT_STRATEGY, P.APPROVE_STRATEGY} <= perms, f"{role} could approve what it submits")
            self.assertFalse({P.PLACE_ORDER, P.RESET_KILL_SWITCH} <= perms, f"{role} could trade and reset its own kill switch")
            self.assertFalse({P.PLACE_ORDER, P.APPROVE_STRATEGY} <= perms, f"{role} could trade and approve strategies")
        traders = {r for r, p in ROLE_PERMISSIONS.items() if P.PLACE_ORDER in p}
        self.assertEqual(traders, {Role.ADMIN, Role.TRADER})                          # research, risk, audit and operations never send orders
        self.assertEqual({r for r, p in ROLE_PERMISSIONS.items() if P.RESET_KILL_SWITCH in p}, {Role.ADMIN, Role.RISK_MANAGER})
        self.assertEqual({r for r, p in ROLE_PERMISSIONS.items() if P.TRAIN_MODEL in p}, {Role.ADMIN, Role.QUANT})

    def test_read_only_roles_cannot_change_anything(self):
        view_only = {P.VIEW_ORDERS, P.VIEW_PORTFOLIO, P.VIEW_RISK, P.VIEW_AUDIT}
        self.assertTrue(ROLE_PERMISSIONS[Role.VIEWER] <= view_only)
        self.assertTrue(ROLE_PERMISSIONS[Role.AUDITOR] <= view_only)
        self.assertIn(P.VIEW_AUDIT, ROLE_PERMISSIONS[Role.AUDITOR])
        self.assertNotIn(P.VIEW_AUDIT, ROLE_PERMISSIONS[Role.VIEWER])

    def test_anyone_who_can_see_risk_can_stop_trading_only_if_entrusted(self):
        can_halt = {r for r, p in ROLE_PERMISSIONS.items() if P.TRIGGER_KILL_SWITCH in p}
        self.assertEqual(can_halt, {Role.ADMIN, Role.TRADER, Role.PORTFOLIO_MANAGER, Role.RISK_MANAGER, Role.OPERATIONS})


class DeskTestCase(unittest.TestCase):
    def setUp(self):
        self.p = TradingPlatform(free_config(log_level="CRITICAL"))
        self.p.users.hasher = PasswordHasher(n=2 ** 4)
        reg = self.p.users.register
        self.admin = reg("root_admin", helpers.STRONG_PW, Role.ADMIN)
        self.users = {Role.ADMIN: self.admin}
        for role in Role:
            if role is not Role.ADMIN:
                self.users[role] = reg(role.value.lower(), helpers.STRONG_PW, role, actor=self.admin)
        self.quant, self.trader, self.pm = self.users[Role.QUANT], self.users[Role.TRADER], self.users[Role.PORTFOLIO_MANAGER]
        self.risk, self.ops, self.auditor, self.viewer = (self.users[r] for r in (Role.RISK_MANAGER, Role.OPERATIONS, Role.AUDITOR, Role.VIEWER))
        self.md = market_from_closes(X=100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0.001, 0.01, 300))))
        idx = pd.bdate_range("2019-01-01", periods=1260)
        self.good = evaluate_strategy(pd.Series(np.random.default_rng(0).normal(0.0008, 0.008, 1260), index=idx), [100.0] * 100 + [-80.0] * 60)

    def paper_session(self, strategies=()):
        acc = self.p.accounts.link(self.trader, "paper", {})
        return self.p.desk(self.trader, "trading").open_session(acc.account_id, list(strategies))


class DeskAccessTests(DeskTestCase):
    def test_each_role_sees_exactly_its_desks(self):
        expected = {Role.ADMIN: set(DESKS), Role.QUANT: {"research", "risk"}, Role.TRADER: {"research", "trading", "risk"},
                    Role.PORTFOLIO_MANAGER: {"research", "portfolio", "risk"}, Role.RISK_MANAGER: {"risk", "compliance"},
                    Role.OPERATIONS: {"operations", "risk"}, Role.AUDITOR: {"compliance", "risk"}, Role.VIEWER: {"risk"}}
        for role, desks in expected.items():
            self.assertEqual(set(desks_for(self.p, self.users[role])), desks, role)
            self.assertEqual(set(self.p.desks(self.users[role])), desks, role)
        self.assertEqual(set(DESKS), {"research", "trading", "portfolio", "risk", "operations", "compliance", "admin"})

    def test_opening_a_desk_outside_the_role_is_refused(self):
        for user, desk in ((self.quant, "trading"), (self.trader, "portfolio"), (self.trader, "admin"), (self.risk, "trading"),
                           (self.auditor, "operations"), (self.viewer, "research"), (self.ops, "compliance"), (self.pm, "trading")):
            with self.assertRaises(AuthorizationError, msg=f"{user.username} -> {desk}"):
                open_desk(self.p, user, desk)
        with self.assertRaises(AuthorizationError):
            self.p.desk(self.admin, "treasury")

    def test_a_deactivated_user_loses_every_desk(self):
        self.p.desk(self.admin, "admin").deactivate(self.trader.user_id)
        self.assertEqual(desks_for(self.p, self.trader), {})
        with self.assertRaises(AuthorizationError):
            self.p.desk(self.trader, "trading")


class ResearchDeskTests(DeskTestCase):
    def test_research_workflow_backtest_optimise_evaluate_propose(self):
        desk = self.p.desk(self.quant, "research")
        result = desk.backtest("sma_crossover", self.md, params={"fast": 5, "slow": 20})
        self.assertGreater(len(result.trades), 0)
        space = ParameterSpace([Param("fast", "int", 3, 10), Param("slow", "int", 15, 40)])
        opt = desk.optimize("sma_crossover", self.md, space, method="bayesian", n_iter=12, n_init=6)
        self.assertEqual(opt.n_trials, 12)
        verdict = desk.evaluate(result, n_trials=opt.n_trials)
        self.assertEqual(verdict.metrics["n_trials"], 12)
        desk.register_strategy("sma_v1", {"params": opt.best_params})
        with self.assertRaises(GovernanceError):
            desk.propose("sma_v1", verdict)                                           # a random-walk backtest does not pass the scorecard
        desk.propose("sma_v1", self.good, note="separate evidence")
        self.assertEqual([r.strategy_id for r in self.p.lifecycle.pending()], ["sma_v1"])
        self.assertEqual(len(self.p.audit.entries("backtest.run")), 1)

    def test_research_cannot_reach_the_market(self):
        desk = self.p.desk(self.quant, "research")
        self.assertFalse(hasattr(desk, "place_order") or hasattr(desk, "open_session"))
        with self.assertRaises(AuthorizationError):
            self.p.accounts.link(self.quant, "paper", {})                             # cannot even link a broker account
        with self.assertRaises(AuthorizationError):
            self.p.open_session(self.quant, "ACC-1", [])

    def test_selection_relationships_and_models(self):
        desk = self.p.desk(self.quant, "research")
        md = market_from_closes(**{f"S{i}": 100 * np.exp(np.cumsum(np.random.default_rng(i).normal(0.0005 * i, 0.01, 300))) for i in range(5)})
        picked = desk.select_universe(UniverseSelector().require_history(200).rank_by("momentum").top(2), md)
        self.assertEqual(len(picked.symbols), 2)
        self.assertEqual(self.p.audit.entries("research.select_universe")[0]["details"]["chosen"], picked.symbols)
        rel = desk.relationships(md.closes(), corr_threshold=0.5)
        self.assertIsInstance(rel, RelationshipManager)
        version = desk.register_model("signal", object(), {"auc": 0.6}, data=np.ones((5, 2)))
        self.assertEqual((version.version, version.created_by), (1, "quant"))
        with self.assertRaises(AuthorizationError):
            self.p.desk(self.trader, "research").register_model("signal", object())   # traders may backtest, not publish models


class TradingAndRiskDeskTests(DeskTestCase):
    def test_live_account_only_accepts_approved_strategies(self):
        strategy = SMACrossover(["X"], strategy_id="sma_v1")
        live = self.p.accounts.link(self.trader, "paper", {"api_key": "k"}, paper=False)
        desk = self.p.desk(self.trader, "trading")
        with self.assertRaises(GovernanceError) as cm:
            desk.open_session(live.account_id, [(strategy, 1.0)])
        self.assertEqual((cm.exception.code, cm.exception.details["stage"]), ("NOT_APPROVED_FOR_LIVE", "UNREGISTERED"))
        research = self.p.desk(self.quant, "research")
        research.register_strategy("sma_v1")
        research.propose("sma_v1", self.good)
        self.p.desk(self.pm, "portfolio").approve("sma_v1")
        with self.assertRaises(GovernanceError) as cm:
            desk.open_session(live.account_id, [(strategy, 1.0)])
        self.assertEqual(cm.exception.details["stage"], "PAPER")                      # approved for paper is still not live
        self.p.lifecycle._now = lambda: datetime.now() + timedelta(days=30)
        self.p.lifecycle.get("sma_v1").stage_since = datetime.now()
        research.propose("sma_v1", self.good)
        self.p.desk(self.risk, "risk").approve("sma_v1")
        self.assertIs(self.p.lifecycle.stage("sma_v1"), Stage.LIVE)
        with self.assertRaises(ConfigError):                                          # governance passed; the platform itself is still in paper mode
            desk.open_session(live.account_id, [(strategy, 1.0)])
        engine = self.paper_session([(SMACrossover(["X"], strategy_id="unapproved"), 1.0)])   # paper accounts take anything
        self.assertEqual(engine.risk.kill_switch.state, KillSwitchState.ARMED)

    def test_trader_can_stop_but_only_risk_can_restart(self):
        engine = self.paper_session()
        trading, risk = self.p.desk(self.trader, "trading"), self.p.desk(self.risk, "risk")
        self.assertTrue(trading.halt(engine, "fat finger"))
        self.assertEqual(engine.risk.kill_switch.state, KillSwitchState.TRIGGERED)
        with self.assertRaises(AuthorizationError):
            self.p.reset_kill_switch(self.trader, engine, "let me trade again")
        with self.assertRaises(AuthorizationError):
            self.p.desk(self.viewer, "risk").resume(engine, "let me trade again")     # the viewer can look, not act
        with self.assertRaises(AuthorizationError):
            self.p.desk(self.viewer, "risk").halt(engine, "just because")
        risk.resume(engine, "position reconciled with the broker")
        self.assertEqual(engine.risk.kill_switch.state, KillSwitchState.ARMED)
        self.assertEqual(trading.cancel_all(engine), 0)

    def test_kill_switch_reaches_the_inbox_as_a_critical_notification(self):
        engine = self.paper_session()
        self.p.desk(self.risk, "risk").halt(engine, "daily loss limit")
        note = self.p.inbox.items[-1]
        self.assertEqual((note.title, note.severity, note.body), ("Kill switch", Severity.CRITICAL, "daily loss limit"))
        compliance = self.p.desk(self.auditor, "compliance")
        self.assertEqual([n["title"] for n in compliance.open_critical_alerts()], ["Kill switch"])
        self.p.notifier.acknowledge(note.id, "risk_manager")
        self.assertEqual(compliance.open_critical_alerts(), [])

    def test_risk_snapshot_and_concentration(self):
        engine = self.paper_session()
        snap = self.p.desk(self.viewer, "risk").snapshot(engine)
        self.assertEqual(snap["equity"], 100_000.0)
        self.assertEqual((snap["positions"], snap["kill_switch"], snap["gross_exposure"]), ({}, "ARMED", 0.0))
        self.assertNotIn("concentration", snap)
        rel = RelationshipManager()
        rel.add("X", "Y", "correlated", 0.9)
        self.assertNotIn("concentration", self.p.desk(self.risk, "risk").snapshot(engine, rel))   # nothing held, nothing to concentrate

    def test_alerts_personal_and_firm_wide(self):
        mine = MemoryChannel("mine")
        trading = self.p.desk(self.trader, "trading")
        trading.subscribe(mine)
        rule = trading.add_alert(price_alert("X", 200.0))
        self.assertEqual(rule.user_id, self.trader.user_id)                           # forced to be the trader's own
        self.p.desk(self.risk, "risk").add_alert(drawdown_alert(0.1))
        self.p.alerts.evaluate({"price.X": 201.0, "drawdown": 0.2})
        self.assertEqual([n.title for n in mine.items], ["Alert: price_X_above_200", "Alert: drawdown"])
        self.assertEqual([n.title for n in self.p.inbox.items], ["Alert: drawdown"])  # the shared inbox never sees the personal alert
        with self.assertRaises(AuthorizationError):
            self.p.desk(self.viewer, "risk").add_alert(drawdown_alert(0.5))
        with self.assertRaises(AuthorizationError):
            self.p.desk(self.quant, "risk").add_alert(drawdown_alert(0.5))


class ControlDeskTests(DeskTestCase):
    def test_portfolio_desk_allocates_and_approves(self):
        desk = self.p.desk(self.pm, "portfolio")
        out = desk.allocate({"trend": 0.6, "reversion": 0.4})
        self.assertAlmostEqual(out["trend"] / out["reversion"], 1.5)
        self.assertEqual(len(self.p.audit.entries("allocation.set")), 1)
        self.assertEqual(self.p.allocate_capital(self.risk, {"a": 1.0})["a"], out["trend"] + out["reversion"])   # risk may allocate too
        with self.assertRaises(AuthorizationError):
            self.p.allocate_capital(self.trader, {"a": 1.0})
        research = self.p.desk(self.quant, "research")
        research.register_strategy("s1")
        research.propose("s1", self.good)
        self.assertEqual([r.strategy_id for r in desk.pending_promotions()], ["s1"])
        desk.reject("s1", "needs a longer history")
        self.assertEqual(desk.pending_promotions(), [])
        research.propose("s1", self.good)
        self.assertIs(desk.approve("s1").stage, Stage.PAPER)

    def test_operations_desk(self):
        desk = self.p.desk(self.ops, "operations")
        self.p.health.register("broker", lambda: True)
        self.assertEqual(desk.health()["status"], "UP")
        report = desk.load_instruments([{"symbol": "AAA", "sector": "Tech"}, {"symbol": "BAD", "tick_size": "-1"}])
        self.assertEqual((report["loaded"], len(report["rejected"])), (1, 1))
        self.assertIn("AAA", self.p.instruments)
        self.assertEqual(self.p.audit.entries("instruments.load")[0]["details"], {"loaded": 1, "rejected": 1, "overwrite": False})

        class Down(MemoryChannel):
            fail = True

            def send(self, notification):
                if self.fail:
                    raise ConnectionError("unreachable")
                super().send(notification)
        pager = Down("pager")
        desk.add_channel(pager, Severity.ERROR)
        self.p.notifier.notify("Broker disconnected", severity=Severity.ERROR)
        self.assertEqual([(u["channel"], u["title"]) for u in desk.undelivered()], [("pager", "Broker disconnected")])
        pager.fail = False
        self.assertEqual(desk.replay_undelivered(), {"delivered": 1, "still_failing": 0})
        self.assertEqual([n.title for n in pager.items], ["Broker disconnected"])
        self.p.scheduler.every(3600, lambda: None, "hourly")
        self.assertEqual([j["name"] for j in desk.jobs()], ["hourly"])
        self.assertEqual(desk.run_scheduled(), [])
        with self.assertRaises(AuthorizationError):
            self.p.desk(self.ops, "operations")._need(Permission.PLACE_ORDER)

    def test_compliance_desk_is_read_only(self):
        self.p.desk(self.quant, "research").register_strategy("s1")
        desk = self.p.desk(self.auditor, "compliance")
        public = {name for name in dir(desk) if not name.startswith("_") and callable(getattr(desk, name))}
        self.assertEqual(public, {"audit_trail", "strategy_history", "strategies", "open_critical_alerts"})   # nothing that changes state
        self.assertIn("strategy.register", [e["action"] for e in desk.audit_trail()])
        self.assertEqual(len(desk.audit_trail("strategy.register")), 1)
        self.assertEqual(desk.strategies(), [{"strategy": "s1", "owner": "quant", "stage": "RESEARCH", "pending": False}])
        self.assertEqual(desk.strategy_history("s1"), [])

    def test_admin_desk(self):
        desk = self.p.desk(self.admin, "admin")
        new = desk.create_user("newbie", helpers.STRONG_PW, Role.QUANT)
        self.assertIs(new.role, Role.QUANT)
        desk.set_role(new.user_id, Role.TRADER)
        self.assertIn("trading", desks_for(self.p, new))
        with self.assertRaises(GovernanceError) as cm:
            desk.set_role(self.admin.user_id, Role.VIEWER)                            # the last admin cannot lock everyone out by accident
        self.assertEqual(cm.exception.code, "SELF_DEMOTION")
        self.assertEqual(len(desk.users()), len(Role) + 1)


class CommandLineTests(unittest.TestCase):
    def run_cli(self, *args):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli_main(list(args))
        return code, out.getvalue()

    def test_structure_command(self):
        code, text = self.run_cli("structure", "--check")
        self.assertEqual(code, 0)
        self.assertIn("dependency rules: OK", text)
        self.assertIn("Learning and AI", text)
        self.assertIn("PORTFOLIO_MANAGER", text)

    def test_functions_command(self):
        code, text = self.run_cli("functions", "--kind", "pattern")
        self.assertEqual(code, 0)
        self.assertIn("engulfing(open, high, low, close)", text)
        self.assertNotIn("operator", text)
        self.assertIn("crossover(a, b)", self.run_cli("functions", "--kind", "operator")[1])
        self.assertGreater(len(self.run_cli("functions")[1].splitlines()), 100)

    def test_optimize_accepts_the_new_methods(self):
        code, text = self.run_cli("optimize", "--strategy", "sma_crossover", "--method", "tpe", "--n-iter", "17")
        self.assertEqual(code, 0)
        self.assertIn('"method": "tpe"', text)
        self.assertIn('"trials": 17', text)


if __name__ == "__main__":
    unittest.main()
