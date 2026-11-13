import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

import helpers  # noqa: F401
from algoplatform.analytics import evaluate_strategy
from algoplatform.core import PlatformConfig, Role
from algoplatform.technical import CATALOG
from algoplatform.web.server import App

PW = "Correct-Horse-9"
ROLES = (Role.QUANT, Role.TRADER, Role.PORTFOLIO_MANAGER, Role.RISK_MANAGER, Role.OPERATIONS, Role.AUDITOR, Role.VIEWER)


def make_app(db_dir=None):
    app = App(PlatformConfig(log_level="CRITICAL"), fast_hash=True, db_dir=db_dir)
    P = app.platform
    if P.users.user_count() == 0:
        admin = P.users.register("admin", PW, Role.ADMIN)
        for role in ROLES:
            P.users.register(role.value.lower(), PW, role, actor=admin)
        app._save_users()
    return app


class Client:
    def __init__(self, app):
        self.app = app
        self.tokens = {}

    def __call__(self, who, method, path, body=None):
        if who not in self.tokens:
            self.tokens[who] = self.app.platform.users.login(who, PW)
        return self.app.dispatch(method, path, body or {}, self.tokens[who])


class WebQuantTestCase(unittest.TestCase):
    def setUp(self):
        self.app = make_app()
        self.call = Client(self.app)

    def ok(self, who, method, path, body=None):
        code, obj = self.call(who, method, path, body)
        self.assertEqual(code, 200, obj)
        return obj


class StructureAndAccessTests(WebQuantTestCase):
    def test_structure_reports_the_role_desks_and_rules(self):
        quant = self.ok("quant", "GET", "/api/structure")
        self.assertEqual((quant["role"], quant["desks"], quant["violations"]), ("QUANT", ["research", "risk"], []))
        self.assertEqual(len(quant["groups"]), 12)
        self.assertEqual(self.ok("viewer", "GET", "/api/structure")["desks"], ["risk"])
        self.assertEqual(len(self.ok("viewer", "GET", "/api/functions")), len(CATALOG))       # other tests may register extra functions
        self.assertIn("momentum", self.ok("viewer", "GET", "/api/lab/options")["rankers"])

    def test_research_endpoints_need_the_research_desk(self):
        for who in ("viewer", "auditor", "risk_manager", "operations"):
            for path in ("/api/lab/technical", "/api/lab/volatility", "/api/lab/select", "/api/governance/evaluate"):
                code, obj = self.call(who, "POST", path, {"symbols": ["AAA", "BBB"], "strategy": "sma_crossover"})
                self.assertEqual((code, obj["error"]), (403, "FORBIDDEN"), f"{who} {path}")

    def test_missing_token_is_refused(self):
        code, _ = self.app.dispatch("GET", "/api/structure", {}, None)
        self.assertEqual(code, 401)


class LabTests(WebQuantTestCase):
    def test_technical_with_condition_and_numeric_formulas(self):
        r = self.ok("quant", "POST", "/api/lab/technical", {"symbols": ["AAA"], "bars": 500,
                                                           "formula": "crossover(ema(close, 10), ema(close, 40)) and rsi(close, 14) < 70"})
        self.assertEqual(r["symbol"], "AAA")
        self.assertEqual(set(r["indicators"]), {"rsi", "adx", "supertrend", "natr", "choppiness", "relative_volume"})
        f = r["formula"]
        self.assertTrue(f["condition"])
        self.assertEqual((f["columns"], f["functions"]), (["close"], ["crossover", "ema", "rsi"]))
        self.assertLessEqual(len(f["events"]), f["true_bars"])
        self.assertIsNone(f["series"])
        dates = [e["date"] for e in f["events"]]
        self.assertEqual(dates, sorted(dates, reverse=True))                              # newest first
        self.assertTrue(all(set(p) == {"name", "direction", "date", "level", "target"} for p in r["chart_patterns"]))
        self.assertTrue(all(c["count"] > 0 for c in r["candles"]))
        numeric = self.ok("quant", "POST", "/api/lab/technical", {"symbols": ["AAA"], "bars": 300, "formula": "zscore(close, 20)"})["formula"]
        self.assertFalse(numeric["condition"])
        self.assertEqual(len(numeric["series"]), 281)                                     # warm-up rows are dropped, not invented
        self.assertNotIn("formula", self.ok("quant", "POST", "/api/lab/technical", {"symbols": ["AAA"], "bars": 300}))
        constant = self.ok("quant", "POST", "/api/lab/technical", {"symbols": ["AAA"], "bars": 300, "formula": "1 < 2"})["formula"]
        self.assertEqual(constant["true_bars"], 300)

    def test_formula_errors_are_client_errors_not_crashes(self):
        for text in ("__import__('os').system('ls')", "close.__class__", "smaa(close, 3)", "macd(close)", "close ** 99"):
            code, obj = self.call("quant", "POST", "/api/lab/technical", {"symbols": ["AAA"], "bars": 200, "formula": text})
            self.assertEqual((code, obj["error"]), (400, "ExpressionError"), text)

    def test_volatility(self):
        r = self.ok("quant", "POST", "/api/lab/volatility", {"symbols": ["AAA"], "bars": 600, "model": "ewma", "target_vol": 0.15, "league_garch": False})
        self.assertEqual({e["name"] for e in r["estimators"]}, {"close_to_close", "parkinson", "garman_klass", "rogers_satchell", "yang_zhang", "ewma"})
        a = r["assessment"]
        self.assertIn(a["regime"], ("low", "normal", "high"))
        self.assertAlmostEqual(a["base_scalar"], min(0.15 / a["forecast_vol"], 2.0))
        self.assertEqual(sorted(x["model"] for x in r["league"]), ["ewma", "har", "hist"])
        self.assertEqual([c["window"] for c in r["cone"]], [10, 21, 63, 126, 252])
        self.assertTrue(r["series"]["yang_zhang"] and r["series"]["ewma"])
        code, obj = self.call("quant", "POST", "/api/lab/volatility", {"symbols": ["AAA"], "model": "lstm"})
        self.assertEqual(code, 400)
        short = self.ok("quant", "POST", "/api/lab/volatility", {"symbols": ["AAA"], "bars": 200})
        self.assertEqual((short["league"], short["cone"]), (None, []))                  # too little history: said so, not faked

    def test_selection_is_audited_and_explains_rejections(self):
        r = self.ok("quant", "POST", "/api/lab/select", {"symbols": ["A1", "A2", "A3", "A4", "A5", "A6"], "bars": 500, "top": 2,
                                                         "rankers": [{"name": "momentum", "weight": 1}, {"name": "trend_quality", "weight": 2}]})
        self.assertEqual(len(r["selected"]), 2)
        self.assertEqual(len(r["scores"]), 6)
        self.assertTrue({"symbol", "momentum", "trend_quality", "composite"} <= set(r["scores"][0]))
        self.assertEqual(len(r["rejected"]), 4)
        self.assertEqual(set(r["correlation"]), {"A1", "A2", "A3", "A4", "A5", "A6"})
        self.assertAlmostEqual(r["concentration"]["effective_bets"], 2.0, delta=1.0)
        entries = self.app.platform.audit.entries("research.select_universe")
        self.assertEqual(entries[-1]["details"]["chosen"], r["selected"])
        code, _ = self.call("quant", "POST", "/api/lab/select", {"symbols": ["ONLY"]})
        self.assertEqual(code, 400)
        code, _ = self.call("quant", "POST", "/api/lab/select", {"symbols": ["A1", "A2"], "rankers": [{"name": "astrology"}]})
        self.assertEqual(code, 400)


def passing_verdict():
    idx = pd.bdate_range("2019-01-01", periods=1260)
    return evaluate_strategy(pd.Series(np.random.default_rng(0).normal(0.0008, 0.008, 1260), index=idx), [100.0] * 100 + [-80.0] * 60)


class GovernanceTests(WebQuantTestCase):
    BT = {"strategy": "sma_crossover", "symbols": ["AAA"], "bars": 600}

    def test_capabilities_follow_the_role(self):
        expected = {"quant": (True, True, False, False), "trader": (True, True, False, False), "portfolio_manager": (False, False, True, True),
                    "risk_manager": (False, False, True, True), "auditor": (False, False, False, False), "viewer": (False, False, False, False)}
        for who, (reg, prop, appr, dem) in expected.items():
            can = self.ok(who, "GET", "/api/governance")["can"]
            self.assertEqual((can["register"], can["propose"], can["approve"], can["demote"]), (reg, prop, appr, dem), who)

    def test_register_score_and_a_failing_proposal_is_refused(self):
        self.ok("quant", "POST", "/api/governance/register", {"strategy_id": "sma_v1", **self.BT})
        verdict = self.ok("quant", "POST", "/api/governance/evaluate", {**self.BT, "n_trials": 20})
        self.assertEqual(len(verdict["checks"]), 9)
        self.assertIn("20 trials", " ".join(c["detail"] for c in verdict["checks"]))
        if not verdict["passed"]:
            code, obj = self.call("quant", "POST", "/api/governance/propose", {"strategy_id": "sma_v1", **self.BT})
            self.assertEqual((code, obj["error"]), (409, "EVIDENCE_FAILS"))
        rows = self.ok("viewer", "GET", "/api/governance")["strategies"]
        self.assertEqual([(r["strategy_id"], r["owner"], r["stage"], r["description"]["strategy"]) for r in rows], [("sma_v1", "quant", "RESEARCH", "sma_crossover")])
        for bad in ("", "has space", "x" * 41, "semi;colon"):
            code, _ = self.call("quant", "POST", "/api/governance/register", {"strategy_id": bad})
            self.assertEqual(code, 400, bad)
        code, obj = self.call("quant", "POST", "/api/governance/register", {"strategy_id": "sma_v1"})
        self.assertEqual((code, obj["error"]), (409, "DUPLICATE_STRATEGY"))
        code, _ = self.call("viewer", "POST", "/api/governance/register", {"strategy_id": "v1"})
        self.assertEqual(code, 403)

    def test_decisions_and_four_eyes(self):
        P = self.app.platform
        quant = P.users.get(P.users._by_name["quant"])
        P.lifecycle.register(quant, "trend")
        P.lifecycle.submit(quant, "trend", passing_verdict())
        pending = self.ok("viewer", "GET", "/api/governance")["strategies"][0]["pending"]
        self.assertEqual((pending["target"], pending["by"], pending["grade"]), ("PAPER", "quant", "A"))
        code, obj = self.call("quant", "POST", "/api/governance/decide", {"strategy_id": "trend", "action": "approve"})
        self.assertEqual(code, 403)                                                       # research cannot approve
        code, obj = self.call("portfolio_manager", "POST", "/api/governance/decide", {"strategy_id": "trend", "action": "reject"})
        self.assertEqual((code, obj["error"]), (409, "REASON_REQUIRED"))
        self.assertEqual(self.ok("portfolio_manager", "POST", "/api/governance/decide", {"strategy_id": "trend", "action": "approve", "reason": "ok"})["stage"], "PAPER")
        code, obj = self.call("risk_manager", "POST", "/api/governance/decide", {"strategy_id": "trend", "action": "demote", "reason": "x"})
        self.assertEqual((code, obj["error"]), (409, "NOT_LIVE"))
        self.assertEqual(self.ok("quant", "POST", "/api/governance/decide", {"strategy_id": "trend", "action": "retire", "reason": "v2 ready"})["stage"], "RETIRED")
        code, _ = self.call("quant", "POST", "/api/governance/decide", {"strategy_id": "trend", "action": "teleport"})
        self.assertEqual(code, 400)
        history = self.ok("auditor", "GET", "/api/governance")["strategies"][0]["history"]
        self.assertEqual([h["action"] for h in history], ["retire", "approve"])           # newest first


class AlertAndInboxTests(WebQuantTestCase):
    def test_personal_and_firm_alerts_fire_from_paper_trading(self):
        self.ok("trader", "POST", "/api/session/open", {"symbols": ["AAA"], "seed": 3})
        self.ok("trader", "POST", "/api/alerts/add", {"kind": "price", "symbol": "aaa", "level": 1.0, "direction": "above"})
        self.ok("risk_manager", "POST", "/api/alerts/add", {"kind": "metric", "metric": "equity", "op": ">", "value": 0, "scope": "firm", "severity": "ERROR"})
        self.ok("trader", "POST", "/api/session/step", {"n": 2})
        trader = self.ok("trader", "GET", "/api/inbox")
        self.assertEqual([n["title"] for n in trader["personal"]], ["Alert: trader: AAA above 1"])
        self.assertEqual([n["title"] for n in trader["platform"]], ["Alert: equity > 0"])
        self.assertEqual({a["name"]: a["personal"] for a in trader["alerts"]}, {"trader: AAA above 1": True, "equity > 0": False})
        self.assertEqual(trader["can_add"], {"personal": True, "firm": True})
        risk = self.ok("risk_manager", "GET", "/api/inbox")
        self.assertEqual(risk["personal"], [])                                            # never sees the trader's own alert ...
        self.assertEqual([a["name"] for a in risk["alerts"]], ["equity > 0"])             # ... or the rule behind it
        self.assertEqual(self.ok("viewer", "GET", "/api/inbox")["can_add"], {"personal": False, "firm": False})

    def test_who_may_add_and_remove(self):
        cases = [("quant", {"kind": "price", "symbol": "AAA", "level": 1}, 403), ("viewer", {"kind": "metric", "metric": "equity", "value": 1, "scope": "firm"}, 403),
                 ("trader", {"kind": "metric", "metric": "secret_sauce", "value": 1}, 400), ("trader", {"kind": "price", "symbol": "A A", "level": 1}, 400),
                 ("trader", {"kind": "price", "symbol": "AAA", "level": 1, "direction": "sideways"}, 400),
                 ("trader", {"kind": "price", "symbol": "AAA", "level": 1, "severity": "APOCALYPTIC"}, 400)]
        for who, body, code in cases:
            self.assertEqual(self.call(who, "POST", "/api/alerts/add", body)[0], code, (who, body))
        self.ok("trader", "POST", "/api/alerts/add", {"kind": "price", "symbol": "AAA", "level": 5})
        self.ok("risk_manager", "POST", "/api/alerts/add", {"kind": "metric", "metric": "drawdown", "value": 0.1, "scope": "firm"})
        self.assertEqual(self.call("risk_manager", "POST", "/api/alerts/remove", {"name": "trader: AAA above 5"})[0], 403)
        self.assertEqual(self.call("trader", "POST", "/api/alerts/remove", {"name": "drawdown > 0.1"})[0], 200)       # traders manage alerts
        self.assertEqual(self.call("viewer", "POST", "/api/alerts/remove", {"name": "trader: AAA above 5"})[0], 403)
        self.ok("trader", "POST", "/api/alerts/remove", {"name": "trader: AAA above 5"})
        self.assertEqual(self.app.platform.alerts.describe("x"), [])
        self.assertEqual(len(self.app.platform.audit.entries("alert.remove")), 2)

    def test_critical_alerts_are_acknowledged_by_risk_not_by_the_trader(self):
        self.ok("trader", "POST", "/api/session/open", {"symbols": ["AAA"]})
        self.ok("trader", "POST", "/api/session/killswitch", {"action": "trip", "reason": "fat finger"})
        crit = self.ok("viewer", "GET", "/api/inbox")["open_critical"]
        self.assertEqual([(n["title"], n["body"]) for n in crit], [("Kill switch", "fat finger")])
        code, obj = self.call("trader", "POST", "/api/inbox/ack", {"id": crit[0]["id"]})
        self.assertEqual((code, obj["error"]), (403, "FORBIDDEN"))
        self.ok("risk_manager", "POST", "/api/inbox/ack", {"id": crit[0]["id"]})
        self.assertEqual(self.ok("viewer", "GET", "/api/inbox")["open_critical"], [])
        self.assertEqual(self.call("operations", "POST", "/api/inbox/ack", {"id": crit[0]["id"]})[0], 400)          # already acknowledged
        self.assertEqual(self.app.platform.audit.entries("notify.acknowledge")[0]["actor"], "risk_manager")

    def test_a_broken_alert_never_stops_the_session(self):
        self.ok("trader", "POST", "/api/session/open", {"symbols": ["AAA"]})
        self.app.platform.alerts.evaluate = lambda metrics: (_ for _ in ()).throw(RuntimeError("alert engine down"))
        snap = self.ok("trader", "POST", "/api/session/step", {"n": 3})
        self.assertEqual(snap["bars"], 4)


class RestartTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, True)

    def test_governance_alerts_and_inboxes_survive_a_restart(self):
        app = make_app(self.dir)
        call = Client(app)
        P = app.platform
        quant = P.users.get(P.users._by_name["quant"])
        self.assertEqual(call("quant", "POST", "/api/governance/register", {"strategy_id": "kept"})[0], 200)
        P.lifecycle.submit(quant, "kept", passing_verdict())
        P.autosave()
        self.assertEqual(call("trader", "POST", "/api/alerts/add", {"kind": "price", "symbol": "AAA", "level": 5})[0], 200)
        self.assertEqual(call("trader", "GET", "/api/inbox")[0], 200)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "platform.db")))
        app.close()

        again = make_app(self.dir)                                                        # a new process on the same directory
        call2 = Client(again)
        rows = call2("viewer", "GET", "/api/governance")[1]["strategies"]
        self.assertEqual([(r["strategy_id"], r["pending"]["target"]) for r in rows], [("kept", "PAPER")])
        self.assertEqual(call2("portfolio_manager", "POST", "/api/governance/decide", {"strategy_id": "kept", "action": "approve", "reason": "ok"})[1]["stage"], "PAPER")
        inbox = call2("trader", "GET", "/api/inbox")[1]
        self.assertEqual([a["name"] for a in inbox["alerts"]], ["trader: AAA above 5"])
        names = [r.channel.name for r in again.platform.notifier.routes]
        self.assertEqual(len(names), len(set((r.channel.name, r.user_id) for r in again.platform.notifier.routes)))   # the personal inbox was reused, not duplicated
        call2("trader", "POST", "/api/session/open", {"symbols": ["AAA"]})
        call2("trader", "POST", "/api/session/step", {"n": 1})
        self.assertEqual([n["title"] for n in call2("trader", "GET", "/api/inbox")[1]["personal"]], ["Alert: trader: AAA above 5"])
        again.close()


if __name__ == "__main__":
    unittest.main()
