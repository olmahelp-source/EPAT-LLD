import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import helpers
from helpers import fast_user_manager
from algoplatform.analytics import evaluate_strategy
from algoplatform.app import TradingPlatform
from algoplatform.core.config import PlatformConfig
from algoplatform.core.enums import Role, Severity
from algoplatform.core.exceptions import ConfigError, GovernanceError, ModelError
from algoplatform.governance import ModelRegistry, Stage, StrategyLifecycle
from algoplatform.instruments import ContractSpec, InstrumentMaster, future_spec, option_spec
from algoplatform.notify import (AlertManager, AlertRule, EmailChannel, FileChannel, LogChannel, MemoryChannel, Notification, Notifier,
                                 SlackChannel, TelegramChannel, WebhookChannel, channel_from_config, drawdown_alert, price_alert, threshold)
from algoplatform.persistence import SqliteStore
from algoplatform.persistence.codec import dumps
from algoplatform.users import PasswordHasher
from algoplatform.utils import json_safe


def through_json(state):
    """What the store does to a value: strict JSON out and back in."""
    return json.loads(dumps(json_safe(state)))


def good_verdict():
    idx = pd.bdate_range("2019-01-01", periods=1260)
    return evaluate_strategy(pd.Series(np.random.default_rng(0).normal(0.0008, 0.008, 1260), index=idx), [100.0] * 100 + [-80.0] * 60)


def refuse(*args):
    raise AssertionError("a test must never reach the network")


class JsonSafeTests(unittest.TestCase):
    def test_everything_becomes_strict_json(self):
        raw = {"nan": float("nan"), "inf": np.float64("inf"), "np": np.int64(3), "arr": np.array([1.5, np.nan]), "when": datetime(2024, 1, 2, 3),
               "role": Role.QUANT, "tuple": (1, 2), "set": {"b", "a"}, 5: "int key", "nested": [{"x": np.bool_(True)}], "obj": object}
        safe = json_safe(raw)
        text = json.dumps(safe, allow_nan=False)                                      # would raise on NaN or infinity
        back = json.loads(text)
        self.assertEqual((back["nan"], back["inf"], back["np"], back["arr"]), (None, None, 3, [1.5, None]))
        self.assertEqual((back["when"], back["role"], back["tuple"], back["set"], back["5"]), ("2024-01-02T03:00:00", "QUANT", [1, 2], ["a", "b"], "int key"))
        self.assertIs(back["nested"][0]["x"], True)
        self.assertIsInstance(back["obj"], str)


class InstrumentStateTests(unittest.TestCase):
    def test_round_trip_keeps_every_field_and_mapping(self):
        m = InstrumentMaster([ContractSpec("AAPL", name="Apple", sector="Tech", isin="US0378331005", broker_symbols={"zerodha": "AAPL-EQ"}),
                              future_spec("ES", "2025-03-21", 50, 0.25, margin_pct=0.06), option_spec("AAPL", "2025-01-17", "put", 150)])
        m.deactivate("AAPL250117P00150000")
        restored = InstrumentMaster()
        report = restored.import_state(through_json(m.export_state()))
        self.assertEqual((report["loaded"], report["rejected"]), (3, []))
        for symbol in ("AAPL", "ESH25", "AAPL250117P00150000"):
            self.assertEqual(restored.get(symbol), m.get(symbol))
        self.assertEqual(restored.resolve("zerodha", "AAPL-EQ").symbol, "AAPL")
        self.assertFalse(restored.get("AAPL250117P00150000").active)

    def test_a_damaged_record_is_reported_not_fatal(self):
        state = through_json(InstrumentMaster([ContractSpec("AAA"), ContractSpec("BBB")]).export_state())
        state[0]["tick_size"] = -1
        report = InstrumentMaster().import_state(state)
        self.assertEqual((report["loaded"], len(report["rejected"])), (1, 1))


class LifecycleStateTests(unittest.TestCase):
    def setUp(self):
        um = fast_user_manager()
        self.admin = um.register("admin", helpers.STRONG_PW, Role.ADMIN)
        self.quant = um.register("quant", helpers.STRONG_PW, Role.QUANT, self.admin)
        self.pm = um.register("pm", helpers.STRONG_PW, Role.PORTFOLIO_MANAGER, self.admin)
        self.risk = um.register("risk", helpers.STRONG_PW, Role.RISK_MANAGER, self.admin)
        self.now = [datetime(2024, 1, 1)]
        self.lc = StrategyLifecycle(now=lambda: self.now[0])
        self.verdict = good_verdict()

    def restart(self):
        fresh = StrategyLifecycle(now=lambda: self.now[0])
        fresh.import_state(through_json(self.lc.export_state()))
        return fresh

    def test_stages_history_and_pending_requests_survive(self):
        self.lc.register(self.quant, "live_one", {"family": "trend"})
        self.lc.submit(self.quant, "live_one", self.verdict)
        self.lc.approve(self.pm, "live_one", "sound")
        self.now[0] += timedelta(days=30)
        self.lc.submit(self.quant, "live_one", self.verdict)
        self.lc.approve(self.risk, "live_one")
        self.lc.register(self.quant, "waiting")
        self.lc.submit(self.quant, "waiting", self.verdict, note="please review")
        after = self.restart()
        self.assertIs(after.stage("live_one"), Stage.LIVE)
        after.require_live("live_one")                                                # an approval given before the restart still stands
        self.assertEqual(after.get("live_one").history, self.lc.get("live_one").history)
        self.assertEqual(after.get("live_one").description, {"family": "trend"})
        self.assertEqual(after.get("live_one").stage_since, self.lc.get("live_one").stage_since)
        pending = after.get("waiting").pending
        self.assertEqual((pending.target, pending.submitted_by, pending.note, pending.evidence["grade"]), (Stage.PAPER, "quant", "please review", "A"))
        self.assertEqual([r.strategy_id for r in after.pending()], ["waiting"])

    def test_the_controls_still_apply_after_a_restart(self):
        self.lc.register(self.admin, "own")
        self.lc.submit(self.admin, "own", self.verdict)
        after = self.restart()
        with self.assertRaises(GovernanceError) as cm:
            after.approve(self.admin, "own")                                          # the submitter is remembered
        self.assertEqual(cm.exception.code, "SELF_APPROVAL")
        after.approve(self.pm, "own")
        self.assertIs(after.stage("own"), Stage.PAPER)
        with self.assertRaises(GovernanceError) as cm:
            after.submit(self.admin, "own", self.verdict)                             # the paper period restarts from the stored date
        self.assertEqual(cm.exception.code, "PAPER_PERIOD")

    def test_evidence_with_non_finite_numbers_can_be_stored(self):
        lottery = evaluate_strategy(pd.Series(np.random.default_rng(1).normal(0.0, 0.01, 300)), [-1.0, -2.0])   # top_share is infinite
        self.assertIn(float("inf"), [c.value for c in lottery.checks] + [lottery.metrics["trades"]["top_share"]])
        dumps(json_safe(lottery.to_dict()))                                           # does not raise


class ModelRegistryStateTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(0).normal(size=(40, 3))
        self.reg = ModelRegistry()
        self.reg.register("signal", "model-1", {"auc": 0.58, "bad": float("nan")}, {"depth": 3}, data=self.X, created_by="quant", notes="first")
        self.reg.register("signal", "model-2", {"auc": 0.61}, data=self.X)
        self.reg.promote("signal", 1, "production")
        self.reg.promote("signal", 2, "production")

    def restart(self):
        fresh = ModelRegistry()
        fresh.import_state(through_json(self.reg.export_state()))
        return fresh

    def test_metadata_survives_but_model_objects_do_not(self):
        text = dumps(json_safe(self.reg.export_state()))
        self.assertNotIn("model-1", text)                                             # the object itself is never written
        after = self.restart()
        self.assertEqual([(v.version, v.stage) for v in after.versions("signal")], [(1, "archived"), (2, "production")])
        prod = after.get("signal")
        self.assertIsNone(prod.model)
        self.assertEqual((prod.metrics, prod.data_fingerprint), ({"auc": 0.61}, self.reg.get("signal").data_fingerprint))
        self.assertEqual(after.versions("signal")[0].metrics, {"auc": 0.58})          # the non-finite metric was dropped, not stored as NaN
        self.assertEqual((after.versions("signal")[0].created_by, after.versions("signal")[0].notes), ("quant", "first"))
        self.assertEqual(after.missing_models(), ["signal:2"])                        # production needs its object back

    def test_attach_checks_the_training_data(self):
        after = self.restart()
        with self.assertRaises(GovernanceError) as cm:
            after.attach("signal", 2, "impostor", data=self.X + 1e-9)
        self.assertEqual(cm.exception.code, "FINGERPRINT_MISMATCH")
        self.assertIsNone(after.get("signal").model)
        after.attach("signal", 2, "model-2", data=self.X)
        self.assertEqual((after.get("signal").model, after.missing_models()), ("model-2", []))
        with self.assertRaises(ModelError):
            after.attach("signal", 9, "x")
        self.assertEqual(after.rollback("signal").version, 1)                         # history is intact, so rollback still works
        after.import_state(through_json(self.reg.export_state()))                     # importing twice does not duplicate versions
        self.assertEqual(len(after.versions("signal")), 2)


class AlertStateTests(unittest.TestCase):
    def setUp(self):
        self.t = [1000.0]
        self.notifier = Notifier(clock=lambda: self.t[0], dedupe_window=0)
        self.box = MemoryChannel()
        self.notifier.add_channel(self.box)
        self.alerts = AlertManager(self.notifier, clock=lambda: self.t[0])

    def test_data_rules_are_restored_and_work(self):
        self.alerts.add(drawdown_alert(0.10, for_count=2, cooldown=120))
        self.alerts.add(price_alert("AAPL", 200.0, "below", user_id="u1"))
        self.alerts.add(threshold("latency", "latency_ms", ">=", 500, severity=Severity.CRITICAL, message="slow: {latency_ms}"))
        self.alerts.add(AlertRule("custom", lambda m: m.get("x", 0) > 1))
        self.alerts.silence("latency", 3600)
        state = through_json(self.alerts.export_state())
        self.assertEqual(state["not_storable"], ["custom"])                           # arbitrary code cannot be written down
        fresh = AlertManager(self.notifier, clock=lambda: self.t[0])
        report = fresh.import_state(state)
        self.assertEqual((report["restored"], report["failed"], report["not_storable"]), (3, [], ["custom"]))
        self.assertEqual(sorted(fresh.states()), ["drawdown", "latency", "price_AAPL_below_200"])
        self.assertEqual(fresh.evaluate({"drawdown": 0.2}), [])                       # for_count=2 was kept: one reading is not enough
        fired = fresh.evaluate({"drawdown": 0.2, "price.AAPL": 150.0, "latency_ms": 900})
        self.assertEqual({(e.rule, e.transition) for e in fired}, {("drawdown", "fired"), ("price_AAPL_below_200", "fired")})   # latency is still silenced
        self.assertEqual(self.box.items[0].body, "drawdown 20.0% exceeds 10%")
        self.assertEqual(fresh._rule("price_AAPL_below_200").user_id, "u1")
        self.assertIs(fresh._rule("latency").severity, Severity.CRITICAL)

    def test_a_bad_stored_rule_is_skipped_and_reported(self):
        self.alerts.add(drawdown_alert(0.10))
        state = through_json(self.alerts.export_state())
        state["rules"].append({"name": "broken", "definition": {"kind": "threshold", "metric": "m", "op": "~", "value": 1}, "severity": "ERROR"})
        state["rules"].append({"name": "alien", "definition": {"kind": "python"}, "severity": "ERROR"})
        report = AlertManager(self.notifier).import_state(state)
        self.assertEqual(report["restored"], 1)
        self.assertEqual([f["rule"] for f in report["failed"]], ["broken", "alien"])


class NotifierStateTests(unittest.TestCase):
    def setUp(self):
        self.t = [1000.0]
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.n = Notifier(clock=lambda: self.t[0], sleep=lambda s: None, max_attempts=1)

    def fresh(self):
        return Notifier(clock=lambda: self.t[0], sleep=lambda s: None, max_attempts=1)

    def test_secrets_are_never_written_to_the_store(self):
        self.n.add_channel(SlackChannel("https://hooks.example.test/T/SLACK-SECRET-PATH", transport=refuse))
        self.n.add_channel(TelegramChannel("123456:TELEGRAM-SECRET", "42", transport=refuse))
        self.n.add_channel(WebhookChannel("https://example.test/hook?key=WEBHOOK-SECRET", secret="HMAC-SECRET", transport=refuse))
        self.n.add_channel(EmailChannel("smtp.example.test", "a@b", ["c@d"], username="bot", password="SMTP-SECRET", smtp_factory=refuse))
        text = dumps(json_safe(self.n.export_state()))
        for secret in ("SLACK-SECRET-PATH", "TELEGRAM-SECRET", "WEBHOOK-SECRET", "HMAC-SECRET", "SMTP-SECRET"):
            self.assertNotIn(secret, text)
        self.assertEqual([r["config"] for r in json.loads(text)["routes"]], [None] * 4)

    def test_routes_come_back_and_secret_channels_are_reported_as_unresolved(self):
        path = os.path.join(self.dir.name, "alerts.jsonl")
        self.n.add_channel(MemoryChannel("inbox", maxlen=50))
        self.n.add_channel(FileChannel(path), min_severity=Severity.ERROR, topics=["risk"], rate_limit=(5, 60))
        self.n.subscribe("u1", LogChannel("u1-log"), include_system=False)
        self.n.add_channel(SlackChannel("https://hooks.example.test/T/x", transport=refuse), min_severity=Severity.CRITICAL)
        state = through_json(self.n.export_state())
        a = self.fresh()
        report = a.import_state(state)
        self.assertEqual(report["routes_restored"], 3)
        self.assertEqual([u["channel"] for u in report["unresolved"]], ["slack"])     # not dropped silently
        routes = {r.channel.name: r for r in a.routes}
        self.assertEqual(sorted(routes), ["file", "inbox", "u1-log"])
        f = routes["file"]
        self.assertEqual((f.min_severity, f.topics, f.limiter.max_calls, f.limiter.period, f.channel.path), (Severity.ERROR, ("risk",), 5, 60.0, path))
        self.assertEqual((routes["u1-log"].user_id, routes["u1-log"].include_system, routes["inbox"].channel.maxlen), ("u1", False, 50))
        b = self.fresh()
        slack = SlackChannel("https://hooks.example.test/T/x", transport=lambda *a: (200, ""))
        report = b.import_state(state, channels={"slack": slack})                     # the operator supplies the secret-bearing channel again
        self.assertEqual((report["routes_restored"], report["unresolved"]), (4, []))
        self.assertIs({r.channel.name: r for r in b.routes}["slack"].min_severity, Severity.CRITICAL)
        b.import_state(state, channels={"slack": slack})
        self.assertEqual(len(b.routes), 4)                                            # loading twice does not duplicate routes

    def test_open_critical_alerts_survive_and_still_escalate(self):
        box = MemoryChannel("inbox")
        self.n.add_channel(box)
        sent = self.n.notify("Kill switch", "daily loss", Severity.CRITICAL, "risk.kill_switch")
        self.n.notify("just information")
        state = through_json(self.n.export_state())
        after = self.fresh()
        after.import_state(state)
        self.assertEqual([(n.id, n.title, n.body, n.topic) for n in after.unacknowledged()], [(sent.notification.id, "Kill switch", "daily loss", "risk.kill_switch")])
        self.t[0] += 400
        inbox = {r.channel.name: r.channel for r in after.routes}["inbox"]
        self.assertEqual(len(after.escalate_due(after=300)), 1)                       # nobody acknowledged it before the restart either
        self.assertEqual(inbox.items[-1].title, "UNACKNOWLEDGED: Kill switch")
        self.assertTrue(after.acknowledge(sent.notification.id, "risk_manager"))

    def test_undelivered_notifications_survive_and_can_be_replayed(self):
        class Flaky(MemoryChannel):
            fail = True

            def send(self, notification):
                if self.fail:
                    raise ConnectionError("unreachable")
                super().send(notification)

            def config(self):
                return None                                                           # behaves like a channel that holds a secret
        self.n.add_channel(Flaky("pager"))
        self.n.notify("Broker disconnected", "heartbeat lost", Severity.ERROR, "system.broker")
        self.assertEqual(len(self.n.dead_letters), 1)
        state = through_json(self.n.export_state())
        after = self.fresh()
        pager = Flaky("pager")
        pager.fail = False
        report = after.import_state(state, channels={"pager": pager})
        self.assertEqual(report["dead_letters"], 1)
        self.assertEqual(after.replay_dead_letters(), (1, 0))
        self.assertEqual([(n.title, n.body, n.severity) for n in pager.items], [("Broker disconnected", "heartbeat lost", Severity.ERROR)])

    def test_channel_descriptions(self):
        self.assertEqual(channel_from_config(MemoryChannel("m", 7).config()).maxlen, 7)
        self.assertEqual(channel_from_config(LogChannel("l").config()).name, "l")
        self.assertEqual(channel_from_config(FileChannel("/tmp/x.jsonl", "f").config()).path, "/tmp/x.jsonl")
        with self.assertRaises(ConfigError):
            channel_from_config({"type": "slack", "name": "s"})
        n = Notification("t", "b", Severity.WARNING, "risk.limit", "k", "u1", {"a": 1})
        self.assertEqual(Notification.from_dict(through_json(n.to_dict())), n)


class PlatformStateTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "platform.db")

    def boot(self, **kwargs):
        store = SqliteStore(self.path)
        self.addCleanup(store.close)
        p = TradingPlatform(PlatformConfig(log_level="CRITICAL"), store=store, **kwargs)
        p.users.hasher = PasswordHasher(n=2 ** 4)
        admin = p.users.register("admin", helpers.STRONG_PW, Role.ADMIN)
        users = {r: p.users.register(r.value.lower(), helpers.STRONG_PW, r, actor=admin) for r in Role if r is not Role.ADMIN}
        return p, users

    def test_changes_made_through_desks_survive_a_restart(self):
        p, u = self.boot()
        self.assertEqual(p.restore_report["lifecycle"], "nothing stored")
        research = p.desk(u[Role.QUANT], "research")
        research.register_strategy("trend", {"family": "trend"})
        research.propose("trend", good_verdict())
        p.desk(u[Role.PORTFOLIO_MANAGER], "portfolio").approve("trend")
        research.register_strategy("waiting")
        research.propose("waiting", good_verdict())
        research.register_model("signal", "the-model", {"auc": 0.6}, data=np.ones((4, 2)))
        ops = p.desk(u[Role.OPERATIONS], "operations")
        ops.load_instruments([{"symbol": "AAA", "sector": "Tech", "broker:zerodha": "AAA-EQ"}])
        ops.add_channel(FileChannel(os.path.join(self.dir.name, "alerts.jsonl")), Severity.ERROR)
        p.desk(u[Role.RISK_MANAGER], "risk").add_alert(drawdown_alert(0.10))
        p.desk(u[Role.TRADER], "trading").add_alert(price_alert("AAA", 50.0))
        p.store.close()

        q, v = self.boot()                                                            # a new process, nothing saved by hand
        r = q.restore_report
        self.assertEqual((r["lifecycle"], r["models"], r["instruments"]["loaded"], r["alerts"]["restored"]), (2, 1, 1, 2))
        self.assertEqual(r["notifier"]["routes_restored"], 1)
        self.assertIs(q.lifecycle.stage("trend"), Stage.PAPER)
        self.assertEqual(q.instruments.resolve("zerodha", "AAA-EQ").symbol, "AAA")
        self.assertEqual(sorted(q.alerts.states()), ["drawdown", "price_AAA_above_50"])
        self.assertEqual(sorted(x.channel.name for x in q.notifier.routes), ["file", "inbox"])   # the platform's own inbox is not duplicated
        self.assertIsNone(q.models.versions("signal")[0].model)
        q.desk(v[Role.PORTFOLIO_MANAGER], "portfolio").approve("waiting")             # the request pending before the restart can be decided after it
        self.assertIs(q.lifecycle.stage("waiting"), Stage.PAPER)

    def test_one_damaged_part_does_not_stop_the_others(self):
        p, u = self.boot()
        p.desk(u[Role.OPERATIONS], "operations").load_instruments([{"symbol": "AAA"}])
        p.desk(u[Role.RISK_MANAGER], "risk").add_alert(drawdown_alert(0.10))
        p.store.put("platform.lifecycle", [{"strategy_id": "x"}])                     # a record missing required fields
        p.store.close()
        q, _ = self.boot()
        self.assertIn("error", q.restore_report["lifecycle"])
        self.assertEqual(q.restore_report["instruments"]["loaded"], 1)
        self.assertEqual(q.restore_report["alerts"]["restored"], 1)
        self.assertIn("AAA", q.instruments)

    def test_explicit_save_load_and_missing_store(self):
        bare = TradingPlatform(PlatformConfig(log_level="CRITICAL"))
        bare.autosave()                                                               # no store: a no-op, not an error
        with self.assertRaises(ConfigError):
            bare.save_state()
        with self.assertRaises(ConfigError):
            bare.load_state()
        bare.instruments.add(ContractSpec("ZZZ"))
        bare.data.library.put("HIST", "1Day", self.bars(), "alpaca")
        store = SqliteStore(":memory:")
        self.addCleanup(store.close)
        result = bare.save_state(store)
        self.assertEqual((sorted(result["saved"]), result["errors"]),
                         (["alerts", "data", "history", "instruments", "lifecycle", "models", "notifier"], {}))
        self.assertEqual([s["symbol"] for s in store.bar_series()], ["HIST"])         # history is copied to a store it is not attached to
        second = SqliteStore(":memory:")                                              # unchanged parts still reach a new store
        self.addCleanup(second.close)
        bare.save_state(second)
        self.assertEqual(second.bar_series()[0]["rows"], 30)
        other = TradingPlatform(PlatformConfig(log_level="CRITICAL"))
        other.load_state(store)
        self.assertIn("ZZZ", other.instruments)
        self.assertEqual(len(other.data.library.get("HIST")), 30)

    @staticmethod
    def bars():
        idx = pd.bdate_range("2024-01-01", periods=30)
        return pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.5, "volume": 1.0}, index=idx)

    def test_a_part_that_cannot_be_saved_is_reported_and_the_rest_is_written(self):
        p, _ = self.boot()
        p.instruments.add(ContractSpec("AAA"))
        p.alerts.export_state = lambda: (_ for _ in ()).throw(RuntimeError("cannot serialise"))
        result = p.save_state()
        self.assertEqual(list(result["errors"]), ["alerts"])
        self.assertIn("instruments", result["saved"])
        self.assertEqual(p.inbox.items[-1].title, "Platform state not fully saved")
        self.assertEqual(p.store.get("platform.instruments")[0]["symbol"], "AAA")


if __name__ == "__main__":
    unittest.main()
