import json
import threading
import unittest
import urllib.error
import urllib.request

import helpers  # noqa: F401
from algoplatform.core import PlatformConfig
from algoplatform.web.server import App, create_server

PW = "Correct-Horse-9"


class WebApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = App(PlatformConfig(log_level="CRITICAL"), fast_hash=True)
        cls.httpd = create_server(cls.app, "127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.admin = cls.login("root_admin")                      # first account => admin
        cls.trader = cls.login("trader1")

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    @classmethod
    def call(cls, method, path, body=None, token=None, raw=None):
        req = urllib.request.Request(cls.base + path, method=method,
                                     data=raw if raw is not None else (json.dumps(body).encode() if body is not None else None))
        if token:
            req.add_header("Authorization", "Bearer " + token)
        try:
            with urllib.request.urlopen(req) as r:
                ctype = r.headers.get("Content-Type", "")
                data = r.read()
                return r.status, (json.loads(data) if "json" in ctype else data.decode()), r.headers
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read()), e.headers

    @classmethod
    def login(cls, name):
        cls.call("POST", "/api/register", {"username": name, "password": PW})
        return cls.call("POST", "/api/login", {"username": name, "password": PW})[1]["token"]

    def test_static_ui_and_security_headers(self):
        code, html, hdr = self.call("GET", "/")
        self.assertEqual(code, 200)
        self.assertIn("<div id=\"root\">", html)
        self.assertIn("default-src 'self'", hdr["Content-Security-Policy"])
        self.assertEqual(hdr["X-Content-Type-Options"], "nosniff")
        self.assertEqual(self.call("GET", "/app.js")[0], 200)
        self.assertEqual(self.call("GET", "/../server.py")[0], 404)               # path traversal blocked
        self.assertEqual(self.call("GET", "/%2e%2e/server.py")[0], 404)

    def test_auth_required_and_errors_are_typed(self):
        self.assertEqual(self.call("GET", "/api/me")[0], 401)
        self.assertEqual(self.call("GET", "/api/me", token="garbage")[0], 401)
        code, body, _ = self.call("POST", "/api/login", {"username": "trader1", "password": "wrong-Pass-1x"})
        self.assertEqual(code, 401)
        self.assertEqual(self.call("POST", "/api/login", raw=b"{bad")[0], 400)
        self.assertEqual(self.call("POST", "/api/login", {"username": "x"})[0], 400)  # missing field
        self.assertEqual(self.call("GET", "/api/nope", token=self.trader)[0], 404)

    def test_me_and_strategies(self):
        code, me, _ = self.call("GET", "/api/me", token=self.trader)
        self.assertEqual(me["role"], "TRADER")
        self.assertIn("order:place", me["permissions"])
        code, strat, _ = self.call("GET", "/api/strategies", token=self.trader)
        self.assertIn("sma_crossover", [s["name"] for s in strat])

    def test_backtest_by_name_and_spec_and_validation(self):
        code, r, _ = self.call("POST", "/api/backtest", {"strategy": "sma_crossover", "bars": 300,
                                                          "params": {"fast": 5, "slow": 20}}, self.trader)
        self.assertEqual(code, 200)
        self.assertIn("sharpe", r["metrics"])
        self.assertGreater(len(r["equity"]), 50)
        self.assertLessEqual(len(r["equity"]), 400)                                # downsampled
        json.dumps(r)                                                              # fully serialisable (no NaN)
        spec = {"name": "t", "params": {}, "indicators": {"f": {"type": "sma", "period": 5}, "s": {"type": "sma", "period": 20}},
                "long_entry": [["f", ">", "s"]], "long_exit": [["f", "<=", "s"]]}
        self.assertEqual(self.call("POST", "/api/backtest", {"spec": spec, "bars": 300}, self.trader)[0], 200)
        bad = dict(spec, long_entry=[["nope", ">", "s"]])
        code, err, _ = self.call("POST", "/api/backtest", {"spec": bad, "bars": 300}, self.trader)
        self.assertEqual(code, 400)
        self.assertIn("unknown operand", err["message"])
        self.assertEqual(self.call("POST", "/api/backtest", {"strategy": "sma_crossover", "bars": 10}, self.trader)[0], 400)
        self.assertEqual(self.call("POST", "/api/backtest", {"strategy": "sma_crossover", "symbols": ["A B"]}, self.trader)[0], 400)

    def test_optimize_limits_and_result(self):
        code, r, _ = self.call("POST", "/api/optimize", {"strategy": "sma_crossover", "bars": 300, "n_iter": 6}, self.trader)
        self.assertEqual(code, 200)
        self.assertEqual(r["summary"]["trials"], 6)
        self.assertEqual(self.call("POST", "/api/optimize", {"strategy": "sma_crossover", "n_iter": 500}, self.trader)[0], 400)
        self.assertEqual(self.call("POST", "/api/optimize", {"strategy": "sma_crossover", "method": "grid"}, self.trader)[0], 400)

    def test_live_session_flow_and_kill_switch_permissions(self):
        t = self.login("live_trader")
        self.assertFalse(self.call("GET", "/api/session", token=t)[1]["open"])
        code, snap, _ = self.call("POST", "/api/session/open", {"symbols": ["AAA", "BBB"]}, t)
        self.assertEqual(code, 200)
        self.assertEqual(self.call("POST", "/api/session/open", {"symbols": ["AAA"]}, t)[0], 400)   # already open
        code, r, _ = self.call("POST", "/api/session/order", {"symbol": "AAA", "side": "BUY", "quantity": 10}, t)
        self.assertEqual(r["order"]["status"], "ACKNOWLEDGED")
        code, r, _ = self.call("POST", "/api/session/step", {"n": 2}, t)
        self.assertIn("AAA", r["portfolio"]["positions"])
        code, r, _ = self.call("POST", "/api/session/order", {"symbol": "AAA", "side": "BUY", "quantity": 10**7}, t)
        self.assertEqual(r["order"]["status"], "REJECTED")
        self.assertIn("max_order_quantity", r["order"]["reason"])
        code, r, _ = self.call("POST", "/api/session/killswitch", {"action": "trip", "reason": "ui test"}, t)
        self.assertEqual(r["kill_switch"]["state"], "TRIGGERED")
        code, r, _ = self.call("POST", "/api/session/order", {"symbol": "AAA", "side": "BUY", "quantity": 1}, t)
        self.assertEqual(r["order"]["status"], "REJECTED")
        self.assertEqual(self.call("POST", "/api/session/killswitch", {"action": "reset", "reason": "trader tries it"}, t)[0], 403)
        # admin promotes the trader's colleague? here: promote this trader to risk manager, then reset works
        uid = [u["id"] for u in self.call("GET", "/api/users", token=self.admin)[1] if u["username"] == "live_trader"][0]
        self.assertEqual(self.call("POST", "/api/users/role", {"user_id": uid, "role": "RISK_MANAGER"}, self.admin)[0], 200)
        code, r, _ = self.call("POST", "/api/session/killswitch", {"action": "reset", "reason": "cleared by risk desk"}, t)
        self.assertEqual(r["kill_switch"]["state"], "ARMED")
        self.assertEqual(self.call("POST", "/api/session/close", {}, t)[1], {"open": False})

    def test_rbac_on_admin_endpoints(self):
        self.assertEqual(self.call("GET", "/api/users", token=self.trader)[0], 403)
        self.assertEqual(self.call("GET", "/api/audit", token=self.trader)[0], 403)
        code, rows, _ = self.call("GET", "/api/audit", token=self.admin)
        self.assertEqual(code, 200)
        self.assertGreater(len(rows), 0)

    def test_oversized_body_rejected(self):
        req = urllib.request.Request(self.base + "/api/login", method="POST", data=b"x" * 10)
        req.add_header("Content-Length", "5000000")
        try:
            urllib.request.urlopen(req)
        except Exception as e:                                                     # server answers 413 and may reset the socket
            self.assertTrue(isinstance(e, (urllib.error.HTTPError, ConnectionError, urllib.error.URLError)))


if __name__ == "__main__":
    unittest.main()


class WebPersistenceTests(unittest.TestCase):
    def test_users_and_paper_session_survive_a_server_restart(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            app1 = App(PlatformConfig(log_level="CRITICAL"), fast_hash=True, db_dir=d)
            code, body = app1.dispatch("POST", "/api/register", {"username": "persist_admin", "password": PW}, None)
            self.assertEqual(code, 200)
            tok = app1.dispatch("POST", "/api/login", {"username": "persist_admin", "password": PW}, None)[1]["token"]
            self.assertEqual(app1.dispatch("POST", "/api/session/open", {"symbols": ["AAA", "BBB"], "seed": 5}, tok)[0], 200)
            app1.dispatch("POST", "/api/session/order", {"symbol": "AAA", "side": "BUY", "quantity": 10}, tok)
            app1.dispatch("POST", "/api/session/step", {"n": 3}, tok)
            px = app1.dispatch("GET", "/api/session", {}, tok)[1]["prices"]["BBB"]
            resting = app1.dispatch("POST", "/api/session/order", {"symbol": "BBB", "side": "BUY", "quantity": 5,
                                                                   "type": "LIMIT", "limit_price": round(px * 0.95, 2)}, tok)[1]
            self.assertEqual(resting["order"]["status"], "ACKNOWLEDGED")
            before = app1.dispatch("GET", "/api/session", {}, tok)[1]
            self.assertEqual(before["portfolio"]["positions"]["AAA"]["qty"], 10)
            # --- restart: a brand-new App on the same directory (old tokens are not honoured, users are) ---
            app2 = App(PlatformConfig(log_level="CRITICAL"), fast_hash=True, db_dir=d)
            self.assertEqual(app2.dispatch("GET", "/api/me", {}, tok)[0], 401)
            tok2 = app2.dispatch("POST", "/api/login", {"username": "persist_admin", "password": PW}, None)[1]["token"]
            code, after = app2.dispatch("GET", "/api/session", {}, tok2)
            self.assertEqual(code, 200)
            self.assertTrue(after["open"] and after["resumed"])
            self.assertTrue(after["recovery"]["clean"], after["recovery"])
            self.assertEqual(after["portfolio"]["positions"]["AAA"]["qty"], 10)
            self.assertAlmostEqual(after["portfolio"]["cash"], before["portfolio"]["cash"], places=4)
            self.assertEqual(after["prices"], before["prices"])
            self.assertEqual(len(after["orders"]), len(before["orders"]))
            self.assertIn("ACKNOWLEDGED", [o["status"] for o in after["orders"]])        # the resting limit order is back
            self.assertEqual(app2.dispatch("POST", "/api/session/step", {"n": 2}, tok2)[0], 200)
            # a deliberate close deletes the saved session so it cannot resurrect
            self.assertEqual(app2.dispatch("POST", "/api/session/close", {}, tok2)[1], {"open": False})
            app3 = App(PlatformConfig(log_level="CRITICAL"), fast_hash=True, db_dir=d)
            tok3 = app3.dispatch("POST", "/api/login", {"username": "persist_admin", "password": PW}, None)[1]["token"]
            self.assertFalse(app3.dispatch("GET", "/api/session", {}, tok3)[1]["open"])
            for a in (app1, app2, app3):
                if a._meta:
                    a._meta.close()
