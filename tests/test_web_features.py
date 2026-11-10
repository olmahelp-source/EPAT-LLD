import http.client
import json
import threading
import time
import unittest
import urllib.error
import urllib.request

import pandas as pd

import helpers  # noqa: F401
from algoplatform.core import PlatformConfig
from algoplatform.data import synthetic
from algoplatform.web.server import App, create_server

PW = "Correct-Horse-9"


def csv_from(df: pd.DataFrame) -> str:
    return df.rename_axis("date").reset_index().to_csv(index=False)


class WebFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = App(PlatformConfig(log_level="CRITICAL"), fast_hash=True)
        cls.httpd = create_server(cls.app, "127.0.0.1", 0)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.admin = cls.login("feat_admin")
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.app.shutting_down = True
        cls.httpd.shutdown()
        cls.httpd.server_close()

    @classmethod
    def call(cls, method, path, body=None, token=None):
        req = urllib.request.Request(f"http://127.0.0.1:{cls.port}{path}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None)
        if token:
            req.add_header("Authorization", "Bearer " + token)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    @classmethod
    def login(cls, name):
        cls.call("POST", "/api/register", {"username": name, "password": PW})
        return cls.call("POST", "/api/login", {"username": name, "password": PW})[1]["token"]

    def user(self):
        WebFeatureTests.n += 1
        return self.login(f"feat_user_{WebFeatureTests.n}")

    # ---- uploads ---------------------------------------------------------------------------------------
    def test_upload_list_backtest_and_clear(self):
        t = self.user()
        df = synthetic.gbm(400, seed=3)
        code, r = self.call("POST", "/api/data/upload", {"symbol": "mine", "csv": csv_from(df)}, t)
        self.assertEqual((code, r["symbol"], r["rows"]), (200, "MINE", 400))
        self.assertEqual([d["symbol"] for d in self.call("GET", "/api/data", token=t)[1]], ["MINE"])
        code, bt = self.call("POST", "/api/backtest", {"strategy": "sma_crossover", "dataset": "uploaded"}, t)
        self.assertEqual(code, 200)
        self.assertEqual(bt["symbols"], ["MINE"])
        other = self.user()                                                               # uploads are private per user
        self.assertEqual(self.call("GET", "/api/data", token=other)[1], [])
        self.assertEqual(self.call("POST", "/api/backtest", {"strategy": "sma_crossover", "dataset": "uploaded"}, other)[0], 400)
        self.call("POST", "/api/data/clear", {}, t)
        self.assertEqual(self.call("GET", "/api/data", token=t)[1], [])

    def test_bad_uploads_are_rejected_with_a_reason(self):
        t = self.user()
        good = csv_from(synthetic.gbm(200, seed=1))
        cases = {
            "no date column": good.replace("date,", "when,", 1),
            "bad dates": good.replace("2020-01-01", "not-a-date", 1),
            "empty": "   ",
            "missing close": good.replace("close", "closing"),
        }
        for label, text in cases.items():
            code, r = self.call("POST", "/api/data/upload", {"symbol": "BAD", "csv": text}, t)
            self.assertEqual(code, 400, label)
            self.assertTrue(r["message"], label)
        df = synthetic.gbm(200, seed=1)
        df.iloc[5, df.columns.get_loc("high")] = df.iloc[5]["low"] * 0.5                    # impossible bar
        code, r = self.call("POST", "/api/data/upload", {"symbol": "BAD", "csv": csv_from(df)}, t)
        self.assertEqual(code, 400)
        self.assertIn("OHLC", r["message"])
        self.assertEqual(self.call("POST", "/api/data/upload", {"symbol": "bad name!", "csv": good}, t)[0], 400)
        for i in range(10):
            self.assertEqual(self.call("POST", "/api/data/upload", {"symbol": f"S{i}", "csv": good}, t)[0], 200)
        self.assertEqual(self.call("POST", "/api/data/upload", {"symbol": "S11", "csv": good}, t)[0], 400)     # per-user cap
        short = csv_from(synthetic.gbm(100, seed=1))
        t2 = self.user()
        self.call("POST", "/api/data/upload", {"symbol": "SHORT", "csv": short}, t2)
        self.assertEqual(self.call("POST", "/api/backtest", {"strategy": "sma_crossover", "dataset": "uploaded"}, t2)[0], 400)

    # ---- strategies and research endpoints -----------------------------------------------------------------
    def test_strategy_list_flags_and_guards(self):
        t = self.user()
        by_name = {s["name"]: s for s in self.call("GET", "/api/strategies", token=t)[1]}
        self.assertTrue(by_name["ml_logistic"]["runnable"] and by_name["ml_logistic"]["slow"])
        self.assertFalse(by_name["factor_ls"]["runnable"])
        self.assertEqual(self.call("POST", "/api/backtest", {"strategy": "factor_ls", "bars": 300}, t)[0], 400)
        self.assertEqual(self.call("POST", "/api/optimize", {"strategy": "ml_logistic", "n_iter": 30}, t)[0], 400)

    def test_backtest_returns_var_and_es(self):
        t = self.user()
        code, r = self.call("POST", "/api/backtest", {"strategy": "sma_crossover", "bars": 400}, t)
        self.assertEqual(code, 200)
        rk = r["risk"]
        self.assertTrue(0 <= rk["var_95"] <= rk["var_99"])
        self.assertGreaterEqual(rk["es_99"], rk["var_99"] - 1e-9)                          # the tail mean is at least the VaR

    def test_ml_strategy_runs_from_the_api(self):
        t = self.user()
        code, r = self.call("POST", "/api/backtest", {"strategy": "ml_logistic", "bars": 600}, t)
        self.assertEqual(code, 200)
        self.assertIn("sharpe", r["metrics"])

    def test_walkforward(self):
        t = self.user()
        code, r = self.call("POST", "/api/walkforward", {"strategy": "sma_crossover", "bars": 700, "train_bars": 250,
                                                         "test_bars": 100, "n_iter": 4}, t)
        self.assertEqual(code, 200, r)
        self.assertGreaterEqual(len(r["folds"]), 2)
        self.assertLess(r["folds"][0]["train_end"], r["folds"][0]["test_start"])
        self.assertTrue(r["oos_equity"])
        self.assertIn("fast", r["stability"])
        self.assertEqual(self.call("POST", "/api/walkforward", {"strategy": "sma_crossover", "bars": 5000}, t)[0], 400)
        self.assertEqual(self.call("POST", "/api/walkforward", {"strategy": "sma_crossover", "n_iter": 99}, t)[0], 400)

    def test_portfolio_of_strategies(self):
        t = self.user()
        body = {"bars": 500, "strategies": [{"name": "sma_crossover", "pct": 50, "params": {"fast": 8, "slow": 30}},
                                              {"name": "zscore_reversion", "pct": 30}]}
        code, r = self.call("POST", "/api/portfolio", body, t)
        self.assertEqual(code, 200, r)
        self.assertEqual(set(r["standalone"]), {"sma_crossover#1", "zscore_reversion#2"})
        self.assertEqual(set(r["suggested_weights"]), {"equal", "inverse_vol", "risk_parity", "min_variance"})
        for w in r["suggested_weights"].values():
            self.assertAlmostEqual(sum(w.values()), 1.0, places=3)
        self.assertAlmostEqual(r["correlation"]["sma_crossover#1"]["sma_crossover#1"], 1.0)
        self.assertTrue(all("strategy_id" in c for c in r["contributions"]))
        self.assertIsNotNone(r["risk"])
        bad = {"bars": 500, "strategies": [{"name": "sma_crossover", "pct": 80}, {"name": "zscore_reversion", "pct": 40}]}
        self.assertEqual(self.call("POST", "/api/portfolio", bad, t)[0], 400)                # over-allocated
        six = {"bars": 500, "strategies": [{"name": "sma_crossover", "pct": 10}] * 6}
        self.assertEqual(self.call("POST", "/api/portfolio", six, t)[0], 400)
        self.assertEqual(self.call("POST", "/api/portfolio", {"strategies": []}, t)[0], 400)

    # ---- live session --------------------------------------------------------------------------------------
    def open(self, t, **kw):
        code, snap = self.call("POST", "/api/session/open", {"symbols": ["AAA", "BBB"], "seed": 2, **kw}, t)
        self.assertEqual(code, 200)
        return snap

    def test_algo_order_works_through_children_and_reports_tca(self):
        t = self.user()
        self.open(t)
        code, r = self.call("POST", "/api/session/order", {"symbol": "AAA", "side": "BUY", "quantity": 40,
                                                           "algo": {"name": "twap", "bars": 4}}, t)
        self.assertEqual(code, 200, r)
        self.assertEqual(r["parent"]["algo"], "twap")
        code, snap = self.call("POST", "/api/session/step", {"n": 8}, t)
        parent = snap["parents"][0]
        self.assertEqual((parent["status"], parent["filled"]), ("DONE", 40))
        self.assertEqual(snap["portfolio"]["positions"]["AAA"]["qty"], 40)
        self.assertGreaterEqual(parent["children"], 4)
        self.assertEqual(snap["tca"]["parents"], 1)
        self.assertIn("twap", snap["tca"]["by_algo"])
        self.assertTrue(all(o["strategy"].startswith("manual:") for o in snap["orders"]))

    def test_algo_validation_and_kill_switch(self):
        t = self.user()
        self.open(t)
        base = {"symbol": "AAA", "side": "BUY", "quantity": 10}
        for algo in ({"name": "nope"}, {"name": "twap", "bars": 1}, {"name": "pov", "participation": 0.9},
                     {"name": "iceberg", "display": -1}):
            self.assertEqual(self.call("POST", "/api/session/order", {**base, "algo": algo}, t)[0], 400, algo)
        self.assertEqual(self.call("POST", "/api/session/order", {**base, "symbol": "ZZZ", "algo": {"name": "twap", "bars": 3}}, t)[0], 400)
        self.call("POST", "/api/session/killswitch", {"action": "trip", "reason": "halt"}, t)
        code, r = self.call("POST", "/api/session/order", {**base, "algo": {"name": "twap", "bars": 3}}, t)
        self.assertEqual(code, 409)                                                         # halted: no new parent orders

    def test_cancel_a_single_order(self):
        t = self.user()
        snap = self.open(t)
        px = snap["prices"]["AAA"]
        r = self.call("POST", "/api/session/order", {"symbol": "AAA", "side": "BUY", "quantity": 5, "type": "LIMIT",
                                                     "limit_price": round(px * 0.95, 2)}, t)[1]
        oid = r["order"]["id"]
        self.assertEqual(r["order"]["status"], "ACKNOWLEDGED")
        code, r2 = self.call("POST", "/api/session/cancel", {"order_id": oid}, t)
        self.assertEqual(code, 200)
        self.assertEqual([o["status"] for o in r2["orders"] if o["id"] == oid], ["CANCELLED"])
        self.assertEqual(self.call("POST", "/api/session/cancel", {"order_id": "ORD-does-not-exist"}, t)[0], 409)

    def test_live_risk_measures(self):
        t = self.user()
        self.open(t)
        self.call("POST", "/api/session/order", {"symbol": "AAA", "side": "BUY", "quantity": 50}, t)
        snap = self.call("POST", "/api/session/step", {"n": 45}, t)[1]
        m = snap["measures"]
        self.assertIsNotNone(m["var"])
        self.assertTrue(m["var"]["var_99"] >= m["var"]["var_95"] >= 0)
        self.assertEqual({s["scenario"] for s in m["stress"]}, {"equity_crash_-20%", "black_monday_-22%", "flash_crash_-9%", "rally_+10%"})
        crash = [s for s in m["stress"] if s["scenario"] == "equity_crash_-20%"][0]
        self.assertAlmostEqual(crash["pnl"], -0.2 * snap["portfolio"]["positions"]["AAA"]["qty"] * snap["portfolio"]["positions"]["AAA"]["last"], places=2)
        self.assertGreater(m["reverse_stress_move"], 0)

    # ---- streaming -----------------------------------------------------------------------------------------
    def read_event(self, resp, timeout=5.0):
        """Parse one Server-Sent Event (skipping keep-alive comments)."""
        end, name, data = time.time() + timeout, None, None
        while time.time() < end:
            line = resp.fp.readline().decode().rstrip("\n")
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:])
            elif line == "" and name:
                return name, data
        raise AssertionError("no event received")

    def test_event_stream_pushes_snapshots_and_ends_when_the_session_closes(self):
        t = self.user()
        self.open(t)
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=6)
        conn.request("GET", "/api/session/stream", headers={"Authorization": "Bearer " + t})
        resp = conn.getresponse()
        self.assertEqual(resp.status, 200)
        self.assertIn("text/event-stream", resp.getheader("Content-Type"))
        name, first = self.read_event(resp)
        self.assertEqual((name, first["open"]), ("snapshot", True))
        bars0 = first["bars"]
        self.call("POST", "/api/session/step", {"n": 3}, t)
        deadline, latest = time.time() + 5, first
        while latest["bars"] <= bars0 and time.time() < deadline:
            name, latest = self.read_event(resp)
        self.assertEqual(latest["bars"], bars0 + 3)                                      # pushed without polling
        self.call("POST", "/api/session/close", {}, t)
        events = []
        deadline = time.time() + 5
        while time.time() < deadline and "closed" not in events:
            events.append(self.read_event(resp)[0])
        self.assertIn("closed", events)
        conn.close()

    def test_event_stream_requires_auth_and_a_session(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/api/session/stream")
        self.assertEqual(conn.getresponse().status, 401)
        t = self.user()
        conn2 = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn2.request("GET", "/api/session/stream", headers={"Authorization": "Bearer " + t})
        self.assertEqual(conn2.getresponse().status, 404)


if __name__ == "__main__":
    unittest.main()
