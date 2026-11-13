"""The web UI follows the platform's structure: one menu group per job group, a page per job, and each page shown to
exactly the roles the server lets use it. Also the routes added for the groups that had no screen."""
import os
import re
import unittest

import helpers  # noqa: F401
from algoplatform.architecture import GROUPS
from algoplatform.foundation.core import Permission, Role
from algoplatform.access.users import ROLE_PERMISSIONS
from test_web_quant import WebQuantTestCase

STATIC = os.path.join(os.path.dirname(__file__), "..", "src", "algoplatform", "application", "web", "static")
JS = os.path.join(STATIC, "js")


def js_files():
    """Every script of the UI: {relative path: source}."""
    out = {}
    for dirpath, _, names in os.walk(JS):
        for name in names:
            if name.endswith(".js"):
                with open(os.path.join(dirpath, name), encoding="utf-8") as fh:
                    out[os.path.relpath(os.path.join(dirpath, name), JS).replace(os.sep, "/")] = fh.read()
    return out
USERS = {"admin": Role.ADMIN, "quant": Role.QUANT, "trader": Role.TRADER, "portfolio_manager": Role.PORTFOLIO_MANAGER,
         "risk_manager": Role.RISK_MANAGER, "operations": Role.OPERATIONS, "auditor": Role.AUDITOR, "viewer": Role.VIEWER}


def frontend_pages():
    """The PAGES table of the UI (js/shell.js) as dicts: id, group, label, build, perm (list); and all script text."""
    files = js_files()
    src = "\n".join(files.values())
    block = src[src.index("const PAGES = ["):src.index("];", src.index("const PAGES = ["))]
    pages = []
    for m in re.finditer(r'\{ id: "([\w-]+)", group: "(\w+)", label: "([^"]+)", build: (\w+)(?:, perm: \[([^\]]*)\])? \}', block):
        perm = re.findall(r'"([^"]+)"', m.group(5) or "")
        pages.append({"id": m.group(1), "group": m.group(2), "label": m.group(3), "build": m.group(4), "perm": perm})
    return pages, src


SMALL = {"symbols": ["AAA"], "bars": 320}
# one request per page that the page itself makes; used to check that hiding a page matches the server's refusal
PROBES = {
    "instruments": ("GET", "/api/instruments", {}),
    "sentiment": ("POST", "/api/lab/sentiment", {"headlines": "Alpha beats estimates"}),
    "options": ("POST", "/api/lab/option_chain", {"days": [30]}),
    "models": ("POST", "/api/lab/models", {**SMALL, "train_size": 150}),
    "regimes": ("POST", "/api/lab/regimes", SMALL),
    "margin": ("POST", "/api/risk/margin", {"positions": [{"symbol": "AAA", "quantity": 10, "price": 100}]}),
    "currency": ("POST", "/api/risk/currency", {"holdings": [{"currency": "USD", "amount": 1}]}),
    "orderbook": ("POST", "/api/trading/orderbook", {"events": 100}),
    "health": ("GET", "/api/ops/health", {}),
    "providers": ("GET", "/api/data/providers", {}),
    "audit": ("GET", "/api/audit", {}),
    "users": ("GET", "/api/users", {}),
    "lab": ("POST", "/api/lab/technical", SMALL),
    "inbox": ("GET", "/api/inbox", {}),
    "governance": ("GET", "/api/governance", {}),
}


class FrontendFollowsTheStructureTests(unittest.TestCase):
    def setUp(self):
        self.pages, self.src = frontend_pages()

    def test_every_page_belongs_to_a_job_group_and_every_job_group_has_a_page(self):
        self.assertGreaterEqual(len(self.pages), 24)
        keys = {g.key for g in GROUPS}
        self.assertTrue({p["group"] for p in self.pages} <= keys, {p["group"] for p in self.pages} - keys)
        without = keys - {p["group"] for p in self.pages}
        self.assertEqual(without, {"foundation"})                 # shared code (errors, contracts) has nothing to show
        self.assertEqual(len({p["id"] for p in self.pages}), len(self.pages))
        for p in self.pages:
            self.assertRegex(self.src, rf"\nfunction {p['build']}\(", f"page {p['id']} has no {p['build']}()")

    def test_page_permissions_are_real_permissions(self):
        valid = {x.value for x in Permission}
        for p in self.pages:
            self.assertTrue(set(p["perm"]) <= valid, p)
            self.assertTrue(p["perm"] or p["id"] == "home", f"{p['id']} is shown to everyone")


class ScriptsFollowTheStructureTests(unittest.TestCase):
    """The UI is one script per job group, plus a shared core and the shell that loads last."""

    def test_each_page_lives_in_its_groups_script(self):
        files = js_files()
        for p in frontend_pages()[0]:
            where = [f for f, text in files.items() if re.search(rf"\nfunction {p['build']}\(", "\n" + text)]
            self.assertEqual(where, [f"pages/{p['group']}.js"], p["id"])
        groups = {g.key for g in GROUPS} - {"foundation"}
        self.assertEqual({f[len("pages/"):-3] for f in files if f.startswith("pages/")}, groups)
        self.assertTrue(all(text.startswith('"use strict";') for text in files.values()))

    def test_index_loads_core_first_every_page_script_once_and_the_shell_last(self):
        with open(os.path.join(STATIC, "index.html"), encoding="utf-8") as fh:
            scripts = re.findall(r'<script src="/js/([^"]+)" defer></script>', fh.read())
        self.assertEqual(scripts[0], "core.js")
        self.assertEqual(scripts[-1], "shell.js")
        self.assertEqual(sorted(scripts), sorted(js_files()))
        self.assertEqual(len(scripts), len(set(scripts)))


class ApiFollowsTheStructureTests(unittest.TestCase):
    """The API has one route module per job group, and every page calls routes of its own group."""

    ROUTES_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "algoplatform", "application", "web", "routes")

    def registered(self):
        out = {}
        for name in os.listdir(self.ROUTES_DIR):
            if name.endswith(".py") and not name.startswith("__"):
                with open(os.path.join(self.ROUTES_DIR, name), encoding="utf-8") as fh:
                    out[name[:-3]] = set(re.findall(r'@r\("(?:GET|POST)", "([^"]+)"', fh.read()))
        return out

    def test_route_modules_are_the_job_groups(self):
        from algoplatform.application.web.routes import MODULES
        modules = self.registered()
        self.assertEqual(set(modules), {g.key for g in GROUPS} - {"foundation"})
        self.assertEqual(set(MODULES), set(modules))
        self.assertTrue(all(paths for paths in modules.values()), "a group module with no routes")
        everything = [p for paths in modules.values() for p in paths]
        self.assertEqual(len(everything), len(set(everything)), "a route registered by two modules")

    def test_each_page_calls_its_own_groups_routes(self):
        modules = self.registered()
        pages = {p["id"]: p for p in frontend_pages()[0]}
        for page_id, (_method, path, _body) in PROBES.items():
            owner = next(g for g, paths in modules.items() if path in paths)
            self.assertEqual(owner, pages[page_id]["group"], f"page {page_id} ({pages[page_id]['group']}) calls {path} from {owner}")


class MenuMatchesServerTests(WebQuantTestCase):
    def test_a_page_is_shown_exactly_when_the_server_lets_the_role_use_it(self):
        pages = {p["id"]: p for p in frontend_pages()[0]}
        self.assertTrue(set(PROBES) <= set(pages))
        for user, role in USERS.items():
            perms = {x.value for x in ROLE_PERMISSIONS[role]}
            for page_id, (method, path, body) in PROBES.items():
                shown = any(x in perms for x in pages[page_id]["perm"])
                code, obj = self.call(user, method, path, body)
                self.assertEqual(code != 403, shown, f"{user} {page_id}: page {'shown' if shown else 'hidden'} but server said {code} {obj}")
                if shown:
                    self.assertEqual(code, 200, f"{user} {page_id}: {obj}")

    def test_structure_route_names_each_group_by_its_key(self):
        groups = self.ok("viewer", "GET", "/api/structure")["groups"]
        self.assertEqual([g["key"] for g in groups], [g.key for g in GROUPS])


class ResearchRouteTests(WebQuantTestCase):
    def test_option_chain(self):
        r = self.ok("quant", "POST", "/api/lab/option_chain", {"spot": 250, "atm_vol": 0.3, "days": [60, 30], "expiry": 1})
        self.assertEqual([e["days"] for e in r["expiries"]], [30, 60])
        self.assertEqual(r["expiry"], r["expiries"][1]["expiry"])
        self.assertAlmostEqual(r["summary"]["atm_iv"], 0.3, delta=0.03)
        self.assertTrue(r["quality"]["ok"])
        self.assertTrue(all(row["call_delta"] > 0 > row["put_delta"] for row in r["chain"] if row["call_delta"] is not None))
        self.assertLess(r["skew"]["risk_reversal"], 0)                # the default smile prices downside protection higher
        for bad in ({"days": []}, {"days": [30, 30]}, {"days": [1000]}, {"atm_vol": 9}, {"expiry": 5}):
            code, obj = self.call("quant", "POST", "/api/lab/option_chain", bad)
            self.assertEqual(code, 400, (bad, obj))

    def test_sentiment_scores_headlines_and_groups_them_by_instrument(self):
        r = self.ok("quant", "POST", "/api/lab/sentiment", {
            "headlines": "Alpha beats estimates and raises guidance\nBolt downgraded after a weak quarter\nThe weather was mild",
            "aliases": "AAA = Alpha\nBBB = Bolt"})
        by = {x["symbol"]: x for x in r["by_symbol"]}
        self.assertGreater(by["AAA"]["average"], 0)
        self.assertLess(by["BBB"]["average"], 0)
        self.assertIn("downgrade", by["BBB"]["events"])
        self.assertEqual(r["neutral"], 1)
        for bad in ({"headlines": ""}, {"headlines": "x\n" * 101}, {"headlines": "x", "aliases": "not a ticker! = y"}):
            self.assertEqual(self.call("quant", "POST", "/api/lab/sentiment", bad)[0], 400)

    def test_models_are_walk_forward_deterministic_and_need_model_training_rights(self):
        body = {"symbols": ["AAA"], "bars": 500, "model": "tree", "horizon": 5, "train_size": 200}
        a = self.ok("quant", "POST", "/api/lab/models", body)
        b = self.ok("quant", "POST", "/api/lab/models", body)
        self.assertEqual((a["accuracy"], a["auc"], a["equity"]), (b["accuracy"], b["auc"], b["equity"]))
        self.assertLessEqual(a["fits"] + a["failed_fits"], 15)
        self.assertEqual(len(a["probability"]), a["rows"] - 200 - 5)  # nothing predicted inside the first window + embargo
        self.assertEqual(a["predictions"], len(a["probability"]) - 5)  # the last 5 bars' outcomes are not known yet
        self.assertEqual(self.call("quant", "POST", "/api/lab/models", {**body, "model": "oracle"})[0], 400)
        for user in ("trader", "portfolio_manager", "risk_manager", "viewer"):
            self.assertEqual(self.call(user, "POST", "/api/lab/models", body)[0], 403, user)
        self.assertIn("mlp", self.ok("admin", "GET", "/api/lab/models")["models"])

    def test_regimes(self):
        r = self.ok("trader", "POST", "/api/lab/regimes", {"symbols": ["AAA"], "bars": 600, "states": 2})
        self.assertEqual(r["labels"], ["calm", "turbulent"])
        self.assertAlmostEqual(sum(r["current"]["probabilities"].values()), 1.0, places=6)
        self.assertAlmostEqual(sum(r["time_in_regime"].values()), 1.0, places=6)
        vols = [x["ann_vol"] for x in r["describe"]]
        self.assertEqual(vols, sorted(vols))                          # ordered calm -> turbulent
        self.assertEqual(r, self.ok("trader", "POST", "/api/lab/regimes", {"symbols": ["AAA"], "bars": 600, "states": 2}))
        self.assertEqual(self.call("trader", "POST", "/api/lab/regimes", {"symbols": ["AAA"], "bars": 150})[0], 400)


class DeskRouteTests(WebQuantTestCase):
    CSV = "symbol,asset_class,name,currency,tick_size,broker:paper\nAAA,equity,Alpha,USD,0.01,AAA.N\nBAD,spaceship,Bad,USD,0.01,\n"

    def test_instruments_load_list_and_replace(self):
        r = self.ok("operations", "POST", "/api/instruments/load", {"csv": self.CSV})
        self.assertEqual((r["loaded"], len(r["rejected"])), (1, 1))     # one bad row does not stop the good one
        self.assertIn("spaceship", r["rejected"][0]["error"])
        again = self.ok("operations", "POST", "/api/instruments/load", {"csv": self.CSV.replace("Alpha", "Alpha Inc")})
        self.assertEqual(again["loaded"], 0)                           # a duplicate is refused unless replacing is asked for
        self.ok("operations", "POST", "/api/instruments/load", {"csv": self.CSV.replace("Alpha", "Alpha Inc"), "overwrite": True})
        listed = self.ok("viewer", "GET", "/api/instruments")
        self.assertEqual([(x["symbol"], x["name"], x["broker_symbols"]) for x in listed["instruments"]], [("AAA", "Alpha Inc", {"paper": "AAA.N"})])
        self.assertFalse(listed["can_load"])
        self.assertTrue(self.ok("operations", "GET", "/api/instruments")["can_load"])
        for user in ("trader", "quant", "risk_manager", "viewer"):
            self.assertEqual(self.call(user, "POST", "/api/instruments/load", {"csv": self.CSV})[0], 403, user)
        self.assertEqual(self.call("operations", "POST", "/api/instruments/load", {"csv": "name\nx\n"})[0], 400)
        self.assertTrue(any(e["action"] == "instruments.load" for e in self.ok("auditor", "GET", "/api/audit")))

    def test_margin_gives_hedges_relief_only_in_the_scenario_method(self):
        stock = {"symbol": "AAA", "quantity": 100, "price": 100}
        call = {"symbol": "AAA C110", "asset_class": "option", "quantity": -1, "price": 2.0, "multiplier": 100, "underlying": "AAA",
                "underlying_price": 100, "strike": 110, "right": "call", "days": 30, "volatility": 0.25}
        alone = self.ok("risk_manager", "POST", "/api/risk/margin", {"equity": 50000, "positions": [stock]})
        covered = self.ok("risk_manager", "POST", "/api/risk/margin", {"equity": 50000, "positions": [stock, call]})
        self.assertGreater(covered["rules"]["initial"], alone["rules"]["initial"])          # rules: the short call adds margin
        self.assertLess(covered["scenario"]["initial"], alone["scenario"]["initial"])       # scenario: it hedges the stock
        self.assertEqual(len(covered["positions"]), 2)
        code, obj = self.call("risk_manager", "POST", "/api/risk/margin", {"positions": [{"symbol": "X", "quantity": "a", "price": 1},
                                                                                         {"symbol": "Y", "asset_class": "option", "quantity": 1, "price": 1}]})
        self.assertEqual(code, 400)
        self.assertIn("row 1", obj["message"])
        self.assertIn("row 2", obj["message"])                         # every bad row is reported, not just the first

    def test_margin_from_the_paper_session(self):
        code, obj = self.call("trader", "POST", "/api/risk/margin", {"source": "session"})
        self.assertEqual(code, 400)
        self.assertIn("no open paper-trading session", obj["message"])
        self.ok("trader", "POST", "/api/session/open", {"symbols": ["AAA"]})
        self.ok("trader", "POST", "/api/session/order", {"symbol": "AAA", "side": "BUY", "quantity": 20})
        self.ok("trader", "POST", "/api/session/step", {"n": 2})
        r = self.ok("trader", "POST", "/api/risk/margin", {"source": "session"})
        self.assertEqual([(p["symbol"], p["quantity"]) for p in r["positions"]], [("AAA", 20.0)])
        self.assertAlmostEqual(r["rules"]["initial"], 0.5 * r["positions"][0]["value"])

    def test_currency_exposure_hedges_and_attribution(self):
        r = self.ok("portfolio_manager", "POST", "/api/risk/currency", {
            "base": "usd", "holdings": [{"currency": "USD", "amount": 1000}, {"currency": "EUR", "amount": 500}, {"currency": "eur", "amount": 500},
                                        {"currency": "JPY", "amount": 150000}],
            "rates": [{"pair": "EURUSD", "rate": 1.1}, {"pair": "USD/JPY", "rate": 150}], "hedge_ratio": 0.5,
            "attribution": {"value_start": 100, "value_end": 110, "rate_start": 1.1, "rate_end": 1.0}})
        x = r["exposure"]
        self.assertAlmostEqual(x["total"], 1000 + 1000 * 1.1 + 150000 / 150)
        self.assertEqual(r["hedges"], {"EUR": 500.0, "JPY": 75000.0})
        a = r["attribution"]
        self.assertAlmostEqual(a["asset"] + a["currency"] + a["interaction"], a["total"])
        self.assertEqual(self.call("viewer", "POST", "/api/risk/currency", {"holdings": [{"currency": "CHF", "amount": 1}]})[0], 400)   # no rate

    def test_order_book_is_reproducible_and_latency_is_compared_on_the_same_tape(self):
        body = {"side": "BUY", "quantity": 600, "order": "market", "latency": 1.0, "events": 400, "seed": 3}
        a = self.ok("trader", "POST", "/api/trading/orderbook", body)
        self.assertEqual(a, self.ok("trader", "POST", "/api/trading/orderbook", body))
        x = a["result"]
        self.assertLessEqual(x["filled"], 600)
        self.assertAlmostEqual(x["filled"] + x["unfilled"], 600)
        self.assertGreater(x["arrival_mid"], 0)
        no_delay = self.ok("trader", "POST", "/api/trading/orderbook", {**body, "latency": 0})
        self.assertEqual(no_delay["result"], no_delay["no_latency"])
        self.assertEqual(no_delay["result"], a["no_latency"])
        limit = self.ok("trader", "POST", "/api/trading/orderbook", {**body, "order": "limit", "offset_ticks": 5})
        self.assertIsNotNone(limit["result"]["limit_price"])
        for bad in ({"order": "stop"}, {"quantity": 0}, {"latency": 60}, {"events": 10}, {"side": "HOLD"}):
            self.assertEqual(self.call("trader", "POST", "/api/trading/orderbook", {**body, **bad})[0], 400, bad)

    def test_operations_health_jobs_and_replay(self):
        r = self.ok("operations", "GET", "/api/ops/health")
        self.assertEqual(r["health"]["status"], "UP")
        self.assertEqual({i["name"] for i in r["health"]["items"]}, {"state store", "notification delivery", "paper sessions", "market data"})
        self.assertFalse(r["persistent"])
        self.assertEqual(self.ok("operations", "POST", "/api/ops/replay", {}), {"delivered": 0, "still_failing": 0})
        self.assertEqual(self.ok("operations", "POST", "/api/ops/run_jobs", {}), {"ran": []})
        for user in ("trader", "quant", "risk_manager", "auditor", "viewer"):
            self.assertEqual(self.call(user, "GET", "/api/ops/health")[0], 403, user)

    def test_audit_log_follows_the_compliance_desk(self):
        for user in ("admin", "auditor", "risk_manager"):
            self.ok(user, "GET", "/api/audit")
        for user in ("trader", "quant", "operations", "viewer", "portfolio_manager"):
            self.assertEqual(self.call(user, "GET", "/api/audit")[0], 403, user)


class OptimiserMethodTests(WebQuantTestCase):
    def test_every_search_method_runs_from_the_web(self):
        body = {"strategy": "sma_crossover", "n_iter": 8, "symbols": ["AAA"], "bars": 500}      # enough bars for every trial to trade
        for method in ("random", "genetic", "bayesian", "tpe", "pso", "de", "annealing", "cmaes"):
            with self.subTest(method=method):
                r = self.ok("quant", "POST", "/api/optimize", {**body, "method": method})
                self.assertGreater(r["summary"]["trials"], 0)
                self.assertLessEqual(r["summary"]["trials"], 30)
        self.assertEqual(self.call("quant", "POST", "/api/optimize", {**body, "method": "grid"})[0], 400)


if __name__ == "__main__":
    unittest.main()
