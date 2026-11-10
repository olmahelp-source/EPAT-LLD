import hashlib
import hmac
import unittest

import helpers  # noqa: F401
from algoplatform.brokers import (AlpacaGateway, BinanceGateway, FakeTransport, ZerodhaGateway, create_gateway)
from algoplatform.core import EventBus, OrderStatus, OrderType, Side, TimeInForce
from algoplatform.core.exceptions import (AuthenticationError, BrokerConnectionError, ConfigError, OrderRejectedError)
from algoplatform.orders import OrderManager

S = OrderStatus
NOSLEEP = lambda s: None
ALPACA = {"api_key": "AK-SECRET-1", "api_secret": "SK-SECRET-2"}


def wire(gw):
    oms = OrderManager(EventBus())
    events = []
    gw.set_event_handler(lambda e: (events.append(e), oms.on_broker_event(e)))
    return oms, events


def send(oms, gw, *a, **kw):
    o = oms.create(*a, **kw)
    oms.mark_sent(o)
    oms.set_broker_id(o, gw.submit_order(o))
    return o


def alpaca(t=None, paper=True):
    t = t or FakeTransport()
    t.on("GET", "/v2/account", (200, {"status": "ACTIVE", "cash": "5000"}))
    gw = AlpacaGateway(ALPACA, paper=paper, transport=t, retry_sleep=NOSLEEP, max_calls_per_sec=10_000)
    return gw, t


class AlpacaTests(unittest.TestCase):
    def test_sandbox_vs_live_url_and_auth_headers(self):
        gw, t = alpaca(paper=True)
        gw.connect()
        self.assertTrue(t.calls[0]["url"].startswith("https://paper-api.alpaca.markets/v2/account"))
        self.assertEqual(t.calls[0]["headers"]["APCA-API-KEY-ID"], "AK-SECRET-1")
        gw2, t2 = alpaca(paper=False)
        gw2.connect()
        self.assertTrue(t2.calls[0]["url"].startswith("https://api.alpaca.markets/"))

    def test_credential_and_account_errors(self):
        with self.assertRaises(ConfigError):
            AlpacaGateway({"api_key": "x"}, transport=FakeTransport())
        t = FakeTransport().on("GET", "/v2/account", (401, {"message": "forbidden"}))
        with self.assertRaises(AuthenticationError):
            AlpacaGateway(ALPACA, transport=t, retry_sleep=NOSLEEP).connect()
        t2 = FakeTransport().on("GET", "/v2/account", (200, {"status": "ACCOUNT_UPDATED"}))
        with self.assertRaises(BrokerConnectionError):
            AlpacaGateway(ALPACA, transport=t2, retry_sleep=NOSLEEP).connect()

    def test_order_payload_ack_and_duplicate_guard(self):
        gw, t = alpaca()
        t.on("POST", "/v2/orders", (200, {"id": "A1", "status": "accepted", "filled_qty": "0"}))
        gw.connect()
        oms, events = wire(gw)
        o = send(oms, gw, "AAPL", Side.BUY, 10, OrderType.LIMIT, limit_price=150.25, tif=TimeInForce.GTC)
        body = t.calls[-1]["body"]
        self.assertEqual((body["symbol"], body["side"], body["qty"], body["type"], body["time_in_force"], body["limit_price"]),
                         ("AAPL", "buy", "10.0", "limit", "gtc", "150.25"))
        self.assertEqual(body["client_order_id"], o.client_order_id)
        self.assertEqual([e.kind for e in events], ["ack"])
        self.assertEqual(o.status, S.ACKNOWLEDGED)
        with self.assertRaises(OrderRejectedError):
            gw.submit_order(o)

    def test_sync_turns_cumulative_status_into_incremental_fills(self):
        gw, t = alpaca()
        t.on("POST", "/v2/orders", (200, {"id": "A1", "status": "accepted"}))
        t.on("GET", "/v2/orders/A1", [(200, {"status": "partially_filled", "filled_qty": "40", "filled_avg_price": "10"}),
                                       (200, {"status": "partially_filled", "filled_qty": "40", "filled_avg_price": "10"}),
                                       (200, {"status": "filled", "filled_qty": "100", "filled_avg_price": "10.6"})])
        gw.connect()
        oms, events = wire(gw)
        o = send(oms, gw, "AAPL", Side.BUY, 100)
        self.assertEqual(gw.sync(), 1)
        self.assertEqual((o.status, o.filled_qty), (S.PARTIALLY_FILLED, 40))
        self.assertEqual(gw.sync(), 0)                                               # same status again: nothing new
        self.assertEqual(gw.sync(), 1)
        self.assertEqual(o.status, S.FILLED)
        self.assertAlmostEqual(o.avg_fill_price, 10.6)                               # 40@10 + 60@11
        self.assertEqual(gw.sync(), 0)                                               # closed orders are not polled
        self.assertEqual(gw.open_order_ids(), set())

    def test_market_order_filled_in_the_submit_response(self):
        gw, t = alpaca()
        t.on("POST", "/v2/orders", (200, {"id": "A9", "status": "filled", "filled_qty": "5", "filled_avg_price": "20"}))
        gw.connect()
        oms, _ = wire(gw)
        o = send(oms, gw, "AAPL", Side.BUY, 5)
        self.assertEqual((o.status, o.avg_fill_price), (S.FILLED, 20.0))

    def test_cancel_confirmed_and_cancel_losing_a_race(self):
        gw, t = alpaca()
        t.on("POST", "/v2/orders", [(200, {"id": "A1", "status": "accepted"}), (200, {"id": "A2", "status": "accepted"})])
        t.on("DELETE", "/v2/orders/", (204, {}))
        t.on("GET", "/v2/orders/A1", (200, {"status": "canceled", "filled_qty": "0"}))
        t.on("GET", "/v2/orders/A2", (200, {"status": "filled", "filled_qty": "10", "filled_avg_price": "5"}))
        gw.connect()
        oms, _ = wire(gw)
        a = send(oms, gw, "AAPL", Side.BUY, 10, OrderType.LIMIT, limit_price=1.0)
        oms.request_cancel(a)
        gw.cancel_order(a)
        self.assertEqual(a.status, S.CANCELLED)
        b = send(oms, gw, "AAPL", Side.BUY, 10, OrderType.LIMIT, limit_price=1.0)
        oms.request_cancel(b)
        gw.cancel_order(b)                                                           # the fill got there first
        self.assertEqual((b.status, b.filled_qty), (S.FILLED, 10))

    def test_retry_policy_by_status_code(self):
        gw, t = alpaca()
        t.on("POST", "/v2/orders", [(429, {"message": "slow down"}), (503, {}), (200, {"id": "A1", "status": "accepted"})])
        gw.connect()
        oms, _ = wire(gw)
        send(oms, gw, "AAPL", Side.BUY, 1)
        self.assertEqual(sum(1 for c in t.calls if c["method"] == "POST"), 3)        # two transient errors then success
        gw2, t2 = alpaca()
        t2.on("POST", "/v2/orders", (422, {"message": "insufficient buying power"}))
        gw2.connect()
        oms2, _ = wire(gw2)
        o = oms2.create("AAPL", Side.BUY, 1)
        oms2.mark_sent(o)
        with self.assertRaises(OrderRejectedError):
            gw2.submit_order(o)
        self.assertEqual(sum(1 for c in t2.calls if c["method"] == "POST"), 1)       # a rejection is never retried
        gw3, t3 = alpaca()
        t3.on("POST", "/v2/orders", (403, {}))
        gw3.connect()
        o3 = OrderManager().create("AAPL", Side.BUY, 1)
        with self.assertRaises(AuthenticationError):
            gw3.submit_order(o3)

    def test_secrets_never_leak_into_logs_repr_or_errors(self):
        gw, t = alpaca()
        t.on("POST", "/v2/orders", [(429, {"message": "slow"}), (200, {"id": "A1", "status": "accepted"})])
        gw.connect()
        oms, _ = wire(gw)
        with self.assertLogs("algoplatform", level="DEBUG") as cm:
            send(oms, gw, "AAPL", Side.BUY, 1)
        blob = "\n".join(cm.output) + repr(gw) + str(gw._secrets)
        for secret in ("AK-SECRET-1", "SK-SECRET-2"):
            self.assertNotIn(secret, blob)
        self.assertFalse(hasattr(gw, "_credentials") and gw._credentials)           # plain-text copy dropped

    def test_positions_cash_and_registry(self):
        gw, t = alpaca()
        t.on("GET", "/v2/positions", (200, [{"symbol": "AAPL", "qty": "10", "side": "long"},
                                             {"symbol": "TSLA", "qty": "3", "side": "short"}]))
        gw.connect()
        self.assertEqual(gw.get_positions(), {"AAPL": 10.0, "TSLA": -3.0})
        self.assertEqual(gw.get_cash(), 5000.0)
        self.assertIsInstance(create_gateway("alpaca", ALPACA, True), AlpacaGateway)
        with self.assertRaises(ConfigError):
            create_gateway("alpaca", {}, True)

    def test_unreachable_sync_is_tolerated(self):
        gw, t = alpaca()
        t.on("POST", "/v2/orders", (200, {"id": "A1", "status": "accepted"}))
        t.on("GET", "/v2/orders/A1", (503, {}))
        gw.connect()
        oms, _ = wire(gw)
        o = send(oms, gw, "AAPL", Side.BUY, 1)
        self.assertEqual(gw.sync(), 0)                                               # broker down: skip, keep tracking
        self.assertEqual(o.status, S.ACKNOWLEDGED)
        self.assertEqual(gw.open_order_ids(), {o.client_order_id})


class ZerodhaTests(unittest.TestCase):
    CREDS = {"api_key": "KITE-KEY", "access_token": "KITE-TOKEN"}

    def gw(self, t):
        t.on("GET", "/user/margins", (200, {"data": {"equity": {"available": {"cash": 25000}}}}))
        g = ZerodhaGateway(self.CREDS, paper=False, transport=t, retry_sleep=NOSLEEP, max_calls_per_sec=10_000)
        g.connect()
        return g

    def test_paper_flag_is_refused(self):
        with self.assertRaises(ConfigError):
            ZerodhaGateway(self.CREDS, paper=True, transport=FakeTransport())
        with self.assertRaises(ConfigError):
            create_gateway("zerodha", self.CREDS, True)

    def test_order_mapping_tag_and_fill(self):
        t = FakeTransport()
        t.on("POST", "/orders/regular", (200, {"status": "success", "data": {"order_id": "2201"}}))
        t.on("GET", "/orders/2201", (200, {"data": [{"status": "OPEN"}, {"status": "COMPLETE", "filled_quantity": 10,
                                                                         "average_price": 101.5}]}))
        g = self.gw(t)
        oms, _ = wire(g)
        o = send(oms, g, "RELIANCE", Side.SELL, 10, OrderType.STOP, stop_price=95.0)
        body = t.calls[-1]["body"]
        self.assertEqual((body["transaction_type"], body["order_type"], body["trigger_price"], body["exchange"]),
                         ("SELL", "SL-M", 95.0, "NSE"))
        self.assertLessEqual(len(body["tag"]), 20)
        self.assertEqual(t.calls[-1]["headers"]["Authorization"], "token KITE-KEY:KITE-TOKEN")
        g.sync()
        self.assertEqual((o.status, o.avg_fill_price), (S.FILLED, 101.5))

    def test_positions_cash_and_unsupported_type(self):
        t = FakeTransport()
        t.on("GET", "/portfolio/positions", (200, {"data": {"net": [{"tradingsymbol": "INFY", "quantity": 5},
                                                                    {"tradingsymbol": "TCS", "quantity": 0}]}}))
        g = self.gw(t)
        self.assertEqual(g.get_positions(), {"INFY": 5.0})
        self.assertEqual(g.get_cash(), 25000.0)
        o = OrderManager().create("INFY", Side.BUY, 1, OrderType.TRAILING_STOP, tags={"trail_pct": 0.05})
        with self.assertRaises(OrderRejectedError):
            g.submit_order(o)


class BinanceTests(unittest.TestCase):
    CREDS = {"api_key": "BN-KEY", "api_secret": "bn-secret"}

    def gw(self, t, paper=True):
        t.on("GET", "/api/v3/account", (200, {"balances": [{"asset": "USDT", "free": "900", "locked": "0"},
                                                          {"asset": "BTC", "free": "0.5", "locked": "0.1"},
                                                          {"asset": "ETH", "free": "0", "locked": "0"}]}))
        g = BinanceGateway(self.CREDS, paper=paper, transport=t, clock=lambda: 1_700_000_000.0, retry_sleep=NOSLEEP, max_calls_per_sec=10_000)
        g.connect()
        return g

    def test_signature_is_hmac_sha256_over_the_query_and_comes_last(self):
        g = BinanceGateway(self.CREDS, transport=FakeTransport(), clock=lambda: 1_700_000_000.0)
        signed = g.sign({"symbol": "BTCUSDT", "side": "BUY"})
        qs = "symbol=BTCUSDT&side=BUY&recvWindow=5000&timestamp=1700000000000"
        expected = hmac.new(b"bn-secret", qs.encode(), hashlib.sha256).hexdigest()
        self.assertEqual(signed, f"{qs}&signature={expected}")

    def test_testnet_vs_live_and_headers(self):
        t = FakeTransport()
        self.gw(t, paper=True)
        self.assertTrue(t.calls[0]["url"].startswith("https://testnet.binance.vision/api/v3/account?"))
        self.assertEqual(t.calls[0]["headers"]["X-MBX-APIKEY"], "BN-KEY")
        self.assertIn("signature=", t.calls[0]["url"])
        self.assertIsNone(t.calls[0]["body"])
        t2 = FakeTransport()
        self.gw(t2, paper=False)
        self.assertTrue(t2.calls[0]["url"].startswith("https://api.binance.com/"))

    def test_order_flow_average_price_from_quote_quantity(self):
        t = FakeTransport()
        t.on("POST", "/api/v3/order", (200, {"orderId": 77, "status": "NEW", "executedQty": "0"}))
        t.on("GET", "/api/v3/order", (200, {"status": "FILLED", "executedQty": "2", "cummulativeQuoteQty": "40100"}))
        g = self.gw(t)
        oms, _ = wire(g)
        o = send(oms, g, "BTCUSDT", Side.BUY, 2, OrderType.LIMIT, limit_price=20050.0, tif=TimeInForce.IOC)
        post = [c for c in t.calls if c["method"] == "POST"][0]
        for part in ("symbol=BTCUSDT", "side=BUY", "type=LIMIT", "timeInForce=IOC", "price=20050.0",
                     f"newClientOrderId={o.client_order_id}"):
            self.assertIn(part, post["url"])
        g.sync()
        self.assertEqual((o.status, o.avg_fill_price), (S.FILLED, 20050.0))

    def test_positions_exclude_quote_asset(self):
        g = self.gw(FakeTransport())
        self.assertEqual(g.get_positions(), {"BTCUSDT": 0.6})
        self.assertEqual(g.get_cash(), 900.0)


if __name__ == "__main__":
    unittest.main()
