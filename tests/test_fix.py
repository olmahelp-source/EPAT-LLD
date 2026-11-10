import os
import tempfile
import time
import unittest

import helpers  # noqa: F401
from algoplatform.brokers import create_gateway
from algoplatform.brokers.fix import FileSeqStore, FixGateway, FixParser, FixProtocolError, FixSession, build, decode, encode
from algoplatform.brokers.fix.message import (EXEC_REPORT, HEARTBEAT, SOH, T_TEXT, checksum, utc_timestamp)
from algoplatform.brokers.fix.session import ACTIVE, DISCONNECTED
from algoplatform.brokers.fix.testing import FixAcceptor, LoopbackChannel, TcpAcceptorServer
from algoplatform.core import EventBus, OrderStatus, OrderType, Side
from algoplatform.core.exceptions import BrokerConnectionError, ConfigError, OrderRejectedError
from algoplatform.orders import OrderManager

S = OrderStatus


class MessageTests(unittest.TestCase):
    def test_encode_decode_roundtrip_and_trailer(self):
        raw = build("D", "CLIENT", "EXCH", 7, "20240102-10:00:00.000", [(11, "ORD1"), (55, "AAPL"), (38, "100")])
        self.assertTrue(raw.startswith(b"8=FIX.4.4\x019="))
        m = decode(raw)
        self.assertEqual((m.type, m.seq, m.get(11), m.get(55), m.get(38)), ("D", 7, "ORD1", "AAPL", "100"))
        tail = raw[-7:]
        self.assertEqual(tail[:3], b"10=")
        self.assertEqual(tail[3:6].decode(), f"{sum(raw[:-7]) % 256:03d}")           # checksum recomputed independently

    def test_corruption_is_detected(self):
        raw = build("D", "C", "E", 1, "t", [(55, "AAPL")])
        bad = bytearray(raw)
        bad[raw.index(b"AAPL")] = ord("B")
        with self.assertRaises(FixProtocolError):
            decode(bytes(bad))                                                        # checksum mismatch
        # correct checksum but a lying body length
        body = f"35=0{SOH}49=C{SOH}56=E{SOH}34=1{SOH}52=t{SOH}"
        head = f"8=FIX.4.4{SOH}9={len(body) + 5}{SOH}".encode()
        forged = head + body.encode()
        forged += f"10={checksum(forged)}{SOH}".encode()
        with self.assertRaisesRegex(FixProtocolError, "body length"):
            decode(forged)
        with self.assertRaises(FixProtocolError):
            decode(b"garbage")
        with self.assertRaises(FixProtocolError):
            encode([(35, "D"), (58, "bad\x01value")])

    def test_parser_handles_coalesced_split_and_noisy_streams(self):
        a = build("0", "C", "E", 1, "t", [])
        b = build("D", "C", "E", 2, "t", [(55, "X")])
        p = FixParser()
        self.assertEqual([m.seq for m in p.feed(a + b)], [1, 2])                       # two messages in one read
        p2, out = FixParser(), []
        for byte in a + b:                                                            # one byte at a time
            out += p2.feed(bytes([byte]))
        self.assertEqual([m.seq for m in out], [1, 2])
        self.assertEqual([m.seq for m in FixParser().feed(b"noise" + a)], [1])        # junk before a message is skipped
        with self.assertRaises(FixProtocolError):
            FixParser().feed(a[:-4] + b"999" + a[-1:])                                # bad checksum surfaces

    def test_timestamp_format(self):
        self.assertRegex(utc_timestamp(0), r"^19700101-00:00:00\.000$")


def make_pair(clock=None, heartbeat=30):
    clk = clock or time.time
    apps = []
    client = FixSession("CLIENT", "EXCH", heartbeat=heartbeat, clock=clk, on_app=lambda m: apps.append(m.get(T_TEXT)))
    acc = FixAcceptor("EXCH", "CLIENT", heartbeat=heartbeat, clock=clk)
    acc.link(client, LoopbackChannel())
    return client, acc, apps


def news(acc, text):
    return acc.session.send_app("B", [(T_TEXT, text)])


class SessionTests(unittest.TestCase):
    def test_logon_handshake_and_sequence_numbers(self):
        client, acc, _ = make_pair()
        client.logon()
        self.assertEqual((client.state, acc.session.state), (ACTIVE, ACTIVE))
        self.assertEqual((client.store.next_send, client.store.next_expected), (2, 2))
        self.assertEqual((acc.session.store.next_send, acc.session.store.next_expected), (2, 2))

    def test_app_messages_delivered_in_order_once(self):
        client, acc, apps = make_pair()
        client.logon()
        for t in ("a", "b", "c"):
            news(acc, t)
        self.assertEqual(apps, ["a", "b", "c"])

    def test_gap_triggers_resend_and_nothing_is_lost_or_duplicated(self):
        client, acc, apps = make_pair()
        client.logon()
        acc.channel.drop = 2                                                          # messages 2 and 3 are lost in transit
        news(acc, "one")
        news(acc, "two")
        news(acc, "three")                                                            # arrives: client sees a gap
        self.assertEqual(apps, ["one", "two", "three"])
        self.assertEqual(client.store.next_expected, 5)
        self.assertEqual(client.state, ACTIVE)

    def test_admin_messages_in_a_gap_are_gap_filled_not_resent(self):
        client, acc, apps = make_pair()
        client.logon()
        acc.channel.drop = 1
        acc.session._send(HEARTBEAT, [])                                              # seq 2: admin, lost
        news(acc, "x")                                                                # seq 3
        self.assertEqual(apps, ["x"])
        self.assertEqual(client.store.next_expected, 4)
        resent = [decode(w) for w in acc.channel.written if decode(w).get(43) == "Y"]
        self.assertEqual(resent[0].type, "4")                                         # SequenceReset gap fill for the heartbeat
        self.assertEqual(resent[0].get(123), "Y")

    def test_duplicates_ignored_and_too_low_sequence_logs_out(self):
        client, acc, apps = make_pair()
        client.logon()
        ts = utc_timestamp()
        m2 = build("B", "EXCH", "CLIENT", 2, ts, [(T_TEXT, "first")])
        client.on_bytes(m2)
        dup = build("B", "EXCH", "CLIENT", 2, ts, [(T_TEXT, "first")], [(43, "Y"), (122, ts)])
        client.on_bytes(dup)
        self.assertEqual(apps, ["first"])
        self.assertEqual(client.state, ACTIVE)
        stale = build("B", "EXCH", "CLIENT", 1, ts, [(T_TEXT, "replayed")])              # not flagged PossDup
        client.on_bytes(stale)
        self.assertEqual(client.state, DISCONNECTED)
        self.assertEqual(acc.session.state, DISCONNECTED)                               # logout was delivered to the peer
        self.assertEqual(apps, ["first"])

    def test_corrupt_bytes_do_not_break_the_session(self):
        client, acc, apps = make_pair()
        client.logon()
        good = build("B", "EXCH", "CLIENT", 2, utc_timestamp(), [(T_TEXT, "ok")])
        client.on_bytes(good[:-4] + b"000" + good[-1:])
        self.assertEqual(client.protocol_errors, 1)
        news(acc, "after")                                                            # seq 3 vs expected 2: gap -> resend -> recovered
        self.assertIn("after", apps)

    def test_heartbeat_test_request_and_dead_link(self):
        clk = [1000.0]
        client, acc, apps = make_pair(clock=lambda: clk[0], heartbeat=30)
        client.logon()
        sent = client.store.next_send
        clk[0] = 1031
        client.tick()
        self.assertEqual(client.store.next_send, sent + 1)                              # heartbeat after 30s of silence on send
        clk[0] = 1050                                                                 # > 1.5 x heartbeat without hearing anything
        client.tick()
        self.assertEqual(client.state, ACTIVE)                                         # peer answered the TestRequest
        acc.channel.drop = 10**6                                                      # peer goes silent
        clk[0] = 1200
        client.tick()                                                                 # test request unanswered, then dead
        clk[0] = 1300
        client.tick()
        self.assertEqual(client.state, DISCONNECTED)

    def test_file_store_persists_sequences_and_sent_log(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "seq.json")
            store = FileSeqStore(path)
            client = FixSession("CLIENT", "EXCH", store=store)
            acc = FixAcceptor("EXCH", "CLIENT")
            acc.link(client, LoopbackChannel())
            client.logon()
            news(acc, "n1")
            reloaded = FileSeqStore(path)
            self.assertEqual((reloaded.next_send, reloaded.next_expected), (store.next_send, store.next_expected))
            self.assertEqual(reloaded.sent[1][0], "A")
            self.assertEqual(reloaded.next_expected, 3)

    def test_reset_on_logon_restarts_both_sides(self):
        client, acc, _ = make_pair()
        client.logon()
        news(acc, "x")
        client.logon(reset=True)
        self.assertEqual((client.store.next_send, acc.session.store.next_send), (2, 2))
        self.assertEqual(client.store.next_expected, 2)

    def test_orderly_logout(self):
        client, acc, _ = make_pair()
        client.logon()
        client.logout("bye")
        self.assertEqual((client.state, acc.session.state), (DISCONNECTED, DISCONNECTED))


def gw_pair(**kw):
    ch = LoopbackChannel()
    gw = FixGateway(sender="CLIENT", target="EXCH", channel=ch, **kw)
    acc = FixAcceptor("EXCH", "CLIENT")
    acc.link(gw.session, ch)
    oms = OrderManager(EventBus())
    gw.set_event_handler(oms.on_broker_event)
    gw.connect()
    return gw, acc, oms, ch


def go(oms, gw, *a, **kw):
    o = oms.create(*a, **kw)
    oms.mark_sent(o)
    oms.set_broker_id(o, gw.submit_order(o))
    return o


class GatewayTests(unittest.TestCase):
    def test_connect_disconnect(self):
        gw, acc, oms, _ = gw_pair()
        self.assertTrue(gw.is_connected)
        gw.disconnect()
        self.assertFalse(gw.is_connected)
        self.assertEqual(acc.session.state, DISCONNECTED)

    def test_full_order_lifecycle_through_the_oms(self):
        gw, acc, oms, ch = gw_pair(initial_cash=10_000)
        o = go(oms, gw, "AAPL", Side.BUY, 100, OrderType.LIMIT, limit_price=10.5)
        self.assertEqual(o.status, S.ACKNOWLEDGED)
        self.assertEqual(o.broker_order_id, o.client_order_id)                          # placeholder until the ER names it
        acc.fill(o.client_order_id, 40, 10.0)
        self.assertEqual((o.status, o.filled_qty), (S.PARTIALLY_FILLED, 40))
        acc.fill(o.client_order_id, 60, 11.0)
        self.assertEqual(o.status, S.FILLED)
        self.assertAlmostEqual(o.avg_fill_price, 10.6)
        self.assertEqual(gw.get_positions(), {"AAPL": 100})
        self.assertAlmostEqual(gw.get_cash(), 10_000 - 1060)
        self.assertEqual(gw.open_order_ids(), set())
        sent = decode([w for w in ch.written if decode(w).type == "D"][0])
        self.assertEqual((sent.get(54), sent.get(40), sent.get(44), sent.get(59), sent.get(38)), ("1", "2", "10.5", "0", "100"))

    def test_reject_and_unsupported_type_and_not_connected(self):
        gw, acc, oms, _ = gw_pair()
        acc.reject_all = "no margin"
        o = go(oms, gw, "AAPL", Side.BUY, 1)
        self.assertEqual(o.status, S.REJECTED)
        self.assertIn("no margin", o.reject_reason)
        with self.assertRaises(OrderRejectedError):
            gw.submit_order(oms.create("AAPL", Side.BUY, 1, OrderType.TRAILING_STOP, tags={"trail_pct": 0.05}))
        gw.disconnect()
        with self.assertRaises(BrokerConnectionError):
            gw.submit_order(oms.create("AAPL", Side.BUY, 1))

    def test_cancel_and_cancel_reject(self):
        gw, acc, oms, _ = gw_pair()
        o = go(oms, gw, "AAPL", Side.BUY, 10, OrderType.LIMIT, limit_price=1.0)
        oms.request_cancel(o)
        gw.cancel_order(o)
        self.assertEqual(o.status, S.CANCELLED)
        self.assertEqual(gw.open_order_ids(), set())
        p = go(oms, gw, "AAPL", Side.BUY, 10, OrderType.LIMIT, limit_price=1.0)
        acc.orders[p.client_order_id]["open"] = False                                   # exchange already filled/closed it
        oms.request_cancel(p)
        self.assertEqual(p.status, S.PENDING_CANCEL)
        gw.cancel_order(p)                                                              # -> OrderCancelReject
        self.assertEqual(p.status, S.ACKNOWLEDGED)                                      # back to live, not stuck

    def test_duplicate_execution_report_does_not_double_fill(self):
        gw, acc, oms, _ = gw_pair()
        o = go(oms, gw, "AAPL", Side.BUY, 100)
        acc.fill(o.client_order_id, 30, 10.0)
        last_seq = acc.session.store.next_send - 1
        body = acc.session.store.sent[last_seq][1]
        acc.session.send_app(EXEC_REPORT, body)                                         # same ExecID, new sequence number
        self.assertEqual(o.filled_qty, 30)
        self.assertEqual(oms.anomalies[-1]["type"], "duplicate_fill")

    def test_lost_execution_report_is_recovered_by_resend(self):
        gw, acc, oms, _ = gw_pair()
        a = go(oms, gw, "AAPL", Side.BUY, 100)
        acc.channel.drop = 1
        acc.fill(a.client_order_id, 100, 10.0)                                          # this report is lost
        self.assertEqual(a.status, S.ACKNOWLEDGED)
        b = go(oms, gw, "MSFT", Side.BUY, 5)                                            # next report reveals the gap
        self.assertEqual(a.status, S.FILLED)
        self.assertEqual(a.filled_qty, 100)
        self.assertEqual(b.status, S.ACKNOWLEDGED)
        self.assertEqual(gw.get_positions(), {"AAPL": 100})

    def test_unacknowledged_logon_times_out(self):
        gw = FixGateway(sender="C", target="E", channel=LoopbackChannel(), logon_timeout=0.05)
        with self.assertRaises(BrokerConnectionError):
            gw.connect()

    def test_registry_and_configuration_errors(self):
        gw = create_gateway("fix", {"sender_comp_id": "A", "target_comp_id": "B"}, True)
        self.assertIsInstance(gw, FixGateway)
        with self.assertRaises(ConfigError):
            gw.connect()                                                                # no channel and no host/port
        with self.assertRaises(ConfigError):
            FixGateway()

    def test_over_a_real_tcp_socket(self):
        server = TcpAcceptorServer().start()
        try:
            gw = FixGateway(credentials={"sender_comp_id": "CLIENT", "target_comp_id": "EXCH", "host": "127.0.0.1",
                                         "port": server.port}, logon_timeout=3)
            oms = OrderManager(EventBus())
            gw.set_event_handler(oms.on_broker_event)
            gw.connect()
            o = go(oms, gw, "AAPL", Side.BUY, 10)
            self._wait(lambda: o.status is S.ACKNOWLEDGED)
            server.acceptor.fill(o.client_order_id, px=12.5)
            self._wait(lambda: o.status is S.FILLED)
            self.assertEqual(o.avg_fill_price, 12.5)
            gw.disconnect()
        finally:
            server.stop()

    @staticmethod
    def _wait(cond, timeout=3.0):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return
            time.sleep(0.01)
        raise AssertionError("condition not reached in time")

    def test_unreachable_endpoint(self):
        gw = FixGateway(credentials={"sender_comp_id": "C", "target_comp_id": "E", "host": "127.0.0.1", "port": 1},
                        logon_timeout=0.2)
        with self.assertRaises(BrokerConnectionError):
            gw.connect()


if __name__ == "__main__":
    unittest.main()
