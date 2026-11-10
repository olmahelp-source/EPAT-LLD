import unittest

import helpers  # noqa: F401  (sets sys.path)
from algoplatform.core import (AuditLog, ConfigError, EventBus, Instrument, Order, PlatformConfig, Registry,
                               Signal, ValidationError, Side, OrderType, Bar)
from algoplatform.core.exceptions import BrokerConnectionError, OrderRejectedError, PlatformError
from algoplatform.utils import CircuitBreaker, IdGenerator, RateLimiter, retry, safe_div, to_array, SimClock, TradingCalendar
from datetime import datetime, timezone


class ConfigTests(unittest.TestCase):
    def test_defaults_valid(self):
        self.assertEqual(PlatformConfig().environment, "paper")

    def test_rejects_bad_values(self):
        with self.assertRaises(ConfigError):
            PlatformConfig(initial_capital=-1)
        with self.assertRaises(ConfigError):
            PlatformConfig(environment="prod")

    def test_from_dict_unknown_key(self):
        with self.assertRaises(ConfigError):
            PlatformConfig.from_dict({"risk": {"nope": 1}})
        with self.assertRaises(ConfigError):
            PlatformConfig.from_dict({"bogus": 1})

    def test_roundtrip(self):
        cfg = PlatformConfig.from_dict({"initial_capital": 5000, "risk": {"max_daily_loss_pct": 0.02}})
        self.assertEqual(cfg.risk.max_daily_loss_pct, 0.02)
        self.assertEqual(PlatformConfig.from_dict(cfg.to_dict()).initial_capital, 5000)


class ModelTests(unittest.TestCase):
    def test_instrument_rounding(self):
        i = Instrument("X", tick_size=0.05, lot_size=10)
        self.assertAlmostEqual(i.round_price(100.07), 100.05)
        self.assertEqual(i.round_qty(37), 30)
        self.assertEqual(i.round_qty(-37), -30)
        self.assertEqual(i.round_qty(5), 0)

    def test_order_validation(self):
        o = Order("1", "X", Side.BUY, 0)
        with self.assertRaises(ValidationError):
            o.validate()
        with self.assertRaises(ValidationError):
            Order("1", "X", Side.BUY, 1, OrderType.LIMIT).validate()
        with self.assertRaises(ValidationError):
            Order("1", "X", Side.BUY, 1, OrderType.MARKET, limit_price=5).validate()

    def test_signal_bounds_and_bar_consistency(self):
        with self.assertRaises(ValidationError):
            Signal("X", 1.5)
        with self.assertRaises(ValidationError):
            Bar(datetime(2020, 1, 1), "X", 10, 9, 8, 10)   # high below open


class InfraTests(unittest.TestCase):
    def test_registry(self):
        r = Registry("thing")

        @r.register("a")
        class A:
            pass
        self.assertIs(r.get("A"), A)
        with self.assertRaises(ConfigError):
            r.get("missing")
        with self.assertRaises(ConfigError):
            r.register("a")(type("B", (), {}))

    def test_event_bus_isolates_failures(self):
        bus, seen = EventBus(), []
        bus.subscribe("t", lambda t, p: (_ for _ in ()).throw(RuntimeError("boom")))
        bus.subscribe("t", lambda t, p: seen.append(p))
        bus.publish("t", 1)
        self.assertEqual(seen, [1])
        self.assertEqual(bus.handler_errors, 1)

    def test_audit_log(self):
        a = AuditLog()
        a.record("u", "x", k=1)
        self.assertEqual(len(a.entries("x")), 1)

    def test_empty_audit_log_is_truthy_so_it_is_shared_not_replaced(self):
        # regression: `audit or AuditLog()` used to discard an empty shared log
        from algoplatform.orders import OrderManager
        shared = AuditLog()
        self.assertTrue(bool(shared))
        self.assertIs(OrderManager(audit=shared).audit, shared)

    def test_retry_only_retryable(self):
        calls = {"n": 0}

        @retry(max_attempts=3, base_delay=0, sleep=lambda s: None)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise BrokerConnectionError("down")
            return "ok"
        self.assertEqual(flaky(), "ok")

        calls["n"] = 0

        @retry(max_attempts=3, base_delay=0, sleep=lambda s: None)
        def rejected():
            calls["n"] += 1
            raise OrderRejectedError("no")
        with self.assertRaises(OrderRejectedError):
            rejected()
        self.assertEqual(calls["n"], 1)         # non-retryable: no second attempt

    def test_circuit_breaker(self):
        t = {"now": 0.0}
        cb = CircuitBreaker(2, reset_timeout=10, clock=lambda: t["now"])

        def bad():
            raise ValueError("x")
        for _ in range(2):
            with self.assertRaises(ValueError):
                cb.call(bad)
        self.assertEqual(cb.state, "open")
        with self.assertRaises(BrokerConnectionError):
            cb.call(lambda: 1)
        t["now"] = 11
        self.assertEqual(cb.state, "half_open")
        self.assertEqual(cb.call(lambda: 5), 5)
        self.assertEqual(cb.state, "closed")

    def test_rate_limiter(self):
        t = {"now": 0.0}
        rl = RateLimiter(2, 1.0, clock=lambda: t["now"])
        self.assertTrue(rl.allow() and rl.allow())
        self.assertFalse(rl.allow())
        t["now"] = 1.5
        self.assertTrue(rl.allow())

    def test_helpers(self):
        self.assertEqual(safe_div(1, 0, -1), -1)
        with self.assertRaises(ValidationError):
            to_array([], min_len=1)
        g = IdGenerator("A", "tok")
        self.assertEqual(g.next(), "A-tok-00000001")

    def test_sim_clock_monotonic_and_calendar(self):
        c = SimClock()
        c.set(datetime(2020, 1, 2))
        with self.assertRaises(ValueError):
            c.set(datetime(2020, 1, 1))
        cal = TradingCalendar(holidays=["2024-12-25"])
        self.assertTrue(cal.is_open(datetime(2024, 12, 24, 15, 0, tzinfo=timezone.utc)))
        self.assertFalse(cal.is_open(datetime(2024, 12, 25, 15, 0, tzinfo=timezone.utc)))   # holiday
        self.assertFalse(cal.is_open(datetime(2024, 12, 28, 15, 0, tzinfo=timezone.utc)))   # Saturday
        self.assertEqual(cal.next_trading_day(datetime(2024, 12, 27)).isoformat(), "2024-12-30")

    def test_error_serialisation(self):
        e = PlatformError("m", code="C", details={"a": 1})
        self.assertEqual(e.to_dict()["error"], "C")


if __name__ == "__main__":
    unittest.main()
