import math
import threading
import time
import unittest

import numpy as np

import helpers  # noqa: F401
from algoplatform.foundation.core.enums import Severity
from algoplatform.foundation.core.exceptions import (AggregateError, BrokerConnectionError, CapacityError, ConfigError, KillSwitchActive,
                                          MissingDataError, OperationTimeout, OrderRejectedError, PlatformError, RateLimitError,
                                          RiskLimitBreach, ValidationError)
from algoplatform.foundation.utils import (Bulkhead, DeadLetterQueue, Deadline, ErrorAction, ErrorBoundary, ErrorBudget, ErrorCollector,
                                ErrorPolicy, Result, attempt, call_with_timeout, classify, error_context, first_success,
                                logsumexp, nelder_mead, partition, rank, softmax, spearman, with_fallback)


class ClassificationTests(unittest.TestCase):
    def test_default_decisions(self):
        cases = [(KillSwitchActive("halt"), ErrorAction.HALT, Severity.CRITICAL), (RateLimitError("slow down"), ErrorAction.RETRY, Severity.WARNING),
                 (BrokerConnectionError("down"), ErrorAction.RETRY, Severity.ERROR), (OrderRejectedError("no"), ErrorAction.SKIP, Severity.WARNING),
                 (RiskLimitBreach("limit"), ErrorAction.SKIP, Severity.WARNING), (ConfigError("bad"), ErrorAction.HALT, Severity.CRITICAL),
                 (MissingDataError("gap"), ErrorAction.FALLBACK, Severity.WARNING), (TimeoutError(), ErrorAction.RETRY, Severity.WARNING),
                 (ZeroDivisionError(), ErrorAction.ESCALATE, Severity.ERROR)]
        for exc, action, severity in cases:
            info = classify(exc)
            self.assertEqual((info.action, info.severity), (action, severity), type(exc).__name__)
        self.assertEqual(classify(ValueError("x")).category, "bug")
        self.assertEqual(classify(RateLimitError("x", details={"a": 1})).to_dict()["details"], {"a": 1})

    def test_most_specific_custom_rule_wins(self):
        policy = ErrorPolicy().on(PlatformError, ErrorAction.ESCALATE).on(ValidationError, ErrorAction.HALT, Severity.CRITICAL)
        self.assertIs(policy.decide(ValidationError("x")).action, ErrorAction.HALT)
        self.assertIs(policy.decide(ConfigError("x")).action, ErrorAction.ESCALATE)

    def test_retry_is_not_offered_for_a_non_retryable_error(self):
        policy = ErrorPolicy().on(ValidationError, ErrorAction.RETRY)
        self.assertIs(policy.decide(ValidationError("x")).action, ErrorAction.ESCALATE)


class ResultTests(unittest.TestCase):
    def test_success_and_failure_paths(self):
        ok = attempt(lambda x: x * 2, 21)
        self.assertTrue(ok.ok)
        self.assertEqual(ok.unwrap(), 42)
        self.assertEqual(ok.map(lambda v: v + 1).unwrap(), 43)
        bad = attempt(lambda: 1 / 0)
        self.assertFalse(bad.ok)
        self.assertEqual(bad.unwrap_or(-1), -1)
        self.assertIs(bad.map(lambda v: v + 1), bad)                    # mapping a failure keeps the original error
        self.assertEqual(bad.recover(lambda e: 0).unwrap(), 0)
        self.assertEqual(bad.info().category, "bug")
        with self.assertRaises(ZeroDivisionError):
            bad.unwrap()

    def test_map_captures_errors_and_partition_splits(self):
        self.assertFalse(Result.success(1).map(lambda v: v / 0).ok)
        self.assertFalse(Result.success(1).and_then(lambda v: 1 / 0).ok)
        values, errors = partition([attempt(int, "1"), attempt(int, "x"), attempt(int, "3")])
        self.assertEqual(values, [1, 3])
        self.assertEqual(len(errors), 1)


class FallbackTests(unittest.TestCase):
    def test_first_working_candidate_is_used(self):
        seen = []
        out = first_success(lambda: 1 / 0, lambda: "backup", lambda: seen.append("never") or "third", on_error=lambda i, e: seen.append(i))
        self.assertEqual((out, seen), ("backup", [0]))

    def test_all_failing_raises_one_aggregate(self):
        with self.assertRaises(AggregateError) as cm:
            first_success(lambda: 1 / 0, lambda: int("x"))
        self.assertEqual(len(cm.exception.errors), 2)
        self.assertEqual(cm.exception.details["count"], 2)

    def test_halt_class_error_is_never_papered_over(self):
        def halted():
            raise KillSwitchActive("stop")
        with self.assertRaises(KillSwitchActive):
            first_success(halted, lambda: "cached price")

    def test_decorator(self):
        @with_fallback(lambda sym: f"cached:{sym}")
        def live(sym):
            raise ConnectionError("feed down")
        self.assertEqual(live("AAA"), "cached:AAA")


class TimeLimitTests(unittest.TestCase):
    def test_timeout_and_passthrough(self):
        self.assertEqual(call_with_timeout(lambda a, b=1: a + b, 1.0, 2, b=3), 5)
        with self.assertRaises(OperationTimeout):
            call_with_timeout(time.sleep, 0.05, 0.5)
        with self.assertRaises(KeyError):                               # the callee's own exception comes through
            call_with_timeout(lambda: {}["x"], 1.0)
        with self.assertRaises(ValidationError):
            call_with_timeout(lambda: 1, 0)

    def test_deadline_is_shared_across_steps(self):
        t = [0.0]
        d = Deadline(10, clock=lambda: t[0])
        self.assertEqual(d.check("step 1"), 10)
        t[0] = 7
        self.assertEqual(d.remaining, 3)
        t[0] = 11
        self.assertTrue(d.expired)
        with self.assertRaises(OperationTimeout) as cm:
            d.check("step 3")
        self.assertEqual(cm.exception.details["step"], "step 3")


class BulkheadTests(unittest.TestCase):
    def test_excess_callers_are_refused_and_slots_are_released(self):
        b = Bulkhead(1, "broker")
        gate, entered = threading.Event(), threading.Event()
        t = threading.Thread(target=lambda: b.call(lambda: (entered.set(), gate.wait(2))))
        t.start()
        entered.wait(2)
        with self.assertRaises(CapacityError):
            b.call(lambda: None)
        gate.set()
        t.join()
        self.assertEqual(b.call(lambda: "ok"), "ok")
        self.assertEqual(b.rejected, 1)

    def test_slot_released_when_the_call_fails(self):
        b = Bulkhead(1)
        with self.assertRaises(ZeroDivisionError):
            b.call(lambda: 1 / 0)
        self.assertEqual(b.call(lambda: 1), 1)


class BoundaryTests(unittest.TestCase):
    def test_budget_slides_with_time(self):
        t = [0.0]
        budget = ErrorBudget(3, 10, clock=lambda: t[0])
        self.assertFalse(budget.record())
        self.assertFalse(budget.record())
        self.assertTrue(budget.record())
        t[0] = 11
        self.assertEqual(budget.used, 0)
        self.assertFalse(budget.exhausted)

    def test_contains_counts_and_escalates_once(self):
        t = [0.0]
        errors, exhausted = [], []
        boundary = ErrorBoundary("strategy", budget=ErrorBudget(2, 60, clock=lambda: t[0]),
                                 on_error=lambda info, exc: errors.append(info.code), on_exhausted=lambda name, info: exhausted.append(name))
        self.assertEqual(boundary.run(lambda: 1 / 0, default="flat"), "flat")
        self.assertEqual(boundary.run(lambda: int("x"), default="flat"), "flat")
        self.assertEqual(boundary.run(lambda: 1 / 0, default="flat"), "flat")
        self.assertEqual(exhausted, ["strategy"])                       # fired once, not on every further error
        self.assertEqual(boundary.stats()["errors"], 3)
        self.assertEqual(boundary.stats()["by_code"]["ZeroDivisionError"], 2)
        self.assertEqual(boundary.run(lambda: 7), 7)

    def test_halt_errors_pass_through_every_form(self):
        boundary = ErrorBoundary("loop")
        with self.assertRaises(KillSwitchActive):
            boundary.run(lambda: (_ for _ in ()).throw(KillSwitchActive("stop")))
        with self.assertRaises(KillSwitchActive):
            with boundary:
                raise KillSwitchActive("stop")
        with boundary:                                                  # ordinary errors are swallowed by the context form
            raise ValueError("contained")

        @boundary
        def flaky():
            raise RuntimeError("contained")
        self.assertIsNone(flaky())
        self.assertEqual(boundary.errors, 4)

    def test_reraise_list_and_broken_callback(self):
        boundary = ErrorBoundary("x", reraise=(KeyError,), on_error=lambda *a: 1 / 0)
        with self.assertRaises(KeyError):
            boundary.run(lambda: {}["k"])
        self.assertEqual(boundary.run(lambda: 1 / 0, default=5), 5)     # a failing callback does not turn into a new failure


class DeadLetterTests(unittest.TestCase):
    def test_replay_removes_only_what_succeeds(self):
        q = DeadLetterQueue()
        for item in (1, 2, 3):
            q.add(item, ValueError("first try"), "orders")
        q.add("other", ValueError("x"), "notify")

        def handler(item):
            if item == 2:
                raise RuntimeError("still broken")
        self.assertEqual(q.replay(handler, "orders"), (2, 1))
        left = q.items("orders")
        self.assertEqual([(d.item, d.attempts) for d in left], [(2, 2)])
        self.assertIn("still broken", left[0].error)
        self.assertEqual(len(q.items("notify")), 1)

    def test_eviction_is_counted_not_silent(self):
        q = DeadLetterQueue(maxlen=2)
        for i in range(5):
            q.add(i, ValueError())
        self.assertEqual((len(q), q.evicted), (2, 3))


class CollectorAndContextTests(unittest.TestCase):
    def test_batch_finishes_then_reports_everything(self):
        errors = ErrorCollector("loading")
        done = []
        for raw in ("1", "x", "3", "y"):
            with errors.catch(raw):
                done.append(int(raw))
        self.assertEqual(done, [1, 3])
        self.assertEqual([e["item"] for e in errors.summary()], ["x", "y"])
        with self.assertRaises(AggregateError) as cm:
            errors.raise_if_any()
        self.assertEqual(cm.exception.details["items"], ["x", "y"])
        ErrorCollector().raise_if_any()                                 # nothing collected: no error

    def test_max_errors_stops_early_and_halt_is_not_collected(self):
        errors = ErrorCollector("x", max_errors=2)
        with self.assertRaises(AggregateError):
            for raw in ("a", "b", "c"):
                with errors.catch(raw):
                    int(raw)
        with self.assertRaises(KillSwitchActive):
            with ErrorCollector().catch("item"):
                raise KillSwitchActive("stop")

    def test_error_context_enriches_and_wraps(self):
        with self.assertRaises(ValidationError) as cm:
            with error_context("load prices", symbol="AAA"):
                raise ValidationError("bad row", details={"symbol": "ORIGINAL"})
        self.assertEqual(cm.exception.details, {"symbol": "ORIGINAL", "operation": "load prices"})   # existing keys win
        with self.assertRaises(PlatformError) as cm:
            with error_context("parse", wrap=ConfigError, path="x.json"):
                raise KeyError("missing")
        self.assertIsInstance(cm.exception, ConfigError)
        self.assertIsInstance(cm.exception.__cause__, KeyError)
        self.assertEqual(cm.exception.details["cause"], "KeyError")


class NumericTests(unittest.TestCase):
    def test_nelder_mead_finds_minimum_and_respects_infeasible_region(self):
        x, f = nelder_mead(lambda v: (v[0] - 3) ** 2 + (v[1] + 1) ** 2, [0.0, 0.0])
        self.assertAlmostEqual(x[0], 3, places=3)
        self.assertAlmostEqual(x[1], -1, places=3)
        self.assertLess(f, 1e-6)
        x, _ = nelder_mead(lambda v: math.inf if v[0] < 1 else (v[0] - 0.2) ** 2, [2.0])
        self.assertGreaterEqual(x[0], 1.0)

    def test_rank_spearman_softmax_logsumexp(self):
        np.testing.assert_allclose(rank(np.array([10.0, 30.0, 20.0, 30.0])), [1, 3.5, 2, 3.5])
        x = np.arange(20.0)
        self.assertAlmostEqual(spearman(x, np.exp(x / 5)), 1.0)        # any monotone map keeps rank correlation at 1
        self.assertAlmostEqual(spearman(x, -x), -1.0)
        self.assertTrue(math.isnan(spearman([1, 2], [1, 2])))
        np.testing.assert_allclose(softmax(np.array([[1000.0, 1000.0]])), [[0.5, 0.5]])
        self.assertAlmostEqual(logsumexp(np.array([1000.0, 1000.0])), 1000 + math.log(2))


if __name__ == "__main__":
    unittest.main()
