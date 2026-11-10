"""Resilience wrappers: retry + circuit breaker + failover, and a REST adapter base."""
from __future__ import annotations

import abc
from typing import Any, Dict, Optional

from ..core.exceptions import BrokerConnectionError, BrokerError, OrderRejectedError, RateLimitError
from ..core.logging import get_logger
from ..core.models import Order
from ..utils.decorators import CircuitBreaker, RateLimiter, retry
from .base import BrokerGateway

log = get_logger(__name__)


class ResilientGateway(BrokerGateway):
    """Wraps a primary gateway (and optional secondary) with retries and a breaker.

    New orders fail over to the secondary when the primary's circuit is open.
    Cancels always go to whichever broker owns the order.
    """
    name = "resilient"

    def __init__(self, primary: BrokerGateway, secondary: Optional[BrokerGateway] = None,
                 max_attempts: int = 3, failure_threshold: int = 3, reset_timeout: float = 30.0,
                 sleep=lambda s: None) -> None:
        super().__init__()
        self.primary, self.secondary = primary, secondary
        self._breaker = CircuitBreaker(failure_threshold, reset_timeout)
        self._owner: Dict[str, BrokerGateway] = {}
        self._attempts, self._sleep = max_attempts, sleep
        for g in filter(None, (primary, secondary)):
            g.set_event_handler(self._emit)

    def connect(self) -> None:
        self.primary.connect()
        if self.secondary:
            try:
                self.secondary.connect()
            except BrokerError as exc:
                log.warning("secondary broker unavailable: %r", exc)
        self._connected = True

    def disconnect(self) -> None:
        for g in filter(None, (self.primary, self.secondary)):
            try:
                g.disconnect()
            except BrokerError as exc:
                log.warning("disconnect failed: %r", exc)
        self._connected = False

    def submit_order(self, order: Order) -> str:
        self._require_connected()

        @retry(max_attempts=self._attempts, base_delay=0.01, sleep=self._sleep)
        def via_primary() -> str:
            return self._breaker.call(self.primary.submit_order, order)

        try:
            bid = via_primary()
            self._owner[order.client_order_id] = self.primary
            return bid
        except OrderRejectedError:
            raise                                  # a rejection is final, never failed over
        except (BrokerConnectionError, RateLimitError) as exc:
            if self.secondary is None:
                raise
            log.error("primary broker failed (%r); failing over for %s", exc, order.client_order_id)
            bid = self.secondary.submit_order(order)
            self._owner[order.client_order_id] = self.secondary
            return bid

    def cancel_order(self, order: Order) -> None:
        self._owner.get(order.client_order_id, self.primary).cancel_order(order)

    def get_positions(self) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for g in filter(None, (self.primary, self.secondary)):
            for s, q in g.get_positions().items():
                merged[s] = merged.get(s, 0.0) + q
        return merged

    def get_cash(self) -> float:
        return self.primary.get_cash()

    def open_order_ids(self) -> set:
        out = set()
        for g in filter(None, (self.primary, self.secondary)):
            out |= g.open_order_ids()
        return out


class RestBrokerBase(BrokerGateway):
    """Template for HTTP brokers (Alpaca, Zerodha, Binance, ...).

    Subclasses implement ``_request`` and ``_parse_*``; this base supplies client
    side rate limiting, retry on transient errors, and error classification, so
    each concrete adapter is only the broker-specific mapping.
    """

    def __init__(self, credentials: Optional[Dict[str, str]] = None, paper: bool = True,
                 max_calls_per_sec: int = 5, retry_attempts: int = 3, retry_sleep=None) -> None:
        super().__init__()
        self._credentials = credentials or {}
        self.paper = paper
        self._limiter = RateLimiter(max_calls_per_sec, 1.0)
        self._retry_attempts = retry_attempts
        self._retry_sleep = retry_sleep or __import__("time").sleep

    @abc.abstractmethod
    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform one HTTP call and return the decoded JSON, raising typed errors."""

    def _call(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self._limiter.allow():
            raise RateLimitError("client-side rate limit")

        @retry(max_attempts=self._retry_attempts, base_delay=0.2, sleep=self._retry_sleep)
        def go() -> Dict[str, Any]:
            return self._request(method, path, payload)
        return go()

    @staticmethod
    def classify_http_error(status: int, body: str = "") -> BrokerError:
        if status == 429:
            return RateLimitError(body or "rate limited")
        if status >= 500:
            return BrokerConnectionError(f"server error {status}")
        return OrderRejectedError(body or f"rejected ({status})", details={"status": status})
