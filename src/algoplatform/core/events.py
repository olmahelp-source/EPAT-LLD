"""Synchronous publish/subscribe event bus with failure isolation."""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List

from .logging import get_logger

log = get_logger(__name__)

Handler = Callable[[str, Any], None]

# Standard topics
ORDER_UPDATED = "order.updated"
FILL = "order.fill"
KILL_SWITCH = "risk.kill_switch"
RISK_BREACH = "risk.breach"
BAR = "market.bar"
ERROR = "system.error"


class EventBus:
    """Handlers run inline in publish order. A failing handler is logged and counted
    but never prevents the others from running, so one bad subscriber cannot stop
    the order path."""

    def __init__(self) -> None:
        self._subs: DefaultDict[str, List[Handler]] = defaultdict(list)
        self._lock = threading.RLock()
        self.handler_errors = 0

    def subscribe(self, topic: str, handler: Handler) -> Callable[[], None]:
        with self._lock:
            self._subs[topic].append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if handler in self._subs[topic]:
                    self._subs[topic].remove(handler)
        return unsubscribe

    def publish(self, topic: str, payload: Any = None) -> int:
        with self._lock:
            handlers = list(self._subs.get(topic, ())) + list(self._subs.get("*", ()))
        delivered = 0
        for h in handlers:
            try:
                h(topic, payload)
                delivered += 1
            except Exception as exc:  # noqa: BLE001 - isolation is the point
                self.handler_errors += 1
                log.error("event handler failed topic=%s handler=%s err=%r", topic,
                          getattr(h, "__name__", h), exc)
        return delivered

    def topics(self) -> Dict[str, int]:
        with self._lock:
            return {t: len(h) for t, h in self._subs.items()}
