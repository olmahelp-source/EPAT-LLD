"""Broker gateway interface (one abstraction for every venue/broker)."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from ..core.exceptions import BrokerConnectionError
from ..core.models import Fill, Order
from ..core.registry import Registry


@dataclass
class BrokerEvent:
    """Asynchronous message from the broker about an order."""
    kind: str                       # ack | fill | rejected | cancelled | expired
    client_order_id: str
    broker_order_id: str = ""
    fill: Optional[Fill] = None
    reason: str = ""


EventHandler = Callable[[BrokerEvent], None]


class BrokerGateway(abc.ABC):
    """Contract every broker adapter implements.

    The execution layer only talks to this interface, so adding a broker means
    writing one subclass and registering it in :data:`BROKERS`.
    """
    name = "abstract"

    def __init__(self) -> None:
        self._handler: Optional[EventHandler] = None
        self._connected = False

    # ---- plumbing -----------------------------------------------------------------
    def set_event_handler(self, handler: EventHandler) -> None:
        self._handler = handler

    def _emit(self, event: BrokerEvent) -> None:
        if self._handler is not None:
            self._handler(event)

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _require_connected(self) -> None:
        if not self._connected:
            raise BrokerConnectionError(f"{self.name} not connected")

    # ---- required API -------------------------------------------------------------
    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def disconnect(self) -> None: ...

    @abc.abstractmethod
    def submit_order(self, order: Order) -> str:
        """Send an order. Returns the broker order id; raises on failure."""

    @abc.abstractmethod
    def cancel_order(self, order: Order) -> None: ...

    @abc.abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """Broker's view of signed quantities, for reconciliation."""

    @abc.abstractmethod
    def get_cash(self) -> float: ...

    def open_order_ids(self) -> set:
        return set()

    def __enter__(self) -> "BrokerGateway":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()


BROKERS: Registry[BrokerGateway] = Registry("broker")


def create_gateway(name: str, credentials: Optional[Dict[str, str]] = None, paper: bool = True) -> BrokerGateway:
    cls = BROKERS.get(name)
    try:
        return cls(credentials=credentials or {}, paper=paper)  # type: ignore[call-arg]
    except TypeError:
        return cls()
