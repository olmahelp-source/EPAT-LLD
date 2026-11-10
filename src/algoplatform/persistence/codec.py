"""JSON (de)serialisation of orders and fills for storage."""
from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict

import numpy as np

from ..core.enums import OrderStatus, OrderType, Side, TimeInForce
from ..core.models import Fill, Order


def _default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"cannot serialise {type(o).__name__}")


def dumps(obj: Any) -> str:
    return json.dumps(obj, default=_default, separators=(",", ":"), allow_nan=False)


def _dt(v: Any) -> Any:
    if isinstance(v, datetime):
        return v
    return datetime.fromisoformat(v) if v else None


def order_to_dict(o: Order) -> Dict[str, Any]:
    return o.to_dict()                                   # enums already values; fills live in their own table


def order_from_dict(d: Dict[str, Any]) -> Order:
    d = dict(d)
    return Order(
        client_order_id=d["client_order_id"], symbol=d["symbol"], side=Side(d["side"]), quantity=float(d["quantity"]),
        order_type=OrderType(d["order_type"]), limit_price=d.get("limit_price"), stop_price=d.get("stop_price"),
        tif=TimeInForce(d["tif"]), strategy_id=d.get("strategy_id", ""), account_id=d.get("account_id", ""),
        reduce_only=bool(d.get("reduce_only", False)), status=OrderStatus(d["status"]), filled_qty=float(d.get("filled_qty", 0.0)),
        avg_fill_price=float(d.get("avg_fill_price", 0.0)), broker_order_id=d.get("broker_order_id", ""),
        reject_reason=d.get("reject_reason", ""), created_at=_dt(d.get("created_at")), updated_at=_dt(d.get("updated_at")),
        parent_id=d.get("parent_id"), tags=dict(d.get("tags") or {}))


def fill_to_dict(f: Fill) -> Dict[str, Any]:
    return {"fill_id": f.fill_id, "client_order_id": f.client_order_id, "symbol": f.symbol, "side": f.side.value,
            "quantity": f.quantity, "price": f.price, "ts": f.ts, "commission": f.commission,
            "strategy_id": f.strategy_id, "account_id": f.account_id, "broker_order_id": f.broker_order_id}


def fill_from_dict(d: Dict[str, Any]) -> Fill:
    return Fill(d["fill_id"], d["client_order_id"], d["symbol"], Side(d["side"]), float(d["quantity"]), float(d["price"]),
                _dt(d.get("ts")), float(d.get("commission", 0.0)), d.get("strategy_id", ""), d.get("account_id", ""),
                d.get("broker_order_id", ""))
