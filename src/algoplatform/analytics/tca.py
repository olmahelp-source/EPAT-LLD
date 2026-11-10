"""Transaction-cost analysis: implementation shortfall, markouts, fill quality."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from ..core.enums import OrderStatus, OrderType
from ..core.models import Fill, Order
from ..utils.helpers import safe_div


def parent_tca(parent) -> Dict[str, Any]:
    """Shortfall of a worked parent order vs the price when the decision was made (positive = cost)."""
    arrival = parent.arrival_price
    sign = parent.side.sign
    slip = sign * (parent.avg_price - arrival) / arrival * 1e4 if (arrival and parent.filled) else 0.0
    notional = parent.fill_notional or 1.0
    return {"parent": parent.parent_id, "symbol": parent.symbol, "side": parent.side.value, "status": parent.status,
            "algo": getattr(parent.algo, "name", "?"), "total": parent.total, "filled": parent.filled,
            "fill_rate": safe_div(parent.filled, parent.total), "arrival": arrival, "avg_price": parent.avg_price,
            "slippage_bps": slip, "commission_bps": parent.commission / notional * 1e4,
            "shortfall_bps": slip + parent.commission / notional * 1e4, "children": len(parent.children),
            "bars": parent.bars_elapsed}


def order_tca(orders: Iterable[Order]) -> pd.DataFrame:
    """Per-order slippage versus the ``arrival_price`` tag the engine attaches."""
    rows = []
    for o in orders:
        arr = o.tags.get("arrival_price")
        if not arr or not o.filled_qty:
            continue
        slip = o.side.sign * (o.avg_fill_price - arr) / arr * 1e4
        rows.append({"order": o.client_order_id, "symbol": o.symbol, "side": o.side.value, "type": o.order_type.value,
                     "qty": o.filled_qty, "arrival": arr, "avg_fill": o.avg_fill_price, "slippage_bps": slip,
                     "parent": o.parent_id})
    return pd.DataFrame(rows)


def markouts(fills: Sequence[Fill], closes: pd.DataFrame, horizons: Sequence[int] = (1, 5)) -> pd.DataFrame:
    """Price move after each fill in the trader's favour, in bps. Persistently negative markouts mean
    you are being adversely selected (trading right before the price moves against you)."""
    rows = []
    for f in fills:
        if f.symbol not in closes or f.ts not in closes.index:
            continue
        col = closes[f.symbol]
        pos = col.index.get_loc(f.ts)
        row: Dict[str, Any] = {"fill": f.fill_id, "symbol": f.symbol}
        for h in horizons:
            row[f"markout_{h}"] = (f.side.sign * (col.iloc[pos + h] - f.price) / f.price * 1e4
                                   if pos + h < len(col) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def fill_quality(orders: Iterable[Order]) -> Dict[str, float]:
    orders = list(orders)
    if not orders:
        return {"orders": 0}
    n = len(orders)
    filled = sum(1 for o in orders if o.status is OrderStatus.FILLED)
    passive = sum(1 for o in orders if o.order_type is OrderType.LIMIT)
    return {"orders": n, "fill_rate": filled / n, "reject_rate": sum(o.status is OrderStatus.REJECTED for o in orders) / n,
            "cancel_rate": sum(o.status is OrderStatus.CANCELLED for o in orders) / n, "passive_share": passive / n}


def tca_summary(parents: Iterable[Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = [parent_tca(p) for p in parents]
    if not rows:
        return {"parents": 0}
    df = pd.DataFrame(rows)
    return {"parents": len(df), "avg_shortfall_bps": float(df["shortfall_bps"].mean()),
            "weighted_shortfall_bps": float(np.average(df["shortfall_bps"], weights=df["filled"]) if df["filled"].sum() else 0.0),
            "avg_fill_rate": float(df["fill_rate"].mean()), "by_algo": df.groupby("algo")["shortfall_bps"].mean().to_dict()}
