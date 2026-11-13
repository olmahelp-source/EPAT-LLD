"""Limit order book and tick-level market simulator.

The paper broker fills against bars. That is right for daily strategies and wrong for anything where the questions are
"where am I in the queue", "how much does it cost to cross the spread for this size" or "what does 50 ms of latency
cost me". This module answers those with a real matching engine:

* :class:`OrderBook`        price-time priority matching for limit and market orders, with IOC / FOK, post-only,
                            self-trade prevention, queue position and book analytics (spread, imbalance, microprice,
                            cost to trade a size);
* :class:`MarketSimulator`  a seeded stream of background orders and cancellations around the book ("zero
                            intelligence" flow), in which your own orders arrive after a configurable latency;
* :func:`ticks_to_bars`     turns the simulated tape into OHLCV bars the rest of the platform already understands.

What it is not: the background flow is random, not a model of real participants, so it shows *mechanics* (queueing,
slippage, latency) and not market behaviour. The trading engine itself still runs on bars; this is a laboratory for
execution questions, not yet a tick-driven backtester.
"""
from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...foundation.core.enums import Side, TimeInForce
from ...foundation.core.exceptions import OrderError, OrderNotFoundError, ValidationError


@dataclass
class BookOrder:
    order_id: str
    side: Side
    tick: int                   # price in integer ticks: no float comparisons inside the engine
    quantity: float             # remaining
    owner: str = ""
    seq: int = 0                # arrival order, the "time" in price-time priority


@dataclass(frozen=True)
class BookTrade:
    price: float
    quantity: float
    maker_id: str               # the resting order
    taker_id: str               # the incoming order
    aggressor: Side
    seq: int


class OrderBook:
    def __init__(self, tick_size: float = 0.01, prevent_self_trade: bool = True) -> None:
        if not tick_size > 0:
            raise ValidationError("tick_size must be positive")
        self.tick_size, self.prevent_self_trade = tick_size, prevent_self_trade
        self._levels: Dict[Side, Dict[int, Deque[BookOrder]]] = {Side.BUY: {}, Side.SELL: {}}
        self._prices: Dict[Side, List[int]] = {Side.BUY: [], Side.SELL: []}        # sorted ascending
        self._orders: Dict[str, BookOrder] = {}
        self._seq = 0
        self.trades: List[BookTrade] = []
        self.last_price: Optional[float] = None

    # ---- helpers --------------------------------------------------------------------------------------------------------
    def _tick(self, price: float) -> int:
        if not (isinstance(price, (int, float)) and math.isfinite(price) and price > 0):
            raise ValidationError("price must be a positive number", details={"price": price})
        return int(round(price / self.tick_size))

    def _price(self, tick: int) -> float:
        return round(tick * self.tick_size, 10)

    @staticmethod
    def _qty(quantity: float) -> float:
        if not (isinstance(quantity, (int, float)) and math.isfinite(quantity) and quantity > 0):
            raise ValidationError("quantity must be a positive number", details={"quantity": quantity})
        return float(quantity)

    def _best(self, side: Side) -> Optional[int]:
        prices = self._prices[side]
        if not prices:
            return None
        return prices[-1] if side is Side.BUY else prices[0]

    def _rest(self, order: BookOrder) -> None:
        level = self._levels[order.side].get(order.tick)
        if level is None:
            level = self._levels[order.side][order.tick] = deque()
            bisect.insort(self._prices[order.side], order.tick)
        level.append(order)
        self._orders[order.order_id] = order

    def _drop_level(self, side: Side, tick: int) -> None:
        del self._levels[side][tick]
        self._prices[side].pop(bisect.bisect_left(self._prices[side], tick))

    def _crosses(self, side: Side, tick: Optional[int], best_opposite: int) -> bool:
        return tick is None or (tick >= best_opposite if side is Side.BUY else tick <= best_opposite)

    def _available(self, side: Side, tick: Optional[int], owner: str) -> float:
        """Quantity an incoming order could take right now (used for fill-or-kill)."""
        opposite = side.opposite
        prices = self._prices[opposite] if opposite is Side.SELL else reversed(self._prices[opposite])
        total = 0.0
        for p in prices:
            if not self._crosses(side, tick, p):
                break
            for o in self._levels[opposite][p]:
                if self.prevent_self_trade and owner and o.owner == owner:
                    return total
                total += o.quantity
        return total

    def _match(self, taker_id: str, side: Side, tick: Optional[int], quantity: float, owner: str) -> Tuple[List[BookTrade], float]:
        trades: List[BookTrade] = []
        opposite = side.opposite
        while quantity > 1e-12:
            best = self._best(opposite)
            if best is None or not self._crosses(side, tick, best):
                break
            level = self._levels[opposite][best]
            maker = level[0]
            if self.prevent_self_trade and owner and maker.owner == owner:
                break                                               # never trade with yourself: the rest of the incoming order is cancelled
            fill = min(quantity, maker.quantity)
            self._seq += 1
            trade = BookTrade(self._price(best), fill, maker.order_id, taker_id, side, self._seq)
            trades.append(trade)
            maker.quantity -= fill
            quantity -= fill
            if maker.quantity <= 1e-12:
                level.popleft()
                del self._orders[maker.order_id]
                if not level:
                    self._drop_level(opposite, best)
        if trades:
            self.trades += trades
            self.last_price = trades[-1].price
        return trades, quantity

    # ---- order entry ------------------------------------------------------------------------------------------------------
    def add_limit(self, order_id: str, side: Side, price: float, quantity: float, owner: str = "", tif: TimeInForce = TimeInForce.GTC,
                  post_only: bool = False) -> List[BookTrade]:
        """Match what crosses, rest the remainder (unless IOC / FOK). Returns the trades this order caused.

        ``post_only``: refuse the order if it would trade at once, so it can only ever add liquidity.
        ``FOK``: fill completely at once or do nothing. ``IOC``: fill what is possible at once, cancel the rest."""
        if order_id in self._orders:
            raise OrderError(f"order '{order_id}' is already resting in the book", code="DUPLICATE_BOOK_ORDER")
        tick, quantity = self._tick(price), self._qty(quantity)
        best_opposite = self._best(side.opposite)
        would_cross = best_opposite is not None and self._crosses(side, tick, best_opposite)
        if post_only and would_cross:
            raise OrderError("post-only order would trade immediately", code="POST_ONLY_WOULD_CROSS")
        if tif is TimeInForce.FOK and self._available(side, tick, owner) < quantity - 1e-12:
            return []
        trades, remaining = self._match(order_id, side, tick, quantity, owner)
        blocked_by_self = remaining > 1e-12 and self._best(side.opposite) is not None and self._crosses(side, tick, self._best(side.opposite))  # type: ignore[arg-type]
        if remaining > 1e-12 and tif not in (TimeInForce.IOC, TimeInForce.FOK) and not blocked_by_self:
            self._seq += 1
            self._rest(BookOrder(order_id, side, tick, remaining, owner, self._seq))
        return trades

    def add_market(self, order_id: str, side: Side, quantity: float, owner: str = "") -> List[BookTrade]:
        """Take liquidity until filled or the other side is empty; whatever is left over is cancelled, never rested."""
        trades, _ = self._match(order_id, side, None, self._qty(quantity), owner)
        return trades

    def cancel(self, order_id: str) -> float:
        """Remove a resting order; returns the quantity that was cancelled."""
        order = self._orders.pop(order_id, None)
        if order is None:
            raise OrderNotFoundError(f"order '{order_id}' is not in the book (filled or already cancelled)")
        level = self._levels[order.side][order.tick]
        level.remove(order)
        if not level:
            self._drop_level(order.side, order.tick)
        return order.quantity

    def reduce(self, order_id: str, new_quantity: float) -> None:
        """Shrink a resting order. It keeps its place in the queue (growing an order would not, so that is refused)."""
        order = self._orders.get(order_id)
        if order is None:
            raise OrderNotFoundError(f"order '{order_id}' is not in the book")
        if not 0 < new_quantity <= order.quantity:
            raise OrderError("reduce needs 0 < new quantity <= resting quantity; cancel and re-enter to increase", code="BAD_REDUCE")
        order.quantity = float(new_quantity)

    # ---- state -----------------------------------------------------------------------------------------------------------------
    def __contains__(self, order_id: str) -> bool:
        return order_id in self._orders

    def __len__(self) -> int:
        return len(self._orders)

    def remaining(self, order_id: str) -> float:
        order = self._orders.get(order_id)
        return order.quantity if order else 0.0

    def best_bid(self) -> Optional[float]:
        t = self._best(Side.BUY)
        return None if t is None else self._price(t)

    def best_ask(self) -> Optional[float]:
        t = self._best(Side.SELL)
        return None if t is None else self._price(t)

    def spread(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        return None if b is None or a is None else round(a - b, 10)

    def mid(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        return None if b is None or a is None else (a + b) / 2.0

    def depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """Aggregate size per price level, best first."""
        def side(s: Side) -> List[Tuple[float, float]]:
            prices = self._prices[s][::-1] if s is Side.BUY else self._prices[s]
            return [(self._price(p), sum(o.quantity for o in self._levels[s][p])) for p in prices[:levels]]
        return {"bids": side(Side.BUY), "asks": side(Side.SELL)}

    def imbalance(self, levels: int = 1) -> float:
        """(bid size - ask size) / (bid size + ask size) over the top ``levels``: +1 all bids, -1 all asks."""
        d = self.depth(levels)
        bid, ask = sum(q for _, q in d["bids"]), sum(q for _, q in d["asks"])
        return (bid - ask) / (bid + ask) if bid + ask > 0 else 0.0

    def microprice(self) -> Optional[float]:
        """Mid weighted by the *opposite* side's size: a large bid pulls it towards the ask, where the next trade is likelier."""
        d = self.depth(1)
        if not d["bids"] or not d["asks"]:
            return None
        (bp, bq), (ap, aq) = d["bids"][0], d["asks"][0]
        return (bp * aq + ap * bq) / (bq + aq)

    def queue_ahead(self, order_id: str) -> float:
        """Size resting ahead of this order at its own price: what must trade (or cancel) before it is next."""
        order = self._orders.get(order_id)
        if order is None:
            raise OrderNotFoundError(f"order '{order_id}' is not in the book")
        ahead = 0.0
        for o in self._levels[order.side][order.tick]:
            if o is order:
                return ahead
            ahead += o.quantity
        return ahead

    def cost_to_trade(self, side: Side, quantity: float) -> Dict[str, float]:
        """What a market order of this size would pay right now, without sending it: average price, how much the book
        can fill, levels consumed, and slippage against the mid in basis points."""
        quantity = self._qty(quantity)
        opposite = side.opposite
        prices = self._prices[opposite] if opposite is Side.SELL else self._prices[opposite][::-1]
        filled = cost = 0.0
        used = 0
        for p in prices:
            if filled >= quantity - 1e-12:
                break
            take = min(quantity - filled, sum(o.quantity for o in self._levels[opposite][p]))
            filled, cost, used = filled + take, cost + take * self._price(p), used + 1
        mid = self.mid()
        avg = cost / filled if filled > 0 else float("nan")
        slip = side.sign * (avg - mid) / mid * 1e4 if filled > 0 and mid else float("nan")
        return {"average_price": avg, "filled": filled, "unfilled": quantity - filled, "levels": float(used), "slippage_bps": slip}


@dataclass
class _Pending:
    arrive: float
    kind: str                   # limit | market | cancel
    order_id: str
    side: Optional[Side] = None
    quantity: float = 0.0
    price: Optional[float] = None


class MarketSimulator:
    """A book fed by random background orders, plus your own orders with latency.

    Background flow: events arrive as a Poisson process around a slowly drifting fair value (``volatility``, in ticks
    per square-root second). Each event is a limit order placed a geometric number of ticks away from fair value on
    its own side, a market order, or the cancellation of a random resting background order. When fair value moves
    through stale quotes, new limit orders trade against them, which is what moves the price.
    Your orders (:meth:`submit`, :meth:`cancel`) are stamped with the current time and reach the book ``latency``
    seconds later, by which time the book may have moved: that gap is what latency costs.

    Same seed, same tape. ``fills`` lists the trades your orders took part in."""

    OWNER = "user"

    def __init__(self, mid: float = 100.0, tick_size: float = 0.01, event_rate: float = 50.0, limit_share: float = 0.55,
                 market_share: float = 0.10, mean_size: float = 100.0, mean_distance: float = 3.0, initial_depth: int = 10,
                 latency: float = 0.0, volatility: float = 2.0, seed: Optional[int] = 0) -> None:
        if event_rate <= 0 or mean_size <= 0 or mean_distance < 1 or initial_depth < 1 or latency < 0 or mid <= 0 or volatility < 0:
            raise ValidationError("need positive mid, event_rate and mean_size, mean_distance >= 1, initial_depth >= 1, latency >= 0, "
                                  "volatility >= 0")
        if not (0 < limit_share < 1 and 0 < market_share < 1 and limit_share + market_share < 1):
            raise ValidationError("limit_share and market_share must be in (0, 1) and sum to less than 1 (the rest are cancellations)")
        self.book = OrderBook(tick_size)
        self.event_rate, self.limit_share, self.market_share = event_rate, limit_share, market_share
        self.mean_size, self.mean_distance, self.latency = mean_size, mean_distance, latency
        self.volatility = volatility                                # of the reference price, in ticks per square-root second
        self._rng = np.random.default_rng(seed)
        self._n = 0
        self.now = 0.0
        self._background: List[str] = []
        self._pending: List[_Pending] = []
        self._user_ids: set = set()
        self.fills: List[BookTrade] = []
        self.rejections: List[Tuple[str, str]] = []
        self.tape: List[Tuple[float, float, float, float, float, float, float]] = []
        self._reference = mid
        self._seed_book(initial_depth)

    # ---- background flow ---------------------------------------------------------------------------------------------------
    def _size(self) -> float:
        return float(max(1, round(self._rng.exponential(self.mean_size))))

    def _new_id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def _seed_book(self, levels: int) -> None:
        tick = self.book.tick_size
        ref = round(round(self._reference / tick) * tick, 10)
        for i in range(1, levels + 1):
            for side, price in ((Side.BUY, ref - i * tick), (Side.SELL, ref + i * tick)):
                if price <= 0:
                    continue
                oid = self._new_id("bg")
                try:
                    self.book.add_limit(oid, side, price, self._size(), post_only=True)
                    self._background.append(oid)
                except OrderError:
                    pass                                            # would cross a quote that is still there: leave that level alone

    def _background_event(self, dt: float = 0.0) -> None:
        tick = self.book.tick_size
        if self.volatility > 0 and dt > 0:                           # the "fair value" that new quotes are placed around drifts
            self._reference = max(self._reference + self._rng.normal(0.0, self.volatility * tick * math.sqrt(dt)), 2 * tick)
        u = self._rng.random()
        side = Side.BUY if self._rng.random() < 0.5 else Side.SELL
        if u < self.limit_share:
            distance = int(self._rng.geometric(1.0 / self.mean_distance))               # at least one tick from fair value
            price = round(round((self._reference - side.sign * distance * tick) / tick) * tick, 10)
            if price <= 0:
                return                                                # when fair value has moved through stale quotes, this order trades
            oid = self._new_id("bg")
            trades = self.book.add_limit(oid, side, price, self._size())
            self._record(trades)
            if oid in self.book:
                self._background.append(oid)
        elif u < self.limit_share + self.market_share:
            self._record(self.book.add_market(self._new_id("bg"), side, self._size()))
        elif self._background:
            i = int(self._rng.integers(len(self._background)))
            oid = self._background[i]
            self._background[i] = self._background[-1]
            self._background.pop()
            if oid in self.book:
                self.book.cancel(oid)
        if self.book.best_bid() is None or self.book.best_ask() is None:
            self._seed_book(5)                                       # a side was swept: liquidity comes back around fair value

    def _record(self, trades: List[BookTrade]) -> None:
        for t in trades:
            if t.maker_id in self._user_ids or t.taker_id in self._user_ids:
                self.fills.append(t)

    # ---- your orders -------------------------------------------------------------------------------------------------------------
    def submit(self, side: Side, quantity: float, price: Optional[float] = None) -> str:
        """Send a limit order (with ``price``) or a market order. It reaches the book after ``latency`` seconds."""
        oid = self._new_id("u")
        self._user_ids.add(oid)
        OrderBook._qty(quantity)
        self._pending.append(_Pending(self.now + self.latency, "market" if price is None else "limit", oid, side, float(quantity), price))
        return oid

    def cancel(self, order_id: str) -> None:
        """Request a cancel; it too takes ``latency`` to arrive, so the order may fill first."""
        if order_id not in self._user_ids:
            raise OrderNotFoundError(f"'{order_id}' is not one of your orders")
        self._pending.append(_Pending(self.now + self.latency, "cancel", order_id))

    def _deliver(self) -> None:
        due = [p for p in self._pending if p.arrive <= self.now]
        if not due:
            return
        self._pending = [p for p in self._pending if p.arrive > self.now]
        for p in sorted(due, key=lambda p: p.arrive):
            try:
                if p.kind == "cancel":
                    self.book.cancel(p.order_id)
                elif p.kind == "market":
                    self._record(self.book.add_market(p.order_id, p.side, p.quantity, self.OWNER))      # type: ignore[arg-type]
                else:
                    self._record(self.book.add_limit(p.order_id, p.side, p.price, p.quantity, self.OWNER))   # type: ignore[arg-type]
            except (OrderError, ValidationError) as exc:           # e.g. cancelling an order that already filled
                self.rejections.append((p.order_id, exc.message))

    def filled(self, order_id: str) -> float:
        return sum(t.quantity for t in self.fills if order_id in (t.maker_id, t.taker_id))

    def average_fill_price(self, order_id: str) -> float:
        mine = [t for t in self.fills if order_id in (t.maker_id, t.taker_id)]
        qty = sum(t.quantity for t in mine)
        return sum(t.price * t.quantity for t in mine) / qty if qty > 0 else float("nan")

    # ---- clock -------------------------------------------------------------------------------------------------------------------
    def step(self) -> None:
        """Advance to the next background event, delivering any of your orders whose latency has elapsed on the way."""
        dt = float(self._rng.exponential(1.0 / self.event_rate))
        self.now += dt
        self._deliver()
        self._background_event(dt)
        d = self.book.depth(1)
        bid, bid_size = d["bids"][0] if d["bids"] else (float("nan"), 0.0)
        ask, ask_size = d["asks"][0] if d["asks"] else (float("nan"), 0.0)
        last = self.book.trades[-1] if self.book.trades else None
        self.tape.append((self.now, bid, ask, bid_size, ask_size, last.price if last else float("nan"), float(len(self.book.trades))))

    def run(self, events: int) -> pd.DataFrame:
        """Run ``events`` background events and return the quote tape so far."""
        if events < 1:
            raise ValidationError("events must be >= 1")
        for _ in range(events):
            self.step()
        return self.quotes()

    def quotes(self) -> pd.DataFrame:
        """One row per event: time, best bid / ask and their sizes, last trade price, cumulative trade count."""
        return pd.DataFrame(self.tape, columns=["time", "bid", "ask", "bid_size", "ask_size", "last", "trade_count"])

    def trade_tape(self) -> pd.DataFrame:
        """Every trade in the book so far: sequence number, price, size, aggressor side."""
        return pd.DataFrame([(t.seq, t.price, t.quantity, t.aggressor.value) for t in self.book.trades],
                            columns=["seq", "price", "quantity", "aggressor"])


def ticks_to_bars(quotes: pd.DataFrame, seconds: float = 60.0, start: Any = "2024-01-02 09:30:00", price: str = "mid") -> pd.DataFrame:
    """Aggregate a quote tape (as returned by :meth:`MarketSimulator.quotes`) into OHLCV bars with a datetime index, ready
    for ``MarketData.add``. ``price`` is ``"mid"`` or ``"last"``; volume is the number of trades in the bar. Buckets with
    no usable price are left out rather than invented."""
    if seconds <= 0 or price not in ("mid", "last"):
        raise ValidationError("seconds must be positive and price 'mid' or 'last'")
    missing = [c for c in ("time", "bid", "ask", "last", "trade_count") if c not in quotes]
    if missing:
        raise ValidationError(f"quote tape is missing column(s) {missing}")
    px = (quotes["bid"] + quotes["ask"]) / 2.0 if price == "mid" else quotes["last"]
    ok = px.notna()
    if not ok.any():
        raise ValidationError("the tape holds no usable prices")
    bucket = np.floor(quotes.loc[ok, "time"] / seconds).astype(int)
    grouped = px[ok].groupby(bucket)
    trades = quotes.loc[ok, "trade_count"].groupby(bucket).max()
    volume = trades.diff().fillna(trades.iloc[0]).clip(lower=0.0)
    bars = pd.DataFrame({"open": grouped.first(), "high": grouped.max(), "low": grouped.min(), "close": grouped.last(), "volume": volume})
    bars.index = pd.Timestamp(start) + pd.to_timedelta(bars.index.to_numpy() * seconds, unit="s")
    return bars
