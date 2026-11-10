"""REST broker adapters: Alpaca, Zerodha (Kite) and Binance spot.

All three share ``HttpBrokerGateway``: signing/auth, error classification, a polling ``sync()`` that turns REST
order status into ack/fill/cancel events (cumulative-quantity diffs, so repeated polls never double-count), and
secret hygiene (credentials are held in ``SecretStr`` and never appear in logs or error messages).

IMPORTANT: these follow the brokers' public API shapes but have only been exercised against a fake transport.
Run them against each broker's sandbox before use, and read the broker's current API documentation.
"""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlencode

from ..core.enums import OrderType, TimeInForce
from ..core.exceptions import (AuthenticationError, BrokerConnectionError, BrokerError, ConfigError,
                               OrderRejectedError, RateLimitError)
from ..core.logging import get_logger
from ..core.models import Fill, Order
from ..users.security import SecretStr
from .base import BROKERS, BrokerEvent
from .http import Transport, UrllibTransport
from .resilient import RestBrokerBase

log = get_logger(__name__)


class HttpBrokerGateway(RestBrokerBase):
    name = "http"
    LIVE_URL = ""
    PAPER_URL = ""
    REQUIRED: Tuple[str, ...] = ()

    def __init__(self, credentials: Optional[Dict[str, str]] = None, paper: bool = True,
                 transport: Optional[Transport] = None, base_url: Optional[str] = None, clock: Callable[[], float] = time.time,
                 **kw: Any) -> None:
        super().__init__(credentials, paper, **kw)
        creds = credentials or {}
        missing = [k for k in self.REQUIRED if not creds.get(k)]
        if missing:
            raise ConfigError(f"{self.name}: missing credentials {missing}")
        self._secrets = {k: SecretStr(str(v)) for k, v in creds.items()}      # never keep plain strings around
        self._credentials = {}                                                # drop the base class's plain copy
        self._transport = transport or UrllibTransport()
        self.base_url = (base_url or (self.PAPER_URL if paper else self.LIVE_URL)).rstrip("/")
        self._clock = clock
        self._tracked: Dict[str, Dict[str, Any]] = {}
        self._seen_fill_keys: set = set()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} paper={self.paper} base={self.base_url}>"

    def _secret(self, key: str) -> str:
        return self._secrets[key].reveal()

    # ---- HTTP -----------------------------------------------------------------------------------
    def _headers(self, method: str, path: str, params: Optional[Dict[str, Any]]) -> Dict[str, str]:
        raise NotImplementedError

    def _url(self, path: str, params: Optional[Dict[str, Any]]) -> str:
        return self.base_url + path

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        status, body = self._transport.request(method, self._url(path, payload), self._headers(method, path, payload),
                                               self._body(payload))
        if 200 <= status < 300:
            return body
        text = (body.get("message") or body.get("msg") or "") if isinstance(body, dict) else ""
        raise self.classify_http_error(status, str(text)[:200])

    def _body(self, payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        return payload

    @staticmethod
    def classify_http_error(status: int, body: str = "") -> BrokerError:
        if status in (401, 403):
            return AuthenticationError(f"broker refused credentials ({status})")
        return RestBrokerBase.classify_http_error(status, body)

    # ---- lifecycle -----------------------------------------------------------------------------
    def connect(self) -> None:
        self._check_account()
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def _check_account(self) -> None:
        raise NotImplementedError

    # ---- order entry ---------------------------------------------------------------------------
    def _submit_request(self, order: Order) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError

    def submit_order(self, order: Order) -> str:
        self._require_connected()
        if order.client_order_id in self._tracked:
            raise OrderRejectedError("duplicate client order id", code="DUPLICATE_ID")
        resp = self._submit_request(order)
        bid = self._extract_id(resp, order)
        self._tracked[order.client_order_id] = {"order": order, "bid": bid, "cum": 0.0, "avg": 0.0, "open": True}
        self._emit(BrokerEvent("ack", order.client_order_id, bid))
        self._ingest(order.client_order_id, resp)                    # a market order may already be filled
        return bid

    def _extract_id(self, resp: Any, order: Order) -> str:
        raise NotImplementedError

    def cancel_order(self, order: Order) -> None:
        self._require_connected()
        t = self._tracked.get(order.client_order_id)
        if t is None or not t["open"]:
            raise OrderRejectedError("order not open at broker", code="UNKNOWN_ORDER")
        self._cancel_request(order, t["bid"])
        # Do not assume the cancel won: a fill may have raced it. Confirm the real state now; if the broker has
        # not settled yet the order stays tracked and a later sync() reports the outcome.
        try:
            self._ingest(order.client_order_id, self._status_request(order.client_order_id, t))
        except (BrokerConnectionError, RateLimitError):
            log.warning("cancel of %s sent; status confirmation deferred to sync()", order.client_order_id)

    def _cancel_request(self, order: Order, bid: str) -> None:
        raise NotImplementedError

    def open_order_ids(self) -> set:
        return {cid for cid, t in self._tracked.items() if t["open"]}

    # ---- status polling ------------------------------------------------------------------------
    def _status_request(self, cid: str, t: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def _parse_status(self, resp: Any) -> Tuple[str, float, float]:
        """-> (kind, cumulative_qty, average_price) with kind in new|partial|filled|cancelled|rejected|expired."""
        raise NotImplementedError

    def sync(self) -> int:
        """Poll every open order once and emit events for anything that changed. Returns events emitted."""
        n = 0
        for cid, t in list(self._tracked.items()):
            if not t["open"]:
                continue
            try:
                resp = self._status_request(cid, t)
            except (BrokerConnectionError, RateLimitError) as exc:
                log.warning("sync of %s skipped: %s", cid, exc.message)
                continue
            n += self._ingest(cid, resp)
        return n

    def _ingest(self, cid: str, resp: Any) -> int:
        t = self._tracked[cid]
        kind, cum, avg = self._parse_status(resp)
        emitted = 0
        if cum > t["cum"] + 1e-12:                                    # new quantity filled since last look
            qty = cum - t["cum"]
            price = (avg * cum - t["avg"] * t["cum"]) / qty if qty else avg
            key = f"{t['bid']}:{cum:.8f}"
            if key not in self._seen_fill_keys:
                self._seen_fill_keys.add(key)
                o = t["order"]
                fill = Fill(key, cid, o.symbol, o.side, qty, price, None, 0.0, o.strategy_id, o.account_id, t["bid"])
                self._emit(BrokerEvent("fill", cid, t["bid"], fill=fill))
                emitted += 1
            t["cum"], t["avg"] = cum, avg
        if kind in ("filled", "cancelled", "rejected", "expired"):
            if t["open"] and kind != "filled":
                self._emit(BrokerEvent(kind, cid, t["bid"], reason=f"broker status {kind}"))
                emitted += 1
            t["open"] = False
        return emitted


# ---------------------------------------------------------------------------------------------------- Alpaca
@BROKERS.register("alpaca")
class AlpacaGateway(HttpBrokerGateway):
    name = "alpaca"
    LIVE_URL = "https://api.alpaca.markets"
    PAPER_URL = "https://paper-api.alpaca.markets"
    REQUIRED = ("api_key", "api_secret")
    _TYPES = {OrderType.MARKET: "market", OrderType.LIMIT: "limit", OrderType.STOP: "stop",
              OrderType.STOP_LIMIT: "stop_limit", OrderType.TRAILING_STOP: "trailing_stop"}
    _TIF = {TimeInForce.DAY: "day", TimeInForce.GTC: "gtc", TimeInForce.IOC: "ioc", TimeInForce.FOK: "fok"}
    _STATUS = {"new": "new", "accepted": "new", "pending_new": "new", "partially_filled": "partial", "filled": "filled",
               "canceled": "cancelled", "pending_cancel": "new", "rejected": "rejected", "expired": "expired",
               "done_for_day": "expired"}

    def _headers(self, method, path, params):
        return {"APCA-API-KEY-ID": self._secret("api_key"), "APCA-API-SECRET-KEY": self._secret("api_secret")}

    def _check_account(self) -> None:
        acct = self._request("GET", "/v2/account")
        if str(acct.get("status", "ACTIVE")).upper() != "ACTIVE":
            raise BrokerConnectionError(f"alpaca account not active: {acct.get('status')}")
        self._cash = float(acct.get("cash", 0.0))

    def _submit_request(self, order: Order) -> Dict[str, Any]:
        p: Dict[str, Any] = {"symbol": order.symbol, "qty": str(order.quantity), "side": order.side.value.lower(),
                             "type": self._TYPES[order.order_type], "time_in_force": self._TIF[order.tif],
                             "client_order_id": order.client_order_id}
        if order.limit_price is not None:
            p["limit_price"] = str(order.limit_price)
        if order.stop_price is not None:
            p["stop_price"] = str(order.stop_price)
        if order.order_type is OrderType.TRAILING_STOP:
            p["trail_percent"] = str(100 * order.tags["trail_pct"]) if order.tags.get("trail_pct") else None
            if order.tags.get("trail_amount"):
                p["trail_price"] = str(order.tags["trail_amount"])
            p = {k: v for k, v in p.items() if v is not None}
        return self._call("POST", "/v2/orders", p)

    def _extract_id(self, resp, order) -> str:
        return str(resp["id"])

    def _cancel_request(self, order, bid) -> None:
        self._call("DELETE", f"/v2/orders/{bid}")

    def _status_request(self, cid, t):
        return self._call("GET", f"/v2/orders/{t['bid']}")

    def _parse_status(self, resp):
        kind = self._STATUS.get(str(resp.get("status", "new")).lower(), "new")
        return kind, float(resp.get("filled_qty") or 0.0), float(resp.get("filled_avg_price") or 0.0)

    def get_positions(self) -> Dict[str, float]:
        out = {}
        for p in self._call("GET", "/v2/positions"):
            q = float(p["qty"])
            out[p["symbol"]] = -abs(q) if str(p.get("side", "long")).lower() == "short" else q
        return out

    def get_cash(self) -> float:
        return float(self._call("GET", "/v2/account").get("cash", 0.0))


# ---------------------------------------------------------------------------------------------------- Zerodha
@BROKERS.register("zerodha")
class ZerodhaGateway(HttpBrokerGateway):
    """Kite Connect. There is NO sandbox: ``paper=True`` is refused so a paper-flagged account can never
    place a real order. Use the built-in paper broker for simulation."""
    name = "zerodha"
    LIVE_URL = "https://api.kite.trade"
    PAPER_URL = ""
    REQUIRED = ("api_key", "access_token")
    _TYPES = {OrderType.MARKET: "MARKET", OrderType.LIMIT: "LIMIT", OrderType.STOP: "SL-M", OrderType.STOP_LIMIT: "SL"}
    _STATUS = {"OPEN": "new", "TRIGGER PENDING": "new", "PUT ORDER REQ RECEIVED": "new", "VALIDATION PENDING": "new",
               "COMPLETE": "filled", "CANCELLED": "cancelled", "REJECTED": "rejected"}

    def __init__(self, credentials=None, paper: bool = True, exchange: str = "NSE", product: str = "MIS", **kw: Any) -> None:
        if paper:
            raise ConfigError("Zerodha has no sandbox; use the paper broker for simulation, or pass paper=False deliberately")
        super().__init__(credentials, paper, **kw)
        self.exchange, self.product = exchange, product

    def _headers(self, method, path, params):
        return {"X-Kite-Version": "3", "Authorization": f"token {self._secret('api_key')}:{self._secret('access_token')}"}

    def _check_account(self) -> None:
        self._request("GET", "/user/margins")

    def _submit_request(self, order: Order) -> Dict[str, Any]:
        if order.order_type not in self._TYPES:
            raise OrderRejectedError(f"order type {order.order_type.value} not supported by this adapter")
        p = {"tradingsymbol": order.symbol, "exchange": self.exchange, "transaction_type": order.side.value,
             "order_type": self._TYPES[order.order_type], "quantity": int(order.quantity), "product": self.product,
             "validity": "IOC" if order.tif in (TimeInForce.IOC, TimeInForce.FOK) else "DAY",
             "tag": order.client_order_id[-20:]}                                    # Kite tags are max 20 chars
        if order.limit_price is not None:
            p["price"] = order.limit_price
        if order.stop_price is not None:
            p["trigger_price"] = order.stop_price
        return self._call("POST", "/orders/regular", p)

    def _extract_id(self, resp, order) -> str:
        return str(resp["data"]["order_id"])

    def _cancel_request(self, order, bid) -> None:
        self._call("DELETE", f"/orders/regular/{bid}")

    def _status_request(self, cid, t):
        return self._call("GET", f"/orders/{t['bid']}")

    def _parse_status(self, resp):
        hist = resp.get("data")
        if not isinstance(hist, list) or not hist:                  # e.g. the place-order reply: {"data": {"order_id": ...}}
            return "new", 0.0, 0.0
        last = hist[-1]
        return (self._STATUS.get(str(last.get("status", "")).upper(), "new"), float(last.get("filled_quantity") or 0.0),
                float(last.get("average_price") or 0.0))

    def get_positions(self) -> Dict[str, float]:
        net = self._call("GET", "/portfolio/positions")["data"]["net"]
        return {p["tradingsymbol"]: float(p["quantity"]) for p in net if float(p["quantity"]) != 0}

    def get_cash(self) -> float:
        return float(self._call("GET", "/user/margins")["data"]["equity"]["available"]["cash"])


# ---------------------------------------------------------------------------------------------------- Binance
@BROKERS.register("binance")
class BinanceGateway(HttpBrokerGateway):
    """Binance spot. Requests are HMAC-SHA256 signed over the query string; the testnet is used when paper=True."""
    name = "binance"
    LIVE_URL = "https://api.binance.com"
    PAPER_URL = "https://testnet.binance.vision"
    REQUIRED = ("api_key", "api_secret")
    _TYPES = {OrderType.MARKET: "MARKET", OrderType.LIMIT: "LIMIT", OrderType.STOP: "STOP_LOSS",
              OrderType.STOP_LIMIT: "STOP_LOSS_LIMIT"}
    _STATUS = {"NEW": "new", "PARTIALLY_FILLED": "partial", "FILLED": "filled", "CANCELED": "cancelled",
               "PENDING_CANCEL": "new", "REJECTED": "rejected", "EXPIRED": "expired"}

    def __init__(self, credentials=None, paper: bool = True, quote_asset: str = "USDT", recv_window: int = 5000, **kw: Any) -> None:
        super().__init__(credentials, paper, **kw)
        self.quote_asset, self.recv_window = quote_asset, recv_window

    def sign(self, params: Dict[str, Any]) -> str:
        """Query string with timestamp and signature appended (signature is last, as Binance requires)."""
        q = {**params, "recvWindow": self.recv_window, "timestamp": int(self._clock() * 1000)}
        qs = urlencode(q)
        sig = hmac.new(self._secret("api_secret").encode(), qs.encode(), hashlib.sha256).hexdigest()
        return f"{qs}&signature={sig}"

    def _headers(self, method, path, params):
        return {"X-MBX-APIKEY": self._secret("api_key")}

    def _url(self, path, params) -> str:
        return f"{self.base_url}{path}?{self.sign(params or {})}"

    def _body(self, payload):
        return None                                                                  # everything travels in the signed query

    def _check_account(self) -> None:
        self._request("GET", "/api/v3/account", {})

    def _submit_request(self, order: Order) -> Dict[str, Any]:
        if order.order_type not in self._TYPES:
            raise OrderRejectedError(f"order type {order.order_type.value} not supported by this adapter")
        p: Dict[str, Any] = {"symbol": order.symbol, "side": order.side.value, "type": self._TYPES[order.order_type],
                             "quantity": order.quantity, "newClientOrderId": order.client_order_id}
        if order.limit_price is not None:
            p["price"] = order.limit_price
            p["timeInForce"] = {"GTC": "GTC", "DAY": "GTC", "IOC": "IOC", "FOK": "FOK"}[order.tif.value]
        if order.stop_price is not None:
            p["stopPrice"] = order.stop_price
        return self._call("POST", "/api/v3/order", p)

    def _extract_id(self, resp, order) -> str:
        return str(resp["orderId"])

    def _cancel_request(self, order, bid) -> None:
        self._call("DELETE", "/api/v3/order", {"symbol": order.symbol, "origClientOrderId": order.client_order_id})

    def _status_request(self, cid, t):
        return self._call("GET", "/api/v3/order", {"symbol": t["order"].symbol, "origClientOrderId": cid})

    def _parse_status(self, resp):
        kind = self._STATUS.get(str(resp.get("status", "NEW")).upper(), "new")
        cum = float(resp.get("executedQty") or 0.0)
        quote = float(resp.get("cummulativeQuoteQty") or 0.0)
        return kind, cum, (quote / cum if cum else 0.0)

    def get_positions(self) -> Dict[str, float]:
        out = {}
        for b in self._call("GET", "/api/v3/account", {})["balances"]:
            qty = float(b["free"]) + float(b["locked"])
            if qty > 0 and b["asset"] != self.quote_asset:
                out[b["asset"] + self.quote_asset] = qty
        return out

    def get_cash(self) -> float:
        for b in self._call("GET", "/api/v3/account", {})["balances"]:
            if b["asset"] == self.quote_asset:
                return float(b["free"])
        return 0.0
