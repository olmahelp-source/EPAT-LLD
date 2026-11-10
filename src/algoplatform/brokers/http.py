"""HTTP transport abstraction so adapters can be tested without a network."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

from ..core.exceptions import BrokerConnectionError

Response = Tuple[int, Any]


class Transport(Protocol):
    def request(self, method: str, url: str, headers: Dict[str, str], body: Optional[Dict[str, Any]] = None) -> Response: ...


class UrllibTransport:
    """Default transport: stdlib urllib, JSON in and out, explicit timeout, TLS verification on."""

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout

    def request(self, method: str, url: str, headers: Dict[str, str], body: Optional[Dict[str, Any]] = None) -> Response:
        if not url.startswith("https://"):
            raise BrokerConnectionError("refusing non-HTTPS broker URL")
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:                       # noqa: S310 (https enforced)
                return r.status, _parse(r.read())
        except urllib.error.HTTPError as e:
            return e.code, _parse(e.read())
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise BrokerConnectionError(f"network error: {type(e).__name__}") from None        # no URL/secret in the message


def _parse(raw: bytes) -> Any:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        return {"raw": raw[:200].decode("utf-8", "replace")}


class FakeTransport:
    """Scriptable transport for tests. Register handlers by (method, path-prefix); every call is recorded."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._routes: List[Tuple[str, str, Union[Response, Callable[..., Response], List[Response]]]] = []

    def on(self, method: str, path: str, response: Union[Response, Callable[..., Response], List[Response]]) -> "FakeTransport":
        """``response`` may be a (status, body) tuple, a list consumed one per call (last repeats), or a callable."""
        self._routes.append((method.upper(), path, response))
        return self

    def request(self, method: str, url: str, headers: Dict[str, str], body: Optional[Dict[str, Any]] = None) -> Response:
        self.calls.append({"method": method, "url": url, "headers": dict(headers), "body": body})
        path = url.split("://", 1)[-1].split("/", 1)[-1]
        path = "/" + path
        for m, prefix, resp in reversed(self._routes):                 # latest registration wins
            if m == method.upper() and path.split("?")[0].startswith(prefix):
                if callable(resp):
                    return resp(method=method, url=url, headers=headers, body=body)
                if isinstance(resp, list):
                    return resp.pop(0) if len(resp) > 1 else resp[0]
                return resp
        return 404, {"message": f"no fake route for {method} {path}"}
