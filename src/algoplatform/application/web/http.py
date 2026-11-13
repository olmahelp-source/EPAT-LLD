"""HTTP middleware for the web UI: request parsing and limits, authentication of the event stream, static files,
security headers, and the threaded server.

Every request is either a static file (confined to ``static/``) or a JSON call handed to ``App.dispatch``, which checks
the bearer token and maps platform errors to HTTP status codes. Bodies are capped at ``MAX_BODY`` bytes and must be a
JSON object; every response carries CSP, nosniff, no-referrer and frame-deny headers. There is no TLS: bind to
127.0.0.1 or put a TLS reverse proxy in front.
"""
from __future__ import annotations

import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

from ...foundation.core.enums import Permission
from ...foundation.core.exceptions import AuthenticationError, PlatformError
from ...foundation.core.logging import get_logger
from ...access.users.manager import UserManager

from .common import clean


log = get_logger(__name__)
STATIC = os.path.join(os.path.dirname(__file__), "static")
MAX_BODY = 1_000_000


def make_handler(app: Any):
    class Handler(BaseHTTPRequestHandler):
        server_version = "algoplatform"

        def log_message(self, fmt, *args):          # quiet by default
            log.debug("%s " + fmt, self.address_string(), *args)

        def _headers(self, code: int, ctype: str, length: int) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        def _json(self, code: int, obj: Any) -> None:
            raw = json.dumps(obj).encode()
            self._headers(code, "application/json; charset=utf-8", len(raw))
            self.wfile.write(raw)

        def _static(self, path: str) -> None:
            rel = "index.html" if path in ("/", "") else path.lstrip("/")
            full = os.path.realpath(os.path.join(STATIC, rel))
            if not full.startswith(os.path.realpath(STATIC) + os.sep) or not os.path.isfile(full):
                return self._json(404, {"error": "NOT_FOUND", "message": "not found"})
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            with open(full, "rb") as fh:
                data = fh.read()
            self._headers(200, ctype + ("; charset=utf-8" if ctype.startswith("text") or "javascript" in ctype else ""), len(data))
            self.wfile.write(data)

        def _stream(self) -> None:
            """Server-Sent Events: push a full snapshot whenever the session changes (auth via the Authorization header,
            so the token never appears in a URL)."""
            auth = self.headers.get("Authorization", "")
            try:
                user = app.platform.users.user_from_token(auth[7:] if auth.startswith("Bearer ") else "")
                UserManager.authorize(user, Permission.VIEW_PORTFOLIO)
            except PlatformError as exc:
                return self._json(401 if isinstance(exc, AuthenticationError) else 403, {"error": exc.code, "message": exc.message})
            sess = app.sessions.get(user.user_id)
            if sess is None:
                return self._json(404, {"error": "NO_SESSION", "message": "no open session"})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'")
            self.end_headers()
            self.close_connection = True
            last, idle = -1, 0
            try:
                while not app.shutting_down:
                    with sess.cond:
                        if sess.version == last:
                            sess.cond.wait(timeout=1.0)
                        ver = sess.version
                    if app.sessions.get(user.user_id) is not sess:
                        self.wfile.write(b"event: closed\ndata: {}\n\n")
                        self.wfile.flush()
                        return
                    if ver != last:
                        payload = json.dumps(clean(app._snapshot(sess)))
                        self.wfile.write(f"event: snapshot\ndata: {payload}\n\n".encode())
                        self.wfile.flush()
                        last, idle = ver, 0
                    else:
                        idle += 1
                        if idle >= 15:
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                            idle = 0
            except (BrokenPipeError, ConnectionResetError, OSError):
                return                                                    # the browser went away

        def _handle(self, method: str) -> None:
            path = urlparse(self.path).path
            if method == "GET" and not path.startswith("/api/"):
                return self._static(path)
            if method == "GET" and path == "/api/session/stream":
                return self._stream()
            body: Dict[str, Any] = {}
            if method == "POST":
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    return self._json(400, {"error": "BAD_REQUEST", "message": "bad Content-Length"})
                if n > MAX_BODY:
                    return self._json(413, {"error": "TOO_LARGE", "message": "body too large"})
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                    if not isinstance(body, dict):
                        raise ValueError
                except ValueError:
                    return self._json(400, {"error": "BAD_REQUEST", "message": "body must be a JSON object"})
            auth = self.headers.get("Authorization", "")
            token = auth[7:] if auth.startswith("Bearer ") else None
            code, obj = app.dispatch(method, path, body, token)
            self._json(code, obj)

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")
    return Handler


def create_server(app: Any, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    """Threaded server whose worker threads never block shutdown (open event streams would otherwise)."""
    httpd = ThreadingHTTPServer((host, port), make_handler(app))
    httpd.daemon_threads = True
    httpd.block_on_close = False
    return httpd


def serve(host: str = "127.0.0.1", port: int = 8080, app: Any = None) -> None:
    if app is None:
        from .server import App                                  # deferred: the server module imports this one
        app = App()
    httpd = create_server(app, host, port)
    print(f"algoplatform UI on http://{host}:{port}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        app.close()


__all__ = ["make_handler", "create_server", "serve", "STATIC", "MAX_BODY"]
