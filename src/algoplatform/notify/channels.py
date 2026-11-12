"""Delivery channels. Each turns a :class:`Notification` into one outbound message and raises on failure:

* ``DeliveryError``      the channel could not deliver *now* (network, timeout, HTTP 429 / 5xx): worth retrying;
* ``NotificationError``  it will never work as configured (bad token, HTTP 4xx): do not retry, fix the configuration.

Network channels take their transport as a parameter, so they are tested against fakes and never touch the network in
the test-suite. Secrets (webhook URLs, bot tokens, passwords) are masked in ``repr`` and never written to logs.
"""
from __future__ import annotations

import abc
import hashlib
import hmac
import json
import os
import smtplib
import threading
import urllib.error
import urllib.request
from email.message import EmailMessage
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.exceptions import ConfigError, DeliveryError, NotificationError
from ..core.logging import get_logger
from .message import Notification

log = get_logger(__name__)
Transport = Callable[[str, Mapping[str, str], bytes, float], Tuple[int, str]]       # (url, headers, body, timeout) -> (status, text)
_LEVELS = {"INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


def urllib_transport(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> Tuple[int, str]:
    """Default HTTP POST using the standard library. Network failures become ``DeliveryError``."""
    req = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:      # noqa: S310 - scheme is validated by the channel
            return resp.status, resp.read(2048).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DeliveryError(f"network error: {type(exc).__name__}") from None       # the URL may hold a secret: keep it out of the message


def _mask(secret: str) -> str:
    return "***" if len(secret) <= 8 else f"{secret[:4]}***"


def _require_https(url: str, what: str) -> str:
    if not (url.startswith("https://") or url.startswith("http://localhost") or url.startswith("http://127.0.0.1")):
        raise ConfigError(f"{what} must be an https:// URL (plain http is accepted for localhost only)")
    return url


def _check_status(status: int, channel: str) -> None:
    if 200 <= status < 300:
        return
    if status == 429 or status >= 500:
        raise DeliveryError(f"{channel} returned HTTP {status}", details={"status": status})
    raise NotificationError(f"{channel} rejected the message (HTTP {status}); check its configuration", details={"status": status})


class Channel(abc.ABC):
    name = "channel"

    @abc.abstractmethod
    def send(self, notification: Notification) -> None:
        """Deliver or raise ``DeliveryError`` (retryable) / ``NotificationError`` (permanent)."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


class MemoryChannel(Channel):
    """Keeps the last ``maxlen`` notifications in memory: the in-app inbox, and the channel used in tests."""

    def __init__(self, name: str = "memory", maxlen: int = 1000) -> None:
        self.name, self.maxlen = name, maxlen
        self.items: List[Notification] = []
        self._lock = threading.Lock()

    def send(self, notification: Notification) -> None:
        with self._lock:
            self.items.append(notification)
            del self.items[:-self.maxlen]


class LogChannel(Channel):
    """Writes to the platform log at the level matching the severity."""

    def __init__(self, name: str = "log") -> None:
        self.name = name

    def send(self, notification: Notification) -> None:
        log.log(_LEVELS[notification.severity.value], "NOTIFY %s | %s | %s", notification.topic, notification.title, notification.body)


class FileChannel(Channel):
    """Appends one JSON line per notification: a durable local record that survives the process."""

    def __init__(self, path: str, name: str = "file") -> None:
        self.name, self.path = name, path
        self._lock = threading.Lock()

    def send(self, notification: Notification) -> None:
        try:
            with self._lock, open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(notification.to_dict(), default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError as exc:
            raise DeliveryError(f"cannot write notification file: {exc.strerror}") from exc


class WebhookChannel(Channel):
    """POSTs the notification as JSON. With ``secret`` the body is signed (HMAC-SHA256, header ``X-Signature``) so the
    receiver can verify that it came from this platform."""

    def __init__(self, url: str, name: str = "webhook", secret: Optional[str] = None, headers: Optional[Mapping[str, str]] = None,
                 transport: Transport = urllib_transport, timeout: float = 5.0) -> None:
        self.name, self._url, self._secret, self._headers = name, _require_https(url, "webhook url"), secret, dict(headers or {})
        self._transport, self.timeout = transport, timeout

    def payload(self, notification: Notification) -> Dict[str, Any]:
        return notification.to_dict()

    def send(self, notification: Notification) -> None:
        body = json.dumps(self.payload(notification), default=str, sort_keys=True).encode()
        headers = {"Content-Type": "application/json", **self._headers}
        if self._secret:
            headers["X-Signature"] = "sha256=" + hmac.new(self._secret.encode(), body, hashlib.sha256).hexdigest()
        status, _ = self._transport(self._url, headers, body, self.timeout)
        _check_status(status, self.name)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} url={_mask(self._url)}>"


class SlackChannel(WebhookChannel):
    """Slack incoming webhook."""

    def __init__(self, webhook_url: str, name: str = "slack", **kwargs: Any) -> None:
        super().__init__(webhook_url, name, **kwargs)

    def payload(self, notification: Notification) -> Dict[str, Any]:
        return {"text": notification.text()}


class TelegramChannel(Channel):
    """Telegram bot ``sendMessage``. The bot token is part of the request URL, so it is never logged or echoed."""

    def __init__(self, bot_token: str, chat_id: str, name: str = "telegram", transport: Transport = urllib_transport, timeout: float = 5.0) -> None:
        if not bot_token or not str(chat_id):
            raise ConfigError("telegram needs a bot token and a chat id")
        self.name, self._token, self._chat, self._transport, self.timeout = name, bot_token, str(chat_id), transport, timeout

    def send(self, notification: Notification) -> None:
        body = json.dumps({"chat_id": self._chat, "text": notification.text()[:4000]}).encode()
        status, _ = self._transport(f"https://api.telegram.org/bot{self._token}/sendMessage", {"Content-Type": "application/json"}, body, self.timeout)
        _check_status(status, self.name)

    def __repr__(self) -> str:
        return f"<TelegramChannel {self.name} token={_mask(self._token)}>"


class EmailChannel(Channel):
    """SMTP e-mail. ``smtp_factory(host, port, timeout)`` defaults to ``smtplib.SMTP`` and is injectable for tests."""

    def __init__(self, host: str, sender: str, recipients: Sequence[str], port: int = 587, username: Optional[str] = None,
                 password: Optional[str] = None, use_tls: bool = True, name: str = "email",
                 smtp_factory: Callable[..., Any] = smtplib.SMTP, timeout: float = 10.0) -> None:
        if not host or not sender or not recipients:
            raise ConfigError("email needs a host, a sender and at least one recipient")
        self.name, self.host, self.port, self.sender, self.recipients = name, host, port, sender, list(recipients)
        self._username, self._password, self.use_tls, self._factory, self.timeout = username, password, use_tls, smtp_factory, timeout

    def send(self, notification: Notification) -> None:
        msg = EmailMessage()
        msg["Subject"] = f"[{notification.severity.value}] {notification.title}"[:200].replace("\n", " ")
        msg["From"], msg["To"] = self.sender, ", ".join(self.recipients)
        msg.set_content(notification.body or notification.title)
        try:
            smtp = self._factory(self.host, self.port, timeout=self.timeout)
            try:
                if self.use_tls:
                    smtp.starttls()
                if self._username:
                    smtp.login(self._username, self._password or "")
                smtp.send_message(msg)
            finally:
                smtp.quit()
        except smtplib.SMTPAuthenticationError as exc:
            raise NotificationError("SMTP login rejected; check the e-mail credentials") from exc
        except smtplib.SMTPRecipientsRefused as exc:
            raise NotificationError("SMTP server refused the recipients") from exc
        except (smtplib.SMTPException, OSError) as exc:
            raise DeliveryError(f"SMTP delivery failed: {type(exc).__name__}") from exc

    def __repr__(self) -> str:
        return f"<EmailChannel {self.name} host={self.host} to={len(self.recipients)}>"
