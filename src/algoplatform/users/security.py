"""Password hashing, signed tokens, secret handling, RBAC table."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Callable, Dict, FrozenSet, Optional

from ..core.enums import Permission, Role
from ..core.exceptions import AuthenticationError, ConfigError

try:  # optional dependency for encryption at rest
    from cryptography.fernet import Fernet, InvalidToken  # type: ignore
    _HAS_FERNET = True
except Exception:  # pragma: no cover - depends on environment
    Fernet = None  # type: ignore
    InvalidToken = Exception  # type: ignore
    _HAS_FERNET = False


class SecretStr:
    """Holds a secret; never reveals it through repr/str/logging."""
    __slots__ = ("_v",)

    def __init__(self, value: str) -> None:
        self._v = value

    def reveal(self) -> str:
        return self._v

    def __repr__(self) -> str:
        return "SecretStr('***')"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SecretStr) and hmac.compare_digest(self._v, other._v)

    def __hash__(self) -> int:
        return hash(("SecretStr", self._v))


class PasswordHasher:
    """scrypt with a per-password random salt; constant-time verification."""

    def __init__(self, n: int = 2 ** 14, r: int = 8, p: int = 1) -> None:
        self.n, self.r, self.p = n, r, p

    def hash(self, password: str) -> str:
        salt = os.urandom(16)
        dk = hashlib.scrypt(password.encode(), salt=salt, n=self.n, r=self.r, p=self.p, dklen=32)
        return "scrypt${}${}${}${}${}".format(self.n, self.r, self.p,
                                              base64.b64encode(salt).decode(), base64.b64encode(dk).decode())

    def verify(self, password: str, stored: str) -> bool:
        try:
            _, n, r, p, salt_b64, dk_b64 = stored.split("$")
            salt, expected = base64.b64decode(salt_b64), base64.b64decode(dk_b64)
            dk = hashlib.scrypt(password.encode(), salt=salt, n=int(n), r=int(r), p=int(p), dklen=len(expected))
            return hmac.compare_digest(dk, expected)
        except Exception:  # malformed hash is a failed login, never a crash
            return False


def check_password_policy(password: str) -> Optional[str]:
    if len(password) < 10:
        return "password must be at least 10 characters"
    if password.lower() == password or password.upper() == password:
        return "password needs upper and lower case letters"
    if not any(c.isdigit() for c in password):
        return "password needs a digit"
    return None


class TokenService:
    """Stateless HMAC-SHA256 signed tokens: ``base64(payload).signature``."""

    def __init__(self, secret: Optional[bytes] = None, ttl_seconds: int = 3600,
                 clock: Callable[[], float] = time.time) -> None:
        if secret is not None and len(secret) < 16:
            raise ConfigError("token secret must be at least 16 bytes")
        self._secret = secret or secrets.token_bytes(32)
        self.ttl = ttl_seconds
        self._clock = clock

    def issue(self, subject: str, **claims: Any) -> str:
        payload = {"sub": subject, "exp": self._clock() + self.ttl, **claims}
        body = base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True).encode()).decode()
        return f"{body}.{self._sign(body)}"

    def verify(self, token: str) -> Dict[str, Any]:
        try:
            body, sig = token.rsplit(".", 1)
        except ValueError:
            raise AuthenticationError("malformed token") from None
        if not hmac.compare_digest(sig, self._sign(body)):
            raise AuthenticationError("bad token signature")
        try:
            payload = json.loads(base64.urlsafe_b64decode(body.encode()))
        except Exception:
            raise AuthenticationError("malformed token") from None
        if payload.get("exp", 0) < self._clock():
            raise AuthenticationError("token expired")
        return payload

    def _sign(self, body: str) -> str:
        return base64.urlsafe_b64encode(hmac.new(self._secret, body.encode(), hashlib.sha256).digest()).decode()


class CredentialVault:
    """Stores broker credentials. Encrypted with Fernet when the optional
    ``cryptography`` package is installed; otherwise kept only in process memory
    (still hidden from logs via SecretStr) and never written to disk."""

    def __init__(self, key: Optional[bytes] = None) -> None:
        self._store: Dict[str, Any] = {}
        self._fernet = None
        if _HAS_FERNET:
            self._fernet = Fernet(key or Fernet.generate_key())

    @property
    def encrypted(self) -> bool:
        return self._fernet is not None

    def put(self, ref: str, credentials: Dict[str, str]) -> None:
        blob = json.dumps(credentials)
        self._store[ref] = self._fernet.encrypt(blob.encode()) if self._fernet else SecretStr(blob)

    def get(self, ref: str) -> Dict[str, str]:
        try:
            item = self._store[ref]
        except KeyError:
            raise AuthenticationError(f"no credentials for {ref}") from None
        if self._fernet:
            try:
                return json.loads(self._fernet.decrypt(item).decode())
            except InvalidToken:
                raise AuthenticationError("credential decryption failed") from None
        return json.loads(item.reveal())

    def delete(self, ref: str) -> None:
        self._store.pop(ref, None)


ROLE_PERMISSIONS: Dict[Role, FrozenSet[Permission]] = {
    Role.ADMIN: frozenset(Permission),
    Role.TRADER: frozenset({Permission.PLACE_ORDER, Permission.CANCEL_ORDER, Permission.VIEW_ORDERS,
                            Permission.MANAGE_STRATEGY, Permission.RUN_BACKTEST,
                            Permission.VIEW_PORTFOLIO, Permission.MANAGE_BROKERS,
                            Permission.TRIGGER_KILL_SWITCH}),
    Role.RISK_MANAGER: frozenset({Permission.VIEW_ORDERS, Permission.VIEW_PORTFOLIO, Permission.MANAGE_RISK,
                                  Permission.CANCEL_ORDER, Permission.TRIGGER_KILL_SWITCH,
                                  Permission.RESET_KILL_SWITCH}),
    Role.VIEWER: frozenset({Permission.VIEW_ORDERS, Permission.VIEW_PORTFOLIO}),
}
