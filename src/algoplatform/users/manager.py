"""User and broker-account management."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from ..core.enums import Permission, Role
from ..core.exceptions import (AuthenticationError, AuthorizationError, ConfigError,
                               UserError, ValidationError)
from ..core.logging import AuditLog, get_logger
from ..utils.helpers import IdGenerator
from .security import (CredentialVault, PasswordHasher, ROLE_PERMISSIONS, TokenService,
                       check_password_policy)

log = get_logger(__name__)


@dataclass
class User:
    user_id: str
    username: str
    password_hash: str = field(repr=False)
    role: Role = Role.TRADER
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None

    def has(self, perm: Permission) -> bool:
        return self.active and perm in ROLE_PERMISSIONS[self.role]


@dataclass
class BrokerAccount:
    account_id: str
    user_id: str
    broker: str                 # registry name, e.g. "paper", "zerodha"
    label: str
    credential_ref: str
    paper: bool = True
    active: bool = True


class UserManager:
    MAX_FAILED = 5
    LOCK_MINUTES = 15

    def __init__(self, audit: Optional[AuditLog] = None, hasher: Optional[PasswordHasher] = None,
                 tokens: Optional[TokenService] = None,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self._users: Dict[str, User] = {}
        self._by_name: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._ids = IdGenerator("USR")
        self.audit = audit or AuditLog()
        self.hasher = hasher or PasswordHasher()
        self.tokens = tokens or TokenService()
        self._now = now

    def export_users(self) -> List[Dict[str, Any]]:
        """Serialisable user records (password HASHES only, never passwords) for persistence."""
        with self._lock:
            return [{"user_id": u.user_id, "username": u.username, "password_hash": u.password_hash, "role": u.role.value,
                     "active": u.active, "created_at": u.created_at.isoformat()} for u in self._users.values()]

    def import_users(self, records: List[Dict[str, Any]]) -> int:
        with self._lock:
            for r in records:
                u = User(r["user_id"], r["username"], r["password_hash"], Role(r["role"]), bool(r["active"]),
                         datetime.fromisoformat(r["created_at"]))
                self._users[u.user_id] = u
                self._by_name[u.username] = u.user_id
            return len(records)

    def user_count(self) -> int:
        return len(self._users)

    # ---- lifecycle ------------------------------------------------------------------
    def register(self, username: str, password: str, role: Role = Role.TRADER,
                 actor: Optional[User] = None) -> User:
        username = (username or "").strip().lower()
        if not username.isidentifier() and "@" not in username:
            raise ValidationError("username must be alphanumeric/underscore or an email")
        problem = check_password_policy(password)
        if problem:
            raise ValidationError(problem)
        with self._lock:
            if role is Role.ADMIN and self._users and not (actor and actor.has(Permission.MANAGE_USERS)):
                raise AuthorizationError("only an admin can create admins")
            if self._users and role is not Role.TRADER and not (actor and actor.has(Permission.MANAGE_USERS)):
                raise AuthorizationError("only an admin can assign elevated roles")
            if username in self._by_name:
                raise UserError("username already exists", code="USER_EXISTS")
            user = User(self._ids.next(), username, self.hasher.hash(password), role)
            self._users[user.user_id] = user
            self._by_name[username] = user.user_id
        self.audit.record(actor.username if actor else "system", "user.register",
                          user=username, role=role.value)
        return user

    def authenticate(self, username: str, password: str) -> User:
        # identical error for unknown user / wrong password avoids user enumeration
        generic = AuthenticationError("invalid credentials")
        with self._lock:
            uid = self._by_name.get((username or "").strip().lower())
            user = self._users.get(uid) if uid else None
            if user is None:
                self.hasher.verify(password, self.hasher.hash("dummy-password-A1"))  # equalise timing
                raise generic
            now = self._now()
            if user.locked_until and now < user.locked_until:
                raise AuthenticationError("account locked; try later", code="ACCOUNT_LOCKED")
            if not user.active:
                raise AuthenticationError("account disabled", code="ACCOUNT_DISABLED")
            if not self.hasher.verify(password, user.password_hash):
                user.failed_attempts += 1
                if user.failed_attempts >= self.MAX_FAILED:
                    user.locked_until = now + timedelta(minutes=self.LOCK_MINUTES)
                    user.failed_attempts = 0
                    self.audit.record(user.username, "user.locked")
                self.audit.record(user.username, "user.login_failed")
                raise generic
            user.failed_attempts, user.locked_until = 0, None
        self.audit.record(user.username, "user.login")
        return user

    def login(self, username: str, password: str) -> str:
        user = self.authenticate(username, password)
        return self.tokens.issue(user.user_id, role=user.role.value)

    def user_from_token(self, token: str) -> User:
        payload = self.tokens.verify(token)
        user = self.get(payload["sub"])
        if not user.active:
            raise AuthenticationError("account disabled")
        return user

    # ---- admin ----------------------------------------------------------------------
    def get(self, user_id: str) -> User:
        try:
            return self._users[user_id]
        except KeyError:
            raise UserError(f"unknown user {user_id}", code="USER_NOT_FOUND") from None

    def set_role(self, actor: User, user_id: str, role: Role) -> None:
        self.authorize(actor, Permission.MANAGE_USERS)
        self.get(user_id).role = role
        self.audit.record(actor.username, "user.set_role", target=user_id, role=role.value)

    def deactivate(self, actor: User, user_id: str) -> None:
        self.authorize(actor, Permission.MANAGE_USERS)
        if actor.user_id == user_id:
            raise UserError("cannot deactivate yourself")
        self.get(user_id).active = False
        self.audit.record(actor.username, "user.deactivate", target=user_id)

    def list_users(self, actor: User) -> List[User]:
        self.authorize(actor, Permission.MANAGE_USERS)
        return list(self._users.values())

    @staticmethod
    def authorize(user: User, perm: Permission) -> None:
        if not user.has(perm):
            raise AuthorizationError(f"{user.username} lacks permission {perm.value}",
                                     code="FORBIDDEN", details={"permission": perm.value})


class BrokerAccountManager:
    """Links users to broker accounts and builds gateways with decrypted credentials."""

    def __init__(self, vault: Optional[CredentialVault] = None, audit: Optional[AuditLog] = None,
                 gateway_factory: Optional[Callable[[str, Dict[str, str], bool], Any]] = None) -> None:
        self.vault = vault or CredentialVault()
        self.audit = audit or AuditLog()
        self._accounts: Dict[str, BrokerAccount] = {}
        self._ids = IdGenerator("ACC")
        self._factory = gateway_factory
        self._lock = threading.RLock()

    def link(self, user: User, broker: str, credentials: Dict[str, str], label: str = "",
             paper: bool = True) -> BrokerAccount:
        UserManager.authorize(user, Permission.MANAGE_BROKERS)
        if not credentials and not paper:
            raise ValidationError("live accounts need credentials")
        with self._lock:
            acc_id = self._ids.next()
            ref = f"cred:{acc_id}"
            self.vault.put(ref, credentials)
            acc = BrokerAccount(acc_id, user.user_id, broker.lower(), label or broker, ref, paper)
            self._accounts[acc_id] = acc
        self.audit.record(user.username, "broker.link", account=acc_id, broker=broker, paper=paper)
        return acc

    def _owned(self, user: User, account_id: str) -> BrokerAccount:
        acc = self._accounts.get(account_id)
        if acc is None:
            raise UserError(f"unknown account {account_id}", code="ACCOUNT_NOT_FOUND")
        if acc.user_id != user.user_id and not user.has(Permission.MANAGE_USERS):
            raise AuthorizationError("account belongs to another user")
        return acc

    def accounts_for(self, user: User) -> List[BrokerAccount]:
        return [a for a in self._accounts.values()
                if a.active and (a.user_id == user.user_id or user.has(Permission.MANAGE_USERS))]

    def unlink(self, user: User, account_id: str) -> None:
        acc = self._owned(user, account_id)
        acc.active = False
        self.vault.delete(acc.credential_ref)
        self.audit.record(user.username, "broker.unlink", account=account_id)

    def open_gateway(self, user: User, account_id: str):
        acc = self._owned(user, account_id)
        if not acc.active:
            raise UserError("account inactive")
        if self._factory is None:
            raise ConfigError("no gateway_factory configured")
        creds = self.vault.get(acc.credential_ref)
        return self._factory(acc.broker, creds, acc.paper)
