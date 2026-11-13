"""Access control: sign-up, sign-in, the signed-in user, users and roles.

Every route checks the caller's permission on the server (through the role desks or ``UserManager.authorize``); hiding a
page in the browser is a convenience, not the control."""
from __future__ import annotations

from typing import Any

from ...core.enums import Permission, Role


def register(app: Any) -> None:
    r, P = app.route, app.platform

    @r("POST", "/api/register", public=True)
    def sign_up(_u, b):
        # bootstrap: on a brand-new install the first account is the administrator. Everyone after
        # that self-registers as a plain trader; only an admin can raise roles.
        with app.lock:
            role = Role.ADMIN if P.users.user_count() == 0 else Role.TRADER
            user = P.users.register(b["username"], b["password"], role)
            app._save_users()
        return {"username": user.username, "role": user.role}

    @r("POST", "/api/login", public=True)
    def login(_u, b):
        token = P.users.login(b["username"], b["password"])
        user = P.users.user_from_token(token)
        return {"token": token, "username": user.username, "role": user.role}

    @r("GET", "/api/me")
    def me(u, _b):
        return {"username": u.username, "role": u.role, "permissions": sorted(p.value for p in Permission if u.has(p))}

    @r("GET", "/api/users")
    def users(u, _b):
        return [{"id": x.user_id, "username": x.username, "role": x.role, "active": x.active}
                for x in P.users.list_users(u)]

    @r("POST", "/api/users/role")
    def set_role(u, b):
        P.users.set_role(u, b["user_id"], Role(b["role"]))
        app._save_users()
        return {"ok": True}
