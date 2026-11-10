import unittest
from datetime import datetime, timedelta, timezone

import helpers
from algoplatform.core import Permission, Role
from algoplatform.core.exceptions import (AuthenticationError, AuthorizationError, UserError, ValidationError)
from algoplatform.users import (BrokerAccountManager, CredentialVault, PasswordHasher, SecretStr, TokenService,
                                UserManager)


class UserTests(unittest.TestCase):
    def setUp(self):
        self.um = helpers.fast_user_manager()
        self.admin = self.um.register("root_admin", helpers.STRONG_PW, Role.ADMIN)   # bootstrap
        self.alice = self.um.register("alice", helpers.STRONG_PW)

    def test_password_policy_and_duplicates(self):
        with self.assertRaises(ValidationError):
            self.um.register("bob", "short")
        with self.assertRaises(UserError):
            self.um.register("alice", helpers.STRONG_PW)

    def test_cannot_self_assign_elevated_role(self):
        with self.assertRaises(AuthorizationError):
            self.um.register("mallory", helpers.STRONG_PW, Role.ADMIN)
        self.um.register("risky", helpers.STRONG_PW, Role.RISK_MANAGER, actor=self.admin)

    def test_authenticate_and_generic_errors(self):
        self.assertEqual(self.um.authenticate("ALICE", helpers.STRONG_PW).username, "alice")
        with self.assertRaises(AuthenticationError) as a:
            self.um.authenticate("alice", "wrong-password-1A")
        with self.assertRaises(AuthenticationError) as b:
            self.um.authenticate("nobody", "wrong-password-1A")
        self.assertEqual(str(a.exception), str(b.exception))      # no user enumeration

    def test_lockout(self):
        now = {"t": datetime(2024, 1, 1, tzinfo=timezone.utc)}
        um = UserManager(hasher=PasswordHasher(n=2 ** 4), now=lambda: now["t"])
        um.register("carol", helpers.STRONG_PW)
        for _ in range(UserManager.MAX_FAILED):
            with self.assertRaises(AuthenticationError):
                um.authenticate("carol", "bad-password-1A")
        with self.assertRaises(AuthenticationError) as e:
            um.authenticate("carol", helpers.STRONG_PW)            # right password, still locked
        self.assertEqual(e.exception.code, "ACCOUNT_LOCKED")
        now["t"] += timedelta(minutes=16)
        self.assertEqual(um.authenticate("carol", helpers.STRONG_PW).username, "carol")

    def test_tokens(self):
        tok = self.um.login("alice", helpers.STRONG_PW)
        self.assertEqual(self.um.user_from_token(tok).username, "alice")
        with self.assertRaises(AuthenticationError):
            self.um.user_from_token(tok[:-3] + "AAA")
        t = {"now": 1000.0}
        ts = TokenService(b"x" * 32, ttl_seconds=10, clock=lambda: t["now"])
        tk = ts.issue("u1")
        self.assertEqual(ts.verify(tk)["sub"], "u1")
        t["now"] += 11
        with self.assertRaises(AuthenticationError):
            ts.verify(tk)

    def test_rbac(self):
        viewer = self.um.register("vee", helpers.STRONG_PW, Role.VIEWER, actor=self.admin)
        self.assertTrue(viewer.has(Permission.VIEW_PORTFOLIO))
        self.assertFalse(viewer.has(Permission.PLACE_ORDER))
        with self.assertRaises(AuthorizationError):
            UserManager.authorize(viewer, Permission.PLACE_ORDER)
        with self.assertRaises(AuthorizationError):
            self.um.set_role(self.alice, viewer.user_id, Role.ADMIN)
        self.um.set_role(self.admin, viewer.user_id, Role.TRADER)
        self.assertTrue(viewer.has(Permission.PLACE_ORDER))

    def test_deactivate(self):
        self.um.deactivate(self.admin, self.alice.user_id)
        with self.assertRaises(AuthenticationError):
            self.um.authenticate("alice", helpers.STRONG_PW)
        with self.assertRaises(UserError):
            self.um.deactivate(self.admin, self.admin.user_id)

    def test_secret_str_hidden_and_vault(self):
        s = SecretStr("topsecret")
        self.assertNotIn("topsecret", repr(s) + str(s))
        v = CredentialVault()
        v.put("r", {"key": "abc"})
        self.assertEqual(v.get("r"), {"key": "abc"})
        v.delete("r")
        with self.assertRaises(AuthenticationError):
            v.get("r")


class BrokerAccountTests(unittest.TestCase):
    def setUp(self):
        self.um = helpers.fast_user_manager()
        self.admin = self.um.register("root_admin", helpers.STRONG_PW, Role.ADMIN)
        self.alice = self.um.register("alice", helpers.STRONG_PW)
        self.bob = self.um.register("bob", helpers.STRONG_PW)
        self.seen = []
        self.am = BrokerAccountManager(gateway_factory=lambda b, c, p: self.seen.append((b, c, p)) or "gw")

    def test_link_and_open(self):
        acc = self.am.link(self.alice, "paper", {"api_key": "k"})
        self.assertEqual(self.am.open_gateway(self.alice, acc.account_id), "gw")
        self.assertEqual(self.seen[0], ("paper", {"api_key": "k"}, True))

    def test_isolation_between_users(self):
        acc = self.am.link(self.alice, "paper", {"api_key": "k"})
        with self.assertRaises(AuthorizationError):
            self.am.open_gateway(self.bob, acc.account_id)
        self.assertEqual(self.am.accounts_for(self.bob), [])
        self.assertEqual(len(self.am.accounts_for(self.admin)), 1)       # admin sees all

    def test_unlink_removes_credentials(self):
        acc = self.am.link(self.alice, "paper", {"api_key": "k"})
        self.am.unlink(self.alice, acc.account_id)
        with self.assertRaises(UserError):
            self.am.open_gateway(self.alice, acc.account_id)
        with self.assertRaises(AuthenticationError):
            self.am.vault.get(acc.credential_ref)

    def test_live_needs_credentials(self):
        with self.assertRaises(ValidationError):
            self.am.link(self.alice, "zerodha", {}, paper=False)


if __name__ == "__main__":
    unittest.main()
