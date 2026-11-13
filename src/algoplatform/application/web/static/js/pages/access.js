"use strict";
/* Access control: users and roles. */

// ----- access control: users and roles -----
const ROLES = ["TRADER", "QUANT", "PORTFOLIO_MANAGER", "RISK_MANAGER", "OPERATIONS", "AUDITOR", "VIEWER", "ADMIN"];
function usersView() {
  const root = h("div"), usersBox = h("div");
  const load = async () => {
    const users = await api("/api/users");
    usersBox.replaceChildren(table([{ key: "username", label: "User" }, { key: "role", label: "Role", render: (v, r) => {
      const sel = h("select", { "aria-label": "role for " + r.username }, ROLES.map((x) => h("option", { value: x }, x)));
      sel.value = v;
      sel.addEventListener("change", async () => { try { await api("/api/users/role", { user_id: r.id, role: sel.value }); toast("Role updated for " + r.username); } catch (e) { toast(e.message, true); sel.value = v; } });
      return sel; } }, { key: "active", label: "Active", render: (v) => v ? "yes" : "no" }], users));
  };
  load().catch((e) => toast(e.message, true));
  root.append(h("div", { class: "card" }, h("h2", null, "Users and roles"), usersBox,
    note("A role decides which desks and pages a user gets. Give a second account the Risk Manager role to try resetting the kill switch, or the Portfolio Manager role to approve a strategy someone else proposed. Role changes apply on the user's next request.")));
  return root;
}
