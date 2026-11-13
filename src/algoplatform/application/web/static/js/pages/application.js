"use strict";
/* Application: Home, the platform map from the signed-in user's point of view. */

// ----- home: the platform map, from the signed-in user's point of view -----
function homeView() {
  const root = h("div"), st = store.structure, role = store.me.role;
  const pages = shell.pages.filter((p) => p.id !== "home");
  root.append(h("div", { class: "kpis" }, kpi("Signed in as", store.me.username), kpi("Role", role),
    kpi("Your desks", st ? (st.desks.join(", ") || "none") : "–"), kpi("Pages you can open", pages.length),
    st ? kpi("Dependency rules", st.violations.length ? st.violations.length + " violation(s)" : "hold", st.violations.length ? "bad" : "good") : null));
  const main = (g) => (g.roles || []).includes(role);
  root.append(h("div", { class: "groupcards" }, [...shell.groups].sort((a, b) => main(b) - main(a)).map((g) => h("section", { class: "card groupcard" + (main(g) ? " main" : "") },
    h("h2", null, g.title, main(g) ? h("span", { class: "badge" }, "your role's area") : null), h("p", { class: "hint" }, g.job),
    h("div", { class: "row" }, g.pages.map((p) => h("a", { class: "btn ghost", href: "#/" + p.id }, p.label)))))));
  if (st) {
    const screens = (key) => {
      const all = PAGES.filter((p) => p.group === key && p.id !== "home"), yours = all.filter((p) => shell.links.has(p.id));
      return key === "application" ? "this web UI" : all.length ? `${yours.length} of ${all.length}` : "none: shared code";
    };
    root.append(h("div", { class: "card full" }, h("h2", null, "The whole platform"),
      table([{ key: "title", label: "Job group" }, { key: "job", label: "The question it answers", render: (v) => h("span", { class: "cell-text" }, v) },
        { key: "packages", label: "Packages", render: (v) => h("span", { class: "cell-text narrow" }, v.join(", ")) },
        { key: "roles", label: "Main roles", render: (v) => h("span", { class: "cell-text narrow" }, v.join(", ") || "everyone, through desks") },
        { key: "key", label: "Screens you can open", render: screens }], st.groups),
      note("Each package may import only what the platform's allow-list says, and a test fails if one reaches further: strategies never see orders or brokers, risk never knows which strategy sent an order, libraries stay pure computation. The menu follows the same groups; a role sees only the pages it may use, and the server checks the permission again on every request.")));
    if (st.violations.length) root.append(h("div", { class: "card" }, h("h2", null, "Violations"), h("p", { class: "code" }, st.violations.join("\n"))));
  }
  return root;
}
