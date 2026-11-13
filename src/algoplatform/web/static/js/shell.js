"use strict";
/* The shell: sign-in, the menu grouped by job group (PAGES), hash routing, the Sample / Real data switch, start-up.
   Loaded last, because PAGES names the page functions defined in js/pages/*.js. */

// ---------- views ----------
function renderLogin() {
  const user = h("input", { id: "u", autocomplete: "username", required: true });
  const pw = h("input", { id: "p", type: "password", autocomplete: "current-password", required: true });
  const msg = h("p", { class: "hint" }, "Passwords need 10+ characters with upper, lower and a digit. The first account created becomes the administrator.");
  const loginBtn = h("button", { class: "btn", type: "submit" }, "Sign in");
  const regBtn = h("button", { class: "btn ghost", type: "button" }, "Create account");
  const doLogin = async () => {
    const r = await api("/api/login", { username: user.value, password: pw.value });
    store.token = r.token;
    try { sessionStorage.setItem("token", r.token); } catch (_) { /* ignore */ }
    await boot();
  };
  regBtn.addEventListener("click", () => busy(regBtn, async () => {
    await api("/api/register", { username: user.value, password: pw.value });
    toast("Account created, signing in…");
    await doLogin();
  }));
  const form = h("form", { class: "card login", onsubmit: (e) => { e.preventDefault(); busy(loginBtn, doLogin); } },
    h("h1", null, "Algo Platform"), h("p", { class: "muted" }, "Backtest, optimise and paper-trade."),
    h("div", { class: "grid" }, field("Username", user), field("Password", pw)),
    h("div", { class: "row", style: "margin-top:14px" }, loginBtn, regBtn), msg);
  return form;
}

function logout(silent) {
  store.token = null; store.me = null; store.structure = null; store.sources = null;
  try { sessionStorage.removeItem("token"); } catch (_) { /* ignore */ }
  stopAuto();
  if (!silent) toast("Signed out");
  mount(renderLogin());
}

/* Pages grouped by the platform's job groups (architecture.py). Group titles and the question each group answers come
   from /api/structure, so the navigation follows the code's own structure. `perm`: holding any one of them shows the
   page; the server checks the permission again on every request, so hiding a page is a convenience, not the control. */
const PAGES = [
  { id: "home", group: "application", label: "Home", build: homeView },
  { id: "data", group: "market_data", label: "Price data", build: dataView, perm: ["backtest:run"] },
  { id: "instruments", group: "market_data", label: "Instruments", build: instrumentsView, perm: ["portfolio:view"] },
  { id: "providers", group: "market_data", label: "Data providers", build: providersView, perm: ["system:operate"] },
  { id: "lab", group: "signals", label: "Technical lab", build: labView, perm: ["backtest:run"] },
  { id: "sentiment", group: "signals", label: "News sentiment", build: sentimentView, perm: ["backtest:run"] },
  { id: "vol", group: "quant", label: "Volatility", build: volView, perm: ["backtest:run"] },
  { id: "options", group: "quant", label: "Option chain", build: optionsView, perm: ["backtest:run"] },
  { id: "models", group: "learning", label: "Prediction models", build: modelsView, perm: ["model:train"] },
  { id: "regimes", group: "learning", label: "Regimes and change points", build: regimesView, perm: ["backtest:run"] },
  { id: "builder", group: "strategy", label: "Strategy builder", build: builderView, perm: ["backtest:run"] },
  { id: "select", group: "strategy", label: "Instrument selection", build: selectView, perm: ["backtest:run"] },
  { id: "portfolio", group: "capital", label: "Multi-strategy portfolio", build: portfolioView, perm: ["backtest:run"] },
  { id: "margin", group: "capital", label: "Margin", build: marginView, perm: ["risk:view"] },
  { id: "currency", group: "capital", label: "Currency exposure", build: currencyView, perm: ["portfolio:view"] },
  { id: "live", group: "trading", label: "Paper trading", build: liveView, perm: ["order:place"] },
  { id: "orderbook", group: "trading", label: "Order-book simulator", build: orderbookView, perm: ["backtest:run"] },
  { id: "backtest", group: "validation", label: "Backtest", build: backtestView, perm: ["backtest:run"] },
  { id: "optimize", group: "validation", label: "Optimise", build: optimizeView, perm: ["backtest:run"] },
  { id: "walkforward", group: "validation", label: "Walk-forward", build: walkForwardView, perm: ["backtest:run"] },
  { id: "governance", group: "operations", label: "Strategy approvals", build: governanceView, perm: ["order:view"] },
  { id: "inbox", group: "operations", label: "Alerts and inbox", build: inboxView, perm: ["risk:view"] },
  { id: "health", group: "operations", label: "Health and jobs", build: healthView, perm: ["system:operate"] },
  { id: "audit", group: "operations", label: "Audit log", build: auditView, perm: ["audit:view"] },
  { id: "users", group: "access", label: "Users and roles", build: usersView, perm: ["users:manage"] },
];

function mount(node) { document.getElementById("root").replaceChildren(node); }

function allowedPages() {
  const perms = store.me.permissions;
  return PAGES.filter((p) => !p.perm || p.perm.some((x) => perms.includes(x)));
}

/* The job groups the user has at least one page in, in the platform's own order. */
function groupInfo(key) {
  const known = (store.structure && store.structure.groups) || [];
  return known.find((g) => g.key === key) || { key, title: key.replace(/_/g, " "), job: "", roles: [], packages: [] };
}
function navGroups(pages) {
  const order = ((store.structure && store.structure.groups) || []).map((g) => g.key);
  const rank = (k) => (order.indexOf(k) + 1) || 99;
  return [...new Set(pages.map((p) => p.group))].sort((a, b) => rank(a) - rank(b))
    .map((k) => ({ ...groupInfo(k), pages: pages.filter((p) => p.group === k) }));
}

const shell = { frame: null, pages: [], groups: [], links: new Map(), head: null, content: null, menu: null };

function renderApp() {
  const pages = allowedPages();
  const groups = navGroups(pages.filter((p) => p.group !== "application"));
  const links = new Map();
  const link = (p) => { const a = h("a", { href: "#/" + p.id, class: "navlink" }, p.label); links.set(p.id, a); return a; };
  const nav = h("nav", { class: "sidenav", id: "sidenav", "aria-label": "Platform sections" }, link(PAGES[0]),
    groups.map((g) => h("div", { class: "navgroup" }, h("div", { class: "navgroup-title", title: g.job }, g.title), g.pages.map(link))));
  const head = h("div", { class: "pagehead" }), content = h("div", { class: "page" });
  const frame = h("div", { class: "shell" }, nav, h("main", { class: "main" }, head, content));
  const menu = h("button", { class: "btn ghost menu", type: "button", "aria-controls": "sidenav", "aria-expanded": "false" }, "Menu");
  const dataSwitch = h("div", { class: "seg", role: "group", "aria-label": "data the pages start on" }, ...dataSwitchParts());
  menu.addEventListener("click", () => { const open = !frame.classList.contains("nav-open"); frame.classList.toggle("nav-open", open); menu.setAttribute("aria-expanded", String(open)); });
  Object.assign(shell, { frame, pages, groups, links, head, content, menu, dataSwitch });
  mount(h("div", null,
    h("header", { class: "top" }, h("div", { class: "bar" }, menu, h("a", { class: "brand", href: "#/home" }, "Algo Platform"), h("span", { class: "spacer" }), dataSwitch,
      h("span", { class: "who" }, store.me.username + " · " + store.me.role),
      h("button", { class: "btn ghost", onclick: () => logout(false) }, "Sign out"))),
    frame));
  showPage();
}

/* The header switch: "Sample" or "Real" data for every data form (this browser's choice). Operations can lock it to
   sample data for everyone; the server enforces that lock, the switch only shows it. */
function dataSwitchParts() {
  const locked = sampleLocked(), sample = useSample();
  const pick = (v) => () => {
    try { localStorage.setItem("sampleData", String(v)); } catch (_) { /* private mode: the switch lasts for this page only */ }
    shell.dataSwitch.replaceChildren(...dataSwitchParts());
    toast(v ? "Pages now start on sample data" : "Pages now start on real data: downloaded history or your uploads");
    showPage();
  };
  const scenarios = (store.sources && store.sources.scenarios) || {};
  const choose = h("select", { class: "seg-select", "aria-label": "sample scenario for every page", title: "Which sample data every page starts with",
    onchange: (e) => {
      try { localStorage.setItem("sampleScenario", e.target.value); } catch (_) { /* private mode */ }
      toast("Sample data: " + (e.target.value ? e.target.value.replaceAll("_", " ") : "mixed universe") + (e.target.value ? " — " + scenarios[e.target.value] : ""));
      showPage();
    } }, h("option", { value: "" }, "mixed universe"), Object.keys(scenarios).map((k) => h("option", { value: k }, k.replaceAll("_", " "))));
  choose.value = sampleScenario();
  return [h("span", { class: "seg-label" }, "Data"),
    h("button", { type: "button", "aria-pressed": String(sample), onclick: pick(true) }, "Sample"),
    sample ? choose : null,
    h("button", { type: "button", "aria-pressed": String(!sample), disabled: locked, title: locked ? "Operations has locked the platform to sample data" : "Downloaded history or your own uploads", onclick: pick(false) }, "Real"),
    locked ? h("span", { class: "badge lock", title: "Sample-data-only mode, set by operations" }, "locked") : null].filter(Boolean);
}

/* Hash routing (#/page-id): links, the back button and bookmarks all work. An unknown page, or one this role cannot
   open, falls back to Home. */
function showPage() {
  if (!store.me || !shell.frame || !shell.frame.isConnected) return;
  const id = (location.hash.match(/^#\/([\w-]+)$/) || [])[1];
  let page = shell.pages.find((p) => p.id === id);
  if (!page) {
    page = shell.pages.find((p) => p.id === "home");
    if (id) history.replaceState(null, "", "#/home");
  }
  if (page.id !== "live") stopAuto();
  for (const [pid, a] of shell.links) { if (pid === page.id) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); }
  const g = groupInfo(page.group);
  shell.head.replaceChildren(page.id === "home"
    ? h("h1", null, "One platform, grouped by job")
    : h("div", null, h("div", { class: "crumb" }, h("span", null, g.title), h("span", { class: "sep", "aria-hidden": "true" }, "›"), h("h1", null, page.label)),
      g.job ? h("p", { class: "hint" }, g.job) : null));
  shell.content.replaceChildren(page.build());
  shell.frame.classList.remove("nav-open");
  shell.menu.setAttribute("aria-expanded", "false");
  document.title = page.label + " · Algo Platform";
}
window.addEventListener("hashchange", showPage);

// ---------- boot ----------
async function boot() {
  if (!store.token) return mount(renderLogin());
  try { store.me = await api("/api/me"); }
  catch (_) { return mount(renderLogin()); }
  try { store.structure = await api("/api/structure"); } catch (_) { store.structure = null; }   // the menu still works, with plainer titles
  await refreshSources();
  renderApp();
}
boot();
