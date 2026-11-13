"use strict";
/* Algo Platform UI: vanilla JS, no build step. All server text is inserted via textContent
   (never innerHTML), so strategy names / reject reasons cannot inject markup. */

// ---------- tiny DOM helper ----------
function h(tag, props, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "value") el.value = v;
    else if (k === "style") el.style.cssText = v;      // CSSOM assignment is allowed by our CSP; a style="" attribute is not
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
const svgNS = "http://www.w3.org/2000/svg";
function s(tag, attrs) {
  const el = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) el.setAttribute(k, v);
  return el;
}

// ---------- state & API ----------
const store = { token: null, me: null, structure: null };
try { store.token = sessionStorage.getItem("token"); } catch (_) { /* private mode */ }
let toastTimer = null;

function toast(msg, isError) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = "toast" + (isError ? " error" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, isError ? 7000 : 3000);
}

async function api(path, body) {
  const opts = { method: body === undefined ? "GET" : "POST", headers: {} };
  if (store.token) opts.headers.Authorization = "Bearer " + store.token;
  if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
  let res;
  try { res = await fetch(path, opts); } catch (_) { throw new Error("Cannot reach the server"); }
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON */ }
  if (res.status === 401 && store.token && path !== "/api/login") { logout(true); throw new Error("Session expired, please sign in again"); }
  if (!res.ok) throw new Error((data && data.message) || ("Request failed (" + res.status + ")"));
  return data;
}

// run an async action with a busy button and error toast
async function busy(btn, fn) {
  const label = btn.textContent;
  btn.disabled = true;
  btn.replaceChildren(h("span", { class: "spinner" }), "Working…");
  try { return await fn(); }
  catch (e) { toast(e.message, true); }
  finally { btn.disabled = false; btn.textContent = label; }
}

// ---------- formatting ----------
const nf = (d) => new Intl.NumberFormat(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
const fmt = {
  pct: (v, d = 2) => v == null ? "–" : nf(d).format(v * 100) + "%",
  num: (v, d = 2) => v == null ? "–" : nf(d).format(v),
  money: (v) => v == null ? "–" : (v < 0 ? "-$" : "$") + nf(2).format(Math.abs(v)),
  tone: (v) => v == null || v === 0 ? "" : (v > 0 ? "good" : "bad"),
};

// ---------- components ----------
function kpi(label, value, tone) {
  return h("div", { class: "kpi" }, h("div", { class: "l" }, label), h("div", { class: "v " + (tone || "") }, value));
}

function table(cols, rows, empty) {
  if (!rows || !rows.length) return h("p", { class: "hint" }, empty || "No data.");
  return h("div", { class: "scroll" }, h("table", null,
    h("thead", null, h("tr", null, cols.map((c) => h("th", null, c.label)))),
    h("tbody", null, rows.map((r) => h("tr", null, cols.map((c) => {
      const raw = r[c.key];
      const cell = c.render ? c.render(raw, r) : (raw == null ? "–" : raw);
      return h("td", { class: c.tone ? fmt.tone(raw) : "" }, cell);
    }))))));
}

/* series: [{name, cls, points: [[label, value], ...]}]; first series drives hover */
function lineChart(series, opts = {}) {
  const W = 800, H = opts.small ? 120 : 220, pad = 6;
  const all = series.flatMap((x) => x.points.map((p) => p[1])).filter((v) => v != null);
  if (!all.length) return h("p", { class: "hint" }, "No data yet.");
  let lo = Math.min(...all, opts.baseline ?? Infinity), hi = Math.max(...all, opts.baseline ?? -Infinity);
  if (lo === hi) { lo -= 1; hi += 1; }
  const y = (v) => H - pad - ((v - lo) / (hi - lo)) * (H - 2 * pad);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: "none", role: "img", "aria-label": opts.label || "chart" });
  svg.setAttribute("class", opts.small ? "small" : "");
  for (let i = 0; i <= 3; i++) svg.append(s("line", { x1: 0, x2: W, y1: pad + i * (H - 2 * pad) / 3, y2: pad + i * (H - 2 * pad) / 3, class: "grid-line", "vector-effect": "non-scaling-stroke" }));
  for (const ser of series) {
    const n = ser.points.length;
    if (n < 2) continue;
    const xs = (i) => pad + (i / (n - 1)) * (W - 2 * pad);
    const d = ser.points.map((p, i) => (i ? "L" : "M") + xs(i).toFixed(1) + " " + y(p[1]).toFixed(1)).join(" ");
    if (opts.fill && ser === series[0]) {
      const base = y(opts.baseline ?? lo);
      svg.append(s("path", { d: d + ` L${xs(n - 1).toFixed(1)} ${base} L${xs(0).toFixed(1)} ${base} Z`, class: ser.fcls || "f0", opacity: ".13" }));
    }
    svg.append(s("path", { d, fill: "none", "stroke-width": "1.8", "vector-effect": "non-scaling-stroke", class: ser.cls || "s0" }));
  }
  const first = series[0].points;
  const tip = h("div", { class: "tip", hidden: true });
  const wrap = h("div", { class: "chart" },
    series.length > 1 || opts.legend ? h("div", { class: "legend" }, series.map((x) => h("span", null, h("span", { class: "dot " + (x.fcls || "f0") }), x.name))) : null,
    svg, tip,
    h("div", { class: "axis" }, h("span", null, first[0][0]), h("span", null, (opts.yFmt || fmt.num)(lo) + " – " + (opts.yFmt || fmt.num)(hi)), h("span", null, first[first.length - 1][0])));
  svg.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const i = Math.max(0, Math.min(first.length - 1, Math.round(((e.clientX - r.left) / r.width) * (first.length - 1))));
    tip.hidden = false;
    tip.textContent = first[i][0] + "  " + series.map((x) => x.points[Math.min(i, x.points.length - 1)]).map((p) => (opts.yFmt || fmt.num)(p[1])).join(" · ");
  });
  svg.addEventListener("pointerleave", () => { tip.hidden = true; });
  return wrap;
}

function field(label, input) { return h("label", null, label, input); }

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
  store.token = null; store.me = null; store.structure = null;
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
  menu.addEventListener("click", () => { const open = !frame.classList.contains("nav-open"); frame.classList.toggle("nav-open", open); menu.setAttribute("aria-expanded", String(open)); });
  Object.assign(shell, { frame, pages, groups, links, head, content, menu });
  mount(h("div", null,
    h("header", { class: "top" }, h("div", { class: "bar" }, menu, h("a", { class: "brand", href: "#/home" }, "Algo Platform"), h("span", { class: "spacer" }),
      h("span", { class: "who" }, store.me.username + " · " + store.me.role),
      h("button", { class: "btn ghost", onclick: () => logout(false) }, "Sign out"))),
    frame));
  showPage();
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

// ----- strategy parameter form -----
let strategyCache = null;
async function strategies() { return strategyCache || (strategyCache = await api("/api/strategies")); }

function paramInputs(container, strat) {
  container.replaceChildren();
  const inputs = {};
  for (const [k, v] of Object.entries(strat.params)) {
    const isBool = typeof v === "boolean";
    const inp = isBool
      ? h("select", null, h("option", { value: "false" }, "false"), h("option", { value: "true" }, "true"))
      : h("input", { type: "number", step: "any", value: v, style: "width:110px" });
    if (isBool) inp.value = String(v);
    inputs[k] = { inp, isBool, def: v };
    container.append(field(k, inp));
  }
  return () => {
    const out = {};
    for (const [k, { inp, isBool }] of Object.entries(inputs)) out[k] = isBool ? inp.value === "true" : Number(inp.value);
    return out;
  };
}

function dataFields() {
  const source = h("select", { "aria-label": "data source" }, h("option", { value: "synthetic" }, "Synthetic"), h("option", { value: "uploaded" }, "Uploaded CSV"));
  const symbols = h("input", { value: "AAA,BBB,CCC", style: "width:150px", "aria-label": "symbols" });
  const bars = h("input", { type: "number", value: 756, min: 150, max: 3000, style: "width:90px" });
  const seed = h("input", { type: "number", value: 7, style: "width:80px" });
  const sync = () => { const up = source.value === "uploaded"; symbols.disabled = bars.disabled = seed.disabled = up; };
  source.addEventListener("change", sync);
  return { source, symbols, bars, seed, nodes: [field("Data", source), field("Symbols (synthetic)", symbols), field("Bars", bars), field("Seed", seed)],
    get: () => source.value === "uploaded" ? { dataset: "uploaded" }
      : { symbols: symbols.value.split(",").map((x) => x.trim()).filter(Boolean), bars: Number(bars.value), seed: Number(seed.value) } };
}

function riskCard(risk, title) {
  title = title || "Risk: value at risk and expected shortfall";
  if (!risk) return h("div", { class: "card" }, h("h2", null, title), h("p", { class: "hint" }, "Needs at least 30 return observations."));
  return h("div", { class: "card" }, h("h2", null, title),
    table([{ key: "k", label: "Measure" }, { key: "v", label: "Loss per bar" }], [
      { k: "VaR 95%", v: fmt.pct(risk.var_95) }, { k: "VaR 99%", v: fmt.pct(risk.var_99) },
      { k: "Expected shortfall 95%", v: fmt.pct(risk.es_95) }, { k: "Expected shortfall 99%", v: fmt.pct(risk.es_99) },
      { k: "Parametric VaR 99% (assumes normal)", v: fmt.pct(risk.parametric_var_99) }]),
    h("p", { class: "hint" }, `From ${risk.bars} bars. A bar worse than the 95% VaR happens about 1 time in 20; expected shortfall is the average loss on those bad bars. A parametric figure well below the historical one points to fat tails.`));
}

// ----- backtest results -----
function backtestResults(r) {
  const m = r.metrics, t = r.trade_stats;
  const out = h("div");
  if (r.kill_switch) out.append(h("div", { class: "banner halt" }, h("strong", null, "Kill switch tripped during this backtest"), " — trading was halted and positions flattened."));
  out.append(h("div", { class: "kpis" },
    kpi("Total return", fmt.pct(m.total_return), fmt.tone(m.total_return)), kpi("CAGR", fmt.pct(m.cagr), fmt.tone(m.cagr)),
    kpi("Sharpe", fmt.num(m.sharpe)), kpi("Sortino", fmt.num(m.sortino)), kpi("Max drawdown", fmt.pct(m.max_drawdown), "bad"),
    kpi("Volatility", fmt.pct(m.volatility)), kpi("Trades", t.trades ?? 0), kpi("Win rate", fmt.pct(t.win_rate, 1)),
    kpi("Profit factor", fmt.num(t.profit_factor))));
  const series = [{ name: "Strategy", cls: "s0", fcls: "f0", points: r.equity }];
  if (r.benchmark) series.push({ name: "Equal-weight buy & hold", cls: "s1", fcls: "f1", points: r.benchmark });
  out.append(h("div", { class: "card" }, h("h2", null, "Equity"), lineChart(series, { yFmt: fmt.money, label: "equity curve" }),
    h("h3", { style: "margin-top:14px" }, "Drawdown"), lineChart([{ name: "Drawdown", cls: "neg", fcls: "fneg", points: r.drawdown }], { yFmt: (v) => fmt.pct(v, 1), fill: true, baseline: 0, small: true, label: "drawdown" })));
  const sig = r.significance;
  out.append(h("div", { class: "grid cols-2" },
    h("div", { class: "card" }, h("h2", null, "Performance"), table([{ key: "k", label: "Metric" }, { key: "v", label: "Value" }],
      Object.entries(m).map(([k, v]) => ({ k: k.replace(/_/g, " "), v: typeof v === "number" ? fmt.num(v, 4) : v })))),
    h("div", { class: "card" }, h("h2", null, "Trade statistics"), table([{ key: "k", label: "Metric" }, { key: "v", label: "Value" }],
      Object.entries(t).map(([k, v]) => ({ k: k.replace(/_/g, " "), v: typeof v === "number" ? fmt.num(v, 3) : v }))),
      sig ? h("p", { class: "hint" }, `Probability the true Sharpe > 0: ${fmt.pct(sig.probabilistic_sharpe_vs_0, 1)} · deflated for ${sig.n_trials} trial(s): ${fmt.pct(sig.deflated_sharpe, 1)}`) : null)));
  out.append(riskCard(r.risk));
  out.append(h("div", { class: "card" }, h("h2", null, `Trades (last ${r.trades.length})`),
    table([{ key: "symbol", label: "Symbol" }, { key: "direction", label: "Side", render: (d) => d > 0 ? "Long" : "Short" },
      { key: "quantity", label: "Qty" }, { key: "entry_price", label: "Entry", render: (v) => fmt.num(v) }, { key: "exit_price", label: "Exit", render: (v) => fmt.num(v) },
      { key: "pnl", label: "P&L", tone: true, render: (v) => fmt.num(v) }, { key: "return_pct", label: "Return", tone: true, render: (v) => fmt.pct(v) },
      { key: "exit_ts", label: "Closed", render: (v) => String(v).slice(0, 10) }], r.trades, "No closed trades.")));
  out.append(h("p", { class: "hint" }, `${r.orders} orders sent, ${r.rejected} rejected by risk. Data is synthetic: results illustrate mechanics, not real-market performance.`));
  return out;
}

function backtestView() {
  const root = h("div"), params = h("div", { class: "row" }), results = h("div"), df = dataFields();
  const sel = h("select");
  let getParams = () => ({}), list = [];
  const run = h("button", { class: "btn", type: "submit" }, "Run backtest");
  const doc = h("p", { class: "hint" });
  const apply = () => { const st = list.find((x) => x.name === sel.value); getParams = paramInputs(params, st); doc.textContent = st.doc; };
  sel.addEventListener("change", apply);
  strategies().then((l) => { list = l.filter((x) => x.runnable); list.forEach((x) => sel.append(h("option", { value: x.name }, x.name))); apply(); }).catch((e) => toast(e.message, true));
  root.append(h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(run, async () => {
    const r = await api("/api/backtest", { strategy: sel.value, params: getParams(), ...df.get() });
    results.replaceChildren(backtestResults(r));
  }); } }, h("h2", null, "Backtest a built-in strategy"),
    h("div", { class: "row" }, field("Strategy", sel), ...df.nodes), h("div", { class: "row", style: "margin-top:10px" }, params), doc,
    h("div", { style: "margin-top:12px" }, run)), results);
  return root;
}

// ----- strategy builder -----
const OPS = ["crosses_above", "crosses_below", ">", "<", ">=", "<=", "==", "!="];
const IND_TYPES = ["sma", "ema", "wma", "rsi", "roc", "zscore", "rolling_std", "rolling_max", "rolling_min", "atr", "bollinger", "macd", "donchian", "stochastic", "adx"];
const IND_HINT = { sma: "period=20", ema: "period=20", wma: "period=20", rsi: "period=14", roc: "period=10", zscore: "period=20", rolling_std: "period=20",
  rolling_max: "period=20", rolling_min: "period=20", atr: "period=14", bollinger: "period=20,k=2,output=lower", macd: "fast=12,slow=26,signal=9,output=hist",
  donchian: "period=20,output=upper", stochastic: "k=14,d=3,output=k", adx: "period=14,output=adx" };
const PRESET = { inds: [["fast", "sma", "period=10"], ["slow", "sma", "period=40"], ["rsi", "rsi", "period=14"], ["atr", "atr", "period=14"]],
  long_entry: "fast crosses_above slow\nrsi < 70", long_exit: "fast crosses_below slow", short_entry: "", short_exit: "", stop: "atr", mult: 2.5 };

function parseArgs(text) {
  const out = {};
  for (const part of text.split(",").map((x) => x.trim()).filter(Boolean)) {
    const [k, ...rest] = part.split("=");
    if (!rest.length) throw new Error(`Indicator argument "${part}" must look like name=value`);
    const v = rest.join("=").trim();
    out[k.trim()] = v !== "" && !isNaN(Number(v)) ? Number(v) : v;
  }
  return out;
}
function parseRules(text, label) {
  return text.split("\n").map((l) => l.trim()).filter(Boolean).map((line) => {
    const tok = line.split(/\s+/);
    if (tok.length !== 3) throw new Error(`${label}: "${line}" must be three words: left operator right`);
    return tok.map((t, i) => (i !== 1 && t !== "" && !isNaN(Number(t)) ? Number(t) : t));
  });
}

function builderView() {
  const root = h("div"), results = h("div"), df = dataFields();
  const name = h("input", { value: "my_strategy", style: "width:170px" });
  const weight = h("input", { type: "number", step: "0.05", min: "0.05", max: "1", value: "1", style: "width:80px" });
  const stopInd = h("input", { placeholder: "e.g. atr", style: "width:90px" });
  const stopMult = h("input", { type: "number", step: "0.1", value: "2.5", style: "width:80px" });
  const indBox = h("div");
  const rules = { long_entry: h("textarea", { "aria-label": "long entry rules" }), long_exit: h("textarea", { "aria-label": "long exit rules" }),
    short_entry: h("textarea", { "aria-label": "short entry rules" }), short_exit: h("textarea", { "aria-label": "short exit rules" }) };
  const preview = h("div", { class: "code" }, "");
  function addInd(n = "", t = "sma", a = "") {
    const nm = h("input", { value: n, placeholder: "name", "aria-label": "indicator name" });
    const ty = h("select", { "aria-label": "indicator type" }, IND_TYPES.map((x) => h("option", { value: x }, x)));
    ty.value = t;
    const ar = h("input", { value: a, placeholder: IND_HINT[t], "aria-label": "indicator arguments" });
    ty.addEventListener("change", () => { ar.placeholder = IND_HINT[ty.value]; });
    const row = h("div", { class: "indrow" }, nm, ty, ar, h("button", { class: "btn ghost", type: "button", onclick: () => row.remove(), "aria-label": "remove indicator" }, "✕"));
    row.get = () => ({ name: nm.value.trim(), type: ty.value, args: ar.value || ar.placeholder });
    indBox.append(row);
  }
  function load(p) {
    indBox.replaceChildren();
    p.inds.forEach((i) => addInd(...i));
    for (const k of Object.keys(rules)) rules[k].value = p[k];
    stopInd.value = p.stop; stopMult.value = p.mult;
  }
  function spec() {
    const indicators = {};
    for (const row of indBox.children) {
      const { name: n, type, args } = row.get();
      if (!n) throw new Error("Every indicator needs a name");
      indicators[n] = { type, ...parseArgs(args) };
    }
    const sp = { name: name.value.trim() || "custom", params: {}, indicators, weight: Number(weight.value) };
    for (const [k, ta] of Object.entries(rules)) { const r = parseRules(ta.value, k.replace("_", " ")); if (r.length) sp[k] = r; }
    if (stopInd.value.trim()) sp.stop_atr = { indicator: stopInd.value.trim(), multiple: Number(stopMult.value) };
    return sp;
  }
  const show = h("button", { class: "btn ghost", type: "button" }, "Show JSON");
  const run = h("button", { class: "btn", type: "submit" }, "Validate & backtest");
  show.addEventListener("click", () => { try { preview.textContent = JSON.stringify(spec(), null, 2); } catch (e) { toast(e.message, true); } });
  load(PRESET);
  root.append(h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(run, async () => {
    const sp = spec();
    preview.textContent = JSON.stringify(sp, null, 2);
    const r = await api("/api/backtest", { spec: sp, ...df.get() });
    results.replaceChildren(backtestResults(r));
  }); } },
    h("h2", null, "Build a strategy without code"),
    h("p", { class: "hint" }, "Define indicators, then rules. A rule is three words: left operator right, one per line; lines in a box are AND-ed. Operands are indicator names, close/open/high/low/volume, numbers, or name@1 for the value one bar ago."),
    h("div", { class: "row" }, field("Strategy name", name), field("Position weight", weight), ...df.nodes),
    h("h3", { style: "margin-top:16px" }, "Indicators"), indBox,
    h("button", { class: "btn ghost", type: "button", onclick: () => addInd() }, "+ Add indicator"),
    h("div", { class: "grid cols-2", style: "margin-top:16px" },
      field("Long entry", rules.long_entry), field("Long exit", rules.long_exit), field("Short entry", rules.short_entry), field("Short exit", rules.short_exit)),
    h("div", { class: "row", style: "margin-top:10px" }, field("Trailing stop indicator (ATR)", stopInd), field("Stop multiple", stopMult)),
    h("p", { class: "hint" }, "Operators: " + OPS.join("  ")),
    h("div", { class: "row", style: "margin-top:12px" }, run, show, h("button", { class: "btn ghost", type: "button", onclick: () => load(PRESET) }, "Reset to example")),
    preview), results);
  return root;
}

// ----- optimiser -----
const OPT_METHODS = [["random", "Random search"], ["genetic", "Genetic algorithm"], ["bayesian", "Bayesian (Gaussian process)"], ["tpe", "Tree-structured Parzen (TPE)"],
  ["pso", "Particle swarm"], ["de", "Differential evolution"], ["annealing", "Simulated annealing"], ["cmaes", "CMA-ES"]];
function optimizeView() {
  const root = h("div"), results = h("div"), df = dataFields();
  df.bars.value = 500;
  const sel = h("select"), method = h("select", { "aria-label": "search method" }, OPT_METHODS.map(([v, l]) => h("option", { value: v }, l)));
  const n = h("input", { type: "number", value: 20, min: 5, max: 60, style: "width:80px" });
  const metric = h("select", null, ["sharpe", "sortino", "calmar", "total_return", "cagr"].map((x) => h("option", { value: x }, x)));
  const run = h("button", { class: "btn", type: "submit" }, "Optimise");
  strategies().then((l) => l.filter((x) => x.runnable && Object.keys(x.bounds).length).forEach((x) => sel.append(h("option", { value: x.name }, x.name)))).catch((e) => toast(e.message, true));
  root.append(h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(run, async () => {
    const r = await api("/api/optimize", { strategy: sel.value, method: method.value, n_iter: Number(n.value), metric: metric.value, ...df.get() });
    const keys = Object.keys(r.summary.best_params);
    const pts = r.scores.map((v, i) => [String(i + 1), v]);
    let bestSoFar = -Infinity;
    const running = pts.map(([i, v]) => { bestSoFar = Math.max(bestSoFar, v); return [i, bestSoFar]; });
    results.replaceChildren(
      h("div", { class: "kpis" }, kpi("Best " + metric.value, fmt.num(r.summary.best_score)), kpi("Trials", r.summary.trials), kpi("Failed", r.summary.failed),
        kpi("Plateau score", r.plateau == null ? "–" : fmt.num(r.plateau), r.plateau == null ? "" : (r.plateau > 0.7 ? "good" : "warn")), kpi("Seconds", fmt.num(r.summary.seconds, 1))),
      h("div", { class: "card" }, h("h2", null, "Best parameters"), h("div", { class: "code" }, JSON.stringify(r.summary.best_params, null, 2)),
        h("p", { class: "hint" }, "Plateau score near 1 means neighbouring parameters score similarly (robust); near 0 means a lone spike, a classic sign of overfitting. Test the winner out-of-sample before trusting it.")),
      h("div", { class: "card" }, h("h2", null, "Best score so far"), lineChart([{ name: "best", cls: "s0", fcls: "f0", points: running }], { small: true, label: "optimisation progress" })),
      h("div", { class: "card" }, h("h2", null, "Top trials"), table([...keys.map((k) => ({ key: k, label: k })), { key: "score", label: metric.value, render: (v) => fmt.num(v, 3) }], r.top)));
  }); } }, h("h2", null, "Optimise strategy parameters"),
    h("div", { class: "row" }, field("Strategy", sel), field("Method", method), field("Trials", n), field("Objective", metric), ...df.nodes),
    h("p", { class: "hint" }, "Each trial is a full backtest. Parameter sets that break the strategy's rules are recorded as failed trials rather than aborting the search. Random and genetic search ignore what earlier trials found; Bayesian, TPE and CMA-ES use it to choose the next trial, which usually finds a good region in fewer backtests. Population methods may run slightly fewer trials than asked, to fit whole generations."),
    h("div", { style: "margin-top:12px" }, run)), results);
  return root;
}

// ----- data upload -----
function dataView() {
  const root = h("div"), listBox = h("div");
  const file = h("input", { type: "file", accept: ".csv,text/csv", "aria-label": "CSV file" });
  const sym = h("input", { placeholder: "symbol", style: "width:110px", "aria-label": "symbol" });
  file.addEventListener("change", () => {
    if (file.files[0] && !sym.value) sym.value = file.files[0].name.replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9]/g, "").slice(0, 8).toUpperCase();
  });
  const up = h("button", { class: "btn", type: "submit" }, "Upload");
  const clear = h("button", { class: "btn ghost", type: "button" }, "Remove all my data");
  const refresh = async () => {
    const rows = await api("/api/data");
    listBox.replaceChildren(table([{ key: "symbol", label: "Symbol" }, { key: "rows", label: "Rows" }, { key: "start", label: "From", render: (v) => String(v).slice(0, 10) },
      { key: "end", label: "To", render: (v) => String(v).slice(0, 10) }], rows, "Nothing uploaded yet."));
  };
  refresh().catch((e) => toast(e.message, true));
  clear.addEventListener("click", () => busy(clear, async () => { await api("/api/data/clear", {}); await refresh(); toast("Uploaded data removed"); }));
  root.append(h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(up, async () => {
    const f = file.files[0];
    if (!f) throw new Error("Choose a CSV file first");
    if (f.size > 900000) throw new Error("File too large (limit about 900 KB, 5,000 rows)");
    const r = await api("/api/data/upload", { symbol: sym.value, csv: await f.text() });
    toast(`Uploaded ${r.symbol}: ${r.rows} rows`);
    file.value = ""; sym.value = "";
    await refresh();
  }); } },
    h("h2", null, "Upload your own price data"),
    h("p", { class: "hint" }, "CSV with a header row and columns date, open, high, low, close and optionally volume (one file per symbol, up to 5,000 rows, 10 symbols). Rows are checked for duplicate dates, gaps in ordering, non-positive prices and impossible highs/lows. Then choose \"Uploaded CSV\" as the data source on any research or validation page (Backtest, Optimise, Walk-forward, Portfolio, Technical lab and the others). Your data stays in server memory and is private to you."),
    h("div", { class: "row" }, field("File", file), field("Symbol", sym), up, clear)),
    h("div", { class: "card" }, h("h2", null, "Your uploaded data"), listBox));
  return root;
}

// ----- walk-forward -----
function walkForwardView() {
  const root = h("div"), results = h("div"), df = dataFields();
  df.bars.value = 800;
  const sel = h("select"), metric = h("select", null, ["sharpe", "sortino", "calmar", "total_return"].map((x) => h("option", { value: x }, x)));
  const train = h("input", { type: "number", value: 300, min: 50, style: "width:90px" }), test = h("input", { type: "number", value: 100, min: 10, style: "width:90px" });
  const n = h("input", { type: "number", value: 8, min: 3, max: 20, style: "width:80px" });
  const run = h("button", { class: "btn", type: "submit" }, "Run walk-forward");
  strategies().then((l) => l.filter((x) => x.runnable && Object.keys(x.bounds).length).forEach((x) => sel.append(h("option", { value: x.name }, x.name)))).catch((e) => toast(e.message, true));
  root.append(h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(run, async () => {
    const r = await api("/api/walkforward", { strategy: sel.value, metric: metric.value, train_bars: Number(train.value), test_bars: Number(test.value), n_iter: Number(n.value), ...df.get() });
    const m = r.oos_metrics;
    const stab = Object.entries(r.stability).map(([k, v]) => ({ k, v: fmt.num(v, 2), verdict: v < 0.25 ? "stable" : v < 0.6 ? "drifting" : "unstable" }));
    results.replaceChildren(
      h("div", { class: "kpis" }, kpi("Folds", r.folds.length), kpi("Walk-forward efficiency", r.efficiency == null ? "–" : fmt.num(r.efficiency, 2), r.efficiency != null && r.efficiency < 0.5 ? "warn" : ""),
        kpi("Out-of-sample return", fmt.pct(m.total_return), fmt.tone(m.total_return)), kpi("Out-of-sample Sharpe", fmt.num(m.sharpe)), kpi("OOS max drawdown", fmt.pct(m.max_drawdown), "bad")),
      h("div", { class: "card" }, h("h2", null, "Stitched out-of-sample equity"), lineChart([{ name: "OOS equity", cls: "s0", fcls: "f0", points: r.oos_equity }], { yFmt: fmt.money, label: "out-of-sample equity" }),
        h("p", { class: "hint" }, "Each fold picks parameters on its training window only, then trades the next window untouched. Efficiency is mean out-of-sample score divided by mean in-sample score: near 1 means the edge survived, well below 0.5 means the optimiser mostly fitted noise.")),
      h("div", { class: "card" }, h("h2", null, "Folds"), table([{ key: "train_start", label: "Train from" }, { key: "train_end", label: "to" }, { key: "test_start", label: "Test from" }, { key: "test_end", label: "to" },
        { key: "train_score", label: "Train score", render: (v) => fmt.num(v, 2) }, { key: "test_score", label: "Test score", tone: true, render: (v) => fmt.num(v, 2) },
        { key: "test_return", label: "Test return", tone: true, render: (v) => fmt.pct(v) }, ...r.params.map((p) => ({ key: p, label: p }))], r.folds)),
      h("div", { class: "card" }, h("h2", null, "Parameter stability"), table([{ key: "k", label: "Parameter" }, { key: "v", label: "Variation across folds" }, { key: "verdict", label: "Verdict" }], stab),
        h("p", { class: "hint" }, "Variation is the standard deviation of the chosen value divided by its mean. Parameters that jump around between folds suggest the strategy has no stable optimum.")));
  }); } }, h("h2", null, "Walk-forward analysis"),
    h("div", { class: "row" }, field("Strategy", sel), field("Objective", metric), field("Train bars", train), field("Test bars", test), field("Trials per fold", n), ...df.nodes),
    h("p", { class: "hint" }, "Total bars (above) must cover at least one train + test window. This runs trials x folds full backtests, so it can take a while."), h("div", { style: "margin-top:12px" }, run)), results);
  return root;
}

// ----- multi-strategy portfolio -----
function portfolioView() {
  const root = h("div"), results = h("div"), rowsBox = h("div"), df = dataFields();
  df.bars.value = 600;
  let list = [];
  const rows = [];
  const total = h("span", { class: "muted" });
  const recount = () => { const t = rows.reduce((a, r) => a + Number(r.pct.value || 0), 0); total.textContent = `Allocated ${fmt.num(t, 0)}% (cash ${fmt.num(Math.max(0, 100 - t), 0)}%)`; total.className = t > 100 ? "bad" : "muted"; };
  function addRow(name, pct) {
    const sel = h("select", { "aria-label": "strategy" }, list.map((x) => h("option", { value: x.name }, x.name)));
    if (name) sel.value = name;
    const pctIn = h("input", { type: "number", min: 1, max: 100, value: pct || 25, style: "width:80px", "aria-label": "allocation percent" });
    pctIn.addEventListener("input", recount);
    const row = h("div", { class: "row", style: "margin-bottom:8px" }, field("Strategy", sel), field("Capital %", pctIn),
      h("button", { class: "btn ghost", type: "button", "aria-label": "remove strategy", onclick: () => { rows.splice(rows.indexOf(entry), 1); row.remove(); recount(); } }, "✕"));
    const entry = { sel, pct: pctIn };
    rows.push(entry);
    rowsBox.append(row);
    recount();
  }
  const run = h("button", { class: "btn", type: "submit" }, "Run portfolio");
  const add = h("button", { class: "btn ghost", type: "button" }, "+ Add strategy");
  add.addEventListener("click", () => { if (rows.length >= 5) return toast("At most 5 strategies", true); addRow(); });
  strategies().then((l) => { list = l.filter((x) => x.runnable && !x.slow); addRow("sma_crossover", 40); addRow("zscore_reversion", 30); }).catch((e) => toast(e.message, true));
  const applyWeights = (w) => { Object.values(w).forEach((v, i) => { if (rows[i]) rows[i].pct.value = Math.max(1, Math.round(v * 100)); }); recount(); toast("Allocations updated: run again to compare"); };
  root.append(h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(run, async () => {
    const r = await api("/api/portfolio", { strategies: rows.map((x) => ({ name: x.sel.value, pct: Number(x.pct.value) })), ...df.get() });
    const m = r.metrics, ids = Object.keys(r.standalone);
    const sw = Object.entries(r.suggested_weights || {});
    results.replaceChildren(
      h("div", { class: "kpis" }, kpi("Total return", fmt.pct(m.total_return), fmt.tone(m.total_return)), kpi("Sharpe", fmt.num(m.sharpe)), kpi("Max drawdown", fmt.pct(m.max_drawdown), "bad"), kpi("Volatility", fmt.pct(m.volatility))),
      h("div", { class: "card" }, h("h2", null, "Combined equity"), lineChart([{ name: "Portfolio", cls: "s0", fcls: "f0", points: r.equity }], { yFmt: fmt.money }),
        h("h3", { style: "margin-top:14px" }, "Drawdown"), lineChart([{ name: "Drawdown", cls: "neg", fcls: "fneg", points: r.drawdown }], { yFmt: (v) => fmt.pct(v, 1), fill: true, baseline: 0, small: true })),
      h("div", { class: "grid cols-2" },
        h("div", { class: "card" }, h("h2", null, "Allocation vs standalone"), table([{ key: "id", label: "Strategy" }, { key: "pct", label: "Capital %" }, { key: "ret", label: "Alone: return", tone: true, render: (v) => fmt.pct(v) }, { key: "sh", label: "Alone: Sharpe", render: (v) => fmt.num(v) }],
          ids.map((id) => ({ id, pct: r.allocation[id], ret: r.standalone[id].total_return, sh: r.standalone[id].sharpe })))),
        h("div", { class: "card" }, h("h2", null, "P&L contribution (closed trades)"), table([{ key: "strategy_id", label: "Strategy" }, { key: "pnl", label: "P&L", tone: true, render: (v) => fmt.num(v) }, { key: "trades", label: "Trades" }, { key: "win_rate", label: "Win rate", render: (v) => fmt.pct(v, 0) }], r.contributions, "No closed trades."),
          h("p", { class: "hint" }, "Orders that serve several strategies at once are netted into one trade and listed under the combined name (a+b); close_out is the final liquidation at the end of the test."))),
      r.correlation ? h("div", { class: "card" }, h("h2", null, "Return correlation"), table([{ key: "id", label: "" }, ...ids.map((c) => ({ key: c, label: c, render: (v) => fmt.num(v, 2) }))], ids.map((id) => ({ id, ...r.correlation[id] }))),
        h("p", { class: "hint" }, "Low or negative correlation is what makes combining strategies worthwhile; two highly correlated strategies mostly duplicate each other.")) : null,
      sw.length ? h("div", { class: "card" }, h("h2", null, "Suggested allocations (from the standalone return streams)"), h("div", { class: "scroll" }, h("table", null,
        h("thead", null, h("tr", null, h("th", null, "Method"), ...ids.map((i) => h("th", null, i)), h("th", null, ""))),
        h("tbody", null, sw.map(([method, w]) => h("tr", null, h("td", null, method), ...ids.map((i) => h("td", null, fmt.pct(w[i], 1))),
          h("td", null, h("button", { class: "btn ghost", type: "button", onclick: () => applyWeights(w) }, "Use"))))))),
        h("p", { class: "hint" }, "Equal, inverse-volatility, risk-parity and minimum-variance weights computed in-sample on the same history, so treat them as starting points, not forecasts.")) : null,
      riskCard(r.risk, "Portfolio risk: value at risk and expected shortfall"));
  }); } }, h("h2", null, "Run several strategies together"), rowsBox, h("div", { class: "row" }, add, total),
    h("div", { class: "row", style: "margin-top:12px" }, ...df.nodes),
    h("p", { class: "hint" }, "Each strategy trades its own slice of capital through the same risk gate; opposing views on the same symbol net out before any order is sent. Parameters use each strategy's defaults."), h("div", { style: "margin-top:12px" }, run)), results);
  return root;
}

// ----- live paper trading -----
let autoTimer = null, streamCtl = null;
function stopStream() { if (streamCtl) { streamCtl.abort(); streamCtl = null; } }
function stopAuto() { if (autoTimer) { clearInterval(autoTimer); autoTimer = null; } stopStream(); }

/* Server-Sent Events over fetch: the bearer token travels in a header and never appears in a URL. */
async function startStream(onSnapshot, onClosed) {
  stopStream();
  const ctl = new AbortController();
  streamCtl = ctl;
  try {
    const res = await fetch("/api/session/stream", { headers: { Authorization: "Bearer " + store.token }, signal: ctl.signal });
    if (!res.ok || !res.body) return;
    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, i);
        buf = buf.slice(i + 2);
        let ev = "message", data = "";
        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) ev = line.slice(6).trim();
          else if (line.startsWith("data:")) data += line.slice(5).trim();
        }
        if (ev === "snapshot" && data) onSnapshot(JSON.parse(data));
        else if (ev === "closed") { onClosed(); return; }
      }
    }
  } catch (e) { /* aborted or lost: buttons still refresh the view directly */ }
}

function liveView() {
  const root = h("div");
  const show = (snap) => { stopAuto(); root.replaceChildren(snap && snap.open ? liveDashboard(snap, show) : liveOpen(show)); };
  api("/api/session").then(show).catch((e) => { toast(e.message, true); show(null); });
  return root;
}

function liveOpen(show) {
  const symbols = h("input", { value: "AAA,BBB", style: "width:150px" });
  const sel = h("select", null, h("option", { value: "" }, "None (manual trading only)"));
  strategies().then((l) => l.filter((x) => x.runnable && !x.slow && x.name !== "pairs_trading" && x.name !== "momentum_rotation")
    .forEach((x) => sel.append(h("option", { value: x.name }, x.name)))).catch(() => {});
  const btn = h("button", { class: "btn", type: "submit" }, "Open paper session");
  return h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(btn, async () => {
    show(await api("/api/session/open", { symbols: symbols.value.split(",").map((x) => x.trim()).filter(Boolean), strategy: sel.value || undefined }));
  }); } }, h("h2", null, "Paper trading session"),
    h("p", { class: "hint" }, "A simulated broker and a synthetic random-walk price feed on simulated time. Every order goes through the same risk gate, order manager and kill switch as the backtester; large orders can be worked by an execution algo. No real money or real market data is involved. With the server started using --db-dir, the session survives a restart."),
    h("div", { class: "row" }, field("Symbols", symbols), field("Run a strategy", sel)), h("div", { style: "margin-top:12px" }, btn));
}

function meter(value, limit, label) {
  const frac = Math.max(0, Math.min(1, limit ? Math.abs(Math.min(value, 0)) / limit : 0));
  const cls = frac > 0.8 ? "crit" : frac > 0.5 ? "hot" : "";
  const bar = h("span");
  bar.style.width = (frac * 100).toFixed(0) + "%";
  return h("div", null, h("div", { class: "row" }, h("span", { class: "muted" }, label), h("span", { class: "spacer" }), h("span", null, fmt.pct(value) + " of −" + fmt.pct(limit, 0) + " limit")),
    h("div", { class: "meter " + cls, role: "progressbar", "aria-valuenow": (frac * 100).toFixed(0), "aria-valuemin": 0, "aria-valuemax": 100 }, bar));
}

const OPEN_STATUSES = new Set(["ACKNOWLEDGED", "PARTIALLY_FILLED", "PENDING"]);

/* Build the controls ONCE and then only refresh the data panels, so streamed updates never wipe what the user is typing. */
function liveDashboard(initial, show) {
  const ui = {
    status: h("span"), kpis: h("div", { class: "kpis" }), equity: h("div"), prices: h("div", { class: "grid" }),
    risk: h("div"), measures: h("div"), positions: h("div"), parents: h("div"), orders: h("div"),
  };
  ui.banner = h("div", { class: "banner" });
  const update = (snap) => renderLive(ui, snap, act);
  const act = (btn, path, body) => busy(btn, async () => { update(await api(path, body)); });
  const stepBtn = (n) => { const b = h("button", { class: "btn ghost", type: "button" }, "Step +" + n); b.addEventListener("click", () => act(b, "/api/session/step", { n })); return b; };
  const auto = h("button", { class: "btn ghost", type: "button" }, "Auto-step");
  auto.addEventListener("click", () => {
    if (autoTimer) { clearInterval(autoTimer); autoTimer = null; auto.textContent = "Auto-step"; return; }
    autoTimer = setInterval(async () => { try { update(await api("/api/session/step", { n: 1 })); } catch (e) { clearInterval(autoTimer); autoTimer = null; auto.textContent = "Auto-step"; toast(e.message, true); } }, 1000);
    auto.textContent = "Stop auto";
  });
  const closeBtn = h("button", { class: "btn ghost", type: "button" }, "Close session");
  closeBtn.addEventListener("click", () => { stopAuto(); busy(closeBtn, async () => { show(await api("/api/session/close", {})); }); });
  ui.banner.append(ui.status, h("span", { class: "spacer" }), stepBtn(1), stepBtn(10), stepBtn(50), auto, closeBtn);

  // ---- order ticket (static) ----
  const sym = h("select", null, initial.symbols.map((x) => h("option", { value: x }, x)));
  const side = h("select", null, h("option", { value: "BUY" }, "Buy"), h("option", { value: "SELL" }, "Sell"));
  const qty = h("input", { type: "number", min: 1, value: 40, style: "width:90px" });
  const type = h("select", null, h("option", { value: "MARKET" }, "Market"), h("option", { value: "LIMIT" }, "Limit"));
  const lim = h("input", { type: "number", step: "0.01", placeholder: "limit price", style: "width:110px" });
  const algo = h("select", { "aria-label": "execution algo" }, h("option", { value: "" }, "Single order"), h("option", { value: "twap" }, "TWAP"), h("option", { value: "vwap" }, "VWAP"),
    h("option", { value: "pov" }, "POV"), h("option", { value: "iceberg" }, "Iceberg"), h("option", { value: "is" }, "Implementation shortfall"));
  const bars = h("input", { type: "number", min: 2, max: 50, value: 5, style: "width:70px", "aria-label": "algo bars" });
  const part = h("input", { type: "number", min: 1, max: 50, value: 10, style: "width:70px", "aria-label": "participation percent" });
  const disp = h("input", { type: "number", min: 1, value: 10, style: "width:80px", "aria-label": "iceberg display size" });
  const barsF = field("Over bars", bars), partF = field("Participation %", part), dispF = field("Display size", disp);
  const syncAlgo = () => {
    const a = algo.value;
    barsF.hidden = !["twap", "vwap", "is"].includes(a); partF.hidden = a !== "pov"; dispF.hidden = a !== "iceberg";
    type.disabled = lim.disabled = !!a;
  };
  algo.addEventListener("change", syncAlgo);
  syncAlgo();
  const send = h("button", { class: "btn", type: "submit" }, "Send order");
  const ticket = h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(send, async () => {
    const body = { symbol: sym.value, side: side.value, quantity: Number(qty.value) };
    if (algo.value) {
      body.algo = { name: algo.value };
      if (!barsF.hidden) body.algo.bars = Number(bars.value);
      if (!partF.hidden) body.algo.participation = Number(part.value) / 100;
      if (!dispF.hidden) body.algo.display = Number(disp.value);
    } else { body.type = type.value; body.limit_price = lim.value || null; }
    const r = await api("/api/session/order", body);
    if (r.parent) toast(`Algo ${r.parent.algo} started: ${r.parent.total} ${r.parent.side} ${r.parent.symbol}`);
    else toast(`Order ${r.order.status}${r.order.reason ? ": " + r.order.reason : ""}`, r.order.status === "REJECTED");
    update(r);
  }); } }, h("h2", null, "Order ticket"),
    h("div", { class: "row" }, field("Symbol", sym), field("Side", side), field("Qty", qty), field("Type", type), field("Limit", lim), field("Execution", algo), barsF, partF, dispF, send),
    h("p", { class: "hint" }, "Market orders fill at the next bar's open. An algo splits the order into child orders, each risk-checked like any other, and reports its cost versus the arrival price (TCA) below."));

  // ---- kill switch (static, so the reason box keeps its text) ----
  const reason = h("input", { placeholder: "reason (required to reset)", style: "min-width:220px" });
  const trip = h("button", { class: "btn danger", type: "button" }, "TRIP kill switch");
  const reset = h("button", { class: "btn ghost", type: "button" }, "Reset");
  trip.addEventListener("click", () => act(trip, "/api/session/killswitch", { action: "trip", reason: reason.value }));
  reset.addEventListener("click", () => act(reset, "/api/session/killswitch", { action: "reset", reason: reason.value }));
  const cancelAll = h("button", { class: "btn ghost", type: "button" }, "Cancel all orders");
  cancelAll.addEventListener("click", () => act(cancelAll, "/api/session/cancel_all", {}));

  const frame = h("div", null, ui.banner, ui.kpis,
    h("div", { class: "grid cols-2" }, h("div", { class: "card" }, h("h2", null, "Equity"), ui.equity), h("div", { class: "card" }, h("h2", null, "Prices"), ui.prices)),
    ticket,
    h("div", { class: "grid cols-2" }, h("div", { class: "card" }, h("h2", null, "Limits"), ui.risk),
      h("div", { class: "card" }, h("h2", null, "Kill switch"), h("div", { class: "row" }, reason, trip, reset, cancelAll),
        h("p", { class: "hint" }, "Tripping cancels resting orders and running algos, flattens positions and blocks new orders. Resetting needs the Risk Manager or Admin role and a written reason (a trader will be refused)."))),
    ui.measures,
    h("div", { class: "card" }, h("h2", null, "Positions"), ui.positions),
    h("div", { class: "card" }, h("h2", null, "Execution algos and cost analysis"), ui.parents),
    h("div", { class: "card" }, h("h2", null, "Orders"), ui.orders));
  update(initial);
  startStream(update, () => show({ open: false }));
  return frame;
}

function renderLive(ui, snap, act) {
  const p = snap.portfolio, ks = snap.kill_switch, halted = ks.state === "TRIGGERED";
  ui.banner.className = "banner" + (halted ? " halt" : "");
  ui.status.replaceChildren(halted ? h("strong", { class: "bad" }, "TRADING HALTED") : h("strong", { class: "good" }, "Kill switch armed"), " ",
    halted ? `${ks.reason} (${ks.source})` : `${snap.bars} bars processed`, snap.resumed ? (snap.recovery && !snap.recovery.clean ? " · resumed with a reconciliation break" : " · resumed after restart") : "");
  ui.kpis.replaceChildren(kpi("Equity", fmt.money(p.equity)), kpi("Cash", fmt.money(p.cash)), kpi("Realized P&L", fmt.money(p.realized), fmt.tone(p.realized)),
    kpi("Unrealized P&L", fmt.money(p.unrealized), fmt.tone(p.unrealized)), kpi("Gross exposure", fmt.money(p.gross)), kpi("Strategy errors", snap.strategy_errors, snap.strategy_errors ? "bad" : ""));
  ui.equity.replaceChildren(lineChart([{ name: "Equity", cls: "s0", fcls: "f0", points: snap.equity_curve }], { yFmt: fmt.money, label: "live equity" }));
  ui.prices.replaceChildren(...snap.symbols.map((x) => h("div", null, h("div", { class: "row" }, h("strong", null, x), h("span", { class: "spacer" }), h("span", null, fmt.num(snap.prices[x]))),
    lineChart([{ name: x, cls: "s0", fcls: "f0", points: snap.history[x].map((v, i) => [String(i), v]) }], { small: true, label: x + " price" }))));
  ui.risk.replaceChildren(meter(snap.risk.daily_pnl_pct, snap.risk.max_daily_loss_pct, "Daily P&L"), h("div", { style: "height:10px" }), meter(snap.risk.drawdown, snap.risk.max_drawdown_pct, "Drawdown"),
    h("p", { class: "hint" }, "Size multiplier from drawdown de-risking: " + fmt.num(snap.risk.exposure_scalar, 2) + (Object.keys(snap.risk.rejections).length ? " · rejections: " + JSON.stringify(snap.risk.rejections) : "")));

  const m = snap.measures || {};
  const stressRows = m.stress || [];
  ui.measures.replaceChildren(h("div", { class: "grid cols-2" },
    m.var ? riskCard(m.var, "Value at risk (this session)") : h("div", { class: "card" }, h("h2", null, "Value at risk (this session)"), h("p", { class: "hint" }, "Needs about 35 bars of history: use Step +50.")),
    h("div", { class: "card" }, h("h2", null, "Stress tests on current positions"),
      stressRows.length ? table([{ key: "scenario", label: "Scenario" }, { key: "pnl", label: "P&L", tone: true, render: (v) => fmt.num(v) }, { key: "pct_equity", label: "% of equity", tone: true, render: (v) => fmt.pct(v) }], stressRows)
        : h("p", { class: "hint" }, "No open positions to stress."),
      m.reverse_stress_move ? h("p", { class: "hint" }, `A uniform adverse market move of about ${fmt.pct(m.reverse_stress_move, 1)} would reach the daily loss limit at the current exposure.`) : null)));

  const posRows = Object.entries(p.positions).map(([k, v]) => ({ symbol: k, ...v }));
  ui.positions.replaceChildren(table([{ key: "symbol", label: "Symbol" }, { key: "qty", label: "Qty" }, { key: "avg", label: "Avg", render: (v) => fmt.num(v) },
    { key: "last", label: "Last", render: (v) => fmt.num(v) }, { key: "upl", label: "Unrealized", tone: true, render: (v) => fmt.num(v) }], posRows, "Flat."));

  const parents = snap.parents || [];
  const costCls = (v) => (v > 0 ? "bad" : v < 0 ? "good" : "");
  ui.parents.replaceChildren(
    parents.length ? table([{ key: "parent", label: "Parent", render: (v) => v.slice(-6) }, { key: "algo", label: "Algo" }, { key: "symbol", label: "Symbol" }, { key: "side", label: "Side" },
      { key: "filled", label: "Filled", render: (v, r) => `${fmt.num(v, 0)} / ${fmt.num(r.total, 0)}` }, { key: "status", label: "Status", render: (v) => h("span", { class: "badge " + v }, v) },
      { key: "children", label: "Child orders" }, { key: "slippage_bps", label: "Slippage (bps)", render: (v) => h("span", { class: costCls(v) }, fmt.num(v, 1)) },
      { key: "shortfall_bps", label: "Shortfall incl. fees (bps)", render: (v) => h("span", { class: costCls(v) }, fmt.num(v, 1)) }], parents)
      : h("p", { class: "hint" }, "No algo orders yet. Pick an execution algo in the order ticket."),
    snap.tca && snap.tca.parents ? h("p", { class: "hint" }, `Average shortfall ${fmt.num(snap.tca.avg_shortfall_bps, 1)} bps over ${snap.tca.parents} algo order(s). Positive means you paid more than the price when the decision was made.`) : "");

  ui.orders.replaceChildren(table([{ key: "id", label: "ID", render: (v) => v.slice(-8) }, { key: "symbol", label: "Symbol" }, { key: "side", label: "Side" }, { key: "qty", label: "Qty" },
    { key: "type", label: "Type" }, { key: "status", label: "Status", render: (v) => h("span", { class: "badge " + v }, v) }, { key: "avg", label: "Avg fill", render: (v) => v ? fmt.num(v) : "–" },
    { key: "strategy", label: "Source" }, { key: "reason", label: "Reason" },
    { key: "id", label: "", render: (v, r) => OPEN_STATUSES.has(r.status) ? h("button", { class: "btn ghost", type: "button", onclick: (e) => act(e.currentTarget, "/api/session/cancel", { order_id: v }) }, "Cancel") : "" }],
    snap.orders, "No orders yet."));
}

// ----- shared helpers for the research tabs -----
function singleSymbolFields(def) {
  const df = dataFields();
  df.symbols.value = def || "AAA";
  return df;
}
const when = (v) => v ? String(v).replace("T", " ").slice(0, 16) : "–";
const dirLabel = (d) => d > 0 ? h("span", { class: "good" }, "bullish") : d < 0 ? h("span", { class: "bad" }, "bearish") : h("span", { class: "muted" }, "neutral");
const sevBadge = (v) => h("span", { class: "badge sev-" + v }, v);
const stageBadge = (v) => h("span", { class: "badge stage-" + v }, v);
function note(text) { return h("p", { class: "hint" }, text); }

// ----- technical lab -----
const FORMULA_EXAMPLES = [
  "crossover(ema(close, 10), ema(close, 40)) and rsi(close, 14) < 70",
  "supertrend(high, low, close, 10, 3).direction > 0 and relative_volume(volume, 20) > 1.5",
  "rsi(close, 14) < 30 and close > sma(close, 200)",
  "barssince(crossover(close, shift(highest(close, 20), 1))) < 5",
  "clamp(-zscore(close, 20) / 2, -1, 1)",
  "macd(close).hist",
];

function labView() {
  const root = h("div"), out = h("div"), df = singleSymbolFields("AAA");
  const formula = h("textarea", { "aria-label": "formula", rows: 2 }, FORMULA_EXAMPLES[0]);
  const pick = h("select", { "aria-label": "example formulas", onchange: () => { if (pick.value) formula.value = pick.value; } },
    h("option", { value: "" }, "Examples…"), FORMULA_EXAMPLES.slice(0, 5).map((f) => h("option", { value: f }, f)));
  const run = h("button", { class: "btn", type: "submit" }, "Analyse");
  const catalogue = h("div");
  const details = h("details", { ontoggle: () => {
    if (!details.open || catalogue.childElementCount) return;
    api("/api/functions").then((rows) => catalogue.replaceChildren(table([{ key: "kind", label: "Kind" }, { key: "category", label: "Category" },
      { key: "signature", label: "Signature", render: (v) => h("code", { class: "cell-text" }, v) },
      { key: "doc", label: "What it does", render: (v) => h("span", { class: "cell-text" }, v || "") }], rows))).catch((e) => toast(e.message, true));
  } }, h("summary", null, "Function reference (indicators, patterns, operators)"), catalogue);
  root.append(h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(run, async () => {
    const r = await api("/api/lab/technical", { ...df.get(), formula: formula.value });
    out.replaceChildren(labResults(r));
  }); } }, h("h2", null, "Indicators, patterns and formulas"),
    h("div", { class: "row" }, ...df.nodes),
    h("div", { style: "margin-top:10px" }, field("Formula (optional): evaluated over the whole history, causally", formula)),
    h("div", { class: "row", style: "margin-top:8px" }, pick, run),
    note("Formulas use the function catalogue below plus close, open, high, low, volume, arithmetic, comparisons, and / or / not, and pick one output with a dot (macd(close).hist). They are interpreted, never executed as code. A comparison involving a value that is not yet known (indicator warm-up) is false."),
    details), out);
  return root;
}

function labResults(r) {
  const out = h("div"), ind = r.indicators;
  out.append(h("div", { class: "kpis" },
    kpi("Symbol", r.symbol), kpi("RSI(14)", fmt.num(ind.rsi, 1), ind.rsi > 70 ? "bad" : ind.rsi < 30 ? "good" : ""),
    kpi("ADX(14)", fmt.num(ind.adx, 1)), kpi("Supertrend", ind.supertrend > 0 ? "up" : ind.supertrend < 0 ? "down" : "–", fmt.tone(ind.supertrend)),
    kpi("ATR % of price", fmt.num(ind.natr, 2) + "%"), kpi("Choppiness", fmt.num(ind.choppiness, 0)), kpi("Relative volume", fmt.num(ind.relative_volume, 2))));
  out.append(h("div", { class: "card" }, h("h2", null, "Price"), lineChart([{ name: "Close", cls: "s0", fcls: "f0", points: r.close }], { label: "price" })));
  if (r.formula) {
    const f = r.formula;
    const body = f.condition
      ? h("div", null, h("p", null, `True on ${f.true_bars} of ${f.bars} bars. Most recent moments it became true:`),
          table([{ key: "date", label: "Date" }, { key: "close", label: "Close", render: (v) => fmt.num(v) }], f.events, "Never true on this data."))
      : lineChart([{ name: "Formula value", cls: "s0", fcls: "f0", points: f.series }], { small: true, label: "formula value" });
    out.append(h("div", { class: "card" }, h("h2", null, "Formula"), h("p", { class: "code" }, f.text),
      note("Reads: " + (f.columns.join(", ") || "–") + ". Calls: " + (f.functions.join(", ") || "–") + "."), body));
  }
  out.append(h("div", { class: "grid cols-2" },
    h("div", { class: "card" }, h("h2", null, "Chart patterns (most recent first)"),
      table([{ key: "date", label: "Confirmed" }, { key: "name", label: "Pattern", render: (v) => v.replaceAll("_", " ") }, { key: "direction", label: "Bias", render: dirLabel },
        { key: "level", label: "Level", render: (v) => fmt.num(v) }, { key: "target", label: "Target", render: (v) => fmt.num(v) }], r.chart_patterns, "No pattern confirmed."),
      note("A pattern is listed on the bar where it became known (the neckline break, for reversals), never earlier. A detection describes the past; it is not a forecast.")),
    h("div", { class: "card" }, h("h2", null, "Candlesticks, last 20 bars"),
      table([{ key: "pattern", label: "Pattern", render: (v) => v.replaceAll("_", " ") }, { key: "count", label: "Times" },
        { key: "last_signal", label: "Latest", render: (v) => v > 0 ? "bullish / present" : "bearish" }], r.candles, "None."),
      note("Net candlestick score over the window: " + r.candle_score + " (bullish minus bearish signals)."),
      h("h3", { style: "margin-top:14px" }, "Support and resistance"),
      table([{ key: "price", label: "Price", render: (v) => fmt.num(v) }, { key: "kind", label: "Kind" }, { key: "touches", label: "Touches" }], r.levels, "No level touched twice."))));
  return out;
}

// ----- volatility -----
function volView() {
  const root = h("div"), out = h("div"), df = singleSymbolFields("AAA");
  const model = h("select", { "aria-label": "forecast model" }, ["ewma", "garch", "har", "hist"].map((m) => h("option", { value: m }, m.toUpperCase())));
  const target = h("input", { type: "number", value: 10, min: 1, max: 60, step: 1, style: "width:80px" });
  const run = h("button", { class: "btn", type: "submit" }, "Assess");
  root.append(h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(run, async () => {
    const r = await api("/api/lab/volatility", { ...df.get(), model: model.value, target_vol: Number(target.value) / 100 });
    out.replaceChildren(volResults(r));
  }); } }, h("h2", null, "Volatility: estimate, forecast, size"),
    h("div", { class: "row" }, ...df.nodes, field("Forecast model", model), field("Target volatility %", target), run),
    note("The exposure multiplier is target / forecast volatility, capped at 2x, and halved again when volatility sits in the top fifth of its own history. The model league refits each model on the past only and scores its out-of-sample forecasts.")), out);
  return root;
}

function volResults(r) {
  const a = r.assessment, out = h("div");
  out.append(h("div", { class: "kpis" }, kpi("Symbol", r.symbol), kpi("Forecast volatility", fmt.pct(a.forecast_vol, 1)),
    kpi("Regime", a.regime, a.regime === "high" ? "bad" : a.regime === "low" ? "good" : ""), kpi("Percentile", fmt.pct(a.percentile, 0)),
    kpi("Exposure multiplier", fmt.num(a.scalar, 2) + "x"), kpi("Model", a.model.toUpperCase())));
  out.append(h("div", { class: "card" }, h("h2", null, "Rolling volatility (annualised)"),
    lineChart([{ name: "Yang-Zhang 20-bar", cls: "s0", fcls: "f0", points: r.series.yang_zhang }, { name: "EWMA", cls: "s1", fcls: "f1", points: r.series.ewma }], { yFmt: (v) => fmt.pct(v, 0), label: "volatility" })));
  out.append(h("div", { class: "grid cols-2" },
    h("div", { class: "card" }, h("h2", null, "Estimators (latest)"), table([{ key: "name", label: "Estimator", render: (v) => v.replaceAll("_", " ") }, { key: "value", label: "Annualised", render: (v) => fmt.pct(v, 1) }], r.estimators),
      note("Range-based estimators (Parkinson onward) use the high and low as well as the close, so they need far fewer bars for the same precision. Yang-Zhang also handles overnight gaps.")),
    h("div", { class: "card" }, h("h2", null, "Out-of-sample model league"),
      r.league ? table([{ key: "model", label: "Model" }, { key: "qlike", label: "QLIKE (lower is better)", render: (v) => fmt.num(v, 3) }, { key: "mz_r2", label: "R²", render: (v) => fmt.num(v, 2) },
        { key: "forecasts", label: "Forecasts" }, { key: "error", label: "Error", render: (v) => v || "" }], r.league) : note("Needs more history (about 310 bars or more)."))));
  if (r.cone.length) out.append(h("div", { class: "card" }, h("h2", null, "Volatility cone"),
    table([{ key: "window", label: "Window (bars)" }, ...["min", "q25", "q50", "q75", "max", "current"].map((k) => ({ key: k, label: k, render: (v) => fmt.pct(v, 1) })),
      { key: "percentile", label: "Current percentile", render: (v) => fmt.pct(v, 0) }], r.cone),
    note("For each horizon, the range realised volatility has covered in this history and where it stands now. Compare an option's implied volatility with the matching row.")));
  return out;
}

// ----- selection -----
function selectView() {
  const root = h("div"), out = h("div"), df = dataFields();
  df.symbols.value = "AAA,BBB,CCC,DDD,EEE,FFF,GGG,HHH";
  df.symbols.style.width = "260px";
  const rows = h("div");
  const addRanker = (name, weight) => {
    const sel = h("select", { "aria-label": "ranker" });
    api("/api/lab/options").then((o) => { o.rankers.forEach((x) => sel.append(h("option", { value: x }, x))); sel.value = name; }).catch(() => {});
    const w = h("input", { type: "number", value: weight, step: 0.25, min: 0.25, style: "width:80px", "aria-label": "weight" });
    const row = h("div", { class: "row", style: "margin-bottom:6px" }, field("Rank by", sel), field("Weight", w),
      h("button", { class: "btn ghost", type: "button", onclick: () => row.remove() }, "Remove"));
    row.get = () => ({ name: sel.value, weight: Number(w.value) });
    rows.append(row);
  };
  addRanker("momentum", 1); addRanker("low_volatility", 0.5);
  const top = h("input", { type: "number", value: 3, min: 1, max: 10, style: "width:70px" });
  const corr = h("input", { type: "number", value: 0.8, min: 0, max: 1, step: 0.05, style: "width:80px" });
  const run = h("button", { class: "btn", type: "submit" }, "Select");
  root.append(h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(run, async () => {
    const r = await api("/api/lab/select", { ...df.get(), rankers: [...rows.children].map((x) => x.get()), top: Number(top.value), max_correlation: corr.value === "" ? null : Number(corr.value) });
    out.replaceChildren(selectResults(r));
  }); } }, h("h2", null, "Choose what to trade"), h("div", { class: "row" }, ...df.nodes), h("div", { style: "margin-top:10px" }, rows),
    h("div", { class: "row" }, h("button", { class: "btn ghost", type: "button", onclick: () => addRanker("trend_quality", 1) }, "Add a ranking"),
      field("How many", top), field("Max correlation between picks", corr), run),
    note("Each ranking turns into a percentile; the composite is their weighted average, so rankings in different units can be combined. Walking down the list, a candidate that correlates above the limit with one already picked is skipped. Every instrument left out is listed with the reason.")), out);
  return root;
}

function selectResults(r) {
  const out = h("div"), syms = Object.keys(r.correlation || {});
  const keys = r.scores.length ? Object.keys(r.scores[0]).filter((k) => k !== "symbol") : [];
  out.append(h("div", { class: "kpis" }, kpi("Selected", r.selected.join(", ") || "none"),
    kpi("Independent bets", r.concentration ? fmt.num(r.concentration.effective_bets, 1) : "–"),
    kpi("Relationships found", Object.entries(r.relationships).map(([k, v]) => `${v} ${k}`).join(", ") || "none")));
  out.append(h("div", { class: "card" }, h("h2", null, "Scores"),
    table([{ key: "symbol", label: "Symbol", render: (v) => r.selected.includes(v) ? h("strong", null, v + " ✓") : v },
      ...keys.map((k) => ({ key: k, label: k.replaceAll("_", " "), render: (v) => fmt.num(v, 3) }))], r.scores),
    h("h3", { style: "margin-top:14px" }, "Left out"), table([{ key: "symbol", label: "Symbol" }, { key: "reason", label: "Reason" }], r.rejected, "Nothing left out.")));
  out.append(h("div", { class: "grid cols-2" },
    h("div", { class: "card" }, h("h2", null, "Cointegrated pairs"), table([{ key: "a", label: "Y" }, { key: "b", label: "X" }, { key: "stat", label: "Test statistic", render: (v) => fmt.num(v, 2) },
      { key: "hedge_ratio", label: "Hedge ratio", render: (v) => fmt.num(v, 3) }, { key: "half_life", label: "Half-life (bars)", render: (v) => fmt.num(v, 1) }], r.pairs, "None at 5%."),
      note("Testing many pairs at 5% finds about 5% by luck: treat these as candidates to validate out of sample."),
      h("h3", { style: "margin-top:14px" }, "Who moves first"), table([{ key: "leader", label: "Leader" }, { key: "follower", label: "Follower" }, { key: "lag", label: "Lag (bars)" },
        { key: "corr", label: "Correlation", render: (v) => fmt.num(v, 2) }], r.leads, "No significant lead-lag.")),
    h("div", { class: "card" }, h("h2", null, "Return correlation"), table([{ key: "id", label: "" }, ...syms.map((c) => ({ key: c, label: c, render: (v) => fmt.num(v, 2) }))],
      syms.map((id) => ({ id, ...r.correlation[id] }))))));
  return out;
}

// ----- governance -----
function governanceView() {
  const root = h("div"), listBox = h("div"), forms = h("div");
  let state = null;
  const load = async () => {
    state = await api("/api/governance");
    const c = state.can;
    const act = async (btn, body) => busy(btn, async () => { await api("/api/governance/decide", body); toast(body.strategy_id + ": " + body.action + " done"); await load(); });
    const reasonFor = (label) => window.prompt(label);
    listBox.replaceChildren(table([{ key: "strategy_id", label: "Strategy" }, { key: "owner", label: "Owner" }, { key: "stage", label: "Stage", render: stageBadge },
      { key: "since", label: "Since", render: when },
      { key: "pending", label: "Pending request", render: (p) => p ? h("span", null, `→ ${p.target} by ${p.by}, grade ${p.grade}`) : "–" },
      { key: "strategy_id", label: "", render: (id, row) => h("div", { class: "row" },
        row.pending && c.approve ? h("button", { class: "btn", type: "button", onclick: (e) => act(e.currentTarget, { strategy_id: id, action: "approve", reason: "approved in UI" }) }, "Approve") : null,
        row.pending && c.approve ? h("button", { class: "btn ghost", type: "button", onclick: (e) => { const why = reasonFor("Reason for rejecting " + id); if (why) act(e.currentTarget, { strategy_id: id, action: "reject", reason: why }); } }, "Reject") : null,
        row.stage === "LIVE" && c.demote ? h("button", { class: "btn danger", type: "button", onclick: (e) => { const why = reasonFor("Reason for taking " + id + " back to paper"); if (why) act(e.currentTarget, { strategy_id: id, action: "demote", reason: why }); } }, "Demote") : null) }],
      state.strategies, "No strategy registered yet."),
      state.strategies.some((x) => x.history.length) ? h("details", null, h("summary", null, "Decision history"),
        table([{ key: "strategy", label: "Strategy" }, { key: "ts", label: "When", render: when }, { key: "actor", label: "By" }, { key: "action", label: "Action" },
          { key: "to", label: "Stage" }, { key: "reason", label: "Reason" }], state.strategies.flatMap((x) => x.history.map((e) => ({ strategy: x.strategy_id, ...e }))))) : "");
    if (!forms.childElementCount) renderForms();          // built once, so a reload keeps what the user typed
  };
  const renderForms = () => {
    forms.replaceChildren();
    const c = state.can;
    if (!c.register && !c.propose) return;
    const sid = h("input", { value: "my_strategy", style: "width:160px", "aria-label": "strategy id" });
    const sel = h("select"), params = h("div", { class: "row" }), df = dataFields(), trials = h("input", { type: "number", value: 1, min: 1, style: "width:80px" });
    df.symbols.value = "AAA";
    let getParams = () => ({}), list = [];
    const apply = () => { getParams = paramInputs(params, list.find((x) => x.name === sel.value)); };
    sel.addEventListener("change", apply);
    strategies().then((l) => { list = l.filter((x) => x.runnable); list.forEach((x) => sel.append(h("option", { value: x.name }, x.name))); apply(); });
    const card = h("div");
    const body = () => ({ strategy_id: sid.value.trim(), strategy: sel.value, params: getParams(), n_trials: Number(trials.value), ...df.get() });
    const reg = h("button", { class: "btn ghost", type: "button", onclick: () => busy(reg, async () => { await api("/api/governance/register", body()); toast("Registered " + sid.value); await load(); }) }, "Register");
    const evalBtn = h("button", { class: "btn ghost", type: "button", onclick: () => busy(evalBtn, async () => { card.replaceChildren(scorecard(await api("/api/governance/evaluate", body()))); }) }, "Score it");
    const prop = h("button", { class: "btn", type: "button", onclick: () => busy(prop, async () => { const r = await api("/api/governance/propose", body()); card.replaceChildren(scorecard(r.verdict)); toast("Proposed: waiting for a second person to approve"); await load(); }) }, "Propose for next stage");
    forms.append(h("div", { class: "card" }, h("h2", null, "Register, score and propose"),
      h("div", { class: "row" }, field("Strategy id", sid), field("Strategy", sel), ...df.nodes, field("Variants tried", trials)),
      h("div", { class: "row", style: "margin-top:10px" }, params), h("div", { class: "row", style: "margin-top:10px" }, c.register ? reg : null, evalBtn, c.propose ? prop : null),
      note(`Promotion needs a passing scorecard and a different person's approval; going live also needs ${state.min_paper_days} days in paper trading. "Variants tried" is how many parameter sets or versions were tested to arrive at this one: the bar for skill rises with it.`), card));
  };
  load().catch((e) => toast(e.message, true));
  root.append(h("div", { class: "card" }, h("h2", null, "Strategies: research → paper → live"), listBox), forms);
  return root;
}

function scorecard(v) {
  return h("div", { style: "margin-top:12px" },
    h("div", { class: "kpis" }, kpi("Grade", v.grade, v.passed ? "good" : "bad"), kpi("Passed", v.passed ? "yes" : "no", v.passed ? "good" : "bad"), kpi("Checks passed", Math.round(v.score * v.checks.length) + " / " + v.checks.length)),
    table([{ key: "name", label: "Check", render: (x) => x.replaceAll("_", " ") }, { key: "passed", label: "Result", render: (x) => x ? h("span", { class: "good" }, "pass") : h("span", { class: "bad" }, "fail") },
      { key: "detail", label: "Detail" }], v.checks));
}

// ----- alerts & inbox -----
function inboxView() {
  const root = h("div");
  const load = async () => {
    const r = await api("/api/inbox");
    const ack = (btn, id) => busy(btn, async () => { await api("/api/inbox/ack", { id }); toast("Acknowledged"); await load(); });
    const remove = (btn, name) => busy(btn, async () => { await api("/api/alerts/remove", { name }); toast("Removed " + name); await load(); });
    const noteCols = [{ key: "ts", label: "When", render: (v) => new Date(v * 1000).toLocaleString() }, { key: "severity", label: "Severity", render: sevBadge },
      { key: "title", label: "Title" }, { key: "body", label: "Detail", render: (v) => (v || "").slice(0, 80) }];
    const parts = [];
    if (r.open_critical.length) parts.push(h("div", { class: "banner halt" }, h("strong", null, r.open_critical.length + " critical alert(s) not yet acknowledged"),
      table([...noteCols, { key: "id", label: "", render: (id) => r.can_ack ? h("button", { class: "btn", type: "button", onclick: (e) => ack(e.currentTarget, id) }, "Acknowledge") : "" }], r.open_critical),
      r.can_ack ? null : note("A risk manager or operations user must acknowledge these; until then they are re-sent.")));
    parts.push(h("div", { class: "card" }, h("h2", null, "Alert rules"),
      table([{ key: "name", label: "Rule" }, { key: "condition", label: "Condition" }, { key: "personal", label: "Scope", render: (v) => v ? "personal" : "firm-wide" },
        { key: "severity", label: "Severity", render: sevBadge }, { key: "state", label: "State" }, { key: "fired", label: "Fired" },
        { key: "name", label: "", render: (n, row) => (row.personal || r.can_add.firm) ? h("button", { class: "btn ghost", type: "button", onclick: (e) => remove(e.currentTarget, n) }, "Remove") : "" }],
        r.alerts, "No alert rules yet."),
      note("Rules are checked after every bar of your paper-trading session. A rule fires once, repeats at most every 5 minutes while it holds, and reports when it clears.")));
    if (r.can_add.personal || r.can_add.firm) parts.push(alertForm(r, load));
    parts.push(h("div", { class: "grid cols-2" },
      h("div", { class: "card" }, h("h2", null, "Your notifications"), table(noteCols, r.personal, "Nothing for you yet.")),
      h("div", { class: "card" }, h("h2", null, "Platform notifications"), table(noteCols, r.platform, "Nothing yet."),
        r.undelivered ? note(r.undelivered + " notification(s) could not be delivered to an external channel; operations can replay them.") : null)));
    root.replaceChildren(...parts);
  };
  load().catch((e) => toast(e.message, true));
  return root;
}

function alertForm(r, reload) {
  const kind = h("select", null, h("option", { value: "price" }, "Price crosses a level"), h("option", { value: "metric" }, "Risk metric"));
  const symbol = h("input", { value: "AAA", style: "width:90px" }), level = h("input", { type: "number", value: 100, step: "any", style: "width:100px" });
  const direction = h("select", null, h("option", { value: "above" }, "above"), h("option", { value: "below" }, "below"));
  const metric = h("select", null, r.metrics.map((m) => h("option", { value: m }, m))), op = h("select", null, [">", ">=", "<", "<="].map((o) => h("option", { value: o }, o)));
  const value = h("input", { type: "number", value: 0.1, step: "any", style: "width:100px" });
  const severity = h("select", null, ["INFO", "WARNING", "ERROR", "CRITICAL"].map((x) => h("option", { value: x }, x)));
  severity.value = "WARNING";
  const scope = h("select", null, r.can_add.personal ? h("option", { value: "personal" }, "Just me") : null, r.can_add.firm ? h("option", { value: "firm" }, "Firm-wide") : null);
  const priceBox = h("div", { class: "row" }, field("Symbol", symbol), field("Direction", direction), field("Level", level));
  const metricBox = h("div", { class: "row", hidden: true }, field("Metric", metric), field("Comparison", op), field("Value", value));
  kind.addEventListener("change", () => { priceBox.hidden = kind.value !== "price"; metricBox.hidden = kind.value === "price"; });
  const add = h("button", { class: "btn", type: "submit" }, "Add rule");
  return h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(add, async () => {
    const body = { kind: kind.value, severity: severity.value, scope: scope.value };
    if (kind.value === "price") Object.assign(body, { symbol: symbol.value, level: Number(level.value), direction: direction.value });
    else Object.assign(body, { metric: metric.value, op: op.value, value: Number(value.value) });
    const res = await api("/api/alerts/add", body);
    toast("Added " + res.name);
    await reload();
  }); } }, h("h2", null, "Add an alert"), h("div", { class: "row" }, field("Type", kind), field("Severity", severity), field("Who is told", scope)),
    h("div", { style: "margin-top:10px" }, priceBox, metricBox), h("div", { style: "margin-top:10px" }, add),
    note("Drawdown and daily P&L are fractions (0.1 = 10%). A personal alert reaches only you; a firm-wide one goes to the platform inbox and every channel operations has configured."));
}

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

// ----- operations: audit log -----
function auditView() {
  const root = h("div"), box = h("div"), filter = h("input", { placeholder: "filter by actor or action", "aria-label": "filter audit log", style: "min-width:220px" });
  let rows = [];
  const draw = () => {
    const q = filter.value.trim().toLowerCase();
    box.replaceChildren(table([{ key: "ts", label: "Time", render: (v) => String(v).replace("T", " ").slice(0, 19) }, { key: "actor", label: "Actor" }, { key: "action", label: "Action" },
      { key: "details", label: "Details", render: (v) => h("span", { class: "cell-text" }, JSON.stringify(v).slice(0, 160)) }],
      q ? rows.filter((r) => (r.actor + " " + r.action).toLowerCase().includes(q)) : rows, rows.length ? "No entry matches." : "No audit entries."));
  };
  filter.addEventListener("input", draw);
  api("/api/audit").then((r) => { rows = r; draw(); }).catch((e) => box.replaceChildren(note(e.message)));
  root.append(h("div", { class: "card" }, h("h2", null, "Audit log (latest 150)"), h("div", { class: "row" }, filter), box,
    note("Every sign-in, order, approval, role change and alert change is recorded with who did it. Auditors and risk managers can read it; nobody can edit it from here.")));
  return root;
}

// ----- shared: an editable list of rows (positions, holdings, rates) -----
/* fields: [{key, label, type?: "number", options?: [...], def?, width?}]. onRow(row) runs for each new row. */
function rowsEditor(fields, initial, addLabel, onRow) {
  const box = h("div");
  const add = (vals = {}) => {
    const inputs = {};
    const cells = fields.map((f) => {
      const inp = f.options ? h("select", { "aria-label": f.label }, f.options.map((o) => h("option", { value: o }, o)))
        : h("input", { type: f.type || "text", step: "any", "aria-label": f.label, style: `width:${f.width || 100}px` });
      inp.value = vals[f.key] ?? f.def ?? "";
      inputs[f.key] = inp;
      return field(f.label, inp);
    });
    const row = h("div", { class: "row editrow" }, cells, h("button", { class: "btn ghost", type: "button", "aria-label": "remove row", onclick: () => row.remove() }, "✕"));
    row.inputs = inputs;
    row.get = () => Object.fromEntries(fields.map((f) => {
      const v = inputs[f.key].value.trim();
      return [f.key, f.type === "number" ? (v === "" ? null : Number(v)) : v];
    }));
    box.append(row);
    if (onRow) onRow(row);
    return row;
  };
  initial.forEach(add);
  return { box, add, get: () => [...box.children].map((r) => r.get()),
    node: h("div", null, box, h("button", { class: "btn ghost", type: "button", onclick: () => add() }, addLabel || "+ Add row")) };
}
const yesNo = (v) => v ? h("span", { class: "bad" }, "yes") : h("span", { class: "good" }, "no");
const bps = (v) => v == null ? "–" : fmt.num(v, 1) + " bps";

// ----- market and reference data: instruments -----
const SAMPLE_INSTRUMENTS = [
  "symbol,asset_class,name,exchange,currency,tick_size,lot_size,multiplier,sector,underlying,expiry,strike,right,margin_pct,broker:paper",
  "AAA,equity,Alpha Industries,NYSE,USD,0.01,1,1,Industrials,,,,,1,AAA.N",
  "BBB,equity,Bolt Motors,NASDAQ,USD,0.01,1,1,Consumer,,,,,1,BBB.O",
  "CCC,equity,Cedar Bank,LSE,GBP,0.5,1,1,Financials,,,,,1,CCC.L",
  "ESZ6,future,E-mini S&P 500 Dec 2026,CME,USD,0.25,1,50,,ES,2026-12-18,,,0.06,ESZ6",
  "AAA261218C160,option,AAA Dec 2026 160 call,CBOE,USD,0.01,1,100,,AAA,2026-12-18,160,call,1,",
].join("\n");

function instrumentsView() {
  const root = h("div"), listBox = h("div"), count = h("span", { class: "muted" });
  const search = h("input", { placeholder: "filter by symbol, name, sector", "aria-label": "filter instruments", style: "min-width:220px" });
  let rows = [];
  const cols = [{ key: "symbol", label: "Symbol" }, { key: "asset_class", label: "Class" }, { key: "name", label: "Name", render: (v) => h("span", { class: "cell-text" }, v || "") },
    { key: "exchange", label: "Exchange" }, { key: "currency", label: "Ccy" }, { key: "tick_size", label: "Tick" }, { key: "lot_size", label: "Lot" },
    { key: "multiplier", label: "Multiplier" }, { key: "sector", label: "Sector" }, { key: "underlying", label: "Underlying" }, { key: "expiry", label: "Expiry" },
    { key: "strike", label: "Strike" }, { key: "right", label: "Right" },
    { key: "broker_symbols", label: "Broker symbols", render: (v) => Object.entries(v || {}).map(([b, x]) => b + ": " + x).join(", ") || "–" }];
  const draw = () => {
    const q = search.value.trim().toLowerCase();
    const shown = q ? rows.filter((r) => [r.symbol, r.name, r.sector, r.exchange, r.asset_class, r.underlying].some((v) => String(v || "").toLowerCase().includes(q))) : rows;
    listBox.replaceChildren(table(cols, shown, rows.length ? "No instrument matches." : "The instrument master is empty. Operations loads it from a CSV of contract specifications."));
  };
  search.addEventListener("input", draw);
  const csv = h("textarea", { "aria-label": "instrument CSV", rows: 7 }, SAMPLE_INSTRUMENTS);
  const loadBtn = h("button", { class: "btn", type: "submit" }, "Load instruments");
  const replace = h("input", { type: "checkbox", "aria-label": "replace instruments that already exist" });
  const report = h("div");
  const form = h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(loadBtn, async () => {
    const r = await api("/api/instruments/load", { csv: csv.value, overwrite: replace.checked });
    toast(`Loaded ${r.loaded}, rejected ${r.rejected.length}`, r.rejected.length > 0);
    report.replaceChildren(r.rejected.length ? table([{ key: "item", label: "Row" }, { key: "error", label: "Why it was rejected", render: (v) => h("span", { class: "cell-text" }, v) }], r.rejected) : note("Every row loaded."));
    await load();
  }); } }, h("h2", null, "Load contract specifications"), csv,
    h("div", { class: "row", style: "margin-top:10px; align-items:center" }, loadBtn, h("label", { class: "check" }, replace, "Replace instruments that already exist")),
    note("One row per instrument. Futures and options need an underlying and an expiry; options also a strike and a right. Columns named broker:<name> map the platform symbol to that broker's own symbol. A symbol that is already in the master is rejected unless you tick Replace, so a careless re-load cannot change contract terms. Bad rows are rejected one by one and the rest still load."), report);
  const load = async () => {
    const r = await api("/api/instruments");
    rows = r.instruments;
    count.textContent = r.count + " instrument(s)";
    draw();
    if (r.can_load && !form.isConnected) root.append(form);
  };
  load().catch((e) => toast(e.message, true));
  root.append(h("div", { class: "card" }, h("h2", null, "Instrument master"), h("div", { class: "row" }, search, count), listBox,
    note("What can be traded and on what terms: tick and lot size, contract multiplier, currency, expiry, and each broker's symbol for it. Sizing, margin and order checks read these terms.")));
  return root;
}

// ----- signal libraries: news sentiment -----
const SAMPLE_HEADLINES = ["Alpha Industries beats estimates and raises full-year guidance", "Bolt Motors downgraded to sell after weak deliveries",
  "Regulator opens probe into Bolt Motors battery fires", "Alpha to acquire a sensor maker in all-cash deal", "Cedar Bank shares little changed ahead of results",
  "Markets wait for the central bank decision"].join("\n");
const SAMPLE_ALIASES = "AAA = Alpha Industries, Alpha\nBBB = Bolt Motors, Bolt\nCCC = Cedar Bank";

function sentimentView() {
  const root = h("div"), out = h("div");
  const headlines = h("textarea", { "aria-label": "headlines", rows: 7 }, SAMPLE_HEADLINES);
  const aliases = h("textarea", { "aria-label": "symbol aliases", rows: 3 }, SAMPLE_ALIASES);
  const run = h("button", { class: "btn", type: "submit" }, "Score headlines");
  root.append(h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(run, async () => {
    const r = await api("/api/lab/sentiment", { headlines: headlines.value, aliases: aliases.value });
    const tone = (v) => h("span", { class: fmt.tone(v) }, fmt.num(v, 2));
    out.replaceChildren(
      h("div", { class: "kpis" }, kpi("Average tone", fmt.num(r.average, 2), fmt.tone(r.average)), kpi("Headlines", r.headlines.length), kpi("No sentiment found", r.neutral)),
      h("div", { class: "card" }, h("h2", null, "By instrument"), table([{ key: "symbol", label: "Symbol" }, { key: "headlines", label: "Headlines" },
        { key: "average", label: "Average tone", render: tone }, { key: "events", label: "Events", render: (v) => v.map((x) => x.replaceAll("_", " ")).join(", ") || "–" }],
        r.by_symbol, "No headline mentions an instrument in the alias list.")),
      h("div", { class: "card" }, h("h2", null, "Each headline"), table([{ key: "headline", label: "Headline", render: (v) => h("span", { class: "cell-text" }, v) },
        { key: "score", label: "Words", render: tone }, { key: "events", label: "Events", render: (v) => v.map((x) => x.replaceAll("_", " ")).join(", ") || "–" },
        { key: "combined", label: "Tone", render: tone }, { key: "symbols", label: "Instruments", render: (v) => v.join(", ") || "–" }], r.headlines)));
  }); } }, h("h2", null, "Score news headlines"),
    h("div", { class: "grid cols-2" }, field("Headlines, one per line", headlines), field("Instruments: SYMBOL = name, other name", aliases)),
    h("div", { style: "margin-top:10px" }, run),
    note("Tone runs from −1 (negative) to +1 (positive). Words are scored from a finance word list that understands negation (\"not strong\"); events such as an earnings beat, a downgrade or a takeover are found by rules and weigh more when present. It runs offline and gives the same answer every time. The platform also has a language-model scorer (sentiment/llm.py) that needs an API key and falls back to these rules if the service fails; it is not used on this page.")), out);
  return root;
}

// ----- quantitative research: option chain -----
function optionsView() {
  const root = h("div"), out = h("div");
  const spot = h("input", { type: "number", value: 100, step: "any", style: "width:90px" });
  const vol = h("input", { type: "number", value: 20, step: "any", style: "width:80px" });
  const skew = h("input", { type: "number", value: -0.1, step: "0.05", style: "width:80px" });
  const smile = h("input", { type: "number", value: 0.3, step: "0.05", style: "width:80px" });
  const days = h("input", { value: "30, 60, 90", style: "width:120px" });
  const seed = h("input", { type: "number", value: 0, style: "width:70px" });
  const expiry = h("select", { "aria-label": "expiry" });
  const parseDays = () => [...new Set(days.value.split(",").map((x) => Number(x.trim())).filter((x) => x > 0))].sort((a, b) => a - b);
  const syncExp = () => { const keep = expiry.value; expiry.replaceChildren(...parseDays().map((d, i) => h("option", { value: i }, d + " days"))); if (keep && Number(keep) < expiry.options.length) expiry.value = keep; };
  days.addEventListener("change", syncExp);
  syncExp();
  const run = h("button", { class: "btn", type: "submit" }, "Build chain");
  root.append(h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(run, async () => {
    out.replaceChildren(optionResults(await api("/api/lab/option_chain", { spot: Number(spot.value), atm_vol: Number(vol.value) / 100, skew: Number(skew.value),
      smile: Number(smile.value), days: parseDays(), seed: Number(seed.value), expiry: Number(expiry.value || 0) })));
  }); } }, h("h2", null, "Option chain analytics"),
    h("div", { class: "row" }, field("Spot", spot), field("ATM volatility %", vol), field("Skew", skew), field("Smile", smile), field("Expiries (days)", days),
      field("Show expiry", expiry), field("Seed", seed), run),
    note("The chain is synthetic: Black-Scholes prices from a skewed smile, with a bid/ask spread and open interest concentrated near the money. Implied volatility and the Greeks are then solved back from the mid prices exactly as they would be for a real chain loaded from a data vendor.")), out);
  return root;
}

function optionResults(r) {
  const s = r.summary, sk = r.skew, em = r.expected_move, out = h("div");
  const atm = r.chain.reduce((best, x) => Math.abs(x.strike - r.spot) < Math.abs(best - r.spot) ? x.strike : best, r.chain[0].strike);
  const days = (r.expiries.find((x) => x.expiry === r.expiry) || {}).days;
  out.append(h("div", { class: "kpis" }, kpi("Expiry " + r.expiry, days + " days"), kpi("ATM implied vol", fmt.pct(s.atm_iv, 1)), kpi("Expected move", "±" + fmt.pct(em.iv_move_pct, 1)),
    kpi("ATM straddle, % of spot", fmt.pct(em.straddle_pct, 1)), kpi("Put/call ratio (OI)", fmt.num(s.put_call_ratio, 2)), kpi("Max pain", fmt.num(s.max_pain, 2)),
    kpi("25Δ risk reversal, vol pts", fmt.num(sk.risk_reversal * 100, 2), fmt.tone(sk.risk_reversal)), kpi("25Δ butterfly, vol pts", fmt.num(sk.butterfly * 100, 2))));
  out.append(h("div", { class: "grid cols-2" },
    h("div", { class: "card" }, h("h2", null, "Volatility smile"), lineChart([{ name: "Implied vol", cls: "s0", fcls: "f0", points: r.smile.map((x) => [fmt.num(x.strike, 1), x.iv]) }],
      { yFmt: (v) => fmt.pct(v, 1), label: "volatility smile" }),
      note("Out-of-the-money options only (puts below the forward, calls above), because they are the liquid side. A negative risk reversal means downside protection costs more than upside calls.")),
    h("div", { class: "card" }, h("h2", null, "Term structure"), table([{ key: "days", label: "Days" }, { key: "expiry", label: "Expiry" }, { key: "atm_iv", label: "ATM implied vol", render: (v) => fmt.pct(v, 1) }], r.expiries),
      h("h3", { style: "margin-top:14px" }, "Positioning"),
      table([{ key: "k", label: "" }, { key: "v", label: "Strikes (open interest)", render: (v) => h("span", { class: "cell-text" }, v) }], [
        { k: "Call walls above spot", v: r.walls.resistance.map(([k, oi]) => `${fmt.num(k, 1)} (${fmt.num(oi, 0)})`).join(", ") || "–" },
        { k: "Put walls below spot", v: r.walls.support.map(([k, oi]) => `${fmt.num(k, 1)} (${fmt.num(oi, 0)})`).join(", ") || "–" },
        { k: "Dealer gamma, per 1% move", v: fmt.money(r.gamma_total) + (r.gamma_total >= 0 ? " (dampens moves)" : " (amplifies moves)") }]),
      note("Large open interest often acts as support or resistance. Dealer gamma assumes dealers are long calls and short puts; it is a convention, not an observation."))));
  const mid = (b, a) => b == null || a == null ? "–" : fmt.num(b, 2) + " / " + fmt.num(a, 2);
  out.append(h("div", { class: "card full" }, h("h2", null, "Chain: calls on the left, puts on the right"),
    table([{ key: "call_open_interest", label: "Call OI", render: (v) => fmt.num(v, 0) }, { key: "call_delta", label: "Δ", render: (v) => fmt.num(v, 2) },
      { key: "call_iv", label: "IV", render: (v) => fmt.pct(v, 1) }, { key: "call_bid", label: "Bid / ask", render: (v, x) => mid(v, x.call_ask) },
      { key: "strike", label: "Strike", render: (v) => v === atm ? h("strong", null, fmt.num(v, 2) + " ◀") : fmt.num(v, 2) },
      { key: "put_bid", label: "Bid / ask", render: (v, x) => mid(v, x.put_ask) }, { key: "put_iv", label: "IV", render: (v) => fmt.pct(v, 1) },
      { key: "put_delta", label: "Δ", render: (v) => fmt.num(v, 2) }, { key: "put_open_interest", label: "Put OI", render: (v) => fmt.num(v, 0) }], r.chain),
    note(r.quality.ok ? `Quote check: ${r.quality.quotes} quotes, none crossed, empty or arbitrageable.`
      : `Quote check found problems: ${r.quality.crossed} crossed, ${r.quality.no_price} without a price, ${r.quality.iv_failures} without an implied vol, ${r.quality.arbitrage.length} arbitrage warning(s).`)));
  return out;
}

// ----- learning and AI: prediction models -----
const MODEL_LABELS = { logistic: "Logistic regression", naive_bayes: "Naive Bayes", knn: "k-nearest neighbours", tree: "Decision tree",
  forest: "Random forest", boosting: "Gradient boosting", mlp: "Neural network (MLP)" };

function modelsView() {
  const root = h("div"), out = h("div"), df = singleSymbolFields("AAA");
  const model = h("select", { "aria-label": "model" }, Object.entries(MODEL_LABELS).map(([k, v]) => h("option", { value: k }, v)));
  const horizon = h("input", { type: "number", value: 5, min: 1, max: 20, style: "width:70px" });
  const train = h("input", { type: "number", value: 250, min: 100, style: "width:80px" });
  const run = h("button", { class: "btn", type: "submit" }, "Train and test");
  root.append(h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(run, async () => {
    const r = await api("/api/lab/models", { ...df.get(), model: model.value, horizon: Number(horizon.value), train_size: Number(train.value) });
    const edge = r.accuracy - r.base_rate;
    out.replaceChildren(
      h("div", { class: "kpis" }, kpi("Hit rate", fmt.pct(r.accuracy, 1), edge > 0.01 ? "good" : edge < -0.01 ? "bad" : ""), kpi("Up moves (base rate)", fmt.pct(r.base_rate, 1)),
        kpi("AUC", fmt.num(r.auc, 3)), kpi("Information coefficient", fmt.num(r.ic, 3), fmt.tone(r.ic)), kpi("Predictions", r.predictions),
        kpi("Refits", r.fits + (r.failed_fits ? ` (${r.failed_fits} failed)` : ""), r.failed_fits ? "warn" : ""), kpi("Time in market", fmt.pct(r.exposure, 0)),
        kpi("Model trading", fmt.pct(r.strategy_return, 1), fmt.tone(r.strategy_return)), kpi("Buy and hold", fmt.pct(r.buy_hold_return, 1), fmt.tone(r.buy_hold_return))),
      h("div", { class: "card" }, h("h2", null, "Trading the predictions, next bar, no costs"),
        lineChart([{ name: MODEL_LABELS[r.model], cls: "s0", fcls: "f0", points: r.equity }, { name: "Buy and hold", cls: "s1", fcls: "f1", points: r.buy_hold }], { label: "model equity" }),
        h("h3", { style: "margin-top:14px" }, "Predicted probability of a rise"),
        lineChart([{ name: "P(up)", cls: "s0", fcls: "f0", points: r.probability }], { small: true, baseline: 0.5, yFmt: (v) => fmt.pct(v, 0), label: "predicted probability" }),
        note(`Latest (${r.latest.date}): P(up over ${r.horizon} bars) = ${fmt.pct(r.latest.probability, 1)}, position ${fmt.num(r.latest.signal, 2)}.`)),
      h("div", { class: "card" }, h("h2", null, "How this was tested"),
        note(`Walk-forward: the model was refitted ${r.fits} times, each time on the past only, and predicted the next ${r.step} bars it had not seen. Training stops ${r.horizon} bars before each prediction block (an embargo), because a label built from the next ${r.horizon} bars would otherwise leak the answer. Features (${r.features.length}): ${r.features.join(", ")}. A hit rate a few points above the base rate, or an information coefficient of 0.02 to 0.05, is already useful on daily data; much more usually means leakage. The data is synthetic, so expect little real skill.`)));
  }); } }, h("h2", null, "Predict the direction of the next move"),
    h("div", { class: "row" }, ...df.nodes, field("Model", model), field("Horizon (bars)", horizon), field("First training window", train), run),
    note("Learns P(price is higher after the horizon) from price and volume features, then trades it: long above 52%, short below 48%, flat in between, sized by conviction. Only quants and admins can train models.")), out);
  return root;
}

// ----- learning and AI: regimes, change points, anomalies -----
function regimesView() {
  const root = h("div"), out = h("div"), df = singleSymbolFields("AAA");
  const states = h("select", { "aria-label": "number of regimes" }, [2, 3].map((n) => h("option", { value: n }, n + " regimes")));
  const run = h("button", { class: "btn", type: "submit" }, "Detect");
  root.append(h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(run, async () => {
    const r = await api("/api/lab/regimes", { ...df.get(), states: Number(states.value) });
    const cur = r.current;
    out.replaceChildren(
      h("div", { class: "kpis" }, kpi("Symbol", r.symbol), kpi("Current regime", cur.label, cur.label === "turbulent" ? "bad" : cur.label === "calm" ? "good" : "warn"),
        kpi("Probability", fmt.pct(cur.probability, 0)), ...Object.entries(r.time_in_regime).map(([k, v]) => kpi("Time " + k, fmt.pct(v, 0))),
        kpi("Change points", r.changepoints.length), kpi("Anomalies", r.anomalies.length)),
      h("div", { class: "card" }, h("h2", null, "Price"), lineChart([{ name: "Close", cls: "s0", fcls: "f0", points: r.close }], { label: "price" }),
        h("h3", { style: "margin-top:14px" }, "Probability of the most turbulent regime"),
        lineChart([{ name: "P(" + r.labels[r.labels.length - 1] + ")", cls: "neg", fcls: "fneg", points: r.turbulent }], { small: true, fill: true, baseline: 0, yFmt: (v) => fmt.pct(v, 0), label: "turbulent probability" }),
        h("h3", { style: "margin-top:14px" }, "Probability that a change point just happened"),
        lineChart([{ name: "P(change)", cls: "s0", fcls: "f0", points: r.change_prob }], { small: true, fill: true, baseline: 0, yFmt: (v) => fmt.pct(v, 0), label: "change probability" })),
      h("div", { class: "grid cols-2" },
        h("div", { class: "card" }, h("h2", null, "The regimes"), table([{ key: "label", label: "Regime" }, { key: "ann_return", label: "Return / year", tone: true, render: (v) => fmt.pct(v, 1) },
          { key: "ann_vol", label: "Volatility / year", render: (v) => fmt.pct(v, 1) }, { key: "expected_bars", label: "Typical length (bars)", render: (v) => fmt.num(v, 0) }], r.describe),
          h("h3", { style: "margin-top:14px" }, "Latest stretches"), table([{ key: "label", label: "Regime" }, { key: "start", label: "From" }, { key: "end", label: "To" }, { key: "bars", label: "Bars" }], r.segments)),
        h("div", { class: "card" }, h("h2", null, "Change points"), table([{ key: "date", label: "New regime from" }, { key: "close", label: "Close", render: (v) => fmt.num(v) }], r.changepoints, "None found."),
          h("h3", { style: "margin-top:14px" }, "Anomalous days"), table([{ key: "date", label: "Date" }, { key: "return", label: "Return", tone: true, render: (v) => fmt.pct(v) },
            { key: "z", label: "Robust z-score", render: (v) => fmt.num(v, 1) }], r.anomalies, "No return beyond 5 robust standard deviations."))),
      note(`A hidden Markov model learns the regimes from the first ${r.trained_on} bars, then labels every bar using only the data up to it; labels in that first stretch are in-sample. Change points come from Bayesian online change-point detection; anomalies compare each return with the median and spread of the 50 before it.`));
  }); } }, h("h2", null, "Market regimes, change points and anomalies"), h("div", { class: "row" }, ...df.nodes, field("Regimes", states), run),
    note("Use the current regime to switch strategies or cut size: trend-following tends to suffer in choppy, turbulent stretches.")), out);
  return root;
}

// ----- portfolio and risk: margin -----
const MARGIN_EXAMPLE = [
  { symbol: "AAA", asset_class: "equity", quantity: 400, price: 150 },
  { symbol: "BBB", asset_class: "equity", quantity: -300, price: 80 },
  { symbol: "ESZ6", asset_class: "future", quantity: 2, price: 5000, multiplier: 50, margin_pct: 6 },
  { symbol: "AAA 160C", asset_class: "option", quantity: -4, price: 3.2, multiplier: 100, underlying: "AAA", underlying_price: 150, strike: 160, right: "call", days: 30, volatility: 25 },
];
const MARGIN_FIELDS = [{ key: "symbol", label: "Symbol", width: 90 }, { key: "asset_class", label: "Type", options: ["equity", "etf", "fx", "crypto", "future", "option"] },
  { key: "quantity", label: "Quantity (− short)", type: "number", width: 100 }, { key: "price", label: "Price", type: "number", width: 90 },
  { key: "multiplier", label: "Multiplier", type: "number", width: 80, def: 1 }, { key: "margin_pct", label: "Margin %", type: "number", width: 75 },
  { key: "underlying", label: "Underlying", width: 80 }, { key: "underlying_price", label: "Underlying price", type: "number", width: 90 },
  { key: "strike", label: "Strike", type: "number", width: 80 }, { key: "right", label: "Right", options: ["call", "put"] },
  { key: "days", label: "Days to expiry", type: "number", width: 80 }, { key: "volatility", label: "Implied vol %", type: "number", width: 80 }];

function marginView() {
  const root = h("div"), out = h("div");
  const showFor = { multiplier: ["future", "option"], margin_pct: ["future"], underlying: ["option"], underlying_price: ["option"], strike: ["option"], right: ["option"],
    days: ["option"], volatility: ["option"] };
  const sync = (row) => { const t = row.inputs.asset_class.value; for (const [k, types] of Object.entries(showFor)) row.inputs[k].parentElement.hidden = !types.includes(t); };
  const ed = rowsEditor(MARGIN_FIELDS, MARGIN_EXAMPLE, "+ Add position", (row) => { row.inputs.asset_class.addEventListener("change", () => sync(row)); sync(row); });
  const equity = h("input", { type: "number", value: 100000, step: "any", style: "width:120px" });
  const run = h("button", { class: "btn", type: "submit" }, "Work out margin");
  const fromSession = h("button", { class: "btn ghost", type: "button" }, "Use my paper-trading positions");
  const show = (r) => out.replaceChildren(marginResults(r));
  fromSession.addEventListener("click", () => busy(fromSession, async () => show(await api("/api/risk/margin", { source: "session" }))));
  root.append(h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(run, async () => {
    const positions = ed.get().map((p) => ({ ...p, margin_pct: p.margin_pct == null ? null : p.margin_pct / 100, volatility: p.volatility == null ? null : p.volatility / 100 }));
    show(await api("/api/risk/margin", { equity: Number(equity.value), positions }));
  }); } }, h("h2", null, "Margin on a set of positions"), ed.node,
    h("div", { class: "row", style: "margin-top:12px" }, field("Account equity", equity), run, fromSession),
    note("Two methods side by side. Rule-based charges each position on its own (50% of a stock's value, the exchange's initial margin for a future, a share of the underlying for a short option), so hedges get no relief. Scenario margin moves each underlying's price and volatility up and down and charges the worst loss, so a covered call or a hedged future is charged for its net risk. The rates are typical defaults, not your broker's or exchange's schedule.")), out);
  return root;
}

function marginResults(r) {
  const a = r.rules, b = r.scenario, out = h("div");
  const rows = [["Initial requirement", "initial", fmt.money], ["Maintenance requirement", "maintenance", fmt.money], ["Available for new positions", "available_funds", fmt.money],
    ["Excess liquidity", "excess_liquidity", fmt.money], ["Utilisation of equity", "utilisation", (v) => fmt.pct(v, 1)], ["Gross leverage", "leverage", (v) => fmt.num(v, 2) + "x"],
    ["Margin call", "margin_call", yesNo]].map(([label, k, f]) => ({ label, rules: f(a[k]), scenario: f(b[k]) }));
  const call = a.margin_call || b.margin_call;
  out.append(h("div", { class: "kpis" }, kpi("Equity", fmt.money(r.equity)), kpi("Rule-based initial", fmt.money(a.initial)), kpi("Scenario initial", fmt.money(b.initial)),
    kpi("Scenario vs rule-based", (b.initial >= a.initial ? "+" : "") + fmt.money(b.initial - a.initial), b.initial < a.initial ? "good" : "warn"),
    kpi("Margin call", call ? "yes" : "no", call ? "bad" : "good")));
  if (call) out.append(h("div", { class: "banner halt" }, h("strong", null, "Margin call"), ` Equity is below maintenance: shortfall ${fmt.money(Math.max(a.shortfall, b.shortfall))}. Add funds or reduce positions.`));
  out.append(h("div", { class: "grid cols-2" },
    h("div", { class: "card" }, h("h2", null, "Account view"), table([{ key: "label", label: "" }, { key: "rules", label: "Rule-based" }, { key: "scenario", label: "Scenario" }], rows)),
    h("div", { class: "card" }, h("h2", null, "Scenario margin by underlying"), table([{ key: "group", label: "Underlying" }, { key: "margin", label: "Margin", render: fmt.money },
      { key: "worst_price_move", label: "Worst price move", render: (v) => fmt.pct(v, 0) }, { key: "worst_vol_move", label: "Vol move", render: (v) => fmt.pct(v, 0) },
      { key: "positions", label: "Positions" }], Object.entries(b.detail.by_underlying).map(([group, v]) => ({ group, ...v }))),
      note("Within one underlying, gains on one leg offset losses on another, so a covered call or a hedged pair costs less here than under the rules. Across underlyings no offset is given, which is the conservative choice. Every underlying is scanned over the same ±15% price range, so a future whose exchange margin is only a few percent of its value can cost more here than under the rules."))));
  out.append(h("div", { class: "card" }, h("h2", null, "Positions, each on its own"), table([{ key: "symbol", label: "Symbol" }, { key: "asset_class", label: "Type" },
    { key: "quantity", label: "Quantity" }, { key: "price", label: "Price", render: (v) => fmt.num(v) }, { key: "value", label: "Value", render: fmt.money },
    { key: "initial", label: "Initial", render: fmt.money }, { key: "maintenance", label: "Maintenance", render: fmt.money }], r.positions)));
  return out;
}

// ----- portfolio and risk: currency exposure -----
function currencyView() {
  const root = h("div"), out = h("div");
  const base = h("input", { value: "USD", maxlength: 3, style: "width:70px", "aria-label": "base currency" });
  const holdings = rowsEditor([{ key: "currency", label: "Currency", width: 70 }, { key: "amount", label: "Value in that currency (positions + cash)", type: "number", width: 150 }],
    [{ currency: "USD", amount: 50000 }, { currency: "EUR", amount: 20000 }, { currency: "JPY", amount: 3000000 }, { currency: "GBP", amount: -5000 }], "+ Add holding");
  const rates = rowsEditor([{ key: "pair", label: "Pair", width: 80 }, { key: "rate", label: "Rate", type: "number", width: 100 }],
    [{ pair: "EURUSD", rate: 1.08 }, { pair: "USDJPY", rate: 150 }, { pair: "GBPUSD", rate: 1.27 }], "+ Add rate");
  const ratio = h("input", { type: "number", value: 100, min: 0, max: 100, style: "width:70px" });
  const att = { value_start: 100000, value_end: 104000, rate_start: 1.08, rate_end: 1.05 };
  const attIn = Object.fromEntries(Object.entries(att).map(([k, v]) => [k, h("input", { type: "number", value: v, step: "any", style: "width:100px" })]));
  const run = h("button", { class: "btn", type: "submit" }, "Value and hedge");
  root.append(h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(run, async () => {
    const r = await api("/api/risk/currency", { base: base.value, holdings: holdings.get(), rates: rates.get(), hedge_ratio: Number(ratio.value) / 100,
      attribution: Object.fromEntries(Object.entries(attIn).map(([k, v]) => [k, Number(v.value)])) });
    const x = r.exposure;
    out.replaceChildren(
      h("div", { class: "kpis" }, kpi("Total in " + x.base, fmt.num(x.total, 0)), kpi("Gross", fmt.num(x.gross, 0)), kpi("Held in other currencies", fmt.pct(x.fx_exposure, 1), x.fx_exposure > 0.3 ? "warn" : "")),
      h("div", { class: "grid cols-2" },
        h("div", { class: "card" }, h("h2", null, "Exposure by currency"), table([{ key: "ccy", label: "Currency" }, { key: "local", label: "Local", render: (v) => fmt.num(v, 0) },
          { key: "rate", label: "Rate to " + x.base, render: (v) => fmt.num(v, 4) }, { key: "base", label: "In " + x.base, tone: true, render: (v) => fmt.num(v, 0) },
          { key: "share", label: "Share of gross", render: (v) => fmt.pct(v, 1) }], Object.entries(x.by_currency).map(([ccy, v]) => ({ ccy, ...v })))),
        h("div", { class: "card" }, h("h2", null, `Forward hedges for ${fmt.pct(r.hedge_ratio, 0)} of the exposure`),
          table([{ key: "ccy", label: "Currency" }, { key: "amount", label: "Sell forward (− = buy)", render: (v) => fmt.num(v, 0) }],
            Object.entries(r.hedges).map(([ccy, amount]) => ({ ccy, amount })), "Nothing to hedge."),
          r.attribution ? h("div", null, h("h3", { style: "margin-top:14px" }, "Where a foreign holding's result came from"),
            table([{ key: "k", label: "Source" }, { key: "v", label: "In " + x.base, tone: true, render: (v) => fmt.num(v, 0) }],
              [{ k: "Asset price", v: r.attribution.asset }, { k: "Exchange rate", v: r.attribution.currency }, { k: "Both together", v: r.attribution.interaction }, { k: "Total", v: r.attribution.total }]),
            note("The holding's value in its own currency and the rate at the start and end; the three sources add up to the total exactly.")) : null)));
  }); } }, h("h2", null, "Multi-currency exposure"),
    h("div", { class: "row" }, field("Base currency", base), field("Hedge ratio %", ratio)),
    h("div", { class: "grid cols-2", style: "margin-top:10px" }, h("div", null, h("h3", null, "Holdings"), holdings.node), h("div", null, h("h3", null, "Exchange rates"), rates.node)),
    h("h3", { style: "margin-top:12px" }, "Attribution example: one holding in a foreign currency"),
    h("div", { class: "row" }, field("Value at start (local)", attIn.value_start), field("Value at end (local)", attIn.value_end), field("Rate at start", attIn.rate_start), field("Rate at end", attIn.rate_end)),
    h("div", { style: "margin-top:12px" }, run),
    note("Rates are quoted as the price of the first currency in the second (EURUSD 1.08: one euro costs 1.08 dollars). Pairs that are not given directly are crossed through the base currency. Negative values are short positions or borrowed cash.")), out);
  return root;
}

// ----- trading: order-book simulator -----
function orderbookView() {
  const root = h("div"), out = h("div");
  const side = h("select", null, h("option", { value: "BUY" }, "Buy"), h("option", { value: "SELL" }, "Sell"));
  const qty = h("input", { type: "number", value: 800, min: 1, style: "width:90px" });
  const order = h("select", null, h("option", { value: "market" }, "Market"), h("option", { value: "limit" }, "Limit"));
  const offset = h("input", { type: "number", value: 0, min: -20, max: 50, style: "width:70px" });
  const offsetF = field("Ticks behind the touch", offset);
  const latency = h("input", { type: "number", value: 0.5, min: 0, max: 10, step: "0.1", style: "width:70px" });
  const vol = h("input", { type: "number", value: 2, min: 0, max: 20, step: "0.5", style: "width:70px" });
  const rate = h("input", { type: "number", value: 50, min: 5, max: 500, style: "width:70px" });
  const events = h("input", { type: "number", value: 1500, min: 100, max: 5000, style: "width:80px" });
  const seed = h("input", { type: "number", value: 1, style: "width:70px" });
  const syncOrder = () => { offsetF.hidden = order.value !== "limit"; };
  order.addEventListener("change", syncOrder);
  syncOrder();
  const run = h("button", { class: "btn", type: "submit" }, "Simulate");
  root.append(h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(run, async () => {
    out.replaceChildren(orderbookResults(await api("/api/trading/orderbook", { side: side.value, quantity: Number(qty.value), order: order.value,
      offset_ticks: Number(offset.value), latency: Number(latency.value), volatility: Number(vol.value), event_rate: Number(rate.value), events: Number(events.value), seed: Number(seed.value) })));
  }); } }, h("h2", null, "Send one order into a simulated order book"),
    h("div", { class: "row" }, field("Side", side), field("Quantity", qty), field("Order", order), offsetF, field("Latency (seconds)", latency),
      field("Price drift (ticks / √s)", vol), field("Market events / second", rate), field("Events after sending", events), field("Seed", seed), run),
    note("Background traders place, cancel and take liquidity around a drifting fair value; your order reaches the book after the latency. The same order is also run with no latency on the same random tape, so the difference is what the delay cost. A limit order at 0 ticks joins the best bid (buy) or best ask (sell); positive values sit further back, negative values cross into the spread.")), out);
  return root;
}

function orderbookResults(r) {
  const x = r.result, z = r.no_latency, out = h("div");
  const cost = x.slippage_bps != null && z.slippage_bps != null ? x.slippage_bps - z.slippage_bps : null;
  out.append(h("div", { class: "kpis" }, kpi("Filled", `${fmt.num(x.filled, 0)} / ${fmt.num(r.quantity, 0)}`, x.fill_rate < 1 ? "warn" : "good"),
    kpi("Average price", fmt.num(x.average_price, 4)), kpi("Arrival mid", fmt.num(x.arrival_mid, 4)), kpi("Slippage vs arrival", bps(x.slippage_bps), fmt.tone(-(x.slippage_bps || 0))),
    kpi("Same order, no latency", bps(z.slippage_bps)), kpi("Cost of latency", bps(cost), cost > 0 ? "bad" : cost < 0 ? "good" : ""),
    x.unfilled > 0 ? kpi("Unfilled part: market moved", bps(x.opportunity_bps), fmt.tone(-(x.opportunity_bps || 0))) : null, kpi("Average spread", bps(x.spread_bps))));
  if (x.rejections.length) out.append(h("div", { class: "banner" }, h("strong", null, "The book refused part of the order: "), x.rejections.join("; ")));
  out.append(h("div", { class: "card" }, h("h2", null, "Mid price over the simulation"), lineChart([{ name: "Mid", cls: "s0", fcls: "f0", points: x.mid }], { label: "mid price" }),
    note(`Your order was sent at ${fmt.num(x.sent_at, 1)} s${x.limit_price != null ? " with a limit of " + fmt.num(x.limit_price, 2) : " at market"} and took part in ${x.fills} trade(s). Positive slippage means you paid more (or sold for less) than the mid when you decided.`)));
  const lvl = (rows) => rows.map(([price, size]) => ({ price, size }));
  const e = x.estimate;
  out.append(h("div", { class: "grid cols-2" },
    h("div", { class: "card" }, h("h2", null, "Book when the order was sent"), h("div", { class: "grid cols-2" },
      table([{ key: "price", label: "Bid", render: (v) => fmt.num(v, 2) }, { key: "size", label: "Size", render: (v) => fmt.num(v, 0) }], lvl(x.depth.bids), "Empty."),
      table([{ key: "price", label: "Ask", render: (v) => fmt.num(v, 2) }, { key: "size", label: "Size", render: (v) => fmt.num(v, 0) }], lvl(x.depth.asks), "Empty."))),
    h("div", { class: "card" }, h("h2", null, "Estimate before sending (market order)"), table([{ key: "k", label: "" }, { key: "v", label: "" }], [
      { k: "Average price", v: fmt.num(e.average_price, 4) }, { k: "Could fill now", v: fmt.num(e.filled, 0) }, { k: "Left unfilled", v: fmt.num(e.unfilled, 0) },
      { k: "Price levels used", v: fmt.num(e.levels, 0) }, { k: "Slippage vs mid", v: bps(e.slippage_bps) }]),
      note("Walking the visible book is what an immediate market order would pay; by the time a delayed order arrives the book has changed."))));
  out.append(note("The background flow is random: this shows how queue position, size and delay affect a fill, not how any real market behaves. Backtests and paper trading still fill on bars."));
  return out;
}

// ----- operations and control: health and jobs -----
function healthView() {
  const root = h("div");
  const load = async () => {
    const r = await api("/api/ops/health");
    const st = r.health.status, cls = st === "UP" ? "good" : st === "DEGRADED" ? "warn" : "bad";
    const replay = h("button", { class: "btn ghost", type: "button" }, "Retry delivery");
    replay.addEventListener("click", () => busy(replay, async () => { const x = await api("/api/ops/replay", {}); toast(`Delivered ${x.delivered}, still failing ${x.still_failing}`); await load(); }));
    const runJobs = h("button", { class: "btn ghost", type: "button" }, "Run due jobs now");
    runJobs.addEventListener("click", () => busy(runJobs, async () => { const x = await api("/api/ops/run_jobs", {}); toast(x.ran.length ? "Ran: " + x.ran.join(", ") : "No job was due"); await load(); }));
    const shown = (v) => Array.isArray(v) ? (v.length ? v.join(", ") : "none") : v && typeof v === "object" ? JSON.stringify(v) : String(v);
    const restore = Object.entries(r.restore || {}).map(([k, v]) => ({ k: k.replaceAll("_", " "), v: shown(v) }));
    root.replaceChildren(
      h("div", { class: "kpis" }, kpi("Platform", st, cls), kpi("Failing checks", r.health.failing.length, r.health.failing.length ? "bad" : "good"),
        kpi("State saved to disk", r.persistent ? "yes" : "no", r.persistent ? "good" : "warn"), kpi("Open paper sessions", r.sessions), kpi("Users", r.users),
        kpi("Undelivered notifications", r.undelivered.length, r.undelivered.length ? "warn" : "")),
      h("div", { class: "card" }, h("h2", null, "Health checks"), table([{ key: "name", label: "Check" }, { key: "ok", label: "State", render: (v) => v ? h("span", { class: "good" }, "ok") : h("span", { class: "bad" }, "failing") },
        { key: "detail", label: "Detail", render: (v) => h("span", { class: "cell-text" }, v || "") }, { key: "critical", label: "Critical", render: (v) => v ? "yes" : "no" },
        { key: "latency_ms", label: "Took (ms)", render: (v) => fmt.num(v, 1) }], r.health.items, "No checks registered."),
        note("A failing critical check marks the platform DOWN; a failing non-critical one marks it DEGRADED. Changes of state are sent as notifications.")),
      h("div", { class: "grid cols-2" },
        h("div", { class: "card" }, h("h2", null, "Scheduled jobs"), table([{ key: "name", label: "Job" }, { key: "enabled", label: "On", render: (v) => v ? "yes" : "no" },
          { key: "runs", label: "Runs" }, { key: "failures", label: "Failures" }, { key: "next_run", label: "Next run", render: when }, { key: "last_error", label: "Last error", render: (v) => v || "–" }],
          r.jobs, "No scheduled jobs (state is not saved to disk, so there is nothing to save periodically)."), h("div", { class: "row" }, runJobs)),
        h("div", { class: "card" }, h("h2", null, "Undelivered notifications"), table([{ key: "channel", label: "Channel" }, { key: "title", label: "Notification" },
          { key: "error", label: "Error", render: (v) => h("span", { class: "cell-text" }, v || "") }, { key: "attempts", label: "Attempts" }], r.undelivered, "Everything was delivered."),
          r.undelivered.length ? h("div", { class: "row" }, replay) : null)),
      h("div", { class: "grid cols-2" },
        h("div", { class: "card" }, h("h2", null, "Notification routes"), table([{ key: "channel", label: "Channel" }, { key: "scope", label: "Scope" },
          { key: "min_severity", label: "From severity", render: sevBadge }, { key: "topics", label: "Topics", render: (v) => v.join(", ") || "all" }], r.routes, "No routes.")),
        h("div", { class: "card" }, h("h2", null, "What was restored at start-up"), table([{ key: "k", label: "Part" }, { key: "v", label: "Result", render: (v) => h("span", { class: "cell-text" }, v) }], restore,
          r.persistent ? "Nothing to restore: this is a fresh database." : "The server was started without --db-dir, so nothing is saved or restored."),
          note("Channels holding a password or token are never written to disk; they are listed here as needing to be added again after a restart."))));
  };
  load().catch((e) => toast(e.message, true));
  return root;
}

// ---------- boot ----------
async function boot() {
  if (!store.token) return mount(renderLogin());
  try { store.me = await api("/api/me"); }
  catch (_) { return mount(renderLogin()); }
  try { store.structure = await api("/api/structure"); } catch (_) { store.structure = null; }   // the menu still works, with plainer titles
  renderApp();
}
boot();
