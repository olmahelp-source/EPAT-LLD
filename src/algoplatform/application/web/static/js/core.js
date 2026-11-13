"use strict";
/* Algo Platform UI, shared core: vanilla JS, no build step. All server text is inserted via textContent
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
const store = { token: null, me: null, structure: null, sources: null };
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

/* Which data a page starts on: the header switch (per browser), unless operations has locked the platform to sample data. */
function sampleLocked() { return !store.sources || !!store.sources.sample_only; }
function useSample() {
  if (sampleLocked()) return true;
  try { return localStorage.getItem("sampleData") !== "false"; } catch (_) { return true; }
}
/* The sample scenario every data form starts with ("" = the mixed universe); chosen beside the header switch. */
function sampleScenario() {
  let v = "";
  try { v = localStorage.getItem("sampleScenario") || ""; } catch (_) { /* private mode */ }
  return store.sources && store.sources.scenarios && store.sources.scenarios[v] ? v : "";
}
function historySymbols() { return ((store.sources && store.sources.library) || []).filter((x) => x.timeframe === "1Day").map((x) => x.symbol); }
async function refreshSources() {
  try { store.sources = await api("/api/data/sources"); } catch (_) { /* keep the last known state */ }
  if (shell.dataSwitch) shell.dataSwitch.replaceChildren(...dataSwitchParts());
  return store.sources;
}

/* The data choice on every research and validation form. `def.symbols` / `def.bars` are the sample defaults; with
   downloaded history the same number of symbols is taken from the library instead. */
function dataFields(def = {}) {
  const st = store.sources || { sample_only: true, library: [], scenarios: {}, uploaded: [] };
  const sampleSymbols = def.symbols || "AAA,BBB,CCC";
  const count = sampleSymbols.split(",").length;
  const hist = historySymbols();
  const source = h("select", { "aria-label": "data source" }, h("option", { value: "sample" }, "Sample data"),
    h("option", { value: "history", disabled: st.sample_only }, "Downloaded history" + (hist.length ? ` (${hist.length})` : "")),
    h("option", { value: "uploaded", disabled: st.sample_only }, "Uploaded CSV"));
  const scenario = h("select", { "aria-label": "sample scenario" }, h("option", { value: "" }, "Mixed universe"),
    Object.keys(st.scenarios || {}).map((k) => h("option", { value: k }, k.replaceAll("_", " "))));
  const symbols = h("input", { value: sampleSymbols, style: `width:${count > 4 ? 260 : 150}px`, "aria-label": "symbols" });
  const bars = h("input", { type: "number", value: def.bars || 756, min: 150, max: 3000, style: "width:90px" });
  const seed = h("input", { type: "number", value: 7, style: "width:80px" });
  const hint = h("span", { class: "src-hint" });
  const scenarioF = field("Scenario", scenario), barsF = field("Bars", bars), seedF = field("Seed", seed), symbolsF = field("Symbols", symbols);
  let lastSample = sampleSymbols;
  const sync = (changed) => {
    const v = source.value;
    scenarioF.hidden = barsF.hidden = seedF.hidden = v !== "sample";
    symbolsF.hidden = v === "uploaded";
    if (changed && v === "history") { lastSample = symbols.value; symbols.value = hist.slice(0, count > 4 ? 10 : count).join(","); }
    if (changed && v === "sample") symbols.value = lastSample;
    hint.textContent = v === "sample" ? (scenario.value ? st.scenarios[scenario.value] : "Generated prices for a correlated universe: every symbol name works.")
      : v === "history" ? (hist.length ? `Daily bars downloaded from a data provider. In the library: ${hist.join(", ")}.` : "Nothing downloaded yet: download history on the Price data page.")
      : ((st.uploaded || []).length ? `Your uploaded files: ${st.uploaded.join(", ")}.` : "Nothing uploaded yet: upload a CSV on the Price data page.");
  };
  scenario.value = sampleScenario();
  source.value = useSample() ? "sample" : (hist.length || !(st.uploaded || []).length ? "history" : "uploaded");
  if (source.value === "history") symbols.value = hist.slice(0, count > 4 ? 10 : count).join(",");
  source.addEventListener("change", () => sync(true));
  scenario.addEventListener("change", () => sync(false));
  sync(false);
  return { source, symbols, bars, seed, scenario, nodes: [field("Data", source), scenarioF, symbolsF, barsF, seedF, hint],
    get: () => source.value === "uploaded" ? { source: "uploaded", dataset: "uploaded" }
      : source.value === "history" ? { source: "history", symbols: symbols.value.split(",").map((x) => x.trim()).filter(Boolean) }
      : { source: "sample", symbols: symbols.value.split(",").map((x) => x.trim()).filter(Boolean), bars: Number(bars.value), seed: Number(seed.value),
          scenario: scenario.value || undefined } };
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

// ----- shared helpers for the research tabs -----
function singleSymbolFields(def) {
  const df = dataFields({ symbols: def || "AAA" });
  return df;
}
const when = (v) => v ? String(v).replace("T", " ").slice(0, 16) : "–";
const dirLabel = (d) => d > 0 ? h("span", { class: "good" }, "bullish") : d < 0 ? h("span", { class: "bad" }, "bearish") : h("span", { class: "muted" }, "neutral");
const sevBadge = (v) => h("span", { class: "badge sev-" + v }, v);
const stageBadge = (v) => h("span", { class: "badge stage-" + v }, v);
function note(text) { return h("p", { class: "hint" }, text); }

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
