"use strict";
/* Strategy: the strategy builder and instrument selection. */

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

// ----- selection -----
function selectView() {
  const root = h("div"), out = h("div"), df = dataFields({ symbols: "AAA,BBB,CCC,DDD,EEE,FFF,GGG,HHH" });
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
