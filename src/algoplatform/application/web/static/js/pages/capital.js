"use strict";
/* Portfolio and risk: the multi-strategy portfolio, margin and currency exposure. */

// ----- multi-strategy portfolio -----
function portfolioView() {
  const root = h("div"), results = h("div"), rowsBox = h("div"), df = dataFields({ bars: 600 });
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
