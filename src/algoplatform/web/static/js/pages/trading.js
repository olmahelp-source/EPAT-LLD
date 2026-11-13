"use strict";
/* Trading: paper trading (order ticket, execution algos, kill switch, live updates) and the order-book simulator. */

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
  const hist = historySymbols();
  const feed = h("select", { "aria-label": "price feed" }, h("option", { value: "sample" }, "Sample: random walk"),
    h("option", { value: "history", disabled: sampleLocked() || !hist.length }, "Replay downloaded history" + (hist.length ? "" : " (none downloaded)")));
  feed.value = !useSample() && hist.length ? "history" : "sample";
  const symbols = h("input", { value: feed.value === "history" ? hist.slice(0, 2).join(",") : "AAA,BBB", style: "width:150px" });
  feed.addEventListener("change", () => { symbols.value = feed.value === "history" ? hist.slice(0, 2).join(",") : "AAA,BBB"; });
  const sel = h("select", null, h("option", { value: "" }, "None (manual trading only)"));
  strategies().then((l) => l.filter((x) => x.runnable && !x.slow && x.name !== "pairs_trading" && x.name !== "momentum_rotation")
    .forEach((x) => sel.append(h("option", { value: x.name }, x.name)))).catch(() => {});
  const btn = h("button", { class: "btn", type: "submit" }, "Open paper session");
  return h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(btn, async () => {
    show(await api("/api/session/open", { symbols: symbols.value.split(",").map((x) => x.trim()).filter(Boolean), strategy: sel.value || undefined, feed: feed.value }));
  }); } }, h("h2", null, "Paper trading session"),
    h("p", { class: "hint" }, "A simulated broker on simulated time, fed either by a sample random walk or by replaying downloaded daily history bar by bar (the strategy sees each day only when it is played). Every order goes through the same risk gate, order manager and kill switch as the backtester; large orders can be worked by an execution algo. No real money is involved. With the server started using --db-dir, the session survives a restart."),
    h("div", { class: "row" }, field("Price feed", feed), field("Symbols", symbols), field("Run a strategy", sel)), h("div", { style: "margin-top:12px" }, btn));
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
    halted ? `${ks.reason} (${ks.source})` : `${snap.bars} bars processed`,
    snap.replay ? (snap.replay.available ? ` · replaying history ${snap.replay.from} to ${snap.replay.to}: bar ${snap.replay.pos} of ${snap.replay.total}` : " · the history being replayed is not available: close the session") : " · sample feed",
    snap.resumed ? (snap.recovery && !snap.recovery.clean ? " · resumed with a reconciliation break" : " · resumed after restart") : "");
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
