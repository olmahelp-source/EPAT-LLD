"use strict";
/* Quantitative research: volatility and the option chain. */

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
