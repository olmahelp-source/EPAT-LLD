"use strict";
/* Validation: backtest (and the results view other pages reuse), optimise, walk-forward. */

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

// ----- optimiser -----
const OPT_METHODS = [["random", "Random search"], ["genetic", "Genetic algorithm"], ["bayesian", "Bayesian (Gaussian process)"], ["tpe", "Tree-structured Parzen (TPE)"],
  ["pso", "Particle swarm"], ["de", "Differential evolution"], ["annealing", "Simulated annealing"], ["cmaes", "CMA-ES"]];
function optimizeView() {
  const root = h("div"), results = h("div"), df = dataFields({ bars: 500 });
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

// ----- walk-forward -----
function walkForwardView() {
  const root = h("div"), results = h("div"), df = dataFields({ bars: 800 });
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
