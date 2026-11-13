"use strict";
/* Learning and AI: prediction models, regimes and change points. */

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
