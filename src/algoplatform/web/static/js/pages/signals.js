"use strict";
/* Signal libraries: the technical lab and news sentiment. */

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
