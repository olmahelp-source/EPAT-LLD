"use strict";
/* Market and reference data: price data (sample generator, downloads, history library, uploads), data providers,
   the instrument master. */

// ----- market and reference data: price data (sample generator, downloads, history library, uploads) -----
function dataView() {
  const root = h("div");
  const draw = () => {
    const st = store.sources || { sample_only: true, providers: [], library: [], scenarios: {}, uploaded: [] };
    const connected = (st.providers || []).filter((p) => p.connected);
    root.replaceChildren(
      h("div", { class: "banner" + (st.sample_only ? " info" : "") },
        st.sample_only ? h("strong", null, "Sample-data-only mode") : h("strong", null, connected.length ? "Real data available" : "No data provider connected"),
        h("span", null, st.sample_only ? " — operations has locked the platform to generated data: downloads, downloaded history and uploads are off."
          : connected.length ? ` — connected: ${connected.map((p) => p.label).join(", ")}. ${st.library.length} series downloaded.`
          : " — sample data and your own uploads work; operations connects Alpaca or Interactive Brokers on the Data providers page."),
        st.can_configure ? h("a", { class: "btn ghost", href: "#/providers" }, "Data providers") : null),
      sampleCard(st), downloadCard(st, connected, draw), libraryCard(st, draw), uploadCard(st, draw));
  };
  draw();
  refreshSources().then(draw);
  return root;
}

function sampleCard(st) {
  const out = h("div");
  const scenario = h("select", { "aria-label": "scenario" }, Object.entries(st.scenarios || {}).map(([k]) => h("option", { value: k }, k.replaceAll("_", " "))));
  if (sampleScenario()) scenario.value = sampleScenario();                      // start on the scenario chosen in the header
  const desc = h("p", { class: "hint" });
  const syncDesc = () => { desc.textContent = (st.scenarios || {})[scenario.value] || ""; };
  scenario.addEventListener("change", syncDesc);
  syncDesc();
  const symbols = h("input", { value: "AAA,BBB,CCC", style: "width:160px", "aria-label": "symbols" });
  const bars = h("input", { type: "number", value: 756, min: 60, max: 3000, style: "width:90px" });
  const seed = h("input", { type: "number", value: 7, style: "width:80px" });
  const run = h("button", { class: "btn", type: "submit" }, "Preview");
  return h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(run, async () => {
    const r = await api("/api/data/sample/preview", { scenario: scenario.value, symbols: symbols.value.split(",").map((x) => x.trim()).filter(Boolean),
      bars: Number(bars.value), seed: Number(seed.value) });
    const names = Object.keys(r.series);
    out.replaceChildren(
      lineChart(names.slice(0, 2).map((n, i) => ({ name: n, cls: "s" + i, fcls: "f" + i, points: r.series[n] })), { legend: true, label: "sample prices, rebased to 100" }),
      table([{ key: "symbol", label: "Symbol" }, { key: "total_return", label: "Total return", tone: true, render: (v) => fmt.pct(v, 1) },
        { key: "volatility", label: "Volatility / year", render: (v) => fmt.pct(v, 1) }, { key: "max_drawdown", label: "Max drawdown", render: (v) => fmt.pct(v, 1) },
        { key: "last", label: "Last price", render: (v) => fmt.num(v) }], r.stats),
      note("The chart shows the first two symbols, rebased to 100. To use this scenario on any page, set Data to Sample data and pick it under Scenario."));
  }); } }, h("h2", null, "Sample data generator"),
    h("div", { class: "row" }, field("Scenario", scenario), field("Symbols", symbols), field("Bars", bars), field("Seed", seed), run), desc,
    note("Generated, reproducible prices (the same scenario, seed and symbol always give the same series). Good for learning the platform, testing a strategy's mechanics and seeing how it behaves in a crash or a sideways market; it says nothing about any real market."), out);
}

function downloadCard(st, connected, redraw) {
  const today = new Date().toISOString().slice(0, 10);
  const provider = h("select", { "aria-label": "provider" }, connected.map((p) => h("option", { value: p.name }, p.label)));
  const symbols = h("input", { value: "SPY,AAPL,MSFT", style: "width:200px", "aria-label": "symbols to download" });
  const timeframe = h("select", { "aria-label": "bar size" }, (st.timeframes || ["1Day"]).map((t) => h("option", { value: t }, t)));
  timeframe.value = "1Day";
  const start = h("input", { type: "date", value: String(new Date().getFullYear() - 3) + today.slice(4), "aria-label": "from" });
  const end = h("input", { type: "date", value: today, "aria-label": "to" });
  const run = h("button", { class: "btn", type: "submit" }, "Download");
  const report = h("div");
  const off = st.sample_only || !connected.length || !st.can_fetch;
  run.disabled = off;
  return h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(run, async () => {
    const r = await api("/api/data/fetch", { provider: provider.value, symbols: symbols.value, timeframe: timeframe.value, start: start.value, end: end.value });
    const ok = Object.keys(r.stored), bad = Object.entries(r.errors);
    toast(`Downloaded ${ok.length} series` + (bad.length ? `, ${bad.length} failed` : ""), bad.length > 0 && !ok.length);
    await refreshSources();
    redraw();
    if (bad.length) toast(bad.map(([k, v]) => `${k}: ${v}`).join(" · "), true);
  }); } }, h("h2", null, "Download history from a data provider"),
    h("div", { class: "row" }, field("Provider", provider), field("Symbols", symbols), field("Bar size", timeframe), field("From", start), field("To", end), run),
    note(st.sample_only ? "Switched off: the platform is in sample-data-only mode."
      : !connected.length ? "No provider is connected yet. Operations connects Alpaca (API keys) or Interactive Brokers (TWS or IB Gateway) on the Data providers page; until then use sample data or upload a CSV."
      : "Bars are split- and dividend-adjusted where the provider offers it, checked and cleaned, then kept in a library everyone on the platform can use. Downloading the same symbol again from the same provider adds to it. Pages use the daily (1Day) series; intraday bars are kept for research in code."),
    report);
}

function libraryCard(st, redraw) {
  const chart = h("div");
  const remove = (btn, row) => busy(btn, async () => { await api("/api/data/history/remove", { symbol: row.symbol, timeframe: row.timeframe }); await refreshSources(); redraw(); });
  const show = (btn, row) => busy(btn, async () => {
    const r = await api("/api/data/history/series", { symbol: row.symbol, timeframe: row.timeframe });
    chart.replaceChildren(h("h3", { style: "margin-top:14px" }, `${r.symbol} close`), lineChart([{ name: r.symbol, cls: "s0", fcls: "f0", points: r.close }], { label: r.symbol + " close" }),
      note(`Total return ${fmt.pct(r.stats.total_return, 1)}, volatility ${fmt.pct(r.stats.volatility, 1)} a year, worst drawdown ${fmt.pct(r.stats.max_drawdown, 1)}.`));
  });
  return h("div", { class: "card" }, h("h2", null, "History library"),
    table([{ key: "symbol", label: "Symbol" }, { key: "timeframe", label: "Bars" }, { key: "provider", label: "From" },
      { key: "start", label: "First", render: (v) => String(v).slice(0, 10) }, { key: "end", label: "Last", render: (v) => String(v).slice(0, 10) },
      { key: "rows", label: "Rows" }, { key: "last_close", label: "Last close", render: (v) => fmt.num(v) },
      { key: "cleaning", label: "Repaired", render: (v) => String(Object.values(v || {}).reduce((a, b) => a + b, 0)) },
      { key: "symbol", label: "", render: (_, row) => h("div", { class: "row" },
        st.sample_only ? null : h("button", { class: "btn ghost", type: "button", onclick: (e) => show(e.currentTarget, row) }, "Chart"),
        st.can_fetch ? h("button", { class: "btn ghost", type: "button", onclick: (e) => remove(e.currentTarget, row) }, "Remove") : null) }],
      st.library || [], "Nothing downloaded yet."),
    note("Shared by every user. \"Repaired\" counts bars the cleaner fixed (high/low envelope, invalid or duplicate bars, isolated bad ticks)."), chart);
}

function uploadCard(st, redraw) {
  const listBox = h("div");
  const file = h("input", { type: "file", accept: ".csv,text/csv", "aria-label": "CSV file" });
  const sym = h("input", { placeholder: "symbol", style: "width:110px", "aria-label": "symbol" });
  file.addEventListener("change", () => {
    if (file.files[0] && !sym.value) sym.value = file.files[0].name.replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9]/g, "").slice(0, 8).toUpperCase();
  });
  const up = h("button", { class: "btn", type: "submit" }, "Upload");
  const clear = h("button", { class: "btn ghost", type: "button" }, "Remove all my data");
  up.disabled = !!st.sample_only;
  const refresh = async () => {
    const rows = await api("/api/data");
    listBox.replaceChildren(table([{ key: "symbol", label: "Symbol" }, { key: "rows", label: "Rows" }, { key: "start", label: "From", render: (v) => String(v).slice(0, 10) },
      { key: "end", label: "To", render: (v) => String(v).slice(0, 10) }], rows, "Nothing uploaded yet."));
  };
  refresh().catch((e) => toast(e.message, true));
  clear.addEventListener("click", () => busy(clear, async () => { await api("/api/data/clear", {}); await refreshSources(); await refresh(); toast("Uploaded data removed"); }));
  return h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(up, async () => {
    const f = file.files[0];
    if (!f) throw new Error("Choose a CSV file first");
    if (f.size > 900000) throw new Error("File too large (limit about 900 KB, 5,000 rows)");
    const r = await api("/api/data/upload", { symbol: sym.value, csv: await f.text() });
    toast(`Uploaded ${r.symbol}: ${r.rows} rows`);
    file.value = ""; sym.value = "";
    await refreshSources();
    await refresh();
  }); } },
    h("h2", null, "Upload your own price data"),
    h("p", { class: "hint" }, "CSV with a header row and columns date, open, high, low, close and optionally volume (one file per symbol, up to 5,000 rows, 10 symbols). Rows are checked for duplicate dates, gaps in ordering, non-positive prices and impossible highs/lows. Then choose \"Uploaded CSV\" as the data on any research or validation page. Your files stay in server memory and are private to you." + (st.sample_only ? " Uploads are switched off in sample-data-only mode." : "")),
    h("div", { class: "row" }, field("File", file), field("Symbol", sym), up, clear), listBox);
}

// ----- market and reference data: data providers (operations) -----
function providersView() {
  const root = h("div");
  const draw = (st) => {
    store.sources = st;
    if (shell.dataSwitch) shell.dataSwitch.replaceChildren(...dataSwitchParts());
    const toggle = h("button", { class: st.sample_only ? "btn" : "btn ghost", type: "button" }, st.sample_only ? "Allow real data" : "Lock to sample data");
    toggle.addEventListener("click", () => busy(toggle, async () => { draw(await api("/api/data/sample_only", { on: !st.sample_only })); toast(st.sample_only ? "Real data allowed" : "Locked to sample data"); }));
    root.replaceChildren(
      h("div", { class: "card" }, h("h2", null, "Sample-data-only mode"),
        h("div", { class: "row", style: "align-items:center" }, h("span", { class: "badge " + (st.sample_only ? "lock" : "ok") }, st.sample_only ? "ON: generated data only" : "OFF: real data allowed"), toggle),
        note("When on, every page runs on generated data for everyone: downloading, downloaded history, uploads and history replay in paper trading are refused by the server. Use it for training, demos, or until a provider is set up. Each user can still choose sample or real data with the switch in the header while it is off.")),
      ...st.providers.map((p) => providerCard(p, draw)),
      Object.keys(st.environment || {}).length ? h("div", { class: "card" }, h("h2", null, "Connected at start-up"),
        table([{ key: "k", label: "Provider" }, { key: "v", label: "Result" }], Object.entries(st.environment).map(([k, v]) => ({ k, v })))) : "",
      h("div", { class: "card" }, h("h2", null, "Connecting from the server's environment"),
        note("Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY (or Alpaca's own APCA_API_KEY_ID / APCA_API_SECRET_KEY), and optionally ALPACA_DATA_FEED, before starting the server; or IB_HOST, IB_PORT and IB_CLIENT_ID for Interactive Brokers. Keys entered on this page are kept only in the server's memory (encrypted when the cryptography package is installed) and must be entered again after a restart; keys from the environment come back by themselves.")));
  };
  api("/api/data/providers").then(draw).catch((e) => toast(e.message, true));
  return root;
}

function providerCard(p, draw) {
  const inputs = {};
  const mk = (key, label, attrs, def) => {
    const inp = attrs.options ? h("select", { "aria-label": label }, attrs.options.map((o) => h("option", { value: o }, o)))
      : h("input", { "aria-label": label, autocomplete: "off", spellcheck: "false", ...attrs });
    inp.value = (p.settings && p.settings[key] !== undefined ? p.settings[key] : def) ?? "";
    inputs[key] = inp;
    return field(label, inp);
  };
  const fields = p.name === "alpaca"
    ? [mk("key_id", "API key ID", { style: "width:200px" }), mk("secret_key", "Secret key", { type: "password", style: "width:220px" }),
       mk("feed", "Feed", { options: ["iex", "sip", "delayed_sip"] }, "iex"), mk("adjustment", "Adjustment", { options: ["all", "split", "dividend", "raw"] }, "all")]
    : [mk("host", "Host", { style: "width:140px" }, "127.0.0.1"), mk("port", "Port", { type: "number", style: "width:90px" }, 7497),
       mk("client_id", "Client id", { type: "number", style: "width:80px" }, 17)];
  const connect = h("button", { class: "btn", type: "submit" }, p.connected ? "Reconnect" : "Connect");
  const test = h("button", { class: "btn ghost", type: "button" }, "Test");
  const disconnect = h("button", { class: "btn ghost", type: "button" }, "Disconnect");
  test.addEventListener("click", () => busy(test, async () => {
    const r = await api("/api/data/providers/test", { name: p.name });
    toast(r.result.ok ? `${p.label} works: ${r.result.bars} bars of ${r.result.symbol}, last close ${fmt.num(r.result.last_close)}` : `${p.label}: ${r.result.error}`, !r.result.ok);
    draw(r);
  }));
  disconnect.addEventListener("click", () => busy(disconnect, async () => { draw(await api("/api/data/providers/disconnect", { name: p.name })); toast(p.label + " disconnected"); }));
  const last = p.status || {};
  return h("form", { class: "card", onsubmit: (e) => { e.preventDefault(); busy(connect, async () => {
    const secrets = {}, settings = {};
    for (const k of p.secret_fields) secrets[k] = inputs[k].value.trim();
    for (const k of Object.keys(inputs)) if (!p.secret_fields.includes(k)) settings[k] = inputs[k].type === "number" ? Number(inputs[k].value) : inputs[k].value;
    const r = await api("/api/data/providers/connect", { name: p.name, settings, secrets });
    for (const k of p.secret_fields) inputs[k].value = "";                     // never keep a key in the page
    toast(p.label + " connected: use Test to check the keys");
    draw(r);
  }); } },
    h("h2", null, p.label, " ", h("span", { class: "badge " + (p.connected ? "ok" : "") }, p.connected ? "connected" : "not connected")),
    note(p.help),
    p.connected ? note(`Connected by ${p.settings.connected_by} at ${when(p.settings.connected_at)}.` + (last.at ? ` Last test ${when(last.at)}: ${last.ok ? "ok" : "failed — " + (last.error || "")}` : "")
      + (last.last_fetch ? ` Last download ${when(last.last_fetch)}: ${last.loaded} loaded, ${last.failed} failed.` : "")) : null,
    h("div", { class: "row" }, ...fields, connect, p.connected ? test : null, p.connected ? disconnect : null),
    p.name === "alpaca" ? note("Keys are sent once to the server, kept in its credential vault and never shown again. The free Alpaca plan gives the IEX feed; SIP needs a market-data subscription.")
      : note("Start TWS or IB Gateway, enable the API (Configure › API › Settings) and allow this server's address. Paper TWS listens on 7497, live on 7496; IB Gateway on 4002 / 4001. The server needs the optional ib_async package."));
}

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
