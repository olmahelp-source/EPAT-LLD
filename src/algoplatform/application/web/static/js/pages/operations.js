"use strict";
/* Operations and control: strategy approvals, alerts and inbox, the audit log, health and jobs. */

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
    const sel = h("select"), params = h("div", { class: "row" }), df = dataFields({ symbols: "AAA" }), trials = h("input", { type: "number", value: 1, min: 1, style: "width:80px" });
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
