(() => {
  "use strict";

  const data = window.BENCHMARK_DATA;
  if (!data) {
    document.body.innerHTML = '<p class="empty">Dashboard data is missing. Run <code>uv run python scripts/build_dashboard.py</code>.</p>';
    return;
  }

  const $ = (selector) => document.querySelector(selector);
  const number = (value, digits = 3) => value == null ? "—" : Number(value).toFixed(digits);
  const percent = (value) => value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
  const latency = (value) => value == null ? "—" : `${Number(value).toFixed(value < 10 ? 2 : 0)} ms`;
  const escape = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[char]);

  const count = Object.values(data.counts).reduce((total, value) => total + value, 0);
  $("#snapshot-count").textContent = `${count} recorded runs · ${data.retrieval.length} retrieval labels`;
  $("#snapshot-date").textContent = data.generated_at ? `latest evidence ${new Date(data.generated_at).toLocaleDateString()}` : "no dated evidence";

  const shippedRetrieval = data.retrieval.find((run) => run.label === "compress-embedding-d20") || data.retrieval.at(-1) || {};
  const shippedAnswer = data.answers.find((run) => run.label === "ragas-d20-a0") || data.answers.at(-1) || {};
  const exact = data.cache.find((run) => run.label === "cache-exact") || {};
  const metrics = [
    ["Recall@5", number(shippedRetrieval.aggregate?.["recall@5"]), shippedRetrieval.label || "not measured"],
    ["Recall@10", number(shippedRetrieval.aggregate?.["recall@10"]), "retrieval, document-level"],
    ["Answer relevancy", number(shippedAnswer.ragas?.relevancy), shippedAnswer.ragas ? "RAGAS judge" : "not measured"],
    ["Exact cache p50", latency(exact.hit_p50_ms), exact.rule_e ? "Rule E passed" : "not measured"],
  ];
  $("#headline-metrics").innerHTML = metrics.map(([label, value, note]) => `<article class="metric"><span class="label">${escape(label)}</span><strong class="value">${escape(value)}</strong><span class="note">${escape(note)}</span></article>`).join("");

  const modes = [...new Set(data.retrieval.map((run) => run.config?.mode || "dense"))].sort();
  $("#mode-filter").insertAdjacentHTML("beforeend", modes.map((mode) => `<option value="${escape(mode)}">${escape(mode)}</option>`).join(""));

  let selectedLabel = shippedRetrieval.label;
  const filteredRuns = () => {
    const query = $("#run-search").value.trim().toLowerCase();
    const mode = $("#mode-filter").value;
    return data.retrieval.filter((run) => {
      const runMode = run.config?.mode || "dense";
      return (!query || run.label.toLowerCase().includes(query)) && (mode === "all" || runMode === mode);
    });
  };

  function renderPlot(runs, metric) {
    const svg = $("#retrieval-plot");
    const points = runs.filter((run) => Number.isFinite(run.aggregate?.[metric]) && Number.isFinite(run.latency_p50_ms));
    $("#plot-title").textContent = `${metric.replace("@", " @ ")} vs p50 latency`;
    $("#plot-count").textContent = `${points.length} plotted`;
    if (!points.length) {
      svg.innerHTML = '<text x="480" y="220" text-anchor="middle" class="axis-label">No matching measured runs</text>';
      return;
    }
    const left = 78, right = 925, top = 35, bottom = 385;
    const maxX = Math.max(...points.map((run) => run.latency_p50_ms), 1);
    const minY = Math.min(...points.map((run) => run.aggregate[metric]));
    const maxY = Math.max(...points.map((run) => run.aggregate[metric]));
    const yFloor = Math.max(0, Math.floor((minY - .04) * 10) / 10);
    const yCeil = Math.min(1, Math.ceil((maxY + .04) * 10) / 10);
    const x = (value) => left + (Math.log10(value + 1) / Math.log10(maxX + 1)) * (right - left);
    const y = (value) => bottom - ((value - yFloor) / Math.max(yCeil - yFloor, .1)) * (bottom - top);
    const grid = [0, .25, .5, .75, 1].map((part) => {
      const gy = top + part * (bottom - top);
      const value = yCeil - part * (yCeil - yFloor);
      return `<line x1="${left}" y1="${gy}" x2="${right}" y2="${gy}" class="grid-line"/><text x="${left - 12}" y="${gy + 4}" text-anchor="end" class="axis-label">${value.toFixed(2)}</text>`;
    }).join("");
    const circles = points.map((run) => `<g><circle tabindex="0" role="button" aria-label="${escape(run.label)}, ${metric} ${number(run.aggregate[metric])}, ${latency(run.latency_p50_ms)}" data-label="${escape(run.label)}" cx="${x(run.latency_p50_ms)}" cy="${y(run.aggregate[metric])}" r="${run.label === selectedLabel ? 9 : 6}" class="point"><title>${escape(run.label)} · ${number(run.aggregate[metric])} · ${latency(run.latency_p50_ms)}</title></circle></g>`).join("");
    svg.innerHTML = `${grid}<line x1="${left}" y1="${bottom}" x2="${right}" y2="${bottom}" class="axis"/><line x1="${left}" y1="${top}" x2="${left}" y2="${bottom}" class="axis"/><text x="${(left + right) / 2}" y="425" text-anchor="middle" class="axis-label">p50 latency (log scale)</text>${circles}`;
    svg.querySelectorAll(".point").forEach((point) => {
      const select = () => selectRun(point.dataset.label);
      point.addEventListener("click", select);
      point.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") select(); });
    });
  }

  function renderRows(runs) {
    const body = $("#retrieval-rows");
    body.innerHTML = runs.map((run) => `<tr tabindex="0" data-label="${escape(run.label)}" class="${run.label === selectedLabel ? "selected" : ""}"><td>${escape(run.label)}</td><td>${escape(run.config?.mode || "dense")}</td><td>${number(run.aggregate?.["recall@5"])}</td><td>${number(run.aggregate?.["recall@10"])}</td><td>${number(run.aggregate?.mrr)}</td><td>${latency(run.latency_p50_ms)}</td><td><code>${escape(run.git_commit || "—")}</code></td></tr>`).join("") || '<tr><td colspan="7" class="empty">No matching runs</td></tr>';
    body.querySelectorAll("tr[data-label]").forEach((row) => {
      const select = () => selectRun(row.dataset.label);
      row.addEventListener("click", select);
      row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") select(); });
    });
  }

  function renderCategory(run) {
    if (!run) { $("#category-detail").innerHTML = '<p class="empty">Select a measured run.</p>'; return; }
    const categories = Object.entries(run.per_category || {});
    const metric = $("#metric-filter").value;
    $("#category-detail").innerHTML = `<h3>${escape(run.label)} / category ${escape(metric)}</h3><div class="category-bars">${categories.map(([name, values]) => `<div class="category-bar"><strong>${escape(name)}</strong><span class="track"><span class="fill" style="width:${Math.max(0, Math.min(100, Number(values[metric] || 0) * 100))}%"></span></span><small>${number(values[metric])} · ${Number(values.questions || 0).toFixed(0)} questions</small></div>`).join("")}</div>`;
  }

  function selectRun(label) {
    selectedLabel = label;
    renderRetrieval();
  }

  function renderRetrieval() {
    const runs = filteredRuns();
    const metric = $("#metric-filter").value;
    renderPlot(runs, metric);
    renderRows(runs);
    renderCategory(data.retrieval.find((run) => run.label === selectedLabel));
  }
  ["#run-search", "#mode-filter", "#metric-filter"].forEach((selector) => $(selector).addEventListener("input", renderRetrieval));
  renderRetrieval();

  $("#answer-rows").innerHTML = data.answers.map((run) => `<tr><td>${escape(run.label)}</td><td>${number(run.ragas?.faithfulness)}</td><td>${number(run.ragas?.relevancy)}</td><td>${percent(run.refusal_rate)}</td><td>${run.prompt_tokens_p50 == null ? "—" : Number(run.prompt_tokens_p50).toFixed(0)}</td><td>${latency(run.latency_p50_ms)}</td></tr>`).join("") || '<tr><td colspan="6" class="empty">No answer runs</td></tr>';

  const cold = data.cache.find((run) => run.label === "cache-cold") || {};
  const trace = data.observability.at(-1) || {};
  const saving = cold.usd_at_list_price && exact.usd_at_list_price ? 1 - exact.usd_at_list_price / cold.usd_at_list_price : null;
  const systemCards = [
    ["Exact cache", exact.rule_e ? "PASS" : "—", `${exact.hits ?? 0} identical hits; ${latency(exact.hit_p50_ms)} p50.`, "accent"],
    ["Generation cost", saving == null ? "—" : percent(saving), "Measured reduction from the cold answer arm to exact-cache traffic.", "orange"],
    ["Trace overhead", latency(trace.p95_ms), `${trace.valid_lines ?? 0}/${trace.writes ?? 0} valid records; Rule O ${trace.rule_o ? "passed" : "unmeasured"}.`, ""],
  ];
  $("#system-cards").innerHTML = systemCards.map(([title, value, copy, style]) => `<article class="system-card ${style}"><span class="kicker">${escape(title)}</span><strong class="big">${escape(value)}</strong><p>${escape(copy)}</p></article>`).join("");
})();
