(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const number = new Intl.NumberFormat("en", { maximumFractionDigits: 1 });
  const integer = new Intl.NumberFormat("en", { maximumFractionDigits: 0 });
  const history = [];
  const conversationsKey = "rag-conversations";
  let conversations = [];
  let activeConversationId = "";
  let busy = false;

  const stored = (key) => {
    try { return sessionStorage.getItem(key) || ""; } catch { return ""; }
  };
  const remember = (key, value) => {
    try { value ? sessionStorage.setItem(key, value) : sessionStorage.removeItem(key); } catch { /* session storage is optional */ }
  };
  const apiBase = () => (stored("rag-api-url") || window.location.origin).replace(/\/$/, "");
  const apiUrl = (path) => `${apiBase()}${path}`;
  const percent = (value) => `${number.format(value * 100)}%`;
  const latency = (value) => value == null ? "--" : `${integer.format(value)} ms`;

  function readConversations() {
    try {
      const value = JSON.parse(localStorage.getItem(conversationsKey) || "[]");
      return Array.isArray(value) ? value : [];
    } catch { return []; }
  }

  function writeConversations() {
    try { localStorage.setItem(conversationsKey, JSON.stringify(conversations.slice(0, 24))); } catch { /* local storage is optional */ }
  }

  function setText(selector, value) {
    const element = $(selector);
    if (element) element.textContent = value;
  }

  function message(role, text, options = {}) {
    const article = document.createElement("article");
    article.className = `message ${role}-message${options.error ? " error-message" : ""}${options.thinking ? " thinking" : ""}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = role === "user" ? "Y" : "R";

    const content = document.createElement("div");
    content.className = "message-content";
    const author = document.createElement("span");
    author.className = "message-author";
    author.textContent = role === "user" ? "You" : "Assistant";
    const body = document.createElement("p");
    body.textContent = text;
    content.append(author, body);

    if (options.answer) {
      const meta = document.createElement("div");
      meta.className = "answer-meta";
      const items = [
        latency(options.answer.latency_ms),
        options.answer.model || "model unknown",
        options.answer.cache ? `${options.answer.cache} cache` : "generated",
        `${options.answer.retrieval?.used ?? 0} chunks used`,
      ];
      items.forEach((item) => {
        const chip = document.createElement("span");
        chip.textContent = item;
        meta.append(chip);
      });
      content.append(meta);

      if (options.answer.sources?.length) {
        const sources = document.createElement("div");
        sources.className = "sources";
        options.answer.sources.forEach((source) => {
          const label = `[${source.index}] ${source.title || source.document_id}${source.section ? ` / ${source.section}` : ""}`;
          if (source.url) {
            const link = document.createElement("a");
            link.href = source.url;
            link.target = "_blank";
            link.rel = "noreferrer";
            link.textContent = label;
            sources.append(link);
          } else {
            const item = document.createElement("span");
            item.textContent = label;
            sources.append(item);
          }
        });
        content.append(sources);
      }
    }

    article.append(avatar, content);
    $("#messages").append(article);
    $("#messages").scrollTop = $("#messages").scrollHeight;
    return article;
  }

  function renderConversationList() {
    const list = $("#conversation-list");
    list.innerHTML = "";
    if (!conversations.length) {
      const empty = document.createElement("p");
      empty.className = "history-empty";
      empty.textContent = "Your conversations will appear here.";
      list.append(empty);
      return;
    }
    conversations.forEach((conversation) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `conversation-item${conversation.id === activeConversationId ? " active" : ""}`;
      button.dataset.conversationId = conversation.id;
      const title = document.createElement("strong");
      title.textContent = conversation.title || "New conversation";
      const date = document.createElement("small");
      date.textContent = new Date(conversation.updatedAt).toLocaleDateString([], { month: "short", day: "numeric" });
      button.append(title, date);
      list.append(button);
    });
  }

  function saveActiveConversation() {
    const firstQuestion = history.find((item) => item.role === "user")?.content || "New conversation";
    const conversation = {
      id: activeConversationId,
      title: firstQuestion.length > 42 ? `${firstQuestion.slice(0, 42)}…` : firstQuestion,
      updatedAt: new Date().toISOString(),
      messages: [...history],
    };
    conversations = [conversation, ...conversations.filter((item) => item.id !== activeConversationId)];
    writeConversations();
    renderConversationList();
  }

  function openConversation(id) {
    const conversation = conversations.find((item) => item.id === id);
    if (!conversation) return;
    activeConversationId = conversation.id;
    history.splice(0, history.length, ...(conversation.messages || []));
    $("#messages").innerHTML = "";
    history.forEach((item) => message(item.role, item.content));
    $("#suggestions").hidden = history.length > 0;
    renderConversationList();
    showWorkspace("chat");
  }

  function startConversation() {
    activeConversationId = globalThis.crypto?.randomUUID?.() || `${Date.now()}`;
    history.length = 0;
    $("#messages").innerHTML = "";
    message("assistant", "New conversation started. What would you like to explore?");
    $("#suggestions").hidden = false;
    renderConversationList();
    showWorkspace("chat");
    $("#question").focus();
  }

  async function responseError(response) {
    try {
      const payload = await response.json();
      if (Array.isArray(payload.detail)) return payload.detail.map((item) => item.msg).join("; ");
      return payload.detail || `Request failed (${response.status})`;
    } catch {
      return `Request failed (${response.status})`;
    }
  }

  async function ask(question) {
    const key = stored("rag-api-key");
    if (!key) {
      openSettings();
      setTimeout(() => $("#api-key").focus(), 50);
      return;
    }

    busy = true;
    $(".send-button").disabled = true;
    $("#suggestions").hidden = true;
    message("user", question);
    const pending = message("assistant", "Searching the knowledge base", { thinking: true });

    try {
      const response = await fetch(apiUrl("/query"), {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": key },
        body: JSON.stringify({ question, history }),
      });
      if (!response.ok) throw new Error(await responseError(response));
      const answer = await response.json();
      pending.remove();
      message("assistant", answer.answer, { answer });
      history.push({ role: "user", content: question }, { role: "assistant", content: answer.answer });
      saveActiveConversation();
      refreshDashboard();
    } catch (error) {
      pending.remove();
      const hint = error instanceof TypeError ? "Could not reach the API. Check the backend URL and its CORS policy." : error.message;
      message("assistant", hint, { error: true });
    } finally {
      busy = false;
      $(".send-button").disabled = false;
      $("#question").focus();
    }
  }

  function renderHealth(health, reachable) {
    const state = $("#service-state");
    state.classList.toggle("online", Boolean(reachable && health?.status === "ok"));
    state.classList.toggle("offline", !reachable || health?.status !== "ok");
    state.querySelector("strong").textContent = reachable && health?.status === "ok" ? "Service online" : "Service unavailable";
    state.querySelector("small").textContent = reachable ? "Live API status" : "Check connection settings";
    setText("#qdrant-state", health?.qdrant ? "Ready" : "Unavailable");
    setText("#redis-state", health?.redis === null ? "Disabled" : health?.redis ? "Ready" : "Unavailable");
  }

  function renderMetrics(metrics) {
    const total = metrics.total || 0;
    const generated = metrics.generated || 0;
    const refused = metrics.refused || 0;
    const failed = metrics.failed || 0;
    const cacheHits = (metrics.exact_hits || 0) + (metrics.semantic_hits || 0);
    const successRate = total ? generated / total : 0;
    const cacheRate = total ? cacheHits / total : 0;
    const share = (value) => total ? `${value / total * 100}%` : "0%";

    setText("#metric-total", integer.format(total));
    setText("#metric-generated", `${integer.format(generated)} generated answers`);
    setText("#metric-success", total ? percent(successRate) : "--");
    setText("#metric-refused", `${integer.format(refused)} refused / ${integer.format(failed)} failed`);
    setText("#metric-latency", latency(metrics.latency_p50_ms));
    setText("#metric-p95", `p95 ${latency(metrics.latency_p95_ms)}`);
    setText("#side-latency", latency(metrics.latency_p50_ms));
    setText("#metric-cache", total ? percent(cacheRate) : "--");
    setText("#metric-cache-count", `${integer.format(cacheHits)} cache hits`);
    setText("#outcome-total", `${integer.format(total)} requests`);
    setText("#generated-count", integer.format(generated));
    setText("#refused-count", integer.format(refused));
    setText("#failed-count", integer.format(failed));
    $("#generated-bar").style.width = share(generated);
    $("#refused-bar").style.width = share(refused);
    $("#failed-bar").style.width = share(failed);
    setText("#prompt-tokens", integer.format(metrics.prompt_tokens || 0));
    setText("#completion-tokens", integer.format(metrics.completion_tokens || 0));
    setText("#known-cost", `$${number.format(metrics.known_cost_usd || 0)}`);
    setText("#follow-ups", integer.format(metrics.follow_ups || 0));
  }

  async function refreshDashboard() {
    $("#refresh-dashboard").disabled = true;
    try {
      const [healthResponse, metricsResponse] = await Promise.all([
        fetch(apiUrl("/health")),
        fetch(apiUrl("/metrics")),
      ]);
      const health = await healthResponse.json();
      renderHealth(health, true);
      if (!metricsResponse.ok) throw new Error(await responseError(metricsResponse));
      renderMetrics(await metricsResponse.json());
      setText("#last-updated", `Updated ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`);
    } catch {
      renderHealth(null, false);
      setText("#last-updated", "Connection unavailable");
    } finally {
      $("#refresh-dashboard").disabled = false;
    }
  }

  function renderBenchmarks() {
    const data = window.BENCHMARK_DATA;
    if (!data) {
      $("#benchmark-cards").textContent = "Benchmark snapshot unavailable.";
      return;
    }
    const retrieval = data.retrieval.find((run) => run.label === "compress-embedding-d20") || data.retrieval.at(-1) || {};
    const answers = data.answers.find((run) => run.label === "ragas-d20-a0") || data.answers.at(-1) || {};
    const cache = data.cache.find((run) => run.label === "cache-exact") || {};
    const cards = [
      ["Promoted retrieval", retrieval.aggregate?.["recall@5"]?.toFixed(3) || "--", "Recall@5", "featured"],
      ["Answer quality", answers.ragas?.relevancy?.toFixed(3) || "--", "RAGAS relevancy", ""],
      ["Exact cache", cache.hit_p50_ms == null ? "--" : latency(cache.hit_p50_ms), cache.rule_e ? "Acceptance rule passed" : "Not measured", ""],
    ];
    cards.forEach(([label, value, note, className]) => {
      const card = document.createElement("article");
      card.className = `benchmark-card ${className}`.trim();
      const eyebrow = document.createElement("span");
      eyebrow.className = "eyebrow";
      eyebrow.textContent = label;
      const score = document.createElement("strong");
      score.className = "number";
      score.textContent = value;
      const copy = document.createElement("p");
      copy.textContent = note;
      card.append(eyebrow, score, copy);
      $("#benchmark-cards").append(card);
    });

    const rows = [...data.retrieval].slice(-8).reverse();
    rows.forEach((run) => {
      const row = document.createElement("tr");
      const values = [
        run.label,
        run.config?.mode || "dense",
        run.aggregate?.["recall@5"]?.toFixed(3) || "--",
        run.aggregate?.["recall@10"]?.toFixed(3) || "--",
        run.aggregate?.mrr?.toFixed(3) || "--",
        latency(run.latency_p50_ms),
      ];
      values.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      $("#benchmark-rows").append(row);
    });
    if (data.generated_at) setText("#snapshot-date", `Snapshot ${new Date(data.generated_at).toLocaleDateString()}`);
  }

  function openSettings() {
    $("#api-url").value = stored("rag-api-url");
    $("#api-key").value = stored("rag-api-key");
    showWorkspace("configuration");
  }

  $("#chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const input = $("#question");
    const question = input.value.trim();
    if (!question || busy) return;
    input.value = "";
    input.style.height = "auto";
    ask(question);
  });
  $("#question").addEventListener("input", (event) => {
    event.target.style.height = "auto";
    event.target.style.height = `${Math.min(event.target.scrollHeight, 140)}px`;
  });
  $("#question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("#chat-form").requestSubmit();
    }
  });
  $("#suggestions").addEventListener("click", (event) => {
    if (event.target.matches("button")) ask(event.target.textContent);
  });
  document.querySelectorAll(".new-conversation").forEach((button) => button.addEventListener("click", startConversation));
  $("#conversation-list").addEventListener("click", (event) => {
    const item = event.target.closest(".conversation-item");
    if (item) openConversation(item.dataset.conversationId);
  });
  $("#refresh-dashboard").addEventListener("click", refreshDashboard);
  $("#settings-form").addEventListener("submit", (event) => {
    event.preventDefault();
    remember("rag-api-url", $("#api-url").value.trim().replace(/\/$/, ""));
    remember("rag-api-key", $("#api-key").value.trim());
    refreshDashboard();
    showWorkspace("chat");
  });
  function showWorkspace(id) {
    const workspace = document.getElementById(id) || $("#chat");
    document.querySelectorAll(".workspace").forEach((item) => {
      const selected = item === workspace || item.dataset.parent === workspace.id;
      item.hidden = !selected;
      item.classList.toggle("active", selected);
    });
    document.querySelectorAll(".nav-link").forEach((item) => {
      const selected = item.dataset.view === workspace.id;
      item.classList.toggle("active", selected);
      if (selected) item.setAttribute("aria-current", "page");
      else item.removeAttribute("aria-current");
    });
    setText("#view-title", workspace.dataset.title);
    setText("#view-kicker", workspace.dataset.kicker);
  }

  document.querySelectorAll(".nav-link").forEach((link) => link.addEventListener("click", (event) => {
    event.preventDefault();
    showWorkspace(link.dataset.view);
    window.history.replaceState(null, "", link.hash);
  }));
  window.addEventListener("hashchange", () => showWorkspace(location.hash.slice(1)));

  showWorkspace(location.hash.slice(1) || "chat");
  conversations = readConversations();
  if (conversations.length) openConversation(conversations[0].id);
  else startConversation();
  renderBenchmarks();
  refreshDashboard();
  window.setInterval(refreshDashboard, 30000);
})();
