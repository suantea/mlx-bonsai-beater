"use strict";
const $ = (id) => document.getElementById(id);
const state = { model: "", route: "", abort: null, msgs: [] };

const chat = $("chat"), input = $("input"), sendBtn = $("send"), stopBtn = $("stop");

// ---------- route heartbeat ----------
async function pollStatus() {
  try {
    const r = await fetch("/api/status");
    const j = await r.json();
    const { ok } = j;
    $("route-dot").className = ok ? "dot ok" : "dot bad";
  } catch { $("route-dot").className = "dot bad"; }
}
setInterval(pollStatus, 5000); pollStatus();

async function loadModels() {
  const sel = $("model-select");
  try {
    const r = await fetch("/v1/models");
    const j = await r.json();
    const models = (j.data || []).map((m) => m.id);
    state.route = r.headers.get("X-Proxied-Route") || "";
    updateBadge();
    if (!models.length) throw 0;
    sel.innerHTML = "";
    for (const m of models) {
      const o = document.createElement("option");
      o.value = m; o.textContent = m;
      sel.appendChild(o);
    }
    state.model = models[0];
  } catch {
    sel.innerHTML = '<option>— 模型服务不可用 —</option>';
    updateBadge(true);
  }
}
function updateBadge(dead) {
  const b = $("route-badge"), label = $("route-label"), dot = $("route-dot");
  if (dead) {
    b.textContent = "服务离线"; b.className = "badge"; dot.className = "dot bad";
    label.textContent = "调度中心不可达";
    return;
  }
  const primary = !state.route || state.route === "primary";
  b.textContent = primary ? "主：对端" : "备：本机";
  b.className = "badge " + (primary ? "primary" : "backup");
  dot.className = "dot " + (primary ? "ok" : "ok");
  label.textContent = primary ? "主路由 · 对端在线" : "兜底路由 · 本机";
}
$("model-select").addEventListener("change", (e) => (state.model = e.target.value));

// ---------- markdown-lite renderer ----------
function render(text) {
  let t = String(text);
  t = t.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const fence = /```(\w*)\n([\s\S]*?)```/g;
  const blocks = [];
  t = t.replace(fence, (m, lang, code) => {
    blocks.push(`<pre><code>${code.replace(/^\n+|\n+$/g, "")}</code></pre>`);
    return `\x00BLOCK${blocks.length - 1}\x00`;
  });
  t = t.replace(/^### (.*)$/gm, "<h3>$1</h3>");
  t = t.replace(/^## (.*)$/gm, "<h2>$1</h2>");
  t = t.replace(/^# (.*)$/gm, "<h1>$1</h1>");
  t = t.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  t = t.replace(/^\s*[-*] (.*)$/gm, "• $1");
  t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  t = t.replace(/\x00BLOCK(\d+)\x00/g, (m, i) => blocks[+i]);
  return t;
}

function addMsg(role, text, streaming) {
  const el = document.createElement("div");
  el.className = "msg " + role;
  el.innerHTML = `<div class="who">${role === "user" ? "你" : "助手"}</div>
    <div class="bubble"></div>`;
  const bubble = el.querySelector(".bubble");
  bubble.innerHTML = render(text);
  if (streaming) {
    bubble.insertAdjacentHTML("beforeend", '<span class="cursor"></span>');
    el.dataset.stream = "1";
  }
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return bubble;
}

// ---------- send ----------
async function send() {
  const text = input.value.trim();
  if (!text || !input.dataset.ok) return;
  const msgs = [
    ...(state.msgs.length ? state.msgs : []),
    { role: "user", content: text },
  ];
  addMsg("user", text, false);
  const bubble = addMsg("assistant", "", true);
  input.value = ""; input.style.height = "auto";
  sendBtn.classList.add("hidden"); stopBtn.classList.remove("hidden");

  const ac = new AbortController();
  state.abort = ac;
  try {
    const resp = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: state.model,
        messages: [{ role: "system", content: $("system-prompt").value || "你是一个乐于助人的助手。" }, ...msgs],
        temperature: parseFloat($("temp").value),
        max_tokens: parseInt($("max-tokens").value) || 1024,
        stream: true,
      }),
      signal: ac.signal,
    });
    const route = resp.headers.get("X-Proxied-Route");
    if (route) { state.route = route; updateBadge(); }
    if (!resp.ok || !resp.body) throw new Error("HTTP " + resp.status);

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = "", acc = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, idx).trim(); buf = buf.slice(idx + 1);
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        try {
          const j = JSON.parse(payload);
          const delta = j.choices?.[0]?.delta?.content || "";
          acc += delta;
          bubble.innerHTML = render(acc) + '<span class="cursor"></span>';
          chat.scrollTop = chat.scrollHeight;
        } catch {}
      }
    }
    if (acc) {
      bubble.innerHTML = render(acc);
      state.msgs = [...msgs, { role: "assistant", content: acc }];
    }
  } catch (e) {
    if (e.name !== "AbortError")
      bubble.innerHTML = render("⚠️ 请求失败：" + e.message);
  } finally {
    state.abort = null;
    sendBtn.classList.remove("hidden"); stopBtn.classList.add("hidden");
    if (!bubble.querySelector(".cursor")) input.focus();
    else input.focus();
  }
}

// ---------- events ----------
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 160) + "px";
  input.dataset.ok = "1";
});
sendBtn.addEventListener("click", send);
stopBtn.addEventListener("click", () => state.abort?.abort());
$("new-chat").addEventListener("click", () => {
  state.msgs = [];
  chat.innerHTML = '<div style="color:var(--muted);text-align:center;margin:60px 0;">新的对话开始</div>';
});
loadModels();