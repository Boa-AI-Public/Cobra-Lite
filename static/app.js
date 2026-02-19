const hasGateway = window.CLAW_HAS_GATEWAY === true;
const hasAnthropicKey = window.CLAW_HAS_ANTHROPIC_KEY === true;
const defaultGatewayUrl = window.CLAW_DEFAULT_GATEWAY_URL || "http://127.0.0.1:18789";

const keyGate = document.getElementById("key-gate");
const mainApp = document.getElementById("main-app");
const composerDock = document.getElementById("composer-dock");
const chatThread = document.getElementById("chat-thread");
const sessionPane = document.getElementById("session-pane");
const sessionList = document.getElementById("session-list");
const sessionStatus = document.getElementById("session-status");
const newSessionBtn = document.getElementById("new-session-btn");
const executionPane = document.getElementById("execution-pane");
const executionFeed = document.getElementById("execution-feed");
const executionTerminalView = document.getElementById("execution-terminal-view");
const executionGraphView = document.getElementById("execution-graph-view");
const executionTabTerminal = document.getElementById("execution-tab-terminal");
const executionTabGraph = document.getElementById("execution-tab-graph");

const keyForm = document.getElementById("key-form");
const keyInput = document.getElementById("api-key-input");
const keyStatus = document.getElementById("key-status");

const promptForm = document.getElementById("prompt-form");
const promptInput = document.getElementById("prompt-input");
const promptStatus = document.getElementById("prompt-status");
const promptSubmit = promptForm ? promptForm.querySelector(".prompt-submit") : null;

const settingsBtn = document.getElementById("settings-btn");
const settingsModal = document.getElementById("settings-modal");
const closeSettingsBtn = document.getElementById("close-settings");
const settingsForm = document.getElementById("settings-form");
const settingsKeyInput = document.getElementById("settings-key-input");
const settingsStatus = document.getElementById("settings-status");

const authModal = document.getElementById("auth-modal");
const authForm = document.getElementById("auth-form");
const authKeyInput = document.getElementById("auth-key-input");
const authStatus = document.getElementById("auth-status");

let isRunning = false;
let autoScrollEnabled = true;
let unseenEventCount = 0;
let jumpLatestBtn = null;
const ACTIVE_SESSION_STORAGE_KEY = "cobraLite.activeSessionId.v1";

let activeSessionId = null;
let activeSessionMessages = [];
let sessionSummaries = [];
let anthropicConfigured = hasAnthropicKey;

function decodeEscapedSequences(text) {
  if (!text || !text.includes("\\")) {
    return text;
  }

  const decodeOnce = (input) =>
    input
      .replace(/\\u([0-9a-fA-F]{4})/g, (_, hex) => String.fromCharCode(parseInt(hex, 16)))
      .replace(/\\([nrt"\\/])/g, (_, token) => {
        if (token === "n") return "\n";
        if (token === "r") return "\r";
        if (token === "t") return "\t";
        if (token === '"') return '"';
        if (token === "/") return "/";
        return "\\";
      });

  let decoded = text;
  for (let i = 0; i < 3; i += 1) {
    const next = decodeOnce(decoded);
    if (next === decoded) {
      break;
    }
    decoded = next;
  }
  return decoded;
}

function normalizeText(value, fallback = "", options = {}) {
  const decodeEscapes = options.decodeEscapes === true;
  if (value === null || value === undefined) {
    return fallback;
  }
  const raw = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  let normalized = String(raw)
    .replace(/\r\n?/g, "\n")
    .replace(/\u0000/g, "")
    .replace(/\x1b\[[0-9;]*[A-Za-z]/g, "")
    .trim();

  if (decodeEscapes) {
    normalized = decodeEscapedSequences(normalized);
  }

  return normalized;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function sanitizeMarkdownUrl(rawUrl) {
  const source = String(rawUrl || "").trim().replace(/&amp;/g, "&");
  if (!source) return "";
  try {
    const parsed = new URL(source, window.location.origin);
    const protocol = parsed.protocol.toLowerCase();
    if (protocol === "http:" || protocol === "https:" || protocol === "mailto:") {
      return parsed.href;
    }
  } catch (_err) {
    return "";
  }
  return "";
}

function renderInlineMarkdown(sourceText) {
  if (!sourceText) return "";
  let working = escapeHtml(sourceText);
  const codeTokens = [];

  working = working.replace(/`([^`]+)`/g, (_match, codeInner) => {
    const token = `@@CODE_${codeTokens.length}@@`;
    codeTokens.push(`<code>${codeInner}</code>`);
    return token;
  });

  working = working.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_match, label, url) => {
    const safeHref = sanitizeMarkdownUrl(url);
    if (!safeHref) {
      return label;
    }
    return `<a href="${escapeHtml(safeHref)}" target="_blank" rel="noopener noreferrer">${label}</a>`;
  });

  working = working.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  working = working.replace(/(^|[\s(])\*([^*]+)\*(?=[\s).,!?:;]|$)/g, "$1<em>$2</em>");

  working = working.replace(/@@CODE_(\d+)@@/g, (_match, idx) => codeTokens[Number(idx)] || "");
  return working;
}

function renderMarkdown(markdownText) {
  const text = normalizeText(markdownText, "");
  if (!text) return "";

  const lines = text.split("\n");
  const htmlParts = [];
  let paragraphLines = [];
  let inUl = false;
  let inOl = false;
  let inCode = false;
  let codeLang = "";
  let codeLines = [];

  const closeLists = () => {
    if (inUl) {
      htmlParts.push("</ul>");
      inUl = false;
    }
    if (inOl) {
      htmlParts.push("</ol>");
      inOl = false;
    }
  };

  const flushParagraph = () => {
    if (!paragraphLines.length) return;
    const paragraphHtml = paragraphLines.map((line) => renderInlineMarkdown(line)).join("<br />");
    htmlParts.push(`<p>${paragraphHtml}</p>`);
    paragraphLines = [];
  };

  const flushCode = () => {
    const escapedCode = escapeHtml(codeLines.join("\n"));
    const langClass = codeLang ? ` class="language-${escapeHtml(codeLang)}"` : "";
    htmlParts.push(`<pre><code${langClass}>${escapedCode}</code></pre>`);
    inCode = false;
    codeLang = "";
    codeLines = [];
  };

  for (const line of lines) {
    if (inCode) {
      if (/^```/.test(line.trim())) {
        flushCode();
      } else {
        codeLines.push(line);
      }
      continue;
    }

    const fenceMatch = line.match(/^```+\s*([^`\s]*)\s*$/);
    if (fenceMatch) {
      flushParagraph();
      closeLists();
      inCode = true;
      codeLang = String(fenceMatch[1] || "").trim();
      continue;
    }

    if (!line.trim()) {
      flushParagraph();
      closeLists();
      continue;
    }

    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      flushParagraph();
      closeLists();
      const level = headingMatch[1].length;
      const headingBody = renderInlineMarkdown(headingMatch[2]);
      htmlParts.push(`<h${level}>${headingBody}</h${level}>`);
      continue;
    }

    const ulMatch = line.match(/^\s*[-*+]\s+(.*)$/);
    if (ulMatch) {
      flushParagraph();
      if (inOl) {
        htmlParts.push("</ol>");
        inOl = false;
      }
      if (!inUl) {
        htmlParts.push("<ul>");
        inUl = true;
      }
      htmlParts.push(`<li>${renderInlineMarkdown(ulMatch[1])}</li>`);
      continue;
    }

    const olMatch = line.match(/^\s*\d+\.\s+(.*)$/);
    if (olMatch) {
      flushParagraph();
      if (inUl) {
        htmlParts.push("</ul>");
        inUl = false;
      }
      if (!inOl) {
        htmlParts.push("<ol>");
        inOl = true;
      }
      htmlParts.push(`<li>${renderInlineMarkdown(olMatch[1])}</li>`);
      continue;
    }

    if (inUl || inOl) {
      closeLists();
    }
    paragraphLines.push(line.trimEnd());
  }

  if (inCode) {
    flushCode();
  }
  flushParagraph();
  closeLists();

  return htmlParts.join("\n");
}

function getMarkdownSource(node) {
  if (!node) return "";
  const stored = node.dataset.rawMarkdown;
  if (typeof stored === "string") {
    return stored;
  }
  return normalizeText(node.textContent, "");
}

function setMarkdownContent(node, value) {
  if (!node) return;
  const normalized = normalizeText(value, "");
  node.dataset.rawMarkdown = normalized;
  node.innerHTML = normalized ? renderMarkdown(normalized) : "";
}

function resizePromptInput() {
  if (!promptInput) return;
  promptInput.style.height = "0px";
  const maxHeight = 256;
  const nextHeight = Math.min(promptInput.scrollHeight, maxHeight);
  promptInput.style.height = `${Math.max(nextHeight, 44)}px`;
  promptInput.style.overflowY = promptInput.scrollHeight > maxHeight ? "auto" : "hidden";
}

function setStatus(node, message, ok) {
  if (!node) return;
  node.textContent = message || "";
  node.classList.remove("ok", "bad");
  if (!message) return;
  node.classList.add(ok ? "ok" : "bad");
}

function setUnlocked(unlocked) {
  keyGate?.classList.toggle("hidden", unlocked);
  mainApp?.classList.toggle("hidden", !unlocked);
  composerDock?.classList.toggle("hidden", !unlocked);
  sessionPane?.classList.toggle("hidden", !unlocked);
  executionPane?.classList.toggle("hidden", !unlocked);
  document.body.classList.toggle("sidebar-enabled", unlocked);
}

function setRunningState(running) {
  isRunning = running;
  if (promptInput) {
    promptInput.disabled = running;
  }
  if (promptSubmit) {
    promptSubmit.disabled = running;
    promptSubmit.textContent = running ? "Running..." : "Send";
  }
}

function normalizeMessages(messages) {
  if (!Array.isArray(messages)) return [];
  return messages
    .filter((m) => m && typeof m === "object")
    .map((m) => ({
      role: m.role === "assistant" ? "assistant" : "user",
      content: typeof m.content === "string" ? m.content : "",
      ts: typeof m.ts === "number" ? m.ts : Date.now(),
    }))
    .filter((m) => m.content.trim().length > 0);
}

function readStoredSessionId() {
  try {
    const raw = localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY);
    const id = typeof raw === "string" ? raw.trim() : "";
    return id || null;
  } catch (_err) {
    return null;
  }
}

function writeStoredSessionId(sessionId) {
  try {
    if (sessionId) {
      localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, sessionId);
    } else {
      localStorage.removeItem(ACTIVE_SESSION_STORAGE_KEY);
    }
  } catch (_err) {
    // Ignore storage errors
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload = {};
  try {
    payload = text ? JSON.parse(text) : {};
  } catch (_err) {
    payload = {};
  }
  if (!response.ok) {
    const err = new Error(payload.message || text || "Request failed.");
    if (payload && typeof payload === "object") {
      err.code = payload.code;
      err.provider = payload.provider;
    }
    throw err;
  }
  return payload;
}

function isMissingAnthropicKeyError(error) {
  if (!error) return false;
  if (error.code === "missing_provider_key") {
    return (error.provider || "").toLowerCase() === "anthropic";
  }
  const message = normalizeText(error.message, "").toLowerCase();
  return message.includes('no api key found for provider "anthropic"');
}

function openAuthModal(message = "") {
  if (!authModal) return;
  authModal.classList.remove("hidden");
  setStatus(authStatus, message || "", false);
  if (authKeyInput) {
    authKeyInput.focus();
  }
}

function closeAuthModal() {
  authModal?.classList.add("hidden");
  setStatus(authStatus, "", true);
}

async function refreshAuthStatus() {
  const payload = await fetchJson("/api/auth-status");
  const configured = payload?.providers?.anthropic?.configured === true;
  anthropicConfigured = configured;
  return configured;
}

async function ensureAnthropicKeyConfigured({ showModal = false } = {}) {
  if (anthropicConfigured) {
    return true;
  }
  const configured = await refreshAuthStatus();
  if (!configured && showModal) {
    openAuthModal("Add your Anthropic key to continue.");
  }
  return configured;
}

function showExecutionTab(tab) {
  const showTerminal = tab !== "graph";
  executionTerminalView?.classList.toggle("hidden", !showTerminal);
  executionGraphView?.classList.toggle("hidden", showTerminal);
  executionTabTerminal?.classList.toggle("active", showTerminal);
  executionTabGraph?.classList.toggle("active", !showTerminal);
  executionTabTerminal?.setAttribute("aria-selected", showTerminal ? "true" : "false");
  executionTabGraph?.setAttribute("aria-selected", showTerminal ? "false" : "true");
}

function scrollExecutionToBottom() {
  if (!executionFeed) return;
  executionFeed.scrollTop = executionFeed.scrollHeight;
}

function createExecutionRunShell(promptText) {
  if (!executionFeed) {
    return { streamEvents: null };
  }
  executionFeed.innerHTML = "";

  const runShell = document.createElement("section");
  runShell.className = "terminal-run";

  const runMeta = document.createElement("div");
  runMeta.className = "terminal-run-meta";
  runMeta.textContent = `Run started · ${new Date().toLocaleTimeString()}`;
  runShell.appendChild(runMeta);

  const runPrompt = document.createElement("div");
  runPrompt.className = "terminal-run-prompt markdown-content";
  setMarkdownContent(runPrompt, `**Prompt**\n${normalizeText(promptText, "")}`);
  runShell.appendChild(runPrompt);

  const streamEvents = document.createElement("div");
  streamEvents.className = "stream-events terminal-stream";
  runShell.appendChild(streamEvents);

  executionFeed.appendChild(runShell);
  showExecutionTab("terminal");
  scrollExecutionToBottom();

  return { streamEvents };
}

function renderExecutionEmpty(message = "Run a prompt to stream terminal output.") {
  if (!executionFeed) return;
  executionFeed.innerHTML = "";
  const empty = document.createElement("div");
  empty.className = "execution-empty";
  empty.textContent = message;
  executionFeed.appendChild(empty);
}

function setActiveSession(session) {
  const sessionId = normalizeText(session?.id, "");
  const messages = normalizeMessages(session?.messages);
  activeSessionId = sessionId || null;
  activeSessionMessages = messages;
  writeStoredSessionId(activeSessionId);
  if (activeSessionId) {
    const existingIndex = sessionSummaries.findIndex((item) => item.id === activeSessionId);
    const nextSummary = {
      id: activeSessionId,
      title: normalizeText(session?.title, "New Chat") || "New Chat",
      created_at: Number(session?.created_at) || Date.now() / 1000,
      updated_at: Number(session?.updated_at) || Date.now() / 1000,
      message_count: messages.length,
    };
    if (existingIndex >= 0) {
      sessionSummaries[existingIndex] = nextSummary;
    } else {
      sessionSummaries.unshift(nextSummary);
    }
  }
  renderSessionList();
  renderExecutionEmpty("Run a prompt to stream terminal output.");
}

function normalizeSessionSummaries(summaries) {
  if (!Array.isArray(summaries)) return [];
  return summaries
    .filter((item) => item && typeof item === "object")
    .map((item) => ({
      id: normalizeText(item.id, ""),
      title: normalizeText(item.title, "New Chat") || "New Chat",
      created_at: Number(item.created_at) || 0,
      updated_at: Number(item.updated_at) || 0,
      message_count: Number(item.message_count) || 0,
    }))
    .filter((item) => item.id);
}

function formatRelativeTime(secondsEpoch) {
  const ts = Number(secondsEpoch);
  if (!Number.isFinite(ts) || ts <= 0) return "";
  const diff = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (diff < 30) return "just now";
  if (diff < 3600) return `${Math.max(1, Math.floor(diff / 60))}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 86400 * 7) return `${Math.floor(diff / 86400)}d ago`;
  return `${Math.floor(diff / (86400 * 7))}w ago`;
}

function setSessionStatus(message, ok) {
  setStatus(sessionStatus, message, ok);
}

function renderSessionList() {
  if (!sessionList) return;
  sessionList.innerHTML = "";

  if (!sessionSummaries.length) {
    const emptyNode = document.createElement("div");
    emptyNode.className = "session-empty";
    emptyNode.textContent = "No sessions yet. Start a new one.";
    sessionList.appendChild(emptyNode);
    return;
  }

  for (const summary of sessionSummaries) {
    const row = document.createElement("div");
    row.className = `session-item ${summary.id === activeSessionId ? "active" : ""}`.trim();

    const selectBtn = document.createElement("button");
    selectBtn.type = "button";
    selectBtn.className = "session-select";
    selectBtn.title = summary.title;

    const titleNode = document.createElement("span");
    titleNode.className = "session-title";
    titleNode.textContent = summary.title;
    selectBtn.appendChild(titleNode);

    const metaNode = document.createElement("span");
    metaNode.className = "session-meta";
    const labelCount = summary.message_count === 1 ? "1 msg" : `${summary.message_count} msgs`;
    const relativeUpdated = formatRelativeTime(summary.updated_at);
    metaNode.textContent = relativeUpdated ? `${labelCount} · ${relativeUpdated}` : labelCount;
    selectBtn.appendChild(metaNode);

    selectBtn.addEventListener("click", async () => {
      if (summary.id === activeSessionId) return;
      if (isRunning) {
        setSessionStatus("Wait for the current run to finish before switching sessions.", false);
        return;
      }
      try {
        await getSession(summary.id);
        renderChatHistory(activeSessionMessages);
        setSessionStatus("", true);
        setStatus(promptStatus, "", true);
        scrollChatToBottom({ behavior: "auto", force: true });
      } catch (error) {
        setSessionStatus(error.message || "Could not switch sessions.", false);
      }
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "session-delete";
    deleteBtn.textContent = "×";
    deleteBtn.title = "Delete session";
    deleteBtn.setAttribute("aria-label", `Delete session ${summary.title}`);
    deleteBtn.addEventListener("click", async (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (isRunning && summary.id === activeSessionId) {
        setSessionStatus("Wait for the current run to finish before deleting this session.", false);
        return;
      }
      const confirmed = window.confirm(`Delete session "${summary.title}"? This cannot be undone.`);
      if (!confirmed) return;

      try {
        await fetchJson(`/api/sessions/${encodeURIComponent(summary.id)}`, { method: "DELETE" });
        sessionSummaries = sessionSummaries.filter((item) => item.id !== summary.id);

        if (summary.id === activeSessionId) {
          if (sessionSummaries.length > 0) {
            await getSession(sessionSummaries[0].id);
          } else {
            await createSession();
          }
          renderChatHistory(activeSessionMessages);
        }
        await refreshSessionSummaries();
        setSessionStatus("Session deleted.", true);
      } catch (error) {
        setSessionStatus(error.message || "Could not delete session.", false);
      }
    });

    row.appendChild(selectBtn);
    row.appendChild(deleteBtn);
    sessionList.appendChild(row);
  }
}

async function refreshSessionSummaries() {
  const { sessions } = await listSessions();
  sessionSummaries = normalizeSessionSummaries(sessions);
  if (activeSessionId && !sessionSummaries.some((item) => item.id === activeSessionId)) {
    activeSessionId = null;
    writeStoredSessionId(null);
  }
  renderSessionList();
}

function touchActiveSessionSummary(role, content) {
  if (!activeSessionId) return;
  const nowSeconds = Date.now() / 1000;
  const idx = sessionSummaries.findIndex((item) => item.id === activeSessionId);
  if (idx < 0) return;
  const next = { ...sessionSummaries[idx] };
  next.updated_at = nowSeconds;
  next.message_count = Math.max(0, Number(next.message_count) || 0) + 1;
  if ((next.title === "New Chat" || !next.title) && role === "user") {
    const firstLine = normalizeText(content, "New Chat").split("\n")[0];
    next.title = firstLine.slice(0, 72) || "New Chat";
  }
  sessionSummaries[idx] = next;
  sessionSummaries.sort((a, b) => Number(b.updated_at || 0) - Number(a.updated_at || 0));
  renderSessionList();
}

async function listSessions() {
  const payload = await fetchJson("/api/sessions");
  const sessions = Array.isArray(payload.sessions) ? payload.sessions : [];
  const lastSessionId = normalizeText(payload.last_session_id, "");
  return { sessions, lastSessionId: lastSessionId || null };
}

async function createSession(title = "") {
  const payload = await fetchJson("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!payload.session || typeof payload.session !== "object") {
    throw new Error("Invalid create session response.");
  }
  setActiveSession(payload.session);
  return payload.session;
}

async function getSession(sessionId) {
  const payload = await fetchJson(`/api/sessions/${encodeURIComponent(sessionId)}`);
  if (!payload.session || typeof payload.session !== "object") {
    throw new Error("Invalid session response.");
  }
  setActiveSession(payload.session);
  return payload.session;
}

async function ensureActiveSession() {
  const preferredId = readStoredSessionId();
  if (preferredId) {
    try {
      return await getSession(preferredId);
    } catch (_err) {
      // Fall through to fallback strategy
    }
  }

  const { sessions, lastSessionId } = await listSessions();
  sessionSummaries = normalizeSessionSummaries(sessions);
  renderSessionList();
  if (lastSessionId) {
    try {
      return await getSession(lastSessionId);
    } catch (_err) {
      // Fall through to first available session
    }
  }
  if (sessions.length > 0) {
    return getSession(sessions[0].id);
  }
  return createSession();
}

function isNearBottom(thresholdPx = 220) {
  if (!chatThread) return true;
  const distance = chatThread.scrollHeight - (chatThread.scrollTop + chatThread.clientHeight);
  return distance <= thresholdPx;
}

function updateAutoScrollEnabled() {
  autoScrollEnabled = isNearBottom();
  if (autoScrollEnabled && jumpLatestBtn) {
    unseenEventCount = 0;
    jumpLatestBtn.classList.add("hidden");
    jumpLatestBtn.textContent = "Jump to latest";
  }
}

function scrollChatToBottom(options = {}) {
  if (!options.force && !autoScrollEnabled) {
    return;
  }
  if (!chatThread) return;
  if ((options.behavior || "auto") === "smooth" && typeof chatThread.scrollTo === "function") {
    chatThread.scrollTo({ top: chatThread.scrollHeight, behavior: "smooth" });
    return;
  }
  chatThread.scrollTop = chatThread.scrollHeight;
}

function ensureJumpLatestButton() {
  if (jumpLatestBtn) return;
  const btn = document.createElement("button");
  btn.type = "button";
  btn.id = "jump-latest-btn";
  btn.className = "jump-latest hidden";
  btn.textContent = "Jump to latest";
  document.body.appendChild(btn);
  btn.addEventListener("click", () => {
    unseenEventCount = 0;
    btn.classList.add("hidden");
    btn.textContent = "Jump to latest";
    scrollChatToBottom({ behavior: "smooth", force: true });
    updateAutoScrollEnabled();
  });
  jumpLatestBtn = btn;
}

function createChatMessageNode({ role, content }) {
  const wrapper = document.createElement("div");
  wrapper.className = `chat-message ${role === "assistant" ? "assistant" : "user"}`;

  const meta = document.createElement("div");
  meta.className = "chat-meta";
  meta.textContent = role === "assistant" ? "Cobra Lite" : "You";
  wrapper.appendChild(meta);

  const bubble = document.createElement("div");
  bubble.className = "chat-bubble markdown-content";
  setMarkdownContent(bubble, content);
  wrapper.appendChild(bubble);

  return { wrapper, bubble };
}

function renderChatHistory(history) {
  if (!chatThread) return;
  chatThread.innerHTML = "";
  for (const msg of history || []) {
    const role = msg.role === "assistant" ? "assistant" : "user";
    const node = createChatMessageNode({ role, content: msg.content || "" });
    chatThread.appendChild(node.wrapper);
  }
}

function createRunUI(promptText) {
  const userMsg = createChatMessageNode({ role: "user", content: promptText });
  const assistantMsg = createChatMessageNode({ role: "assistant", content: "" });
  const terminalRun = createExecutionRunShell(promptText);

  return {
    userMsg,
    assistantMsg,
    assistantInserted: false,
    streamEvents: terminalRun.streamEvents,
    runningExecutionBlocks: new Map(),
    reasoningLiveNode: null,
    committedAssistant: false,
    commitAssistant: null,
  };
}

function ensureAssistantNode(run) {
  if (!run || !run.assistantMsg || run.assistantInserted) return;
  chatThread?.appendChild(run.assistantMsg.wrapper);
  run.assistantInserted = true;
}

function setAssistantBubble(run, content) {
  if (!run || !run.assistantMsg || !run.assistantMsg.bubble) return;
  ensureAssistantNode(run);
  setMarkdownContent(run.assistantMsg.bubble, content);
}

function appendStreamEvent(run, content, kind) {
  if (!run || !run.streamEvents) return;
  const block = document.createElement("div");
  block.className = `stream-event ${kind || ""}`.trim();

  const safeContent = normalizeText(content, "");
  const body = document.createElement("div");
  body.className = "stream-event-body markdown-content";
  setMarkdownContent(body, safeContent);
  block.appendChild(body);

  run.streamEvents.appendChild(block);
  scrollExecutionToBottom();
}

function createCommandBlock({ stepIndex, toolName, command, rationale, isRunning }) {
  const details = document.createElement("details");
  details.className = `stream-event command-block ${isRunning ? "running" : "completed"}`;
  if (isRunning) {
    details.open = true;
  }

  const summary = document.createElement("summary");
  summary.textContent = isRunning
    ? `Action ${stepIndex} · ${toolName} · running`
    : `Action ${stepIndex} · ${toolName}`;
  details.appendChild(summary);

  const summaryLine = document.createElement("div");
  summaryLine.className = "stream-event-summary";
  summaryLine.textContent = command;
  details.appendChild(summaryLine);

  let rationaleLine = null;
  if (rationale) {
    rationaleLine = document.createElement("div");
    rationaleLine.className = "stream-event-summary";
    rationaleLine.textContent = `Rationale: ${rationale}`;
    details.appendChild(rationaleLine);
  }

  const outputNode = document.createElement("div");
  outputNode.className = "command-output markdown-content";
  setMarkdownContent(outputNode, isRunning ? "(running...)" : "");
  details.appendChild(outputNode);

  return {
    details,
    summary,
    summaryLine,
    rationaleLine,
    outputNode,
  };
}

function appendCommandStart(run, data) {
  const stepIndex = data.action_index_1based || data.step_index_1based || "?";
  const toolName = data.tool_name || "unknown tool";
  const command = normalizeText(data.command, "(no command)");
  const rationale = normalizeText(data.rationale, "");
  const executionId = normalizeText(data.execution_id, "");

  const blockRef = createCommandBlock({
    stepIndex,
    toolName,
    command,
    rationale,
    isRunning: true,
  });

  if (executionId) {
    blockRef.details.dataset.executionId = executionId;
    run.runningExecutionBlocks.set(executionId, blockRef);
  }

  run.streamEvents.appendChild(blockRef.details);
  scrollExecutionToBottom();
}

function appendCommandExecution(run, data) {
  const stepIndex = data.action_index_1based || data.step_index_1based || "?";
  const toolName = data.tool_name || "unknown tool";
  const command = normalizeText(data.command, "(no command)");
  const output = normalizeText(data.tool_output, "(no output)", { decodeEscapes: true });
  const rationale = normalizeText(data.rationale, "");
  const executionId = normalizeText(data.execution_id, "");
  const runningBlock = executionId ? run.runningExecutionBlocks.get(executionId) : null;

  if (runningBlock) {
    runningBlock.details.classList.remove("running");
    runningBlock.details.classList.add("completed");
    runningBlock.summary.textContent = `Action ${stepIndex} · ${toolName}`;
    runningBlock.summaryLine.textContent = command;
    if (rationale) {
      if (!runningBlock.rationaleLine) {
        const rationaleLine = document.createElement("div");
        rationaleLine.className = "stream-event-summary";
        runningBlock.details.insertBefore(rationaleLine, runningBlock.outputNode);
        runningBlock.rationaleLine = rationaleLine;
      }
      runningBlock.rationaleLine.textContent = `Rationale: ${rationale}`;
    }
    setMarkdownContent(
      runningBlock.outputNode,
      mergeCommandOutput(getMarkdownSource(runningBlock.outputNode), output)
    );
    run.runningExecutionBlocks.delete(executionId);
    scrollExecutionToBottom();
    return;
  }

  const blockRef = createCommandBlock({
    stepIndex,
    toolName,
    command,
    rationale,
    isRunning: false,
  });
  setMarkdownContent(blockRef.outputNode, output);
  run.streamEvents.appendChild(blockRef.details);
  scrollExecutionToBottom();
}

function mergeCommandOutput(existingText, incomingText) {
  const existing = normalizeText(existingText, "");
  const incoming = normalizeText(incomingText, "");
  const base = existing === "(running...)" ? "" : existing;

  if (!incoming) {
    return base || "(running...)";
  }
  if (!base) {
    return incoming;
  }
  if (incoming.startsWith(base)) {
    return incoming;
  }
  if (base.endsWith(incoming)) {
    return base;
  }
  return `${base}\n${incoming}`;
}

function mergeIncrementalText(existingText, incomingText) {
  const existing = existingText || "";
  const incoming = normalizeText(incomingText, "", { decodeEscapes: true });
  if (!incoming) return existing;
  if (!existing) return incoming;
  if (incoming.startsWith(existing)) return incoming;
  if (existing.endsWith(incoming)) return existing;
  if (/^[,.;:!?)]/.test(incoming)) return `${existing}${incoming}`;
  if (/^\s/.test(incoming) || /\s$/.test(existing)) return `${existing}${incoming}`;
  return `${existing}\n${incoming}`;
}

function ensureReasoningLiveNode(run) {
  if (run.reasoningLiveNode) {
    return run.reasoningLiveNode;
  }
  const details = document.createElement("details");
  details.className = "stream-event note";
  details.open = true;

  const summary = document.createElement("summary");
  summary.textContent = "Agent notes (live)";
  details.appendChild(summary);

  const outputNode = document.createElement("div");
  outputNode.className = "stream-event-multi markdown-content";
  setMarkdownContent(outputNode, "");
  details.appendChild(outputNode);

  run.streamEvents.appendChild(details);
  scrollExecutionToBottom();
  run.reasoningLiveNode = { details, outputNode };
  return run.reasoningLiveNode;
}

function appendCommandUpdate(run, data) {
  const executionId = normalizeText(data.execution_id, "");
  const runningBlock = executionId ? run.runningExecutionBlocks.get(executionId) : null;
  const output = normalizeText(data.tool_output, "", { decodeEscapes: true });
  if (!output) return;

  if (runningBlock) {
    setMarkdownContent(
      runningBlock.outputNode,
      mergeCommandOutput(getMarkdownSource(runningBlock.outputNode), output)
    );
    scrollExecutionToBottom();
    return;
  }

  const stepIndex = data.action_index_1based || data.step_index_1based || "?";
  const toolName = data.tool_name || "unknown tool";
  const command = normalizeText(data.command, "(no command)");
  const rationale = normalizeText(data.rationale, "");
  const blockRef = createCommandBlock({
    stepIndex,
    toolName,
    command,
    rationale,
    isRunning: true,
  });
  setMarkdownContent(
    blockRef.outputNode,
    mergeCommandOutput(getMarkdownSource(blockRef.outputNode), output)
  );
  if (executionId) {
    blockRef.details.dataset.executionId = executionId;
    run.runningExecutionBlocks.set(executionId, blockRef);
  }
  run.streamEvents.appendChild(blockRef.details);
  scrollExecutionToBottom();
}

function handleStreamEvent(type, data, run) {
  if (!data || !run) return;

  switch (type) {
    case "llm_decide_start":
    case "llm_decide_end":
      // Internal debug events
      break;
    case "tool_start":
      appendCommandStart(run, data);
      break;
    case "tool_update":
      appendCommandUpdate(run, data);
      break;
    case "tool_execution":
      appendCommandExecution(run, data);
      break;
    case "reasoning": {
      const text = normalizeText(data.text, "");
      if (text) {
        const ref = ensureReasoningLiveNode(run);
        setMarkdownContent(
          ref.outputNode,
          mergeIncrementalText(getMarkdownSource(ref.outputNode), text)
        );
      }
      break;
    }
    case "assistant_delta": {
      // Keep center chat focused on final assistant output only.
      break;
    }
    case "run_status":
      if (data.phase === "start") {
        appendStreamEvent(run, "Run started.", "note");
      } else if (data.phase === "end") {
        appendStreamEvent(run, "Run finished.", "note");
      }
      break;
    case "final_observation": {
      const text = normalizeText(data.final_observation, "(no final observation)");
      setAssistantBubble(run, text);
      run.commitAssistant?.(text);
      break;
    }
    case "final_result":
      if (data.result && typeof data.result === "object" && typeof data.result.final_observation === "string") {
        const text = normalizeText(data.result.final_observation, "(no final observation)");
        setAssistantBubble(run, text);
        run.commitAssistant?.(text);
      }
      break;
    case "error": {
      const message = normalizeText(data.message, "Backend error while running prompt.");
      const code = normalizeText(data.code, "");
      const provider = normalizeText(data.provider, "").toLowerCase();
      if (code === "missing_provider_key" && provider === "anthropic") {
        anthropicConfigured = false;
        openAuthModal("Add your Anthropic key, then resend your prompt.");
      }
      setStatus(promptStatus, message, false);
      appendStreamEvent(run, `ERROR: ${message}`, "note");
      if (!run.committedAssistant) {
        setAssistantBubble(run, `Error: ${message}`);
        run.commitAssistant?.(`Error: ${message}`);
      }
      break;
    }
    case "done":
      if (data.ok === false) {
        setStatus(promptStatus, data.message || "Run ended with errors.", false);
      } else {
        setStatus(promptStatus, "Run complete.", true);
      }
      break;
    default:
      break;
  }

  ensureJumpLatestButton();
  if (!autoScrollEnabled && jumpLatestBtn) {
    unseenEventCount += 1;
    jumpLatestBtn.classList.remove("hidden");
    jumpLatestBtn.textContent = unseenEventCount > 1 ? `Jump to latest (${unseenEventCount})` : "Jump to latest";
  }

  scrollChatToBottom({ behavior: "auto" });
}

function parseSseFrame(rawFrame) {
  const lines = rawFrame.split("\n");
  let eventName = "message";
  const dataParts = [];
  let payload = null;

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataParts.push(line.slice(5).trimStart());
    }
  }

  const dataText = dataParts.join("\n").trim();
  if (!dataText) return null;

  try {
    payload = JSON.parse(dataText);
  } catch (_err) {
    return null;
  }

  return { type: eventName, payload: payload };
}

async function runPromptStream({ prompt, sessionId, run }) {
  const response = await fetch("/api/prompt/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, session_id: sessionId }),
  });

  if (!response.ok) {
    const text = await response.text();
    let message = "Prompt submission failed.";
    let code = "";
    let provider = "";
    try {
      const data = JSON.parse(text);
      message = data.message || message;
      code = data.code || "";
      provider = data.provider || "";
    } catch (_err) {
      message = text || message;
    }
    const err = new Error(message);
    err.code = code;
    err.provider = provider;
    throw err;
  }

  if (!response.body) {
    throw new Error("Streaming not supported by this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let doneReceived = false;
  let doneStatus = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      if (buffer.trim()) {
        const parsed = parseSseFrame(buffer.trim());
        if (parsed) {
          handleStreamEvent(parsed.type, parsed.payload, run);
          if (parsed.type === "done") {
            doneReceived = true;
            doneStatus = parsed.payload.ok !== false;
          }
        }
      }
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    let idx = buffer.indexOf("\n\n");
    while (idx !== -1) {
      const frame = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const parsed = parseSseFrame(frame.trim());
      if (parsed) {
        handleStreamEvent(parsed.type, parsed.payload, run);
        if (parsed.type === "done") {
          doneReceived = true;
          doneStatus = parsed.payload.ok !== false;
        }
      }
      idx = buffer.indexOf("\n\n");
    }
  }

  if (!doneReceived) {
    appendStreamEvent(run, "Stream ended unexpectedly without a done event.", "note");
    doneStatus = false;
  }

  return doneStatus;
}

async function verifyGateway(gatewayUrl, statusNode) {
  const response = await fetch("/api/verify-gateway", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ gateway_url: gatewayUrl }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.message || "Gateway connection failed.");
  }
  setStatus(statusNode, data.message || "Gateway connected.", true);
  return data;
}

keyForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const value = keyInput.value.trim() || defaultGatewayUrl;

  setStatus(keyStatus, "Connecting to gateway...", true);
  try {
    await verifyGateway(value, keyStatus);
    await ensureActiveSession();
    await refreshSessionSummaries();
    await ensureAnthropicKeyConfigured({ showModal: true });
    renderChatHistory(activeSessionMessages);
    setUnlocked(true);
    resizePromptInput();
    promptInput?.focus();
    scrollChatToBottom({ behavior: "smooth", force: true });
  } catch (error) {
    setStatus(keyStatus, error.message, false);
  }
});

authForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const value = authKeyInput?.value?.trim() || "";
  if (!value) {
    setStatus(authStatus, "Please enter your Anthropic API key.", false);
    return;
  }
  setStatus(authStatus, "Saving Anthropic key...", true);
  try {
    await fetchJson("/api/auth/anthropic", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: value }),
    });
    anthropicConfigured = true;
    if (authKeyInput) {
      authKeyInput.value = "";
    }
    closeAuthModal();
    setStatus(promptStatus, "Anthropic key saved. You can run prompts now.", true);
    promptInput?.focus();
  } catch (error) {
    setStatus(authStatus, error.message || "Could not save API key.", false);
  }
});

newSessionBtn?.addEventListener("click", async () => {
  if (isRunning) {
    setSessionStatus("Wait for the current run to finish before creating a new session.", false);
    return;
  }
  try {
    await createSession();
    await refreshSessionSummaries();
    renderChatHistory(activeSessionMessages);
    setStatus(promptStatus, "Started a new session.", true);
    setSessionStatus("", true);
    promptInput?.focus();
    scrollChatToBottom({ behavior: "auto", force: true });
  } catch (error) {
    setSessionStatus(error.message || "Could not create a new session.", false);
  }
});

promptForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt) {
    setStatus(promptStatus, "Prompt cannot be empty.", false);
    return;
  }
  if (isRunning) {
    return;
  }

  if (!activeSessionId) {
    try {
      await ensureActiveSession();
    } catch (error) {
      setStatus(promptStatus, error.message || "Could not initialize session.", false);
      return;
    }
  }
  try {
    const configured = await ensureAnthropicKeyConfigured({ showModal: true });
    if (!configured) {
      setStatus(promptStatus, "Anthropic API key required before running prompts.", false);
      return;
    }
  } catch (error) {
    setStatus(promptStatus, error.message || "Could not verify provider auth.", false);
    return;
  }
  const run = createRunUI(prompt);

  run.commitAssistant = (text) => {
    if (run.committedAssistant) return;
    const normalized = normalizeText(text, "");
    activeSessionMessages.push({ role: "assistant", content: normalized, ts: Date.now() });
    touchActiveSessionSummary("assistant", normalized);
    run.committedAssistant = true;
  };

  chatThread?.appendChild(run.userMsg.wrapper);
  scrollChatToBottom({ behavior: "smooth", force: true });

  activeSessionMessages.push({ role: "user", content: prompt, ts: Date.now() });
  touchActiveSessionSummary("user", prompt);

  promptInput.value = "";
  resizePromptInput();

  setRunningState(true);
  setStatus(promptStatus, "Starting Cobra Lite run...", true);

  try {
    const ok = await runPromptStream({ prompt, sessionId: activeSessionId, run });
    if (ok) {
      setStatus(promptStatus, "Run complete.", true);
      if (!run.committedAssistant) {
        const fallback = run.assistantMsg.bubble.textContent || "(no response)";
        setAssistantBubble(run, fallback);
        run.commitAssistant(fallback);
      }
    }
  } catch (error) {
    if (isMissingAnthropicKeyError(error)) {
      anthropicConfigured = false;
      openAuthModal("Add your Anthropic key, then resend your prompt.");
      setStatus(promptStatus, "Anthropic API key required.", false);
      appendStreamEvent(run, "Error: Anthropic API key required.", "note");
    } else {
      setStatus(promptStatus, error.message, false);
      appendStreamEvent(run, `Error: ${error.message}`, "note");
    }
    if (!run.committedAssistant) {
      const message = isMissingAnthropicKeyError(error) ? "Error: Anthropic API key required." : `Error: ${error.message}`;
      setAssistantBubble(run, message);
      run.commitAssistant(message);
    }
  } finally {
    try {
      await refreshSessionSummaries();
    } catch (_err) {
      // Ignore sidebar refresh errors after prompt completion.
    }
    setRunningState(false);
    promptInput?.focus();
  }
});

promptInput?.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    promptForm?.requestSubmit();
  }
});
promptInput?.addEventListener("input", resizePromptInput);

executionTabTerminal?.addEventListener("click", () => {
  showExecutionTab("terminal");
});

executionTabGraph?.addEventListener("click", () => {
  showExecutionTab("graph");
});

settingsBtn?.addEventListener("click", () => {
  settingsModal.classList.remove("hidden");
  settingsStatus.textContent = "";
  if (settingsKeyInput && !settingsKeyInput.value.trim()) {
    settingsKeyInput.value = defaultGatewayUrl;
  }
  settingsKeyInput.focus();
});

closeSettingsBtn?.addEventListener("click", () => {
  settingsModal.classList.add("hidden");
});

settingsModal?.addEventListener("click", (event) => {
  if (event.target === settingsModal) {
    settingsModal.classList.add("hidden");
  }
});

settingsForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const value = settingsKeyInput.value.trim();
  if (!value) {
    setStatus(settingsStatus, "Please enter a gateway URL.", false);
    return;
  }

  setStatus(settingsStatus, "Updating gateway...", true);
  try {
    await verifyGateway(value, settingsStatus);
    settingsKeyInput.value = "";
    setUnlocked(true);
    setTimeout(() => settingsModal.classList.add("hidden"), 400);
  } catch (error) {
    setStatus(settingsStatus, error.message, false);
  }
});

async function bootstrap() {
  setUnlocked(hasGateway);
  showExecutionTab("terminal");
  renderExecutionEmpty();
  if (hasGateway) {
    try {
      await ensureActiveSession();
      await refreshSessionSummaries();
      await ensureAnthropicKeyConfigured({ showModal: true });
    } catch (error) {
      setStatus(promptStatus, error.message || "Could not load session.", false);
    }
  }
  renderSessionList();
  renderChatHistory(activeSessionMessages);
  resizePromptInput();
  scrollChatToBottom({ behavior: "auto", force: true });
  ensureJumpLatestButton();
  updateAutoScrollEnabled();
  chatThread?.addEventListener("scroll", updateAutoScrollEnabled, { passive: true });
  const authModalVisible = !!authModal && !authModal.classList.contains("hidden");
  if (hasGateway && !authModalVisible) {
    promptInput?.focus();
  }
}

bootstrap();
