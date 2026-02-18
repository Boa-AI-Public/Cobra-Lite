const hasGateway = window.CLAW_HAS_GATEWAY === true;
const defaultGatewayUrl = window.CLAW_DEFAULT_GATEWAY_URL || "http://127.0.0.1:18789";

const keyGate = document.getElementById("key-gate");
const mainApp = document.getElementById("main-app");
const composerDock = document.getElementById("composer-dock");
const chatThread = document.getElementById("chat-thread");
const clearChatBtn = document.getElementById("clear-chat-btn");

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

let isRunning = false;
let autoScrollEnabled = true;
let unseenEventCount = 0;
let jumpLatestBtn = null;

const CHAT_STORAGE_KEY = "cobraLite.chat.v1";
const CHAT_MAX_MESSAGES = 2000;
const CHAT_MAX_CHARS = 240000;

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

function safeJsonParse(text, fallback) {
  try {
    return JSON.parse(text);
  } catch (_err) {
    return fallback;
  }
}

function loadChatHistory() {
  try {
    const raw = localStorage.getItem(CHAT_STORAGE_KEY);
    const parsed = safeJsonParse(raw || "[]", []);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((m) => m && typeof m === "object")
      .map((m) => ({
        role: m.role === "assistant" ? "assistant" : "user",
        content: typeof m.content === "string" ? m.content : "",
        ts: typeof m.ts === "number" ? m.ts : Date.now(),
      }))
      .filter((m) => m.content.trim().length > 0);
  } catch (_err) {
    return [];
  }
}

function pruneChatHistory(history) {
  let out = Array.isArray(history) ? history.slice() : [];
  if (CHAT_MAX_MESSAGES > 0 && out.length > CHAT_MAX_MESSAGES) {
    out = out.slice(out.length - CHAT_MAX_MESSAGES);
  }
  if (CHAT_MAX_CHARS <= 0) {
    return out;
  }

  let total = 0;
  const keptRev = [];
  for (let i = out.length - 1; i >= 0; i -= 1) {
    const msg = out[i];
    const content = typeof msg.content === "string" ? msg.content : "";
    total += content.length + 12;
    if (total > CHAT_MAX_CHARS) {
      break;
    }
    keptRev.push(msg);
  }
  keptRev.reverse();
  return keptRev;
}

function saveChatHistory(history) {
  try {
    localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(pruneChatHistory(history)));
  } catch (_err) {
    // Ignore storage errors
  }
}

function clearChatHistory() {
  try {
    localStorage.removeItem(CHAT_STORAGE_KEY);
  } catch (_err) {
    // ignore
  }
}

function isNearBottom(thresholdPx = 220) {
  const doc = document.documentElement;
  const distance = doc.scrollHeight - (window.scrollY + window.innerHeight);
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
  const last = chatThread.lastElementChild;
  if (!last || typeof last.scrollIntoView !== "function") return;
  last.scrollIntoView({
    block: "end",
    behavior: options.behavior || "auto",
  });
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

  const bubble = document.createElement("pre");
  bubble.className = "chat-bubble";
  bubble.textContent = normalizeText(content, "");
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
  const assistantMsg = createChatMessageNode({ role: "assistant", content: "(running...)" });

  const details = document.createElement("details");
  details.className = "run-details";
  details.open = true;

  const summary = document.createElement("summary");
  summary.textContent = "Run details";
  details.appendChild(summary);

  const body = document.createElement("div");
  body.className = "run-details-body";
  details.appendChild(body);

  const eventsSection = document.createElement("div");
  eventsSection.className = "stream-section";
  const eventsTitle = document.createElement("h3");
  eventsTitle.textContent = "Live execution";
  eventsSection.appendChild(eventsTitle);
  const streamEvents = document.createElement("div");
  streamEvents.className = "stream-events";
  eventsSection.appendChild(streamEvents);
  body.appendChild(eventsSection);

  assistantMsg.wrapper.appendChild(details);

  return {
    userMsg,
    assistantMsg,
    details,
    streamEvents,
    runningExecutionBlocks: new Map(),
    reasoningLiveNode: null,
    committedAssistant: false,
    commitAssistant: null,
  };
}

function setAssistantBubble(run, content) {
  if (!run || !run.assistantMsg || !run.assistantMsg.bubble) return;
  run.assistantMsg.bubble.textContent = normalizeText(content, "");
}

function appendStreamEvent(run, content, kind) {
  if (!run || !run.streamEvents) return;
  const block = document.createElement("div");
  block.className = `stream-event ${kind || ""}`.trim();

  const safeContent = normalizeText(content, "");
  if (safeContent.includes("\n")) {
    const pre = document.createElement("pre");
    pre.className = "stream-event-multi";
    pre.textContent = safeContent;
    block.appendChild(pre);
  } else {
    block.textContent = safeContent;
  }

  run.streamEvents.appendChild(block);
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

  const outputNode = document.createElement("pre");
  outputNode.className = "command-output";
  outputNode.textContent = isRunning ? "(running...)" : "";
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
    runningBlock.outputNode.textContent = mergeCommandOutput(runningBlock.outputNode.textContent, output);
    run.runningExecutionBlocks.delete(executionId);
    return;
  }

  const blockRef = createCommandBlock({
    stepIndex,
    toolName,
    command,
    rationale,
    isRunning: false,
  });
  blockRef.outputNode.textContent = output;
  run.streamEvents.appendChild(blockRef.details);
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

  const outputNode = document.createElement("pre");
  outputNode.className = "stream-event-multi";
  outputNode.textContent = "";
  details.appendChild(outputNode);

  run.streamEvents.appendChild(details);
  run.reasoningLiveNode = { details, outputNode };
  return run.reasoningLiveNode;
}

function appendCommandUpdate(run, data) {
  const executionId = normalizeText(data.execution_id, "");
  const runningBlock = executionId ? run.runningExecutionBlocks.get(executionId) : null;
  const output = normalizeText(data.tool_output, "", { decodeEscapes: true });
  if (!output) return;

  if (runningBlock) {
    runningBlock.outputNode.textContent = mergeCommandOutput(runningBlock.outputNode.textContent, output);
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
  blockRef.outputNode.textContent = mergeCommandOutput(blockRef.outputNode.textContent, output);
  if (executionId) {
    blockRef.details.dataset.executionId = executionId;
    run.runningExecutionBlocks.set(executionId, blockRef);
  }
  run.streamEvents.appendChild(blockRef.details);
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
        ref.outputNode.textContent = mergeIncrementalText(ref.outputNode.textContent, text);
      }
      break;
    }
    case "assistant_delta": {
      const text = normalizeText(data.text, "");
      if (text) {
        setAssistantBubble(run, text);
      }
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
      if (run.details) {
        run.details.open = false;
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

async function runPromptStream({ prompt, history, run }) {
  const response = await fetch("/api/prompt/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, history }),
  });

  if (!response.ok) {
    const text = await response.text();
    let message = "Prompt submission failed.";
    try {
      const data = JSON.parse(text);
      message = data.message || message;
    } catch (_err) {
      message = text || message;
    }
    throw new Error(message);
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
    setUnlocked(true);
    resizePromptInput();
    promptInput?.focus();
    scrollChatToBottom({ behavior: "smooth", force: true });
  } catch (error) {
    setStatus(keyStatus, error.message, false);
  }
});

clearChatBtn?.addEventListener("click", () => {
  const ok = window.confirm("Clear chat history on this device?");
  if (!ok) return;
  clearChatHistory();
  renderChatHistory([]);
  setStatus(promptStatus, "", true);
  promptInput?.focus();
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

  const history = loadChatHistory();
  const run = createRunUI(prompt);

  run.commitAssistant = (text) => {
    if (run.committedAssistant) return;
    const current = loadChatHistory();
    current.push({ role: "assistant", content: normalizeText(text, ""), ts: Date.now() });
    saveChatHistory(current);
    run.committedAssistant = true;
  };

  chatThread?.appendChild(run.userMsg.wrapper);
  chatThread?.appendChild(run.assistantMsg.wrapper);
  scrollChatToBottom({ behavior: "smooth", force: true });

  history.push({ role: "user", content: prompt, ts: Date.now() });
  saveChatHistory(history);

  promptInput.value = "";
  resizePromptInput();

  setRunningState(true);
  setStatus(promptStatus, "Starting Cobra Lite run...", true);

  try {
    const historyForBackend = pruneChatHistory(history).slice(0, -1);
    const ok = await runPromptStream({ prompt, history: historyForBackend, run });
    if (ok) {
      setStatus(promptStatus, "Run complete.", true);
      if (!run.committedAssistant) {
        run.commitAssistant(run.assistantMsg.bubble.textContent || "(no response)");
      }
    }
  } catch (error) {
    setStatus(promptStatus, error.message, false);
    appendStreamEvent(run, `Error: ${error.message}`, "note");
    if (!run.committedAssistant) {
      setAssistantBubble(run, `Error: ${error.message}`);
      run.commitAssistant(`Error: ${error.message}`);
    }
  } finally {
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

setUnlocked(hasGateway);
renderChatHistory(loadChatHistory());
resizePromptInput();
scrollChatToBottom({ behavior: "auto", force: true });
ensureJumpLatestButton();
updateAutoScrollEnabled();
window.addEventListener("scroll", updateAutoScrollEnabled, { passive: true });
if (hasGateway) {
  promptInput?.focus();
}
