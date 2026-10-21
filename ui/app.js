const tauri = window.__TAURI__;
const invoke = tauri?.core?.invoke;

const elements = {
  tabs: document.querySelector("#tabs"),
  conversation: document.querySelector("#conversation"),
  prompt: document.querySelector("#prompt"),
  send: document.querySelector("#send-button"),
  stop: document.querySelector("#stop-button"),
  newTab: document.querySelector("#new-tab"),
  settingsButton: document.querySelector("#settings-button"),
  modelSelect: document.querySelector("#model-select"),
  modelAction: document.querySelector("#model-action"),
  manageModels: document.querySelector("#manage-models"),
  settingsModal: document.querySelector("#settings-modal"),
  settingsClose: document.querySelector("#settings-close"),
  settingsCancel: document.querySelector("#settings-cancel"),
  settingsSave: document.querySelector("#settings-save"),
  temperature: document.querySelector("#temperature"),
  maxTokens: document.querySelector("#max-tokens"),
  systemPrompt: document.querySelector("#system-prompt"),
  openaiUrl: document.querySelector("#openai-url"),
  copyOpenAIUrl: document.querySelector("#copy-openai-url"),
  setupModal: document.querySelector("#model-setup-modal"),
  setupClose: document.querySelector("#setup-close"),
  setupTitle: document.querySelector("#setup-title"),
  setupCopy: document.querySelector("#setup-copy"),
  setupCancel: document.querySelector("#setup-cancel"),
  setupDownload: document.querySelector("#setup-download"),
  modelChoiceList: document.querySelector("#model-choice-list"),
  storageSummary: document.querySelector("#storage-summary"),
  setupAlert: document.querySelector("#setup-alert"),
  downloadStatus: document.querySelector("#download-status"),
  downloadStage: document.querySelector("#download-stage"),
  downloadPercent: document.querySelector("#download-percent"),
  downloadBar: document.querySelector("#download-progress-bar"),
  downloadDetail: document.querySelector("#download-detail"),
  setupRuntimeStatus: document.querySelector("#setup-runtime-status"),
  connectionDot: document.querySelector("#connection-dot"),
  connectionLabel: document.querySelector("#connection-label"),
  statusDetail: document.querySelector("#status-detail"),
};

const defaultSettings = {
  temperature: 0.6,
  maxTokens: "",
  systemPrompt: "",
};

const TEXT_SCALE_STORAGE_KEY = "hybridchat-text-scale";
const MIN_TEXT_SCALE = 0.8;
const MAX_TEXT_SCALE = 1.6;
const TEXT_SCALE_STEP = 0.1;
const disclosureState = new Map();
const disclosureTags = new Set(["think", "latent", "search", "result"]);
let conversationRenderPending = false;

function normalizeTextScale(value) {
  const scale = Number(value);
  if (!Number.isFinite(scale) || scale === 0) return 1;
  return Math.round(Math.min(MAX_TEXT_SCALE, Math.max(MIN_TEXT_SCALE, scale)) * 100) / 100;
}

let settings = { ...defaultSettings };
let sessions = [];
let activeSessionId = null;
let saveTimer = null;
let healthTimer = null;
let healthInFlight = false;
let newestHealth = null;
let modelCatalog = null;
let selectedModelId = localStorage.getItem("hybridchat-selected-model") || "";
let textScale = normalizeTextScale(localStorage.getItem(TEXT_SCALE_STORAGE_KEY));
let modelActionBusy = false;
let modelDownloadActive = false;
let pendingDownloadIds = [];

document.documentElement.style.zoom = String(textScale);

function setTextScale(value) {
  textScale = normalizeTextScale(value);
  document.documentElement.style.zoom = String(textScale);
  localStorage.setItem(TEXT_SCALE_STORAGE_KEY, textScale.toFixed(2));
}

function makeId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function activeSession() {
  return sessions.find((session) => session.id === activeSessionId) || null;
}

function normalizeSession(session) {
  if (!session || typeof session !== "object") return null;
  const messages = Array.isArray(session.messages) ? session.messages : [];
  return {
    id: String(session.id || makeId()),
    title: String(session.title || "New chat"),
    createdAt: Number(session.createdAt) || Date.now(),
    updatedAt: Number(session.updatedAt) || Date.now(),
    messages: messages
      .filter((message) => message && (message.role === "user" || message.role === "assistant"))
      .map((message) => ({
        id: String(message.id || makeId()),
        role: message.role,
        text: String(message.text || ""),
        modelId: message.modelId ? String(message.modelId) : "",
        createdAt: Number(message.createdAt) || Date.now(),
        streaming: false,
        stopped: Boolean(message.stopped),
        error: message.error ? String(message.error) : "",
        meta: message.meta && typeof message.meta === "object" ? message.meta : null,
      })),
  };
}

function createSession(render = true) {
  const now = Date.now();
  const session = {
    id: makeId(),
    title: "New chat",
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
  sessions.push(session);
  activeSessionId = session.id;
  if (render) {
    renderAll();
    elements.prompt.focus();
    scheduleSave();
  }
  return session;
}

function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    if (!invoke) return;
    try {
      await invoke("save_sessions", { sessionsJson: JSON.stringify(sessions) });
    } catch (error) {
      console.error("Could not save chat sessions", error);
    }
  }, 350);
}

function loadSettings() {
  try {
    const stored = JSON.parse(localStorage.getItem("hybridchat-settings") || "{}");
    settings = { ...defaultSettings, ...stored };
    settings.systemPrompt = typeof settings.systemPrompt === "string" ? settings.systemPrompt : "";
  } catch {
    settings = { ...defaultSettings };
  }
  elements.temperature.value = settings.temperature;
  elements.maxTokens.value = settings.maxTokens;
  elements.systemPrompt.value = settings.systemPrompt;
}

function saveSettings() {
  const temperature = Number(elements.temperature.value);
  const maxTokens = elements.maxTokens.value.trim();
  settings = {
    temperature: Number.isFinite(temperature) ? Math.max(0, Math.min(2, temperature)) : defaultSettings.temperature,
    maxTokens: maxTokens && Number.isFinite(Number(maxTokens)) ? String(Math.max(1, Number(maxTokens))) : "",
    systemPrompt: elements.systemPrompt.value,
  };
  localStorage.setItem("hybridchat-settings", JSON.stringify(settings));
  elements.temperature.value = settings.temperature;
  elements.maxTokens.value = settings.maxTokens;
  elements.settingsModal.classList.add("hidden");
  renderStatus();
}

function formatGB(bytes) {
  const value = Number(bytes) || 0;
  return `${(value / 1_000_000_000).toFixed(1)} GB`;
}

function selectedModel() {
  return modelCatalog?.models?.find((model) => model.id === selectedModelId) || null;
}

function modelName(id) {
  if (id?.includes("14B")) return "HybridAgent 14B";
  if (id?.includes("8B")) return "HybridAgent 8B";
  return "HybridAgent";
}

function renderModelControls() {
  if (!modelCatalog?.models) return;
  elements.modelSelect.replaceChildren();
  for (const model of modelCatalog.models) {
    const option = document.createElement("option");
    option.value = model.id;
    const state = model.installed ? "ready" : model.sourceAvailable ? "source" : "download";
    const size = model.id.includes("14B") ? "14B" : "8B";
    option.textContent = `${size} · MLX Q6 · ${state}`;
    elements.modelSelect.append(option);
  }

  if (!modelCatalog.models.some((model) => model.id === selectedModelId)) {
    selectedModelId = modelCatalog.models.find((model) => model.installed)?.id || modelCatalog.models[0]?.id || "";
  }
  elements.modelSelect.value = selectedModelId;
  localStorage.setItem("hybridchat-selected-model", selectedModelId);

  const model = selectedModel();
  const loaded = newestHealth?.loaded && newestHealth.model === model?.id;
  elements.modelAction.classList.toggle("unload-action", Boolean(loaded));
  elements.modelAction.disabled = modelActionBusy || modelDownloadActive || !model;
  if (modelActionBusy) elements.modelAction.textContent = "working…";
  else if (modelDownloadActive) elements.modelAction.textContent = "downloading…";
  else if (!model) elements.modelAction.textContent = "load";
  else if (!model.installed) elements.modelAction.textContent = model.sourceAvailable ? "install" : "download";
  else elements.modelAction.textContent = loaded ? "unload" : newestHealth?.loaded ? "switch / load" : "load";
}

function selectedSetupModels() {
  if (!elements.modelChoiceList) return [];
  const selectedIds = new Set(
    [...elements.modelChoiceList.querySelectorAll("input[type=checkbox]:checked")].map((input) => input.value),
  );
  return (modelCatalog?.models || []).filter((model) => selectedIds.has(model.id) && !model.installed);
}

function estimatePeakSpace(models) {
  const ordered = [...models].sort((a, b) => Number(b.sourceBytes) - Number(a.sourceBytes));
  let retainedOutputs = 0;
  let peak = 0;
  let finalSize = 0;
  for (const model of ordered) {
    const sourceNeeded = model.sourceAvailable
      ? 0
      : Math.max(0, Number(model.sourceBytes) - Number(model.downloadedBytes || 0));
    const output = Number(model.quantizedBytes) || 0;
    peak = Math.max(peak, retainedOutputs + sourceNeeded + output);
    retainedOutputs += output;
    finalSize += output;
  }
  const runtimeReserve = models.length ? Number(modelCatalog?.runtimeReserveBytes || 0) : 0;
  return { peak, finalSize, required: peak + runtimeReserve, runtimeReserve };
}

function updateStorageSummary() {
  const selected = selectedSetupModels();
  const { peak, finalSize, required, runtimeReserve } = estimatePeakSpace(selected);
  const free = Number(modelCatalog?.availableBytes) || 0;
  const enough = !selected.length || free >= required;
  elements.storageSummary.classList.toggle("insufficient", !enough);
  if (!selected.length) {
    const allInstalled = modelCatalog?.models?.length && modelCatalog.models.every((model) => model.installed);
    elements.storageSummary.textContent = allInstalled
      ? `Both models are installed. ${formatGB(free)} remains free on the model drive; choose a model in the toolbar to load it.`
      : `Free on the model drive: ${formatGB(free)}. Select one or both models to see the required space.`;
    elements.setupAlert.classList.add("hidden");
  } else {
    elements.storageSummary.replaceChildren();
    const line1 = document.createElement("div");
    line1.textContent = `Selected MLX 6-bit models use about ${formatGB(finalSize)} once installed.`;
    const line2 = document.createElement("div");
    line2.textContent = `Setup needs about ${formatGB(required)} free at peak (${formatGB(peak)} for download/conversion + ${formatGB(runtimeReserve)} runtime reserve).`;
    const line3 = document.createElement("div");
    line3.textContent = `Available on the destination drive: ${formatGB(free)}.`;
    elements.storageSummary.append(line1, line2, line3);
    if (!enough) {
      elements.setupAlert.textContent = `Not enough free disk space for this selection. Free up ${formatGB(required - free)} or select fewer models.`;
      elements.setupAlert.classList.remove("hidden");
    } else if (!modelDownloadActive) {
      elements.setupAlert.classList.add("hidden");
    }
  }
  elements.setupDownload.disabled = !selected.length || !enough || modelDownloadActive;
}

function renderModelChoices(preselectId = "") {
  elements.modelChoiceList.replaceChildren();
  const models = modelCatalog?.models || [];
  const allInstalled = models.length > 0 && models.every((model) => model.installed);
  elements.setupTitle.textContent = allInstalled ? "Models on this Mac" : "Confirm HybridAgent model download?";
  elements.setupCopy.textContent = allInstalled
    ? "Both checkpoints are installed. Select a model in the toolbar and use Load/Unload to manage MLX unified memory."
    : "HybridChat downloads the official BF16 checkpoint, converts it to a Mac-native MLX 6-bit model, and keeps the HybridLink adapter for latent reasoning.";
  const notInstalled = models.filter((model) => !model.installed);
  const defaultId = preselectId || selectedModelId || notInstalled.find((model) => model.id.includes("8B"))?.id || notInstalled[0]?.id;

  for (const model of models) {
    const label = document.createElement("label");
    label.className = "model-choice";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = model.id;
    checkbox.checked = !model.installed && model.id === defaultId;
    checkbox.disabled = model.installed || modelDownloadActive;
    checkbox.addEventListener("change", updateStorageSummary);

    const details = document.createElement("span");
    const title = document.createElement("span");
    title.className = "model-choice-name";
    title.textContent = modelName(model.id);
    const description = document.createElement("span");
    description.className = "model-choice-description";
    description.textContent = model.installed
      ? "Already installed · MLX 6-bit · HybridLink enabled"
      : model.sourceAvailable
        ? `Local BF16 files found; convert to MLX 6-bit · final size ~${formatGB(model.quantizedBytes)}`
        : `Official Hugging Face BF16 source ~${formatGB(model.sourceBytes)} · final MLX 6-bit size ~${formatGB(model.quantizedBytes)}`;
    details.append(title, description);

    const size = document.createElement("span");
    size.className = "model-choice-size";
    size.textContent = model.installed ? "ready" : formatGB(model.quantizedBytes);
    label.append(checkbox, details, size);
    elements.modelChoiceList.append(label);
  }
  if (allInstalled && !modelDownloadActive) {
    elements.setupDownload.textContent = "all installed";
    elements.setupCancel.textContent = "close";
  } else if (!modelDownloadActive) {
    elements.setupDownload.textContent = "download & install";
    elements.setupCancel.textContent = "later";
  }
  updateStorageSummary();
}

function openModelSetup(preselectId = "") {
  elements.setupAlert.classList.add("hidden");
  elements.setupRuntimeStatus.textContent = "";
  if (!modelCatalog) {
    elements.setupAlert.textContent = "Model information is not ready yet. Try again in a moment.";
    elements.setupAlert.classList.remove("hidden");
  }
  renderModelChoices(preselectId);
  elements.downloadStatus.classList.add("hidden");
  elements.setupModal.classList.remove("hidden");
}

function closeModelSetup() {
  if (modelDownloadActive) return;
  elements.setupModal.classList.add("hidden");
  localStorage.setItem("hybridchat-setup-dismissed", "1");
}

function shortTitle(text) {
  const compact = text.replace(/\s+/g, " ").trim();
  if (!compact) return "New chat";
  return compact.length > 31 ? `${compact.slice(0, 30)}…` : compact;
}

function renderTabs() {
  elements.tabs.replaceChildren();
  for (const session of sessions) {
    const tab = document.createElement("button");
    tab.type = "button";
    tab.className = `tab${session.id === activeSessionId ? " active" : ""}`;
    tab.title = session.title;
    tab.addEventListener("click", () => {
      activeSessionId = session.id;
      renderAll();
    });

    const label = document.createElement("span");
    label.className = "tab-label";
    label.textContent = session.title;
    tab.append(label);

    const close = document.createElement("span");
    close.className = "tab-close";
    close.setAttribute("role", "button");
    close.setAttribute("aria-label", `Close ${session.title}`);
    close.tabIndex = 0;
    close.textContent = "×";
    close.addEventListener("click", (event) => {
      event.stopPropagation();
      closeSession(session.id);
    });
    close.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        event.stopPropagation();
        closeSession(session.id);
      }
    });
    tab.append(close);
    elements.tabs.append(tab);
  }
}

function closeSession(sessionId) {
  const index = sessions.findIndex((session) => session.id === sessionId);
  if (index < 0) return;
  const session = sessions[index];
  if (session.messages.some((message) => message.streaming)) {
    invoke?.("stop_chat", { sessionId }).catch(() => {});
  }
  sessions.splice(index, 1);
  if (activeSessionId === sessionId) {
    activeSessionId = sessions[Math.max(0, index - 1)]?.id || null;
  }
  if (!sessions.length) createSession(false);
  renderAll();
  scheduleSave();
}

function formatTime(timestamp) {
  return new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" }).format(new Date(timestamp));
}

function appendPlain(parent, text) {
  if (!text) return;
  const node = document.createElement("div");
  node.className = "plain-segment";
  node.textContent = text;
  parent.append(node);
}

function stripBoxedDelimiters(text, hideUnclosed = false) {
  const marker = "\\boxed{";
  let result = "";
  let cursor = 0;
  while (cursor < text.length) {
    const start = text.indexOf(marker, cursor);
    if (start < 0) break;
    result += text.slice(cursor, start);

    const contentStart = start + marker.length;
    let depth = 1;
    let closing = -1;
    for (let index = contentStart; index < text.length; index += 1) {
      const character = text[index];
      if (character !== "{" && character !== "}") continue;
      let backslashes = 0;
      for (let previous = index - 1; previous >= 0 && text[previous] === "\\"; previous -= 1) {
        backslashes += 1;
      }
      if (backslashes % 2 === 1) continue;
      if (character === "{") depth += 1;
      else depth -= 1;
      if (depth === 0) {
        closing = index;
        break;
      }
    }

    if (closing < 0) {
      return result + text.slice(hideUnclosed ? contentStart : start);
    }
    result += stripBoxedDelimiters(text.slice(contentStart, closing), hideUnclosed);
    cursor = closing + 1;
  }
  return result + text.slice(cursor);
}

function cleanFinalAnswer(text, hideUnclosed = false) {
  let answer = stripBoxedDelimiters(text, hideUnclosed).trimStart();
  const preamble = "the final answer is";
  const beginning = answer.trimStart().toLowerCase();
  if (hideUnclosed && preamble.startsWith(beginning) && beginning.length < preamble.length) return "";
  answer = answer.replace(/^the\s+final\s+answer\s+is\s*:?\s*/i, "");

  const opening = "\\[";
  const closing = "\\]";
  if (answer.startsWith(opening)) {
    const end = answer.lastIndexOf(closing);
    if (end >= opening.length) {
      const suffix = answer.slice(end + closing.length).trim();
      if (!suffix || /^[.!?]$/.test(suffix)) {
        answer = answer.slice(opening.length, end).trim();
      }
    } else if (hideUnclosed) {
      answer = answer.slice(opening.length);
    }
  }
  return answer.trim();
}

function appendDisclosure(parent, tag, body, depth, traceContext) {
  const thinking = tag === "think";
  const latent = tag === "latent";
  const disclosure = document.createElement("details");
  disclosure.className = thinking ? "reasoning-disclosure think-disclosure" : `hybrid-block ${tag}-block reasoning-disclosure`;
  if (traceContext) {
    const traceIndex = traceContext.counts[tag] || 0;
    traceContext.counts[tag] = traceIndex + 1;
    const traceKey = `${traceContext.messageId}:${tag}:${traceIndex}`;
    disclosure.dataset.traceKey = traceKey;
    disclosure.open = disclosureState.get(traceKey) === true;
    disclosure.addEventListener("toggle", () => {
      if (!disclosure.isConnected) return;
      if (disclosure.open) disclosureState.set(traceKey, true);
      else disclosureState.delete(traceKey);
    });
  }

  const summary = document.createElement("summary");
  summary.className = thinking ? "reasoning-summary" : "hybrid-block-label";
  summary.textContent = thinking ? "thinking" : latent ? "latent thought" : tag;

  const content = document.createElement("div");
  content.className = thinking
    ? "reasoning-segment"
    : `hybrid-block-body${latent ? " latent-thought-body" : ""}`;
  if (thinking) appendSegments(content, body, depth + 1, traceContext);
  else content.textContent = body;

  disclosure.append(summary, content);
  parent.append(disclosure);
}

function appendHybridBlock(parent, tag, body, partial = false) {
  const block = document.createElement("section");
  block.className = `hybrid-block ${tag}-block`;
  const label = document.createElement("div");
  label.className = "hybrid-block-label";
  label.textContent = tag;
  const content = document.createElement("div");
  content.className = "hybrid-block-body";
  if (tag === "answer") content.textContent = cleanFinalAnswer(body, partial);
  else content.textContent = body;
  block.append(label, content);
  parent.append(block);
}

function appendSegments(parent, text, depth = 0, traceContext = null) {
  if (!text) return;
  if (depth > 3) {
    appendPlain(parent, text);
    return;
  }

  const tags = ["think", "latent", "search", "python", "result", "answer"];
  let cursor = 0;
  while (cursor < text.length) {
    let next = null;
    let pendingTag = null;
    for (const tag of tags) {
      const openTag = `<${tag}>`;
      const closeTag = `</${tag}>`;
      const open = text.indexOf(openTag, cursor);
      if (open < 0) continue;
      const close = text.indexOf(closeTag, open + openTag.length);
      if (close < 0) {
        if (!pendingTag || open < pendingTag.open) pendingTag = { tag, open, openTag };
        continue;
      }
      if (!next || open < next.open) next = { tag, open, close, openTag, closeTag };
    }
    if (pendingTag && (!next || pendingTag.open < next.open)) {
      appendPlain(parent, text.slice(cursor, pendingTag.open));
      const body = text.slice(pendingTag.open + pendingTag.openTag.length);
      if (disclosureTags.has(pendingTag.tag)) {
        appendDisclosure(parent, pendingTag.tag, body, depth, traceContext);
      } else {
        appendHybridBlock(parent, pendingTag.tag, body, true);
      }
      break;
    }
    if (!next) {
      appendPlain(parent, text.slice(cursor));
      break;
    }

    appendPlain(parent, text.slice(cursor, next.open));
    const body = text.slice(next.open + next.openTag.length, next.close);
    if (disclosureTags.has(next.tag)) {
      appendDisclosure(parent, next.tag, body, depth, traceContext);
    } else {
      appendHybridBlock(parent, next.tag, body);
    }
    cursor = next.close + next.closeTag.length;
  }
}

function appendMeta(parent, meta) {
  if (!meta) return;
  const items = [];
  if (Number.isFinite(meta.generation_steps)) items.push(`${meta.generation_steps} steps`);
  if (Number.isFinite(meta.latent_steps)) items.push(`${meta.latent_steps} latent`);
  if (Number.isFinite(meta.latent_spans)) items.push(`${meta.latent_spans} spans`);
  const toolCalls = (Number(meta.python_calls) || 0) + (Number(meta.search_calls) || 0);
  if (toolCalls) {
    const details = [];
    if (meta.python_calls) details.push(`${meta.python_calls} python`);
    if (meta.search_calls) details.push(`${meta.search_calls} search`);
    items.push(`${toolCalls} tool${toolCalls === 1 ? "" : "s"} (${details.join(", ")})`);
  }
  if (Number.isFinite(meta.wall_time_s)) items.push(`${Number(meta.wall_time_s).toFixed(1)}s`);
  if (!items.length) return;

  const footer = document.createElement("div");
  footer.className = "message-meta";
  for (const item of items) {
    const chip = document.createElement("span");
    chip.className = "meta-chip";
    chip.textContent = item;
    footer.append(chip);
  }
  parent.append(footer);
}

function renderMessage(message) {
  const article = document.createElement("article");
  article.className = `message ${message.role}`;
  article.dataset.messageId = message.id;

  const header = document.createElement("div");
  header.className = "message-head";
  const role = document.createElement("span");
  role.className = "role-badge";
  role.textContent = message.role === "user" ? "you" : modelName(message.modelId || newestHealth?.model);
  const time = document.createElement("span");
  time.className = "message-time";
  time.textContent = formatTime(message.createdAt);
  header.append(role, time);
  article.append(header);

  if (message.error) {
    const error = document.createElement("div");
    error.className = "message-error";
    error.textContent = message.error;
    article.append(error);
  } else {
    const content = document.createElement("div");
    content.className = "message-text";
    if (message.role === "assistant") {
      const traceContext = { messageId: message.id, counts: { think: 0, latent: 0 } };
      appendSegments(content, message.text.replace(/<\|(?:endoftext|im_end|im_start)\|>/g, ""), 0, traceContext);
    }
    else appendPlain(content, message.text);
    if (message.streaming) {
      if (!message.text) {
        const waiting = document.createElement("span");
        waiting.className = "thinking-label";
        waiting.textContent = "thinking in text + latent space";
        content.append(waiting);
      }
      const cursor = document.createElement("span");
      cursor.className = "stream-cursor";
      content.append(cursor);
    }
    article.append(content);
  }

  if (message.stopped) {
    const stopped = document.createElement("div");
    stopped.className = "message-stopped";
    stopped.textContent = "Display stopped. The local server may finish the current generation in the background.";
    article.append(stopped);
  }
  if (message.role === "assistant") appendMeta(article, message.meta);
  return article;
}

function renderWelcome(parent) {
  const welcome = document.createElement("section");
  welcome.className = "welcome";
  const kicker = document.createElement("div");
  kicker.className = "welcome-kicker";
  kicker.textContent = "LOCAL · HYBRID REASONING";
  const heading = document.createElement("h1");
  heading.textContent = "A different kind of thinking.";
  const copy = document.createElement("p");
  copy.className = "welcome-copy";
  copy.textContent = "HybridAgent moves between readable text and continuous latent thought, then acts through its Python and search tools. Your sessions stay on this Mac.";
  const grid = document.createElement("div");
  grid.className = "welcome-grid";
  const cards = [
    ["latent spans", "Continuous inner reasoning, rendered as pseudo-text."],
    ["tools", "Python and optional Tavily search, executed by the local server."],
    ["session tabs", "Keep separate questions and experiments open side by side."],
  ];
  for (const [title, description] of cards) {
    const card = document.createElement("div");
    card.className = "welcome-card";
    const strong = document.createElement("strong");
    strong.textContent = title;
    const span = document.createElement("span");
    span.textContent = description;
    card.append(strong, span);
    grid.append(card);
  }
  welcome.append(kicker, heading, copy, grid);
  parent.append(welcome);
}

function syncMessageNode(current, next) {
  if (current.nodeType !== next.nodeType
      || current.nodeName !== next.nodeName
      || (current.nodeType === Node.ELEMENT_NODE
        && (current.className !== next.className
          || current.dataset.traceKey !== next.dataset.traceKey))) {
    current.replaceWith(next);
    return;
  }
  if (current.nodeType === Node.TEXT_NODE) {
    if (current.nodeValue !== next.nodeValue) current.nodeValue = next.nodeValue;
    return;
  }

  // Keep native disclosure state and focused controls on the existing nodes.
  for (const attribute of [...current.attributes]) {
    if (current.nodeName === "DETAILS" && attribute.name === "open") continue;
    if (!next.hasAttribute(attribute.name)) current.removeAttribute(attribute.name);
  }
  for (const attribute of [...next.attributes]) {
    if (current.nodeName === "DETAILS" && attribute.name === "open") continue;
    if (current.getAttribute(attribute.name) !== attribute.value) {
      current.setAttribute(attribute.name, attribute.value);
    }
  }
  const previousChildren = [...current.childNodes];
  const nextChildren = [...next.childNodes];
  for (let index = 0; index < Math.max(previousChildren.length, nextChildren.length); index += 1) {
    if (!nextChildren[index]) previousChildren[index].remove();
    else if (!previousChildren[index]) current.append(nextChildren[index]);
    else syncMessageNode(previousChildren[index], nextChildren[index]);
  }
}

function renderConversation() {
  const session = activeSession();
  const container = elements.conversation;
  const pinToBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 48;
  if (container.dataset.sessionId !== session?.id) {
    container.replaceChildren();
    container.dataset.sessionId = session?.id || "";
  }
  if (!session || session.messages.length === 0) {
    if (!container.querySelector(".welcome")) {
      container.replaceChildren();
      renderWelcome(container);
    }
  } else {
    container.querySelector(".welcome")?.remove();
    const existing = new Map([...container.children].map((node) => [node.dataset.messageId, node]));
    for (const [index, message] of session.messages.entries()) {
      let node = existing.get(message.id);
      const signature = JSON.stringify(message);
      if (!node) {
        node = renderMessage(message);
      } else if (node.renderSignature !== signature) {
        // Capture the actual state even if the browser's toggle event is pending.
        for (const disclosure of node.querySelectorAll("details[data-trace-key]")) {
          if (disclosure.open) disclosureState.set(disclosure.dataset.traceKey, true);
          else disclosureState.delete(disclosure.dataset.traceKey);
        }
        syncMessageNode(node, renderMessage(message));
      }
      node.renderSignature = signature;
      if (container.children[index] !== node) container.insertBefore(node, container.children[index] || null);
      existing.delete(message.id);
    }
    for (const node of existing.values()) node.remove();
  }
  if (pinToBottom) container.scrollTop = container.scrollHeight;
  renderStatus();
}

function renderStatus() {
  const session = activeSession();
  const activeMessage = session?.messages.find((message) => message.role === "assistant" && message.streaming);
  const modelLoaded = Boolean(newestHealth?.loaded && newestHealth.model === selectedModelId);
  elements.stop.classList.toggle("hidden", !activeMessage);
  elements.send.classList.toggle("hidden", Boolean(activeMessage));
  elements.prompt.disabled = !modelLoaded || Boolean(activeMessage);
  elements.prompt.placeholder = modelLoaded ? "Ask HybridAgent…" : "Load a model to start chatting…";
  elements.send.disabled = !modelLoaded || !elements.prompt.value.trim();
  if (activeMessage) {
    elements.statusDetail.textContent = activeMessage.text ? "receiving response" : "thinking in latent space…";
  } else if (modelLoaded) {
    elements.statusDetail.textContent = `${modelName(newestHealth.model)} · MLX · unified memory`;
  } else if (newestHealth?.running) {
    elements.statusDetail.textContent = "MLX engine ready · no model loaded";
  } else {
    elements.statusDetail.textContent = "select a model to load it";
  }
}

function renderAll() {
  renderTabs();
  renderModelControls();
  renderConversation();
}

function updateSessionTitle(session, text) {
  if (session.title === "New chat") session.title = shortTitle(text);
}

function findStreamingAssistant(session) {
  for (let index = session.messages.length - 1; index >= 0; index -= 1) {
    const message = session.messages[index];
    if (message.role === "assistant" && message.streaming) return message;
  }
  return null;
}

function compactAssistantAnswer(text) {
  const answers = [...text.matchAll(/<answer>([\s\S]*?)<\/answer>/g)];
  const answer = answers.at(-1)?.[1]?.trim();
  return answer
    ? cleanFinalAnswer(answer)
    : text.replace(/<latent>[\s\S]*?<\/latent>/g, "").trim().slice(0, 1800);
}

function conversationQuestion(session, latestUserMessage) {
  const history = session.messages
    .filter((message) => message.id !== latestUserMessage.id && !message.error)
    .slice(-8)
    .map((message) => {
      if (message.role === "user") return `User: ${message.text}`;
      const answer = compactAssistantAnswer(message.text);
      return answer ? `Assistant: ${answer}` : "";
    })
    .filter(Boolean);
  if (!history.length) return latestUserMessage.text;
  return `Conversation so far:\n${history.join("\n")}\n\nCurrent question:\n${latestUserMessage.text}`;
}

async function sendMessage() {
  const text = elements.prompt.value.trim();
  const session = activeSession();
  if (!newestHealth?.loaded || newestHealth.model !== selectedModelId) {
    elements.statusDetail.textContent = "Load the selected model before sending a message";
    return;
  }
  if (!text || !session || findStreamingAssistant(session)) return;

  const userMessage = { id: makeId(), role: "user", text, createdAt: Date.now(), streaming: false, meta: null };
  const assistantMessage = {
    id: makeId(), role: "assistant", text: "", modelId: selectedModelId, createdAt: Date.now(), streaming: true, stopped: false, error: "", meta: null,
  };
  session.messages.push(userMessage, assistantMessage);
  session.updatedAt = Date.now();
  updateSessionTitle(session, text);
  elements.prompt.value = "";
  resizePrompt();
  renderAll();
  scheduleSave();

  const request = {
    model: selectedModelId,
    messages: [
      ...(settings.systemPrompt.trim() ? [{ role: "system", content: settings.systemPrompt }] : []),
      { role: "user", content: conversationQuestion(session, userMessage) },
    ],
    stream: true,
    temperature: settings.temperature,
  };
  if (settings.maxTokens) request.max_tokens = Number(settings.maxTokens);

  try {
    await invoke("start_chat", {
      sessionId: session.id,
      requestBody: JSON.stringify(request),
    });
  } catch (error) {
    assistantMessage.streaming = false;
    assistantMessage.error = String(error);
    renderConversation();
    scheduleSave();
  }
}

async function stopActiveResponse() {
  const session = activeSession();
  const message = session && findStreamingAssistant(session);
  if (!session || !message) return;
  message.streaming = false;
  message.stopped = true;
  renderConversation();
  scheduleSave();
  try {
    await invoke("stop_chat", { sessionId: session.id });
  } catch (error) {
    console.error("Could not stop response stream", error);
  }
}

function scheduleConversationRender() {
  if (conversationRenderPending) return;
  conversationRenderPending = true;
  requestAnimationFrame(() => {
    conversationRenderPending = false;
    renderConversation();
  });
}

function handleChatEvent(payload) {
  if (!payload || !payload.sessionId) return;
  const session = sessions.find((candidate) => candidate.id === payload.sessionId);
  if (!session) return;
  const message = findStreamingAssistant(session);
  if (!message) return;

  if (payload.kind === "delta") {
    message.text += String(payload.content || "");
  } else if (payload.kind === "meta") {
    message.meta = payload.meta || null;
  } else if (payload.kind === "error") {
    message.error = String(payload.error || "Unknown server error");
    message.streaming = false;
  } else if (payload.kind === "stopped") {
    message.streaming = false;
    message.stopped = true;
  } else if (payload.kind === "done") {
    message.streaming = false;
  }

  session.updatedAt = Date.now();
  if (payload.kind === "done" || payload.kind === "stopped" || payload.kind === "error") scheduleSave();
  if (session.id === activeSessionId) {
    if (payload.kind === "delta") scheduleConversationRender();
    else renderConversation();
  }
  else renderTabs();
}

async function checkHealth() {
  if (!invoke || healthInFlight) return;
  healthInFlight = true;
  try {
    const health = await invoke("engine_status");
    newestHealth = health;
    elements.connectionDot.className = `connection-dot${health?.loaded ? " online" : health?.running ? " idle" : " offline"}`;
    elements.connectionLabel.textContent = health?.loaded
      ? `${modelName(health.model)} · MLX`
      : health?.running ? "engine ready" : "engine stopped";
    elements.openaiUrl.textContent = health?.baseUrl || "Load a model to start the local API";
  } catch {
    newestHealth = null;
    elements.connectionDot.className = "connection-dot offline";
    elements.connectionLabel.textContent = "engine unavailable";
    elements.openaiUrl.textContent = "Load a model to start the local API";
  } finally {
    healthInFlight = false;
    renderModelControls();
    renderStatus();
  }
}

async function copyOpenAIUrl() {
  const url = newestHealth?.baseUrl;
  if (!url) {
    elements.openaiUrl.textContent = "Load a model to start the local API";
    return;
  }
  try {
    await navigator.clipboard.writeText(url);
    elements.copyOpenAIUrl.textContent = "copied";
    setTimeout(() => { elements.copyOpenAIUrl.textContent = "copy"; }, 1200);
  } catch (error) {
    console.error("Could not copy API URL", error);
  }
}

async function refreshModelCatalog() {
  if (!invoke) return;
  try {
    modelCatalog = await invoke("model_catalog");
    if (!modelCatalog.models.some((model) => model.id === selectedModelId)) {
      selectedModelId = modelCatalog.models.find((model) => model.installed)?.id || modelCatalog.models[0]?.id || "";
    }
    renderModelControls();
    renderStatus();
  } catch (error) {
    console.error("Could not load model catalog", error);
    elements.connectionLabel.textContent = "model catalog unavailable";
  }
}

async function handleModelAction() {
  if (elements.modelSelect.value) {
    selectedModelId = elements.modelSelect.value;
    localStorage.setItem("hybridchat-selected-model", selectedModelId);
  }
  const model = selectedModel();
  if (!model || modelActionBusy || modelDownloadActive) return;
  if (!model.installed) {
    openModelSetup(model.id);
    return;
  }

  const loaded = newestHealth?.loaded && newestHealth.model === model.id;
  modelActionBusy = true;
  renderModelControls();
  elements.statusDetail.textContent = loaded ? "unloading model from unified memory…" : `loading ${modelName(model.id)} into unified memory…`;
  try {
    newestHealth = loaded
      ? await invoke("unload_model")
      : await invoke("load_model", { modelId: model.id });
    await checkHealth();
  } catch (error) {
    elements.connectionLabel.textContent = "model operation failed";
    elements.statusDetail.textContent = String(error);
    elements.connectionDot.className = "connection-dot offline";
  } finally {
    modelActionBusy = false;
    renderModelControls();
    renderStatus();
  }
}

async function beginModelDownload() {
  const selected = selectedSetupModels();
  if (!selected.length || modelDownloadActive) return;
  pendingDownloadIds = selected.map((model) => model.id);
  modelDownloadActive = true;
  elements.setupDownload.textContent = "preparing…";
  elements.setupCancel.textContent = "cancel";
  elements.downloadStatus.classList.remove("hidden");
  elements.downloadStage.textContent = "Preparing local runtime and downloads";
  elements.downloadPercent.textContent = "0%";
  elements.downloadBar.style.width = "0%";
  elements.downloadDetail.textContent = "Checking model files and disk space…";
  elements.setupRuntimeStatus.textContent = "";
  updateStorageSummary();
  try {
    await invoke("start_model_download", { modelIds: pendingDownloadIds });
  } catch (error) {
    modelDownloadActive = false;
    elements.setupAlert.textContent = String(error);
    elements.setupAlert.classList.remove("hidden");
    elements.setupDownload.textContent = "download & install";
    renderModelChoices();
  }
}

async function cancelModelDownload() {
  if (!modelDownloadActive) {
    closeModelSetup();
    return;
  }
  elements.setupRuntimeStatus.textContent = "Stopping after the current file finishes…";
  try {
    await invoke("cancel_model_download");
  } catch (error) {
    console.error("Could not cancel model setup", error);
  }
}

function handleModelEvent(payload) {
  if (!payload) return;
  if (payload.kind === "runtime-setup") {
    elements.downloadStatus.classList.remove("hidden");
    elements.downloadStage.textContent = "Preparing local inference runtime";
    elements.setupRuntimeStatus.textContent = payload.message || "Setting up MLX…";
  } else if (payload.kind === "started") {
    elements.downloadStatus.classList.remove("hidden");
    elements.downloadStage.textContent = "Downloading Hugging Face files";
    elements.downloadDetail.textContent = `${formatGB(payload.downloadedBytes)} / ${formatGB(payload.totalBytes)}`;
  } else if (payload.kind === "progress") {
    const total = Number(payload.totalBytes) || 0;
    const done = Number(payload.downloadedBytes) || 0;
    const percent = total ? Math.min(100, Math.floor((done / total) * 100)) : 0;
    elements.downloadStage.textContent = "Downloading model files";
    elements.downloadPercent.textContent = `${percent}%`;
    elements.downloadBar.style.width = `${percent}%`;
    elements.downloadDetail.textContent = `${payload.modelId || ""} · ${payload.fileName || ""} · ${formatGB(done)} / ${formatGB(total)}`;
  } else if (payload.kind === "quantizing") {
    elements.downloadStage.textContent = `Compiling ${modelName(payload.modelId)} to MLX 6-bit`;
    elements.downloadPercent.textContent = "";
    elements.downloadBar.style.width = "100%";
    elements.downloadDetail.textContent = "Converting weights and keeping the HybridLink adapter…";
  } else if (payload.kind === "model-complete") {
    elements.downloadStage.textContent = `${modelName(payload.modelId)} installed`;
    elements.downloadPercent.textContent = "100%";
    elements.downloadBar.style.width = "100%";
    refreshModelCatalog();
  } else if (payload.kind === "complete") {
    modelDownloadActive = false;
    if (pendingDownloadIds.length && pendingDownloadIds.includes(selectedModelId)) {
      selectedModelId = pendingDownloadIds[0];
      localStorage.setItem("hybridchat-selected-model", selectedModelId);
    }
    elements.downloadStage.textContent = "Selected models ready";
    elements.downloadPercent.textContent = "100%";
    elements.downloadBar.style.width = "100%";
    elements.downloadDetail.textContent = "Choose a model in the toolbar, then press Load.";
    elements.setupRuntimeStatus.textContent = "";
    refreshModelCatalog();
    elements.setupDownload.textContent = "download & install";
    elements.setupCancel.textContent = "done";
  } else if (payload.kind === "cancelled") {
    modelDownloadActive = false;
    elements.downloadStage.textContent = "Setup paused";
    elements.downloadDetail.textContent = "Downloaded files are kept so setup can resume later.";
    elements.setupDownload.textContent = "download & install";
    elements.setupCancel.textContent = "later";
    refreshModelCatalog();
  } else if (payload.kind === "error") {
    modelDownloadActive = false;
    elements.setupAlert.textContent = String(payload.message || "Model setup failed");
    elements.setupAlert.classList.remove("hidden");
    elements.setupDownload.textContent = "retry setup";
    elements.setupCancel.textContent = "later";
    refreshModelCatalog();
  }
  renderModelControls();
}

function handleRuntimeEvent(payload) {
  if (!payload) return;
  if (payload.kind === "progress") {
    elements.downloadStatus.classList.remove("hidden");
    elements.downloadStage.textContent = "Preparing local inference runtime";
    elements.setupRuntimeStatus.textContent = String(payload.message || "Setting up MLX…");
    if (modelActionBusy) elements.statusDetail.textContent = String(payload.message || "Preparing MLX runtime…");
  }
}

function resizePrompt() {
  elements.prompt.style.height = "auto";
  elements.prompt.style.height = `${Math.min(elements.prompt.scrollHeight, 180)}px`;
  elements.send.disabled = !elements.prompt.value.trim();
}

function openSettings() {
  elements.temperature.value = settings.temperature;
  elements.maxTokens.value = settings.maxTokens;
  elements.systemPrompt.value = settings.systemPrompt;
  elements.openaiUrl.textContent = newestHealth?.baseUrl || "Load a model to start the local API";
  elements.settingsModal.classList.remove("hidden");
  elements.temperature.focus();
}

function closeSettings() {
  elements.settingsModal.classList.add("hidden");
}

async function initialize() {
  if (!invoke || !tauri?.event?.listen) {
    elements.connectionLabel.textContent = "launch with Tauri";
    elements.statusDetail.textContent = "Tauri runtime required";
    return;
  }

  loadSettings();
  try {
    const stored = await invoke("load_sessions");
    const parsed = JSON.parse(stored || "[]");
    sessions = Array.isArray(parsed) ? parsed.map(normalizeSession).filter(Boolean) : [];
  } catch (error) {
    console.error("Could not load saved sessions", error);
    sessions = [];
  }
  if (!sessions.length) createSession(false);
  activeSessionId = sessions[sessions.length - 1]?.id || sessions[0].id;

  await tauri.event.listen("chat-event", (event) => handleChatEvent(event.payload));
  await tauri.event.listen("model-event", (event) => handleModelEvent(event.payload));
  await tauri.event.listen("runtime-event", (event) => handleRuntimeEvent(event.payload));
  renderAll();
  await refreshModelCatalog();
  checkHealth();
  healthTimer = setInterval(checkHealth, 7000);
  scheduleSave();
  if (modelCatalog?.models?.every((model) => !model.installed)
      && localStorage.getItem("hybridchat-setup-dismissed") !== "1") {
    openModelSetup();
  }
}

elements.newTab.addEventListener("click", () => createSession());
elements.send.addEventListener("click", sendMessage);
elements.stop.addEventListener("click", stopActiveResponse);
elements.modelSelect.addEventListener("change", () => {
  selectedModelId = elements.modelSelect.value;
  localStorage.setItem("hybridchat-selected-model", selectedModelId);
  renderModelControls();
  renderStatus();
});
elements.modelAction.addEventListener("click", handleModelAction);
elements.manageModels.addEventListener("click", () => openModelSetup());
elements.settingsButton.addEventListener("click", openSettings);
elements.settingsClose.addEventListener("click", closeSettings);
elements.settingsCancel.addEventListener("click", closeSettings);
elements.settingsSave.addEventListener("click", saveSettings);
elements.copyOpenAIUrl.addEventListener("click", copyOpenAIUrl);
elements.setupClose.addEventListener("click", closeModelSetup);
elements.setupCancel.addEventListener("click", cancelModelDownload);
elements.setupDownload.addEventListener("click", beginModelDownload);
elements.setupModal.addEventListener("click", (event) => {
  if (event.target === elements.setupModal && !modelDownloadActive) closeModelSetup();
});
elements.settingsModal.addEventListener("click", (event) => {
  if (event.target === elements.settingsModal) closeSettings();
});
elements.prompt.addEventListener("input", resizePrompt);
elements.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  } else if (event.key === "Escape") {
    stopActiveResponse();
  }
});

document.addEventListener("keydown", (event) => {
  const commandKey = event.metaKey || event.ctrlKey;
  const increaseTextSize = event.key === "+" || event.key === "=" || event.code === "Equal" || event.code === "NumpadAdd";
  const decreaseTextSize = event.key === "-" || event.key === "_" || event.code === "NumpadSubtract";
  if (commandKey && increaseTextSize) {
    event.preventDefault();
    setTextScale(textScale + TEXT_SCALE_STEP);
  } else if (commandKey && decreaseTextSize) {
    event.preventDefault();
    setTextScale(textScale - TEXT_SCALE_STEP);
  } else if (commandKey && event.key === "0") {
    event.preventDefault();
    setTextScale(1);
  } else if (commandKey && event.key.toLowerCase() === "t") {
    event.preventDefault();
    createSession();
  } else if (commandKey && event.key.toLowerCase() === "w") {
    event.preventDefault();
    if (!elements.setupModal.classList.contains("hidden")) closeModelSetup();
    else if (!elements.settingsModal.classList.contains("hidden")) closeSettings();
    else if (activeSessionId) closeSession(activeSessionId);
  } else if (event.key === "Escape" && !elements.settingsModal.classList.contains("hidden")) {
    closeSettings();
  } else if (event.key === "Escape" && !elements.setupModal.classList.contains("hidden")) {
    cancelModelDownload();
  }
});

initialize();
