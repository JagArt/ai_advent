const form = document.getElementById("form");
const questionInput = document.getElementById("question");
const submitButton = document.getElementById("submit");
const statusEl = document.getElementById("status");
const presetsEl = document.getElementById("presets");
const recheckButton = document.getElementById("recheck");
const resetButton = document.getElementById("reset");
const fileListEl = document.getElementById("fileList");
const filesMeta = document.getElementById("filesMeta");
const messagesEl = document.getElementById("messages");
const chatMeta = document.getElementById("chatMeta");
const clearButton = document.getElementById("clearChat");
const previewEmpty = document.getElementById("previewEmpty");
const previewBody = document.getElementById("previewBody");
const previewTitle = document.getElementById("previewTitle");
const previewMeta = document.getElementById("previewMeta");
const tabFile = document.getElementById("tabFile");
const tabDiff = document.getElementById("tabDiff");

const dots = {
    mcp: [document.getElementById("mcpDot"), document.getElementById("mcpText")],
    local: [document.getElementById("localDot"), document.getElementById("localText")],
    cloud: [document.getElementById("cloudDot"), document.getElementById("cloudText")],
};

const submitLabel = submitButton.textContent;

let controller = null;
let history = [];
let selectedPath = null;
let latestDiff = "";
let previewMode = "file";
let changedFiles = [];

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", isError);
}

function setBusy(busy) {
    submitButton.textContent = busy ? "Остановить" : submitLabel;
    submitButton.classList.toggle("stop", busy);
    for (const button of presetsEl.querySelectorAll("button")) button.disabled = busy;
    resetButton.disabled = busy;
}

function setDot(name, state, text) {
    const [dot, label] = dots[name];
    dot.className = `dot ${state}`;
    label.textContent = text;
}

function selectedProvider() {
    return document.querySelector('input[name="provider"]:checked').value;
}

function formatArgs(args) {
    return Object.entries(args)
        .map(([key, value]) => {
            const shown = typeof value === "string" && value.length > 40 ? `${value.slice(0, 40)}…` : value;
            return `${key}=${JSON.stringify(shown)}`;
        })
        .join(", ");
}

function scrollMessages() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setMode(mode) {
    previewMode = mode;
    tabFile.classList.toggle("active", mode === "file");
    tabDiff.classList.toggle("active", mode === "diff");
}

function renderDiffText(text) {
    previewBody.textContent = "";
    if (!text) {
        previewBody.textContent = "(нет отличий от эталона)";
        return;
    }
    for (const line of text.split("\n")) {
        const span = el("span");
        if (line.startsWith("+") && !line.startsWith("+++")) span.className = "add";
        else if (line.startsWith("-") && !line.startsWith("---")) span.className = "del";
        else if (line.startsWith("@@")) span.className = "hunk";
        span.textContent = `${line}\n`;
        previewBody.append(span);
    }
}

async function showFile(path) {
    selectedPath = path;
    setMode("file");
    for (const item of fileListEl.children) item.classList.toggle("active", item.dataset.path === path);
    const encoded = path.split("/").map(encodeURIComponent).join("/");
    const response = await fetch(`/api/files/${encoded}`);
    if (!response.ok) {
        previewEmpty.hidden = true;
        previewBody.hidden = false;
        previewBody.textContent = `Не удалось открыть ${path}`;
        return;
    }
    const file = await response.json();
    previewEmpty.hidden = true;
    previewBody.hidden = false;
    previewTitle.textContent = file.path;
    previewMeta.textContent = `${file.lines} строк · ${file.bytes} байт${file.truncated ? " · обрезано" : ""}`;
    previewBody.textContent = file.content;
}

function showDiff(text, changed) {
    latestDiff = text || "";
    if (Array.isArray(changed)) changedFiles = changed;
    setMode("diff");
    previewEmpty.hidden = true;
    previewBody.hidden = false;
    previewTitle.textContent = "Diff относительно эталона";
    previewMeta.textContent = changedFiles.length ? `изменено: ${changedFiles.join(", ")}` : "отличий нет";
    renderDiffText(latestDiff);
}

async function loadFiles() {
    const response = await fetch("/api/files");
    const payload = await response.json();
    changedFiles = payload.changed || [];
    fileListEl.innerHTML = "";
    filesMeta.textContent = `${payload.count} файлов${changedFiles.length ? ` · правок ${changedFiles.length}` : ""}`;
    for (const file of payload.files) {
        const dirty = changedFiles.includes(file.path);
        const li = el("li", `file${dirty ? " dirty" : ""}`);
        li.dataset.path = file.path;
        li.append(el("span", "file-path", file.path));
        if (dirty) li.append(el("span", "badge changed", "правка"));
        if (li.dataset.path === selectedPath) li.classList.add("active");
        li.addEventListener("click", () => showFile(file.path));
        fileListEl.append(li);
    }
}

async function checkStatus() {
    setDot("mcp", "", "MCP...");
    setDot("local", "", "LM Studio...");
    setDot("cloud", "", "Облако...");
    try {
        const response = await fetch("/api/status");
        const status = await response.json();

        const tools = status.mcp.tools.map((tool) => tool.name).join(", ");
        const dirty = status.workspace.dirty ? ` · правок ${status.workspace.changed.length}` : "";
        setDot(
            "mcp",
            "on",
            `${status.mcp.server} · ${status.mcp.tools.length} инструментов: ${tools} · файлов ${status.workspace.files}${dirty}`,
        );

        if (!status.local.online) {
            setDot("local", "off", status.local.error);
        } else if (!status.local.model_loaded) {
            setDot("local", "off", `${status.local.base_url} · модели ${status.local.model} нет в списке: ${status.local.models.join(", ")}`);
        } else {
            setDot("local", "on", `${status.local.base_url} · ${status.local.model}`);
        }
        document.getElementById("localModel").textContent = status.local.model;

        if (status.cloud.configured) {
            setDot("cloud", "on", `DeepSeek · ${status.cloud.model}`);
        } else {
            setDot("cloud", "off", "DEEPSEEK_API_KEY не задан — работает только локальная модель");
        }
        document.getElementById("cloudModel").textContent = status.cloud.model;

        if (!presetsEl.childElementCount) {
            for (const preset of status.presets) {
                const button = el("button", "preset", preset);
                button.type = "button";
                button.addEventListener("click", () => {
                    questionInput.value = preset;
                    form.requestSubmit();
                });
                presetsEl.append(button);
            }
        }
    } catch (error) {
        setDot("mcp", "off", `Не удалось проверить: ${error.message}`);
    }
}

function renderChat() {
    messagesEl.innerHTML = "";
    clearButton.hidden = history.length === 0;
    chatMeta.textContent = history.length ? `ходов: ${Math.floor(history.length / 2)}` : "";
    if (!history.length) {
        messagesEl.append(el("div", "placeholder-box", "Задайте цель словами — ассистент сам откроет файлы"));
        return;
    }
    for (const turn of history) {
        if (turn.role === "user") {
            const node = el("div", "turn");
            node.append(el("div", "turn-q", turn.content));
            messagesEl.append(node);
        } else if (turn.node) {
            messagesEl.append(turn.node);
        } else {
            const node = el("div", "turn-a");
            const answer = el("pre", "answer");
            answer.textContent = turn.content;
            node.append(answer);
            messagesEl.append(node);
        }
    }
    scrollMessages();
}

function newTurn(question) {
    const q = el("div", "turn");
    q.append(el("div", "turn-q", question));
    const a = el("div", "turn-a");
    const trace = el("div", "trace");
    const reasoning = el("details", "reasoning");
    reasoning.hidden = true;
    reasoning.append(el("summary", "", "Рассуждения модели"), el("pre"));
    const answer = el("pre", "answer");
    const meta = el("div", "turn-meta");
    a.append(trace, reasoning, answer, meta);
    if (messagesEl.querySelector(".placeholder-box")) messagesEl.innerHTML = "";
    messagesEl.append(q, a);
    scrollMessages();
    return { node: a, trace, reasoning, answer, meta, text: "", reasoningText: "" };
}

function parseFrame(frame) {
    let event = "message";
    const dataLines = [];
    for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    return { event, data: dataLines.join("\n") };
}

function formatDone(done) {
    const parts = [`${done.model}`, `раундов ${done.rounds}`, `вызовов ${done.calls}`];
    if (done.nudged) parts.push("толчок");
    if (done.files_touched && done.files_touched.length) parts.push(`файлы ${done.files_touched.join(", ")}`);
    if (done.ttft_ms !== null) parts.push(`первый токен ${(done.ttft_ms / 1000).toFixed(1)} с`);
    parts.push(`всего ${(done.total_ms / 1000).toFixed(1)} с`);
    if (done.prompt_tokens !== null) parts.push(`токены ${done.prompt_tokens} → ${done.completion_tokens}`);
    if (done.finish_reason && done.finish_reason !== "stop") parts.push(`finish_reason: ${done.finish_reason}`);
    return parts.join(" · ");
}

function addTrace(turn, line) {
    turn.trace.append(line);
    scrollMessages();
}

async function ask(question, provider) {
    const turn = newTurn(question);
    const priorHistory = history.map(({ role, content }) => ({ role, content }));
    history.push({ role: "user", content: question });
    const assistantTurn = { role: "assistant", content: "", node: turn.node };
    history.push(assistantTurn);
    renderChatMeta();

    const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, provider, history: priorHistory }),
        signal: controller.signal,
    });
    if (!response.ok) throw new Error(`Сервер вернул ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let boundary;
        while ((boundary = buffer.indexOf("\n\n")) !== -1) {
            const { event, data } = parseFrame(buffer.slice(0, boundary));
            buffer = buffer.slice(boundary + 2);
            const payload = JSON.parse(data);

            if (event === "error") throw new Error(payload);
            if (event === "tool") {
                const line = el("div", `trace-line${payload.ok ? "" : " fail"}`);
                line.append(
                    el("span", "", `⚙ r${payload.round}`),
                    el("span", "t-name", ` ${payload.name}`),
                    el("span", "t-args", payload.ok ? `(${formatArgs(payload.arguments)})` : `(${formatArgs(payload.arguments)}) · отказ: ${payload.error}`),
                    el("span", "t-ms", `${payload.ms} мс`),
                );
                line.title = payload.ok ? formatArgs(payload.arguments) : payload.error;
                addTrace(turn, line);
                continue;
            }
            if (event === "clear") {
                turn.text = "";
                turn.answer.textContent = "";
                continue;
            }
            if (event === "nudge") {
                const draft = el("pre", "answer draft");
                draft.textContent = payload.draft || "(пустой черновик)";
                turn.answer.before(draft);
                const line = el("div", "trace-line nudge");
                line.append(el("span", "", `↻ раунд ${payload.round} прошёл без вызовов — черновик отброшен, модели дан толчок`));
                addTrace(turn, line);
                turn.text = "";
                turn.answer.textContent = "";
                continue;
            }
            if (event === "reasoning") {
                turn.reasoningText += payload;
                turn.reasoning.hidden = false;
                turn.reasoning.querySelector("pre").textContent = turn.reasoningText.trim();
                continue;
            }
            if (event === "done") {
                turn.text = payload.text;
                turn.answer.textContent = payload.text || "(модель не дала текста)";
                assistantTurn.content = payload.text;
                turn.meta.textContent = formatDone(payload);
                await loadFiles();
                showDiff(payload.diff, payload.changed);
                scrollMessages();
                return;
            }
            turn.text += payload;
            turn.answer.textContent = turn.text;
            scrollMessages();
        }
    }
}

function renderChatMeta() {
    clearButton.hidden = history.length === 0;
    chatMeta.textContent = `ходов: ${Math.floor(history.length / 2)}`;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (controller) {
        controller.abort();
        return;
    }
    const question = questionInput.value.trim();
    if (!question) {
        setStatus("Введите цель", true);
        return;
    }

    controller = new AbortController();
    setBusy(true);
    setStatus("Модель работает с файлами...");
    questionInput.value = "";
    const turnNode = () => messagesEl.lastElementChild;

    try {
        await ask(question, selectedProvider());
        setStatus("Готово");
    } catch (error) {
        const node = turnNode();
        const meta = node && node.querySelector(".turn-meta");
        if (error.name === "AbortError") {
            if (meta) meta.textContent = "остановлено";
            setStatus("Остановлено");
        } else {
            const answer = node && node.querySelector(".answer:not(.draft)");
            if (answer) {
                answer.textContent = error.message;
                answer.classList.add("error");
            }
            if (meta) {
                meta.textContent = "ошибка";
                meta.classList.add("error");
            }
            setStatus("Ошибка", true);
        }
    } finally {
        controller = null;
        setBusy(false);
    }
});

clearButton.addEventListener("click", () => {
    history = [];
    renderChat();
});

resetButton.addEventListener("click", async () => {
    if (controller) return;
    const response = await fetch("/api/reset", { method: "POST" });
    if (!response.ok) {
        setStatus("Сброс не удался", true);
        return;
    }
    selectedPath = null;
    latestDiff = "";
    changedFiles = [];
    previewEmpty.hidden = false;
    previewBody.hidden = true;
    previewTitle.textContent = "Просмотр";
    previewMeta.textContent = "";
    setMode("file");
    await loadFiles();
    setStatus("Песочница сброшена к эталону");
});

tabFile.addEventListener("click", () => {
    if (selectedPath) showFile(selectedPath);
    else {
        setMode("file");
        previewEmpty.hidden = false;
        previewBody.hidden = true;
        previewTitle.textContent = "Просмотр";
        previewMeta.textContent = "";
    }
});

tabDiff.addEventListener("click", async () => {
    if (latestDiff) {
        showDiff(latestDiff, changedFiles);
        return;
    }
    const response = await fetch("/api/diff");
    const payload = await response.json();
    showDiff(payload.diff, payload.changed);
});

questionInput.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        form.requestSubmit();
    }
});

recheckButton.addEventListener("click", checkStatus);

checkStatus();
loadFiles();
renderChat();
