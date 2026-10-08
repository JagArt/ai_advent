const form = document.getElementById("form");
const questionInput = document.getElementById("question");
const thinkInput = document.getElementById("think");
const submitButton = document.getElementById("submit");
const statusEl = document.getElementById("status");
const presetsEl = document.getElementById("presets");
const recheckButton = document.getElementById("recheck");
const expectedEl = document.getElementById("expected");
const sourcesWrap = document.getElementById("sourcesWrap");
const sourcesEl = document.getElementById("sources");
const sourcesMeta = document.getElementById("sourcesMeta");

const dots = {
    index: [document.getElementById("indexDot"), document.getElementById("indexText")],
    local: [document.getElementById("localDot"), document.getElementById("localText")],
    cloud: [document.getElementById("cloudDot"), document.getElementById("cloudText")],
};

const columns = {};
for (const column of document.querySelectorAll(".column")) {
    columns[column.dataset.provider] = {
        root: column,
        model: column.querySelector(".model"),
        meta: column.querySelector(".meta"),
        reasoningWrap: column.querySelector(".reasoning"),
        reasoning: column.querySelector(".reasoning-text"),
        result: column.querySelector(".result:not(.reasoning-text)"),
        answerText: "",
        reasoningText: "",
    };
}

const submitLabel = submitButton.textContent;

let controller = null;
let questions = [];
let contextShown = false;

function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", isError);
}

function setBusy(busy) {
    submitButton.textContent = busy ? "Остановить" : submitLabel;
    submitButton.classList.toggle("stop", busy);
}

function selectedProvider() {
    return document.querySelector('input[name="provider"]:checked').value;
}

function stickToBottom(el, update) {
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    update();
    if (atBottom) {
        el.scrollTop = el.scrollHeight;
    }
}

function resetColumn(column) {
    column.answerText = "";
    column.reasoningText = "";
    column.result.textContent = "";
    column.result.classList.remove("error");
    column.reasoning.textContent = "";
    column.reasoningWrap.hidden = true;
    column.meta.textContent = "";
    column.meta.classList.remove("error");
}

function appendAnswer(column, text) {
    column.answerText += text;
    stickToBottom(column.result, () => {
        column.result.textContent = column.answerText.replace(/^\s+/, "");
    });
}

function appendReasoning(column, text) {
    column.reasoningText += text;
    stickToBottom(column.reasoning, () => {
        column.reasoning.textContent = column.reasoningText.trim();
    });
    column.reasoningWrap.hidden = !column.reasoningText.trim();
}

function setDot(name, state, text) {
    const [dot, label] = dots[name];
    dot.className = `dot ${state}`;
    label.textContent = text;
}

async function checkStatus() {
    setDot("index", "", "Индекс...");
    setDot("local", "", "LM Studio...");
    setDot("cloud", "", "Облако...");
    try {
        const response = await fetch("/api/status");
        const status = await response.json();

        if (status.index.ready) {
            setDot(
                "index",
                "on",
                `индекс day25 · ${status.index.documents} документов, ${status.index.chunks} чанков · ${status.index.retriever}, top-${status.index.top_k}`,
            );
        } else {
            setDot("index", "off", status.index.error);
        }

        if (!status.local.online) {
            setDot("local", "off", status.local.error);
        } else if (!status.local.model_available) {
            setDot("local", "off", `${status.local.base_url} · модели ${status.local.model} нет в списке: ${status.local.models.join(", ")}`);
        } else {
            setDot("local", "on", `${status.local.base_url} · ${status.local.model}`);
        }
        columns.local.model.textContent = status.local.model;

        if (status.cloud.configured) {
            setDot("cloud", "on", `DeepSeek · ${status.cloud.model}`);
        } else {
            setDot("cloud", "off", "DEEPSEEK_API_KEY не задан — работает только локальный режим");
        }
        columns.cloud.model.textContent = status.cloud.model;
    } catch (error) {
        setDot("index", "off", `Не удалось проверить: ${error.message}`);
    }
}

async function loadQuestions() {
    const response = await fetch("/api/questions");
    questions = await response.json();
    for (const item of questions) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "preset";
        button.innerHTML = `<span class="preset-level"></span><span class="preset-text"></span>`;
        button.querySelector(".preset-level").textContent = `#${item.id} · ${item.kind}`;
        button.querySelector(".preset-text").textContent = item.question;
        button.addEventListener("click", () => {
            questionInput.value = item.question;
            form.requestSubmit();
        });
        presetsEl.append(button);
    }
}

function expectedFor(question) {
    const item = questions.find((entry) => entry.question === question);
    return item ? `Ожидается: ${item.expect}` : "";
}

function parseFrame(frame) {
    let event = "message";
    const dataLines = [];
    for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) {
            event = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
            dataLines.push(line.slice(5).trimStart());
        }
    }
    return { event, data: dataLines.join("\n") };
}

function formatDone(done) {
    const parts = [];
    if (done.ttft_ms !== null) parts.push(`первый токен ${done.ttft_ms} мс`);
    if (done.answer_ms !== null && done.answer_ms !== done.ttft_ms) parts.push(`ответ с ${done.answer_ms} мс`);
    parts.push(`всего ${(done.total_ms / 1000).toFixed(1)} с`);
    if (done.completion_tokens !== null) {
        const reasoning = done.reasoning_tokens ? `, из них рассуждения ${done.reasoning_tokens}` : "";
        parts.push(`токенов ${done.prompt_tokens} → ${done.completion_tokens}${reasoning}`);
    }
    if (done.tokens_per_sec !== null) parts.push(`${done.tokens_per_sec} ток/с`);
    if (done.refused) parts.push("отказ");
    if (done.finish_reason && done.finish_reason !== "stop") parts.push(`finish_reason: ${done.finish_reason}`);
    return parts.join(" · ");
}

function showContext(context) {
    if (contextShown) return;
    contextShown = true;
    sourcesWrap.hidden = false;
    sourcesMeta.textContent =
        `${context.retriever} · ${context.sources.length} выдержек, ${context.tokens} токенов · поиск ${Math.round(context.seconds * 1000)} мс · файлы: ${context.paths.join(", ")}`;
    sourcesEl.innerHTML = "";
    for (const source of context.sources) {
        const li = document.createElement("li");
        li.innerHTML = `<div class="source-head"><span class="source-path"></span><span class="source-score"></span></div><div class="source-snippet"></div>`;
        li.querySelector(".source-path").textContent = `[${source.number}] ${source.path} · ${source.section}`;
        li.querySelector(".source-score").textContent = `bm25 ${source.score} · ${source.tokens} ток.`;
        li.querySelector(".source-snippet").textContent = source.snippet;
        sourcesEl.append(li);
    }
}

async function ask(provider, question, think) {
    const column = columns[provider];
    const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, provider, think }),
        signal: controller.signal,
    });
    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }

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

            if (event === "error") {
                throw new Error(payload);
            }
            if (event === "context") {
                showContext(payload);
                continue;
            }
            if (event === "reasoning") {
                appendReasoning(column, payload);
                continue;
            }
            if (event === "done") {
                column.meta.textContent = formatDone(payload);
                return;
            }
            appendAnswer(column, payload);
        }
    }
}

async function askColumn(provider, question, think) {
    const column = columns[provider];
    try {
        await ask(provider, question, think);
    } catch (error) {
        if (error.name === "AbortError") {
            column.meta.textContent = "остановлено";
            return;
        }
        column.result.textContent = error.message;
        column.result.classList.add("error");
        column.meta.textContent = "ошибка";
        column.meta.classList.add("error");
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (controller) {
        controller.abort();
        return;
    }

    const question = questionInput.value.trim();
    if (!question) {
        setStatus("Введите вопрос", true);
        return;
    }

    const provider = selectedProvider();
    const providers = provider === "both" ? ["local", "cloud"] : [provider];

    for (const [name, column] of Object.entries(columns)) {
        column.root.hidden = !providers.includes(name);
        resetColumn(column);
    }
    contextShown = false;
    sourcesWrap.hidden = true;
    expectedEl.textContent = expectedFor(question);

    controller = new AbortController();
    setBusy(true);
    setStatus("Поиск и генерация...");

    try {
        await Promise.all(providers.map((name) => askColumn(name, question, thinkInput.checked)));
        setStatus(controller.signal.aborted ? "Остановлено" : "Готово");
    } finally {
        controller = null;
        setBusy(false);
    }
});

questionInput.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        form.requestSubmit();
    }
});

recheckButton.addEventListener("click", checkStatus);

checkStatus();
loadQuestions();
