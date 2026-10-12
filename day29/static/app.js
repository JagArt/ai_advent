const form = document.getElementById("form");
const questionInput = document.getElementById("question");
const optimizedInput = document.getElementById("optimized");
const thinkInput = document.getElementById("think");
const thinkLabel = document.getElementById("thinkLabel");
const submitButton = document.getElementById("submit");
const statusEl = document.getElementById("status");
const profileLine = document.getElementById("profileLine");
const presetsEl = document.getElementById("presets");
const recheckButton = document.getElementById("recheck");
const expectedEl = document.getElementById("expected");
const sourcesWrap = document.getElementById("sourcesWrap");
const sourcesEl = document.getElementById("sources");
const sourcesMeta = document.getElementById("sourcesMeta");

const dots = {
    index: [document.getElementById("indexDot"), document.getElementById("indexText")],
    local: [document.getElementById("localDot"), document.getElementById("localText")],
    model: [document.getElementById("modelDot"), document.getElementById("modelText")],
};

const columns = {};
for (const column of document.querySelectorAll(".column")) {
    columns[column.dataset.profile] = {
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
let profiles = {};
let contexts = {};

function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", isError);
}

function setBusy(busy) {
    submitButton.textContent = busy ? "Остановить" : submitLabel;
    submitButton.classList.toggle("stop", busy);
}

function selectedMode() {
    return document.querySelector('input[name="mode"]:checked').value;
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

function describeProfiles() {
    const optimized = profiles.optimized;
    const base = profiles.base;
    if (!optimized || !base) {
        profileLine.textContent = "";
        return;
    }
    const active = optimizedInput.checked ? optimized : base;
    profileLine.textContent = `${optimizedInput.checked ? "оптимизация включена" : "оптимизация выключена"}: ${active.summary}`;
    const allowThink = base.think_allowed && !optimizedInput.checked;
    thinkInput.disabled = !allowThink;
    thinkLabel.classList.toggle("disabled", !allowThink);
    thinkLabel.title = allowThink ? "" : "оптимизированный профиль всегда добавляет /no_think";
}

async function checkStatus() {
    setDot("index", "", "Индекс...");
    setDot("local", "", "LM Studio...");
    setDot("model", "", "Модель...");
    try {
        const response = await fetch("/api/status");
        const status = await response.json();
        profiles = status.profiles || {};
        describeProfiles();

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
            setDot("model", "off", "—");
        } else if (!status.local.model_available) {
            setDot("local", "on", status.local.base_url);
            setDot("model", "off", `модели ${status.local.model} нет в списке: ${status.local.models.join(", ")}`);
        } else {
            setDot("local", "on", `${status.local.base_url} · ${status.local.model}`);
            const info = status.local.info;
            if (!info) {
                setDot("model", "", status.local.info_error || "сведений о модели нет");
            } else {
                const quant = info.quantization ? `${info.quantization}${info.bits_per_weight ? ` · ${info.bits_per_weight} бит/вес` : ""}` : "квантование неизвестно";
                const size = info.size_gb ? ` · ${info.size_gb} ГБ` : "";
                const cache = info.parallel !== null ? ` · KV-кэш: ${info.parallel} слот., ${info.context_checkpoints} снимков` : "";
                const loaded = info.loaded
                    ? `загружена, контекст ${info.loaded_context_length} из ${info.max_context_length}${cache}`
                    : `не загружена — загрузится первым запросом с окном по умолчанию (максимум ${info.max_context_length})`;
                setDot("model", info.loaded ? "on" : "", `${quant}${size} · ${loaded}`);
            }
        }
        for (const column of Object.values(columns)) {
            column.model.textContent = status.local.model;
        }
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

function renderSources() {
    const shown = Object.values(contexts);
    if (!shown.length) return;
    sourcesWrap.hidden = false;

    const full = contexts.base || contexts.optimized;
    const fitted = contexts.optimized;
    const metaParts = [`${full.retriever} · поиск ${Math.round(full.seconds * 1000)} мс · файлы: ${full.paths.join(", ")}`];
    if (contexts.base) metaParts.push(`базовая: ${contexts.base.sources.length} выдержек, ${contexts.base.tokens} ток. целиком`);
    if (fitted) {
        const cut = fitted.trimmed ? `, обрезано ${fitted.trimmed}` : "";
        const dropped = fitted.dropped ? `, отброшено ${fitted.dropped}` : "";
        metaParts.push(`оптимизированная: ${fitted.sources.length} выдержек, ${fitted.tokens} ток. при бюджете ${fitted.budget}${cut}${dropped}`);
    }
    sourcesMeta.textContent = metaParts.join(" · ");

    const byNumber = new Map();
    for (const context of [contexts.base, contexts.optimized]) {
        if (!context) continue;
        for (const source of context.sources) {
            if (!byNumber.has(source.number)) byNumber.set(source.number, { ...source, fitted: null });
            if (context === fitted) byNumber.get(source.number).fitted = source;
        }
    }

    sourcesEl.innerHTML = "";
    for (const source of [...byNumber.values()].sort((a, b) => a.number - b.number)) {
        const li = document.createElement("li");
        li.innerHTML = `<div class="source-head"><span class="source-path"></span><span class="source-score"></span></div><div class="source-snippet"></div>`;
        li.querySelector(".source-path").textContent = `[${source.number}] ${source.path} · ${source.section}`;
        let tokens = `${source.tokens} ток.`;
        if (fitted) {
            if (!source.fitted) tokens += " → отброшена";
            else if (source.fitted.trimmed) tokens += ` → ${source.fitted.tokens} ток. (обрезана)`;
        }
        li.querySelector(".source-score").textContent = `bm25 ${source.score} · ${tokens}`;
        li.querySelector(".source-snippet").textContent = source.snippet;
        sourcesEl.append(li);
    }
}

async function ask(profileName, question, think) {
    const column = columns[profileName];
    const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, optimized: profileName === "optimized", think }),
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
                contexts[profileName] = payload;
                renderSources();
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

async function askColumn(profileName, question, think) {
    const column = columns[profileName];
    column.meta.textContent = "ждёт...";
    try {
        await ask(profileName, question, think);
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

    const active = optimizedInput.checked ? "optimized" : "base";
    const order = selectedMode() === "compare" ? ["base", "optimized"] : [active];

    for (const [name, column] of Object.entries(columns)) {
        column.root.hidden = !order.includes(name);
        resetColumn(column);
    }
    contexts = {};
    sourcesWrap.hidden = true;
    expectedEl.textContent = expectedFor(question);

    controller = new AbortController();
    setBusy(true);
    setStatus(order.length > 1 ? "Сначала базовая, затем оптимизированная..." : "Поиск и генерация...");

    try {
        for (const name of order) {
            if (controller.signal.aborted) break;
            await askColumn(name, question, thinkInput.checked && !thinkInput.disabled);
        }
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

optimizedInput.addEventListener("change", describeProfiles);
recheckButton.addEventListener("click", checkStatus);

checkStatus();
loadQuestions();
