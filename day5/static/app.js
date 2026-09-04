const form = document.getElementById("form");
const promptInput = document.getElementById("prompt");
const submitButton = document.getElementById("submit");
const statusEl = document.getElementById("status");
const compareEl = document.getElementById("compare");

const submitLabel = submitButton.textContent;

// Список моделей целиком приходит из GET /api/models, своих значений страница не держит.
let columns = new Map();
let maxTokens = null;
let controller = null;

function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", isError);
}

// Пока идёт стрим, та же кнопка работает на остановку.
function setBusy(busy) {
    submitButton.textContent = busy ? "Остановить" : submitLabel;
    submitButton.classList.toggle("stop", busy);
}

function isNearBottom(el, threshold = 48) {
    return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
}

function formatMs(ms) {
    if (ms === null || ms === undefined) return "—";
    return ms >= 1000 ? `${(ms / 1000).toFixed(2)} с` : `${ms} мс`;
}

function formatCost(cost, paid) {
    if (cost === null || cost === undefined) return "—";
    if (!paid && cost === 0) return "бесплатно";
    return `$${cost.toFixed(6)} (за 1000 запросов $${(cost * 1000).toFixed(3)})`;
}

function link(href, text) {
    const el = document.createElement("a");
    el.href = href;
    el.target = "_blank";
    el.rel = "noopener";
    el.textContent = text;
    return el;
}

function buildColumn(spec) {
    const wrap = document.createElement("article");
    wrap.className = "result-wrap";

    const head = document.createElement("div");
    head.className = "result-head";

    const info = document.createElement("div");
    const title = document.createElement("span");
    title.className = "result-title";
    title.textContent = spec.tier;
    const name = document.createElement("span");
    name.className = "model-name";
    name.textContent = spec.model;
    const links = document.createElement("div");
    links.className = "model-links";
    links.append(link(spec.link, spec.provider));
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = spec.note;
    info.append(title, name, links, meta);

    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "copy";
    copy.textContent = "Копировать";
    copy.hidden = true;

    head.append(info, copy);

    const metrics = document.createElement("dl");
    metrics.className = "metrics";

    const result = document.createElement("pre");
    result.className = "result";

    wrap.append(head, metrics, result);
    compareEl.append(wrap);

    const column = { spec, meta, metrics, result, copy, text: "", reasoningText: "" };

    copy.addEventListener("click", async () => {
        await navigator.clipboard.writeText(column.text);
        copy.textContent = "Скопировано";
        setTimeout(() => {
            copy.textContent = "Копировать";
        }, 1500);
    });

    resetColumn(column, "Здесь появится ответ");
    return column;
}

function resetColumn(column, placeholder) {
    column.text = "";
    column.reasoningText = "";
    column.metrics.replaceChildren();
    column.result.classList.remove("error");
    column.result.replaceChildren();
    const span = document.createElement("span");
    span.className = "placeholder";
    span.textContent = placeholder;
    column.result.append(span);
    column.copy.hidden = true;
    column.meta.textContent = column.spec.note;
    column.meta.classList.remove("error");
}

function appendAnswer(column, text) {
    const follow = isNearBottom(column.result);
    if (!column.text) {
        column.result.replaceChildren();
    }
    column.text += text;
    column.result.textContent = column.text;
    column.copy.hidden = false;
    if (follow) {
        column.result.scrollTop = column.result.scrollHeight;
    }
}

function setMetrics(column, payload) {
    const rows = [
        ["первый токен", formatMs(payload.ttft_ms)],
    ];

    // У модели с обязательным рассуждением ответ начинается позже первого токена.
    if (payload.reasoning_tokens > 0) {
        rows.push(["первый токен ответа", formatMs(payload.answer_ttft_ms)]);
    }

    rows.push(
        ["всего", formatMs(payload.total_ms)],
        ["токены", `${payload.prompt_tokens} вход · ${payload.completion_tokens} ответ`],
    );

    if (payload.reasoning_tokens > 0) {
        rows.push(["из них рассуждение", `${payload.reasoning_tokens} токенов`]);
    }

    rows.push(
        ["скорость", payload.output_tps === null ? "—" : `${payload.output_tps} ток/с`],
        ["стоимость", formatCost(payload.cost_usd, column.spec.paid)],
    );

    column.metrics.replaceChildren();
    for (const [label, value] of rows) {
        const dt = document.createElement("dt");
        dt.textContent = label;
        const dd = document.createElement("dd");
        dd.textContent = value;
        column.metrics.append(dt, dd);
    }
}

function notice(column, text) {
    const span = document.createElement("span");
    span.className = "notice";
    span.textContent = text;
    column.result.append(span);
}

// Слабая модель может уйти в рассуждение целиком и не начать ответ: в этом случае
// колонка не должна оставаться с плейсхолдером, причину надо назвать.
function applyOutcome(column, payload) {
    const truncated = payload.finish_reason === "length";

    if (column.text) {
        if (truncated) {
            notice(column, `\n\n[ответ обрезан: исчерпан лимит max_tokens = ${maxTokens}]`);
        }
        return;
    }

    column.result.replaceChildren();

    if (payload.reasoning_tokens > 0 && truncated) {
        notice(
            column,
            `Ответа нет: на рассуждение ушёл весь лимит max_tokens = ${maxTokens} `
                + `(${payload.reasoning_tokens} токенов), до самого ответа модель не дошла.`,
        );
    } else if (truncated) {
        notice(column, `Ответа нет: исчерпан лимит max_tokens = ${maxTokens}.`);
    } else {
        notice(column, "Модель закончила генерацию, но текста ответа не вернула.");
    }

    if (column.reasoningText) {
        const dump = document.createElement("span");
        dump.className = "reasoning";
        dump.textContent = `\n\n--- рассуждение модели ---\n${column.reasoningText}`;
        column.result.append(dump);
    }
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

function handleFrame(event, data) {
    if (event === "end") {
        return true;
    }

    const payload = JSON.parse(data);
    const column = columns.get(payload.model);
    if (!column) {
        return false;
    }

    if (event === "error") {
        column.result.classList.add("error");
        column.result.textContent = payload.message;
        column.meta.textContent = "ошибка";
        column.meta.classList.add("error");
        return false;
    }

    if (event === "done") {
        setMetrics(column, payload);
        column.meta.textContent = payload.finish_reason
            ? `finish_reason: ${payload.finish_reason}`
            : column.spec.note;
        applyOutcome(column, payload);
        return false;
    }

    if (payload.kind === "reasoning") {
        column.reasoningText += payload.text;
        // Текст рассуждения не показываем, пока ответ идёт своим ходом.
        if (!column.text) {
            column.meta.textContent = "рассуждает...";
        }
        return false;
    }

    appendAnswer(column, payload.text);
    return false;
}

async function ask(prompt) {
    const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
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
            if (handleFrame(event, data)) {
                return;
            }
        }
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (controller) {
        controller.abort();
        return;
    }

    const prompt = promptInput.value.trim();
    if (!prompt) {
        setStatus("Введите запрос", true);
        return;
    }

    controller = new AbortController();
    setBusy(true);
    setStatus("Три модели отвечают параллельно...");
    for (const column of columns.values()) {
        resetColumn(column, "Ожидание ответа");
    }

    try {
        await ask(prompt);
        setStatus("Готово");
    } catch (error) {
        if (error.name === "AbortError") {
            setStatus("Остановлено");
        } else {
            setStatus(error.message, true);
        }
    } finally {
        controller = null;
        setBusy(false);
    }
});

promptInput.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        form.requestSubmit();
    }
});

async function loadModels() {
    const response = await fetch("/api/models");
    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }

    const payload = await response.json();
    promptInput.value = payload.prompt;
    maxTokens = payload.max_tokens;

    compareEl.replaceChildren();
    columns = new Map(payload.models.map((spec) => [spec.key, buildColumn(spec)]));

    const rates = payload.peak ? "пиковые" : "непиковые";
    setStatus(`temperature ${payload.temperature} · max_tokens ${payload.max_tokens} · тарифы ${rates}`);
    submitButton.disabled = false;
}

loadModels().catch((error) => {
    setStatus(error.message, true);
});
