const form = document.getElementById("form");
const promptInput = document.getElementById("prompt");
const submitButton = document.getElementById("submit");
const statusEl = document.getElementById("status");
const rangeEl = document.getElementById("tempRange");
const numberEl = document.getElementById("tempNumber");
const bandEl = document.getElementById("tempBand");
const resultEl = document.getElementById("result");
const copyButton = document.getElementById("copy");
const metaEl = document.getElementById("meta");

// Шкала температуры целиком приходит из GET /api/defaults, своих значений страница не держит.
let config = null;
let temperature = null;
let answerText = "";

function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", isError);
}

function setMeta(text, isError = false) {
    metaEl.textContent = text;
    metaEl.classList.toggle("error", isError);
}

function isNearBottom(el, threshold = 48) {
    return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
}

function tempDecimals() {
    const text = String(config.tempStep);
    const dot = text.indexOf(".");
    return dot === -1 ? 0 : text.length - dot - 1;
}

function formatTemp(value) {
    return value.toFixed(tempDecimals());
}

function normalizeTemp(raw, fallback) {
    const parsed = typeof raw === "number" ? raw : Number.parseFloat(raw);
    if (!Number.isFinite(parsed)) {
        return fallback;
    }
    const clamped = Math.min(config.tempMax, Math.max(config.tempMin, parsed));
    const snapped = Math.round(clamped / config.tempStep) * config.tempStep;
    return Number(snapped.toFixed(tempDecimals()));
}

function bandFor(value) {
    const band = config.bands.find((item) => value <= item.max + 1e-9);
    return band ? band.label : "";
}

function setTemperature(raw) {
    temperature = normalizeTemp(raw, temperature);
    rangeEl.value = String(temperature);
    numberEl.value = formatTemp(temperature);
    bandEl.textContent = bandFor(temperature);
}

function setPlaceholder() {
    answerText = "";
    resultEl.replaceChildren();
    const span = document.createElement("span");
    span.className = "placeholder";
    span.textContent = "Здесь появится ответ";
    resultEl.append(span);
    copyButton.hidden = true;
}

function setResult(text) {
    const follow = isNearBottom(resultEl);
    answerText = text;
    resultEl.textContent = text;
    copyButton.hidden = !text;
    if (follow) {
        resultEl.scrollTop = resultEl.scrollHeight;
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

function applyDone(payload) {
    setMeta(`temperature: ${formatTemp(temperature)} · слов: ${payload.word_count}`);
}

async function ask(prompt) {
    const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, temperature }),
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

            if (event === "error") {
                throw new Error(JSON.parse(data));
            }
            if (event === "done") {
                applyDone(JSON.parse(data));
                return;
            }
            setResult(answerText + JSON.parse(data));
        }
    }
}

rangeEl.addEventListener("input", () => {
    setTemperature(rangeEl.value);
});

// Пока пользователь печатает, поле не переписываем — синхронизируем только слайдер и подпись.
numberEl.addEventListener("input", () => {
    temperature = normalizeTemp(numberEl.value, temperature);
    rangeEl.value = String(temperature);
    bandEl.textContent = bandFor(temperature);
});

numberEl.addEventListener("change", () => {
    setTemperature(numberEl.value);
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const prompt = promptInput.value.trim();
    if (!prompt) {
        setStatus("Введите запрос", true);
        return;
    }

    submitButton.disabled = true;
    setStatus("Генерация ответа...");
    setMeta("");
    setPlaceholder();

    try {
        await ask(prompt);
        setStatus("Готово");
    } catch (error) {
        setStatus(error.message, true);
    } finally {
        submitButton.disabled = false;
    }
});

promptInput.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        form.requestSubmit();
    }
});

copyButton.addEventListener("click", async () => {
    await navigator.clipboard.writeText(answerText);
    copyButton.textContent = "Скопировано";
    setTimeout(() => {
        copyButton.textContent = "Копировать";
    }, 1500);
});

async function loadDefaults() {
    const response = await fetch("/api/defaults");
    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }

    const payload = await response.json();
    config = {
        tempMin: payload.temp_min,
        tempMax: payload.temp_max,
        tempStep: payload.temp_step,
        bands: payload.bands,
    };

    promptInput.value = payload.prompt;

    for (const input of [rangeEl, numberEl]) {
        input.min = String(config.tempMin);
        input.max = String(config.tempMax);
        input.step = String(config.tempStep);
    }

    setTemperature(payload.temperature);

    for (const control of [rangeEl, numberEl, submitButton]) {
        control.disabled = false;
    }
}

loadDefaults().catch((error) => {
    setStatus(error.message, true);
});
