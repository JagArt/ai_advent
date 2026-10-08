const form = document.getElementById("form");
const promptInput = document.getElementById("prompt");
const thinkInput = document.getElementById("think");
const submitButton = document.getElementById("submit");
const statusEl = document.getElementById("status");
const presetsEl = document.getElementById("presets");
const serverDot = document.getElementById("serverDot");
const serverText = document.getElementById("serverText");
const recheckButton = document.getElementById("recheck");
const reasoningWrap = document.getElementById("reasoningWrap");
const reasoningEl = document.getElementById("reasoning");
const resultEl = document.getElementById("result");
const metaEl = document.getElementById("meta");
const copyButton = document.getElementById("copy");
const expectedEl = document.getElementById("expected");

const submitLabel = submitButton.textContent;

let controller = null;
let answerText = "";
let reasoningText = "";
let prompts = [];

function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", isError);
}

// Пока идёт стрим, та же кнопка работает на остановку.
function setBusy(busy) {
    submitButton.textContent = busy ? "Остановить" : submitLabel;
    submitButton.classList.toggle("stop", busy);
}

// Прокручиваем вниз, только если пользователь и так был внизу: иначе стрим отнимал бы прокрутку у читающего.
function stickToBottom(el, update) {
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    update();
    if (atBottom) {
        el.scrollTop = el.scrollHeight;
    }
}

function setAnswer(text) {
    answerText = text;
    stickToBottom(resultEl, () => {
        resultEl.textContent = text.replace(/^\s+/, "");
    });
    copyButton.hidden = !text.trim();
}

function setReasoning(text) {
    reasoningText = text;
    stickToBottom(reasoningEl, () => {
        reasoningEl.textContent = text.trim();
    });
    reasoningWrap.hidden = !text.trim();
}

async function checkServer() {
    serverDot.className = "dot";
    serverText.textContent = "Проверка сервера...";
    try {
        const response = await fetch("/api/status");
        const status = await response.json();
        if (!status.online) {
            serverDot.classList.add("off");
            serverText.textContent = status.error;
            return;
        }
        serverDot.classList.add(status.model_available ? "on" : "off");
        const note = status.model_available ? "" : " — модели нет в списке сервера";
        serverText.textContent =
            `${status.base_url} · модель ${status.model}${note} · доступно: ${status.models.join(", ")}`;
    } catch (error) {
        serverDot.classList.add("off");
        serverText.textContent = `Не удалось проверить сервер: ${error.message}`;
    }
}

async function loadPrompts() {
    const response = await fetch("/api/prompts");
    prompts = await response.json();
    for (const preset of prompts) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "preset";
        button.innerHTML = `<span class="preset-level"></span><span class="preset-text"></span>`;
        button.querySelector(".preset-level").textContent = preset.level;
        button.querySelector(".preset-text").textContent = preset.text;
        button.addEventListener("click", () => {
            promptInput.value = preset.text;
            form.requestSubmit();
        });
        presetsEl.append(button);
    }
}

function expectedFor(prompt) {
    const preset = prompts.find((item) => item.text === prompt);
    return preset ? `Ожидается: ${preset.expected}` : "";
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
    if (done.answer_ms !== null) parts.push(`ответ с ${done.answer_ms} мс`);
    parts.push(`всего ${(done.total_ms / 1000).toFixed(1)} с`);
    if (done.completion_tokens !== null) {
        const reasoning = done.reasoning_tokens ? `, из них рассуждения ${done.reasoning_tokens}` : "";
        parts.push(`токенов ${done.prompt_tokens} → ${done.completion_tokens}${reasoning}`);
    }
    if (done.tokens_per_sec !== null) parts.push(`${done.tokens_per_sec} ток/с`);
    if (done.finish_reason) parts.push(`finish_reason: ${done.finish_reason}`);
    return parts.join(" · ");
}

async function ask(prompt, think) {
    const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, think }),
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

            if (event === "error") {
                throw new Error(JSON.parse(data));
            }
            if (event === "done") {
                metaEl.textContent = formatDone(JSON.parse(data));
                return;
            }
            if (event === "reasoning") {
                setReasoning(reasoningText + JSON.parse(data));
                continue;
            }
            setAnswer(answerText + JSON.parse(data));
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
    setStatus(thinkInput.checked ? "Модель думает..." : "Генерация ответа...");
    setAnswer("");
    setReasoning("");
    metaEl.textContent = "";
    expectedEl.textContent = expectedFor(prompt);

    try {
        await ask(prompt, thinkInput.checked);
        setStatus("Готово");
    } catch (error) {
        setStatus(error.name === "AbortError" ? "Остановлено" : error.message, error.name !== "AbortError");
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

copyButton.addEventListener("click", async () => {
    await navigator.clipboard.writeText(answerText.trim());
    copyButton.textContent = "Скопировано";
    setTimeout(() => {
        copyButton.textContent = "Копировать";
    }, 1500);
});

recheckButton.addEventListener("click", checkServer);

checkServer();
loadPrompts();
