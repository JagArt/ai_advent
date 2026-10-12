const form = document.getElementById("form");
const promptInput = document.getElementById("prompt");
const thinkInput = document.getElementById("think");
const submitButton = document.getElementById("submit");
const statusEl = document.getElementById("status");
const serviceDot = document.getElementById("serviceDot");
const serviceText = document.getElementById("serviceText");
const modelDot = document.getElementById("modelDot");
const modelText = document.getElementById("modelText");
const limitsText = document.getElementById("limitsText");
const recheckButton = document.getElementById("recheck");
const resetButton = document.getElementById("reset");
const thread = document.getElementById("thread");
const empty = document.getElementById("empty");

const submitLabel = submitButton.textContent;

let controller = null;
const messages = [];

function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", isError);
}

function setBusy(busy) {
    submitButton.textContent = busy ? "Остановить" : submitLabel;
    submitButton.classList.toggle("stop", busy);
    promptInput.disabled = busy;
    thinkInput.disabled = busy;
    resetButton.disabled = busy;
}

function stickToBottom(update) {
    const atBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 48;
    update();
    if (atBottom) {
        thread.scrollTop = thread.scrollHeight;
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

function formatLimits(done) {
    const parts = [];
    if (done.queue_wait_ms !== null && done.queue_wait_ms !== undefined) {
        parts.push(`очередь ${done.queue_wait_ms} мс`);
    }
    if (done.context_dropped) {
        parts.push(`срезано ходов ${done.context_dropped}`);
    }
    if (done.context_truncated) {
        parts.push("реплика укорочена");
    }
    if (done.context_tokens !== null && done.context_tokens !== undefined) {
        parts.push(`контекст ${done.context_tokens} ток.`);
    }
    if (done.prompt_tokens !== null && done.prompt_tokens !== undefined) {
        const completion = done.completion_tokens === null || done.completion_tokens === undefined
            ? "—"
            : done.completion_tokens;
        parts.push(`${done.prompt_tokens} → ${completion} ток.`);
    }
    if (done.ttft_ms !== null && done.ttft_ms !== undefined) {
        parts.push(`первый токен ${done.ttft_ms} мс`);
    }
    if (done.total_ms !== null && done.total_ms !== undefined) {
        parts.push(`всего ${(done.total_ms / 1000).toFixed(1)} с`);
    }
    if (done.rate_remaining !== null && done.rate_remaining !== undefined && done.rate_limit) {
        parts.push(`осталось ${done.rate_remaining}/${done.rate_limit}`);
    }
    return parts.join(" · ");
}

function addMessage(role, text) {
    empty.hidden = true;
    const article = document.createElement("article");
    article.className = `msg ${role}`;
    const who = document.createElement("span");
    who.className = "who";
    who.textContent = role === "user" ? "вы" : "сервис";
    const body = document.createElement("pre");
    body.className = "body";
    body.textContent = text;
    article.append(who, body);
    thread.append(article);
    stickToBottom(() => {});
    return article;
}

function ensureReasoning(article) {
    let block = article.querySelector(".reasoning");
    if (block) {
        return block.querySelector("pre");
    }
    block = document.createElement("details");
    block.className = "reasoning";
    block.open = true;
    const summary = document.createElement("summary");
    summary.textContent = "Рассуждения модели";
    const pre = document.createElement("pre");
    block.append(summary, pre);
    article.querySelector(".body").before(block);
    return pre;
}

async function checkServer() {
    serviceDot.className = "dot";
    modelDot.className = "dot";
    serviceText.textContent = "Проверка сервиса...";
    modelText.textContent = "Модель...";
    limitsText.textContent = "Лимиты...";
    try {
        const response = await fetch("/api/status");
        const status = await response.json();
        const service = status.service;
        serviceDot.classList.add("on");
        const keyNote = service.default_key ? " · ключ API по умолчанию, только localhost" : "";
        const from = service.client ? ` · вы с ${service.client}` : "";
        serviceText.textContent = `${location.origin}${from}${keyNote}`;
        modelDot.classList.add(service.model_available ? "on" : "off");
        modelText.textContent = service.model_available
            ? service.model
            : `${service.model} — ${service.upstream_error || "нет в списке LM Studio"}`;
        const left = service.rate_remaining === null || service.rate_remaining === undefined
            ? "—"
            : service.rate_remaining;
        const waiting = service.waiting ?? 0;
        limitsText.textContent =
            `слот ${service.slots}, в очереди ${waiting}/${service.queue_limit}`
            + ` · частота ${left}/${service.rate_limit} за ${service.rate_window_s} с для вашего IP`
            + ` · контекст ${service.max_context}`;
    } catch (error) {
        serviceDot.classList.add("off");
        modelDot.classList.add("off");
        serviceText.textContent = "Сервис не отвечает. Запустите `python day30/server.py`.";
        modelText.textContent = "Модель недоступна";
        limitsText.textContent = "Лимиты неизвестны, пока сервис не поднят";
    }
}

async function ask(article, body) {
    const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages, think: thinkInput.checked }),
        signal: controller.signal,
    });
    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let answer = "";
    let reasoning = "";

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
                const limits = article.querySelector(".limits") || document.createElement("span");
                limits.className = "limits";
                limits.textContent = formatLimits(JSON.parse(data));
                article.append(limits);
                return answer;
            }
            if (event === "reasoning") {
                reasoning += JSON.parse(data);
                ensureReasoning(article).textContent = reasoning.trim();
                stickToBottom(() => {});
                continue;
            }
            if (event === "meta") {
                continue;
            }
            const piece = JSON.parse(data);
            stickToBottom(() => {
                answer += piece;
                body.textContent = answer.replace(/^\s+/, "");
            });
        }
    }
    return answer;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (controller) {
        controller.abort();
        return;
    }

    const prompt = promptInput.value.trim();
    if (!prompt) {
        setStatus("Введите реплику", true);
        return;
    }

    messages.push({ role: "user", content: prompt });
    promptInput.value = "";
    addMessage("user", prompt);
    const article = addMessage("assistant", "");
    const body = article.querySelector(".body");
    body.textContent = "…";

    controller = new AbortController();
    setBusy(true);
    setStatus(thinkInput.checked ? "Модель думает..." : "Генерация ответа...");

    try {
        const answer = await ask(article, body);
        const text = (answer || "").trim();
        if (!text) {
            body.textContent = "Пустой ответ.";
        }
        messages.push({ role: "assistant", content: text || "Пустой ответ." });
        setStatus("Готово");
        checkServer();
    } catch (error) {
        const stopped = error.name === "AbortError";
        article.classList.add("error");
        body.textContent = stopped ? "Остановлено" : error.message;
        setStatus(stopped ? "Остановлено" : error.message, !stopped);
    } finally {
        controller = null;
        setBusy(false);
        promptInput.focus();
    }
});

promptInput.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        form.requestSubmit();
    }
});

resetButton.addEventListener("click", () => {
    messages.length = 0;
    thread.replaceChildren(empty);
    empty.hidden = false;
    setStatus("");
});

recheckButton.addEventListener("click", checkServer);
checkServer();
setInterval(checkServer, 8000);
