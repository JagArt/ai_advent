const form = document.getElementById("form");
const promptInput = document.getElementById("prompt");
const submitButton = document.getElementById("submit");
const resetButton = document.getElementById("reset");
const statusEl = document.getElementById("status");
const paramsEl = document.getElementById("params");
const chatEl = document.getElementById("chat");
const placeholderEl = document.getElementById("placeholder");

const submitLabel = submitButton.textContent;
const SESSION_KEY = "day6.session";

// Параметры агента и стартовый запрос приходят из GET /api/defaults.
let config = null;
let sessionId = null;
let historySize = 0;
let memoryParam = null;
let controller = null;

function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", isError);
}

// Пока идёт стрим, та же кнопка работает на остановку.
function setBusy(busy) {
    submitButton.textContent = busy ? "Остановить" : submitLabel;
    submitButton.classList.toggle("stop", busy);
    resetButton.disabled = busy;
    promptInput.readOnly = busy;
}

function addParam(text) {
    const param = document.createElement("span");
    param.className = "param";
    param.textContent = text;
    paramsEl.append(param);
    return param;
}

function setHistorySize(size) {
    historySize = size;
    memoryParam.textContent = `в памяти ${size} из ${config.historyLimit} сообщений`;
}

function isNearBottom(el, threshold = 64) {
    return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
}

function scrollToBottom() {
    chatEl.scrollTop = chatEl.scrollHeight;
}

function addMessage(role, text) {
    const follow = isNearBottom(chatEl);
    placeholderEl.hidden = true;

    const message = document.createElement("article");
    message.className = `message message-${role}`;

    const author = document.createElement("span");
    author.className = "message-author";
    author.textContent = role === "user" ? "Вы" : "Агент";

    const body = document.createElement("pre");
    body.className = "message-body";
    body.textContent = text;

    message.append(author, body);
    chatEl.append(message);

    if (follow) {
        scrollToBottom();
    }
    return { message, body, thinking: null };
}

// Пузырь агента появляется пустым сразу после отправки, поэтому до первого токена
// в нём живут анимированные точки: видно, что запрос ушёл и ответ готовится.
function showThinking(bubble) {
    const thinking = document.createElement("span");
    thinking.className = "thinking";
    thinking.append(...Array.from({ length: 3 }, () => document.createElement("span")));
    bubble.body.append(thinking);
    bubble.thinking = thinking;
}

function stopThinking(bubble) {
    if (!bubble.thinking) {
        return false;
    }
    bubble.thinking.remove();
    bubble.thinking = null;
    return true;
}

function appendChunk(bubble, text) {
    const follow = isNearBottom(chatEl);
    bubble.body.textContent += text;
    if (follow) {
        scrollToBottom();
    }
}

function clearChat() {
    for (const message of [...chatEl.querySelectorAll(".message")]) {
        message.remove();
    }
    placeholderEl.hidden = false;
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

async function ask(prompt, bubble) {
    const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, prompt }),
        signal: controller.signal,
    });

    if (response.status === 404) {
        throw new Error("Сессия истекла");
    }
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
                const payload = JSON.parse(data);
                stopThinking(bubble);
                setHistorySize(payload.history_size);
                return payload;
            }

            const text = JSON.parse(data);
            if (stopThinking(bubble)) {
                setStatus("Агент отвечает...");
            }
            appendChunk(bubble, text);
        }
    }

    throw new Error("Поток оборвался");
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (controller) {
        controller.abort();
        return;
    }

    const prompt = promptInput.value.trim();
    if (!prompt) {
        setStatus("Введите сообщение", true);
        return;
    }

    const question = addMessage("user", prompt);
    const answer = addMessage("agent", "");
    showThinking(answer);
    promptInput.value = "";
    controller = new AbortController();
    setBusy(true);
    setStatus("Агент думает...");

    try {
        const payload = await ask(prompt, answer);
        setStatus(payload.finish_reason === "length" ? "Ответ обрезан по лимиту" : "Готово");
    } catch (error) {
        // Оборванный ход агент в память не записал — убираем его и из ленты,
        // чтобы на экране не осталось того, чего собеседник не помнит.
        question.message.remove();
        answer.message.remove();
        placeholderEl.hidden = chatEl.querySelector(".message") !== null;
        promptInput.value = prompt;

        if (error.name === "AbortError") {
            setStatus("Остановлено, ход не сохранён в памяти");
        } else if (error.message === "Сессия истекла") {
            await startSession();
            promptInput.value = prompt;
            setStatus("Сессия истекла, начат новый диалог", true);
        } else {
            setStatus(error.message, true);
        }
    } finally {
        controller = null;
        setBusy(false);
        promptInput.focus();
    }
});

promptInput.addEventListener("keydown", (event) => {
    // Shift + Enter — перенос строки. event.isComposing отсекает Enter, которым
    // подтверждают подсказку IME: это ввод слова, а не отправка сообщения.
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) {
        return;
    }
    event.preventDefault();
    // Пока идёт ответ, та же кнопка означает «Остановить» — прерывать диалог
    // случайным Enter не стоит, остановка остаётся осознанным кликом.
    if (!controller) {
        form.requestSubmit();
    }
});

resetButton.addEventListener("click", async () => {
    try {
        const response = await fetch(`/api/session/${sessionId}/reset`, { method: "POST" });
        if (!response.ok) {
            throw new Error(`Сервер вернул ${response.status}`);
        }
        const payload = await response.json();
        clearChat();
        setHistorySize(payload.history_size);
        promptInput.value = config.prompt;
        setStatus("Память очищена");
    } catch (error) {
        setStatus(error.message, true);
    }
    promptInput.focus();
});

async function startSession() {
    const response = await fetch("/api/session", { method: "POST" });
    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }

    const payload = await response.json();
    sessionId = payload.session_id;
    localStorage.setItem(SESSION_KEY, sessionId);
    clearChat();
    setHistorySize(0);
    promptInput.value = config.prompt;
}

// Диалог хранится на сервере, поэтому после перезагрузки страницы лента
// восстанавливается из памяти агента, а не из браузера.
async function restoreSession(saved) {
    const response = await fetch(`/api/session/${saved}`);
    if (!response.ok) {
        return false;
    }

    const payload = await response.json();
    sessionId = saved;
    clearChat();
    for (const message of payload.history) {
        addMessage(message.role === "user" ? "user" : "agent", message.content);
    }
    setHistorySize(payload.history_size);
    promptInput.value = payload.history.length ? "" : config.prompt;
    scrollToBottom();
    return true;
}

async function init() {
    const response = await fetch("/api/defaults");
    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }

    const payload = await response.json();
    config = {
        prompt: payload.prompt,
        historyLimit: payload.history_limit,
    };

    // Параметры агента постоянны для всего диалога, меняется только счётчик памяти.
    addParam(payload.model);
    addParam(`temperature ${payload.temperature}`);
    addParam(`лимит ответа ${payload.max_tokens} токенов`);
    memoryParam = addParam("");

    const saved = localStorage.getItem(SESSION_KEY);
    if (!saved || !(await restoreSession(saved))) {
        await startSession();
    }

    for (const control of [submitButton, resetButton]) {
        control.disabled = false;
    }
}

init().catch((error) => {
    setStatus(error.message, true);
});
