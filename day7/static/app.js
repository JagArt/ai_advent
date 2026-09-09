const form = document.getElementById("form");
const promptInput = document.getElementById("prompt");
const submitButton = document.getElementById("submit");
const newDialogButton = document.getElementById("new-dialog");
const dialogsEl = document.getElementById("dialogs");
const statusEl = document.getElementById("status");
const paramsEl = document.getElementById("params");
const chatEl = document.getElementById("chat");
const placeholderEl = document.getElementById("placeholder");

const submitLabel = submitButton.textContent;
const SESSION_KEY = "day7.session";

// Параметры агента и стартовый запрос приходят из GET /api/defaults.
let config = null;
let sessionId = null;
let historySize = 0;
let dialogs = [];
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
    newDialogButton.disabled = busy;
    // Панель гаснет вместе с кнопкой: уход в другой диалог посреди ответа увёл бы
    // текущий стрим в чужую ленту.
    dialogsEl.classList.toggle("busy", busy);
    promptInput.readOnly = busy;
}

function addParam(text) {
    const param = document.createElement("span");
    param.className = "param";
    param.textContent = text;
    paramsEl.append(param);
    return param;
}

function plural(count, one, few, many) {
    if (count % 100 >= 11 && count % 100 <= 14) {
        return many;
    }
    if (count % 10 === 1) {
        return one;
    }
    return count % 10 >= 2 && count % 10 <= 4 ? few : many;
}

// Два числа, а не одно: база хранит диалог целиком, модель видит только окно.
function setMemory(payload) {
    historySize = payload.history_size;
    const word = plural(historySize, "сообщение", "сообщения", "сообщений");
    memoryParam.textContent = `в диалоге ${historySize} ${word}, в контексте ${payload.context_size} из ${config.historyLimit}`;
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
                setMemory(payload);
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
        refreshDialogs();
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
            await createSession();
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

newDialogButton.addEventListener("click", async () => {
    // Открытый диалог пуст — вторая такая же сессия только замусорит панель.
    if (historySize === 0) {
        setStatus("Этот диалог ещё пуст");
        promptInput.focus();
        return;
    }

    try {
        await createSession();
        setStatus("Начат новый диалог");
    } catch (error) {
        setStatus(error.message, true);
    }
    promptInput.focus();
});

function renderDialogs() {
    dialogsEl.replaceChildren(...dialogs.map(dialogItem));
}

function dialogItem(dialog) {
    const item = document.createElement("li");
    item.className = "dialog";
    item.classList.toggle("active", dialog.id === sessionId);

    const open = document.createElement("button");
    open.type = "button";
    open.className = "dialog-open";
    // Заголовок сессии — суть первого запроса, у пустой его ещё нет.
    const title = dialog.title || "Новый диалог";
    open.title = title;
    if (dialog.id === sessionId) {
        open.setAttribute("aria-current", "true");
    }

    const name = document.createElement("span");
    name.className = "dialog-name";
    name.textContent = title;

    const size = document.createElement("span");
    size.className = "dialog-size";
    size.textContent = dialog.size;

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "dialog-delete";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Удалить диалог «${title}»`);
    remove.title = "Удалить диалог";

    open.append(name, size);
    open.addEventListener("click", () => switchSession(dialog.id));
    remove.addEventListener("click", () => deleteDialog(dialog.id));
    item.append(open, remove);
    return item;
}

async function loadDialogs() {
    const response = await fetch("/api/sessions");
    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }

    dialogs = (await response.json()).sessions;
    renderDialogs();
}

// Список обновляется после каждого хода: после первого появляется заголовок,
// а диалог поднимается наверх. Панель — не лента, сбой обновления не стоит ошибки.
function refreshDialogs() {
    loadDialogs().catch(() => {});
}

async function createSession() {
    const response = await fetch("/api/session", { method: "POST" });
    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }

    const payload = await response.json();
    sessionId = payload.session_id;
    localStorage.setItem(SESSION_KEY, sessionId);
    clearChat();
    setMemory({ history_size: 0, context_size: 0 });
    promptInput.value = config.prompt;
    await loadDialogs();
}

// Диалог хранится в базе, поэтому лента восстанавливается запросом истории —
// и при переключении, и после перезагрузки страницы, и после перезапуска сервера.
async function openSession(id) {
    const response = await fetch(`/api/session/${id}`);
    if (!response.ok) {
        return false;
    }

    const payload = await response.json();
    sessionId = id;
    localStorage.setItem(SESSION_KEY, id);
    clearChat();
    for (const message of payload.history) {
        addMessage(message.role === "user" ? "user" : "agent", message.content);
    }
    setMemory(payload);
    promptInput.value = payload.history.length ? "" : config.prompt;
    scrollToBottom();
    renderDialogs();
    return true;
}

async function switchSession(id) {
    if (controller || id === sessionId) {
        promptInput.focus();
        return;
    }

    try {
        if (!(await openSession(id))) {
            // Диалог удалили из другой вкладки — панель показывает то, чего нет.
            await loadDialogs();
            throw new Error("Диалог не найден");
        }
        setStatus("");
    } catch (error) {
        setStatus(error.message, true);
    }
    promptInput.focus();
}

async function deleteDialog(id) {
    if (controller) {
        return;
    }

    try {
        const response = await fetch(`/api/session/${id}`, { method: "DELETE" });
        // 404 — диалога уже нет, цель достигнута без нас.
        if (!response.ok && response.status !== 404) {
            throw new Error(`Сервер вернул ${response.status}`);
        }

        await loadDialogs();
        if (id === sessionId) {
            // Удалён открытый диалог: показываем соседний, а если список опустел —
            // заводим новый, странице всегда нужна живая сессия.
            await openDialogsHead();
        }
        setStatus("Диалог удалён");
    } catch (error) {
        setStatus(error.message, true);
    }
    promptInput.focus();
}

async function openDialogsHead() {
    if (!dialogs.length || !(await openSession(dialogs[0].id))) {
        await createSession();
    }
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

    await loadDialogs();

    // Вкладка помнит последний открытый диалог, но переписка принадлежит базе:
    // если сессии там уже нет, страница поднимает самую свежую из списка.
    const saved = localStorage.getItem(SESSION_KEY);
    if (!saved || !(await openSession(saved))) {
        await openDialogsHead();
    }

    for (const control of [submitButton, newDialogButton]) {
        control.disabled = false;
    }
}

init().catch((error) => {
    setStatus(error.message, true);
});
