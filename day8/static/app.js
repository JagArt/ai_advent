const form = document.getElementById("form");
const promptInput = document.getElementById("prompt");
const submitButton = document.getElementById("submit");
const newDialogButton = document.getElementById("new-dialog");
const dialogsEl = document.getElementById("dialogs");
const statusEl = document.getElementById("status");
const paramsEl = document.getElementById("params");
const chatEl = document.getElementById("chat");
const placeholderEl = document.getElementById("placeholder");
const turnsEl = document.getElementById("turns");
const totalsEl = document.getElementById("totals");
const tokensNoteEl = document.getElementById("tokens-note");
const tokensEmptyEl = document.getElementById("tokens-empty");

const submitLabel = submitButton.textContent;
const SESSION_KEY = "day8.session";
const BUDGET_KEY = "day8.budget";

// Параметры агента, прайс и шкала бюджетов приходят из GET /api/defaults.
let config = null;
let sessionId = null;
let historySize = 0;
let dialogs = [];
let memoryParam = null;
let contextParam = null;
let costParam = null;
let budgetSelect = null;
let budget = null;
let pendingRow = null;
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
    // Бюджет — часть уходящего запроса, менять его посреди хода нечестно.
    budgetSelect.disabled = busy;
}

function addParam(text) {
    const param = document.createElement("span");
    param.className = "param";
    param.textContent = text;
    paramsEl.append(param);
    return param;
}

function addBudgetParam() {
    const label = document.createElement("label");
    label.className = "param param-control";
    label.textContent = "бюджет контекста ";

    const select = document.createElement("select");
    select.className = "budget";
    select.append(...config.budgetOptions.map((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = `${formatTokens(value)} ток.`;
        option.selected = value === budget;
        return option;
    }));

    label.append(select);
    paramsEl.append(label);
    return select;
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

function formatTokens(count) {
    return count.toLocaleString("ru-RU");
}

// Ход стоит десятитысячные доли цента, поэтому округление до центов бесполезно.
function formatCost(value) {
    return `$${value.toFixed(6)}`;
}

function formatDrift(estimated, actual) {
    if (!actual) {
        return "";
    }
    const drift = ((estimated - actual) / actual) * 100;
    const sign = drift > 0 ? "+" : "";
    return `${sign}${drift.toFixed(1)}%`;
}

// Три числа, а не одно: база хранит диалог целиком, модель видит только окно
// в пределах бюджета, а деньги считаются по факту из usage.
function setMemory(payload) {
    historySize = payload.history_size;
    const word = plural(historySize, "сообщение", "сообщения", "сообщений");
    memoryParam.textContent =
        `в диалоге ${historySize} ${word}, ${formatTokens(payload.history_tokens)} ток.`;

    const dropped = payload.history_size - payload.context_size;
    const tail = dropped > 0 ? `, выпало ${dropped}` : "";
    contextParam.textContent =
        `контекст ${formatTokens(payload.context_tokens)} из ${formatTokens(payload.context_budget)} ток.`
        + ` (${payload.context_size} сообщ.${tail})`;
    contextParam.classList.toggle("param-warning", dropped > 0);
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

function cell(text, className = "") {
    const td = document.createElement("td");
    td.textContent = text;
    if (className) {
        td.className = className;
    }
    return td;
}

// Таблица целиком приходит из базы: заголовок диалога — тоже запрос к модели,
// и в панели он стоит отдельной строкой рядом с ходами.
function renderTurns(turns) {
    let number = 0;
    let spent = 0;

    turnsEl.replaceChildren(...turns.map((turn) => {
        spent += turn.cost_usd;
        const total = turn.prompt_tokens + turn.completion_tokens;
        const row = document.createElement("tr");
        const label = turn.kind === "turn" ? String(++number) : "заголовок";

        row.className = turn.kind === "turn" ? "turn" : "turn turn-service";
        row.append(
            cell(label, "turn-label"),
            cell(formatTokens(turn.prompt_tokens)),
            cell(formatTokens(turn.cached_tokens)),
            cell(formatTokens(turn.completion_tokens)),
            cell(formatTokens(total)),
            cell(`${formatTokens(turn.estimated_tokens)} (${formatDrift(turn.estimated_tokens, turn.prompt_tokens)})`, "turn-estimate"),
            cell(formatCost(turn.cost_usd)),
            cell(formatCost(spent)),
        );
        return row;
    }));

    pendingRow = null;
    setTotals(turns, spent);
}

function setTotals(turns, spent) {
    const sum = (field) => turns.reduce((total, turn) => total + turn[field], 0);
    const prompt = sum("prompt_tokens");
    const completion = sum("completion_tokens");

    document.getElementById("total-prompt").textContent = formatTokens(prompt);
    document.getElementById("total-cached").textContent = formatTokens(sum("cached_tokens"));
    document.getElementById("total-completion").textContent = formatTokens(completion);
    document.getElementById("total-tokens").textContent = formatTokens(prompt + completion);
    document.getElementById("total-cost").textContent = formatCost(spent);

    totalsEl.hidden = turns.length === 0;
    tokensEmptyEl.hidden = turns.length !== 0;
    costParam.textContent = `потрачено ${formatCost(spent)}`;
}

// Оценка появляется в таблице до первого токена ответа: цена запроса известна
// раньше, чем модель начнёт отвечать. После ответа строку заменяет факт.
function addPendingRow(payload) {
    const row = document.createElement("tr");
    row.className = "turn turn-pending";
    row.append(
        cell("сейчас", "turn-label"),
        cell(`~${formatTokens(payload.request_tokens)}`),
        cell("—"),
        cell("—"),
        cell("—"),
        cell("оценка", "turn-estimate"),
        cell("—"),
        cell("—"),
    );

    turnsEl.append(row);
    tokensEmptyEl.hidden = true;
    pendingRow = row;
}

function dropPendingRow() {
    if (pendingRow) {
        pendingRow.remove();
        pendingRow = null;
        tokensEmptyEl.hidden = turnsEl.children.length !== 0;
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

async function ask(prompt, bubble) {
    const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, prompt, context_budget: budget }),
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
                throw new Error(JSON.parse(data).message);
            }
            if (event === "prompt") {
                const payload = JSON.parse(data);
                setMemory(payload);
                addPendingRow(payload);
                setStatus(promptStatus(payload));
                continue;
            }
            if (event === "done") {
                const payload = JSON.parse(data);
                stopThinking(bubble);
                setMemory(payload);
                renderTurns(payload.turns);
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

function promptStatus(payload) {
    const parts = [`Запрос ~${formatTokens(payload.request_tokens)} ток.`];
    if (payload.dropped_messages > 0) {
        parts.push(`вне контекста ${payload.dropped_messages} сообщ.`);
    }
    if (payload.over_budget) {
        parts.push("бюджет превышен одним вопросом");
    }
    return parts.join(", ");
}

function doneStatus(payload) {
    const parts = [
        `Запрос ${formatTokens(payload.prompt_tokens)} ток.`
        + ` (оценка ${formatTokens(payload.estimated_tokens)}, ${formatDrift(payload.estimated_tokens, payload.prompt_tokens)})`,
        `кэш ${formatTokens(payload.cached_tokens)}`,
        `ответ ${formatTokens(payload.completion_tokens)}`,
        `ход ${formatCost(payload.cost_usd)}`,
    ];
    if (payload.finish_reason === "length") {
        parts.unshift("Ответ обрезан по лимиту");
    }
    return parts.join(" · ");
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
        setStatus(doneStatus(payload));
        refreshDialogs();
    } catch (error) {
        // Оборванный ход агент в память не записал — убираем его и из ленты,
        // чтобы на экране не осталось того, чего собеседник не помнит.
        question.message.remove();
        answer.message.remove();
        placeholderEl.hidden = chatEl.querySelector(".message") !== null;
        promptInput.value = prompt;
        dropPendingRow();

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
    setMemory({
        history_size: 0,
        history_tokens: 0,
        context_size: 0,
        context_tokens: 0,
        context_budget: budget,
    });
    renderTurns([]);
    promptInput.value = config.prompt;
    await loadDialogs();
}

// Диалог хранится в базе, поэтому и лента, и таблица токенов восстанавливаются
// запросом сессии — и при переключении, и после перезагрузки страницы.
async function openSession(id) {
    const state = await loadState(id);
    if (!state) {
        return false;
    }

    sessionId = id;
    localStorage.setItem(SESSION_KEY, id);
    clearChat();
    for (const message of state.history) {
        addMessage(message.role === "user" ? "user" : "agent", message.content);
    }
    setMemory(state);
    renderTurns(state.turns);
    promptInput.value = state.history.length ? "" : config.prompt;
    scrollToBottom();
    renderDialogs();
    return true;
}

// Бюджет уходит в запрос: окно контекста считает сервер, и от бюджета зависит,
// сколько сообщений в него попадёт.
async function loadState(id) {
    const response = await fetch(`/api/session/${id}?context_budget=${budget}`);
    return response.ok ? await response.json() : null;
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

// Смена бюджета не меняет переписку, но меняет окно: пересчитываем его на сервере
// и показываем, сколько сообщений теперь остаётся за границей контекста.
async function changeBudget(value) {
    budget = value;
    localStorage.setItem(BUDGET_KEY, String(value));

    const state = await loadState(sessionId);
    if (!state) {
        return;
    }
    setMemory(state);
    const dropped = state.history_size - state.context_size;
    setStatus(dropped > 0
        ? `Бюджет ${formatTokens(value)} токенов: вне контекста ${dropped} сообщ.`
        : `Бюджет ${formatTokens(value)} токенов: весь диалог влезает в контекст`);
}

function describePricing(pricing, peak) {
    const rate = (value) => `$${(value * pricing.multiplier).toFixed(3)}/1M`;
    return [
        `оценка на ${config.encoding}, факт из usage`,
        `вход ${rate(pricing.input_miss)}, кэш ${rate(pricing.input_hit)}, выход ${rate(pricing.output)}`,
        peak ? "пиковый тариф ×2" : "непиковый тариф",
        `лимит модели ${formatTokens(config.modelLimit)} ток.`,
    ].join(" · ");
}

async function init() {
    const response = await fetch("/api/defaults");
    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }

    const payload = await response.json();
    config = {
        prompt: payload.prompt,
        budgetOptions: payload.budget_options,
        modelLimit: payload.model_context_limit,
        encoding: payload.encoding,
    };

    // Вкладка помнит выбранный бюджет, но шкалу задаёт сервер: чужое значение
    // из localStorage в список не попадёт.
    const saved = Number(localStorage.getItem(BUDGET_KEY));
    budget = config.budgetOptions.includes(saved) ? saved : payload.context_budget;

    // Параметры агента постоянны для всего диалога, меняются только счётчики.
    addParam(payload.model);
    addParam(`temperature ${payload.temperature}`);
    addParam(`лимит ответа ${formatTokens(payload.max_tokens)} ток.`);
    budgetSelect = addBudgetParam();
    memoryParam = addParam("");
    contextParam = addParam("");
    costParam = addParam("");
    tokensNoteEl.textContent = describePricing(payload.pricing, payload.peak);

    budgetSelect.addEventListener("change", () => {
        changeBudget(Number(budgetSelect.value)).catch((error) => setStatus(error.message, true));
    });

    await loadDialogs();

    // Вкладка помнит последний открытый диалог, но переписка принадлежит базе:
    // если сессии там уже нет, страница поднимает самую свежую из списка.
    const savedSession = localStorage.getItem(SESSION_KEY);
    if (!savedSession || !(await openSession(savedSession))) {
        await openDialogsHead();
    }

    for (const control of [submitButton, newDialogButton]) {
        control.disabled = false;
    }
}

init().catch((error) => {
    setStatus(error.message, true);
});
