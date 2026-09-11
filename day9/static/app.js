const form = document.getElementById("form");
const promptInput = document.getElementById("prompt");
const submitButton = document.getElementById("submit");
const newDialogButton = document.getElementById("new-dialog");
const dialogsEl = document.getElementById("dialogs");
const statusEl = document.getElementById("status");
const paramsEl = document.getElementById("params");
const chatEl = document.getElementById("chat");
const placeholderEl = document.getElementById("placeholder");
const tokensEl = document.getElementById("tokens");
const turnsEl = document.getElementById("turns");
const totalsEl = document.getElementById("totals");
const tokensNoteEl = document.getElementById("tokens-note");
const tokensEmptyEl = document.getElementById("tokens-empty");
const summaryEl = document.getElementById("summary");
const summaryMetaEl = document.getElementById("summary-meta");
const summaryTextEl = document.getElementById("summary-text");

const submitLabel = submitButton.textContent;
const SESSION_KEY = "day9.session";
const BUDGET_KEY = "day9.budget";
const COMPRESSION_KEY = "day9.compression";

// Служебные запросы в таблице ходов подписаны словом, а не номером: в нумерацию
// диалога они не попадают.
const TURN_LABELS = { title: "заголовок", summary: "свёртка" };

// Параметры агента, прайс и шкала бюджетов приходят из GET /api/defaults.
let config = null;
let sessionId = null;
let historySize = 0;
let dialogs = [];
let memoryParam = null;
let contextParam = null;
let summaryParam = null;
let costParam = null;
let budgetSelect = null;
let compressionToggle = null;
let budget = null;
let compression = true;
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
    // Бюджет и сжатие — часть уходящего запроса, менять их посреди хода нечестно.
    budgetSelect.disabled = busy;
    compressionToggle.disabled = busy;
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

function addCompressionParam() {
    const label = document.createElement("label");
    label.className = "param param-control param-toggle";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = compression;

    label.append(checkbox, document.createTextNode(" сжатие истории"));
    paramsEl.append(label);
    return checkbox;
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

// База хранит диалог целиком, модель видит хвост дословно и всё остальное
// конспектом, а деньги считаются по факту из usage.
function setMemory(payload) {
    historySize = payload.history_size;
    const word = plural(historySize, "сообщение", "сообщения", "сообщений");
    memoryParam.textContent =
        `в диалоге ${historySize} ${word}, ${formatTokens(payload.history_tokens)} ток.`;

    // Реплика, которой нет ни дословно, ни в конспекте, забыта по-настоящему:
    // свёрнутые в счёт потерь не идут.
    const dropped = payload.history_size - payload.context_size - payload.summary_size;
    const tail = dropped > 0 ? `, выпало ${dropped}` : "";
    contextParam.textContent =
        `контекст ${formatTokens(payload.context_tokens)} из ${formatTokens(payload.context_budget)} ток.`
        + ` (${payload.context_size} сообщ.${tail})`;
    contextParam.classList.toggle("param-warning", dropped > 0);

    summaryParam.textContent = describeSummary(payload);
    summaryParam.classList.toggle("param-warning", payload.summary_size > 0);
}

function describeSummary(payload) {
    if (!payload.compression) {
        return "сжатие выключено";
    }

    const left = Math.max(config.compressEvery - payload.pending_size, 0);
    const word = plural(left, "сообщение", "сообщения", "сообщений");
    const tail = `, до свёртки ${left} ${word}`;
    if (!payload.summary_size) {
        return `конспекта нет${tail}`;
    }
    return `конспект ${formatTokens(payload.summary_tokens)} ток.`
        + ` вместо ${payload.summary_size} сообщ.${tail}`;
}

// Текст конспекта — единственное место, где видно, что именно агент помнит
// вместо выброшенных из запроса реплик. С выключенным сжатием он в запрос не
// идёт, поэтому и на странице его нет, хотя в базе конспект остаётся.
function setSummary(summary) {
    summaryEl.hidden = !summary || !compression;
    if (!summary) {
        summaryTextEl.textContent = "";
        return;
    }

    summaryMetaEl.textContent =
        `${summary.messages} сообщ. → ${formatTokens(summary.tokens)} ток.`;
    summaryTextEl.textContent = summary.text;
}

function isNearBottom(el, threshold = 64) {
    return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
}

function scrollToBottom(el) {
    el.scrollTop = el.scrollHeight;
}

// Таблица ходов следует за своим хвостом, а не за низом блока: под таблицей стоят
// итоги и конспект, и прокрутка «в самый низ» увела бы взгляд с только что
// появившегося хода.
function isTurnsTailVisible(threshold = 24) {
    const last = turnsEl.lastElementChild;
    if (!last) {
        return true;
    }
    const view = tokensEl.getBoundingClientRect();
    const row = last.getBoundingClientRect();
    return row.bottom >= view.top && row.bottom - view.bottom <= threshold;
}

// block: "nearest" прокручивает минимально и только сам блок статистики, поэтому
// липкий заголовок таблицы строку не перекрывает, а страница не сдвигается.
function scrollToTurnsTail() {
    turnsEl.lastElementChild?.scrollIntoView({ block: "nearest" });
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
        scrollToBottom(chatEl);
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
        scrollToBottom(chatEl);
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

// Таблица целиком приходит из базы: заголовок диалога и свёртка истории — тоже
// запросы к модели, и в панели они стоят отдельными строками рядом с ходами.
function renderTurns(turns, force = false) {
    const follow = force || isTurnsTailVisible();
    let number = 0;
    let spent = 0;

    turnsEl.replaceChildren(...turns.map((turn) => {
        spent += turn.cost_usd;
        const total = turn.prompt_tokens + turn.completion_tokens;
        const row = document.createElement("tr");
        const label = turn.kind === "turn" ? String(++number) : TURN_LABELS[turn.kind];

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

    if (follow) {
        scrollToTurnsTail();
    }
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
    const follow = isTurnsTailVisible();
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

    if (follow) {
        scrollToTurnsTail();
    }
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
        body: JSON.stringify({
            session_id: sessionId,
            prompt,
            context_budget: budget,
            compression,
        }),
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
    let compaction = null;

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
            if (event === "compaction") {
                // Свёртка идёт после ответа: он уже на экране, а строка состояния
                // до итогов хода показывает, что агент пересказывает себе старое.
                compaction = JSON.parse(data);
                setStatus(compactionStatus(compaction));
                continue;
            }
            if (event === "done") {
                const payload = JSON.parse(data);
                stopThinking(bubble);
                setMemory(payload);
                setSummary(payload.summary);
                renderTurns(payload.turns);
                return { ...payload, compaction };
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

function compactionStatus(payload) {
    return `Свёртка: ${payload.messages} сообщ. вместо ${formatTokens(payload.replaced_tokens)} ток.`
        + ` пересказаны в ${formatTokens(payload.summary_tokens)}, ${formatCost(payload.cost_usd)}`;
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
    // Свёртка на этом ходу стоила своих токенов: без отдельной строки её цена
    // растворилась бы в итогах и выглядела бы как подорожавший ход.
    if (payload.compaction) {
        parts.push(`свёртка ${payload.compaction.messages} сообщ. за ${formatCost(payload.compaction.cost_usd)}`);
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
        compression,
        summary_size: 0,
        summary_tokens: 0,
        pending_size: 0,
    });
    setSummary(null);
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
    setSummary(state.summary);
    // Диалог открывается на последних ходах: и лента, и таблица показывают конец.
    renderTurns(state.turns, true);
    promptInput.value = state.history.length ? "" : config.prompt;
    scrollToBottom(chatEl);
    renderDialogs();
    return true;
}

// Бюджет и сжатие уходят в запрос: окно контекста считает сервер, и от этих двух
// настроек зависит, что в него попадёт — хвост с конспектом или вся история.
async function loadState(id) {
    const query = new URLSearchParams({ context_budget: budget, compression });
    const response = await fetch(`/api/session/${id}?${query}`);
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

    const state = await reloadState();
    if (!state) {
        return;
    }
    const dropped = state.history_size - state.context_size - state.summary_size;
    setStatus(dropped > 0
        ? `Бюджет ${formatTokens(value)} токенов: вне контекста ${dropped} сообщ.`
        : `Бюджет ${formatTokens(value)} токенов: весь диалог влезает в контекст`);
}

// Выключенное сжатие не удаляет конспект: он остаётся в базе и снова попадёт в
// запрос, как только галочку вернут. Один диалог так можно провести дважды.
async function changeCompression(value) {
    compression = value;
    localStorage.setItem(COMPRESSION_KEY, String(value));

    const state = await reloadState();
    if (!state) {
        return;
    }
    setStatus(value
        ? `Сжатие включено: в запрос идут конспект и последние ${config.tailMessages} сообщ.`
        : `Сжатие выключено: в запрос идёт вся история под бюджет ${formatTokens(budget)} ток.`);
}

async function reloadState() {
    const state = await loadState(sessionId);
    if (state) {
        setMemory(state);
        setSummary(state.summary);
    }
    return state;
}

function describePricing(pricing, peak) {
    const rate = (value) => `$${(value * pricing.multiplier).toFixed(3)}/1M`;
    return [
        `хвост ${config.tailMessages} сообщ. дословно, свёртка каждые ${config.compressEvery}`,
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
        tailMessages: payload.tail_messages,
        compressEvery: payload.compress_every,
    };

    // Вкладка помнит выбранный бюджет, но шкалу задаёт сервер: чужое значение
    // из localStorage в список не попадёт.
    const saved = Number(localStorage.getItem(BUDGET_KEY));
    budget = config.budgetOptions.includes(saved) ? saved : payload.context_budget;

    const savedCompression = localStorage.getItem(COMPRESSION_KEY);
    compression = savedCompression === null ? payload.compression : savedCompression === "true";

    // Параметры агента постоянны для всего диалога, меняются только счётчики.
    addParam(payload.model);
    addParam(`temperature ${payload.temperature}`);
    addParam(`лимит ответа ${formatTokens(payload.max_tokens)} ток.`);
    budgetSelect = addBudgetParam();
    compressionToggle = addCompressionParam();
    memoryParam = addParam("");
    contextParam = addParam("");
    summaryParam = addParam("");
    costParam = addParam("");
    tokensNoteEl.textContent = describePricing(payload.pricing, payload.peak);

    budgetSelect.addEventListener("change", () => {
        changeBudget(Number(budgetSelect.value)).catch((error) => setStatus(error.message, true));
    });

    compressionToggle.addEventListener("change", () => {
        changeCompression(compressionToggle.checked).catch((error) => setStatus(error.message, true));
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
