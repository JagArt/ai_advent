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
const factsEl = document.getElementById("facts");
const factsMetaEl = document.getElementById("facts-meta");
const factsRowsEl = document.getElementById("facts-rows");
const branchesEl = document.getElementById("branches");
const branchListEl = document.getElementById("branch-list");

const submitLabel = submitButton.textContent;
const SESSION_KEY = "day10.session";
const BUDGET_KEY = "day10.budget";
const STRATEGY_KEY = "day10.strategy";
const WINDOW_KEY = "day10.window";

// Служебные запросы в таблице ходов подписаны словом, а не номером: в нумерацию
// диалога они не попадают.
const TURN_LABELS = { title: "заголовок", facts: "факты" };

const STRATEGY_LABELS = {
    window: "Sliding Window",
    facts: "Sticky Facts",
    branching: "Branching",
};
const STRATEGY_TITLES = {
    facts: "Sticky Facts / Key-Value Memory",
    branching: "Branching — ветки диалога",
};

// Параметры агента, прайс и шкалы приходят из GET /api/defaults.
let config = null;
let sessionId = null;
let branchId = null;
let historySize = 0;
let dialogs = [];
let branches = [];
let facts = [];
let freshFacts = new Set();
let memoryParam = null;
let contextParam = null;
let factsParam = null;
let costParam = null;
let budgetSelect = null;
let windowSelect = null;
let strategyControl = null;
let budget = null;
let strategy = null;
let windowMessages = null;
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
    // текущий стрим в чужую ленту. Ветки и checkpoint — тот же случай, поэтому их
    // кнопки на время хода выключаются.
    dialogsEl.classList.toggle("busy", busy);
    promptInput.readOnly = busy;
    // Стратегия, окно и бюджет — часть уходящего запроса, менять их посреди хода
    // нечестно: числа хода перестали бы сходиться с настройками на экране.
    budgetSelect.disabled = busy;
    windowSelect.disabled = busy;
    strategyControl.setDisabled(busy);
    for (const button of branchListEl.querySelectorAll(".branch")) {
        button.disabled = busy;
    }
    for (const button of chatEl.querySelectorAll(".message-fork")) {
        button.disabled = busy;
    }
}

function addParam(text) {
    const param = document.createElement("span");
    param.className = "param";
    param.textContent = text;
    paramsEl.append(param);
    return param;
}

// Стратегия — переключатель на три положения: контекст собирается либо только из
// последних реплик, либо из картотеки фактов рядом с ними, либо из ветки диалога.
function addStrategyParam() {
    const group = document.createElement("div");
    group.className = "param param-control param-strategy";
    group.setAttribute("role", "radiogroup");
    group.setAttribute("aria-label", "Стратегия контекста");
    group.append(document.createTextNode("стратегия "));

    const options = config.strategies.map((value) => {
        const label = document.createElement("label");
        label.className = "strategy";

        const input = document.createElement("input");
        input.type = "radio";
        input.name = "strategy";
        input.value = value;
        input.checked = value === strategy;

        if (STRATEGY_TITLES[value]) {
            label.title = STRATEGY_TITLES[value];
        }
        label.append(input, document.createTextNode(STRATEGY_LABELS[value] || value));
        group.append(label);
        return { value, label, input };
    });

    paramsEl.append(group);

    const paint = () => {
        for (const option of options) {
            option.label.classList.toggle("active", option.value === strategy);
        }
    };
    paint();

    return {
        options,
        paint,
        setDisabled(disabled) {
            group.classList.toggle("disabled", disabled);
            for (const option of options) {
                option.input.disabled = disabled;
            }
        },
    };
}

function addSelectParam(text, values, selected, format) {
    const label = document.createElement("label");
    label.className = "param param-control";
    label.textContent = text;

    const select = document.createElement("select");
    select.className = "budget";
    select.append(...values.map((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = format(value);
        option.selected = value === selected;
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

function messageWord(count) {
    return plural(count, "сообщение", "сообщения", "сообщений");
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

// База хранит ветку целиком, модель видит из неё окно последних реплик, а деньги
// считаются по факту из usage.
function setContext(payload) {
    historySize = payload.history_size;
    memoryParam.textContent =
        `в ветке ${historySize} ${messageWord(historySize)}, ${formatTokens(payload.history_tokens)} ток.`;

    // Реплики за окном для стратегии окна забыты по-настоящему, для фактов —
    // пересказаны картотекой: разницу и показывает счётчик.
    const dropped = payload.history_size - payload.context_size;
    const tail = dropped > 0 ? `, за окном ${dropped}` : "";
    contextParam.textContent =
        `контекст ${formatTokens(payload.context_tokens)} из ${formatTokens(payload.context_budget)} ток.`
        + ` (${payload.context_size} сообщ.${tail})`;
    contextParam.classList.toggle("param-warning", dropped > 0);

    factsParam.textContent = describeFacts(payload);
    factsParam.classList.toggle("param-warning", payload.facts_size > 0);
}

function describeFacts(payload) {
    if (payload.strategy !== "facts") {
        return `окно ${payload.window_messages} ${messageWord(payload.window_messages)} дословно`;
    }
    if (!payload.facts_size) {
        return `картотека пуста, окно ${payload.window_messages} сообщ.`;
    }
    return `картотека ${payload.facts_size} из ${config.factsLimit} фактов`
        + `, ${formatTokens(payload.facts_tokens)} ток.`;
}

// Картотека — единственное место, где видно, что агент помнит помимо окна.
// В других стратегиях она в запрос не идёт, поэтому и на странице её нет, хотя в
// базе факты остаются и вернутся вместе со стратегией.
function setFacts(items, fresh = new Set()) {
    facts = items;
    freshFacts = fresh;
    factsEl.hidden = strategy !== "facts" || items.length === 0;
    factsMetaEl.textContent = `${items.length} ${plural(items.length, "факт", "факта", "фактов")}`;
    factsRowsEl.replaceChildren(...items.map((fact) => {
        const row = document.createElement("tr");
        row.className = freshFacts.has(fact.key) ? "fact-fresh" : "";
        row.append(cell(fact.key, "fact-key"), cell(fact.value, "fact-value"));
        return row;
    }));
}

// Что записалось или уточнилось на этом ходу: модель возвращает картотеку целиком,
// а изменения видны сравнением с той, что была на экране.
function factsDiff(before, after) {
    const previous = new Map(before.map((fact) => [fact.key, fact.value]));
    return new Set(
        after.filter((fact) => previous.get(fact.key) !== fact.value).map((fact) => fact.key),
    );
}

// Полоса веток появляется, как только их стало больше одной: даже вернувшись в
// другую стратегию, из диалога с ветками нужно уметь выбраться.
function renderBranches(items) {
    branches = items;
    branchesEl.hidden = items.length === 0 || (items.length < 2 && strategy !== "branching");
    branchListEl.replaceChildren(...items.map((branch) => {
        const item = document.createElement("li");
        const button = document.createElement("button");
        button.type = "button";
        button.className = branch.active ? "branch active" : "branch";
        button.disabled = Boolean(controller);
        if (branch.active) {
            button.setAttribute("aria-current", "true");
        }

        const name = document.createElement("span");
        name.textContent = branch.name;

        const size = document.createElement("span");
        size.className = "branch-size";
        size.textContent = branch.size;

        button.append(name, size);
        button.addEventListener("click", () => switchBranch(branch.id));
        item.append(button);
        return item;
    }));
}

function isNearBottom(el, threshold = 64) {
    return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
}

function scrollToBottom(el) {
    el.scrollTop = el.scrollHeight;
}

// Таблица ходов следует за своим хвостом, а не за низом блока: под таблицей стоят
// итоги и картотека, и прокрутка «в самый низ» увела бы взгляд с только что
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

function addMessage(role, text, messageId = 0) {
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
    const bubble = { message, body, thinking: null, id: messageId, fork: null };
    setFork(bubble);
    chatEl.append(message);

    if (follow) {
        scrollToBottom(chatEl);
    }
    return bubble;
}

// Checkpoint — это реплика: кнопка живёт в пузыре и появляется только в стратегии
// веток, где от неё есть толк.
function setFork(bubble) {
    const needed = strategy === "branching" && bubble.id > 0;
    if (!needed) {
        bubble.fork?.remove();
        bubble.fork = null;
        return;
    }
    if (bubble.fork) {
        return;
    }

    const button = document.createElement("button");
    button.type = "button";
    button.className = "message-fork";
    button.textContent = "ветка отсюда";
    button.title = "Продолжить диалог с этого места в новой ветке";
    button.disabled = Boolean(controller);
    button.addEventListener("click", () => forkFrom(bubble.id));
    bubble.message.append(button);
    bubble.fork = button;
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

// Таблица целиком приходит из базы: заголовок диалога и разбор фактов — тоже
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
            cell(branches.length > 1 ? `${label} · ${branchName(turn.branch_id)}` : label, "turn-label"),
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

// Счёт остаётся общим на диалог, но с ветками одной нумерации ходов мало: без
// имени ветки непонятно, почему запрос вдруг подешевел.
function branchName(id) {
    return branches.find((branch) => branch.id === id)?.name || "—";
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

async function ask(prompt, question, answer) {
    const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            session_id: sessionId,
            prompt,
            context_budget: budget,
            strategy,
            window_messages: windowMessages,
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
    let update = null;

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
                setContext(payload);
                addPendingRow(payload);
                setStatus(promptStatus(payload));
                continue;
            }
            if (event === "facts") {
                // Разбор фактов идёт после ответа: он уже на экране, а строка
                // состояния до итогов хода показывает, что агент правит картотеку.
                update = JSON.parse(data);
                setFacts(update.facts, factsDiff(facts, update.facts));
                setStatus(factsStatus(update));
                continue;
            }
            if (event === "done") {
                const payload = JSON.parse(data);
                stopThinking(answer);
                setContext(payload);
                // Картотеку уже нарисовал кадр facts, и отметки о свежих фактах
                // переживают конец хода: перерисовывать её нечем.
                if (!update) {
                    setFacts(payload.facts);
                }
                // Реплики хода получили id — теперь от них можно ветвиться.
                question.id = payload.question_id;
                answer.id = payload.answer_id;
                setFork(question);
                setFork(answer);
                renderBranches(payload.branches);
                renderTurns(payload.turns);
                return { ...payload, update };
            }

            const text = JSON.parse(data);
            if (stopThinking(answer)) {
                setStatus("Агент отвечает...");
            }
            appendChunk(answer, text);
        }
    }

    throw new Error("Поток оборвался");
}

function promptStatus(payload) {
    const parts = [`Запрос ~${formatTokens(payload.request_tokens)} ток.`];
    if (payload.dropped_messages > 0) {
        parts.push(payload.strategy === "facts"
            ? `за окном ${payload.dropped_messages} сообщ., их помнит картотека`
            : `за окном ${payload.dropped_messages} сообщ.`);
    }
    if (payload.over_budget) {
        parts.push("бюджет превышен одним вопросом");
    }
    return parts.join(", ");
}

function factsStatus(payload) {
    const changes = [
        payload.added ? `+${payload.added}` : "",
        payload.changed ? `уточнено ${payload.changed}` : "",
        payload.removed ? `вычеркнуто ${payload.removed}` : "",
    ].filter(Boolean).join(", ") || "без изменений";
    return `Картотека: ${changes}`
        + ` — ${payload.facts_size} ${plural(payload.facts_size, "факт", "факта", "фактов")}`
        + `, ${formatTokens(payload.facts_tokens)} ток., ${formatCost(payload.cost_usd)}`;
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
    // Разбор фактов на этом ходу стоил своих токенов: без отдельной строки его
    // цена растворилась бы в итогах и выглядела бы как подорожавший ход.
    if (payload.update) {
        parts.push(`картотека ${formatCost(payload.update.cost_usd)}`);
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
        const payload = await ask(prompt, question, answer);
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
    open.title = dialog.branches > 1 ? `${title} — ${dialog.branches} ветки` : title;
    if (dialog.id === sessionId) {
        open.setAttribute("aria-current", "true");
    }

    const name = document.createElement("span");
    name.className = "dialog-name";
    name.textContent = title;

    const size = document.createElement("span");
    size.className = "dialog-size";
    // Реплики всех веток вместе: в списке важно, какой диалог больше, а не как он
    // разошёлся, — ветки видны после открытия.
    size.textContent = dialog.branches > 1 ? `${dialog.size} · ${dialog.branches}` : dialog.size;

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

    // Пустое состояние страница не выдумывает, а берёт с сервера тем же запросом,
    // что и при открытии готового диалога: у новой сессии уже есть корневая ветка.
    const payload = await response.json();
    await loadDialogs();
    if (!(await openSession(payload.session_id))) {
        throw new Error("Новый диалог не открылся");
    }
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
    showState(state);
    promptInput.value = state.history.length ? "" : config.prompt;
    renderDialogs();
    return true;
}

// Одно и то же состояние приходит и при открытии диалога, и после переключения
// ветки: и там и там меняется вся лента целиком.
function showState(state) {
    branchId = state.branch_id;
    clearChat();
    for (const message of state.history) {
        addMessage(message.role === "user" ? "user" : "agent", message.content, message.id);
    }
    setContext(state);
    setFacts(state.facts);
    renderBranches(state.branches);
    // Диалог открывается на последних ходах: и лента, и таблица показывают конец.
    renderTurns(state.turns, true);
    scrollToBottom(chatEl);
}

// Стратегия, окно и бюджет уходят в запрос: контекст считает сервер, и от этих
// настроек зависит, что в него попадёт.
async function loadState(id) {
    const query = new URLSearchParams({
        context_budget: budget,
        strategy,
        window_messages: windowMessages,
    });
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

// Ветка от checkpoint: реплики после него остаются в старой ветке, а разговор
// продолжается с этого места заново — и с той картотекой, что была на нём.
async function forkFrom(messageId) {
    if (controller) {
        return;
    }

    try {
        const state = await send(`/api/session/${sessionId}/branch`, { message_id: messageId });
        showState(state);
        const branch = state.branches.find((item) => item.active);
        setStatus(`Ветка «${branch.name}» от реплики ${messageId}: ${branch.size} ${messageWord(branch.size)}`);
        refreshDialogs();
    } catch (error) {
        setStatus(error.message, true);
    }
    promptInput.focus();
}

async function switchBranch(id) {
    if (controller || id === branchId) {
        promptInput.focus();
        return;
    }

    try {
        const state = await send(`/api/session/${sessionId}/branch/${id}`);
        showState(state);
        const branch = state.branches.find((item) => item.active);
        setStatus(`Ветка «${branch.name}»: ${branch.size} ${messageWord(branch.size)},`
            + ` картотека ${state.facts_size} ${plural(state.facts_size, "факт", "факта", "фактов")}`);
    } catch (error) {
        setStatus(error.message, true);
    }
    promptInput.focus();
}

// Ветвление и переключение возвращают то же состояние, что и открытие диалога:
// настройки контекста уходят в query, чтобы окно считалось по ним же.
async function send(path, body = null) {
    const query = new URLSearchParams({
        context_budget: budget,
        strategy,
        window_messages: windowMessages,
    });
    const response = await fetch(`${path}?${query}`, {
        method: "POST",
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
    });
    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }
    return await response.json();
}

// Смена стратегии не меняет переписку, но меняет запрос: сервер пересчитывает
// окно, а страница показывает, что теперь остаётся за его границей.
async function changeStrategy(value) {
    strategy = value;
    localStorage.setItem(STRATEGY_KEY, value);
    strategyControl.paint();

    const state = await reloadState();
    if (!state) {
        return;
    }
    setStatus(strategyStatus(state));
}

function strategyStatus(state) {
    const dropped = state.history_size - state.context_size;
    if (strategy === "window") {
        return `Sliding Window: в запрос идут последние ${state.window_messages} сообщ.`
            + (dropped > 0 ? `, остальные ${dropped} отброшены` : "");
    }
    if (strategy === "facts") {
        return `Sticky Facts / Key-Value Memory: картотека ${state.facts_size}`
            + ` ${plural(state.facts_size, "факт", "факта", "фактов")}`
            + ` плюс последние ${state.window_messages} сообщ.`;
    }
    return `Branching: ${branches.length} ${plural(branches.length, "ветка", "ветки", "веток")},`
        + ` checkpoint ставится кнопкой на реплике`;
}

async function changeWindow(value) {
    windowMessages = value;
    localStorage.setItem(WINDOW_KEY, String(value));

    const state = await reloadState();
    if (!state) {
        return;
    }
    const dropped = state.history_size - state.context_size;
    setStatus(dropped > 0
        ? `Окно ${value} ${messageWord(value)}: за границей ${dropped} сообщ.`
        : `Окно ${value} ${messageWord(value)}: вся ветка влезает в запрос`);
}

// Смена бюджета не меняет переписку, но может укоротить окно: длинный ход
// не влезает в оставшееся место, и его в запросе не будет.
async function changeBudget(value) {
    budget = value;
    localStorage.setItem(BUDGET_KEY, String(value));

    const state = await reloadState();
    if (!state) {
        return;
    }
    setStatus(state.context_size < Math.min(state.history_size, windowMessages)
        ? `Бюджет ${formatTokens(value)} токенов: окно короче ${windowMessages} сообщ.`
        : `Бюджет ${formatTokens(value)} токенов: окно влезает целиком`);
}

// Лента перерисовывается целиком, хотя переписка не менялась: вместе со стратегией
// у реплик появляются и исчезают кнопки checkpoint.
async function reloadState() {
    const state = await loadState(sessionId);
    if (state) {
        showState(state);
    }
    return state;
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
        strategies: payload.strategies,
        windowOptions: payload.window_options,
        factsLimit: payload.facts_limit,
        modelLimit: payload.model_context_limit,
        encoding: payload.encoding,
    };

    // Вкладка помнит выбранные настройки, но шкалы задаёт сервер: чужое значение
    // из localStorage в список не попадёт.
    const saved = (key, options, fallback) => {
        const value = Number(localStorage.getItem(key));
        return options.includes(value) ? value : fallback;
    };
    budget = saved(BUDGET_KEY, config.budgetOptions, payload.context_budget);
    windowMessages = saved(WINDOW_KEY, config.windowOptions, payload.window_messages);
    const savedStrategy = localStorage.getItem(STRATEGY_KEY);
    strategy = config.strategies.includes(savedStrategy) ? savedStrategy : payload.strategy;

    // Параметры агента постоянны для всего диалога, меняются только счётчики.
    addParam(payload.model);
    addParam(`temperature ${payload.temperature}`);
    addParam(`лимит ответа ${formatTokens(payload.max_tokens)} ток.`);
    strategyControl = addStrategyParam();
    windowSelect = addSelectParam("окно ", config.windowOptions, windowMessages, (value) => `${value} сообщ.`);
    budgetSelect = addSelectParam("бюджет ", config.budgetOptions, budget, (value) => `${formatTokens(value)} ток.`);
    memoryParam = addParam("");
    contextParam = addParam("");
    factsParam = addParam("");
    costParam = addParam("");
    tokensNoteEl.textContent = describePricing(payload.pricing, payload.peak);

    for (const option of strategyControl.options) {
        option.input.addEventListener("change", () => {
            if (option.input.checked) {
                changeStrategy(option.value).catch((error) => setStatus(error.message, true));
            }
        });
    }

    windowSelect.addEventListener("change", () => {
        changeWindow(Number(windowSelect.value)).catch((error) => setStatus(error.message, true));
    });

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
