const form = document.getElementById("form");
const questionInput = document.getElementById("question");
const submitButton = document.getElementById("submit");
const statusEl = document.getElementById("status");
const presetsEl = document.getElementById("presets");
const recheckButton = document.getElementById("recheck");
const ticketListEl = document.getElementById("ticketList");
const ticketsMeta = document.getElementById("ticketsMeta");
const messagesEl = document.getElementById("messages");
const chatMeta = document.getElementById("chatMeta");
const clearButton = document.getElementById("clearChat");

const card = {
    empty: document.getElementById("cardEmpty"),
    body: document.getElementById("cardBody"),
    id: document.getElementById("cardId"),
    meta: document.getElementById("cardMeta"),
    status: document.getElementById("cardStatus"),
    subject: document.getElementById("cardSubject"),
    thread: document.getElementById("thread"),
    resolution: document.getElementById("resolution"),
    userMeta: document.getElementById("userMeta"),
    userGrid: document.getElementById("userGrid"),
    history: document.getElementById("history"),
};

const dots = {
    mcp: [document.getElementById("mcpDot"), document.getElementById("mcpText")],
    local: [document.getElementById("localDot"), document.getElementById("localText")],
    cloud: [document.getElementById("cloudDot"), document.getElementById("cloudText")],
};

const submitLabel = submitButton.textContent;

let controller = null;
let currentTicket = null;
const histories = {}; // ticket_id → [{role, content}]

// --- утилиты -----------------------------------------------------------------

function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", isError);
}

function setBusy(busy) {
    submitButton.textContent = busy ? "Остановить" : submitLabel;
    submitButton.classList.toggle("stop", busy);
    for (const button of presetsEl.querySelectorAll("button")) button.disabled = busy;
}

function setDot(name, state, text) {
    const [dot, label] = dots[name];
    dot.className = `dot ${state}`;
    label.textContent = text;
}

function selectedProvider() {
    return document.querySelector('input[name="provider"]:checked').value;
}

function formatDate(iso) {
    const date = new Date(iso);
    return date.toLocaleString("ru-RU", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

function formatArgs(args) {
    return Object.entries(args)
        .map(([key, value]) => `${key}=${JSON.stringify(value)}`)
        .join(", ");
}

function scrollMessages() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderAnswerText(node, text) {
    node.textContent = "";
    const parts = text.replace(/^\s+/, "").split(/(\[\d+(?:,\s*\d+)*\])/g);
    for (const part of parts) {
        if (/^\[\d/.test(part)) node.append(el("span", "cite", part));
        else node.append(document.createTextNode(part));
    }
}

// --- статус ------------------------------------------------------------------

async function checkStatus() {
    setDot("mcp", "", "MCP...");
    setDot("local", "", "LM Studio...");
    setDot("cloud", "", "Облако...");
    try {
        const response = await fetch("/api/status");
        const status = await response.json();

        const tools = status.mcp.tools.map((tool) => tool.name).join(", ");
        setDot(
            "mcp",
            "on",
            `${status.mcp.server} · ${status.mcp.tools.length} инструментов: ${tools} · база знаний: ${status.kb.articles} статей, ${status.kb.sections} разделов · CRM: ${status.crm.users} пользователей, ${status.crm.tickets} тикетов`,
        );

        if (!status.local.online) {
            setDot("local", "off", status.local.error);
        } else if (!status.local.model_loaded) {
            setDot("local", "off", `${status.local.base_url} · модели ${status.local.model} нет в списке: ${status.local.models.join(", ")}`);
        } else {
            setDot("local", "on", `${status.local.base_url} · ${status.local.model}`);
        }
        document.getElementById("localModel").textContent = status.local.model;

        if (status.cloud.configured) {
            setDot("cloud", "on", `DeepSeek · ${status.cloud.model}`);
        } else {
            setDot("cloud", "off", "DEEPSEEK_API_KEY не задан — работает только локальная модель");
        }
        document.getElementById("cloudModel").textContent = status.cloud.model;

        if (!presetsEl.childElementCount) {
            for (const preset of status.presets) {
                const button = el("button", "preset", preset);
                button.type = "button";
                button.addEventListener("click", () => {
                    questionInput.value = preset;
                    form.requestSubmit();
                });
                presetsEl.append(button);
            }
        }
    } catch (error) {
        setDot("mcp", "off", `Не удалось проверить: ${error.message}`);
    }
}

// --- тикеты ------------------------------------------------------------------

async function loadTickets() {
    const response = await fetch("/api/tickets");
    const payload = await response.json();
    ticketListEl.innerHTML = "";
    const open = payload.tickets.filter((ticket) => ticket.status !== "closed").length;
    ticketsMeta.textContent = `${payload.total} всего · ${open} в работе`;

    for (const ticket of payload.tickets) {
        const li = el("li", "ticket");
        li.dataset.id = ticket.id;
        const top = el("div", "ticket-top");
        top.append(el("span", "", ticket.id), el("span", `badge ${ticket.status}`, ticket.status_title));
        li.append(top, el("div", "ticket-subject", ticket.subject));
        li.append(el("div", "ticket-user", `${ticket.user_name || ticket.user_id} · ${ticket.user_plan || ""} · ${formatDate(ticket.created_at)}`));
        li.addEventListener("click", () => selectTicket(ticket.id));
        ticketListEl.append(li);
    }
}

const USER_FIELDS = [
    ["Почта", (u) => u.email, "mono"],
    ["Тариф", (u) => u.plan_title],
    ["Статус аккаунта", (u) => u.status_title, (u) => (u.status !== "active" ? "flag" : "")],
    ["Почта подтверждена", (u) => (u.email_confirmed ? "да" : "нет"), (u) => (u.email_confirmed ? "" : "flag")],
    ["2FA", (u) => (u.two_factor ? "включена" : "выключена")],
    ["Неудачных входов", (u) => String(u.failed_logins), (u) => (u.failed_logins >= 5 ? "flag" : "")],
    ["Заблокирован до", (u) => (u.locked_until ? formatDate(u.locked_until) : null), () => "flag"],
    ["Платформа", (u) => (u.app_version ? `${u.platform} · приложение ${u.app_version}` : u.platform)],
    ["Команда", (u) => (u.team ? `${u.team.name} · ${u.team.role}` : null)],
    ["SSO команды", (u) => (u.team && u.team.sso_domain ? `${u.team.sso_domain}${u.team.sso_only ? " · только SSO" : ""}` : null), "mono"],
    ["Зарегистрирован", (u) => u.registered_at],
    ["Заметки", (u) => u.notes],
];

function renderUser(user) {
    card.userGrid.innerHTML = "";
    if (user.error) {
        card.userMeta.textContent = user.error;
        return;
    }
    card.userMeta.textContent = `${user.name} · ${user.id}`;
    for (const [label, getter, cls] of USER_FIELDS) {
        const value = getter(user);
        if (value === null || value === undefined || value === "") continue;
        const dd = el("dd", typeof cls === "function" ? cls(user) : cls || "", value);
        card.userGrid.append(el("dt", "", label), dd);
    }
    if (user.billing) {
        const attempts = user.billing.last_payment_attempts.map((a) => `${a.date}: ${a.result} (${a.code})`).join("; ");
        card.userGrid.append(el("dt", "", "Оплата"), el("dd", "flag", `был ${user.billing.previous_plan}, понижен ${user.billing.downgraded_at}; попытки: ${attempts}; в архиве проектов: ${user.billing.archived_projects}`));
    }
    if (user.integrations) {
        card.userGrid.append(el("dt", "", "Интеграции"), el("dd", "mono", JSON.stringify(user.integrations)));
    }
    if (user.api) {
        card.userGrid.append(el("dt", "", "API за час"), el("dd", "flag", `запросов ${user.api.requests_last_hour}, из них 429: ${user.api.rate_limited_last_hour}`));
    }
}

async function selectTicket(ticketId) {
    if (controller) controller.abort();
    currentTicket = ticketId;
    for (const li of ticketListEl.children) li.classList.toggle("active", li.dataset.id === ticketId);

    const response = await fetch(`/api/tickets/${encodeURIComponent(ticketId)}`);
    if (!response.ok) {
        setStatus(`Тикет не загрузился: ${response.status}`, true);
        return;
    }
    const { ticket, user, history } = await response.json();

    card.empty.hidden = true;
    card.body.hidden = false;
    card.id.textContent = ticket.id;
    card.meta.textContent = `${ticket.channel} · приоритет ${ticket.priority_title} · ${formatDate(ticket.created_at)} · ${ticket.tags.join(", ")}`;
    card.status.textContent = ticket.status_title;
    card.status.className = `badge ${ticket.status}`;
    card.subject.textContent = ticket.subject;

    card.thread.innerHTML = "";
    for (const message of ticket.messages) {
        const node = el("div", `msg ${message.from}`);
        node.append(el("div", "msg-head", `${message.from === "user" ? "клиент" : "поддержка"} · ${formatDate(message.at)}`), document.createTextNode(message.text));
        card.thread.append(node);
    }
    card.resolution.hidden = !ticket.resolution;
    if (ticket.resolution) {
        card.resolution.innerHTML = "";
        card.resolution.append(el("b", "", "Решение: "), document.createTextNode(ticket.resolution));
    }

    renderUser(user);

    card.history.innerHTML = "";
    for (const item of history) {
        const li = el("li");
        li.append(el("span", "hid", item.id), el("span", "", item.subject), el("span", `badge ${item.status}`, item.status_title));
        if (item.resolution) li.append(el("span", "hres", `— ${item.resolution}`));
        card.history.append(li);
    }

    renderChat();
    questionInput.focus();
}

// --- чат ---------------------------------------------------------------------

function renderChat() {
    messagesEl.innerHTML = "";
    const history = histories[currentTicket] || [];
    clearButton.hidden = history.length === 0;
    chatMeta.textContent = currentTicket ? `по тикету ${currentTicket} · ходов в истории: ${history.length / 2}` : "";
    if (!history.length) {
        messagesEl.append(el("div", "placeholder-box", "Задайте вопрос по выбранному тикету — модель откроет его сама"));
        return;
    }
    for (const turn of history) {
        if (turn.role === "user") {
            const node = el("div", "turn");
            node.append(el("div", "turn-q", turn.content));
            messagesEl.append(node);
        } else if (turn.node) {
            messagesEl.append(turn.node);
        } else {
            const node = el("div", "turn-a");
            const answer = el("pre", "answer");
            renderAnswerText(answer, turn.content);
            node.append(answer);
            messagesEl.append(node);
        }
    }
    scrollMessages();
}

function newTurn(question) {
    const q = el("div", "turn");
    q.append(el("div", "turn-q", question));
    const a = el("div", "turn-a");
    const trace = el("div", "trace");
    const reasoning = el("details", "reasoning");
    reasoning.hidden = true;
    reasoning.append(el("summary", "", "Рассуждения модели"), el("pre"));
    const answer = el("pre", "answer");
    const sources = el("ul", "sources");
    const meta = el("div", "turn-meta");
    a.append(trace, reasoning, answer, sources, meta);
    if (messagesEl.querySelector(".placeholder-box")) messagesEl.innerHTML = "";
    messagesEl.append(q, a);
    scrollMessages();
    return { node: a, trace, reasoning, answer, sources, meta, text: "", reasoningText: "" };
}

function parseFrame(frame) {
    let event = "message";
    const dataLines = [];
    for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    return { event, data: dataLines.join("\n") };
}

function formatDone(done) {
    const parts = [`${done.model}`, `раундов ${done.rounds}`, `вызовов ${done.calls}`];
    if (done.nudged) parts.push("толчок");
    if (done.ttft_ms !== null) parts.push(`первый токен ${(done.ttft_ms / 1000).toFixed(1)} с`);
    parts.push(`всего ${(done.total_ms / 1000).toFixed(1)} с`);
    if (done.prompt_tokens !== null) parts.push(`токены ${done.prompt_tokens} → ${done.completion_tokens}`);
    if (done.finish_reason && done.finish_reason !== "stop") parts.push(`finish_reason: ${done.finish_reason}`);
    return parts.join(" · ");
}

function addTrace(turn, line) {
    turn.trace.append(line);
    scrollMessages();
}

async function ask(question, provider) {
    const ticketId = currentTicket;
    const history = (histories[ticketId] ||= []);
    const turn = newTurn(question);
    const priorHistory = history.map(({ role, content }) => ({ role, content }));
    history.push({ role: "user", content: question });
    const assistantTurn = { role: "assistant", content: "", node: turn.node };
    history.push(assistantTurn);
    renderChatMeta();

    const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticket_id: ticketId, question, provider, history: priorHistory }),
        signal: controller.signal,
    });
    if (!response.ok) throw new Error(`Сервер вернул ${response.status}`);

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

            if (event === "error") throw new Error(payload);
            if (event === "tool") {
                const line = el("div", `trace-line${payload.ok ? "" : " fail"}`);
                line.append(el("span", "", `⚙ r${payload.round}`), el("span", "t-name", ` ${payload.name}`), el("span", "t-args", payload.ok ? `(${formatArgs(payload.arguments)})` : `(${formatArgs(payload.arguments)}) · отказ: ${payload.error}`), el("span", "t-ms", `${payload.ms} мс`));
                line.title = payload.ok ? formatArgs(payload.arguments) : payload.error;
                addTrace(turn, line);
                continue;
            }
            if (event === "clear") {
                turn.text = "";
                turn.answer.textContent = "";
                continue;
            }
            if (event === "nudge") {
                // Черновик без вызовов остаётся на экране зачёркнутым: видно, что модель
                // ответила бы, не заглянув в данные, и что этот ответ не пошёл в дело.
                const draft = el("pre", "answer draft");
                draft.textContent = payload.draft || "(пустой черновик)";
                turn.answer.before(draft);
                const line = el("div", "trace-line nudge");
                line.append(el("span", "", `↻ раунд ${payload.round} прошёл без вызовов — черновик отброшен, модели дан толчок`));
                addTrace(turn, line);
                turn.text = "";
                turn.answer.textContent = "";
                continue;
            }
            if (event === "reasoning") {
                turn.reasoningText += payload;
                turn.reasoning.hidden = false;
                turn.reasoning.querySelector("pre").textContent = turn.reasoningText.trim();
                continue;
            }
            if (event === "done") {
                turn.text = payload.text;
                renderAnswerText(turn.answer, payload.text || "(модель не дала текста)");
                assistantTurn.content = payload.text;
                for (const source of payload.sources) {
                    const li = el("li");
                    li.append(el("span", "n", `[${source.number}]`), document.createTextNode(` ${source.title} — «${source.heading}»`));
                    turn.sources.append(li);
                }
                turn.meta.textContent = formatDone(payload);
                scrollMessages();
                return;
            }
            turn.text += payload;
            renderAnswerText(turn.answer, turn.text);
            scrollMessages();
        }
    }
}

function renderChatMeta() {
    const history = histories[currentTicket] || [];
    clearButton.hidden = history.length === 0;
    chatMeta.textContent = `по тикету ${currentTicket} · ходов в истории: ${Math.floor(history.length / 2)}`;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (controller) {
        controller.abort();
        return;
    }
    if (!currentTicket) {
        setStatus("Сначала выберите тикет", true);
        return;
    }
    const question = questionInput.value.trim();
    if (!question) {
        setStatus("Введите вопрос", true);
        return;
    }

    controller = new AbortController();
    setBusy(true);
    setStatus("Модель работает...");
    questionInput.value = "";
    const turnNode = () => messagesEl.lastElementChild;

    try {
        await ask(question, selectedProvider());
        setStatus("Готово");
    } catch (error) {
        const node = turnNode();
        const meta = node && node.querySelector(".turn-meta");
        if (error.name === "AbortError") {
            if (meta) meta.textContent = "остановлено";
            setStatus("Остановлено");
        } else {
            const answer = node && node.querySelector(".answer:not(.draft)");
            if (answer) {
                answer.textContent = error.message;
                answer.classList.add("error");
            }
            if (meta) {
                meta.textContent = "ошибка";
                meta.classList.add("error");
            }
            setStatus("Ошибка", true);
        }
    } finally {
        controller = null;
        setBusy(false);
    }
});

clearButton.addEventListener("click", () => {
    histories[currentTicket] = [];
    renderChat();
});

questionInput.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        form.requestSubmit();
    }
});

recheckButton.addEventListener("click", checkStatus);

checkStatus();
loadTickets();
