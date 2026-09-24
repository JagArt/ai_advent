const form = document.getElementById("form");
const promptInput = document.getElementById("prompt");
const submitButton = document.getElementById("submit");
const checkButton = document.getElementById("check");
const statusEl = document.getElementById("status");

const watcherEl = document.getElementById("watcher");
const feedEl = document.getElementById("feed");
const jobsBody = document.getElementById("jobsBody");
const jobsMeta = document.getElementById("jobsMeta");
const refreshJobsButton = document.getElementById("refreshJobs");

const connectionPanel = document.getElementById("connection");
const connectionBody = document.getElementById("connectionBody");
const connectionMeta = document.getElementById("connectionMeta");
const callsPanel = document.getElementById("calls");
const callsBody = document.getElementById("callsBody");
const callsMeta = document.getElementById("callsMeta");
const answerEl = document.getElementById("answer");
const answerMeta = document.getElementById("answerMeta");
const copyButton = document.getElementById("copy");

const submitLabel = submitButton.textContent;
// После этих вызовов расписание на странице устарело.
const SCHEDULE_TOOLS = new Set(["schedule_job", "cancel_job"]);
const JOBS_REFRESH_MS = 30000;

let controller = null;
let answerText = "";
let callCount = 0;

function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", isError);
}

// Пока идёт стрим, та же кнопка работает на остановку.
function setBusy(busy) {
    submitButton.textContent = busy ? "Остановить" : submitLabel;
    submitButton.classList.toggle("stop", busy);
    checkButton.disabled = busy;
}

function setAnswer(text) {
    answerText = text;
    answerEl.textContent = text;
    copyButton.hidden = !text;
}

function setMeta(element, text, isError = false) {
    element.textContent = text;
    element.classList.toggle("error", isError);
}

function placeholder(text) {
    const span = document.createElement("span");
    span.className = "placeholder";
    span.textContent = text;
    return span;
}

function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

function formatTime(iso) {
    if (!iso) return "—";
    return new Date(iso).toLocaleString("ru-RU", {
        day: "2-digit",
        month: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
    });
}

function formatPeriod(period) {
    const from = period.from ? formatTime(period.from) : "начала сбора";
    return `с ${from} по ${formatTime(period.to)}`;
}

// --- лента сводок ----------------------------------------------------------

function setWatcher(status) {
    const labels = {
        starting: "агент запускается",
        online: "агент на связи",
        offline: "MCP-сервер недоступен",
    };
    watcherEl.textContent = labels[status.state] || status.state;
    watcherEl.className = `badge ${status.state}`;
    watcherEl.title = [status.detail, status.checked_at && `проверено ${formatTime(status.checked_at)}`]
        .filter(Boolean)
        .join("\n");
}

function addFeedItem(item) {
    feedEl.querySelector(":scope > .placeholder")?.remove();

    const card = element("article", item.error ? "entry error" : "entry");

    const head = element("div", "entry-head");
    head.append(
        element("span", "entry-title", `Сводка #${item.summary_id}`),
        element("span", "meta", `${formatPeriod(item.period)} · снимков: ${item.summary.snapshots}`),
    );

    const text = element(
        "p",
        "entry-text",
        item.text || `Пересказ не удался: ${item.error}. Агрегат ниже.`,
    );

    const details = element("details", "entry-raw");
    details.append(
        element("summary", "", "агрегат от MCP-сервера"),
        element("pre", "call-result", JSON.stringify(item.summary, null, 2)),
    );

    card.append(head, text, details);
    // Новые сверху: свежая сводка важнее вчерашней.
    feedEl.prepend(card);
}

function subscribeFeed() {
    const source = new EventSource("/api/feed");

    source.addEventListener("status", (event) => setWatcher(JSON.parse(event.data)));
    source.addEventListener("item", (event) => {
        addFeedItem(JSON.parse(event.data));
        loadJobs();
    });
    source.addEventListener("open", () => {
        // Переподключение присылает ленту заново целиком.
        feedEl.replaceChildren(
            placeholder("Сводок пока нет: первая появится по расписанию задачи summary"),
        );
    });
    source.addEventListener("error", () => {
        watcherEl.textContent = "веб-агент недоступен, переподключаемся...";
        watcherEl.className = "badge offline";
    });
}

// --- расписание ------------------------------------------------------------

function renderJobs(jobs) {
    const table = element("table", "jobs");
    const header = element("tr");
    for (const title of ["#", "задача", "расписание", "следующий запуск", "последний", "запусков", ""]) {
        header.append(element("th", "", title));
    }
    table.append(header);

    for (const job of jobs) {
        const row = element("tr", job.active ? "" : "inactive");
        row.append(
            element("td", "", String(job.id)),
            element("td", "mono", job.kind),
            element("td", "", job.schedule),
            element("td", "", job.active ? formatTime(job.next_run_at) : "завершена"),
            element("td", "", formatTime(job.last_run_at)),
            element("td", "", String(job.runs)),
            element("td", job.last_error ? "job-error" : "", job.last_error || ""),
        );
        table.append(row);
    }

    jobsBody.replaceChildren(table);
}

async function loadJobs() {
    try {
        const response = await fetch("/api/jobs");
        const payload = await response.json();
        if (payload.error) throw new Error(payload.error);

        renderJobs(payload.jobs);
        const active = payload.jobs.filter((job) => job.active).length;
        setMeta(jobsMeta, `активных: ${active} · обновлено ${formatTime(new Date().toISOString())}`);
    } catch (error) {
        jobsBody.replaceChildren(placeholder("Расписание недоступно: MCP-сервер не отвечает"));
        setMeta(jobsMeta, error.message, true);
    }
}

// --- чат -------------------------------------------------------------------

function argsOf(schema) {
    const properties = schema.properties || {};
    const required = new Set(schema.required || []);
    return Object.keys(properties)
        .map((name) => (required.has(name) ? name : `${name}?`))
        .join(", ");
}

function showConnection(info) {
    connectionPanel.hidden = false;
    connectionBody.replaceChildren();
    setMeta(connectionMeta, `установлено за ${info.elapsed_ms} мс`);

    const facts = element("dl", "facts");
    const rows = [
        ["адрес", info.url],
        ["сервер", info.server],
        ["протокол", info.protocol],
        ["возможности", info.capabilities.join(", ")],
    ];
    for (const [term, value] of rows) {
        facts.append(element("dt", "", term), element("dd", "", value));
    }
    connectionBody.append(facts);

    const list = element("div", "tools");
    for (const tool of info.tools) {
        const card = element("article", "tool");
        card.append(
            element("code", "tool-name", `${tool.name}(${argsOf(tool.input_schema)})`),
            element("p", "tool-description", tool.description),
        );
        list.append(card);
    }
    connectionBody.append(list);
}

function addCall(call) {
    callCount += 1;
    callsPanel.hidden = false;
    setMeta(callsMeta, `вызовов: ${callCount}`);

    const card = element("article", call.is_error ? "call error" : "call");
    const row = element("div", "call-row");
    row.append(
        element("code", "call-head", `${call.name}(${JSON.stringify(call.arguments)})`),
        element("span", "meta", `${call.elapsed_ms} мс`),
    );
    card.append(row, element("pre", "call-result", call.result));
    callsBody.append(card);

    if (SCHEDULE_TOOLS.has(call.name) && !call.is_error) {
        loadJobs();
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
    const parts = [`раундов: ${payload.rounds}`, `вызовов: ${payload.calls}`];
    if (payload.finish_reason) {
        parts.push(`finish_reason: ${payload.finish_reason}`);
    }
    setMeta(answerMeta, parts.join(" · "));
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
            const payload = JSON.parse(data);

            if (event === "error") {
                throw new Error(payload);
            }
            if (event === "mcp") {
                showConnection(payload);
            } else if (event === "tool") {
                addCall(payload);
            } else if (event === "done") {
                applyDone(payload);
                return;
            } else {
                setAnswer(answerText + payload);
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
    setStatus("Подключаемся к MCP-серверу...");
    setAnswer("");
    setMeta(answerMeta, "");
    callCount = 0;
    callsBody.replaceChildren();
    callsPanel.hidden = true;
    setMeta(callsMeta, "");

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

checkButton.addEventListener("click", async () => {
    setStatus("Проверяем соединение...");

    try {
        const response = await fetch("/api/connect", { method: "POST" });
        const payload = await response.json();

        if (payload.error) {
            throw new Error(payload.error);
        }

        showConnection(payload);
        setStatus(`Соединение установлено, инструментов: ${payload.tools.length}`);
    } catch (error) {
        connectionPanel.hidden = false;
        connectionBody.replaceChildren(placeholder("Соединение не установлено"));
        setMeta(connectionMeta, "");
        setStatus(error.message, true);
    }
});

document.getElementById("examples").addEventListener("click", (event) => {
    if (event.target.classList.contains("chip")) {
        promptInput.value = event.target.textContent;
        promptInput.focus();
    }
});

refreshJobsButton.addEventListener("click", loadJobs);

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

subscribeFeed();
loadJobs();
setInterval(loadJobs, JOBS_REFRESH_MS);
