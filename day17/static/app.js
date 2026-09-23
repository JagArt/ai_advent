const form = document.getElementById("form");
const promptInput = document.getElementById("prompt");
const mcpToggle = document.getElementById("mcp");
const submitButton = document.getElementById("submit");
const checkButton = document.getElementById("check");
const statusEl = document.getElementById("status");

const connectionBody = document.getElementById("connectionBody");
const connectionMeta = document.getElementById("connectionMeta");
const callsPanel = document.getElementById("calls");
const callsBody = document.getElementById("callsBody");
const callsMeta = document.getElementById("callsMeta");
const answerEl = document.getElementById("answer");
const answerMeta = document.getElementById("answerMeta");
const copyButton = document.getElementById("copy");

const submitLabel = submitButton.textContent;

let controller = null;
let answerText = "";
let callCount = 0;
let follow = true;

// Лента едет за ответом, пока пользователь сам не ушёл вверх.
function scrollToEnd() {
    answerEl.scrollTop = answerEl.scrollHeight;
    window.scrollTo(0, document.documentElement.scrollHeight);
}

function nearEnd() {
    const left =
        document.documentElement.scrollHeight - window.innerHeight - window.scrollY;
    return left <= 48;
}

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

function argsOf(schema) {
    const properties = schema.properties || {};
    const required = new Set(schema.required || []);
    return Object.keys(properties)
        .map((name) => (required.has(name) ? name : `${name}?`))
        .join(", ");
}

function showConnection(info) {
    connectionBody.replaceChildren();
    setMeta(connectionMeta, `установлено за ${info.elapsed_ms} мс`);

    const facts = document.createElement("dl");
    facts.className = "facts";
    const rows = [
        ["команда", info.command],
        ["сервер", info.server],
        ["протокол", info.protocol],
        ["возможности", info.capabilities.join(", ")],
    ];
    for (const [term, value] of rows) {
        const dt = document.createElement("dt");
        dt.textContent = term;
        const dd = document.createElement("dd");
        dd.textContent = value;
        facts.append(dt, dd);
    }
    connectionBody.append(facts);

    const list = document.createElement("div");
    list.className = "tools";
    for (const tool of info.tools) {
        const card = document.createElement("article");
        card.className = "tool";

        const name = document.createElement("code");
        name.className = "tool-name";
        name.textContent = `${tool.name}(${argsOf(tool.input_schema)})`;

        const description = document.createElement("p");
        description.className = "tool-description";
        description.textContent = tool.description;

        card.append(name, description);
        list.append(card);
    }
    connectionBody.append(list);
}

function showDisconnected() {
    connectionBody.replaceChildren(
        placeholder("MCP выключен: инструментов у модели нет, отвечает по памяти"),
    );
    setMeta(connectionMeta, "");
}

function addCall(call) {
    callCount += 1;
    callsPanel.hidden = false;
    setMeta(callsMeta, `вызовов: ${callCount}`);

    const card = document.createElement("article");
    card.className = call.is_error ? "call error" : "call";

    const head = document.createElement("code");
    head.className = "call-head";
    head.textContent = `${call.name}(${JSON.stringify(call.arguments)})`;

    const time = document.createElement("span");
    time.className = "meta";
    time.textContent = `${call.elapsed_ms} мс`;

    const row = document.createElement("div");
    row.className = "call-row";
    row.append(head, time);

    const result = document.createElement("pre");
    result.className = "call-result";
    result.textContent = call.result;

    card.append(row, result);
    callsBody.append(card);
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

async function ask(prompt, mcp) {
    const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, mcp }),
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

            if (follow) {
                scrollToEnd();
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

    const mcp = mcpToggle.checked;

    follow = true;
    controller = new AbortController();
    setBusy(true);
    setStatus(mcp ? "Поднимаем MCP-сервер..." : "Спрашиваем без MCP...");
    setAnswer("");
    setMeta(answerMeta, "");
    callCount = 0;
    callsBody.replaceChildren();
    callsPanel.hidden = true;
    setMeta(callsMeta, "");
    if (mcp) {
        connectionBody.replaceChildren(placeholder("Подключаемся..."));
        setMeta(connectionMeta, "");
    } else {
        showDisconnected();
    }

    try {
        await ask(prompt, mcp);
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
        // Итог хода дописывается под ответом уже после последнего куска текста.
        if (follow) {
            scrollToEnd();
        }
    }
});

checkButton.addEventListener("click", async () => {
    setStatus("Проверяем соединение...");
    connectionBody.replaceChildren(placeholder("Подключаемся..."));
    setMeta(connectionMeta, "");

    try {
        const response = await fetch("/api/connect", { method: "POST" });
        const payload = await response.json();

        if (payload.error) {
            throw new Error(payload.error);
        }

        showConnection(payload);
        setStatus(`Соединение установлено, инструментов: ${payload.tools.length}`);
    } catch (error) {
        connectionBody.replaceChildren(placeholder("Соединение не установлено"));
        setStatus(error.message, true);
    }
});

mcpToggle.addEventListener("change", () => {
    if (mcpToggle.checked) {
        connectionBody.replaceChildren(
            placeholder("MCP включён: соединение поднимется на первом запросе"),
        );
        setMeta(connectionMeta, "");
    } else {
        showDisconnected();
    }
});

window.addEventListener("scroll", () => {
    follow = nearEnd();
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
