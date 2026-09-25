const pipelineForm = document.getElementById("pipelineForm");
const queryInput = document.getElementById("query");
const limitInput = document.getElementById("limit");
const filenameInput = document.getElementById("filename");
const runButton = document.getElementById("run");
const pipelineStatus = document.getElementById("pipelineStatus");
const stepsEl = document.getElementById("steps");
const stepsMeta = document.getElementById("stepsMeta");
const resultPanel = document.getElementById("resultPanel");
const resultBody = document.getElementById("resultBody");
const resultMeta = document.getElementById("resultMeta");
const runsBody = document.getElementById("runsBody");
const runsMeta = document.getElementById("runsMeta");

const form = document.getElementById("form");
const promptInput = document.getElementById("prompt");
const submitButton = document.getElementById("submit");
const checkButton = document.getElementById("check");
const statusEl = document.getElementById("status");
const connectionPanel = document.getElementById("connection");
const connectionBody = document.getElementById("connectionBody");
const connectionMeta = document.getElementById("connectionMeta");
const callsPanel = document.getElementById("calls");
const callsBody = document.getElementById("callsBody");
const callsMeta = document.getElementById("callsMeta");
const answerEl = document.getElementById("answer");
const answerMeta = document.getElementById("answerMeta");
const copyButton = document.getElementById("copy");

const PIPELINE = ["search", "summarize", "save_to_file"];
const STEP_TITLES = {
    search: "ищет разделы документации",
    summarize: "сводит найденное в тезисы",
    save_to_file: "пишет сводку файлом",
};
const STATES = { waiting: "ждёт", started: "идёт", ok: "готово", failed: "ошибка" };

const runLabel = runButton.textContent;
const submitLabel = submitButton.textContent;

let pipelineController = null;
let controller = null;
let answerText = "";
let callCount = 0;
let steps = {};

// --- мелочи ----------------------------------------------------------------

function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

function placeholder(text) {
    const span = document.createElement("span");
    span.className = "placeholder";
    span.textContent = text;
    return span;
}

function setMeta(node, text, isError = false) {
    node.textContent = text;
    node.classList.toggle("error", isError);
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

// Один поток SSE на все запросы: кадры разбираются здесь, смысл им придают handlers.
async function streamSse(url, body, signal, handlers) {
    const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
    });

    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { value, done } = await reader.read();
        if (done) return;

        buffer += decoder.decode(value, { stream: true });

        let boundary;
        while ((boundary = buffer.indexOf("\n\n")) !== -1) {
            const { event, data } = parseFrame(buffer.slice(0, boundary));
            buffer = buffer.slice(boundary + 2);
            if (!data) continue;

            const payload = JSON.parse(data);
            if (event === "error") {
                throw new Error(payload);
            }
            handlers[event]?.(payload);
        }
    }
}

// --- шаги пайплайна --------------------------------------------------------

function resetSteps() {
    steps = {};
    for (const tool of PIPELINE) {
        steps[tool] = { tool, status: "waiting" };
    }
    renderSteps();
}

function artifactDetails(artifactId) {
    const details = element("details", "artifact");
    details.append(element("summary", "", `что в артефакте #${artifactId}`));
    const body = element("pre", "call-result", "загружаем...");
    details.append(body);

    details.addEventListener(
        "toggle",
        async () => {
            if (!details.open) return;
            try {
                const response = await fetch(`/api/artifacts/${artifactId}`);
                const payload = await response.json();
                body.textContent = JSON.stringify(payload, null, 2);
            } catch (error) {
                body.textContent = error.message;
            }
        },
        { once: true },
    );

    return details;
}

function stepFacts(step) {
    const handle = step.handle || {};
    const facts = [];

    if (step.tool === "search" && handle.hits) {
        facts.push(`разделов: ${handle.hits.length} · ${handle.strategy} · ${handle.total_chars} символов`);
        for (const hit of handle.hits) {
            facts.push(`${hit.number}. ${hit.path} — «${hit.heading}», ${hit.chars} символов`);
        }
    }
    if (step.tool === "summarize" && handle.bullets) {
        facts.push(`тезисов: ${handle.bullets.length} · модель ${handle.model}`);
        for (const bullet of handle.bullets) {
            const refs = bullet.refs.length ? ` [${bullet.refs.join(", ")}]` : "";
            facts.push(`— ${bullet.text}${refs}`);
        }
    }
    if (step.tool === "save_to_file" && handle.path) {
        facts.push(`${handle.path} · ${handle.file_bytes} байт`);
        facts.push(`sha256 файла: ${handle.file_sha256}`);
    }

    return facts;
}

function renderSteps() {
    const cards = PIPELINE.map((tool, index) => {
        const step = steps[tool];
        const card = element("article", `step ${step.status}`);

        const head = element("div", "step-head");
        head.append(
            element("code", "step-name", `${index + 1}. ${tool}`),
            element("span", "badge " + step.status, STATES[step.status]),
        );
        if (step.elapsed_ms !== undefined) {
            head.append(element("span", "meta", `${step.elapsed_ms} мс`));
        }
        card.append(head, element("p", "step-title", STEP_TITLES[tool]));

        const from = step.artifact_in ? `#${step.artifact_in}` : "запрос";
        if (step.status === "ok") {
            card.append(element("div", "step-flow", `${from} → артефакт #${step.artifact_out}`));
            for (const line of stepFacts(step)) {
                card.append(element("div", "step-fact", line));
            }
            card.append(artifactDetails(step.artifact_out));
        } else if (step.status === "failed") {
            card.append(element("div", "step-error", step.error));
        }

        return card;
    });

    stepsEl.replaceChildren(...cards);
}

function applyStepFrame(frame) {
    const step = steps[frame.tool];
    if (!step) return;

    Object.assign(step, frame);
    renderSteps();
}

// Режим «по шагам»: кадр вызова инструмента — это тот же шаг, только от модели.
function applyToolCall(call) {
    if (!PIPELINE.includes(call.name)) return;

    const handle = call.structured || {};
    applyStepFrame({
        tool: call.name,
        status: call.is_error ? "failed" : "ok",
        elapsed_ms: call.elapsed_ms,
        artifact_in: call.arguments.artifact_id ?? null,
        artifact_out: handle.artifact_id,
        error: call.is_error ? call.result : null,
        handle,
    });
}

async function showFile(filename) {
    const view = element("pre", "call-result file-view", "загружаем...");
    resultBody.append(view);

    try {
        const response = await fetch(`/api/files/${encodeURIComponent(filename)}`);
        view.textContent = response.ok ? await response.text() : `Файл не отдался: ${response.status}`;
    } catch (error) {
        view.textContent = error.message;
    }
}

function showResult(file, note) {
    resultPanel.hidden = false;
    resultBody.replaceChildren();

    if (note) {
        resultBody.append(element("p", "step-title", note));
    }

    if (!file) {
        resultBody.append(placeholder("Файла нет: цепочка не дошла до последнего шага"));
        return;
    }

    const facts = element("dl", "facts");
    for (const [term, value] of [
        ["файл", file.path],
        ["размер", `${file.bytes} байт`],
        ["sha256", file.sha256],
    ]) {
        facts.append(element("dt", "", term), element("dd", "", value));
    }

    const button = element("button", "copy", "Показать файл");
    button.type = "button";
    button.addEventListener("click", () => {
        button.disabled = true;
        showFile(file.path.split("/").pop());
    });

    resultBody.append(facts, button);
}

// --- прогоны ---------------------------------------------------------------

function renderRuns(runs) {
    if (!runs.length) {
        runsBody.replaceChildren(placeholder("Прогонов пока не было"));
        return;
    }

    const table = element("table", "jobs");
    const header = element("tr");
    for (const title of ["#", "режим", "запрос", "шаги", "статус", "начат"]) {
        header.append(element("th", "", title));
    }
    table.append(header);

    for (const run of runs) {
        const chain = run.steps
            .map((step) => `${step.tool} ${step.status === "ok" ? `${step.elapsed_ms} мс` : step.status}`)
            .join(" → ");

        const row = element("tr", run.status === "failed" ? "job-error" : "");
        row.append(
            element("td", "", String(run.run_id)),
            element("td", "", run.mode === "auto" ? "цепочкой" : "по шагам"),
            element("td", "", run.query),
            element("td", "mono", chain || "—"),
            element("td", "", run.status),
            element("td", "", formatTime(run.started_at)),
        );
        table.append(row);
    }

    runsBody.replaceChildren(table);
}

async function loadRuns() {
    try {
        const response = await fetch("/api/runs?limit=8");
        const payload = await response.json();
        if (payload.error) throw new Error(payload.error);

        renderRuns(payload.runs);
        setMeta(runsMeta, `последних: ${payload.runs.length} · обновлено ${formatTime(new Date().toISOString())}`);
    } catch (error) {
        runsBody.replaceChildren(placeholder("Прогоны недоступны: MCP-сервер не отвечает"));
        setMeta(runsMeta, error.message, true);
    }
}

// --- запуск пайплайна ------------------------------------------------------

function mode() {
    return document.querySelector("input[name=mode]:checked").value;
}

function setPipelineBusy(busy) {
    runButton.textContent = busy ? "Остановить" : runLabel;
    runButton.classList.toggle("stop", busy);
}

async function runPipeline(body) {
    let note = "";

    await streamSse("/api/pipeline", body, pipelineController.signal, {
        step: applyStepFrame,
        tool: applyToolCall,
        message: (text) => {
            note += text;
        },
        mcp: (info) => showConnection(info),
        pipeline: (result) => {
            setMeta(
                stepsMeta,
                `прогон #${result.run_id} · один вызов run_pipeline · ${result.elapsed_ms} мс`,
                result.status === "failed",
            );
            showResult(result.file, result.error ? `Цепочка встала на ${result.failed_at}: ${result.error}` : "");
        },
        done: (payload) => {
            setMeta(
                stepsMeta,
                `модель: раундов ${payload.rounds} · вызовов ${payload.calls} · ${payload.finish_reason}`,
                payload.finish_reason === "max_rounds",
            );
            const saved = steps.save_to_file;
            showResult(
                saved.status === "ok"
                    ? {
                          path: saved.handle.path,
                          bytes: saved.handle.file_bytes,
                          sha256: saved.handle.file_sha256,
                      }
                    : null,
                note.trim(),
            );
        },
    });
}

pipelineForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (pipelineController) {
        pipelineController.abort();
        return;
    }

    const query = queryInput.value.trim();
    if (!query) {
        setMeta(pipelineStatus, "Введите запрос", true);
        return;
    }

    const body = {
        query,
        limit: Number(limitInput.value),
        filename: filenameInput.value.trim() || null,
        mode: mode(),
    };

    pipelineController = new AbortController();
    setPipelineBusy(true);
    resetSteps();
    resultPanel.hidden = true;
    setMeta(stepsMeta, "");
    setMeta(
        pipelineStatus,
        body.mode === "auto" ? "Цепочка идёт одним вызовом..." : "Цепочку собирает модель...",
    );

    try {
        await runPipeline(body);
        setMeta(pipelineStatus, "Готово");
    } catch (error) {
        if (error.name === "AbortError") {
            setMeta(pipelineStatus, "Остановлено");
        } else {
            setMeta(pipelineStatus, error.message, true);
        }
    } finally {
        pipelineController = null;
        setPipelineBusy(false);
        loadRuns();
    }
});

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
    for (const [term, value] of [
        ["команда", info.command],
        ["сервер", info.server],
        ["протокол", info.protocol],
        ["возможности", info.capabilities.join(", ")],
    ]) {
        facts.append(element("dt", "", term), element("dd", "", value));
    }
    connectionBody.append(facts);

    const list = element("div", "tools");
    for (const tool of info.tools) {
        const card = element("article", "tool");
        card.append(
            element("code", "tool-name", `${tool.name}(${argsOf(tool.input_schema)})`),
            element("p", "tool-description", tool.description.split("\n")[0]),
        );
        list.append(card);
    }
    connectionBody.append(list);
}

function setStatus(text, isError = false) {
    setMeta(statusEl, text, isError);
}

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
        await streamSse("/api/ask", { prompt }, controller.signal, {
            mcp: showConnection,
            tool: (call) => {
                addCall(call);
                if (PIPELINE.includes(call.name)) loadRuns();
            },
            message: (text) => setAnswer(answerText + text),
            done: (payload) => {
                const parts = [`раундов: ${payload.rounds}`, `вызовов: ${payload.calls}`];
                if (payload.finish_reason) parts.push(`finish_reason: ${payload.finish_reason}`);
                setMeta(answerMeta, parts.join(" · "));
            },
        });
        setStatus("Готово");
    } catch (error) {
        setStatus(error.name === "AbortError" ? "Остановлено" : error.message, error.name !== "AbortError");
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
        if (payload.error) throw new Error(payload.error);

        showConnection(payload);
        setStatus(`Соединение установлено, инструментов: ${payload.tools.length}`);
    } catch (error) {
        connectionPanel.hidden = false;
        connectionBody.replaceChildren(placeholder("Соединение не установлено"));
        setMeta(connectionMeta, "");
        setStatus(error.message, true);
    }
});

for (const [container, input] of [
    [document.getElementById("queries"), queryInput],
    [document.getElementById("examples"), promptInput],
]) {
    container.addEventListener("click", (event) => {
        if (event.target.classList.contains("chip")) {
            input.value = event.target.textContent;
            input.focus();
        }
    });
}

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

document.getElementById("refreshRuns").addEventListener("click", loadRuns);

resetSteps();
loadRuns();
