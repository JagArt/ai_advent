const serversBody = document.getElementById("serversBody");
const serversMeta = document.getElementById("serversMeta");
const catalogPanel = document.getElementById("catalogPanel");
const catalogBody = document.getElementById("catalogBody");
const catalogMeta = document.getElementById("catalogMeta");

const flowForm = document.getElementById("flowForm");
const queryInput = document.getElementById("query");
const limitInput = document.getElementById("limit");
const bulletsInput = document.getElementById("bullets");
const nameInput = document.getElementById("name");
const runButton = document.getElementById("run");
const flowStatus = document.getElementById("flowStatus");
const stepsEl = document.getElementById("steps");
const stepsMeta = document.getElementById("stepsMeta");
const verdictPanel = document.getElementById("verdictPanel");
const verdictBody = document.getElementById("verdictBody");
const verdictMeta = document.getElementById("verdictMeta");
const resultPanel = document.getElementById("resultPanel");
const resultBody = document.getElementById("resultBody");
const resultMeta = document.getElementById("resultMeta");
const routesBody = document.getElementById("routesBody");
const routesMeta = document.getElementById("routesMeta");

const form = document.getElementById("form");
const promptInput = document.getElementById("prompt");
const submitButton = document.getElementById("submit");
const statusEl = document.getElementById("status");
const callsPanel = document.getElementById("calls");
const callsBody = document.getElementById("callsBody");
const callsMeta = document.getElementById("callsMeta");
const answerEl = document.getElementById("answer");
const answerMeta = document.getElementById("answerMeta");
const copyButton = document.getElementById("copy");

const STATES = { waiting: "ждёт", started: "идёт", ok: "готово", failed: "ошибка", refused: "отказ" };
const runLabel = runButton.textContent;
const submitLabel = submitButton.textContent;

let contract = [];
let slots = [];
let extras = [];
let causes = {};
let flowController = null;
let controller = null;
let answerText = "";
let callCount = 0;

// --- мелочи ----------------------------------------------------------------

function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

function placeholder(text) {
    return element("span", "placeholder", text);
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

function size(bytes) {
    return bytes > 1024 ? `${(bytes / 1024).toFixed(1)} КБ` : `${bytes} Б`;
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

// --- серверы и каталог -----------------------------------------------------

function renderServers(info) {
    const cards = info.servers.map((server) => {
        const card = element("article", server.online ? "server" : "server offline");
        const head = element("div", "step-head");
        head.append(
            element("code", "step-name", server.name),
            element("span", `badge ${server.online ? "online" : "offline"}`, server.online ? "на связи" : "нет связи"),
            element("span", "meta", server.transport === "http" ? "streamable HTTP" : "stdio"),
        );
        if (server.online) head.append(element("span", "meta", `${server.elapsed_ms} мс`));
        card.append(head, element("p", "step-title", server.about));

        const facts = element("dl", "facts");
        const rows = server.online
            ? [
                  ["адрес", server.endpoint],
                  ["жизнь", server.lifetime],
                  ["сервер", `${server.server}, протокол ${server.protocol}`],
                  ["инструменты", server.tools.join(", ")],
              ]
            : [
                  ["адрес", server.endpoint],
                  ["причина", server.error || "не подключился"],
              ];
        for (const [term, value] of rows) {
            facts.append(element("dt", "", term), element("dd", "", value));
        }
        card.append(facts);
        return card;
    });

    serversBody.replaceChildren(...cards);

    const online = info.servers.filter((server) => server.online).length;
    setMeta(
        serversMeta,
        `на связи ${online} из ${info.servers.length} · инструментов ${info.tools}`,
        online < info.servers.length,
    );
}

function renderCatalog(info) {
    catalogPanel.hidden = false;
    const collisions = Object.entries(info.collisions);
    setMeta(
        catalogMeta,
        collisions.length
            ? `одноимённых: ${collisions.map(([name, list]) => `${name} у ${list.length} серверов`).join(", ")}`
            : "одноимённых инструментов нет",
    );

    const table = element("table", "jobs");
    const header = element("tr");
    for (const title of ["инструмент", "сервер", "что делает", "отдаёт", "ждёт ссылку"]) {
        header.append(element("th", "", title));
    }
    table.append(header);

    for (const entry of info.catalog) {
        const row = element("tr", entry.collides ? "collides" : "");
        const name = element("td", "mono");
        name.append(element("code", "", entry.qualified));
        if (entry.collides) name.append(element("span", "badge warn", "одноимённый"));

        const needs =
            entry.needs
                .map((need) => `${need.arg} ← ${need.kind}${need.optional ? "?" : ""}`)
                .join(", ") || "—";

        row.append(
            name,
            element("td", "", entry.server),
            element("td", "", entry.description.split("\n")[0]),
            element("td", "mono", entry.produces),
            element("td", "mono", needs),
        );
        table.append(row);
    }

    catalogBody.replaceChildren(table);
}

async function loadServers() {
    setMeta(serversMeta, "подключаемся...");
    try {
        const response = await fetch("/api/servers");
        const info = await response.json();
        if (info.error) throw new Error(info.error);

        causes = info.causes || {};
        renderServers(info);
        renderCatalog(info);
    } catch (error) {
        serversBody.replaceChildren(placeholder("Реестр не собрался"));
        setMeta(serversMeta, error.message, true);
    }
}

// --- шаги ------------------------------------------------------------------

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

// Превью артефакта человеку: у каждого вида важно своё, поэтому по виду и разбираем.
function handleFacts(handle) {
    const preview = handle.preview || {};
    const facts = [];
    const lines = (items, format) =>
        (items || []).forEach((item) => {
            if (typeof item === "string") facts.push(item.startsWith("…") ? item : `— ${item}`);
            else facts.push(format(item));
        });

    if (handle.kind === "commits") {
        facts.push(`коммитов: ${preview.count} · ${preview.strategy}`);
        lines(preview.commits, (commit) => `${commit.short} ${commit.subject}`);
    }
    if (handle.kind === "changed") {
        facts.push(`файлов: ${(preview.files || []).length} · +${preview.insertions} −${preview.deletions}`);
        facts.push(`папки для поиска: ${(preview.filter || []).join(", ")}`);
    }
    if (handle.kind === "found") {
        facts.push(
            `разделов: ${preview.count} · ${preview.strategy}` +
                (preview.filter ? ` · в папках ${preview.filter.join(", ")}` : " · без фильтра"),
        );
        lines(preview.hits, (hit) => `${hit.number}. ${hit.path} — «${hit.heading}», ${hit.chars} симв.`);
    }
    if (handle.kind === "summary") {
        facts.push(`тезисов: ${(preview.bullets || []).length} · модель ${preview.model}`);
        lines(preview.bullets, (bullet) => `— ${bullet.text}${bullet.refs?.length ? ` [${bullet.refs.join(", ")}]` : ""}`);
    }
    if (handle.kind === "file") {
        facts.push(`${preview.path} · ${preview.bytes} байт`);
        facts.push(`sha256: ${preview.sha256}`);
    }
    if (handle.kind === "entry") {
        facts.push(`запись #${preview.id} в журнале хранилища, всего записей ${preview.total}`);
    }
    if (handle.kind === "check") {
        facts.push(preview.ok ? "sha256 на диске совпал с артефактом" : "sha256 не сошёлся");
        facts.push(`на диске: ${preview.sha256_on_disk}`);
        facts.push(`в журнале: ${preview.in_journal ? "есть" : "нет"}`);
    }

    return facts;
}

function stepCard(step, position) {
    const card = element("article", `step ${step.status}${step.extra ? " extra" : ""}`);
    const head = element("div", "step-head");
    head.append(
        element("code", "step-name", `${position}. ${step.tool}`),
        element("span", "badge " + step.status, STATES[step.status] || step.status),
    );
    if (step.server) head.append(element("span", "badge server-tag", step.server));
    if (step.elapsed_ms !== undefined) head.append(element("span", "meta", `${step.elapsed_ms} мс`));
    if (step.extra) head.append(element("span", "badge warn", "вне контракта"));
    card.append(head);

    if (step.why) card.append(element("p", "step-title", step.why));
    // Имя, которым позвали, показывается только если оно не совпало с маршрутом:
    // голое `search` реестр развёл сам или отказался это делать.
    if (step.requested && step.requested !== step.tool) {
        const how = causes[step.resolution] || { resolved: "разрешилось однозначно" }[step.resolution] || step.resolution;
        card.append(element("div", "step-fact", `позвали ${step.requested} → ${how}`));
    }

    if (step.status === "ok" && step.handle) {
        const refs = (step.refs || [])
            .map((ref) => `${ref.arg} ← #${ref.artifact_id}.${ref.field} (${size(ref.bytes)})`)
            .join(", ");
        card.append(
            element(
                "div",
                "step-flow",
                `${refs || "аргументы запроса"} → артефакт #${step.handle.artifact_id} (${step.handle.kind})`,
            ),
        );
        for (const line of handleFacts(step.handle)) {
            card.append(element("div", "step-fact", line));
        }
        card.append(artifactDetails(step.handle.artifact_id));
    } else if (step.error) {
        card.append(element("div", "step-error", step.error));
    }

    return card;
}

function renderSteps() {
    const cards = slots.map((step, index) => stepCard(step, index + 1));
    extras.forEach((step) => cards.push(stepCard(step, "+")));
    stepsEl.replaceChildren(...cards);
}

function resetSteps() {
    slots = contract.map((step) => ({ ...step, status: "waiting" }));
    extras = [];
    renderSteps();
}

function applyStep(frame) {
    const slot = slots[frame.position - 1];
    if (!slot) return;
    // В кадре `tool` — короткое имя инструмента на сервере, а в карточке нужно полное.
    Object.assign(slot, frame, {
        tool: frame.qualified || slot.tool,
        status: frame.stage === "started" ? "started" : frame.status,
    });
    renderSteps();
}

// Режим «агентом»: вызов встаёт в слот контракта, а лишний — отдельной карточкой.
function applyCall(frame) {
    const slot = slots.find((step) => step.tool === frame.qualified && step.status === "waiting");
    if (slot) {
        Object.assign(slot, frame, { tool: frame.qualified });
    } else {
        extras.push({ ...frame, tool: frame.qualified || frame.requested, extra: true });
    }
    renderSteps();
}

// --- проверка порядка ------------------------------------------------------

function chips(list, className) {
    const box = element("div", "chips");
    if (!list.length) box.append(placeholder("—"));
    for (const name of list) box.append(element("code", `chip-static ${className || ""}`, name));
    return box;
}

function showVerdict(verdict, mode) {
    if (!verdict) {
        verdictPanel.hidden = true;
        return;
    }

    verdictPanel.hidden = false;
    setMeta(
        verdictMeta,
        verdict.order_ok ? "порядок совпал с контрактом" : "порядок отклонился от контракта",
        !verdict.order_ok,
    );

    const facts = element("dl", "facts");
    const rows = [
        ["контракт", null, chips(verdict.contract)],
        ["как вышло", null, chips(verdict.actual, verdict.order_ok ? "ok" : "")],
        ["совпало", `${verdict.matched} из ${verdict.contract.length}`],
        ["серверы", verdict.servers.join(" → ")],
        ["вызовов", `${verdict.calls} · отказов реестра ${verdict.refused} · сбоев ${verdict.failed}`],
    ];
    if (verdict.missing.length) rows.push(["не позвали", verdict.missing.join(", ")]);
    if (verdict.extra.length) rows.push(["лишние", verdict.extra.join(", ")]);
    if (verdict.resolved_bare) rows.push(["имя без префикса", `${verdict.resolved_bare} раз`]);
    if (Object.keys(verdict.causes).length) {
        rows.push([
            "причины отказов",
            Object.entries(verdict.causes)
                .map(([cause, count]) => `${causes[cause] || cause}: ${count}`)
                .join(", "),
        ]);
    }
    rows.push(["режим", mode === "flow" ? "контракт проходил реестр" : "инструменты выбирала модель"]);

    for (const [term, value, node] of rows) {
        facts.append(element("dt", "", term));
        const dd = element("dd", "");
        if (node) dd.append(node);
        else dd.textContent = value;
        facts.append(dd);
    }

    verdictBody.replaceChildren(facts);
}

// --- отчёт -----------------------------------------------------------------

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

function showResult(file, check) {
    if (!file) {
        resultPanel.hidden = true;
        return;
    }

    resultPanel.hidden = false;
    resultBody.replaceChildren();
    setMeta(resultMeta, check ? (check.ok ? "sha256 сверен" : "sha256 не сошёлся") : "", check && !check.ok);

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

// --- журнал маршрутизации --------------------------------------------------

function renderRoutes(routes) {
    if (!routes.length) {
        routesBody.replaceChildren(placeholder("Вызовов пока не было"));
        return;
    }

    const table = element("table", "jobs");
    const header = element("tr");
    for (const title of ["прогон", "позвали", "сервер", "инструмент", "как разрешилось", "итог", "мс", "когда"]) {
        header.append(element("th", "", title));
    }
    table.append(header);

    for (const route of routes) {
        const row = element("tr", route.status === "ok" ? "" : "job-error");
        const resolution =
            route.status === "refused"
                ? causes[route.resolution] || route.resolution
                : { qualified: "полное имя", resolved: "имя без префикса" }[route.resolution] || route.resolution;
        const result = route.status === "ok" ? `артефакт #${route.artifact_out}` : STATES[route.status] || route.status;

        row.append(
            element("td", "", `#${route.run_id}`),
            element("td", "mono", route.requested),
            element("td", "", route.server || "—"),
            element("td", "mono", route.tool || "—"),
            element("td", "", resolution),
            element("td", "", result),
            element("td", "", String(route.elapsed_ms)),
            element("td", "", formatTime(route.at)),
        );

        if (route.error) {
            const note = element("tr", "job-error");
            const cell = element("td", "route-error", route.error);
            cell.colSpan = 8;
            note.append(cell);
            table.append(row, note);
        } else {
            table.append(row);
        }
    }

    routesBody.replaceChildren(table);
}

async function loadRoutes() {
    try {
        const response = await fetch("/api/routes?limit=20");
        const payload = await response.json();
        causes = payload.causes || causes;
        renderRoutes(payload.routes);
        setMeta(routesMeta, `последних вызовов: ${payload.routes.length}`);
    } catch (error) {
        routesBody.replaceChildren(placeholder("Журнал недоступен"));
        setMeta(routesMeta, error.message, true);
    }
}

// --- запуск флоу -----------------------------------------------------------

function mode() {
    return document.querySelector("input[name=mode]:checked").value;
}

function setFlowBusy(busy) {
    runButton.textContent = busy ? "Остановить" : runLabel;
    runButton.classList.toggle("stop", busy);
}

async function runFlow(body) {
    let note = "";
    const saved = { file: null, check: null };

    await streamSse("/api/flow", body, flowController.signal, {
        mcp: (info) => {
            renderServers(info);
            renderCatalog(info);
            setMeta(stepsMeta, `прогон #${info.run_id}`);
        },
        step: applyStep,
        call: (frame) => {
            applyCall(frame);
            if (frame.handle?.kind === "file") saved.file = frame.handle.preview;
            if (frame.handle?.kind === "check") saved.check = frame.handle.preview;
        },
        message: (text) => {
            note += text;
            setAnswer(note);
        },
        flow: (result) => {
            setMeta(
                stepsMeta,
                `прогон #${result.run_id} · контрактом · ${result.elapsed_ms} мс` +
                    (result.error ? ` · встал на ${result.failed_at}` : ""),
                result.status === "failed",
            );
            showVerdict(result.verdict, "flow");
            showResult(result.file, result.check);
        },
        done: (payload) => {
            setMeta(
                stepsMeta,
                `прогон #${payload.run_id} · агентом · раундов ${payload.rounds} · вызовов ${payload.calls} · ${payload.finish_reason}`,
                payload.finish_reason === "max_rounds",
            );
            setMeta(answerMeta, `раундов: ${payload.rounds} · вызовов: ${payload.calls}`);
            showVerdict(payload.verdict, "agent");
            showResult(saved.file, saved.check);
        },
    });
}

flowForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    if (flowController) {
        flowController.abort();
        return;
    }

    const query = queryInput.value.trim();
    if (!query) {
        setMeta(flowStatus, "Введите тему отчёта", true);
        return;
    }

    const body = {
        query,
        limit: Number(limitInput.value),
        bullets: Number(bulletsInput.value),
        name: nameInput.value.trim() || null,
        mode: mode(),
    };

    flowController = new AbortController();
    setFlowBusy(true);
    resetSteps();
    verdictPanel.hidden = true;
    resultPanel.hidden = true;
    setAnswer("");
    setMeta(
        flowStatus,
        body.mode === "flow" ? "Реестр проходит контракт..." : "Инструменты выбирает модель...",
    );

    try {
        await runFlow(body);
        setMeta(flowStatus, "Готово");
    } catch (error) {
        setMeta(flowStatus, error.name === "AbortError" ? "Остановлено" : error.message, error.name !== "AbortError");
    } finally {
        flowController = null;
        setFlowBusy(false);
        loadRoutes();
    }
});

// --- чат -------------------------------------------------------------------

function setStatus(text, isError = false) {
    setMeta(statusEl, text, isError);
}

function setBusy(busy) {
    submitButton.textContent = busy ? "Остановить" : submitLabel;
    submitButton.classList.toggle("stop", busy);
}

function setAnswer(text) {
    answerText = text;
    answerEl.textContent = text;
    copyButton.hidden = !text;
}

function addCall(frame) {
    callCount += 1;
    callsPanel.hidden = false;
    setMeta(callsMeta, `вызовов: ${callCount}`);

    const card = element("article", frame.status === "ok" ? "call" : "call error");
    const row = element("div", "call-row");
    row.append(
        element("code", "call-head", `${frame.requested}(${JSON.stringify(frame.arguments)})`),
        element("span", "meta", `${frame.server || "—"} · ${frame.elapsed_ms} мс`),
    );
    card.append(
        row,
        element(
            "pre",
            "call-result",
            frame.error || JSON.stringify(frame.handle, null, 2),
        ),
    );
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
    setStatus("Подключаемся ко всем серверам реестра...");
    setAnswer("");
    setMeta(answerMeta, "");
    callCount = 0;
    callsBody.replaceChildren();
    callsPanel.hidden = true;
    setMeta(callsMeta, "");

    try {
        await streamSse("/api/ask", { prompt }, controller.signal, {
            mcp: (info) => {
                renderServers(info);
                renderCatalog(info);
                setStatus(`Реестр собран, инструментов ${info.tools}`);
            },
            call: addCall,
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
        loadRoutes();
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

document.getElementById("refreshServers").addEventListener("click", loadServers);
document.getElementById("refreshRoutes").addEventListener("click", loadRoutes);

// Контракт статичен и приезжает мгновенно: шаги видны до того, как что-то запущено.
fetch("/api/contract")
    .then((response) => response.json())
    .then((payload) => {
        contract = payload.steps.map((step) => ({
            tool: step.tool,
            server: step.server,
            why: step.why,
        }));
        resetSteps();
    });

loadServers();
loadRoutes();
