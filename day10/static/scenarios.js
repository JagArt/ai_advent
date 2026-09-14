const paramsEl = document.getElementById("params");
const scenariosEl = document.getElementById("scenarios");
const statusEl = document.getElementById("status");
const compareButton = document.getElementById("compare");
const controlsEl = document.getElementById("report-controls");
const rawToggle = document.getElementById("raw-toggle");
const copyButton = document.getElementById("copy");
const reportEl = document.getElementById("report");
const placeholderEl = document.getElementById("placeholder");
const transcriptEl = document.getElementById("transcript");
const renderedEl = document.getElementById("report-rendered");
const rawEl = document.getElementById("report-raw");

const RAW_KEY = "day10.report.raw";

// Прогоны сессии: на каждый сценарий не больше одного, новый затирает старый.
// Чистый прогон — точка отсчёта для всех остальных, он дороже прочих и переживает
// сессию, поэтому лежит отдельно и в localStorage.
const RUNS_KEY = "day10.runs";
const CLEAN_KEY = "day10.run.clean";
const CLEAN = "clean";

// Максимум судьи за контрольный вопрос: тот же, что в scenarios.py.
const MAX_SCORE = 2;

// Реплики приходят из stderr с пометкой, кто говорит: `say()` в scenarios.py.
const SPEAKERS = { "вы: ": "user", "агент: ": "agent" };

// Отчёт — это markdown, и он же уходит в README: строки хранятся как пришли, а
// отрендеренный вид собирается из них заново.
let lines = [];
let running = null;
let controller = null;
let pending = null;
let result = null;

// Отпечаток замера с сервера: по нему видно, что перенесённый прогон посчитан по
// другой версии диалога и сравнивать его напрямую нельзя.
let fingerprint = "";
let runs = {};
// Что прогнали в этой вкладке: остальное — принесённое из прошлой сессии.
let ownRuns = new Set();
let cards = new Map();

function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", isError);
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

// Прогоны читаются из двух хранилищ, потому что живут разное время: стратегии — до
// закрытия вкладки, чистый прогон — до следующей правки диалога.
function loadRuns() {
    runs = { ...parse(sessionStorage.getItem(RUNS_KEY)) };
    const clean = parse(localStorage.getItem(CLEAN_KEY));
    if (clean && clean.payload) {
        runs[CLEAN] = clean;
    }
}

function parse(raw) {
    if (!raw) {
        return null;
    }
    try {
        return JSON.parse(raw);
    } catch {
        // Запись оставил другой формат страницы: чинить её нечем, а прогон
        // повторяется кнопкой.
        return null;
    }
}

function persist() {
    const session = {};
    for (const [name, record] of Object.entries(runs)) {
        if (name !== CLEAN) {
            session[name] = record;
        }
    }
    sessionStorage.setItem(RUNS_KEY, JSON.stringify(session));

    if (runs[CLEAN]) {
        localStorage.setItem(CLEAN_KEY, JSON.stringify(runs[CLEAN]));
    } else {
        localStorage.removeItem(CLEAN_KEY);
    }
}

function storeRun(name, payload, reportLines) {
    runs[name] = {
        payload,
        lines: reportLines,
        saved_at: new Date().toISOString(),
    };
    ownRuns.add(name);
    persist();
    paintCards();
}

function dropRun(name) {
    delete runs[name];
    ownRuns.delete(name);
    persist();
    paintCards();
}

function saved() {
    // Порядок как на странице: чистый прогон первый, дальше стратегии.
    return [...cards.keys()].filter((name) => runs[name]).map((name) => runs[name]);
}

function score(payload) {
    const checks = payload.checks || [];
    const total = checks.reduce((sum, check) => sum + Math.max(check.score, 0), 0);
    return { total, max: MAX_SCORE * checks.length };
}

function when(record) {
    const moment = new Date(record.saved_at);
    const today = new Date().toDateString() === moment.toDateString();
    const time = moment.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });
    return today ? time : `${moment.toLocaleDateString("ru-RU")} ${time}`;
}

function describeRun(name, record) {
    const { total, max } = score(record.payload);
    const parts = [
        `прогон ${when(record)}`,
        `судья ${total} из ${max}`,
        `вход ${record.payload.input_tokens} ток.`,
        `$${record.payload.total_cost.toFixed(6)}`,
    ];
    if (name === CLEAN && !ownRuns.has(name)) {
        // Через сессии переносится только чистый прогон, и это надо сказать прямо:
        // иначе непонятно, откуда в свежей вкладке взялись числа.
        parts.push("перенесён из прошлой сессии");
    }
    return parts.join(", ");
}

function stale(record) {
    return Boolean(fingerprint) && record.payload.fingerprint !== fingerprint;
}

// Пока идёт прогон, кнопка своего сценария работает на остановку, остальные гаснут:
// два прогона разом упрутся в рейт-лимиты и испортят замер друг другу.
function setBusy(name) {
    running = name;
    for (const card of cards.values()) {
        const own = card.name === name;
        card.root.classList.toggle("active", Boolean(name) && own);
        card.run.disabled = Boolean(name) && !own;
        card.run.textContent = Boolean(name) && own ? "Остановить" : "Прогнать";
        card.run.classList.toggle("stop", Boolean(name) && own);
    }
    paintCards();
}

function paintCards() {
    for (const card of cards.values()) {
        const record = runs[card.name];
        const busy = Boolean(running);

        if (record) {
            card.state.textContent = describeRun(card.name, record);
            card.state.classList.toggle("scenario-state-stale", stale(record));
            if (stale(record)) {
                card.state.textContent += " — другая версия диалога";
            }
        } else {
            card.state.textContent = "прогона в сессии нет";
            card.state.classList.remove("scenario-state-stale");
        }

        card.show.hidden = !record;
        card.drop.hidden = !record;
        card.show.disabled = busy;
        card.drop.disabled = busy;
    }

    const count = saved().length;
    // Сравнивать нечего, пока в сессии меньше двух прогонов: одна колонка — это
    // отчёт самого прогона, он уже есть.
    compareButton.disabled = count < 2 || Boolean(running);
    compareButton.textContent = count ? `Сравнить прогоны (${count})` : "Сравнить прогоны";
}

function scenarioCard(scenario) {
    const root = document.createElement("article");
    root.className = "scenario";
    root.dataset.name = scenario.name;

    const title = document.createElement("h2");
    title.className = "scenario-title";
    title.textContent = scenario.title;

    const about = document.createElement("p");
    about.className = "scenario-about";
    about.textContent = scenario.about;

    const cost = document.createElement("span");
    cost.className = "scenario-cost";
    // Число запросов считает сервер по самим сценариям: прогон платный, и знать
    // это лучше до нажатия, а не по счёту от DeepSeek.
    cost.textContent = `${scenario.requests} ${plural(scenario.requests, "запрос", "запроса", "запросов")} к модели`;

    const command = document.createElement("code");
    command.className = "scenario-command";
    command.textContent = `python day10/scenarios.py ${scenario.name}`;

    const run = document.createElement("button");
    run.type = "button";
    run.className = "button scenario-run";
    run.textContent = "Прогнать";
    run.addEventListener("click", () => {
        if (running === scenario.name) {
            controller?.abort();
            return;
        }
        start(scenario).catch((error) => setStatus(error.message, true));
    });

    const state = document.createElement("span");
    state.className = "scenario-state";

    const show = document.createElement("button");
    show.type = "button";
    show.className = "button button-quiet scenario-saved";
    show.textContent = "Отчёт";
    show.hidden = true;
    show.addEventListener("click", () => showStored(scenario));

    const drop = document.createElement("button");
    drop.type = "button";
    drop.className = "button button-quiet scenario-saved";
    drop.textContent = "Убрать";
    drop.hidden = true;
    drop.addEventListener("click", () => {
        dropRun(scenario.name);
        setStatus(`${scenario.title}: прогон убран из сессии`);
    });

    const footer = document.createElement("div");
    footer.className = "scenario-footer";
    footer.append(state, show, drop);

    root.append(title, about, cost, command, run, footer);
    cards.set(scenario.name, { name: scenario.name, root, run, state, show, drop });
    return root;
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

async function start(scenario) {
    lines = [];
    result = null;
    rawEl.textContent = "";
    renderedEl.replaceChildren();
    transcriptEl.replaceChildren();
    pending = null;
    placeholderEl.hidden = false;
    controlsEl.hidden = true;
    controller = new AbortController();
    // Пока прогон идёт, на экране стенограмма: отчёт печатается только в конце, и
    // до тех пор и рендер, и сырой текст были бы пустой панелью.
    transcriptEl.hidden = false;
    rawEl.hidden = true;
    renderedEl.hidden = true;
    setBusy(scenario.name);
    setStatus(`${scenario.title}: прогон начался, ${scenario.requests} запросов к модели`);

    try {
        const response = await fetch(`/api/scenario/${scenario.name}`, {
            method: "POST",
            signal: controller.signal,
        });
        if (response.status === 409) {
            throw new Error("Прогон уже идёт — дождитесь конца или остановите его");
        }
        if (!response.ok) {
            throw new Error(`Сервер вернул ${response.status}`);
        }
        await read(response, scenario);
    } catch (error) {
        if (error.name === "AbortError") {
            setStatus(lines.length
                ? `Прогон остановлен, успело напечататься ${lines.length}`
                    + ` ${plural(lines.length, "строка", "строки", "строк")} отчёта`
                : "Прогон остановлен, до отчёта дело не дошло");
        } else {
            setStatus(error.message, true);
        }
        // Оборванный прогон оставляет на экране то, что успел напечатать: полтора
        // сценария — это тоже данные, и их незачем стирать. В сессию он не попадает:
        // сравнивать половину прогона не с чем.
        showReport(lines);
    } finally {
        // Прогон кончился — текущего хода больше нет, даже если стенограмма осталась.
        transcriptEl.querySelector(".progress-line.current")?.classList.remove("current");
        // Ход оборвали посреди ответа: пузырь с точками так и остался бы думать.
        pending?.parentElement?.remove();
        pending = null;
        controller = null;
        setBusy(null);
    }
}

async function read(response, scenario) {
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
            if (event === "progress") {
                // Ход работы идёт из stderr: пока сценарий считает, в stdout пусто,
                // и без этих строк страница выглядела бы повисшей.
                const { text } = JSON.parse(data);
                const said = speaker(text);
                if (said) {
                    addBubble(said.role, said.text);
                } else {
                    // В статусе только фазы прогона: целая реплика в одну строку
                    // всё равно не влезет, её место — в стенограмме.
                    addNote(text);
                    setStatus(`${scenario.title}: ${text}`);
                }
                placeholderEl.hidden = true;
                reportEl.scrollTop = reportEl.scrollHeight;
                continue;
            }
            if (event === "result") {
                // Машинный результат прогона: сам отчёт для сравнения не годится,
                // а этот payload сервер разберёт обратно и сведёт с другими.
                result = JSON.parse(data);
                continue;
            }
            if (event === "done") {
                const payload = JSON.parse(data);
                if (result) {
                    storeRun(scenario.name, result, lines);
                }
                showReport(lines);
                setStatus(doneStatus(scenario, payload));
                return;
            }

            addLine(JSON.parse(data));
        }
    }

    throw new Error("Поток оборвался");
}

// Сырой текст дописывается живьём, а таблицы собираются в конце: до последней
// строки неизвестно, где кончается блок, и перерисовывать его на каждой строке
// значит перерисовывать весь отчёт.
function addLine(text) {
    lines.push(text);
    rawEl.textContent += `${text}\n`;
}

// Реплика или строка хода работы — по пометке в начале. Прогресс вроде «судья:
// вопрос 3 из 5» на неё не похож: пометок ровно две, и обе про говорящего.
function speaker(text) {
    const mark = Object.keys(SPEAKERS).find((prefix) => text.startsWith(prefix));
    return mark ? { role: SPEAKERS[mark], text: text.slice(mark.length) } : null;
}

// Стенограмма копится, а не сменяет сама себя: по ней видно, о чём агента спросили,
// что он ответил и на каком ходу прогон оборвали.
function addNote(text) {
    const note = document.createElement("p");
    note.className = "progress-line";
    note.textContent = text;

    transcriptEl.querySelector(".progress-line.current")?.classList.remove("current");
    note.classList.add("current");
    transcriptEl.append(note);
}

function addBubble(role, text) {
    // Ответ приходит целиком и заметно позже вопроса: до него в стенограмме стоит
    // пузырь агента с точками — тот же приём, что в чате.
    if (role === "agent" && pending) {
        pending.textContent = text;
        pending = null;
        return;
    }

    const body = bubble(role);
    body.textContent = text;
    if (role === "user") {
        pending = bubble("agent");
        pending.append(thinking());
    }
}

function bubble(role) {
    const message = document.createElement("article");
    message.className = `message message-${role}`;

    const author = document.createElement("span");
    author.className = "message-author";
    author.textContent = role === "user" ? "Вы" : "Агент";

    const body = document.createElement("pre");
    body.className = "message-body";

    message.append(author, body);
    transcriptEl.append(message);
    return body;
}

function thinking() {
    const dots = document.createElement("span");
    dots.className = "thinking";
    dots.append(...Array.from({ length: 3 }, () => document.createElement("span")));
    return dots;
}

// Отчёт прогона, сохранённый прогон и сравнение — это один и тот же markdown,
// поэтому и показываются они одинаково.
function showReport(source) {
    lines = source;
    if (!lines.length) {
        // Отчёта нет — прогон оборвали в самом начале. Список ходов остаётся: это
        // всё, что от него осталось.
        return;
    }
    placeholderEl.hidden = true;
    controlsEl.hidden = false;
    // Отчёт встаёт на место стенограммы: то же место, но результат вместо хода.
    transcriptEl.hidden = true;
    renderedEl.replaceChildren(...render(lines));
    reportEl.scrollTop = 0;
    setView();
}

function showStored(scenario) {
    const record = runs[scenario.name];
    if (!record) {
        return;
    }
    transcriptEl.replaceChildren();
    showReport(record.lines);
    setStatus(`${scenario.title}: отчёт прогона от ${when(record)} из сессии`);
}

function doneStatus(scenario, payload) {
    if (payload.exit_code !== 0) {
        return `${scenario.title}: сценарий вышел с кодом ${payload.exit_code}`
            + `, строк отчёта ${payload.lines}`;
    }
    const minutes = payload.seconds >= 60
        ? `${Math.floor(payload.seconds / 60)} мин ${Math.round(payload.seconds % 60)} с`
        : `${payload.seconds} с`;
    return `${scenario.title}: готово за ${minutes}, ${payload.lines}`
        + ` ${plural(payload.lines, "строка", "строки", "строк")} отчёта`
        + (result ? ", прогон сохранён в сессии" : "");
}

// Мини-рендерер markdown: сценарии печатают только заголовки, таблицы и абзацы,
// поэтому полноценный разбор не нужен. Текст попадает в DOM через textContent —
// в отчёте лежат ответы модели и комментарии судьи, доверять им нельзя.
function render(source) {
    const nodes = [];
    let table = [];

    const flush = () => {
        if (table.length) {
            nodes.push(renderTable(table));
            table = [];
        }
    };

    for (const line of source) {
        if (line.startsWith("|")) {
            table.push(line);
            continue;
        }
        flush();

        if (line.startsWith("#### ")) {
            nodes.push(heading("h4", line.slice(5)));
        } else if (line.startsWith("### ")) {
            nodes.push(heading("h3", line.slice(4)));
        } else if (line.startsWith("## ")) {
            nodes.push(heading("h2", line.slice(3)));
        } else if (line.trim()) {
            const paragraph = document.createElement("p");
            paragraph.textContent = line;
            nodes.push(paragraph);
        }
    }

    flush();
    return nodes;
}

function heading(tag, text) {
    const node = document.createElement(tag);
    node.textContent = text;
    return node;
}

function renderTable(rows) {
    const table = document.createElement("table");
    table.className = "tokens-table report-table";
    const head = document.createElement("thead");
    const body = document.createElement("tbody");

    // Вторая строка блока — разделитель из дефисов: он и отличает заголовок от
    // обычных строк, а сам в таблицу не попадает.
    const separator = rows.length > 1 && cells(rows[1]).every((value) => /^-{3,}$/.test(value));
    rows.forEach((row, index) => {
        if (separator && index === 1) {
            return;
        }
        const header = separator && index === 0;
        const tr = document.createElement("tr");
        for (const value of cells(row)) {
            const td = document.createElement(header ? "th" : "td");
            td.textContent = value;
            tr.append(td);
        }
        (header ? head : body).append(tr);
    });

    table.append(head, body);
    return table;
}

// cell() в сценариях экранирует вертикальную черту как \|, потому что она есть в
// ответах модели: перед разбивкой строки экранированные черты прячутся, а после —
// возвращаются на место уже внутри ячейки.
function cells(row) {
    return row
        .replace(/\\\|/g, "\u0000")
        .split("|")
        .slice(1, -1)
        .map((value) => value.replace(/\u0000/g, "|").trim());
}

// Сравнение считается по сохранённым прогонам, а не прогоном заново: судья уже
// поставил оценки внутри каждого, и сводить их — работа на сотые доли секунды.
async function compare() {
    const records = saved();
    if (records.length < 2) {
        setStatus("Сравнивать пока нечего: нужно два прогона в сессии", true);
        return;
    }

    compareButton.disabled = true;
    setStatus(`Сравнение ${records.length} ${plural(records.length, "прогона", "прогонов", "прогонов")}…`);

    try {
        const response = await fetch("/api/compare", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ runs: records.map((record) => record.payload) }),
        });
        if (!response.ok) {
            throw new Error(`Сервер вернул ${response.status}`);
        }
        const payload = await response.json();
        transcriptEl.replaceChildren();
        showReport(payload.lines);
        setStatus(`Сравнение готово: ${records.map((record) => record.payload.label).join(", ")}`);
    } catch (error) {
        setStatus(`Сравнить не удалось: ${error.message}`, true);
    } finally {
        paintCards();
    }
}

compareButton.addEventListener("click", () => {
    compare().catch((error) => setStatus(error.message, true));
});

rawToggle.addEventListener("change", () => {
    localStorage.setItem(RAW_KEY, String(rawToggle.checked));
    setView();
});

function setView() {
    rawEl.hidden = !rawToggle.checked;
    renderedEl.hidden = rawToggle.checked;
}

copyButton.addEventListener("click", async () => {
    try {
        await navigator.clipboard.writeText(lines.join("\n"));
        setStatus(`Отчёт скопирован: ${lines.length} ${plural(lines.length, "строка", "строки", "строк")}`);
    } catch (error) {
        setStatus(`Скопировать не удалось: ${error.message}`, true);
    }
});

async function init() {
    const [defaults, scenarios] = await Promise.all([
        fetch("/api/defaults").then((response) => response.json()),
        fetch("/api/scenarios").then((response) => response.json()),
    ]);

    addParam(defaults.model);
    addParam(`окно ${defaults.window_messages} сообщ.`);
    addParam(`картотека до ${defaults.facts_limit} фактов`);
    addParam(defaults.peak ? "пиковый тариф ×2" : "непиковый тариф");

    fingerprint = scenarios.fingerprint || "";
    scenariosEl.replaceChildren(...scenarios.scenarios.map(scenarioCard));
    loadRuns();
    paintCards();
    rawToggle.checked = localStorage.getItem(RAW_KEY) === "true";
    setView();

    const count = saved().length;
    if (scenarios.busy) {
        // Прогон идёт в другой вкладке: свой запустить всё равно не выйдет, и
        // честнее сказать это сразу, а не после отказа сервера.
        setStatus("Прогон уже идёт — вероятно, в другой вкладке", true);
    } else if (count) {
        setStatus(`В сессии ${count} ${plural(count, "прогон", "прогона", "прогонов")}`
            + (count > 1 ? " — их можно сравнить" : ": для сравнения нужен второй"));
    } else {
        setStatus("Выберите сценарий");
    }
}

init().catch((error) => {
    setStatus(error.message, true);
});
