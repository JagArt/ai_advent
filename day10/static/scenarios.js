const paramsEl = document.getElementById("params");
const scenariosEl = document.getElementById("scenarios");
const statusEl = document.getElementById("status");
const controlsEl = document.getElementById("report-controls");
const rawToggle = document.getElementById("raw-toggle");
const copyButton = document.getElementById("copy");
const reportEl = document.getElementById("report");
const placeholderEl = document.getElementById("placeholder");
const transcriptEl = document.getElementById("transcript");
const renderedEl = document.getElementById("report-rendered");
const rawEl = document.getElementById("report-raw");

const RAW_KEY = "day10.report.raw";

// Реплики приходят из stderr с пометкой, кто говорит: `say()` в scenarios.py.
const SPEAKERS = { "вы: ": "user", "агент: ": "agent" };

// Отчёт — это markdown, и он же уходит в README: строки хранятся как пришли, а
// отрендеренный вид собирается из них заново.
let lines = [];
let running = null;
let controller = null;
let pending = null;

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

// Пока идёт прогон, кнопка своего сценария работает на остановку, остальные гаснут:
// два прогона разом упрутся в рейт-лимиты и испортят замер друг другу.
function setBusy(name) {
    running = name;
    for (const card of scenariosEl.querySelectorAll(".scenario")) {
        const own = card.dataset.name === name;
        const button = card.querySelector(".scenario-run");
        card.classList.toggle("active", Boolean(name) && own);
        button.disabled = Boolean(name) && !own;
        button.textContent = Boolean(name) && own ? "Остановить" : "Прогнать";
        button.classList.toggle("stop", Boolean(name) && own);
    }
}

function scenarioCard(scenario) {
    const card = document.createElement("article");
    card.className = "scenario";
    card.dataset.name = scenario.name;

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

    const button = document.createElement("button");
    button.type = "button";
    button.className = "button scenario-run";
    button.textContent = "Прогнать";
    button.addEventListener("click", () => {
        if (running === scenario.name) {
            controller?.abort();
            return;
        }
        run(scenario).catch((error) => setStatus(error.message, true));
    });

    card.append(title, about, cost, command, button);
    return card;
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

async function run(scenario) {
    lines = [];
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
        // сценария — это тоже данные, и их незачем стирать.
        showReport();
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
            if (event === "done") {
                showReport();
                setStatus(doneStatus(scenario, JSON.parse(data)));
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

function showReport() {
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

function doneStatus(scenario, payload) {
    if (payload.exit_code !== 0) {
        return `${scenario.title}: сценарий вышел с кодом ${payload.exit_code}`
            + `, строк отчёта ${payload.lines}`;
    }
    const minutes = payload.seconds >= 60
        ? `${Math.floor(payload.seconds / 60)} мин ${Math.round(payload.seconds % 60)} с`
        : `${payload.seconds} с`;
    return `${scenario.title}: готово за ${minutes}, ${payload.lines}`
        + ` ${plural(payload.lines, "строка", "строки", "строк")} отчёта`;
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

        if (line.startsWith("### ")) {
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

    scenariosEl.replaceChildren(...scenarios.scenarios.map(scenarioCard));
    rawToggle.checked = localStorage.getItem(RAW_KEY) === "true";
    setView();

    if (scenarios.busy) {
        // Прогон идёт в другой вкладке: свой запустить всё равно не выйдет, и
        // честнее сказать это сразу, а не после отказа сервера.
        setStatus("Прогон уже идёт — вероятно, в другой вкладке", true);
    }
}

init().catch((error) => {
    setStatus(error.message, true);
});
