const el = (id) => document.getElementById(id);

const sayForm = el("sayForm");
const questionInput = el("question");
const modePick = el("mode");
const gateInput = el("gate");
const sendButton = el("send");
const sayStatus = el("sayStatus");
const restartButton = el("restart");
const sessionMeta = el("sessionMeta");
const stream = el("stream");
const stateBody = el("stateBody");
const stateMeta = el("stateMeta");
const memoryBody = el("memoryBody");
const chunkPanel = el("chunkPanel");
const chunkMeta = el("chunkMeta");
const chunkBody = el("chunkBody");
const runBody = el("runBody");
const runMeta = el("runMeta");
const runButton = el("runAll");
const sweepBody = el("sweepBody");
const sweepMeta = el("sweepMeta");
const sweepButton = el("runSweep");
const corpusBody = el("corpusBody");
const corpusMeta = el("corpusMeta");
const buildButton = el("build");

let state = null;
let session = null;
let turns = 0;

const spaced = (value) =>
    Math.round(value).toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
const escaped = (text) =>
    String(text).replace(/[&<>"]/g, (char) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char]);
const share = (value) => (value === null || value === undefined ? "—" : Number(value).toFixed(2));
const score = share;

// Вердикт сверки красит цитату: подкреплённая зелёным, выдуманная красным,
// сбитый адрес — отдельно, потому что лечится он не тем же самым.
const VERDICT_TONE = {
    "дословно": "ok",
    "с правкой": "ok",
    "не из той выдержки": "warn",
    "нет такой выдержки": "bad",
    "не найдена": "bad",
    "цитаты нет": "bad",
};

function table(header, rows, className = "jobs") {
    const head = header.map((name) => `<th>${name}</th>`).join("");
    const body = rows
        .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
        .join("");
    return `<table class="${className}"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function fillSelect(node, items, selected) {
    node.innerHTML = items
        .map((item) => `<option value="${item.name}"${item.name === selected ? " selected" : ""}>${escaped(item.title)}</option>`)
        .join("");
}

async function readFrames(response, handle) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const frames = buffer.split("\n\n");
        buffer = frames.pop();

        for (const frame of frames) {
            const event = (frame.match(/^event: (.+)$/m) || [])[1];
            const payload = JSON.parse(
                frame.split("\n").filter((line) => line.startsWith("data: ")).map((line) => line.slice(6)).join("\n"),
            );
            handle(event, payload);
        }
    }
}

// --- память задачи -------------------------------------------------------------

// Подсветка «изменилось последним ходом» живёт здесь, а не в разметке самого
// пункта: пункт один и тот же, а новым он бывает только один ход.
function renderState(data, applied) {
    const fresh = new Set((applied?.added || []).map((item) => item.id));
    const gone = applied?.dropped || [];

    if (!data.goal && !data.size) {
        stateMeta.textContent = "пусто";
        stateBody.innerHTML = `<span class="placeholder">Режим без памяти задачи: условия разговора нигде не накапливаются.</span>`;
        return;
    }

    stateMeta.textContent = `${data.size} ${data.size === 1 ? "пункт" : "пунктов"}`;

    const goal = data.goal
        ? `<div class="state-goal${applied?.goal_set ? " fresh" : ""}">
               <span class="state-label">цель диалога</span>
               <p>${escaped(data.goal.text)}</p>
               <span class="inactive">поставлена на ходе ${data.goal.turn}${data.goal.why ? `, смена: ${escaped(data.goal.why)}` : ""}</span>
           </div>`
        : `<div class="state-goal"><span class="state-label">цель диалога</span><p class="inactive">ещё не названа</p></div>`;

    const blocks = state.sections
        .map((section) => {
            const items = data.sections[section] || [];
            if (!items.length) return "";
            const rows = items
                .map((item) => `<li class="${fresh.has(item.id) ? "fresh" : ""}"><span class="inactive mono">${item.id}</span> ${escaped(item.text)}</li>`)
                .join("");
            return `<div class="state-section">
                <span class="state-label">${escaped(section)}</span>
                <ul class="state-list">${rows}</ul>
            </div>`;
        })
        .join("");

    const kept = applied?.kept_goal
        ? `<p class="route-error">код не дал сменить цель: ${escaped(applied.kept_goal)}</p>`
        : "";
    const dropped = gone.length
        ? `<p class="hint">снято: ${gone.map((item) => escaped(item.text)).join("; ")}</p>`
        : "";
    const refused = (applied?.refused || []).length
        ? `<p class="hint">не записано: ${applied.refused.map((item) => `${escaped(item.text)} <span class="inactive">(${escaped(item.reason)})</span>`).join("; ")}</p>`
        : "";

    stateBody.innerHTML = goal + blocks + kept + dropped + refused;
}

function renderMemory() {
    const mode = modePick.value;
    const flags = state.memory[mode];
    memoryBody.innerHTML = `
        <dl class="facts">
            <dt>история</dt><dd>${flags.history ? `последние ${state.defaults.tail} сообщений дословно` : `<span class="inactive">не помнит</span>`}</dd>
            <dt>память задачи</dt><dd>${flags.tracked ? `цель и до ${state.defaults.cap} пунктов на раздел` : `<span class="inactive">не ведёт</span>`}</dd>
            <dt>поисковый запрос</dt><dd>${flags.history ? "реплика разрешается по разговору" : `<span class="inactive">реплика как есть</span>`}</dd>
        </dl>
        <p class="hint">${escaped(state.section_about[state.sections[1]])} — раздел «${escaped(state.sections[1])}»; его пункты попадают в каждый следующий запрос.</p>`;
}

// --- ход -----------------------------------------------------------------------

function claimCard(item) {
    const tone = VERDICT_TONE[item.verdict] || "warn";
    const where = item.path
        ? `<span class="chunk-link mono" data-chunk="${item.chunk_id}" data-start="${item.start}" data-end="${item.end}">${escaped(item.path)} · ${escaped(item.section)}</span>`
        : `<span class="inactive">источник [${item.source ?? "—"}] не найден</span>`;

    return `<div class="claim">
        <p class="claim-text">${escaped(item.text)}</p>
        <blockquote class="claim-quote ${tone}">${escaped(item.quote || "цитаты нет")}</blockquote>
        <div class="claim-foot">
            ${where}
            <span class="chip-static ${tone}">${escaped(item.verdict)}</span>
        </div>
    </div>`;
}

function turnShell(number, question) {
    return `<div class="turn" id="turn-${number}">
        <div class="bubble mine"><span class="inactive">${number}</span> ${escaped(question)}</div>
        <p class="searched" id="searched-${number}"><span class="inactive">ищем...</span></p>
        <div class="bubble theirs">
            <pre class="result" id="text-${number}"></pre>
            <div id="claims-${number}"></div>
            <div class="answer-foot" id="foot-${number}"></div>
        </div>
    </div>`;
}

function renderSearched(number, payload) {
    const resolved = payload.resolve;
    const context = payload.context;
    const gate = payload.gate;

    const asked = resolved.changed
        ? `искали по: <span class="mono">${escaped(resolved.standalone)}</span>`
        : `искали по самой реплике`;
    const found = context
        ? `пул ${context.pool} → ${context.kept} выдержек, лучшая оценка ${share(context.best_relevance)}`
        : "ничего не нашлось";
    const verdict = gate.passed
        ? ""
        : ` · <span class="chip-static bad">порог: ${escaped(gate.reason)}</span>`;

    el(`searched-${number}`).innerHTML = `${asked} <span class="inactive">·</span> ${found}${verdict}`;
}

function renderTurn(number, data) {
    el(`text-${number}`).textContent = data.text;

    const box = el(`claims-${number}`);
    const checked = data.verify ? data.verify.items : [];

    if (data.refusal) {
        box.innerHTML = `<div class="claim refusal">
            <p class="claim-text">Отказ — ${escaped(data.refusal)}.</p>
            <p class="hint">${escaped(data.clarify || "уточнения нет")}</p>
        </div>`;
    } else if (checked.length) {
        const broken = data.structured && data.structured.broken
            ? `<p class="route-error">${escaped(data.structured.broken)}</p>`
            : "";
        box.innerHTML = broken + checked.map(claimCard).join("");
    } else {
        box.innerHTML = "";
    }

    const verified = data.verify && data.verify.claims
        ? `<span class="chip-static ${data.verify.exact === data.verify.claims ? "ok" : "warn"}">дословных ${data.verify.exact} из ${data.verify.claims}</span> `
        : "";
    const cited = data.paths.length
        ? data.paths.map((path) => `<span class="chip-static ok mono">${escaped(path)}</span>`).join(" ")
        : `<span class="inactive">источников нет</span>`;
    const cost = data.refusal && !data.prompt_tokens
        ? "0 токенов: к модели не ходили"
        : `${spaced(data.prompt_tokens)} + ${spaced(data.stage_tokens)} на этапы · ${spaced(data.completion_tokens)} ответа · ${data.seconds} с`;

    el(`foot-${number}`).innerHTML =
        `${verified}${cited} <span class="inactive">· ${escaped(cost)}${data.memory_tokens ? `, память ≈${data.memory_tokens} ток.` : ""}</span>`;

    renderState(data.state, data.track ? data.track.applied : null);
}

async function startSession(mode) {
    const response = await fetch("/api/session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, title: "" }),
    });
    const data = await response.json();
    if (!response.ok) {
        sayStatus.classList.add("error");
        sayStatus.textContent = data.detail;
        return;
    }

    session = data.session_id;
    turns = 0;
    stream.innerHTML = `<span class="placeholder">Диалог начат в режиме «${escaped(data.title)}». Первая реплика задаёт цель разговора.</span>`;
    sessionMeta.textContent = `${data.title} · ходов 0`;
    renderState(data.state, null);
    renderMemory();
}

async function say(event) {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;
    if (!session) await startSession(modePick.value);

    const number = turns + 1;
    if (!turns) stream.innerHTML = "";
    stream.insertAdjacentHTML("beforeend", turnShell(number, question));
    el(`turn-${number}`).scrollIntoView({ behavior: "smooth", block: "nearest" });

    sendButton.disabled = true;
    questionInput.value = "";
    sayStatus.classList.remove("error");
    sayStatus.textContent = "Разрешаем ссылки, ищем в базе, отвечаем...";

    const gate = gateInput.value.trim();

    try {
        const response = await fetch(`/api/session/${session}/say`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                gate_threshold: gate === "" ? null : Number(gate),
            }),
        });

        await readFrames(response, (event, payload) => {
            if (event === "context") {
                renderSearched(number, payload);
            } else if (event === "delta") {
                el(`text-${number}`).textContent += payload;
            } else if (event === "turn") {
                turns = payload.number;
                renderTurn(number, payload);
                sessionMeta.textContent = `${payload.title} · ходов ${turns}`;
            } else if (event === "error") {
                sayStatus.classList.add("error");
                sayStatus.textContent = payload;
            }
        });

        if (!sayStatus.classList.contains("error")) {
            sayStatus.textContent = "Память задачи обновляется после ответа, отдельным вызовом. Цель код сменить не даст, пока модель не объяснит, зачем.";
        }
    } finally {
        sendButton.disabled = false;
        questionInput.focus();
    }
}

async function showChunk(chunkId, start, end) {
    const response = await fetch(`/api/chunk/${chunkId}`, { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) return;

    // Подсветка клеится по смещениям сверки, поэтому экранируются три куска по
    // отдельности: вставлять тег в уже экранированную строку значило бы искать
    // позицию в тексте, длина которого изменилась.
    const body = start >= 0 && end > start
        ? `${escaped(data.text.slice(0, start))}<mark>${escaped(data.text.slice(start, end))}</mark>${escaped(data.text.slice(end))}`
        : escaped(data.text);

    chunkPanel.hidden = false;
    chunkMeta.textContent = start >= 0 ? `#${data.chunk_id}, цитата подсвечена` : `#${data.chunk_id}`;
    chunkBody.innerHTML = `
        <dl class="facts">
            <dt>файл</dt><dd class="mono">${escaped(data.path)}</dd>
            <dt>документ</dt><dd>${escaped(data.title)}</dd>
            <dt>раздел</dt><dd>${escaped(data.section)}</dd>
            <dt>источник</dt><dd>${escaped(state.source_titles[data.source] || data.source)}</dd>
            <dt>диапазон</dt><dd>${spaced(data.start_char)}–${spaced(data.end_char)}, ${spaced(data.chars)} символов, ${data.tokens} токенов</dd>
            <dt>соседи</dt><dd>${data.neighbours
                .map((item) =>
                    item.chunk_id === data.chunk_id
                        ? `<span class="chip-static ok">#${item.chunk_id} ${escaped(item.section)}</span>`
                        : `<span class="chip-static chunk-link" data-chunk="${item.chunk_id}">#${item.chunk_id} ${escaped(item.section)}</span>`,
                )
                .join(" ")}</dd>
        </dl>
        <pre class="file-view">${body}</pre>`;
    chunkPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// --- сценарии --------------------------------------------------------------------

function renderDialogues(data) {
    runMeta.textContent = `${data.dialogues.length} сценария, ${data.dialogues.reduce((sum, item) => sum + item.length, 0)} реплик`;
    runBody.innerHTML =
        data.dialogues
            .map(
                (dialogue) => `<div class="strategy-block">
                    <div class="strategy-head">
                        ${escaped(dialogue.title)}
                        <span class="inactive">${dialogue.length} реплик · цель: ${escaped(dialogue.goal)}</span>
                    </div>
                    ${table(
                        ["#", "Род", "Реплика", "Ждём файл", "Рамка"],
                        dialogue.says.map((say, position) => [
                            String(position + 1),
                            `<span class="chip-static">${escaped(say.kind_title)}</span>`,
                            escaped(say.text),
                            say.expect_paths.map((path) => `<span class="mono">${escaped(path)}</span>`).join(" ") || `<span class="inactive">—</span>`,
                            [
                                ...say.must_say.map((word) => `есть «${escaped(word)}»`),
                                ...say.must_not_say.map((word) => `нет «${escaped(word)}»`),
                            ].join(", ") || `<span class="inactive">—</span>`,
                        ]),
                    )}
                </div>`,
            )
            .join("") +
        `<p class="hint">«Прогнать» проведёт оба сценария во всех трёх режимах — 81 ход, у каждого свой поиск, ответ и обновление памяти, — и позовёт судью на каждый ход. Это самая дорогая кнопка на странице.</p>`;
}

function renderRun(report) {
    const modes = state.modes.map((item) => item.name);
    const titles = state.mode_titles;
    const folded = report.by_mode;

    const main = table(
        ["Режим", "Источники есть", "Цитата дословна", "Ссылка разрешена", "Рамка соблюдена", "Судья цели", "На 2 балла", "Пробы", "Отказов"],
        modes.map((mode) => [
            `<strong>${escaped(titles[mode])}</strong>`,
            share(folded[mode].sources),
            share(folded[mode].exact),
            `${share(folded[mode].resolved)} <span class="inactive">(${folded[mode].resolved_count} из ${folded[mode].referential})</span>`,
            share(folded[mode].framed),
            score(folded[mode].goal),
            `${folded[mode].goal_two} из ${folded[mode].goal_scored}`,
            score(folded[mode].probe_goal),
            `${folded[mode].refused} из ${folded[mode].turns}`,
        ]),
    );

    const kinds = table(
        ["Род реплики", ...modes.map((mode) => titles[mode])],
        Object.entries(report.by_kind).map(([kind, row]) => [
            `${escaped(state.kind_titles[kind])} <span class="inactive">(${Object.values(row)[0].turns})</span>`,
            ...modes.map((mode) => score(row[mode]?.goal)),
        ]),
    );

    const cost = table(
        ["", ...modes.map((mode) => titles[mode])],
        [
            ["Токенов запроса", ...modes.map((mode) => spaced(folded[mode].prompt_tokens))],
            ["Токенов ответа", ...modes.map((mode) => spaced(folded[mode].completion_tokens))],
            ["Токенов на этапы", ...modes.map((mode) => spaced(folded[mode].stage_tokens))],
            ["Из них память, оценка", ...modes.map((mode) => spaced(folded[mode].memory_tokens))],
            ["Секунд на ход", ...modes.map((mode) => folded[mode].seconds.toFixed(1))],
            ["Переписано реплик", ...modes.map((mode) => String(folded[mode].rewritten))],
        ],
    );

    const goals = table(
        ["Сценарий", "Режим", "Цель в памяти", "Сверка", "Код не дал сменить"],
        report.runs
            .filter((run) => state.memory[run.summary.mode].tracked)
            .map((run) => [
                `<span class="mono">${escaped(run.summary.dialogue)}</span>`,
                escaped(titles[run.summary.mode]),
                escaped(run.summary.goal_text || "—"),
                `<span class="chip-static ${run.summary.goal_verdict === "та же" ? "ok" : "warn"}">${escaped(run.summary.goal_verdict)}</span>`,
                String(run.summary.kept_goal),
            ]),
    );

    const misses = report.runs.flatMap((run) =>
        run.turns
            .filter((turn) => turn.resolved_ok === false || turn.framed_ok === false || turn.goal === 0)
            .map((turn) => [
                `<span class="mono">${escaped(run.summary.dialogue)}</span>`,
                escaped(titles[run.summary.mode]),
                `#${turn.number}`,
                escaped(turn.question),
                [
                    turn.resolved_ok === false ? "ссылка не разрешена" : "",
                    turn.missing.length ? `нет «${turn.missing.map(escaped).join("», «")}»` : "",
                    turn.forbidden.length ? `есть «${turn.forbidden.map(escaped).join("», «")}»` : "",
                    turn.goal === 0 ? "мимо цели" : "",
                ].filter(Boolean).join("; "),
                `<span class="inactive">${escaped(turn.why)}</span>`,
            ]),
    );

    runMeta.textContent = `${report.runs.length} прогонов, ${folded[modes[0]].turns} ходов на режим`;
    runBody.innerHTML = `
        <div class="strategy-block"><div class="strategy-head">три режима памяти</div>${main}</div>
        <div class="strategy-block"><div class="strategy-head">судья цели по роду реплики</div>${kinds}</div>
        <div class="strategy-block"><div class="strategy-head">цель к последнему ходу</div>${goals}</div>
        <div class="strategy-block"><div class="strategy-head">цена памяти, на ход</div>${cost}</div>
        <div class="strategy-block"><div class="strategy-head">где просадки</div>${
            misses.length
                ? table(["Сценарий", "Режим", "Ход", "Реплика", "Что не так", "Судья"], misses)
                : `<p class="hint">Ни одной: все ссылки разрешены, рамки соблюдены, мимо цели ни одного хода.</p>`
        }</div>
        <p class="hint">«Ссылка разрешена» и «рамка соблюдена» считаются детекторами в коде. Судья видит цель сценария и зафиксированные пользователем ограничения — но не память агента: эталон обязан быть один для всех трёх режимов.</p>`;
}

async function runAll() {
    runButton.disabled = true;
    runBody.innerHTML = `<span class="placeholder">Проводим оба сценария в трёх режимах, ход за ходом, и зовём судью на каждый ход...</span>`;

    try {
        const response = await fetch("/api/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ gate_threshold: gateInput.value.trim() === "" ? null : Number(gateInput.value) }),
        });
        const data = await response.json();
        if (!response.ok) {
            runBody.innerHTML = `<span class="route-error">${escaped(data.detail)}</span>`;
            return;
        }
        renderRun(data);
    } finally {
        runButton.disabled = false;
    }
}

async function runSweep() {
    sweepButton.disabled = true;
    sweepBody.innerHTML = `<span class="placeholder">Проводим сценарий при четырёх порогах отказа, без судьи...</span>`;

    try {
        const response = await fetch("/api/sweep", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
        });
        const data = await response.json();
        if (!response.ok) {
            sweepBody.innerHTML = `<span class="route-error">${escaped(data.detail)}</span>`;
            return;
        }

        sweepMeta.textContent = `сценарий «${data.key}», ${data.rows[0].turns} реплик`;
        sweepBody.innerHTML =
            table(
                ["Порог", "Отказов", "Из них кодом", "Из них моделью", "Источники есть", "Цитата дословна", "Выдумано"],
                data.rows.map((row) => [
                    row.threshold.toFixed(2),
                    `${row.refused} из ${row.turns}`,
                    String(row.by_code),
                    String(row.by_model),
                    share(row.sources),
                    share(row.exact),
                    String(row.fabricated),
                ]),
            ) +
            `<p class="hint">Слой в коде отказывает по числу до вызова модели, слой в промпте — прочитав выдержки. Первый с порогом считается, второй нет: его доля показывает, сколько ходов отказались бы и без порога.</p>`;
    } finally {
        sweepButton.disabled = false;
    }
}

// --- корпус ----------------------------------------------------------------------

function renderCorpus(data) {
    const rows = Object.entries(data.corpus.by_source).map(([source, slot]) => [
        escaped(data.source_titles[source] || source),
        slot.files,
        spaced(slot.chars),
        Math.round(slot.chars / 1800),
    ]);
    rows.push([
        "<strong>всего</strong>",
        `<strong>${data.corpus.files}</strong>`,
        `<strong>${spaced(data.corpus.chars)}</strong>`,
        `<strong>${Math.round(data.corpus.pages)}</strong>`,
    ]);

    const built = data.built;
    corpusMeta.textContent = built ? `${spaced(built.chunks)} чанков, ${built.model}` : "индекс не собран";

    const indexBlock = built
        ? table(
              ["Чанков", "Медиана, ток.", "p95", "Макс.", "Чанкинг", "Эмбеддинг", "FTS5", "Векторы"],
              [[
                  spaced(built.chunks),
                  Math.round(built.median_tokens),
                  spaced(built.p95_tokens),
                  spaced(built.max_tokens),
                  `${built.chunk_seconds.toFixed(1)} с`,
                  `${built.embed_seconds.toFixed(1)} с`,
                  `${built.fts_seconds.toFixed(2)} с`,
                  `${(built.vector_bytes / 1e6).toFixed(2)} МБ`,
              ]],
          )
        : `<span class="placeholder">Индекса нет. Нажмите «Собрать заново» или запустите <span class="mono">python day25/scenarios.py build</span>.</span>`;

    corpusBody.innerHTML = `
        <div class="strategy-block"><div class="strategy-head">корпус</div>${table(["Источник", "Файлов", "Символов", "Страниц"], rows)}</div>
        <div class="strategy-block"><div class="strategy-head">индекс · ${escaped(data.strategy.title)}</div>${indexBlock}</div>
        <p class="hint">Из корпуса исключены четыре папки: своя, day22, day23 и day24 — их README разбирают контрольные вопросы вместе с ответами.</p>
        ${data.skipped.map((note) => `<p class="route-error">мимо корпуса: ${escaped(note)}</p>`).join("")}`;
}

async function build() {
    buildButton.disabled = true;
    corpusBody.innerHTML = `<span class="placeholder">Режем корпус, считаем векторы, строим FTS5...</span>`;

    try {
        const response = await fetch("/api/build", { method: "POST" });
        await readFrames(response, (event, payload) => {
            if (event === "error") {
                corpusBody.innerHTML = `<span class="route-error">${escaped(payload)}</span>`;
            }
        });
    } finally {
        buildButton.disabled = false;
        await loadState();
    }
}

// --- состояние -------------------------------------------------------------------

async function loadState() {
    state = await (await fetch("/api/state", { cache: "no-store" })).json();

    renderCorpus(state);
    renderDialogues(state);

    fillSelect(modePick, state.modes, state.defaults.mode);
    gateInput.placeholder = share(state.defaults.gate_threshold);
    renderMemory();

    // Примеры — первые реплики обоих сценариев: они же ставят цель, и по ним
    // сразу видно, с чего разговор вообще начинается.
    el("examples").innerHTML = state.dialogues
        .map((dialogue) => `<button type="button" class="chip">${escaped(dialogue.says[0].text)}</button>`)
        .join("");

    if (!session) await startSession(modePick.value);
}

// --- события -----------------------------------------------------------------------

document.addEventListener("click", (event) => {
    const link = event.target.closest(".chunk-link");
    if (link) {
        showChunk(Number(link.dataset.chunk), Number(link.dataset.start ?? -1), Number(link.dataset.end ?? -1));
    }
});

el("examples").addEventListener("click", (event) => {
    if (event.target.classList.contains("chip")) {
        questionInput.value = event.target.textContent;
        sayForm.requestSubmit();
    }
});

el("closeChunk").addEventListener("click", () => {
    chunkPanel.hidden = true;
});

modePick.addEventListener("change", () => startSession(modePick.value));
restartButton.addEventListener("click", () => startSession(modePick.value));
sayForm.addEventListener("submit", say);
buildButton.addEventListener("click", build);
runButton.addEventListener("click", runAll);
sweepButton.addEventListener("click", runSweep);

loadState();
