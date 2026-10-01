const el = (id) => document.getElementById(id);

const askForm = el("askForm");
const questionInput = el("question");
const retrieverPick = el("retriever");
const rewriterPick = el("rewriter");
const rerankerPick = el("reranker");
const poolInput = el("pool");
const topKInput = el("topK");
const thresholdInput = el("threshold");
const runButton = el("run");
const askStatus = el("askStatus");
const answersPanel = el("answersPanel");
const answersMeta = el("answersMeta");
const answersBox = el("answers");
const contextPanel = el("contextPanel");
const contextMeta = el("contextMeta");
const contextBody = el("contextBody");
const chunkPanel = el("chunkPanel");
const chunkMeta = el("chunkMeta");
const chunkBody = el("chunkBody");
const corpusBody = el("corpusBody");
const corpusMeta = el("corpusMeta");
const buildButton = el("build");
const questionsBody = el("questionsBody");
const questionsMeta = el("questionsMeta");
const questionsButton = el("runQuestions");
const matrixBody = el("matrixBody");
const matrixMeta = el("matrixMeta");
const matrixButton = el("runMatrix");
const sweepBody = el("sweepBody");
const sweepMeta = el("sweepMeta");
const sweepButton = el("runSweep");

let state = null;

const spaced = (value) =>
    Math.round(value).toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
const escaped = (text) =>
    String(text).replace(/[&<>"]/g, (char) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char]);
const share = (value) => (value === null || value === undefined ? "—" : Number(value).toFixed(2));
const score = (value) => (value === null || value === undefined ? "—" : Number(value).toFixed(2));

function table(header, rows, className = "jobs") {
    const head = header.map((name) => `<th>${name}</th>`).join("");
    const body = rows
        .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
        .join("");
    return `<table class="${className}"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function fillSelect(node, items) {
    node.innerHTML = items
        .map((item) => `<option value="${item.name}"${item.default ? " selected" : ""}>${escaped(item.title)}</option>`)
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

// --- вопрос ------------------------------------------------------------------

function answerCard(mode, title, note) {
    return `<div class="answer" id="answer-${mode}">
        <div class="answer-head">
            <span class="answer-title">${escaped(title)}</span>
            <span class="meta" id="cost-${mode}">${escaped(note)}</span>
        </div>
        <pre class="result" id="text-${mode}"></pre>
        <div class="answer-foot" id="foot-${mode}"></div>
    </div>`;
}

function renderCost(mode, data) {
    const stage = data.context && data.context.stage_tokens
        ? ` + ${spaced(data.context.stage_tokens)} на этапы`
        : "";
    el(`cost-${mode}`).textContent =
        `${spaced(data.prompt_tokens)}${stage} ток. · ${spaced(data.completion_tokens)} ответа · ${data.seconds} с`;

    const foot = el(`foot-${mode}`);
    if (!data.context) {
        foot.innerHTML = `<span class="inactive">контекста нет: вопрос ушёл к модели как есть</span>`;
        return;
    }

    const cited = data.cited.length
        ? data.cited_paths.map((path) => `<span class="chip-static ok mono">${escaped(path)}</span>`).join(" ")
        : `<span class="inactive">ответ не сослался ни на одну выдержку</span>`;
    foot.innerHTML = `<span class="inactive">подкреплено:</span> ${cited}`;
}

function sourceRow(item, fate) {
    return [
        item.number ? `<span class="chip-static">${item.number}</span>` : `<span class="inactive">—</span>`,
        score(item.relevance),
        item.place ?? "—",
        `<span class="mono">${escaped(item.path || item.hit?.path || "")}</span>`,
        `<span class="chunk-link" data-chunk="${item.chunk_id || item.hit?.chunk_id}">${escaped(item.section || item.hit?.section || "")}</span>`,
        fate,
    ];
}

function renderContextBlock(mode, data) {
    const title = state.mode_titles[mode];
    if (!data) {
        return `<div class="strategy-block">
            <div class="strategy-head">${escaped(title)}</div>
            <p class="hint">Контекста нет: вопрос ушёл к модели как есть.</p>
        </div>`;
    }

    const plan = data.plan;
    const rewrite = data.rewrite;
    const rewriteLine = rewrite.changed
        ? `<p class="hint">запрос: <span class="mono">${escaped(rewrite.query)}</span></p>`
        : "";
    const kept = (data.sources || []).map((source) => sourceRow(source, "дошло"));
    const dropped = (data.dropped_sources || []).map((item) => sourceRow(item, escaped(item.reason || "отсеяно")));

    return `<div class="strategy-block">
        <div class="strategy-head">
            ${escaped(title)}
            <span class="inactive">${escaped(plan.rewriter_title)} · ${escaped(plan.reranker_title)} · пул ${data.pool} → ${data.kept}, порог ${score(plan.threshold)}</span>
        </div>
        ${rewriteLine}
        ${table(["№", "Оценка", "Место", "Файл", "Раздел", "Судьба"], kept.concat(dropped))}
    </div>`;
}

function renderContexts(payload) {
    contextPanel.hidden = false;
    const modes = state.modes.filter((mode) => mode.name !== "plain");
    const kept = modes.reduce((sum, mode) => sum + ((payload.contexts[mode.name] || {}).kept || 0), 0);
    contextMeta.textContent = `${modes.length} конвейера, ${kept} выдержек дошло до промпта`;
    contextBody.innerHTML =
        modes.map((mode) => renderContextBlock(mode.name, payload.contexts[mode.name])).join("") +
        `<p class="hint">Номер слева — тот самый, которым ответ ссылается на выдержку. «Ниже порога», «потолок на файл» и «не вошёл в top-K» — три разные причины отсева, и путать их нельзя.</p>`;
}

function askPayload() {
    const threshold = thresholdInput.value.trim();
    return {
        question: questionInput.value.trim(),
        retriever: retrieverPick.value,
        rewriter: rewriterPick.value,
        reranker: rerankerPick.value,
        pool: Number(poolInput.value),
        top_k: Number(topKInput.value),
        threshold: threshold === "" ? null : Number(threshold),
    };
}

async function ask(event) {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    runButton.disabled = true;
    askStatus.classList.remove("error");
    askStatus.textContent = "Собираем контексты и спрашиваем все пять режимов...";

    answersPanel.hidden = false;
    answersMeta.textContent = `«${question}»`;
    answersBox.innerHTML = state.modes
        .map((mode) => answerCard(mode.name, mode.title, mode.name === "plain" ? "спрашиваем..." : "ищем..."))
        .join("");
    contextPanel.hidden = true;

    try {
        const response = await fetch("/api/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(askPayload()),
        });

        await readFrames(response, (event, payload) => {
            if (event === "context") {
                renderContexts(payload);
            } else if (event === "delta") {
                el(`text-${payload.mode}`).textContent += payload.content;
            } else if (event === "answer") {
                el(`text-${payload.mode}`).textContent = payload.text;
                renderCost(payload.mode, payload);
            } else if (event === "error") {
                askStatus.classList.add("error");
                askStatus.textContent = payload;
            }
        });

        if (!askStatus.classList.contains("error")) {
            askStatus.textContent = "Разница между колонками — это вклад переписывания и фильтра по отдельности, а не их сумма.";
        }
    } finally {
        runButton.disabled = false;
    }
}

async function showChunk(chunkId) {
    const response = await fetch(`/api/chunk/${chunkId}`, { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) return;

    chunkPanel.hidden = false;
    chunkMeta.textContent = `#${data.chunk_id}`;
    chunkBody.innerHTML = `
        <dl class="facts">
            <dt>файл</dt><dd class="mono">${escaped(data.path)}</dd>
            <dt>документ</dt><dd>${escaped(data.title)}</dd>
            <dt>раздел</dt><dd>${escaped(data.section)}</dd>
            <dt>источник</dt><dd>${escaped(state.source_titles[data.source] || data.source)}</dd>
            <dt>диапазон</dt><dd>${spaced(data.start_char)}–${spaced(data.end_char)}, ${spaced(data.chars)} символов, ${data.tokens} токенов</dd>
            <dt>sha256</dt><dd class="mono">${data.sha256.slice(0, 32)}…</dd>
            <dt>соседи</dt><dd>${data.neighbours
                .map((item) =>
                    item.chunk_id === data.chunk_id
                        ? `<span class="chip-static ok">#${item.chunk_id} ${escaped(item.section)}</span>`
                        : `<span class="chip-static chunk-link" data-chunk="${item.chunk_id}">#${item.chunk_id} ${escaped(item.section)}</span>`,
                )
                .join(" ")}</dd>
        </dl>
        <pre class="file-view">${escaped(data.text)}</pre>`;
    chunkPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// --- контрольный набор --------------------------------------------------------

function renderQuestionsSet(data) {
    questionsMeta.textContent = `${data.count} вопросов, написаны руками`;
    questionsBody.innerHTML =
        table(
            ["#", "Тип", "Вопрос", "Что должно быть в ответе", "Источники"],
            data.items.map((item) => [
                item.id,
                `<span class="chip-static">${escaped(item.kind)}</span>`,
                escaped(item.question),
                `<span class="inactive">${escaped(item.expect)}</span>`,
                item.sources.length
                    ? item.sources.map((path) => `<span class="mono">${escaped(path)}</span>`).join("<br>")
                    : `<span class="inactive">—</span>`,
            ]),
        ) +
        `<p class="hint">Ожидание и источники записаны заранее и те же, что в day22. «Прогнать» задаст каждый вопрос в пяти режимах и сверит ответы — это самая дорогая кнопка на странице.</p>`;
}

function renderQuestionsRun(report) {
    const modes = report.modes;
    const titles = report.mode_titles;
    const summary = table(
        ["Режим", "Факты", "Источник", "Точность", "Сослался", "Судья", "На 2", "Отказы", "Выдержек", "Токенов"],
        modes.map((mode) => {
            const row = report.summary[mode];
            return [
                `<strong>${escaped(titles[mode])}</strong>`,
                share(row.facts),
                share(row.sources),
                share(row.precision),
                share(row.cited),
                share(row.judge),
                `${row.judge_full} из ${row.questions}`,
                `${row.refused_outside} из ${row.outside}`,
                share(row.kept),
                spaced(row.total_prompt_tokens),
            ];
        }),
    );

    const kinds = Object.keys(report.by_kind[modes[0]]);
    const perKind = table(
        ["Тип вопроса", ...modes.map((mode) => titles[mode])],
        kinds.map((kind) => [
            `${escaped(kind)} <span class="inactive">(${report.by_kind[modes[0]][kind].questions})</span>`,
            ...modes.map((mode) => share(report.by_kind[mode][kind].judge)),
        ]),
    );

    const rows = report.graded[modes[0]].map((_, position) => [
        report.graded[modes[0]][position].question_id,
        `<span class="chip-static">${escaped(report.graded[modes[0]][position].kind)}</span>`,
        escaped(report.graded[modes[0]][position].question),
        ...modes.map((mode) => String(report.graded[mode][position].judge ?? "—")),
    ]);

    questionsMeta.textContent = `${report.graded[modes[0]].length} вопросов, пять режимов`;
    questionsBody.innerHTML = `
        ${report.notes.map((note) => `<p class="route-error">${escaped(note)}</p>`).join("")}
        <div class="strategy-block"><div class="strategy-head">итог по всему набору</div>${summary}</div>
        <div class="strategy-block"><div class="strategy-head">судья по типам вопросов</div>${perKind}</div>
        <div class="strategy-block"><div class="strategy-head">повопросно</div>${table(["#", "Тип", "Вопрос", ...modes.map((mode) => titles[mode])], rows)}</div>
        <p class="hint">Судья видит вопрос, ожидание и ответ, но не знает режима и не видит контекста. Точность контекста — доля выдержек из файлов, по которым на вопрос можно ответить.</p>`;
}

async function runQuestions() {
    questionsButton.disabled = true;
    questionsBody.innerHTML = `<span class="placeholder">Задаём десять вопросов в пяти режимах и зовём судью...</span>`;

    try {
        const payload = askPayload();
        const response = await fetch("/api/questions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                rewriter: payload.rewriter,
                reranker: payload.reranker,
                threshold: payload.threshold,
            }),
        });
        const data = await response.json();
        if (!response.ok) {
            questionsBody.innerHTML = `<span class="route-error">${escaped(data.detail)}</span>`;
            return;
        }
        renderQuestionsRun(data);
    } finally {
        questionsButton.disabled = false;
    }
}

// --- матрица и развертка ------------------------------------------------------

function renderMatrix(report) {
    const metric = (rewriter, reranker, key) => report.results[rewriter][reranker].metrics[key];
    const cell = (rewriter, reranker, key, digits = 2) => {
        const value = metric(rewriter, reranker, key);
        if (value === null || value === undefined) return "—";
        return digits === 0 ? spaced(value) : Number(value).toFixed(digits);
    };

    const recall = table(
        ["Реранкер", ...report.rewriters.map((name) => `<span class="mono">${name}</span>`)],
        report.rerankers.map((reranker) => [
            `<span class="mono">${reranker}</span> <span class="inactive">${escaped(report.reranker_titles[reranker])}</span>`,
            ...report.rewriters.map((rewriter) => cell(rewriter, reranker, "recall@5")),
        ]),
    );

    const cost = table(
        ["Переписывание", "Реранкер", "recall@5", "MRR@5", "Выдержек", "Файлов", "Символов", "Пусто", "Потеряно"],
        report.rewriters.flatMap((rewriter) =>
            report.rerankers.map((reranker) => [
                `<span class="mono">${rewriter}</span>`,
                `<span class="mono">${reranker}</span>`,
                cell(rewriter, reranker, "recall@5"),
                cell(rewriter, reranker, "mrr@5"),
                cell(rewriter, reranker, "kept"),
                cell(rewriter, reranker, "paths"),
                cell(rewriter, reranker, "context_chars", 0),
                cell(rewriter, reranker, "empty"),
                share(metric(rewriter, reranker, "lost")),
            ]),
        ),
    );

    matrixMeta.textContent = `${report.probes.length} проб, пул ${report.pool} → top-${report.top_k}`;
    matrixBody.innerHTML = `
        <p class="hint">Попадание — чанк из того же файла накрыл не меньше ${Math.round(report.overlap_share * 100)}% эталонного отрывка. «Потеряно» — нужный чанк был в пуле, фильтр его выбросил.</p>
        <div class="strategy-block"><div class="strategy-head">recall@5</div>${recall}</div>
        <div class="strategy-block"><div class="strategy-head">цена контекста и цена фильтрации</div>${cost}</div>`;
}

async function runMatrix() {
    matrixButton.disabled = true;
    matrixBody.innerHTML = `<span class="placeholder">Считаем матрицу на 119 пробах. Если кэш оценок неполный, сначала спросим модель.</span>`;

    try {
        const response = await fetch("/api/matrix", { method: "POST" });
        const data = await response.json();
        if (!response.ok) {
            matrixBody.innerHTML = `<span class="route-error">${escaped(data.detail)}</span>`;
            return;
        }
        renderMatrix(data);
    } finally {
        matrixButton.disabled = false;
    }
}

function renderSweep(report) {
    const blocks = report.rerankers.map((reranker) => {
        const chosen = report.chosen[reranker];
        const rows = report.results[reranker].map((row) => [
            `${share(row.threshold)}${chosen !== null && Math.abs(row.threshold - chosen) < 1e-6 ? " ←" : ""}`,
            row.metrics["recall@5"].toFixed(2),
            row.metrics["recall@1"].toFixed(2),
            row.metrics["mrr@5"].toFixed(2),
            row.metrics.kept.toFixed(2),
            row.metrics.paths.toFixed(2),
            spaced(row.metrics.context_chars),
            row.metrics.empty.toFixed(2),
            share(row.metrics.lost),
        ]);
        return `<div class="strategy-block">
            <div class="strategy-head">${escaped(report.reranker_titles[reranker])}</div>
            ${table(["Порог", "recall@5", "recall@1", "MRR@5", "Выдержек", "Файлов", "Символов", "Пусто", "Потеряно"], rows)}
        </div>`;
    }).join("");

    sweepMeta.textContent = `${report.probes} проб, переписывание ${report.rewriter_title}`;
    sweepBody.innerHTML = `
        <p class="hint">Стрелка — выбранный порог. Оценки из кэша, меняется только отсечка.</p>
        ${blocks}`;
}

async function runSweep() {
    sweepButton.disabled = true;
    sweepBody.innerHTML = `<span class="placeholder">Развертываем порог на трёх реранкерах...</span>`;

    try {
        const response = await fetch("/api/sweep", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ rewriter: "none" }),
        });
        const data = await response.json();
        if (!response.ok) {
            sweepBody.innerHTML = `<span class="route-error">${escaped(data.detail)}</span>`;
            return;
        }
        renderSweep(data);
    } finally {
        sweepButton.disabled = false;
    }
}

// --- корпус и индекс -----------------------------------------------------------

function renderCorpus(data) {
    const titles = data.source_titles;
    const rows = Object.entries(data.corpus.by_source).map(([source, slot]) => [
        titles[source] || source,
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
    corpusMeta.textContent = built
        ? `${spaced(built.chunks)} чанков, ${built.model}`
        : "индекс не собран";

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
        : `<span class="placeholder">Индекса нет. Нажмите «Собрать заново» или запустите <span class="mono">python day23/scenarios.py build</span>.</span>`;

    corpusBody.innerHTML = `
        <div class="strategy-block"><div class="strategy-head">корпус</div>${table(["Источник", "Файлов", "Символов", "Страниц"], rows)}</div>
        <div class="strategy-block"><div class="strategy-head">индекс · ${escaped(data.strategy.title)}</div>${indexBlock}</div>
        <p class="hint">Из корпуса исключены две папки: своя и day22 — его README разбирает все десять контрольных вопросов вместе с ответами.</p>
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

// --- состояние -----------------------------------------------------------------

async function loadState() {
    state = await (await fetch("/api/state", { cache: "no-store" })).json();

    renderCorpus(state);
    renderQuestionsSet(state.questions);

    fillSelect(retrieverPick, state.retrievers);
    fillSelect(rewriterPick, state.rewriters);
    fillSelect(rerankerPick, state.rerankers);
    poolInput.value = state.defaults.pool;
    topKInput.value = state.defaults.top_k;

    el("examples").innerHTML = state.questions.items
        .slice(0, 5)
        .map((item) => `<button type="button" class="chip">${escaped(item.question)}</button>`)
        .join("");

    if (state.probes.count) {
        matrixMeta.textContent = `набор: ${state.probes.count} проб`;
    }
}

// --- события ---------------------------------------------------------------------

document.addEventListener("click", (event) => {
    const link = event.target.closest(".chunk-link");
    if (link) showChunk(Number(link.dataset.chunk));
});

el("examples").addEventListener("click", (event) => {
    if (event.target.classList.contains("chip")) {
        questionInput.value = event.target.textContent;
        askForm.requestSubmit();
    }
});

el("closeChunk").addEventListener("click", () => {
    chunkPanel.hidden = true;
});

askForm.addEventListener("submit", ask);
buildButton.addEventListener("click", build);
questionsButton.addEventListener("click", runQuestions);
matrixButton.addEventListener("click", runMatrix);
sweepButton.addEventListener("click", runSweep);

loadState();
