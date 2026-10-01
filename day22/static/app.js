const el = (id) => document.getElementById(id);

const askForm = el("askForm");
const questionInput = el("question");
const retrieverPick = el("retriever");
const limitInput = el("limit");
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
const retrievalBody = el("retrievalBody");
const retrievalMeta = el("retrievalMeta");
const retrievalButton = el("runRetrieval");

let state = null;

const spaced = (value) =>
    Math.round(value).toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
const escaped = (text) =>
    String(text).replace(/[&<>"]/g, (char) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char]);
const share = (value) => (value === null || value === undefined ? "—" : value.toFixed(2));

function table(header, rows, className = "jobs") {
    const head = header.map((name) => `<th>${name}</th>`).join("");
    const body = rows
        .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
        .join("");
    return `<table class="${className}"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
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
    el(`cost-${mode}`).textContent =
        `${spaced(data.prompt_tokens)} ток. запроса · ${spaced(data.completion_tokens)} ответа · ${data.seconds} с`;

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

function renderContext(data) {
    contextPanel.hidden = false;
    contextMeta.textContent =
        `${data.title}, ${data.sources.length} выдержек, ${spaced(data.tokens)} токенов, поиск ${(data.seconds * 1000).toFixed(1)} мс`;

    const rows = data.sources.map((source) => [
        `<span class="chip-static">${source.number}</span>`,
        `<span class="mono">${escaped(source.path)}</span>`,
        `<span class="chunk-link" data-chunk="${source.chunk_id}">${escaped(source.section)}</span>`,
        source.tokens,
        source.score.toFixed(3),
    ]);

    contextBody.innerHTML =
        table(["№", "Файл", "Раздел", "Токенов", "Счёт"], rows) +
        `<p class="hint">Номер слева — тот самый, которым ответ ссылается на выдержку. Раздел — ссылка: покажет чанк целиком с метаданными.</p>`;
}

async function ask(event) {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    runButton.disabled = true;
    askStatus.classList.remove("error");
    askStatus.textContent = "Ищем и спрашиваем оба режима сразу...";

    answersPanel.hidden = false;
    answersMeta.textContent = `«${question}»`;
    answersBox.innerHTML = state.modes
        .map((mode) => answerCard(mode.name, mode.title, mode.name === "rag" ? "ищем..." : "спрашиваем..."))
        .join("");
    contextPanel.hidden = true;

    try {
        const response = await fetch("/api/ask", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                retriever: retrieverPick.value,
                limit: Number(limitInput.value),
            }),
        });

        await readFrames(response, (event, payload) => {
            if (event === "context") {
                renderContext(payload);
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
            askStatus.textContent = "Разница в токенах запроса — это и есть цена контекста.";
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
        `<p class="hint">Ожидание и источники записаны заранее. «Прогнать» задаст каждый вопрос в обоих режимах и сверит ответы — это единственная кнопка на странице, которой нужен ключ.</p>`;
}

function renderQuestionsRun(report) {
    const modes = report.modes;
    const summary = table(
        ["Режим", "Факты", "Источник найден", "Ответ сослался", "Судья, 0–2", "На 2 балла", "Отказы «вне базы»", "Токенов запроса"],
        modes.map((mode) => {
            const row = report.summary[mode];
            return [
                `<strong>${escaped(report.mode_titles[mode])}</strong>`,
                share(row.facts),
                share(row.sources),
                share(row.cited),
                share(row.judge),
                `${row.judge_full} из ${row.questions}`,
                `${row.refused_outside} из ${row.outside}`,
                spaced(row.prompt_tokens),
            ];
        }),
    );

    const kinds = Object.keys(report.by_kind[modes[0]]);
    const perKind = table(
        ["Тип вопроса", ...modes.map((mode) => `${report.mode_titles[mode]}, судья`), ...modes.map((mode) => `${report.mode_titles[mode]}, факты`)],
        kinds.map((kind) => [
            `${escaped(kind)} <span class="inactive">(${report.by_kind[modes[0]][kind].questions})</span>`,
            ...modes.map((mode) => share(report.by_kind[mode][kind].judge)),
            ...modes.map((mode) => share(report.by_kind[mode][kind].facts)),
        ]),
    );

    const rows = report.graded[modes[0]].map((plain, position) => {
        const rag = report.graded[modes[1]][position];
        return [
            plain.question_id,
            `<span class="chip-static">${escaped(plain.kind)}</span>`,
            escaped(plain.question),
            `${plain.judge ?? "—"} → <strong>${rag.judge ?? "—"}</strong>`,
            `${share(plain.facts_share)} → <strong>${share(rag.facts_share)}</strong>`,
            rag.sources_hit === null ? "—" : rag.sources_hit ? `<span class="chip-static ok">да</span>` : `<span class="inactive">мимо</span>`,
        ];
    });

    questionsMeta.textContent = `${report.graded[modes[0]].length} вопросов, поиск — ${report.retriever_title}`;
    questionsBody.innerHTML = `
        ${report.notes.map((note) => `<p class="route-error">${escaped(note)}</p>`).join("")}
        <div class="strategy-block"><div class="strategy-head">итог по всему набору</div>${summary}</div>
        <div class="strategy-block"><div class="strategy-head">по типам вопросов</div>${perKind}</div>
        <div class="strategy-block"><div class="strategy-head">повопросно: без RAG → с RAG</div>${table(["#", "Тип", "Вопрос", "Судья", "Факты", "Источник"], rows)}</div>
        <p class="hint">Судья видит вопрос, ожидание и ответ, но не знает режима и не видит контекста. Механические колонки считаются без модели.</p>`;
}

async function runQuestions() {
    questionsButton.disabled = true;
    questionsBody.innerHTML = `<span class="placeholder">Задаём десять вопросов в двух режимах и зовём судью...</span>`;

    try {
        const response = await fetch("/api/questions", { method: "POST" });
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

// --- ретриверы ----------------------------------------------------------------

function renderRetrieval(report) {
    const names = report.retrievers;
    const metric = (kind, name, key) => report.results[kind][name].metrics[key];

    const summary = table(
        ["Ретривер", ...report.kinds.map((kind) => state.probes.kinds[kind])],
        names.map((name) => [
            `<span class="mono">${name}</span> <span class="inactive">${escaped(report.titles[name])}</span>`,
            ...report.kinds.map((kind) => metric(kind, name, "recall@5").toFixed(2)),
        ]),
    );

    const detailed = report.kinds
        .map((kind) => {
            const rows = names.map((name) => [
                `<span class="mono">${name}</span>`,
                metric(kind, name, "recall@1").toFixed(2),
                metric(kind, name, "recall@3").toFixed(2),
                metric(kind, name, "recall@5").toFixed(2),
                metric(kind, name, "mrr@5").toFixed(2),
                spaced(metric(kind, name, "chars_to_hit")),
            ]);
            return `<div class="strategy-block">
                <div class="strategy-head">проба — ${escaped(state.probes.kinds[kind])}</div>
                ${table(["Ретривер", "recall@1", "recall@3", "recall@5", "MRR@5", "Символов до ответа"], rows)}
            </div>`;
        })
        .join("");

    retrievalMeta.textContent = `${report.probes.length} проб, top-${report.top_k}`;
    retrievalBody.innerHTML = `
        <p class="hint">Попадание — чанк из того же файла накрыл не меньше ${Math.round(report.overlap_share * 100)}% эталонного отрывка. Колонка «вопрос» — та, что решает: в RAG приходит вопрос человека, а не отрывок из документа.</p>
        <div class="strategy-block"><div class="strategy-head">recall@5 по способу спросить</div>${summary}</div>
        ${detailed}`;
}

async function runRetrieval() {
    retrievalButton.disabled = true;
    retrievalBody.innerHTML = `<span class="placeholder">Прогоняем набор проб по трём ретриверам...</span>`;

    try {
        const response = await fetch("/api/retrieval", { method: "POST" });
        const data = await response.json();
        if (!response.ok) {
            retrievalBody.innerHTML = `<span class="route-error">${escaped(data.detail)}</span>`;
            return;
        }
        renderRetrieval(data);
    } finally {
        retrievalButton.disabled = false;
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
        : `<span class="placeholder">Индекса нет. Нажмите «Собрать заново» или запустите <span class="mono">python day22/scenarios.py build</span>.</span>`;

    corpusBody.innerHTML = `
        <div class="strategy-block"><div class="strategy-head">корпус</div>${table(["Источник", "Файлов", "Символов", "Страниц"], rows)}</div>
        <div class="strategy-block"><div class="strategy-head">индекс · ${escaped(data.strategy.title)}</div>${indexBlock}</div>
        <p class="hint">Папка day22 в корпус не входит: её докстринги разбирают контрольные вопросы числами, и в индексе она была бы шпаргалкой.</p>
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

    retrieverPick.innerHTML = state.retrievers
        .map((item) => `<option value="${item.name}"${item.default ? " selected" : ""}>${escaped(item.title)}</option>`)
        .join("");

    el("examples").innerHTML = state.questions.items
        .slice(0, 5)
        .map((item) => `<button type="button" class="chip">${escaped(item.question)}</button>`)
        .join("");

    if (state.probes.count) {
        retrievalMeta.textContent = `набор: ${state.probes.count} проб`;
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
retrievalButton.addEventListener("click", runRetrieval);

loadState();
