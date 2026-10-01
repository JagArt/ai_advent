const el = (id) => document.getElementById(id);

const corpusBody = el("corpusBody");
const corpusMeta = el("corpusMeta");
const buildBody = el("buildBody");
const buildMeta = el("buildMeta");
const buildButton = el("build");
const searchForm = el("searchForm");
const queryInput = el("query");
const limitInput = el("limit");
const runButton = el("run");
const searchStatus = el("searchStatus");
const hitsPanel = el("hitsPanel");
const hitsBody = el("hitsBody");
const hitsMeta = el("hitsMeta");
const chunkPanel = el("chunkPanel");
const chunkBody = el("chunkBody");
const chunkMeta = el("chunkMeta");
const cutsBody = el("cutsBody");
const cutsMeta = el("cutsMeta");
const compareButton = el("compare");
const compareBody = el("compareBody");
const compareMeta = el("compareMeta");

let state = null;
let cutsPath = null;

const spaced = (value) =>
    Math.round(value).toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
const seconds = (value) => `${value.toFixed(1)} с`;
const megabytes = (value) => `${(value / 1e6).toFixed(2)} МБ`;
const escaped = (text) =>
    String(text).replace(/[&<>"]/g, (char) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[char]);

function table(header, rows, className = "jobs") {
    const head = header.map((name) => `<th>${name}</th>`).join("");
    const body = rows
        .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
        .join("");
    return `<table class="${className}"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

// --- корпус и индексы --------------------------------------------------------

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

    corpusMeta.textContent = `${data.corpus.files} файлов, ${Math.round(data.corpus.pages)} страниц`;
    corpusBody.innerHTML = table(["Источник", "Файлов", "Символов", "Страниц"], rows);

    if (!data.corpus.by_source.pdf) {
        corpusBody.insertAdjacentHTML(
            "beforeend",
            `<p class="hint">PDF в корпусе нет: положите файлы в <span class="mono">day21/corpus/</span>, они подхватятся сами.</p>`,
        );
    }

    for (const note of data.skipped) {
        corpusBody.insertAdjacentHTML(
            "beforeend",
            `<p class="route-error">мимо корпуса: ${escaped(note)}</p>`,
        );
    }
}

function renderBuilt(built, strategies) {
    const known = strategies.filter((item) => built[item.strategy]);

    if (!known.length) {
        buildMeta.textContent = "не собраны";
        buildBody.innerHTML = `<span class="placeholder">Индексов пока нет. Нажмите «Собрать заново» или запустите <span class="mono">python day21/scenarios.py build</span>.</span>`;
        return;
    }

    const rows = known.map((item) => {
        const row = built[item.strategy];
        return [
            `<span class="mono">${item.strategy}</span><br><span class="inactive">${item.title}</span>`,
            spaced(row.chunks),
            Math.round(row.median_tokens),
            spaced(row.p95_tokens),
            spaced(row.max_tokens),
            seconds(row.chunk_seconds),
            seconds(row.embed_seconds),
            megabytes(row.vector_bytes),
        ];
    });

    const first = built[known[0].strategy];
    buildMeta.textContent = `${first.model}, ${first.dimensions} измерений`;
    buildBody.innerHTML = table(
        ["Стратегия", "Чанков", "Медиана, ток.", "p95", "Макс.", "Чанкинг", "Эмбеддинг", "Векторы"],
        rows,
    );
    buildBody.insertAdjacentHTML(
        "beforeend",
        `<p class="hint">Параметры: ${known
            .map(
                (item) =>
                    `<span class="mono">${item.strategy}</span> — ${Object.entries(item.params)
                        .map(([key, value]) => `${key}=${value}`)
                        .join(", ")}`,
            )
            .join(" · ")}</p>`,
    );
}

async function loadState() {
    const response = await fetch("/api/state", { cache: "no-store" });
    state = await response.json();
    renderCorpus(state);
    renderBuilt(state.built, state.strategies);
    if (state.probes.count) {
        compareMeta.textContent = `набор: ${state.probes.count} проб`;
    }
    await loadCuts(cutsPath);
}

async function build() {
    buildButton.disabled = true;
    const started = {};

    const response = await fetch("/api/build", { method: "POST" });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    buildBody.innerHTML = `<span class="placeholder">Считаем чанки и эмбеддинги...</span>`;

    const draw = () => {
        buildBody.innerHTML = Object.values(started)
            .map((item) =>
                item.done
                    ? `<div class="step ok"><div class="step-head"><span class="step-name mono">${item.strategy}</span><span class="meta">${spaced(item.chunks)} чанков, чанкинг ${seconds(item.chunk_seconds)}, эмбеддинг ${seconds(item.embed_seconds)}, ${megabytes(item.vector_bytes)}</span></div></div>`
                    : `<div class="step started"><div class="step-head"><span class="step-name mono">${item.strategy}</span><span class="meta">режем и считаем векторы...</span></div></div>`,
            )
            .join("");
    };

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

            if (event === "building") {
                started[payload.strategy] = { strategy: payload.strategy, done: false };
                draw();
            } else if (event === "built") {
                started[payload.strategy] = { ...payload, done: true };
                draw();
            } else if (event === "error") {
                buildBody.innerHTML = `<span class="route-error">${escaped(payload)}</span>`;
            }
        }
    }

    buildButton.disabled = false;
    await loadState();
}

// --- поиск -------------------------------------------------------------------

function renderHits(data) {
    const strategies = Object.keys(data.results);
    hitsPanel.hidden = false;
    hitsMeta.textContent = `«${data.query}»`;

    hitsBody.innerHTML = strategies
        .map((strategy) => {
            const rows = data.results[strategy].map((hit) => [
                hit.score.toFixed(3),
                `<span class="mono">${escaped(hit.path)}</span>`,
                `<span class="chunk-link" data-chunk="${hit.chunk_id}">${escaped(hit.section)}</span>`,
                hit.tokens,
                `${spaced(hit.start)}–${spaced(hit.end)}`,
            ]);
            return `<div class="strategy-block">
                <div class="strategy-head"><span class="mono">${strategy}</span><span class="inactive">${state.strategies.find((item) => item.strategy === strategy).title}</span></div>
                ${table(["Близость", "Файл", "Раздел", "Токенов", "Символы"], rows)}
            </div>`;
        })
        .join("");
}

async function search(event) {
    event.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    runButton.disabled = true;
    searchStatus.classList.remove("error");
    searchStatus.textContent = "Ищем...";

    try {
        const response = await fetch("/api/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query, limit: Number(limitInput.value) }),
        });
        const data = await response.json();

        if (!response.ok) {
            searchStatus.classList.add("error");
            searchStatus.textContent = data.detail;
            return;
        }

        renderHits(data);
        searchStatus.textContent = "Раздел — ссылка: покажет чанк целиком с метаданными.";
    } finally {
        runButton.disabled = false;
    }
}

async function showChunk(chunkId) {
    const response = await fetch(`/api/chunk/${chunkId}`, { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) return;

    chunkPanel.hidden = false;
    chunkMeta.textContent = `#${data.chunk_id}, стратегия ${data.strategy}`;
    chunkBody.innerHTML = `
        <dl class="facts">
            <dt>файл</dt><dd class="mono">${escaped(data.path)}</dd>
            <dt>документ</dt><dd>${escaped(data.title)}</dd>
            <dt>раздел</dt><dd>${escaped(data.section)}</dd>
            <dt>источник</dt><dd>${escaped(state.source_titles[data.source] || data.source)}</dd>
            <dt>номер в файле</dt><dd>${data.ordinal}</dd>
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

// --- границы чанков ----------------------------------------------------------

async function loadCuts(path) {
    if (!state || !Object.keys(state.built).length) {
        cutsBody.innerHTML = `<span class="placeholder">Сначала соберите индексы.</span>`;
        return;
    }

    const list = await (await fetch("/api/documents?strategy=structural", { cache: "no-store" })).json();
    if (!list.documents.length) return;

    cutsPath = path && list.documents.some((item) => item.path === path) ? path : list.documents[0].path;
    const data = await (await fetch(`/api/document?path=${encodeURIComponent(cutsPath)}`, { cache: "no-store" })).json();

    const options = list.documents
        .map((item) => `<option value="${escaped(item.path)}"${item.path === cutsPath ? " selected" : ""}>${escaped(item.path)}</option>`)
        .join("");

    cutsMeta.textContent = `${spaced(data.document.chars)} символов`;
    cutsBody.innerHTML = `
        <select class="number wide" id="cutsPick">${options}</select>
        <p class="hint">Одна и та же полоса — текст документа. Каждый блок — чанк: наведите, чтобы увидеть раздел, нажмите, чтобы открыть.</p>
        ${Object.entries(data.cuts)
            .map(([strategy, chunks]) => {
                const total = data.document.chars;
                const blocks = chunks
                    .map(
                        (chunk) =>
                            `<span class="cut chunk-link" data-chunk="${chunk.chunk_id}" title="${escaped(chunk.section)} — ${chunk.tokens} ток." style="width:${((chunk.end_char - chunk.start_char) / total) * 100}%"></span>`,
                    )
                    .join("");
                return `<div class="cuts-row">
                    <div class="cuts-label"><span class="mono">${strategy}</span><span class="inactive">${chunks.length} чанков</span></div>
                    <div class="cuts-bar">${blocks}</div>
                </div>`;
            })
            .join("")}`;

    el("cutsPick").addEventListener("change", (event) => loadCuts(event.target.value));
}

// --- сравнение ---------------------------------------------------------------

function renderCompare(report) {
    const kinds = report.kinds;
    const strategies = report.strategies;
    const metric = (kind, strategy, name) => report.results[kind][strategy].metrics[name];

    const summary = table(
        ["Стратегия", ...kinds.map((kind) => state.probes.kinds[kind])],
        strategies.map((strategy) => [
            `<span class="mono">${strategy}</span>`,
            ...kinds.map((kind) => metric(kind, strategy, "recall@5").toFixed(2)),
        ]),
    );

    const detailed = kinds
        .map((kind) => {
            const rows = strategies.map((strategy) => [
                `<span class="mono">${strategy}</span>`,
                spaced(report.results[kind][strategy].index.chunks),
                Math.round(report.results[kind][strategy].index.median_tokens),
                metric(kind, strategy, "recall@1").toFixed(2),
                metric(kind, strategy, "recall@3").toFixed(2),
                metric(kind, strategy, "recall@5").toFixed(2),
                metric(kind, strategy, "mrr@5").toFixed(2),
                spaced(metric(kind, strategy, "chars_to_hit")),
            ]);
            return `<div class="strategy-block">
                <div class="strategy-head">проба — ${escaped(state.probes.kinds[kind])}</div>
                ${table(
                    ["Стратегия", "Чанков", "Медиана, ток.", "recall@1", "recall@3", "recall@5", "MRR@5", "Символов до ответа"],
                    rows,
                )}
            </div>`;
        })
        .join("");

    const diverged = report.probes
        .map((probe, position) => ({
            probe,
            ranks: strategies.map((strategy) => ({
                strategy,
                rank: report.results.query[strategy].scored[position].hit_rank,
            })),
        }))
        .filter((item) => new Set(item.ranks.map((entry) => entry.rank !== null)).size > 1)
        .slice(0, 8);

    compareMeta.textContent = `${report.probes.length} проб, top-${report.top_k}`;
    compareBody.innerHTML = `
        <p class="hint">Попадание — чанк из того же файла накрыл не меньше ${Math.round(report.overlap_share * 100)}% эталонного отрывка. «Символов до ответа» — сколько текста пришлось бы прочитать до первого попадания: критерий по перекрытию сам по себе выгоден большим чанкам, и без этой колонки recall читается неверно.</p>
        <div class="strategy-block"><div class="strategy-head">recall@5 по способу спросить</div>${summary}</div>
        ${detailed}
        <div class="strategy-block">
            <div class="strategy-head">где стратегии разошлись, поисковый запрос</div>
            ${table(
                ["Запрос", "Файл", ...strategies],
                diverged.map((item) => [
                    escaped(item.probe.query),
                    `<span class="mono">${escaped(item.probe.path)}</span>`,
                    ...item.ranks.map((entry) =>
                        entry.rank ? `<span class="chip-static ok">#${entry.rank}</span>` : `<span class="inactive">мимо</span>`,
                    ),
                ]),
            )}
        </div>`;
}

async function compare() {
    compareButton.disabled = true;
    compareBody.innerHTML = `<span class="placeholder">Прогоняем набор проб по всем индексам...</span>`;

    try {
        const response = await fetch("/api/compare", { method: "POST" });
        const data = await response.json();
        if (!response.ok) {
            compareBody.innerHTML = `<span class="route-error">${escaped(data.detail)}</span>`;
            return;
        }
        renderCompare(data);
    } finally {
        compareButton.disabled = false;
    }
}

// --- события -----------------------------------------------------------------

document.addEventListener("click", (event) => {
    const link = event.target.closest(".chunk-link");
    if (link) showChunk(Number(link.dataset.chunk));
});

el("examples").addEventListener("click", (event) => {
    if (event.target.classList.contains("chip")) {
        queryInput.value = event.target.textContent;
        searchForm.requestSubmit();
    }
});

el("closeChunk").addEventListener("click", () => {
    chunkPanel.hidden = true;
});

searchForm.addEventListener("submit", search);
buildButton.addEventListener("click", build);
compareButton.addEventListener("click", compare);

loadState();
