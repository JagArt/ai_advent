const el = (id) => document.getElementById(id);

const askForm = el("askForm");
const questionInput = el("question");
const retrieverPick = el("retriever");
const rewriterPick = el("rewriter");
const rerankerPick = el("reranker");
const poolInput = el("pool");
const topKInput = el("topK");
const gateInput = el("gate");
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
const weakBody = el("weakBody");
const weakMeta = el("weakMeta");
const weakButton = el("runWeak");
const gateBody = el("gateBody");
const gateMeta = el("gateMeta");
const gateButton = el("runGate");

let state = null;

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
        <div id="claims-${mode}"></div>
        <div class="answer-foot" id="foot-${mode}"></div>
    </div>`;
}

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

function renderClaims(mode, data) {
    const box = el(`claims-${mode}`);
    const checked = data.verify ? data.verify.items : [];

    if (data.refusal) {
        box.innerHTML = `<div class="claim refusal">
            <p class="claim-text">Отказ — ${escaped(data.refusal)}.</p>
            <p class="hint">${escaped(data.clarify || "уточнения нет")}</p>
        </div>`;
        return;
    }

    if (!checked.length) {
        box.innerHTML = "";
        return;
    }

    const broken = data.structured && data.structured.broken
        ? `<p class="route-error">${escaped(data.structured.broken)}</p>`
        : "";
    box.innerHTML = broken + checked.map(claimCard).join("");
}

function renderCost(mode, data) {
    const stage = data.context && data.context.stage_tokens
        ? ` + ${spaced(data.context.stage_tokens)} на этапы`
        : "";
    el(`cost-${mode}`).textContent = data.refusal && !data.prompt_tokens
        ? "0 токенов: к модели не ходили"
        : `${spaced(data.prompt_tokens)}${stage} ток. · ${spaced(data.completion_tokens)} ответа · ${data.seconds} с`;

    // У структурированных режимов текст собирается из утверждений, и показывать
    // рядом ещё и сырой json незачем: он уже разложен карточками ниже.
    el(`text-${mode}`).textContent = data.text;
    renderClaims(mode, data);

    const foot = el(`foot-${mode}`);
    if (!data.context) {
        foot.innerHTML = `<span class="inactive">контекста нет: вопрос ушёл к модели как есть</span>`;
        return;
    }

    const verified = data.verify
        ? `<span class="inactive">дословных цитат:</span> <span class="chip-static ${data.verify.exact === data.verify.claims ? "ok" : "warn"}">${data.verify.exact} из ${data.verify.claims}</span> `
        : "";
    const cited = data.cited_paths.length
        ? data.cited_paths.map((path) => `<span class="chip-static ok mono">${escaped(path)}</span>`).join(" ")
        : `<span class="inactive">ответ не сослался ни на одну выдержку</span>`;
    foot.innerHTML = `${verified}<span class="inactive">подкреплено:</span> ${cited}`;
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

function renderContexts(payload) {
    contextPanel.hidden = false;

    // Конвейер у трёх режимов с базой один, поэтому и контекст показывается
    // один: четыре одинаковые таблицы рядом не сказали бы ничего нового.
    const mode = state.modes.map((item) => item.name).find((name) => payload.contexts[name]);
    const data = mode ? payload.contexts[mode] : null;

    if (!data) {
        contextMeta.textContent = "контекста нет";
        contextBody.innerHTML = `<p class="hint">Поиск не принёс ни одной выдержки.</p>`;
        return;
    }

    const gates = Object.entries(payload.gates)
        .filter(([, verdict]) => verdict)
        .map(([name, verdict]) =>
            `<span class="chip-static ${verdict.passed ? "ok" : "bad"}">${escaped(state.mode_titles[name])}: ${verdict.passed ? "пропускает" : escaped(verdict.reason)}</span>`)
        .join(" ");

    const rows = (data.sources || []).map((source) => sourceRow(source, "дошло"))
        .concat((data.dropped_sources || []).map((item) => sourceRow(item, escaped(item.reason || "отсеяно"))));

    contextMeta.textContent = `пул ${data.pool} → ${data.kept} выдержек, лучшая оценка ${share(data.best_relevance)}`;
    contextBody.innerHTML = `
        <div class="strategy-block">
            <div class="strategy-head">
                общий конвейер трёх режимов
                <span class="inactive">${escaped(data.plan.rewriter_title)} · ${escaped(data.plan.reranker_title)} · порог фильтра ${score(data.plan.threshold)}</span>
            </div>
            ${data.rewrite.changed ? `<p class="hint">запрос: <span class="mono">${escaped(data.rewrite.query)}</span></p>` : ""}
            ${table(["№", "Оценка", "Место", "Файл", "Раздел", "Судьба"], rows)}
        </div>
        <p class="hint">Порог отказа смотрит на лучшую оценку среди дошедших: ${gates || "—"}</p>`;
}

function askPayload() {
    const gateThreshold = gateInput.value.trim();
    return {
        question: questionInput.value.trim(),
        retriever: retrieverPick.value,
        rewriter: rewriterPick.value,
        reranker: rerankerPick.value,
        pool: Number(poolInput.value),
        top_k: Number(topKInput.value),
        gate_threshold: gateThreshold === "" ? null : Number(gateThreshold),
    };
}

async function ask(event) {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    runButton.disabled = true;
    askStatus.classList.remove("error");
    askStatus.textContent = "Собираем контекст и спрашиваем все четыре режима...";

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
                renderCost(payload.mode, payload);
            } else if (event === "error") {
                askStatus.classList.add("error");
                askStatus.textContent = payload;
            }
        });

        if (!askStatus.classList.contains("error")) {
            askStatus.textContent = "Цитата сверяется с текстом той выдержки, на которую сослалось утверждение. Клик по источнику открывает чанк с подсвеченной цитатой.";
        }
    } finally {
        runButton.disabled = false;
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

// --- контрольный набор --------------------------------------------------------

function renderQuestionsSet(data) {
    questionsMeta.textContent = `${data.count} вопросов, те же, что в day22 и day23`;
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
        `<p class="hint">«Прогнать» задаст каждый вопрос в четырёх режимах, сверит все цитаты с чанками и позовёт двух судей — качества и сверщика. Это самая дорогая кнопка на странице.</p>`;
}

function renderQuestionsRun(report) {
    const modes = report.modes;
    const titles = report.mode_titles;
    const summary = report.summary;

    const checks = table(
        ["Режим", "Источники есть", "Цитаты есть", "Цитата дословна", "Цитата нашлась", "Смысл следует", "Утверждений", "Выдумано"],
        modes.map((mode) => [
            `<strong>${escaped(titles[mode])}</strong>`,
            share(summary[mode].has_sources),
            share(summary[mode].has_quotes),
            share(summary[mode].exact),
            share(summary[mode].grounded),
            share(summary[mode].entail),
            share(summary[mode].claims),
            String(summary[mode].fabricated),
        ]),
    );

    const quality = table(
        ["Режим", "Факты", "Источник", "Точность", "Сослался", "Судья", "На 2", "Выдержек", "Токенов"],
        modes.map((mode) => [
            `<strong>${escaped(titles[mode])}</strong>`,
            share(summary[mode].facts),
            share(summary[mode].sources),
            share(summary[mode].precision),
            share(summary[mode].cited),
            share(summary[mode].judge),
            `${summary[mode].judge_full} из ${summary[mode].questions}`,
            share(summary[mode].kept),
            spaced(summary[mode].total_prompt_tokens),
        ]),
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

    questionsMeta.textContent = `${report.graded[modes[0]].length} вопросов, четыре режима`;
    questionsBody.innerHTML = `
        ${report.notes.map((note) => `<p class="route-error">${escaped(note)}</p>`).join("")}
        <div class="strategy-block"><div class="strategy-head">три проверки задания</div>${checks}</div>
        <div class="strategy-block"><div class="strategy-head">качество ответа</div>${quality}</div>
        <div class="strategy-block"><div class="strategy-head">судья по типам вопросов</div>${perKind}</div>
        <div class="strategy-block"><div class="strategy-head">повопросно</div>${table(["#", "Тип", "Вопрос", ...modes.map((mode) => titles[mode])], rows)}</div>
        <p class="hint">«Цитата дословна» считается в коде поиском подстроки. «Смысл следует» ставит судья-сверщик, который видит только утверждение и его цитату — ни вопроса, ни базы.</p>`;
}

async function runSet(button, body, meta, url, render, note) {
    button.disabled = true;
    body.innerHTML = `<span class="placeholder">${escaped(note)}</span>`;

    try {
        const payload = askPayload();
        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                rewriter: payload.rewriter,
                reranker: payload.reranker,
                gate_threshold: payload.gate_threshold,
            }),
        });
        const data = await response.json();
        if (!response.ok) {
            body.innerHTML = `<span class="route-error">${escaped(data.detail)}</span>`;
            return;
        }
        render(data);
    } finally {
        button.disabled = false;
    }
}

// --- слабый контекст ----------------------------------------------------------

function renderWeakSet(data) {
    weakMeta.textContent = `${data.count} вопросов, верный ответ везде — отказ`;
    weakBody.innerHTML =
        table(
            ["#", "Тип", "Вопрос", "Почему отвечать нельзя"],
            data.items.map((item) => [
                item.id,
                `<span class="chip-static">${escaped(item.kind)}</span>`,
                escaped(item.question),
                `<span class="inactive">${escaped(item.expect)}</span>`,
            ]),
        ) +
        `<p class="hint">Половина вопросов вне базы — их должен ловить порог. Половина неоднозначных: в базе есть несколько разных верных ответов, оценки высокие, и порог такую пробу пропускает — ловить её может только правило в промпте.</p>`;
}

function renderWeakRun(report) {
    const modes = report.modes;
    const titles = report.mode_titles;
    const summary = report.summary;

    const refusals = table(
        ["Режим", "Отказов", "Доля", "С уточнением", "Судья", "На 2", "Чем вызван"],
        modes.map((mode) => [
            `<strong>${escaped(titles[mode])}</strong>`,
            `${summary[mode].refused} из ${summary[mode].questions}`,
            share(summary[mode].refused_rightly),
            share(summary[mode].asked_back),
            share(summary[mode].judge),
            `${summary[mode].judge_full} из ${summary[mode].questions}`,
            Object.entries(summary[mode].reasons).map(([reason, count]) => `${escaped(reason)}: ${count}`).join(", ") || "—",
        ]),
    );

    const kinds = Object.keys(report.by_kind[modes[0]]);
    const perKind = table(
        ["Тип вопроса", ...modes.map((mode) => titles[mode])],
        kinds.map((kind) => [
            `${escaped(kind)} <span class="inactive">(${report.by_kind[modes[0]][kind].questions})</span>`,
            ...modes.map((mode) => share(report.by_kind[mode][kind].refused_rightly)),
        ]),
    );

    const rows = report.graded[modes[0]].map((_, position) => [
        report.graded[modes[0]][position].question_id,
        `<span class="chip-static">${escaped(report.graded[modes[0]][position].kind)}</span>`,
        escaped(report.graded[modes[0]][position].question),
        ...modes.map((mode) => {
            const row = report.graded[mode][position];
            return row.refused
                ? `<span class="chip-static ok">отказ${row.refusal ? ` · ${escaped(row.refusal)}` : ""}</span>`
                : `<span class="chip-static bad">ответил</span>`;
        }),
    ]);

    const clarifications = report.gated_modes.flatMap((mode) =>
        report.graded[mode]
            .filter((row) => row.clarify)
            .map((row) => `<div class="claim"><p class="claim-text">${escaped(row.question)}</p><p class="hint">${escaped(row.clarify)}</p></div>`),
    ).join("");

    weakMeta.textContent = `${report.graded[modes[0]].length} вопросов, четыре режима`;
    weakBody.innerHTML = `
        <div class="strategy-block"><div class="strategy-head">отказы</div>${refusals}</div>
        <div class="strategy-block"><div class="strategy-head">по типам — доля отказов</div>${perKind}</div>
        <div class="strategy-block"><div class="strategy-head">повопросно</div>${table(["#", "Тип", "Вопрос", ...modes.map((mode) => titles[mode])], rows)}</div>
        <div class="strategy-block"><div class="strategy-head">как звучит уточнение</div>${clarifications || `<p class="hint">уточнений не было</p>`}</div>`;
}

// --- развертка по порогу --------------------------------------------------------

function renderGate(report) {
    const rows = report.results.map((row) => [
        `${share(row.threshold)}${Math.abs(row.threshold - report.chosen) < 1e-6 ? " ←" : ""}`,
        `${share(row.right)} <span class="inactive">(${row.right_count} из ${row.right_total})</span>`,
        `${share(row.wrong)} <span class="inactive">(${row.wrong_count} из ${row.wrong_total})</span>`,
        `${row.by_kind["вне базы"]} из ${report.weak_total["вне базы"]}`,
        `${row.by_kind["неоднозначный"]} из ${report.weak_total["неоднозначный"]}`,
    ]);

    const perQuestion = table(
        ["Набор", "#", "Тип", "Вопрос", "Выдержек", "Лучшая оценка"],
        report.rows.map((row) => [
            row.set,
            row.id,
            `<span class="chip-static">${escaped(row.kind)}</span>`,
            escaped(row.question),
            row.kept,
            `<span class="chip-static ${row.best === null ? "bad" : row.best >= report.chosen ? "ok" : "warn"}">${share(row.best)}</span>`,
        ]),
    );

    gateMeta.textContent = `порог ${share(report.chosen)}`;
    gateBody.innerHTML = `
        <div class="strategy-block"><div class="strategy-head">размен</div>${table(["Порог", "Отказал верно", "Отказал зря", "Вне базы", "Неоднозначный"], rows)}</div>
        <div class="strategy-block"><div class="strategy-head">оценка лучшей выдержки повопросно</div>${perQuestion}</div>
        <p class="hint">Отказ зря — это вопрос контрольного набора, ответ на который в базе есть, а порог его не пустил. На этой линейке таких нет ни при одном пороге: на всех восьми хотя бы одна выдержка получила от реранкера 1.00.</p>`;
}

async function runGate() {
    gateButton.disabled = true;
    gateBody.innerHTML = `<span class="placeholder">Считаем контексты обоих наборов и разворачиваем порог...</span>`;

    try {
        const response = await fetch("/api/gate", { method: "POST" });
        const data = await response.json();
        if (!response.ok) {
            gateBody.innerHTML = `<span class="route-error">${escaped(data.detail)}</span>`;
            return;
        }
        renderGate(data);
    } finally {
        gateButton.disabled = false;
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
        : `<span class="placeholder">Индекса нет. Нажмите «Собрать заново» или запустите <span class="mono">python day24/scenarios.py build</span>.</span>`;

    corpusBody.innerHTML = `
        <div class="strategy-block"><div class="strategy-head">корпус</div>${table(["Источник", "Файлов", "Символов", "Страниц"], rows)}</div>
        <div class="strategy-block"><div class="strategy-head">индекс · ${escaped(data.strategy.title)}</div>${indexBlock}</div>
        <p class="hint">Из корпуса исключены три папки: своя, day22 и day23 — их README разбирают все десять контрольных вопросов вместе с ответами.</p>
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
    renderWeakSet(state.weak);

    fillSelect(retrieverPick, state.retrievers);
    fillSelect(rewriterPick, state.rewriters);
    fillSelect(rerankerPick, state.rerankers);
    poolInput.value = state.defaults.pool;
    topKInput.value = state.defaults.top_k;
    gateInput.placeholder = share(state.defaults.gate_threshold);
    gateMeta.textContent = `порог по умолчанию ${share(state.defaults.gate_threshold)}`;

    el("examples").innerHTML = state.questions.items
        .slice(0, 3)
        .concat(state.weak.items.slice(4, 6))
        .map((item) => `<button type="button" class="chip">${escaped(item.question)}</button>`)
        .join("");
}

// --- события ---------------------------------------------------------------------

document.addEventListener("click", (event) => {
    const link = event.target.closest(".chunk-link");
    if (link) {
        showChunk(Number(link.dataset.chunk), Number(link.dataset.start ?? -1), Number(link.dataset.end ?? -1));
    }
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
questionsButton.addEventListener("click", () =>
    runSet(questionsButton, questionsBody, questionsMeta, "/api/questions", renderQuestionsRun,
        "Задаём десять вопросов в четырёх режимах, сверяем цитаты и зовём двух судей..."));
weakButton.addEventListener("click", () =>
    runSet(weakButton, weakBody, weakMeta, "/api/weak", renderWeakRun,
        "Задаём восемь вопросов слабого контекста в четырёх режимах..."));
gateButton.addEventListener("click", runGate);

loadState();
