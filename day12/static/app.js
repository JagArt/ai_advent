const form = document.getElementById("form");
const promptInput = document.getElementById("prompt");
const submitButton = document.getElementById("submit");
const autorunButton = document.getElementById("autorun");
const compareButton = document.getElementById("compare-run");
const newDialogButton = document.getElementById("new-dialog");
const resetButton = document.getElementById("reset-db");
const dialogsEl = document.getElementById("dialogs");
const statusEl = document.getElementById("status");
const paramsEl = document.getElementById("params");
const chatEl = document.getElementById("chat");
const placeholderEl = document.getElementById("placeholder");
const profileBodyEl = document.getElementById("profile-body");
const profileMetaEl = document.getElementById("profile-meta");
const scalesEl = document.getElementById("scales");
const longtermBodyEl = document.getElementById("longterm-body");
const longtermMetaEl = document.getElementById("longterm-meta");
const workingBodyEl = document.getElementById("working-body");
const workingMetaEl = document.getElementById("working-meta");
const copySpecButton = document.getElementById("copy-spec");
const compareDialog = document.getElementById("compare");
const compareQuestionEl = document.getElementById("compare-question");
const compareBodyEl = document.getElementById("compare-body");
const compareCloseButton = document.getElementById("compare-close");

const submitLabel = submitButton.textContent;
const compareLabel = compareButton.textContent;
const SESSION_KEY = "day12.session";
const WINDOW_KEY = "day12.window";
const AUTOSAVE_KEY = "day12.autosave";

const PROFILE = "profile";
const LONGTERM = "longterm";
const WORKING = "working";

// Вид находки: формулировка в раздел или правка шкалы профиля.
const SCALE = "scale";

const TARGET_NAMES = {
    [PROFILE]: "профиль",
    [LONGTERM]: "долговременная",
    [WORKING]: "рабочая",
};
const TARGET_BUTTONS = {
    [PROFILE]: "В профиль",
    [LONGTERM]: "В долговременную",
    [WORKING]: "В рабочую",
};

// Параметры агента, разделы, шкалы профиля и список профилей приходят из
// GET /api/defaults.
let config = null;
let sessionId = null;
let historySize = 0;
let dialogs = [];
// Профиль и два уровня памяти: всё, что живёт дольше реплики и рисуется блоками
// рядом с лентой. Профиль хранится тут же пунктами, чтобы отметки о свежих записях
// и удаление работали для него теми же тремя строками кода.
let memory = { [PROFILE]: [], [LONGTERM]: [], [WORKING]: [] };
let profile = null;
let profileId = null;
let freshItems = new Set();
// Карточки — ещё не память: они ждут решения и хранятся только на этой странице.
let cards = new Map();
let memoryParam = null;
let contextParam = null;
let tierParam = null;
let windowSelect = null;
let profileSelect = null;
let autosaveInput = null;
let windowMessages = null;
let autosave = false;
let controller = null;
// Автопрогон: ходы печатает страница, и до конца прогона диалог принадлежит ему.
let run = null;
// Сравнение профилей идёт мимо диалога, но к той же модели: пока оно не кончилось,
// второй раз его запускать нечего.
let comparing = false;

// Ход идёт или прогон не кончился — трогать память и диалоги нельзя. Прогон
// заперт и между ходами: в паузе между ними диалог всё ещё не свободен.
function locked() {
    return Boolean(controller) || Boolean(run);
}

function setStatus(text, isError = false) {
    // Во время прогона к каждой строке приписан его счётчик: ходы идут сами, и
    // без номера непонятно, сколько их ещё будет.
    statusEl.textContent = run ? `Прогон ${run.index}/${run.total} · ${text}` : text;
    statusEl.classList.toggle("error", isError);
}

// Пока идёт стрим, та же кнопка работает на остановку. Во время прогона она
// выключена совсем: свою реплику в его диалог не вставить, а останавливают прогон
// его собственной кнопкой.
function setBusy(busy) {
    const stopping = busy && !run;
    submitButton.textContent = stopping ? "Остановить" : submitLabel;
    submitButton.classList.toggle("stop", stopping);
    submitButton.disabled = Boolean(run);
    newDialogButton.disabled = locked();
    resetButton.disabled = locked();
    setCompareButton();
    // Панель гаснет вместе с кнопкой: уход в другой диалог посреди ответа увёл бы
    // текущий стрим в чужую ленту.
    dialogsEl.classList.toggle("busy", locked());
    promptInput.readOnly = locked();
    // Окно — часть уходящего запроса, менять его посреди хода нечестно: состав
    // хода перестал бы сходиться с настройками на экране.
    windowSelect.disabled = locked();
    // Профиль — часть уходящего запроса, как и окно: сменить собеседника посреди
    // ответа значило бы получить ответ, которого не просил ни один из них.
    profileSelect.disabled = locked();
    autosaveInput.disabled = locked();
    // Память и профиль во время хода не трогаем: блоки уже ушли в запрос, и правка
    // на этом месте разошлась бы с тем, что видит модель.
    setMemoryDisabled(locked());
    setAutorunButton();
}

function setCompareButton() {
    compareButton.textContent = comparing ? "Спрашиваем..." : compareLabel;
    compareButton.disabled = comparing || locked() || !config;
}

function setAutorunButton() {
    const total = config.autorun.length;
    autorunButton.textContent = run
        ? "Остановить прогон"
        : `Автопрогон · ${total} ${plural(total, "ход", "хода", "ходов")}`;
    // Красная кнопка остановки не должна оставаться приглушённой: класс уходит
    // вместе со спокойным состоянием.
    autorunButton.classList.toggle("stop", Boolean(run));
    autorunButton.classList.toggle("button-quiet", !run);
    autorunButton.disabled = !total || (!run && Boolean(controller));
}

// Кто выбирает уровень — видно по галочке, поэтому она и значение всегда меняются
// вместе: прогон включает автосохранение на время и возвращает как было.
function setAutosave(value) {
    autosave = value;
    autosaveInput.checked = value;
}

function setMemoryDisabled(disabled) {
    for (const control of document.querySelectorAll(".tier-item-delete, .scale-select")) {
        control.disabled = disabled;
    }
    // Карточки перерисовываются, а не гасятся скопом: у сохранённой кнопка своего
    // уровня выключена и в спокойном состоянии.
    for (const card of cards.values()) {
        paintCard(card);
    }
}

function addParam(text) {
    const param = document.createElement("span");
    param.className = "param";
    param.textContent = text;
    paramsEl.append(param);
    return param;
}

// Значения приходят парами «что уйдёт на сервер» и «что видно в списке»: у окна это
// одно и то же число, у профиля — id и имя человека.
function addSelectParam(text, options, selected) {
    const label = document.createElement("label");
    label.className = "param param-control";
    label.textContent = text;

    const select = document.createElement("select");
    select.className = "param-select";
    select.append(...options.map(({ value, label: caption }) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = caption;
        option.selected = String(value) === String(selected);
        return option;
    }));

    label.append(select);
    paramsEl.append(label);
    return select;
}

// Галочка отдаёт выбор уровня агенту. Смысл задания в обратном — в том, что
// уровень выбирают явно, — поэтому по умолчанию она снята, а карточка появляется
// в любом случае: даже сохранив сам, агент показывает, куда именно положил.
function addToggleParam(text, checked, title) {
    const label = document.createElement("label");
    label.className = "param param-control param-toggle";
    label.title = title;

    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = checked;

    label.append(input, document.createTextNode(text));
    paramsEl.append(label);
    return input;
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

function messageWord(count) {
    return plural(count, "сообщение", "сообщения", "сообщений");
}

function itemWord(count) {
    return plural(count, "пункт", "пункта", "пунктов");
}

// Три уровня в трёх счётчиках: сколько реплик помнит лента, сколько из них ушло
// в запрос и сколько пунктов лежит в блоках памяти рядом с ними. Профиля среди
// счётчиков нет: он уходит в запрос целиком и всегда, поэтому в шапке стоит его имя.
function setContext(payload) {
    historySize = payload.history_size;
    memoryParam.textContent = `диалог ${historySize} ${messageWord(historySize)}`;

    if (payload.profile_id) {
        profileId = payload.profile_id;
        profileSelect.value = payload.profile_id;
    }

    const dropped = payload.history_size - payload.context_size;
    const tail = dropped > 0 ? `, за окном ${dropped}` : "";
    contextParam.textContent = `в запросе ${payload.context_size} сообщ.${tail}`;
    contextParam.classList.toggle("param-warning", dropped > 0);

    const stored = payload.longterm_size + payload.working_size;
    tierParam.textContent = `память ${payload.longterm_size}+${payload.working_size} ${itemWord(stored)}`;
    tierParam.classList.toggle("param-warning", stored > 0);
}

// Блок уровня: разделы в том порядке, что задал сервер, внутри — пункты в порядке
// появления. Раздел без пунктов не показывается, пустая рубрика только шумит.
function renderTier(tier, bodyEl, metaEl) {
    const items = memory[tier];
    if (metaEl) {
        metaEl.textContent = items.length ? `${items.length} ${itemWord(items.length)}` : "пусто";
    }

    if (!items.length) {
        const empty = document.createElement("p");
        empty.className = "tier-empty";
        empty.textContent = tier === LONGTERM
            ? "Пусто — решений и знаний команды у агента нет"
            : tier === WORKING
                ? "Пусто — требования этой задачи ещё не подтверждены"
                : "Свободных пунктов нет — форму ответа задают только шкалы";
        bodyEl.replaceChildren(empty);
        return;
    }

    const nodes = [];
    for (const section of config.sections[tier]) {
        const chosen = items.filter((item) => item.section === section);
        if (!chosen.length) {
            continue;
        }

        const name = document.createElement("div");
        name.className = "tier-section-name";
        name.textContent = section;

        const list = document.createElement("ul");
        list.className = "tier-items";
        list.append(...chosen.map((item) => tierItem(tier, item)));

        nodes.push(name, list);
    }
    bodyEl.replaceChildren(...nodes);
}

function tierItem(tier, item) {
    const row = document.createElement("li");
    row.className = freshItems.has(`${tier}:${item.id}`) ? "tier-item tier-item-fresh" : "tier-item";

    const text = document.createElement("span");
    text.className = "tier-item-text";
    text.textContent = item.text;

    // Откуда пункт взялся, важно именно у долговременной памяти: seed был здесь
    // до разговора, agent записал сам, user отправил кнопкой.
    const origin = document.createElement("span");
    origin.className = "tier-item-origin";
    origin.textContent = item.origin === "seed" ? "seed" : item.origin === "agent" ? "агент" : "";

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "tier-item-delete";
    remove.textContent = "×";
    remove.title = "Забыть этот пункт";
    remove.setAttribute("aria-label", `Забыть: ${item.text}`);
    remove.disabled = locked();
    remove.addEventListener("click", () => forgetItem(tier, item.id));

    row.append(text, origin, remove);
    return row;
}

// Профиль и оба блока перерисовываются вместе: перенос пункта меняет сразу два
// адреса, и рисовать их по отдельности значило бы показать состояние в промежуточном
// виде.
function setMemory(payload) {
    profile = payload.profile;
    memory = {
        [PROFILE]: profile ? profile.items : [],
        [LONGTERM]: payload.longterm,
        [WORKING]: payload.working,
    };
    renderProfile();
    renderTier(LONGTERM, longtermBodyEl, longtermMetaEl);
    renderTier(WORKING, workingBodyEl, workingMetaEl);
    copySpecButton.disabled = payload.working.length === 0;
}

// Шкалы стоят выше свободных пунктов: это костяк профиля, у него всегда есть
// значение, и правится он селектором, а не карточкой.
function renderProfile() {
    profileMetaEl.textContent = profile ? profile.name : "нет профиля";
    if (!profile) {
        scalesEl.replaceChildren();
        profileBodyEl.replaceChildren();
        return;
    }

    scalesEl.replaceChildren(...config.scales.map(scaleRow));
    renderTier(PROFILE, profileBodyEl, null);
}

function scaleRow(scale) {
    const row = document.createElement("label");
    row.className = "scale";
    row.title = scale.about;

    const name = document.createElement("span");
    name.className = "scale-name";
    name.textContent = scale.key;

    const select = document.createElement("select");
    select.className = "scale-select";
    select.disabled = locked();
    select.append(...scale.options.map((option) => {
        const node = document.createElement("option");
        node.value = option;
        node.textContent = option;
        node.selected = option === profile.scales[scale.key];
        return node;
    }));
    select.addEventListener("change", () => tuneScale(scale.key, select.value));

    row.append(name, select);
    return row;
}

// Пункт узнаётся по паре «адрес и id»: один и тот же номер строки может встретиться
// у профиля и на обоих уровнях памяти, таблицы-то разные.
function markers(payload) {
    return [
        ...(payload.profile ? payload.profile.items : []).map((item) => `${PROFILE}:${item.id}`),
        ...payload.longterm.map((item) => `${LONGTERM}:${item.id}`),
        ...payload.working.map((item) => `${WORKING}:${item.id}`),
    ];
}

function stored() {
    return new Set(markers({
        profile: profile ? { items: memory[PROFILE] } : null,
        longterm: memory[LONGTERM],
        working: memory[WORKING],
    }));
}

// Что записалось за последнее действие: без отметки блок выглядит неподвижным.
function added(payload, before) {
    return new Set(markers(payload).filter((marker) => !before.has(marker)));
}

// ТЗ уходит из памяти агента в чужой документ markdown: разделы заголовками,
// пункты списком — так его можно вставить в задачу и не переписывать руками.
function specMarkdown() {
    const lines = ["# Техническое задание"];
    for (const section of config.sections[WORKING]) {
        const chosen = memory[WORKING].filter((item) => item.section === section);
        if (!chosen.length) {
            continue;
        }
        lines.push("", `## ${section[0].toUpperCase()}${section.slice(1)}`);
        lines.push(...chosen.map((item) => `- ${item.text}`));
    }
    return lines.join("\n");
}

function isNearBottom(el, threshold = 64) {
    return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
}

function scrollToBottom(el) {
    el.scrollTop = el.scrollHeight;
}

function addMessage(role, text, id = 0) {
    const follow = isNearBottom(chatEl);
    placeholderEl.hidden = true;

    const message = document.createElement("article");
    message.className = `message message-${role}`;
    // Реплику из базы видно по id: под ответом стоят карточки его хода, и по
    // этому же номеру они возвращаются на место после перезагрузки страницы.
    if (id) {
        message.dataset.id = String(id);
    }

    const author = document.createElement("span");
    author.className = "message-author";
    author.textContent = role === "user" ? "Вы" : "Агент";

    const body = document.createElement("pre");
    body.className = "message-body";
    body.textContent = text;

    message.append(author, body);
    chatEl.append(message);

    if (follow) {
        scrollToBottom(chatEl);
    }
    return { message, body, thinking: null };
}

// Краткосрочная память — это не вся лента, а её хвост: в запрос уходят последние
// N реплик, остальные остались на экране и только на нём. Граница подписана,
// потому что иначе «агент забыл» выглядит как «агент сломался».
function markWindow(contextSize) {
    chatEl.querySelector(".window-mark")?.remove();

    const messages = [...chatEl.querySelectorAll(".message")];
    const dropped = Math.max(messages.length - contextSize, 0);
    messages.forEach((message, index) => {
        message.classList.toggle("message-dropped", index < dropped);
    });

    if (!dropped || dropped >= messages.length) {
        return;
    }

    const mark = document.createElement("div");
    mark.className = "window-mark";
    mark.textContent = `ниже — краткосрочная память, ${contextSize} ${messageWord(contextSize)}`;
    messages[dropped].before(mark);
}

// Пузырь агента появляется пустым сразу после отправки, поэтому до первого токена
// в нём живут анимированные точки: видно, что запрос ушёл и ответ готовится.
function showThinking(bubble) {
    const thinking = document.createElement("span");
    thinking.className = "thinking";
    thinking.append(...Array.from({ length: 3 }, () => document.createElement("span")));
    bubble.body.append(thinking);
    bubble.thinking = thinking;
}

function stopThinking(bubble) {
    if (!bubble.thinking) {
        return false;
    }
    bubble.thinking.remove();
    bubble.thinking = null;
    return true;
}

function appendChunk(bubble, text) {
    const follow = isNearBottom(chatEl);
    bubble.body.textContent += text;
    if (follow) {
        scrollToBottom(chatEl);
    }
}

function clearChat() {
    for (const node of [...chatEl.querySelectorAll(".message, .window-mark, .proposals")]) {
        node.remove();
    }
    placeholderEl.hidden = false;
}

// Карточка — предложение, а не запись: пока пользователь не нажал кнопку, этого
// пункта нет ни на одном уровне памяти. Живёт она в ленте под тем ответом, из
// которого её достали, и остаётся там, когда разговор идёт дальше: предложение
// без своей реплики — это просто список, и понять, откуда он взялся, нельзя.
function renderProposals(items) {
    const follow = isNearBottom(chatEl);
    cards = new Map();
    for (const group of [...chatEl.querySelectorAll(".proposals")]) {
        group.remove();
    }

    for (const [answerId, proposals] of byAnswer(items)) {
        const group = document.createElement("div");
        group.className = "proposals";
        group.append(...proposals.map(proposalCard));

        const anchor = answerId && chatEl.querySelector(`.message-agent[data-id="${answerId}"]`);
        // Ответа может не быть в ленте: очередь карточек переживает перезагрузку
        // страницы, а разбор старого хода мог и не дойти до id. Такая группа
        // встаёт в конец — решение всё равно нужно.
        if (anchor) {
            anchor.after(group);
        } else {
            chatEl.append(group);
        }
    }

    if (follow) {
        scrollToBottom(chatEl);
    }
}

// Ход за ходом, в порядке появления: карточки одного ответа стоят одной группой.
function byAnswer(items) {
    const groups = new Map();
    for (const proposal of items) {
        const key = proposal.answer_id || 0;
        groups.set(key, [...(groups.get(key) ?? []), proposal]);
    }
    return groups;
}

function proposalCard(proposal) {
    const root = document.createElement("div");
    root.className = "proposal";

    // Решённая карточка сворачивается в строку: уровень, раздел и сама
    // формулировка — всё, что нужно, чтобы вспомнить решение или передумать.
    const summary = document.createElement("div");
    summary.className = "proposal-summary";
    const badge = document.createElement("span");
    badge.className = "proposal-badge";
    const line = document.createElement("span");
    line.className = "proposal-line";
    const edit = document.createElement("button");
    edit.type = "button";
    edit.className = "proposal-edit";
    edit.textContent = "Изменить";
    summary.append(badge, line, edit);

    const head = document.createElement("div");
    head.className = "proposal-head";
    const state = document.createElement("span");
    state.className = "proposal-state";
    const why = document.createElement("span");
    why.className = "proposal-why";
    why.textContent = proposal.why;
    const collapse = document.createElement("button");
    collapse.type = "button";
    collapse.className = "proposal-collapse";
    collapse.textContent = "Свернуть";
    head.append(state, why, collapse);

    const text = document.createElement("textarea");
    text.className = "proposal-text";
    text.rows = 2;
    text.value = proposal.text;
    text.setAttribute("aria-label", "Формулировка для памяти");

    const actions = document.createElement("div");
    actions.className = "proposal-actions";

    const section = document.createElement("select");
    section.className = "proposal-section";
    section.setAttribute("aria-label", "Раздел памяти");

    const skip = document.createElement("button");
    skip.type = "button";
    skip.className = "proposal-skip";
    skip.textContent = "Отклонить";

    const card = {
        id: proposal.id,
        tier: proposal.tier,
        section: proposal.section,
        saved: proposal.saved,
        // Правка шкалы — вторая порода карточки: у неё нет ни формулировки, ни
        // раздела, зато есть шкала и новое значение, и лечь она может только в профиль.
        kind: proposal.kind,
        scale: proposal.scale,
        value: proposal.value,
        previous: proposal.previous,
        // Решённая карточка открыта, только пока её правят: решение сворачивает её
        // обратно, иначе очередь ответов заслоняла бы разговор.
        expanded: false,
        // Формулировка, которая лежит в памяти. В поле она может быть уже другой:
        // правку записывает кнопка адреса, а не сам ввод.
        storedText: proposal.text,
        root,
        state,
        sectionSelect: section,
        text,
        badge,
        line,
        edit,
        collapse,
        skip,
        buttons: {},
    };

    // У правки шкалы адрес один, и предлагать выбор незачем: кнопка сразу говорит,
    // что она делает. У обычной находки адресов три, и профиль стоит первым — он
    // ближе всего к тому, о чём его чаще всего спрашивают.
    const targets = card.kind === SCALE ? [PROFILE] : [PROFILE, LONGTERM, WORKING];
    for (const target of targets) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "button proposal-button";
        button.textContent = TARGET_BUTTONS[target];
        button.addEventListener("click", () => decide(card, target));
        card.buttons[target] = button;
        actions.append(button);
    }

    actions.append(section, skip);
    skip.addEventListener("click", () => decide(card, "skip"));
    // Раздел можно поправить и после записи: пункт переедет внутри уровня, для
    // чего его достаточно переписать на месте.
    section.addEventListener("change", () => {
        card.section = section.value;
        if (card.saved) {
            decide(card, card.saved);
        }
    });

    edit.addEventListener("click", () => {
        card.expanded = true;
        paintCard(card);
        card.text.focus();
    });

    collapse.addEventListener("click", () => {
        // Свернуть — значит отказаться от правки: в строке стоит то, что лежит в
        // памяти, и поле возвращается к ней же.
        card.text.value = card.storedText;
        card.expanded = false;
        paintCard(card);
    });

    root.append(summary, head, text, actions);
    cards.set(card.id, card);
    paintCard(card);
    return root;
}

// Вид карточки целиком определяется тем, решён ли вопрос: до решения подсвечена
// рекомендация агента, после — тот адрес, куда пункт лёг, и карточка сжимается
// в строку. Разворачивают её кнопкой, когда решение хотят поменять.
function paintCard(card) {
    if (card.kind === SCALE) {
        paintScaleCard(card);
        return;
    }

    const target = card.saved || card.tier;
    const compact = Boolean(card.saved) && !card.expanded;
    card.root.classList.toggle("proposal-saved", Boolean(card.saved));
    card.root.classList.toggle("proposal-compact", compact);
    card.state.textContent = card.saved
        ? `Сохранено в ${TARGET_NAMES[card.saved]}`
        : "Сохранить в память или в профиль?";

    const sections = config.sections[target];
    if (!sections.includes(card.section)) {
        card.section = sections[0];
    }

    card.badge.textContent = card.saved ? `${TARGET_NAMES[card.saved]} · ${card.section}` : "";
    card.line.textContent = card.storedText;
    card.edit.disabled = locked();
    // Сворачивать нечего, пока решения нет: у нерешённой карточки строки-итога
    // ещё не существует.
    card.collapse.hidden = !card.saved;
    card.skip.textContent = "Отклонить";
    card.sectionSelect.hidden = false;
    card.sectionSelect.disabled = locked();
    card.sectionSelect.replaceChildren(...sections.map((name) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        option.selected = name === card.section;
        return option;
    }));

    for (const [name, button] of Object.entries(card.buttons)) {
        const active = name === target;
        button.classList.toggle("proposal-button-quiet", !active);
        // Уже занятый адрес не предлагается повторно: на его кнопке теперь написано,
        // что пункт там, а нажимают другую — чтобы перенести.
        button.textContent = card.saved === name ? "Здесь" : TARGET_BUTTONS[name];
        button.disabled = card.saved === name || locked();
    }
}

// Карточка правки шкалы устроена проще: править нечего, переносить некуда, и весь
// вопрос — применить или нет. Зато у неё есть то, чего нет у пункта, — прежнее
// значение: без него «длина: коротко» не выглядит переменой.
function paintScaleCard(card) {
    const from = card.saved ? card.previous : profile && profile.scales[card.scale];
    const change = `${card.scale}: ${from || "—"} → ${card.value}`;

    card.root.classList.toggle("proposal-saved", Boolean(card.saved));
    card.root.classList.toggle("proposal-compact", Boolean(card.saved) && !card.expanded);
    card.state.textContent = card.saved ? "Профиль обновлён" : "Изменить профиль?";
    card.badge.textContent = card.saved ? TARGET_NAMES[PROFILE] : "";
    card.line.textContent = change;
    card.text.hidden = true;
    card.sectionSelect.hidden = true;
    card.edit.hidden = true;
    card.collapse.hidden = true;
    // Отмена у шкалы возвращает прежнее значение, а не просто убирает карточку:
    // стереть шкалу нельзя, у профиля она есть всегда.
    card.skip.textContent = card.saved ? "Отменить" : "Отклонить";
    card.skip.disabled = locked();

    const button = card.buttons[PROFILE];
    button.textContent = card.saved ? "Применено" : `Применить · ${change}`;
    button.classList.remove("proposal-button-quiet");
    button.disabled = Boolean(card.saved) || locked();
}

async function decide(card, target) {
    setCardBusy(card, true);
    try {
        const before = stored();
        const payload = await send(`/api/session/${sessionId}/memory`, {
            proposal_id: card.id,
            target,
            section: card.section,
            text: card.text.value,
        });

        applyMemory(payload, added(payload, before));

        const updated = payload.proposals.find((item) => item.id === card.id);
        if (!updated) {
            // Отклонённая карточка уходит с экрана: вопрос закрыт.
            card.root.remove();
            cards.delete(card.id);
            setStatus(card.kind === SCALE
                ? "Отменено, шкала осталась прежней"
                : "Отклонено, в память не записано");
            return;
        }

        card.tier = updated.tier;
        card.section = updated.section;
        card.saved = updated.saved;
        card.previous = updated.previous;
        // Формулировку могли почистить на сервере: в строке-итоге должно стоять
        // ровно то, что легло в память.
        card.storedText = updated.text;
        card.text.value = updated.text;
        card.expanded = false;
        paintCard(card);
        setStatus(card.kind === SCALE
            ? `Профиль: ${card.scale} — ${card.value}. Следующий ответ уйдёт уже по нему`
            : `Записано в ${TARGET_NAMES[updated.saved]} · ${updated.section}`
                + " — блок уходит в каждый запрос");
    } catch (error) {
        setStatus(error.message, true);
    } finally {
        setCardBusy(card, false);
    }
}

function setCardBusy(card, busy) {
    if (!busy) {
        // Разрешения расставляет paintCard: у сохранённой карточки часть кнопок
        // выключена и без всякого запроса.
        paintCard(card);
        return;
    }
    card.sectionSelect.disabled = true;
    card.edit.disabled = true;
    card.skip.disabled = true;
    for (const button of Object.values(card.buttons)) {
        button.disabled = true;
    }
}

// Ждёт полный снимок: и оба блока памяти, и счётчики запроса. Столько отдают
// открытие диалога, запись пункта и удаление — то есть всё, кроме кадра memory
// посреди хода, которому хватает setMemory.
function applyMemory(payload, fresh = new Set()) {
    freshItems = fresh;
    setMemory(payload);
    setContext(payload);
}

async function forgetItem(target, itemId) {
    if (locked()) {
        return;
    }

    // У профиля свой путь: он не уровень памяти, и стирать его пункты через ручку
    // памяти значило бы делать вид, что это одно и то же хранилище.
    const path = target === PROFILE
        ? `/api/session/${sessionId}/profile/item/${itemId}`
        : `/api/session/${sessionId}/memory/${target}/${itemId}`;

    try {
        const payload = await send(path, null, "DELETE");
        applyMemory(payload);
        // Карточка удалённого пункта снова ждёт решения: сервер вернул её без
        // отметки, и страница это показывает.
        renderProposals(payload.proposals);
        setStatus(`Забыто из ${TARGET_NAMES[target]}`);
    } catch (error) {
        setStatus(error.message, true);
    }
}

// Шкала правится без карточки и сразу: выбор из трёх слов не нуждается ни в
// формулировке, ни в решении, куда его положить.
async function tuneScale(scale, value) {
    try {
        const payload = await send(`/api/session/${sessionId}/profile/scale`, { scale, value }, "PUT");
        applyMemory(payload);
        setStatus(`Профиль: ${scale} — ${value}. Следующий ответ уйдёт уже по нему`);
    } catch (error) {
        setStatus(error.message, true);
        // Селектор показывает выбранное, а профиль остался прежним: возвращаем блок
        // к тому, что на самом деле уйдёт в запрос.
        renderProfile();
    }
}

// Профиль меняется у диалога, а не у приложения: два диалога рядом могут идти от
// лица разных людей, и один и тот же вопрос в них отличается только этим.
async function switchProfile(id) {
    try {
        const payload = await send(`/api/session/${sessionId}/profile`, null, "PATCH", { profile_id: id });
        applyMemory(payload);
        // В панели у диалога подписан профиль — после смены подпись должна съехать
        // вместе с ним.
        refreshDialogs();
        setStatus(`Профиль диалога: ${payload.profile.name}.`
            + " Память та же, форма ответов другая");
    } catch (error) {
        setStatus(error.message, true);
        profileSelect.value = profileId;
    }
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

async function ask(prompt, question, answer) {
    const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            session_id: sessionId,
            prompt,
            window_messages: windowMessages,
            autosave,
        }),
        signal: controller.signal,
    });

    if (response.status === 404) {
        throw new Error("Сессия истекла");
    }
    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const before = stored();
    let buffer = "";
    let update = null;

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
            if (event === "prompt") {
                const payload = JSON.parse(data);
                setContext(payload);
                setStatus(promptStatus(payload));
                continue;
            }
            if (event === "turn") {
                // Ход лёг в базу и получил номера реплик. Приходят они до разбора
                // хода: карточки встают под свой ответ, а найти его можно только
                // по id, который до этого кадра пузырь на экране не знал.
                const payload = JSON.parse(data);
                question.message.dataset.id = String(payload.question_id);
                answer.message.dataset.id = String(payload.answer_id);
                continue;
            }
            if (event === "memory") {
                // Разбор хода идёт после ответа: он уже на экране, а строка
                // состояния показывает, что агент решает, что из него запомнить.
                update = JSON.parse(data);
                // В этом кадре едет только память: состав запроса на момент разбора
                // не пересобирали, и счётчики в шапке трогает следующий кадр.
                freshItems = added(update, before);
                setMemory(update);
                renderProposals(update.proposals);
                setStatus(memoryStatus(update));
                continue;
            }
            if (event === "done") {
                const payload = JSON.parse(data);
                stopThinking(answer);
                // Блоки уже нарисовал кадр memory, и отметки о свежих пунктах
                // переживают конец хода: перерисовывать их нечем.
                if (!update) {
                    applyMemory(payload);
                    renderProposals(payload.proposals);
                } else {
                    setContext(payload);
                }
                markWindow(payload.context_size);
                return { ...payload, update };
            }

            const text = JSON.parse(data);
            if (stopThinking(answer)) {
                setStatus("Агент отвечает...");
            }
            appendChunk(answer, text);
        }
    }

    throw new Error("Поток оборвался");
}

function promptStatus(payload) {
    const parts = [`В запросе профиль «${payload.profile_name}»`];
    parts.push(`${payload.context_size} ${messageWord(payload.context_size)}`);
    const stored = payload.longterm_size + payload.working_size;
    if (stored > 0) {
        parts.push(`память ${stored} ${itemWord(stored)}`);
    }
    if (payload.dropped_messages > 0) {
        parts.push(`за окном ${payload.dropped_messages} сообщ.`);
    }
    return parts.join(", ");
}

function memoryStatus(payload) {
    if (!payload.found) {
        return "Разбор хода: записывать нечего";
    }

    const waiting = payload.proposals.filter((item) => !item.saved).length;
    const tail = payload.saved
        ? `агент сохранил ${payload.saved}`
        : `${waiting} ${plural(waiting, "ждёт", "ждут", "ждут")} решения`;
    return `Разбор хода: ${payload.found} ${plural(payload.found, "находка", "находки", "находок")}, ${tail}`;
}

function doneStatus(payload) {
    const parts = [
        `Ход в памяти: лента ${payload.history_size} ${messageWord(payload.history_size)},`
        + ` в запросе было ${payload.context_size}`,
        `ответ ${payload.word_count} ${plural(payload.word_count, "слово", "слова", "слов")}`,
    ];
    if (payload.finish_reason === "length") {
        parts.unshift("Ответ обрезан по лимиту");
    }
    return parts.join(" · ");
}

// Один ход диалога — и для реплики из поля, и для реплики прогона: возвращает
// ошибку, из-за которой ход не дошёл, или null. Прогон по ней и решает,
// продолжать ли, а поле ввода получает свой текст назад только при ручной отправке.
async function turn(prompt, { restore = true } = {}) {
    const question = addMessage("user", prompt);
    const answer = addMessage("agent", "");
    showThinking(answer);
    controller = new AbortController();
    setBusy(true);
    setStatus("Агент думает...");

    try {
        const payload = await ask(prompt, question, answer);
        setStatus(doneStatus(payload));
        refreshDialogs();
        return null;
    } catch (error) {
        // Оборванный ход агент в память не записал — убираем его и из ленты,
        // чтобы на экране не осталось того, чего собеседник не помнит. Карточки
        // этого хода уходят вместе с ответом: без своей реплики они беспризорны.
        const orphans = answer.message.nextElementSibling;
        if (orphans?.classList.contains("proposals")) {
            orphans.remove();
        }
        question.message.remove();
        answer.message.remove();
        placeholderEl.hidden = chatEl.querySelector(".message") !== null;

        if (error.name === "AbortError") {
            setStatus("Остановлено, ход не сохранён в памяти");
        } else if (error.message === "Сессия истекла" && !run) {
            await createSession();
            setStatus("Сессия истекла, начат новый диалог", true);
        } else {
            setStatus(error.message, true);
        }
        if (restore) {
            // Новая сессия успела подставить в поле приветственный вопрос: реплику
            // возвращаем после неё, иначе текст пропал бы вместе с диалогом.
            promptInput.value = prompt;
        }
        return error;
    } finally {
        controller = null;
        setBusy(false);
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
        setStatus("Введите сообщение", true);
        return;
    }

    promptInput.value = "";
    await turn(prompt);
    promptInput.focus();
});

// Автопрогон печатает за пользователя тот же диалог, что проходит замер в
// scenarios.py: ходы идут подряд через тот же запрос, что и ручная отправка, и
// профиль с памятью и карточками меняется на экране обычным путём.
async function startAutorun() {
    const prompts = config.autorun;
    const chosenAutosave = autosave;
    run = { index: 1, total: prompts.length, stopped: false };
    let failure = null;
    let done = 0;

    setBusy(false);
    setStatus("готовим чистый диалог, адрес для находок выбирает агент");

    try {
        // Прогон начинается с пустой рабочей памяти: в непустом диалоге было бы
        // видно не то, как собирается ТЗ, а как оно смешивается с прошлой задачей.
        if (historySize > 0) {
            await createSession();
        }
        // Прогон идёт от лица того же профиля, что и замер: в диалоге пользователь
        // просит отвечать короче и без списков, и у этого профиля обе просьбы —
        // настоящая правка, а не подтверждение уже стоящего значения.
        if (profileId !== config.autorunProfile) {
            await switchProfile(config.autorunProfile);
        }
        // Адрес на время прогона выбирает агент: к кнопке в карточке прогон не
        // пойдёт, а без записи смотреть было бы не на что. Выбор пользователя при
        // этом не переписывается — в localStorage временная галочка не уходит.
        setAutosave(true);

        for (const [index, prompt] of prompts.entries()) {
            if (run.stopped) {
                break;
            }
            run.index = index + 1;
            failure = await turn(prompt, { restore: false });
            if (failure) {
                break;
            }
            done += 1;
        }
    } catch (error) {
        failure = error;
    } finally {
        const { total, stopped } = run;
        run = null;
        setAutosave(chosenAutosave);
        setBusy(false);

        const stored = memory[LONGTERM].length + memory[WORKING].length;
        if (done === total) {
            setStatus(`Прогон завершён: ${total} ${plural(total, "ход", "хода", "ходов")},`
                + ` профиль «${profile ? profile.name : "—"}» правился по ходу,`
                + ` память ${memory[LONGTERM].length}+${memory[WORKING].length} ${itemWord(stored)}`);
        } else if (stopped || failure?.name === "AbortError") {
            setStatus(`Прогон остановлен на ходу ${done + 1} из ${total}:`
                + ` пройденное осталось в памяти, ${stored} ${itemWord(stored)}`);
        } else {
            setStatus(`Прогон прерван на ходу ${done + 1} из ${total}: ${failure?.message ?? "ход не дошёл"}`, true);
        }
        promptInput.focus();
    }
}

autorunButton.addEventListener("click", () => {
    if (run) {
        // Между ходами обрывать нечего: флаг остановит цикл перед следующей репликой.
        run.stopped = true;
        controller?.abort();
        setStatus("останавливаем прогон...");
        return;
    }
    startAutorun();
});

// Один вопрос всем профилям сразу. Идёт мимо диалога: у ответов нет ни ленты, ни
// ТЗ — только общая долговременная память и разные профили. Иначе сравнение мерило
// бы заодно и то, что у одного из профилей была история.
async function askEveryone() {
    const prompt = promptInput.value.trim() || config.compare[0];

    comparing = true;
    setCompareButton();
    setStatus(`Спрашиваем все профили: «${prompt}»`);

    try {
        const response = await fetch("/api/compare", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt }),
        });
        if (!response.ok) {
            throw new Error(`Сервер вернул ${response.status}`);
        }

        const payload = await response.json();
        renderCompare(payload);
        compareDialog.showModal();
        setStatus(`Сравнение готово: ${payload.answers.length} ${plural(payload.answers.length, "ответ", "ответа", "ответов")}`
            + " на один и тот же вопрос");
    } catch (error) {
        setStatus(error.message, true);
    } finally {
        comparing = false;
        setCompareButton();
    }
}

function renderCompare(payload) {
    compareQuestionEl.textContent = payload.prompt;
    compareBodyEl.replaceChildren(...payload.answers.map(compareColumn));
}

function compareColumn(item) {
    const column = document.createElement("article");
    column.className = "compare-column";

    const name = document.createElement("h3");
    name.className = "compare-name";
    name.textContent = item.name;

    // Шкалы над ответом: без них разница в ответах выглядит капризом модели.
    const scales = document.createElement("p");
    scales.className = "compare-scales";
    scales.textContent = Object.values(item.scales ?? {}).join(" · ");

    const body = document.createElement("pre");
    body.className = "compare-answer";
    body.textContent = item.answer ?? item.error;

    // Мерки считает сервер: в панели и в таблице README стоят одни и те же числа.
    const shape = document.createElement("p");
    shape.className = "compare-shape";
    shape.textContent = item.shape ? item.shape.line : "";

    column.append(name, scales, body, shape);
    return column;
}

compareButton.addEventListener("click", askEveryone);

compareCloseButton.addEventListener("click", () => {
    compareDialog.close();
    promptInput.focus();
});

promptInput.addEventListener("keydown", (event) => {
    // Shift + Enter — перенос строки. event.isComposing отсекает Enter, которым
    // подтверждают подсказку IME: это ввод слова, а не отправка сообщения.
    if (event.key !== "Enter" || event.shiftKey || event.isComposing) {
        return;
    }
    event.preventDefault();
    // Пока идёт ответ, та же кнопка означает «Остановить» — прерывать диалог
    // случайным Enter не стоит, остановка остаётся осознанным кликом.
    if (!locked()) {
        form.requestSubmit();
    }
});

newDialogButton.addEventListener("click", async () => {
    // Открытый диалог пуст — вторая такая же сессия только замусорит панель.
    if (historySize === 0) {
        setStatus("Этот диалог ещё пуст");
        promptInput.focus();
        return;
    }

    try {
        await createSession();
        setStatus("Новый диалог: рабочая память пуста, профиль и долговременная на месте");
    } catch (error) {
        setStatus(error.message, true);
    }
    promptInput.focus();
});

resetButton.addEventListener("click", async () => {
    if (locked()) {
        return;
    }
    if (!confirm("Сбросить базу до исходной? Диалоги, правки профилей и всё, что добавили в память, пропадут.")) {
        return;
    }

    resetButton.disabled = true;
    try {
        const response = await fetch("/api/reset", { method: "POST" });
        if (!response.ok) {
            throw new Error(`Сервер вернул ${response.status}`);
        }
        const payload = await response.json();
        await loadDialogs();
        if (!(await openSession(payload.session_id))) {
            throw new Error("После сброса диалог не открылся");
        }
        const seeded = memory[LONGTERM].length;
        setStatus(`База сброшена: профили как в seed, память ${seeded} ${itemWord(seeded)}, диалоги пустые`);
    } catch (error) {
        setStatus(error.message, true);
    }
    resetButton.disabled = locked();
    promptInput.focus();
});

copySpecButton.addEventListener("click", async () => {
    try {
        await navigator.clipboard.writeText(specMarkdown());
        setStatus(`ТЗ скопировано: ${memory[WORKING].length} ${itemWord(memory[WORKING].length)}`);
    } catch (error) {
        setStatus("Буфер обмена недоступен", true);
    }
});

function renderDialogs() {
    dialogsEl.replaceChildren(...dialogs.map(dialogItem));
}

function dialogItem(dialog) {
    const item = document.createElement("li");
    item.className = "dialog";
    item.classList.toggle("active", dialog.id === sessionId);

    const open = document.createElement("button");
    open.type = "button";
    open.className = "dialog-open";
    // Заголовок сессии — суть первого запроса, у пустой его ещё нет.
    const title = dialog.title || "Новый диалог";
    open.title = dialog.working
        ? `${title} — ТЗ на ${dialog.working} ${itemWord(dialog.working)}`
        : title;
    if (dialog.id === sessionId) {
        open.setAttribute("aria-current", "true");
    }

    const name = document.createElement("span");
    name.className = "dialog-name";
    name.textContent = title;

    // Реплики и размер ТЗ: по второму числу видно, в какой задаче рабочая память
    // уже собрана, а какая только началась.
    const size = document.createElement("span");
    size.className = "dialog-size";
    size.textContent = dialog.working ? `${dialog.size} · ${dialog.working}` : dialog.size;

    // От чьего лица шёл разговор: один и тот же вопрос от двух профилей иначе не
    // отличить, а в этом дне два таких диалога рядом — обычное дело.
    const person = document.createElement("span");
    person.className = "dialog-profile";
    person.textContent = dialog.profile || "";

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "dialog-delete";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Удалить диалог «${title}»`);
    remove.title = "Удалить диалог вместе с его рабочей памятью";

    open.append(name, size, person);
    open.addEventListener("click", () => switchSession(dialog.id));
    remove.addEventListener("click", () => deleteDialog(dialog.id));
    item.append(open, remove);
    return item;
}

async function loadDialogs() {
    const response = await fetch("/api/sessions");
    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }

    dialogs = (await response.json()).sessions;
    renderDialogs();
}

// Список обновляется после каждого хода: после первого появляется заголовок,
// а диалог поднимается наверх. Панель — не лента, сбой обновления не стоит ошибки.
function refreshDialogs() {
    loadDialogs().catch(() => {});
}

// Новый диалог наследует профиль открытого: человек за клавиатурой не меняется от
// того, что он начал вторую задачу.
async function createSession() {
    const query = profileId ? `?profile_id=${encodeURIComponent(profileId)}` : "";
    const response = await fetch(`/api/session${query}`, { method: "POST" });
    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }

    // Пустое состояние страница не выдумывает, а берёт с сервера тем же запросом,
    // что и при открытии готового диалога: у новой сессии уже есть память.
    const payload = await response.json();
    await loadDialogs();
    if (!(await openSession(payload.session_id))) {
        throw new Error("Новый диалог не открылся");
    }
}

// Диалог хранится в базе, поэтому и лента, и оба блока памяти восстанавливаются
// запросом сессии — и при переключении, и после перезагрузки страницы.
async function openSession(id) {
    const state = await loadState(id);
    if (!state) {
        return false;
    }

    sessionId = id;
    localStorage.setItem(SESSION_KEY, id);
    showState(state);
    promptInput.value = state.history.length ? "" : config.prompt;
    renderDialogs();
    return true;
}

function showState(state) {
    clearChat();
    for (const message of state.history) {
        addMessage(message.role === "user" ? "user" : "agent", message.content, message.id);
    }
    applyMemory(state);
    // Нерешённые карточки переживают перезагрузку страницы, но не перезапуск
    // сервера: они часть краткосрочной памяти и живут в агенте, а не в базе.
    renderProposals(state.proposals);
    markWindow(state.context_size);
    scrollToBottom(chatEl);
}

// Окно уходит в запрос вместе с настройками: кадр контекста собирает сервер, и
// от размера окна зависит, что в него попадёт.
function settings(extra = {}) {
    return new URLSearchParams({ window_messages: windowMessages, ...extra });
}

async function loadState(id) {
    const response = await fetch(`/api/session/${id}?${settings()}`);
    return response.ok ? await response.json() : null;
}

async function send(path, body = null, method = "POST", extra = {}) {
    const response = await fetch(`${path}?${settings(extra)}`, {
        method,
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
    });
    if (!response.ok) {
        const detail = await response.json().catch(() => null);
        throw new Error(detail?.detail || `Сервер вернул ${response.status}`);
    }
    return await response.json();
}

async function switchSession(id) {
    if (locked() || id === sessionId) {
        promptInput.focus();
        return;
    }

    try {
        if (!(await openSession(id))) {
            // Диалог удалили из другой вкладки — панель показывает то, чего нет.
            await loadDialogs();
            throw new Error("Диалог не найден");
        }
        setStatus("");
    } catch (error) {
        setStatus(error.message, true);
    }
    promptInput.focus();
}

async function deleteDialog(id) {
    if (locked()) {
        return;
    }

    try {
        const response = await fetch(`/api/session/${id}`, { method: "DELETE" });
        // 404 — диалога уже нет, цель достигнута без нас.
        if (!response.ok && response.status !== 404) {
            throw new Error(`Сервер вернул ${response.status}`);
        }

        await loadDialogs();
        if (id === sessionId) {
            // Удалён открытый диалог: показываем соседний, а если список опустел —
            // заводим новый, странице всегда нужна живая сессия.
            await openDialogsHead();
        }
        setStatus("Диалог удалён вместе с его рабочей памятью");
    } catch (error) {
        setStatus(error.message, true);
    }
    promptInput.focus();
}

async function openDialogsHead() {
    if (!dialogs.length || !(await openSession(dialogs[0].id))) {
        await createSession();
    }
}

async function changeWindow(value) {
    windowMessages = value;
    localStorage.setItem(WINDOW_KEY, String(value));

    const state = await reloadState();
    if (!state) {
        return;
    }
    const dropped = state.history_size - state.context_size;
    setStatus(dropped > 0
        ? `Окно ${value} ${messageWord(value)}: за границей ${dropped} сообщ.,`
        + ` из них в памяти только то, что подтвердили`
        : `Окно ${value} ${messageWord(value)}: весь диалог влезает в запрос`);
}

async function reloadState() {
    const state = await loadState(sessionId);
    if (state) {
        showState(state);
    }
    return state;
}

async function init() {
    const response = await fetch("/api/defaults");
    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }

    const payload = await response.json();
    config = {
        prompt: payload.prompt,
        windowOptions: payload.window_options,
        sections: payload.sections,
        // Шкалы профиля со всеми значениями: свои значения в шкалу не придумать,
        // список задаёт сервер.
        scales: payload.scales,
        profiles: payload.profiles,
        // Реплики прогона приходят с сервера: тот же диалог, что проходит замер.
        autorun: payload.autorun,
        autorunProfile: payload.autorun_profile,
        compare: payload.compare,
    };

    // Вкладка помнит выбранные настройки, но шкалу задаёт сервер: чужое значение
    // из localStorage в список не попадёт.
    const saved = (key, options, fallback) => {
        const value = Number(localStorage.getItem(key));
        return options.includes(value) ? value : fallback;
    };
    windowMessages = saved(WINDOW_KEY, config.windowOptions, payload.window_messages);
    const savedAutosave = localStorage.getItem(AUTOSAVE_KEY);
    autosave = savedAutosave === null ? payload.autosave : savedAutosave === "1";
    // Профиль страница не помнит: он принадлежит диалогу, и его приносит снимок
    // сессии. Селектор до этого стоит на первом из списка.
    profileId = config.profiles.length ? config.profiles[0].id : null;

    // Параметры агента постоянны для всего диалога, меняются только счётчики.
    addParam(payload.model);
    addParam(`temperature ${payload.temperature}`);
    profileSelect = addSelectParam(
        "профиль ",
        config.profiles.map((person) => ({ value: person.id, label: person.name })),
        profileId,
    );
    windowSelect = addSelectParam(
        "окно ",
        config.windowOptions.map((value) => ({ value, label: `${value} сообщ.` })),
        windowMessages,
    );
    autosaveInput = addToggleParam(
        "агент сохраняет сам",
        autosave,
        "Агент записывает находки по своей рекомендации, не дожидаясь кнопки. Карточка всё равно покажет, куда именно",
    );
    memoryParam = addParam("");
    contextParam = addParam("");
    tierParam = addParam("");

    profileSelect.addEventListener("change", () => {
        switchProfile(profileSelect.value);
    });

    windowSelect.addEventListener("change", () => {
        changeWindow(Number(windowSelect.value)).catch((error) => setStatus(error.message, true));
    });

    autosaveInput.addEventListener("change", () => {
        setAutosave(autosaveInput.checked);
        localStorage.setItem(AUTOSAVE_KEY, autosave ? "1" : "0");
        setStatus(autosave
            ? "Адрес выбирает агент — карточка покажет, куда он положил"
            : "Адрес выбираете вы: агент только предлагает");
    });

    await loadDialogs();

    // Вкладка помнит последний открытый диалог, но переписка принадлежит базе:
    // если сессии там уже нет, страница поднимает самую свежую из списка.
    const savedSession = localStorage.getItem(SESSION_KEY);
    if (!savedSession || !(await openSession(savedSession))) {
        await openDialogsHead();
    }

    for (const control of [submitButton, newDialogButton, resetButton]) {
        control.disabled = false;
    }
    autorunButton.title = `Прогон ${config.autorun.length} реплик в новом диалоге:`
        + " адрес для находок на это время выбирает агент,"
        + " а профиль на время прогона — тот, у которого просьбы из диалога меняют шкалы";
    compareButton.title = "Текст из поля (или первый готовый вопрос) уходит каждому профилю"
        + " в отдельном запросе без диалога: видно только разницу профилей";
    setAutorunButton();
    setCompareButton();
}

init().catch((error) => {
    setStatus(error.message, true);
});
