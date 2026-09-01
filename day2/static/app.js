const form = document.getElementById("form");
const promptInput = document.getElementById("prompt");
const submitButton = document.getElementById("submit");
const statusEl = document.getElementById("status");
const constraintsEl = document.getElementById("constraints");

const columns = {
    free: {
        result: document.getElementById("resultFree"),
        copy: document.getElementById("copyFree"),
        meta: document.getElementById("metaFree"),
        text: "",
    },
    controlled: {
        result: document.getElementById("resultControlled"),
        copy: document.getElementById("copyControlled"),
        meta: document.getElementById("metaControlled"),
        text: "",
    },
};

function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", isError);
}

function setColumnText(column, text) {
    column.text = text;
    column.result.textContent = text;
    column.copy.hidden = !text;
}

function setColumnMeta(column, text, isError = false) {
    column.meta.textContent = text;
    column.meta.classList.toggle("error", isError);
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

async function ask(prompt, constrained, column) {
    const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, constrained }),
    });

    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }

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
                throw new Error(JSON.parse(data));
            }
            if (event === "done") {
                applyDone(column, JSON.parse(data));
                return;
            }
            setColumnText(column, column.text + JSON.parse(data));
        }
    }
}

function applyDone(column, payload) {
    const parts = [`слов: ${payload.word_count}`];
    if (payload.finish_reason) {
        parts.push(`finish_reason: ${payload.finish_reason}`);
    }
    setColumnMeta(column, parts.join(" · "));
}

async function renderConstraints() {
    const response = await fetch("/api/constraints");
    if (!response.ok) {
        constraintsEl.textContent = "Не удалось загрузить ограничения";
        return;
    }

    const { sections, params } = await response.json();
    constraintsEl.replaceChildren();

    for (const section of sections) {
        const block = document.createElement("div");
        block.className = "constraint";

        const title = document.createElement("span");
        title.className = "constraint-title";
        title.textContent = section.title;
        block.append(title);

        const list = document.createElement("ul");
        list.className = "constraint-rules";
        for (const rule of section.rules) {
            const item = document.createElement("li");
            item.textContent = rule;
            list.append(item);
        }
        block.append(list);

        constraintsEl.append(block);
    }

    const paramsEntries = Object.entries(params);
    if (paramsEntries.length) {
        const block = document.createElement("div");
        block.className = "constraint";

        const title = document.createElement("span");
        title.className = "constraint-title";
        title.textContent = "Параметры API";
        block.append(title);

        const list = document.createElement("ul");
        list.className = "constraint-rules";
        for (const [key, value] of paramsEntries) {
            const item = document.createElement("li");
            item.textContent = `${key}: ${JSON.stringify(value)}`;
            list.append(item);
        }
        block.append(list);

        constraintsEl.append(block);
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const prompt = promptInput.value.trim();
    if (!prompt) {
        setStatus("Введите запрос", true);
        return;
    }

    submitButton.disabled = true;
    setStatus("Генерация ответов...");
    setColumnText(columns.free, "");
    setColumnText(columns.controlled, "");
    setColumnMeta(columns.free, "");
    setColumnMeta(columns.controlled, "");

    try {
        await Promise.all([
            ask(prompt, false, columns.free),
            ask(prompt, true, columns.controlled),
        ]);
        setStatus("Готово");
    } catch (error) {
        setStatus(error.message, true);
    } finally {
        submitButton.disabled = false;
    }
});

promptInput.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        form.requestSubmit();
    }
});

function bindCopy(column) {
    column.copy.addEventListener("click", async () => {
        await navigator.clipboard.writeText(column.text);
        column.copy.textContent = "Скопировано";
        setTimeout(() => {
            column.copy.textContent = "Копировать";
        }, 1500);
    });
}

bindCopy(columns.free);
bindCopy(columns.controlled);
renderConstraints();
