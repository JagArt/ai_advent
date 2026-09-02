const form = document.getElementById("form");
const promptInput = document.getElementById("prompt");
const submitButton = document.getElementById("submit");
const statusEl = document.getElementById("status");
const modeStep = document.getElementById("modeStep");
const modeMeta = document.getElementById("modeMeta");
const modeExperts = document.getElementById("modeExperts");
const expertsPanel = document.getElementById("expertsPanel");
const expertChips = document.getElementById("expertChips");
const expertName = document.getElementById("expertName");
const addExpert = document.getElementById("addExpert");
const generatedBox = document.getElementById("generatedBox");
const generatedPrompt = document.getElementById("generatedPrompt");
const resultEl = document.getElementById("result");
const copyButton = document.getElementById("copy");
const metaEl = document.getElementById("meta");

let experts = [];
let answerText = "";
let generatedText = "";

function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", isError);
}

function isNearBottom(el, threshold = 48) {
    return el.scrollHeight - el.scrollTop - el.clientHeight <= threshold;
}

function scrollToBottom(el) {
    el.scrollTop = el.scrollHeight;
}

function setPlaceholder() {
    answerText = "";
    resultEl.replaceChildren();
    const span = document.createElement("span");
    span.className = "placeholder";
    span.textContent = "Здесь появится ответ";
    resultEl.append(span);
    copyButton.hidden = true;
}

function setResult(text) {
    const follow = isNearBottom(resultEl);
    answerText = text;
    resultEl.textContent = text;
    copyButton.hidden = !text;
    if (follow) {
        scrollToBottom(resultEl);
    }
}

function setMeta(text, isError = false) {
    metaEl.textContent = text;
    metaEl.classList.toggle("error", isError);
}

function setGeneratedPrompt(text) {
    const follow = isNearBottom(generatedPrompt);
    generatedText = text;
    generatedPrompt.textContent = text;
    generatedBox.hidden = !text;
    if (follow) {
        scrollToBottom(generatedPrompt);
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

function applyDone(payload) {
    const parts = [`слов: ${payload.word_count}`];
    if (payload.finish_reason) {
        parts.push(`finish_reason: ${payload.finish_reason}`);
    }
    setMeta(parts.join(" · "));
}

async function ask(prompt, expertNames) {
    const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            prompt,
            step_by_step: modeStep.checked,
            meta_prompt: modeMeta.checked,
            experts: modeExperts.checked ? expertNames : [],
        }),
    });

    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let stage = "answer";

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
            if (event === "stage") {
                stage = JSON.parse(data);
                if (stage === "prompt") {
                    setGeneratedPrompt("");
                }
                continue;
            }
            if (event === "done") {
                applyDone(JSON.parse(data));
                continue;
            }

            const chunk = JSON.parse(data);
            if (stage === "prompt") {
                setGeneratedPrompt(generatedText + chunk);
            } else {
                setResult(answerText + chunk);
            }
        }
    }
}

function renderExpertChips() {
    expertChips.replaceChildren();
    experts.forEach((name, index) => {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.append(name);

        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "chip-remove";
        remove.setAttribute("aria-label", `Удалить ${name}`);
        remove.textContent = "×";
        remove.addEventListener("click", () => {
            experts.splice(index, 1);
            renderExpertChips();
        });

        chip.append(remove);
        expertChips.append(chip);
    });
}

function addExpertName() {
    const name = expertName.value.trim();
    if (!name) {
        return;
    }
    if (experts.some((item) => item.toLowerCase() === name.toLowerCase())) {
        expertName.value = "";
        return;
    }
    experts.push(name);
    expertName.value = "";
    renderExpertChips();
}

function syncExpertsPanel() {
    expertsPanel.hidden = !modeExperts.checked;
}

modeExperts.addEventListener("change", syncExpertsPanel);
addExpert.addEventListener("click", addExpertName);
expertName.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        event.preventDefault();
        addExpertName();
    }
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const prompt = promptInput.value.trim();
    if (!prompt) {
        setStatus("Введите задачу", true);
        return;
    }

    if (modeExperts.checked && experts.length === 0) {
        setStatus("Добавьте хотя бы одного эксперта", true);
        return;
    }

    submitButton.disabled = true;
    setStatus("Генерация ответа...");
    setMeta("");
    setPlaceholder();
    setGeneratedPrompt("");

    try {
        await ask(prompt, experts);
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

copyButton.addEventListener("click", async () => {
    await navigator.clipboard.writeText(answerText);
    copyButton.textContent = "Скопировано";
    setTimeout(() => {
        copyButton.textContent = "Копировать";
    }, 1500);
});

async function loadDefaults() {
    const response = await fetch("/api/defaults");
    if (!response.ok) {
        throw new Error(`Сервер вернул ${response.status}`);
    }
    const payload = await response.json();
    if (typeof payload.prompt === "string") {
        promptInput.value = payload.prompt;
    }
    experts = Array.isArray(payload.experts)
        ? payload.experts.filter((name) => typeof name === "string" && name.trim())
        : [];
    renderExpertChips();
}

syncExpertsPanel();
loadDefaults().catch((error) => {
    setStatus(error.message, true);
});
