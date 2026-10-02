"""Этап до поиска: вопрос человека переписывается в то, чем лучше искать.

Нужда в этом этапе измерена, а не выдумана. В [day21](../day21/README.md) один и
тот же индекс спрашивали трижды: коротким запросом, вопросом на естественном
языке и самим эталонным отрывком. Отрывок находил свой чанк в 84 случаях из ста,
вопрос про тот же отрывок — в 23. Разрыв в 0.6 recall лежит не в индексе и не в
разбиении: он в том, что вопрос и ответ написаны разными словами. Человек
спрашивает «на каком порту слушает хранилище», а в тексте стоит
`PORT = 8770` и «постоянный сервер поднимается на 127.0.0.1:8770».

Отсюда два режима, и они бьют в этот разрыв с двух сторон.

`keyword` остаётся на стороне вопроса: модель выбрасывает связки и добавляет
слова, которыми ответ скорее всего записан, — имена модулей, функций, терминов.
Запрос остаётся запросом, просто более похожим на текст.

`hyde` переходит на сторону ответа целиком: модель пишет короткую выдержку —
такую, какой она была бы, если бы отвечала на вопрос, — и искать идут по ней.
Верна эта выдержка или нет, неважно: она не показывается никому и ни во что не
попадает, кроме выражения FTS5. Это и есть приём HyDE, и на этом корпусе он
интересен тем, что воспроизводит ровно ту пробу day21, которая дала 0.84.

Цена — вызов модели до поиска, то есть задержка и токены раньше, чем начался сам
RAG. Поэтому переписанные запросы кэшируются в [rewrites.json](rewrites.json) и
коммитятся: матрица режимов на 119 пробах иначе стоила бы сотни вызовов на каждый
прогон и не воспроизводилась бы между ними. Кэш лежит списком, а не словарём по
хэшу, нарочно — его надо читать глазами: что модель придумала из вопроса, видно
только так.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

import llm

CACHE_PATH = Path(__file__).resolve().parent / "rewrites.json"

MODES = ("none", "keyword", "hyde")
MODE_TITLES = {
    "none": "как есть",
    "keyword": "термы лексики",
    "hyde": "гипотетический ответ · HyDE",
}

# Ниже этого переписанный запрос считается сорванным, и в поиск уходит исходный
# вопрос. Модель иногда отвечает отказом или пустой строкой, и молча искать по
# такому «запросу» значило бы получить нулевой recall и записать его режиму.
MIN_CHARS = 8

TEMPERATURE = 0.0
MAX_TOKENS = 220
CONCURRENCY = 5

KEYWORD = (
    "Ты переписываешь вопрос о проекте AI Advent в поисковый запрос для "
    "полнотекстового индекса: bm25 по словам, без понимания смысла.\n"
    "В индексе лежат README дней, тексты заданий и код на Python — имена файлов "
    "вида day20/servers/vault.py, идентификаторы вида sse_frame и AgentRegistry, "
    "номера портов, названия разделов.\n"
    "Правила:\n"
    "— сохрани значимые слова вопроса и добавь те, которыми ответ скорее всего "
    "записан в тексте: имена модулей, функций, таблиц, параметров, синонимы, "
    "термин по-английски рядом с русским;\n"
    "— выброси связки и вежливость: они ничего не отсеивают;\n"
    "— не придумывай чисел, имён файлов и версий, которых в вопросе нет;\n"
    "— ответь одной строкой слов через пробел, без запятых и без пояснений."
)

HYDE = (
    "Ты пишешь короткую выдержку из документации проекта AI Advent — такую, какой "
    "она была бы, если бы отвечала на заданный вопрос.\n"
    "Это не ответ пользователю, а приманка для поиска по словам. Важно не то, "
    "верна ли выдержка, а то, какими словами она написана.\n"
    "Правила:\n"
    "— два-три предложения, по-русски, в стиле технической документации;\n"
    "— называй вещи так, как их называют в коде и README: имена файлов, модулей, "
    "функций, таблиц, параметров, номера портов;\n"
    "— без вступлений, без повтора вопроса и без оговорок о незнании."
)

PROMPTS = {"keyword": KEYWORD, "hyde": HYDE}


@dataclass(frozen=True)
class Rewritten:
    """Что ушло в поиск и чего это стоило.

    `query` — то, чем искали; `question` — то, что спросил человек. Они разные, и
    путать их нельзя: в контекст и в промпт ответа идёт вопрос, а в индекс запрос.
    """

    mode: str
    question: str
    query: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cached: bool = False
    note: str = ""

    @property
    def changed(self) -> bool:
        return self.query != self.question

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "title": MODE_TITLES[self.mode],
            "question": self.question,
            "query": self.query,
            "changed": self.changed,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "seconds": round(self.seconds, 3),
            "cached": self.cached,
            "note": self.note,
        }


def plain(question: str) -> Rewritten:
    """Точка отсчёта: вопрос уходит в поиск как есть, без модели и без цены."""
    return Rewritten(mode="none", question=question, query=question, cached=True)


# --- кэш ---------------------------------------------------------------------

_CACHE: dict[tuple[str, str], Rewritten] | None = None


def _load() -> dict[tuple[str, str], Rewritten]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    _CACHE = {}
    if CACHE_PATH.is_file():
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        for item in payload.get("rewrites", []):
            entry = Rewritten(**item, cached=True)
            _CACHE[(entry.mode, entry.question)] = entry

    return _CACHE


def _save() -> None:
    """Кэш на диск: порядок устойчивый, чтобы диff показывал только новое."""
    entries = sorted(_load().values(), key=lambda item: (item.mode, item.question))
    CACHE_PATH.write_text(
        json.dumps(
            {
                "model": llm.MODEL,
                "modes": [mode for mode in MODES if mode != "none"],
                "rewrites": [
                    {
                        "mode": item.mode,
                        "question": item.question,
                        "query": item.query,
                        "prompt_tokens": item.prompt_tokens,
                        "completion_tokens": item.completion_tokens,
                        "seconds": round(item.seconds, 3),
                        "note": item.note,
                    }
                    for item in entries
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def cached(mode: str, question: str) -> Rewritten | None:
    """Готовый запрос из кэша, если он там есть. Без сети и без ключа."""
    question = question.strip()
    if mode == "none":
        return plain(question)
    return _load().get((mode, question))


def require(mode: str, question: str) -> Rewritten:
    """То же, но с внятным отказом: замеры не должны молча считаться по исходному вопросу."""
    found = cached(mode, question)
    if found is None:
        raise RuntimeError(
            f"Запрос режима {mode!r} не переписан и в кэше его нет."
            " Прогрейте кэш: python day24/scenarios.py rewrite"
        )
    return found


# --- переписывание -----------------------------------------------------------


async def apply(mode: str, question: str) -> Rewritten:
    """Переписать вопрос, спросив модель, или взять готовое из кэша."""
    if mode not in MODES:
        raise ValueError(f"Режима {mode!r} нет. Есть: {', '.join(MODES)}.")

    question = question.strip()
    if not question:
        raise ValueError("Пустой вопрос переписывать нечем.")

    found = cached(mode, question)
    if found is not None:
        return found

    started = time.perf_counter()
    try:
        reply = await llm.complete(
            [
                {"role": "system", "content": PROMPTS[mode]},
                {"role": "user", "content": question},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    except Exception as exc:
        # Сорванный вызов не должен останавливать прогон и не должен тихо
        # притворяться удачным: в поиск уходит вопрос, а причина едет рядом.
        return Rewritten(
            mode=mode,
            question=question,
            query=question,
            seconds=time.perf_counter() - started,
            note=f"модель не ответила: {type(exc).__name__}",
        )

    query = " ".join(reply.text.split())
    entry = Rewritten(
        mode=mode,
        question=question,
        query=query if len(query) >= MIN_CHARS else question,
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
        seconds=reply.seconds,
        note="" if len(query) >= MIN_CHARS else f"ответ короче {MIN_CHARS} символов, искали вопросом",
    )

    _load()[(mode, question)] = entry
    _save()
    return entry


async def warm(modes: tuple[str, ...], questions: list[str]) -> dict[str, object]:
    """Прогреть кэш на весь набор сразу: то, что уже есть, модели не показывается."""
    wanted = [
        (mode, question.strip())
        for mode in modes
        if mode != "none"
        for question in questions
        if cached(mode, question.strip()) is None
    ]

    gate = asyncio.Semaphore(CONCURRENCY)

    async def one(mode: str, question: str) -> Rewritten:
        async with gate:
            return await apply(mode, question)

    done = await asyncio.gather(*(one(mode, question) for mode, question in wanted))

    return {
        "asked": len(done),
        "modes": [mode for mode in modes if mode != "none"],
        "questions": len(questions),
        "prompt_tokens": sum(item.prompt_tokens for item in done),
        "completion_tokens": sum(item.completion_tokens for item in done),
        "seconds": round(sum(item.seconds for item in done), 2),
        "failed": [item.note for item in done if item.note],
    }


def cost(modes: tuple[str, ...], questions: list[str]) -> dict[str, dict[str, object]]:
    """Цена этапа по кэшу: токены и время первого прогона, в среднем на вопрос."""
    report: dict[str, dict[str, object]] = {}

    for mode in modes:
        entries = [found for question in questions if (found := cached(mode, question))]
        if not entries:
            report[mode] = {"queries": 0, "prompt_tokens": 0, "completion_tokens": 0,
                            "seconds": 0.0, "chars": 0}
            continue

        # Токены считаются по тем запросам, за которые платили. Длина — по всем:
        # у режима `none` цена нулевая, а длину запроса сравнить с остальными надо.
        paid = [item for item in entries if item.completion_tokens]
        report[mode] = {
            "queries": len(entries),
            "prompt_tokens": round(sum(item.prompt_tokens for item in paid) / len(paid))
            if paid
            else 0,
            "completion_tokens": round(
                sum(item.completion_tokens for item in paid) / len(paid)
            )
            if paid
            else 0,
            "seconds": round(sum(item.seconds for item in paid) / len(paid), 2) if paid else 0.0,
            "chars": round(sum(len(item.query) for item in entries) / len(entries)),
        }

    return report
