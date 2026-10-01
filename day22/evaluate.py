"""Контрольный набор: десять вопросов, два режима, механика и судья.

«Стало лучше» — не результат. Результатом его делает то, что можно пересчитать,
поэтому у каждого вопроса в [questions.json](questions.json) заранее записано
ожидание, список обязательных фактов и список источников, которыми ответ должен
быть подкреплён. Набор маленький и написан руками намеренно: ожидание должен
задавать тот, кто знает ответ. Модель, придумывающая вопросы по отрывку, как в
[day21](../day21/evaluate.py), дала бы сто двадцать штук за пять минут — и вместе
с ними вопросы, подогнанные под текст отрывка, то есть под лексический поиск.

Оценок две, и они намеренно разной природы.

Механическая считается без модели и не плавает между прогонами: доля
обязательных фактов, найденных в ответе; дошёл ли поиск хоть до одного файла, по
которому на вопрос можно ответить; и сослался ли ответ на такой файл номером.
Последняя колонка отвечает на вопрос, который легко пропустить: ответ бывает
верным и при этом не опирающимся на контекст — модель написала его по памяти,
имея выдержки перед глазами.

Судья нужен там, где механика слепа: факт можно перечислить и не ответить, а
можно ответить верно другими словами. Судья видит вопрос, ожидание и ответ — но
не знает, какой режим его писал, и не видит контекста. Иначе он оценивал бы
полноту контекста, а не ответа.

Отдельно считаются вопросы «вне базы». Там верный ответ — отказ, и интересна не
доля фактов, а то, удержится ли режим от выдумки. Детектор отказа здесь грубый,
по оборотам речи: режим `rag` отказывается заданной формулировкой, а режим без
RAG — как придётся, и свести их к одному шаблону нельзя. Поэтому решающее слово
и тут за судьёй, а детектор стоит рядом как быстрая проверка.
"""

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import answer
import corpus
import index
import llm
import retrieve

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"

CONCURRENCY = 5
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 160

KINDS = ("в базе", "общий", "вне базы")

# Обороты, которыми ответ признаёт незнание. Список заведомо неполон — он и не
# может быть полным, — поэтому итоговую оценку «вне базы» ставит судья.
REFUSALS = (
    "в базе ответа нет",
    "нет данных",
    "нет информации",
    "не указан",
    "не указана",
    "не названа",
    "не названо",
    "не нашёл",
    "не нашел",
    "не знаю",
    "отсутству",
    "не содержится",
    "не приводится",
    "не могу",
)

JUDGE = (
    "Ты проверяешь ответ на вопрос о проекте AI Advent. Тебе даны вопрос, "
    "ожидаемое содержание ответа и сам ответ. Оцени, насколько ответ "
    "соответствует ожидаемому содержанию:\n"
    "2 — отвечает по существу, ничего важного не упущено и ничего не выдумано;\n"
    "1 — отвечает частично: верно, но неполно, либо верное смешано с лишним;\n"
    "0 — не отвечает, уходит от вопроса или противоречит ожидаемому.\n"
    "Если ожидаемое содержание — отказ, то отказ оценивай на 2, а уверенный "
    "выдуманный ответ на 0.\n"
    "Оценивай только соответствие ожиданию, а не стиль и не длину.\n"
    "Ответь ровно двумя строками:\n"
    "оценка: 0, 1 или 2\n"
    "почему: одно короткое предложение"
)

SCORE = re.compile(r"^\s*оценка\s*:\s*([012])", re.IGNORECASE | re.MULTILINE)
WHY = re.compile(r"^\s*почему\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)

DIGIT_GROUP = re.compile(r"(\d)[\s\u00a0\u202f](?=\d{3}\b)")


@dataclass(frozen=True)
class Question:
    """Контрольный вопрос вместе с тем, по чему его будут проверять."""

    id: int
    kind: str
    question: str
    expect: str
    facts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Graded:
    """Один ответ со всеми оценками: механические доли и балл судьи."""

    question_id: int
    kind: str
    question: str
    mode: str
    text: str
    facts_found: list[str]
    facts_missed: list[str]
    sources_found: list[str]
    sources_missed: list[str]
    cited_paths: list[str]
    refused: bool
    judge: int | None
    judge_why: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    context: dict[str, object] | None

    @property
    def facts_share(self) -> float | None:
        total = len(self.facts_found) + len(self.facts_missed)
        return len(self.facts_found) / total if total else None

    @property
    def expected_sources(self) -> set[str]:
        return set(self.sources_found) | set(self.sources_missed)

    @property
    def sources_hit(self) -> bool | None:
        """Попал ли в выдачу хоть один файл, по которому можно ответить.

        Главная колонка по источникам — именно эта, а не доля. Ответить хватает
        одного верного файла, и требовать все сразу значило бы штрафовать поиск
        за то, что он поставил README day20 выше, чем сам `vault.py`, хотя порт
        написан в обоих.
        """
        return bool(self.sources_found) if self.expected_sources else None

    @property
    def sources_share(self) -> float | None:
        total = len(self.expected_sources)
        return len(self.sources_found) / total if total else None

    @property
    def cited_hit(self) -> bool | None:
        """Сослался ли ответ хотя бы на один из ожидаемых источников."""
        if not self.expected_sources:
            return None
        return bool(self.expected_sources & set(self.cited_paths))

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["facts_share"] = self.facts_share
        data["sources_hit"] = self.sources_hit
        data["sources_share"] = self.sources_share
        data["cited_hit"] = self.cited_hit
        return data


# --- набор -------------------------------------------------------------------


def load() -> list[Question]:
    payload = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return [Question(**item) for item in payload["questions"]]


def normal(text: str) -> str:
    """К виду, в котором факт из набора сравним с тем, что написала модель.

    Разрядка чисел схлопывается: в документации стоит «15 493», а модель пишет
    «15493», и это один и тот же факт. Остальное — регистр и пробелы.
    """
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    while DIGIT_GROUP.search(text):
        text = DIGIT_GROUP.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def check() -> list[str]:
    """Проверить сам набор: факты обязаны находиться в названных источниках.

    Без этой проверки набор тихо протухает: документация правится, число в ней
    меняется, а в `questions.json` остаётся прежнее — и оба режима получают ноль
    за ответ, который на самом деле верен.
    """
    documents = {document.path: normal(document.text) for document in corpus.load()}
    notes: list[str] = []

    for item in load():
        if item.kind == "вне базы":
            if item.facts or item.sources:
                notes.append(f"#{item.id}: вопрос «вне базы» не может иметь фактов и источников")
            continue

        for path in item.sources:
            if path not in documents:
                notes.append(f"#{item.id}: источника {path} нет в корпусе")

        texts = [documents[path] for path in item.sources if path in documents]
        for fact in item.facts:
            if not any(normal(fact) in text for text in texts):
                notes.append(f"#{item.id}: факта «{fact}» нет ни в одном из названных источников")

    return notes


# --- оценка ------------------------------------------------------------------


def looks_refused(text: str) -> bool:
    body = normal(text)
    return any(marker in body for marker in REFUSALS)


async def judge(item: Question, text: str, gate: asyncio.Semaphore) -> tuple[int | None, str]:
    """Балл судьи. Режим и контекст ему не сообщаются — только вопрос и ожидание."""
    task = (
        f"Вопрос: {item.question}\n\n"
        f"Ожидаемое содержание: {item.expect}\n\n"
        f"Ответ: {text or '(пусто)'}"
    )

    async with gate:
        try:
            reply = await llm.complete(
                [{"role": "system", "content": JUDGE}, {"role": "user", "content": task}],
                temperature=JUDGE_TEMPERATURE,
                max_tokens=JUDGE_MAX_TOKENS,
            )
        except Exception as exc:
            return None, f"судья не ответил: {type(exc).__name__}"

    score = SCORE.search(reply.text)
    why = WHY.search(reply.text)
    return (int(score.group(1)) if score else None), (why.group(1).strip() if why else "")


def grade(item: Question, given: answer.Answer, score: int | None, why: str) -> Graded:
    """Механические доли по одному ответу. Без модели и без случайности."""
    body = normal(given.text)
    found = [fact for fact in item.facts if normal(fact) in body]

    retrieved = given.context.paths if given.context else []
    in_context = [path for path in item.sources if path in retrieved]

    return Graded(
        question_id=item.id,
        kind=item.kind,
        question=item.question,
        mode=given.mode,
        text=given.text,
        facts_found=found,
        facts_missed=[fact for fact in item.facts if fact not in found],
        sources_found=in_context,
        sources_missed=[path for path in item.sources if path not in in_context],
        cited_paths=given.cited_paths,
        refused=looks_refused(given.text),
        judge=score,
        judge_why=why,
        prompt_tokens=given.reply.prompt_tokens,
        completion_tokens=given.reply.completion_tokens,
        seconds=given.reply.seconds,
        context=given.context.as_dict() if given.context else None,
    )


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(graded: list[Graded]) -> dict[str, object]:
    """Сводка по одному режиму. Пустые колонки остаются пустыми, а не нулевыми."""
    judged = [item.judge for item in graded if item.judge is not None]
    outside = [item for item in graded if item.kind == "вне базы"]

    return {
        "questions": len(graded),
        "facts": _mean([share for item in graded if (share := item.facts_share) is not None]),
        "sources": _mean(
            [float(hit) for item in graded if (hit := item.sources_hit) is not None]
        ),
        "sources_share": _mean(
            [share for item in graded if (share := item.sources_share) is not None]
        ),
        "cited": _mean([float(hit) for item in graded if (hit := item.cited_hit) is not None]),
        "judge": _mean([float(score) for score in judged]),
        "judge_full": sum(1 for score in judged if score == 2),
        "refused_outside": sum(1 for item in outside if item.refused),
        "outside": len(outside),
        "prompt_tokens": round(sum(item.prompt_tokens for item in graded) / len(graded)),
        "completion_tokens": round(sum(item.completion_tokens for item in graded) / len(graded)),
        "seconds": round(sum(item.seconds for item in graded) / len(graded), 2),
    }


def by_kind(graded: list[Graded]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Graded]] = {}
    for item in graded:
        grouped.setdefault(item.kind, []).append(item)

    return {kind: summarize(grouped[kind]) for kind in KINDS if kind in grouped}


# --- прогон -------------------------------------------------------------------


async def _one(
    item: Question, retriever: str, gate: asyncio.Semaphore
) -> tuple[Graded, Graded]:
    """Один вопрос в обоих режимах: контекст считается один раз на оба."""
    context = await asyncio.to_thread(answer.prepare, item.question, retriever=retriever)

    async with gate:
        plain = await answer.ask(item.question, "plain")
    async with gate:
        rag = await answer.ask(item.question, "rag", context=context)

    scores = await asyncio.gather(
        judge(item, plain.text, gate), judge(item, rag.text, gate)
    )
    return (
        grade(item, plain, *scores[0]),
        grade(item, rag, *scores[1]),
    )


async def run(retriever: str = retrieve.DEFAULT) -> dict[str, object]:
    """Весь набор в обоих режимах со всеми оценками."""
    info = index.require()
    questions = load()
    notes = check()

    gate = asyncio.Semaphore(CONCURRENCY)
    pairs = await asyncio.gather(*(_one(item, retriever, gate) for item in questions))

    graded = {
        "plain": [pair[0] for pair in pairs],
        "rag": [pair[1] for pair in pairs],
    }

    return {
        "index": info,
        "retriever": retriever,
        "retriever_title": retrieve.RETRIEVER_TITLES[retriever],
        "model": llm.MODEL,
        "top_k": answer.TOP_K,
        "notes": notes,
        "questions": [asdict(item) for item in questions],
        "modes": list(answer.MODES),
        "mode_titles": answer.MODE_TITLES,
        "summary": {mode: summarize(items) for mode, items in graded.items()},
        "by_kind": {mode: by_kind(items) for mode, items in graded.items()},
        "graded": {
            mode: [item.as_dict() for item in items] for mode, items in graded.items()
        },
    }
