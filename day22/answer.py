"""Два режима одного агента: вопрос как есть и вопрос вместе с найденным.

Весь день держится на том, что между режимами ровно одно различие. Модель одна,
температура одна, системный промпт один и тот же, вопрос дословно тот же — а в
режиме `rag` перед вопросом стоит блок с найденными чанками и три правила о том,
как с ним обращаться. Если бы режимы отличались ещё и промптом, сравнение
показывало бы разницу промптов, и сказать, что дал поиск, было бы нельзя.

Контекст собирается пронумерованным, и номер — не украшение. По нему модель
ссылается на источник, по нему же потом механически проверяется, чем именно она
подкрепила ответ: процитированный номер разворачивается обратно в путь к файлу.
Ответ без ссылок в режиме `rag` — это ответ, который модель написала по памяти,
имея контекст перед глазами, и такой случай надо уметь отличать.

Правило про отказ («в базе ответа нет») добавлено не ради вежливости. Без него
модель, получив пять нерелевантных чанков, пересказывает ближайший из них — и
выходит ответ, который выглядит подкреплённым источниками и при этом неверен.
Это худший исход из возможных, хуже отказа и хуже честного незнания.

У правила есть цена, и её тоже надо показать: на вопросе, ответ на который модель
знает и так, но которого нет в базе проекта, режим `rag` обязан промолчать там,
где режим без RAG отвечает. В наборе контрольных вопросов такие есть нарочно.
"""

import asyncio
import re
from dataclasses import dataclass, field

import llm
import retrieve
from retrieve import Hit

MODES = ("plain", "rag")
MODE_TITLES = {"plain": "без RAG", "rag": "с RAG"}

TOP_K = 5

# Потолок контекста. Пять чанков `hybrid` — это максимум 2000 токенов, так что
# упираться в него нечем; он стоит против чанка-гиганта, если разбиение изменят.
CONTEXT_TOKENS = 2400

TEMPERATURE = 0.2
MAX_TOKENS = 600

REFUSAL = "В базе ответа нет"

SYSTEM = (
    "Ты отвечаешь на вопросы о проекте AI Advent — репозитории с дневником итераций, "
    "где есть документация, код на Python и PDF.\n"
    "Отвечай по-русски, коротко и по делу: два-четыре предложения, без вступлений "
    "и без пересказа вопроса.\n"
    "Если чего-то не знаешь — скажи об этом прямо, не придумывай."
)

RULES = (
    "Ниже выдержки из базы проекта, каждая под своим номером.\n"
    "Отвечай только по ним, не добавляя ничего от себя.\n"
    "Каждое утверждение помечай номером выдержки в квадратных скобках, например [2].\n"
    "Числа, имена файлов и названия приводи ровно так, как они стоят в выдержках.\n"
    f"Если ответа в выдержках нет — ответь «{REFUSAL}» и ничего не добавляй."
)

CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")


@dataclass(frozen=True)
class Source:
    """Выдержка под своим номером: по нему ответ на неё ссылается."""

    number: int
    chunk_id: int
    source: str
    path: str
    title: str
    section: str
    start: int
    end: int
    chars: int
    tokens: int
    score: float
    snippet: str

    @classmethod
    def of(cls, number: int, hit: Hit) -> "Source":
        return cls(
            number=number,
            chunk_id=hit.chunk_id,
            source=hit.source,
            path=hit.path,
            title=hit.title,
            section=hit.section,
            start=hit.start,
            end=hit.end,
            chars=hit.chars,
            tokens=hit.tokens,
            score=round(hit.score, 4),
            snippet=hit.snippet,
        )

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class Context:
    """Блок, который пристёгивается к вопросу, и всё, что о нём надо знать."""

    block: str
    sources: list[Source]
    retriever: str
    seconds: float

    @property
    def tokens(self) -> int:
        return sum(source.tokens for source in self.sources)

    @property
    def paths(self) -> list[str]:
        """Файлы в выдаче, без повторов и в порядке мест."""
        seen: list[str] = []
        for source in self.sources:
            if source.path not in seen:
                seen.append(source.path)
        return seen

    def as_dict(self) -> dict[str, object]:
        return {
            "retriever": self.retriever,
            "title": retrieve.RETRIEVER_TITLES[self.retriever],
            "seconds": round(self.seconds, 4),
            "tokens": self.tokens,
            "chars": sum(source.chars for source in self.sources),
            "paths": self.paths,
            "sources": [source.as_dict() for source in self.sources],
        }


@dataclass(frozen=True)
class Answer:
    """Ответ одного режима со всем, что нужно, чтобы его оценить и оплатить."""

    mode: str
    question: str
    text: str
    reply: llm.Reply
    context: Context | None = None
    cited: list[int] = field(default_factory=list)

    @property
    def cited_paths(self) -> list[str]:
        """Файлы, на которые ответ сослался номером."""
        if self.context is None:
            return []

        by_number = {source.number: source.path for source in self.context.sources}
        seen: list[str] = []
        for number in self.cited:
            path = by_number.get(number)
            if path and path not in seen:
                seen.append(path)
        return seen

    @property
    def refused(self) -> bool:
        return self.text.lower().startswith(REFUSAL.lower())

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "title": MODE_TITLES[self.mode],
            "question": self.question,
            "text": self.text,
            "cited": self.cited,
            "cited_paths": self.cited_paths,
            "refused": self.refused,
            "context": self.context.as_dict() if self.context else None,
            **self.reply.as_dict(),
        }


# --- контекст ----------------------------------------------------------------


def _block(sources: list[Source], hits: list[Hit]) -> str:
    """Выдержки под номерами, с путём и разделом в шапке каждой.

    Путь и раздел нужны не только человеку: по ним модель отвечает «в day8», а не
    «в одном из файлов», и ответ становится проверяемым.
    """
    return "\n\n".join(
        f"[{source.number}] {source.path} · {source.section}\n{hit.text.strip()}"
        for source, hit in zip(sources, hits, strict=True)
    )


def prepare(question: str, *, retriever: str = retrieve.DEFAULT, limit: int = TOP_K) -> Context:
    """Найти чанки и сложить из них блок контекста. Без модели и без ключа."""
    found = retrieve.search(question, limit, retriever)

    sources: list[Source] = []
    kept: list[Hit] = []
    budget = CONTEXT_TOKENS

    for hit in found.hits:
        if hit.tokens > budget:
            continue
        budget -= hit.tokens
        sources.append(Source.of(len(sources) + 1, hit))
        kept.append(hit)

    return Context(
        block=_block(sources, kept),
        sources=sources,
        retriever=found.retriever,
        seconds=found.seconds,
    )


def messages(question: str, context: Context | None) -> list[dict[str, str]]:
    """Запрос к модели. Разница между режимами — ровно в блоке контекста."""
    if context is None or not context.sources:
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ]

    return [
        {"role": "system", "content": f"{SYSTEM}\n\n{RULES}"},
        {"role": "user", "content": f"{context.block}\n\nВопрос: {question}"},
    ]


def citations(text: str, sources: list[Source]) -> list[int]:
    """Номера выдержек, на которые сослался ответ.

    Выдуманные номера отбрасываются: модель иногда ставит [7] там, где выдержек
    пять, и считать такую ссылку подкреплением было бы неправильно.
    """
    known = {source.number for source in sources}
    found: list[int] = []

    for group in CITATION.findall(text):
        for part in re.split(r"[,;]", group):
            number = int(part.strip())
            if number in known and number not in found:
                found.append(number)

    return sorted(found)


def finish(question: str, mode: str, context: Context | None, reply: llm.Reply) -> Answer:
    """Собрать ответ из текста модели: ссылки разбираются здесь, а не у вызывающего."""
    return Answer(
        mode=mode,
        question=question,
        text=reply.text,
        reply=reply,
        context=context,
        cited=citations(reply.text, context.sources) if context else [],
    )


# --- спросить ----------------------------------------------------------------


async def ask(
    question: str,
    mode: str,
    *,
    retriever: str = retrieve.DEFAULT,
    limit: int = TOP_K,
    context: Context | None = None,
) -> Answer:
    """Один вопрос в одном режиме. Готовый контекст можно передать снаружи."""
    if mode not in MODES:
        raise ValueError(f"Режима {mode!r} нет. Есть: {', '.join(MODES)}.")

    question = question.strip()
    if not question:
        raise ValueError("Пустой вопрос задавать нечего.")

    if mode == "rag" and context is None:
        context = await asyncio.to_thread(prepare, question, retriever=retriever, limit=limit)
    if mode == "plain":
        context = None

    reply = await llm.complete(
        messages(question, context), temperature=TEMPERATURE, max_tokens=MAX_TOKENS
    )
    return finish(question, mode, context, reply)


async def both(
    question: str, *, retriever: str = retrieve.DEFAULT, limit: int = TOP_K
) -> dict[str, Answer]:
    """Оба режима на один вопрос, одновременно: ждать их по очереди незачем."""
    question = question.strip()
    context = await asyncio.to_thread(prepare, question, retriever=retriever, limit=limit)

    plain, rag = await asyncio.gather(
        ask(question, "plain"),
        ask(question, "rag", context=context),
    )
    return {"plain": plain, "rag": rag}
