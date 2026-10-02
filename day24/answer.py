"""Четыре режима одного агента: от ответа без базы до ответа, который нельзя дать.

[day23](../day23/answer.py) держал пять режимов и менял между ними то, что
происходит **до** промпта: переписывание запроса и фильтрацию кандидатов. Здесь
конвейер у всех трёх режимов с базой один и тот же — переписывание `keyword`,
реранкер `llm`, пул 20, в контекст пять выдержек, — и это сознательно. День
меряет не поиск, а форму ответа, и любое различие в контексте смазало бы замер.

| Режим | Контекст | Форма ответа | Отказ по порогу |
| --- | --- | --- | --- |
| `plain` | нет | свободный текст | — |
| `day23` | есть | свободный текст с `[n]` | — |
| `cited` | есть | json: утверждение, источник, цитата | — |
| `guarded` | есть | json | да |

`day23` — точка отсчёта, а не «плохой режим». Без него нельзя сказать, что
обязательная цитата что-то дала: у свободного текста со ссылкой `[2]` колонки
«цитаты есть» просто не существует, а колонка «ответ сослался» у него своя и
вполне приличная.

Два последних режима различаются одним: в `guarded` между контекстом и моделью
стоит порог из [gate.py](gate.py). Разводить их приходится потому, что отказ —
это размен, а не улучшение: он убирает выдумку на вопросах вне базы и
одновременно съедает верные ответы на вопросах из базы, где нужная выдержка
получила от реранкера оценку 2. Измерить обе стороны можно, только если режимы
стоят рядом.

Ответ, не прошедший порог, к модели не уходит вовсе. Это не оптимизация:
показать модели контекст и попросить ею же признать его негодным — значит
вернуть решение тому слою, который как раз и ошибается.
"""

import asyncio
import re
from dataclasses import dataclass, field

import cite
import gate
import llm
import pipeline
import verify
from cite import Cited
from gate import Verdict
from pipeline import Context, Plan, Source

# Конвейер взят у day23 целиком и в день не входит: `keyword` + `llm` — лучшая
# клетка его матрицы, recall@5 0.81 на 119 пробах. Менять эти две строки значит
# мерить другой поиск, а не другую форму ответа.
REWRITER = "keyword"
RERANKER = "llm"

MODES = ("plain", "day23", "cited", "guarded")
MODE_TITLES = {
    "plain": "без RAG",
    "day23": "как в day23",
    "cited": "+ цитаты",
    "guarded": "+ порог «не знаю»",
}

# Режимы, которые требуют от модели json по схеме [cite.py](cite.py).
STRUCTURED = ("cited", "guarded")

# Режимы, перед которыми стоит слой отказа в коде.
GATED = ("guarded",)

PLANS: dict[str, Plan | None] = {
    "plain": None,
    "day23": Plan.of(rewriter=REWRITER, reranker=RERANKER),
    "cited": Plan.of(rewriter=REWRITER, reranker=RERANKER),
    "guarded": Plan.of(rewriter=REWRITER, reranker=RERANKER),
}

TOP_K = pipeline.TOP_K

TEMPERATURE = 0.2
MAX_TOKENS = 600

# Структурированному ответу нужно заметно больше: к тексту утверждений
# добавляются цитаты, а они по условию длинные — до двух предложений каждая.
# Упереться в потолок здесь дороже, чем в day23: обрыв рвёт json, а не фразу.
STRUCTURED_MAX_TOKENS = 1000

REFUSAL = cite.REFUSAL

# Промпт свободного режима перенесён из day23 дословно. Точка отсчёта обязана
# быть той же самой, иначе сравнение мерило бы разницу промптов.
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


def plan_of(mode: str, **overrides: object) -> Plan | None:
    """План режима, с возможностью переопределить ручки из CLI или со страницы."""
    if mode not in MODES:
        raise ValueError(f"Режима {mode!r} нет. Есть: {', '.join(MODES)}.")

    base = PLANS[mode]
    if base is None:
        return None

    changed = {name: value for name, value in overrides.items() if value is not None}
    if not changed:
        return base

    reranker = changed.get("reranker", base.reranker)
    threshold = changed.get("threshold")

    # Смена реранкера сбрасывает порог фильтра к его собственному: шкалы у них
    # разные, и порог от прежнего отсекал бы наугад.
    if threshold is None and reranker == base.reranker:
        threshold = base.threshold

    return Plan.of(
        rewriter=changed.get("rewriter", base.rewriter),
        reranker=reranker,
        retriever=changed.get("retriever", base.retriever),
        pool=changed.get("pool", base.pool),
        top_k=changed.get("top_k", base.top_k),
        threshold=threshold,
        per_path=changed.get("per_path"),
    )


@dataclass(frozen=True)
class Answer:
    """Ответ одного режима со всем, по чему его можно проверить и оплатить."""

    mode: str
    question: str
    text: str
    reply: llm.Reply
    context: Context | None = None
    cited: list[int] = field(default_factory=list)
    structured: Cited | None = None
    report: verify.Report | None = None
    verdict: Verdict | None = None
    refusal: str = ""

    @property
    def cited_paths(self) -> list[str]:
        """Файлы, на которые ответ сослался. У структурированного — по утверждениям."""
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
        if self.refusal:
            return True
        if self.structured is not None:
            return self.structured.refused
        return self.text.lower().startswith(REFUSAL.lower())

    @property
    def claims(self) -> list[cite.Claim]:
        return self.structured.claims if self.structured else []

    @property
    def clarify(self) -> str:
        """Уточняющий вопрос: от слоя в коде либо от самой модели."""
        if self.verdict is not None and not self.verdict.passed:
            return self.verdict.clarify
        return self.structured.clarify if self.structured else ""

    @property
    def prompt_tokens(self) -> int:
        """Токены запроса вместе с этапами: переписывание и оценка тоже оплачены."""
        return self.reply.prompt_tokens + (self.context.stage_tokens if self.context else 0)

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "title": MODE_TITLES[self.mode],
            "question": self.question,
            "text": self.text,
            "cited": self.cited,
            "cited_paths": self.cited_paths,
            "refused": self.refused,
            "refusal": self.refusal,
            "clarify": self.clarify,
            "structured": self.structured.as_dict() if self.structured else None,
            "verify": self.report.as_dict() if self.report else None,
            "gate": self.verdict.as_dict() if self.verdict else None,
            "total_prompt_tokens": self.prompt_tokens,
            "context": self.context.as_dict() if self.context else None,
            **self.reply.as_dict(),
        }


# --- запрос ------------------------------------------------------------------


def structured(mode: str) -> bool:
    return mode in STRUCTURED


def messages(question: str, context: Context | None, mode: str = "day23") -> list[dict[str, str]]:
    """Запрос к модели. Форму задаёт режим, содержание — контекст."""
    system = cite.SYSTEM if structured(mode) else SYSTEM
    rules = cite.RULES if structured(mode) else RULES

    if context is None or not context.sources:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

    return [
        {"role": "system", "content": f"{system}\n\n{rules}"},
        {"role": "user", "content": f"{context.block}\n\nВопрос: {question}"},
    ]


def passages(context: Context) -> dict[int, verify.Passage]:
    """Выдержки глазами сверки: номер, адрес и текст, который видела модель."""
    return {
        source.number: verify.Passage(
            number=source.number,
            chunk_id=source.chunk_id,
            path=source.path,
            section=source.section,
            text=context.texts.get(source.number, ""),
        )
        for source in context.sources
    }


def citations(text: str, sources: list[Source]) -> list[int]:
    """Номера выдержек, на которые сослался свободный ответ.

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


def refusal_of(verdict: Verdict | None, parsed: Cited | None) -> str:
    """Какой слой сказал «не знаю». Пустая строка — значит ответ дан."""
    if verdict is not None and not verdict.passed:
        return verdict.reason
    if parsed is not None and parsed.refused:
        return gate.MODEL
    return ""


def finish(
    question: str,
    mode: str,
    context: Context | None,
    reply: llm.Reply,
    verdict: Verdict | None = None,
) -> Answer:
    """Собрать ответ из текста модели: разбор, сверка и вердикт — здесь."""
    if not structured(mode):
        return Answer(
            mode=mode,
            question=question,
            text=reply.text,
            reply=reply,
            context=context,
            cited=citations(reply.text, context.sources) if context else [],
            verdict=verdict,
        )

    parsed = cite.parse(reply.text)
    known = {source.number for source in context.sources} if context else set()
    report = verify.review(parsed.claims, passages(context)) if context else None

    return Answer(
        mode=mode,
        question=question,
        text=parsed.text,
        reply=reply,
        context=context,
        cited=sorted({claim.source for claim in parsed.claims if claim.source in known}),
        structured=parsed,
        report=report,
        verdict=verdict,
        refusal=refusal_of(verdict, parsed),
    )


def stopped(question: str, mode: str, context: Context | None, verdict: Verdict) -> Answer:
    """Отказ слоя в коде. Модель не вызывалась, поэтому и цена нулевая."""
    return Answer(
        mode=mode,
        question=question,
        text=verdict.text,
        reply=llm.Reply(text=verdict.text, prompt_tokens=0, completion_tokens=0, seconds=0.0),
        context=context,
        structured=Cited(unknown=True, clarify=verdict.clarify) if structured(mode) else None,
        report=verify.Report() if structured(mode) else None,
        verdict=verdict,
        refusal=verdict.reason,
    )


def checkpoint(mode: str, context: Context | None, threshold: float | None = None) -> Verdict | None:
    """Вердикт слоя в коде для режимов, у которых он есть."""
    if mode not in GATED:
        return None
    return gate.inspect(context, gate.THRESHOLD if threshold is None else threshold)


# --- спросить ----------------------------------------------------------------


async def ask(
    question: str,
    mode: str,
    *,
    context: Context | None = None,
    plan: Plan | None = None,
    threshold: float | None = None,
) -> Answer:
    """Один вопрос в одном режиме. Готовый контекст можно передать снаружи."""
    if mode not in MODES:
        raise ValueError(f"Режима {mode!r} нет. Есть: {', '.join(MODES)}.")

    question = question.strip()
    if not question:
        raise ValueError("Пустой вопрос задавать нечего.")

    if mode == "plain":
        context = None
    elif context is None:
        context = await pipeline.prepare(question, plan or PLANS[mode])

    verdict = checkpoint(mode, context, threshold)
    if verdict is not None and not verdict.passed:
        return stopped(question, mode, context, verdict)

    reply = await llm.complete(
        messages(question, context, mode),
        temperature=TEMPERATURE,
        max_tokens=STRUCTURED_MAX_TOKENS if structured(mode) else MAX_TOKENS,
        json=structured(mode),
    )
    return finish(question, mode, context, reply, verdict)


async def contexts(
    question: str, *, plans: dict[str, Plan | None] | None = None
) -> dict[str, Context | None]:
    """Контексты всех режимов сразу. Одинаковые планы считаются один раз.

    Со значениями по умолчанию план у трёх режимов с базой общий, так что
    конвейер отрабатывает ровно однажды: переписать запрос трижды и трижды
    спросить реранкер значило бы заплатить втрое за один и тот же контекст.
    """
    plans = plans or PLANS
    unique = {plan: None for plan in plans.values() if plan is not None}

    ready = await asyncio.gather(
        *(pipeline.prepare(question, plan) for plan in unique)
    )
    by_plan = dict(zip(unique, ready, strict=True))

    return {mode: by_plan.get(plan) if plan else None for mode, plan in plans.items()}


async def every(
    question: str,
    *,
    plans: dict[str, Plan | None] | None = None,
    threshold: float | None = None,
) -> dict[str, Answer]:
    """Все четыре режима на один вопрос, одновременно."""
    question = question.strip()
    plans = plans or PLANS
    prepared = await contexts(question, plans=plans)

    answers = await asyncio.gather(
        *(
            ask(question, mode, context=prepared[mode], threshold=threshold)
            for mode in plans
        )
    )
    return dict(zip(plans, answers, strict=True))
