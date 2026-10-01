"""Пять режимов одного агента: от вопроса без контекста до полного конвейера.

[day22](../day22/answer.py) держался на том, что между двумя режимами ровно одно
различие — блок выдержек. Здесь режимов пять, и правило то же, только различий
теперь два, и оба лежат **до** промпта:

| Режим | Контекст | Переписывание | Фильтр |
| --- | --- | --- | --- |
| `plain` | нет | — | — |
| `rag` | есть | нет | нет |
| `rag_rewrite` | есть | да | нет |
| `rag_filter` | есть | нет | да |
| `rag_both` | есть | да | да |

Модель, температура, системный промпт и три правила у всех пяти одни и те же, и
вопрос всем пяти задаётся дословно одинаковый. Меняется только то, какие выдержки
попали в блок и попали ли вообще. Иначе сравнение мерило бы разницу промптов.

`rag` — это буквальный day22, точка отсчёта, а не «плохой режим»: без него нельзя
сказать, что фильтрация улучшила, и в задании это требование названо прямо —
качество без фильтра и rewriting против качества с ними.

Два средних режима нужны затем, что иначе улучшение нельзя разложить. Если
измерять только `rag` и `rag_both`, разница окажется суммой двух непохожих
вмешательств, и какое из них работает — а какое, может быть, мешает, — останется
неизвестным.

Пустой контекст здесь, в отличие от day22, стал обычным делом: фильтр вправе
отсечь всех кандидатов. Тогда режим уходит к модели без блока выдержек вовсе —
то есть как `plain`, — и это честнее, чем показывать ей то, что сам же признал
нерелевантным. Правило про отказ при этом никуда не девается: модель всё равно
обязана сказать, что не знает, вместо того чтобы придумать.
"""

import asyncio
import re
from dataclasses import dataclass, field

import llm
import pipeline
from pipeline import Context, Plan, Source

# Что ставится в режимы, где этап включён. Выбрано матрицей на 119 пробах — см.
# README. Менять эти две строки значит менять то, что день измерил.
#
# Реранкер — `llm`: на исходном вопросе recall@5 0.80 против 0.66 у day22, и в
# контексте 3.33 выдержки вместо пяти. `heuristic` сам по себе recall роняет
# (0.60), `vector` — тем более (0.43).
#
# Переписывание — `keyword`, и это не то, с чего день начинался. HyDE должен был
# закрыть разрыв «вопрос vs отрывок» из day21, но гипотетический ответ обрастает
# выдуманными именами, и лексика ищет уже их: пул падает с 0.85 до 0.81, а
# recall@5 без фильтра — с 0.66 до 0.55. Keyword чуть слабее исходного вопроса
# (0.63), зато вместе с `llm` даёт лучшую клетку матрицы: 0.81.
REWRITER = "keyword"
RERANKER = "llm"

MODES = ("plain", "rag", "rag_rewrite", "rag_filter", "rag_both")
MODE_TITLES = {
    "plain": "без RAG",
    "rag": "RAG как в day22",
    "rag_rewrite": "+ переписывание",
    "rag_filter": "+ фильтр",
    "rag_both": "+ оба",
}

# Режим — это значение `Plan`, а не ветка в коде. `rag` обязан остаться буквальным
# day22: без реранкера оценки нет, порог и потолок на файл не ставятся, и фильтр
# вырождается в обрезку по top-K.
PLANS: dict[str, Plan | None] = {
    "plain": None,
    "rag": Plan.of(),
    "rag_rewrite": Plan.of(rewriter=REWRITER),
    "rag_filter": Plan.of(reranker=RERANKER),
    "rag_both": Plan.of(rewriter=REWRITER, reranker=RERANKER),
}

TOP_K = pipeline.TOP_K

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

    rewriter = changed.get("rewriter", base.rewriter)
    reranker = changed.get("reranker", base.reranker)

    # Ручки со страницы и из CLI настраивают уже включённый этап, а не добавляют
    # новый. Иначе `rag` перестаёт быть day22: выбрали keyword в селекте — и
    # точка отсчёта тоже переписывается, сравнивать больше не с чем.
    if base.rewriter == "none":
        rewriter = "none"
    if base.reranker == "none":
        reranker = "none"

    threshold = changed.get("threshold")

    # Смена реранкера сбрасывает порог к его собственному: шкалы у них разные, и
    # порог от прежнего отсекал бы наугад.
    if threshold is None and reranker == base.reranker:
        threshold = base.threshold

    return Plan.of(
        rewriter=rewriter,
        reranker=reranker,
        retriever=changed.get("retriever", base.retriever),
        pool=changed.get("pool", base.pool),
        top_k=changed.get("top_k", base.top_k),
        threshold=threshold,
        per_path=changed.get("per_path"),
    )


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
            "total_prompt_tokens": self.prompt_tokens,
            "context": self.context.as_dict() if self.context else None,
            **self.reply.as_dict(),
        }


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
    context: Context | None = None,
    plan: Plan | None = None,
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

    reply = await llm.complete(
        messages(question, context), temperature=TEMPERATURE, max_tokens=MAX_TOKENS
    )
    return finish(question, mode, context, reply)


async def contexts(
    question: str, *, plans: dict[str, Plan | None] | None = None
) -> dict[str, Context | None]:
    """Контексты всех режимов сразу: конвейеры независимы, ждать их по очереди незачем.

    Одинаковые планы считаются один раз. Со значениями по умолчанию `rag` и
    `rag_filter` ищут одним и тем же запросом, и переписывать его дважды, а тем
    более дважды спрашивать реранкер — значит платить за то же самое.
    """
    plans = plans or PLANS
    unique = {plan: None for plan in plans.values() if plan is not None}

    ready = await asyncio.gather(
        *(pipeline.prepare(question, plan) for plan in unique)
    )
    by_plan = dict(zip(unique, ready, strict=True))

    return {mode: by_plan.get(plan) if plan else None for mode, plan in plans.items()}


async def every(
    question: str, *, plans: dict[str, Plan | None] | None = None
) -> dict[str, Answer]:
    """Все пять режимов на один вопрос, одновременно."""
    question = question.strip()
    plans = plans or PLANS
    prepared = await contexts(question, plans=plans)

    answers = await asyncio.gather(
        *(ask(question, mode, context=prepared[mode]) for mode in plans)
    )
    return dict(zip(plans, answers, strict=True))
