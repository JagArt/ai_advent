"""Путь от вопроса до контекста: переписать, найти, переставить, отсечь, собрать.

В [day22](../day22/answer.py) этого модуля не было: между вопросом и промптом
стоял один вызов поиска, и собирать было нечего. Здесь этапов четыре, у каждого
свои ручки, и держать их россыпью по вызывающим — значит получить пять разных
конвейеров вместо одного.

Поэтому конфигурация собрана в один `Plan`, и «режим» дня — это не ветка в коде, а
значение этого объекта:

| Режим | `rewriter` | `reranker` |
| --- | --- | --- |
| как в day22 | `none` | `none` |
| + переписывание | `keyword` / `hyde` | `none` |
| + фильтр | `none` | `heuristic` / `vector` / `llm` |
| + оба | `keyword` / `hyde` | `heuristic` / `vector` / `llm` |

Два top-K, и путать их нельзя. `pool` — сколько кандидатов достаёт поиск **до**
фильтрации; `top_k` — сколько выдержек доезжает до промпта **после**. В day22 они
были одним числом: искали пять, брали пять. Разводить их приходится потому, что
фильтру нужен запас — отсекать из пяти значило бы приносить в контекст три
выдержки вместо пяти и выдавать обеднение за улучшение.

Входов два, и это не дублирование. `prepare` умеет спросить модель и доплатить за
переписывание и оценку; `prepare_cached` не ходит в сеть вовсе и падает, если
нужного в кэше нет. Второй нужен замерам: матрица режимов на 119 пробах и
развертка по порогу — это десятки тысяч расстановок, и каждая из них обязана быть
локальным счётом, иначе их нельзя ни прогнать, ни воспроизвести.
"""

import asyncio
from dataclasses import dataclass, field

import rerank
import retrieve
import rewrite
from rerank import Ranked, Scored
from retrieve import Found, Hit
from rewrite import Rewritten

# Сколько кандидатов достаёт поиск до фильтрации. Двадцать — потому что пул должен
# быть заметно шире выдачи, но оставаться посильным для единственного вызова
# LLM-реранкера: двадцать чанков по медиане разбиения — это около шести тысяч
# токенов на запрос, сорок было бы уже двенадцать.
POOL = 20

# Сколько выдержек доезжает до промпта. Пять — как в day22, иначе сравнивать не с чем.
TOP_K = 5

# Потолок контекста. Пять чанков — максимум 2000 токенов, так что упираться в него
# нечем; он стоит против чанка-гиганта, если разбиение изменят.
CONTEXT_TOKENS = 2400


@dataclass(frozen=True)
class Plan:
    """Что именно делает конвейер. Один объект на режим, ветвлений в коде нет."""

    retriever: str = retrieve.DEFAULT
    rewriter: str = "none"
    reranker: str = "none"
    pool: int = POOL
    top_k: int = TOP_K
    threshold: float | None = None
    per_path: int | None = None

    @classmethod
    def of(
        cls,
        *,
        rewriter: str = "none",
        reranker: str = "none",
        threshold: float | None = None,
        per_path: int | None = None,
        **rest: object,
    ) -> "Plan":
        """План с порогом и потолком по умолчанию.

        Порог берётся у реранкера, а не из общей константы: у `heuristic` оценка
        относительная и жмётся к единице, у `vector` это косинус, у `llm` — четыре
        деления. Одно число значило бы у каждого своё.

        Потолок на файл ставится только там, где есть что фильтровать. Режим без
        реранкера обязан остаться буквальным day22, иначе точка отсчёта поехала.
        """
        return cls(
            rewriter=rewriter,
            reranker=reranker,
            threshold=rerank.THRESHOLDS[reranker] if threshold is None else threshold,
            per_path=(None if reranker == "none" else rerank.PER_PATH)
            if per_path is None
            else per_path,
            **rest,  # type: ignore[arg-type]
        )

    @property
    def filters(self) -> bool:
        return self.reranker != "none"

    def as_dict(self) -> dict[str, object]:
        return {
            "retriever": self.retriever,
            "retriever_title": retrieve.RETRIEVER_TITLES[self.retriever],
            "rewriter": self.rewriter,
            "rewriter_title": rewrite.MODE_TITLES[self.rewriter],
            "reranker": self.reranker,
            "reranker_title": rerank.RERANKER_TITLES[self.reranker],
            "pool": self.pool,
            "top_k": self.top_k,
            "threshold": self.threshold,
            "per_path": self.per_path,
        }


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
    relevance: float | None
    place: int
    snippet: str

    @classmethod
    def of(cls, number: int, item: Scored) -> "Source":
        hit = item.hit
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
            relevance=None if item.relevance is None else round(item.relevance, 3),
            place=item.place,
            snippet=hit.snippet,
        )

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class Context:
    """Блок, который пристёгивается к вопросу, и вся история его появления."""

    question: str
    block: str
    sources: list[Source]
    plan: Plan
    rewritten: Rewritten
    ranked: Ranked
    search_seconds: float
    dropped: list[Scored] = field(default_factory=list)

    # Тексты дошедших чанков целиком, по номеру выдержки. В day23 их можно было
    # не держать: ответ ссылался номером, и сверять было нечего. Здесь ответ
    # приносит цитату, и её ищут в том самом тексте, который видела модель, —
    # не в сниппете и не в перечитанном из индекса чанке.
    texts: dict[int, str] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return sum(source.tokens for source in self.sources)

    @property
    def best_relevance(self) -> float | None:
        """Оценка лучшей дошедшей выдержки. По ней решается, отвечать ли вообще."""
        scores = [source.relevance for source in self.sources if source.relevance is not None]
        return max(scores) if scores else None

    @property
    def paths(self) -> list[str]:
        """Файлы в выдаче, без повторов и в порядке мест."""
        seen: list[str] = []
        for source in self.sources:
            if source.path not in seen:
                seen.append(source.path)
        return seen

    @property
    def stage_tokens(self) -> int:
        """Токены, уплаченные до того, как начался сам ответ: цена второго этапа."""
        return (
            self.rewritten.prompt_tokens
            + self.rewritten.completion_tokens
            + self.ranked.prompt_tokens
            + self.ranked.completion_tokens
        )

    @property
    def stage_seconds(self) -> float:
        return self.rewritten.seconds + self.search_seconds + self.ranked.seconds

    def as_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "plan": self.plan.as_dict(),
            "best_relevance": self.best_relevance,
            "rewrite": self.rewritten.as_dict(),
            "rerank": self.ranked.as_dict(),
            "seconds": round(self.search_seconds, 4),
            "stage_seconds": round(self.stage_seconds, 4),
            "stage_tokens": self.stage_tokens,
            "tokens": self.tokens,
            "chars": sum(source.chars for source in self.sources),
            "pool": self.ranked.pool,
            "kept": len(self.sources),
            "dropped": len(self.dropped),
            "paths": self.paths,
            "sources": [source.as_dict() for source in self.sources],
            "dropped_sources": [item.as_dict() for item in self.dropped],
        }


# --- сборка ------------------------------------------------------------------


def _block(sources: list[Source], hits: list[Hit]) -> str:
    """Выдержки под номерами, с путём и разделом в шапке каждой.

    Путь и раздел нужны не только человеку: по ним модель отвечает «в day8», а не
    «в одном из файлов», и ответ становится проверяемым.
    """
    return "\n\n".join(
        f"[{source.number}] {source.path} · {source.section}\n{hit.text.strip()}"
        for source, hit in zip(sources, hits, strict=True)
    )


def _assemble(
    question: str, plan: Plan, rewritten: Rewritten, found: Found, ranked: Ranked
) -> Context:
    """Выдержки в блок, с нумерацией и потолком контекста."""
    sources: list[Source] = []
    kept: list[Hit] = []
    budget = CONTEXT_TOKENS

    for item in ranked.kept:
        if item.hit.tokens > budget:
            continue
        budget -= item.hit.tokens
        sources.append(Source.of(len(sources) + 1, item))
        kept.append(item.hit)

    return Context(
        question=question,
        block=_block(sources, kept),
        sources=sources,
        plan=plan,
        rewritten=rewritten,
        ranked=ranked,
        search_seconds=found.seconds,
        dropped=ranked.dropped,
        texts={source.number: hit.text for source, hit in zip(sources, kept, strict=True)},
    )


async def prepare(question: str, plan: Plan | None = None) -> Context:
    """Весь конвейер на один вопрос. Может спросить модель — за переписывание и оценку."""
    plan = plan or Plan.of()
    question = question.strip()
    if not question:
        raise ValueError("Пустой вопрос искать нечем.")

    rewritten = await rewrite.apply(plan.rewriter, question)
    found = await asyncio.to_thread(
        retrieve.search, rewritten.query, plan.pool, plan.retriever
    )
    ranked = await rerank.apply(
        plan.reranker,
        found.hits,
        query=rewritten.query,
        question=question,
        top_k=plan.top_k,
        threshold=plan.threshold,
        per_path=plan.per_path,
    )

    return _assemble(question, plan, rewritten, found, ranked)


def prepare_cached(question: str, plan: Plan | None = None) -> Context:
    """Тот же конвейер без сети: всё нужное обязано лежать в кэшах."""
    plan = plan or Plan.of()
    question = question.strip()

    rewritten = rewrite.require(plan.rewriter, question)
    found = retrieve.search(rewritten.query, plan.pool, plan.retriever)
    ranked = rerank.apply_cached(
        plan.reranker,
        found.hits,
        query=rewritten.query,
        question=question,
        top_k=plan.top_k,
        threshold=plan.threshold,
        per_path=plan.per_path,
    )

    return _assemble(question, plan, rewritten, found, ranked)


def pool_of(question: str, plan: Plan) -> tuple[Rewritten, Found]:
    """Кандидаты до второго этапа. Нужны прогреву кэша: оценки ставятся по пулу."""
    rewritten = rewrite.require(plan.rewriter, question.strip())
    return rewritten, retrieve.search(rewritten.query, plan.pool, plan.retriever)


async def heat(questions: list[str], plans: list[Plan]) -> dict[str, object]:
    """Прогреть кэши под набор планов — всё, за что конвейер платит сетью.

    Порядок обязателен и не переставляется: пул зависит от того, чем искали, а
    подпись оценки — от пула. Прогревать оценки до переписывания нечем.

    После этого шага `prepare_cached` отвечает на всё, что встретится в замерах, и
    матрица с разверткой становятся локальным счётом. Это не оптимизация, а
    условие воспроизводимости: одни и те же числа между прогонами получаются
    только тогда, когда модель не спрашивают заново.
    """
    rewriters = tuple(sorted({plan.rewriter for plan in plans} - {"none"}))
    rewrites = await rewrite.warm(rewriters, questions)

    # Пул определяется тем, чем искали и сколько брали, а не реранкером: планы с
    # одинаковой тройкой дают один пул, и спрашивать по нему дважды незачем.
    wanted = {
        (plan.rewriter, plan.retriever, plan.pool)
        for plan in plans
        if plan.reranker == "llm"
    }

    pools: list[tuple[str, list[Hit]]] = []
    for rewriter, retriever, pool in sorted(wanted):
        plan = Plan.of(rewriter=rewriter, reranker="llm", retriever=retriever, pool=pool)
        pools.extend(
            await asyncio.to_thread(
                lambda inner=plan: [
                    (question, pool_of(question, inner)[1].hits) for question in questions
                ]
            )
        )

    return {"rewrites": rewrites, "relevance": await rerank.warm(pools)}
