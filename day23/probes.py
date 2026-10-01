"""Матрица режимов на эталонном наборе day21: что даёт переписывание и что — фильтр.

Набор тот же и переносится как есть: [day22/probes.json](../day22/probes.json),
120 проб, из которых живыми доезжают 119. Пересобирать его заново значило бы
сравнивать день с днём по разным линейкам. Устройство стоит повторить, потому что
от него зависит смысл чисел: эталон — не «правильный чанк», а диапазон
`[start, end)` в каноническом тексте документа, и чанк засчитан, если он из того
же файла и накрыл не меньше половины отрывка.

Спрашиваем здесь только вопросом. В day21 и day22 рядом стояли ещё короткий
запрос и сам отрывок, и третья проба была потолком — она мерила индекс в отрыве
от того, умеет ли поиск связать формулировку вопроса с текстом ответа. Этот день
занят ровно тем разрывом, который потолок и показал, так что лишние способы
спросить только мешали бы: в RAG приходит вопрос человека.

## Чего не покажет recall

Фильтр умеет только убирать, и по одному recall он всегда выглядит как ухудшение
или как ничья. Это не значит, что он не работает: он торгует присутствием нужного
чанка на чистоту и цену контекста. Поэтому рядом с recall считаются три колонки,
без которых фильтр читается неверно:

* `выдержек` — сколько их доехало в среднем. Пять было всегда; стало меньше, и
  разница — это ровно то, за что мы не платим токенами.
* `символов` — та же экономия в том, чем за неё платят.
* `пусто` — доля проб, где фильтр не оставил ничего. Величина двуликая: когда
  нужного чанка в пуле и не было, пустой контекст лучше пяти нерелевантных, а
  когда был — это потеря, и смотреть на неё надо вместе со следующей колонкой.
* `потеряно` — доля проб, где нужный чанк в пуле был, фильтр его выбросил. Это
  единственная честная цена фильтрации, и её нельзя списать на шум.

## Почему всё считается локально

Матрица — три режима переписывания на четыре реранкера, 119 проб, то есть 1 428
прогонов конвейера. Развертка по порогу — ещё семь значений на каждом из них.
Каждый такой прогон обязан быть локальным счётом, иначе набор не прогнать и, что
важнее, не воспроизвести. Поэтому сеть вынесена в два шага прогрева: сначала
переписываются запросы, потом по готовым пулам ставятся оценки реранкера, и
только после этого считается матрица — целиком из кэшей.
"""

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import corpus
import index
import pipeline
import rerank
import retrieve
import rewrite
from corpus import Document
from pipeline import Plan
from retrieve import Hit

PROBES_PATH = Path(__file__).resolve().parent / "probes.json"

# Половина эталонного отрывка: чанк, накрывший меньше, ответа целиком не содержит.
OVERLAP_SHARE = 0.5
RANKS = (1, 3, 5)

# Значения порога для развертки. Шкалы у реранкеров разные, поэтому развертка
# идёт по доле от 0 до 1, а читать её надо отдельно для каждого. Сетка сгущена
# там, где у них изломы: у `vector` между 0.3 и 0.5, у `heuristic` около 2/3.
SWEEP = (0.0, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.65, 2 / 3, 0.7, 0.75, 0.8, 0.85, 0.9)


@dataclass(frozen=True)
class Probe:
    """Эталон: файл, диапазон, сам отрывок и вопрос про него."""

    query: str
    question: str
    path: str
    section: str
    start: int
    end: int
    passage: str
    doc_sha256: str


@dataclass(frozen=True)
class Scored:
    """Что конвейер принёс на одну пробу — до фильтра и после."""

    question: str
    path: str
    pool_rank: int | None
    hit_rank: int | None
    kept: int
    context_chars: int
    chars_to_hit: int
    top: list[dict[str, object]]

    @property
    def in_pool(self) -> bool:
        return self.pool_rank is not None

    @property
    def lost(self) -> bool:
        """Нужный чанк был в пуле, но до промпта не доехал: цена фильтрации."""
        return self.in_pool and self.hit_rank is None

    @property
    def empty(self) -> bool:
        return self.kept == 0


def load() -> tuple[list[Probe], list[str]]:
    """Набор из файла с привязкой к текущему тексту документов.

    Файлы репозитория живут и меняются, а смещения эталона привязаны к тексту.
    Поэтому у пробы хранится и sha документа, и сам отрывок: если файл изменился,
    отрывок ищется в новом тексте заново, а если не нашёлся — проба выбывает.
    Молча считать метрики по съехавшим эталонам хуже, чем потерять часть набора.
    """
    if not PROBES_PATH.is_file():
        return [], ["Набора проб нет: probes.json должен лежать рядом с модулем."]

    payload = json.loads(PROBES_PATH.read_text(encoding="utf-8"))
    documents: dict[str, Document] = {item.path: item for item in corpus.load()}
    probes: list[Probe] = []
    notes: list[str] = []

    for item in payload["probes"]:
        probe = Probe(**item)
        document = documents.get(probe.path)

        if document is None:
            notes.append(f"{probe.path} больше нет в корпусе — проба выбыла")
            continue

        if document.sha256 == probe.doc_sha256:
            probes.append(probe)
            continue

        position = document.text.find(probe.passage)
        if position < 0:
            notes.append(f"{probe.path} изменился, отрывок не нашёлся — проба выбыла")
            continue

        notes.append(f"{probe.path} изменился, отрывок переехал на {position}")
        probes.append(
            Probe(
                **{
                    **item,
                    "start": position,
                    "end": position + len(probe.passage),
                    "doc_sha256": document.sha256,
                }
            )
        )

    return probes, notes


# --- метрики -----------------------------------------------------------------


def _flags(probe: Probe, hits: list[Hit]) -> list[bool]:
    """Накрыл ли чанк эталонный отрывок хотя бы наполовину."""
    needed = (probe.end - probe.start) * OVERLAP_SHARE
    return [
        hit.path == probe.path
        and max(0, min(hit.end, probe.end) - max(hit.start, probe.start)) >= needed
        for hit in hits
    ]


def _rank(flags: list[bool]) -> int | None:
    return flags.index(True) + 1 if any(flags) else None


def _score(probe: Probe, context: pipeline.Context) -> Scored:
    """Одна проба: где нужный чанк был в пуле и где оказался в контексте."""
    # Пул восстанавливается по местам поиска, а не по порядку после фильтра:
    # `pool_rank` должен отвечать на вопрос «что нашла лексика», и только.
    everyone = sorted(context.ranked.kept + context.ranked.dropped, key=lambda item: item.place)
    pool = [item.hit for item in everyone]
    kept = [item.hit for item in context.ranked.kept]

    hit_rank = _rank(_flags(probe, kept))
    read = kept[:hit_rank] if hit_rank else kept

    return Scored(
        question=probe.question,
        path=probe.path,
        pool_rank=_rank(_flags(probe, pool)),
        hit_rank=hit_rank,
        kept=len(kept),
        context_chars=sum(hit.chars for hit in kept),
        chars_to_hit=sum(hit.chars for hit in read),
        top=[
            {**item.as_dict(), "hit": flag}
            for item, flag in zip(context.ranked.kept, _flags(probe, kept), strict=True)
        ],
    )


def _metrics(scored: list[Scored]) -> dict[str, object]:
    total = len(scored)
    if not total:
        return {}

    found = [item for item in scored if item.hit_rank]
    in_pool = [item for item in scored if item.in_pool]

    return {
        "probes": total,
        **{
            f"recall@{rank}": round(
                sum(1 for item in scored if item.hit_rank and item.hit_rank <= rank) / total, 3
            )
            for rank in RANKS
        },
        # Потолок режима: нужный чанк попал в пул, а дальше дело за фильтром.
        "pool_recall": round(len(in_pool) / total, 3),
        "mrr@5": round(sum(1 / item.hit_rank for item in found) / total, 3),
        "kept": round(sum(item.kept for item in scored) / total, 2),
        "context_chars": round(sum(item.context_chars for item in scored) / total),
        "chars_to_hit": round(sum(item.chars_to_hit for item in scored) / total),
        "empty": round(sum(1 for item in scored if item.empty) / total, 3),
        "lost": round(sum(1 for item in in_pool if item.lost) / len(in_pool), 3)
        if in_pool
        else None,
        "paths": round(
            sum(len({row["path"] for row in item.top}) for item in scored) / total, 2
        ),
    }


# --- прогон -------------------------------------------------------------------


def _plan(rewriter: str, reranker: str, *, threshold: float | None = None) -> Plan:
    return Plan.of(rewriter=rewriter, reranker=reranker, threshold=threshold)


def _run(probes: list[Probe], plan: Plan) -> list[Scored]:
    return [_score(probe, pipeline.prepare_cached(probe.question, plan)) for probe in probes]


async def compare(
    rewriters: tuple[str, ...] = rewrite.MODES,
    rerankers: tuple[str, ...] = rerank.RERANKERS,
) -> dict[str, object]:
    """Матрица «переписывание × реранкер» на всём наборе."""
    loaded, notes = load()
    if not loaded:
        raise RuntimeError(notes[0] if notes else "Набор проб пуст.")

    info = index.require()
    plans = {
        (rewriter, reranker): _plan(rewriter, reranker)
        for rewriter in rewriters
        for reranker in rerankers
    }
    await pipeline.heat([probe.question for probe in loaded], list(plans.values()))

    results: dict[str, dict[str, object]] = {}
    for rewriter in rewriters:
        by_reranker: dict[str, object] = {}

        for reranker in rerankers:
            plan = plans[(rewriter, reranker)]
            scored = await asyncio.to_thread(_run, loaded, plan)
            by_reranker[reranker] = {
                "plan": plan.as_dict(),
                "metrics": _metrics(scored),
                "scored": [asdict(item) for item in scored],
            }

        results[rewriter] = by_reranker

    return {
        "index": info,
        "probes": [asdict(probe) for probe in loaded],
        "notes": notes,
        "rewriters": list(rewriters),
        "rerankers": list(rerankers),
        "rewriter_titles": rewrite.MODE_TITLES,
        "reranker_titles": rerank.RERANKER_TITLES,
        "retriever": retrieve.DEFAULT,
        "pool": pipeline.POOL,
        "top_k": pipeline.TOP_K,
        "thresholds": rerank.THRESHOLDS,
        "per_path": rerank.PER_PATH,
        "overlap_share": OVERLAP_SHARE,
        "cost": rewrite.cost(rewriters, [probe.question for probe in loaded]),
        "results": results,
    }


async def sweep(
    rerankers: tuple[str, ...] = ("heuristic", "vector", "llm"),
    rewriter: str = "none",
    values: tuple[float, ...] = SWEEP,
) -> dict[str, object]:
    """Развертка по порогу: оценки те же, меняется только отсечка.

    Поэтому она и бесплатна — оценки уже в кэшах, и каждое значение порога стоит
    одной расстановки на пробу. Выбирать порог иначе, чем по этой таблице,
    значило бы ставить его на глаз.
    """
    loaded, notes = load()
    if not loaded:
        raise RuntimeError(notes[0] if notes else "Набор проб пуст.")

    info = index.require()
    await pipeline.heat(
        [probe.question for probe in loaded],
        [_plan(rewriter, reranker) for reranker in rerankers],
    )

    results: dict[str, list[dict[str, object]]] = {}
    for reranker in rerankers:
        rows: list[dict[str, object]] = []
        for value in values:
            plan = _plan(rewriter, reranker, threshold=value)
            scored = await asyncio.to_thread(_run, loaded, plan)
            rows.append({"threshold": round(value, 3), "metrics": _metrics(scored)})
        results[reranker] = rows

    return {
        "index": info,
        "notes": notes,
        "probes": len(loaded),
        "rewriter": rewriter,
        "rewriter_title": rewrite.MODE_TITLES[rewriter],
        "rerankers": list(rerankers),
        "reranker_titles": rerank.RERANKER_TITLES,
        "values": [round(value, 3) for value in values],
        "chosen": rerank.THRESHOLDS,
        "pool": pipeline.POOL,
        "top_k": pipeline.TOP_K,
        "per_path": rerank.PER_PATH,
        "results": results,
    }
