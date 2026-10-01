"""Сравнение трёх ретриверов на эталонном наборе day21.

Выбирать ретривер под RAG на глаз нельзя: выдача правдоподобна у всех троих, а
отличаются они ровно в тех случаях, которые на глаз и не посмотришь. Поэтому
набор взят готовый — [day21/probes.json](../day21/probes.json), 120 эталонных
отрывков, и переносится сюда как есть. Пересобирать его моделью заново значило бы
сравнивать день с днём по разным линейкам.

Устройство набора стоит повторить, потому что от него зависит смысл чисел. Эталон
— это не «правильный чанк», а диапазон `[start, end)` в каноническом тексте
документа: отрывок в две-четыре фразы, начинающийся в случайном месте. Чанк
засчитан, если он из того же файла и накрыл не меньше половины отрывка. Эталон
живёт на уровне, где ретриверов ещё нет, и ни к одному из них не подогнан.

Спросить про один отрывок можно по-разному, и у каждого эталона три пробы:
короткий поисковый запрос, вопрос на естественном языке и сам отрывок. Третья —
потолок: она меряет индекс в отрыве от того, умеет ли поиск связать формулировку
вопроса с текстом ответа. В day21 разрыв между потолком и вопросом составил 0.6
recall и оказался главным выводом дня. Здесь важна именно колонка «вопрос»: в
RAG приходит вопрос человека, а не отрывок из документа.

Рядом с `recall` обязательно стоит «символов до ответа». Критерий попадания по
перекрытию сам по себе выгоден тому, кто тащит больше текста, а в RAG этот текст
ещё и оплачивается токенами запроса. Одно число без другого читается неверно.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import corpus
import index
import retrieve
from corpus import Document

PROBES_PATH = Path(__file__).resolve().parent / "probes.json"

# Половина эталонного отрывка: чанк, накрывший меньше, ответа целиком не содержит.
OVERLAP_SHARE = 0.5
TOP_K = 5
RANKS = (1, 3, 5)

KINDS = ("query", "question", "passage")
KIND_TITLES = {
    "query": "поисковый запрос",
    "question": "вопрос",
    "passage": "эталонный отрывок (потолок)",
}


@dataclass(frozen=True)
class Probe:
    """Эталон и три способа его спросить: файл, диапазон, отрывок и две формулировки."""

    query: str
    question: str
    path: str
    section: str
    start: int
    end: int
    passage: str
    doc_sha256: str

    def text(self, kind: str) -> str:
        return {"query": self.query, "question": self.question, "passage": self.passage}[kind]


@dataclass(frozen=True)
class Scored:
    """Что ретривер ответил на одну пробу."""

    probe: str
    path: str
    hit_rank: int | None
    chars_to_hit: int
    context_chars: int
    top: list[dict[str, object]]


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


def _flags(probe: Probe, hits: list[retrieve.Hit]) -> list[bool]:
    """Накрыл ли чанк эталонный отрывок хотя бы наполовину."""
    needed = (probe.end - probe.start) * OVERLAP_SHARE
    return [
        hit.path == probe.path
        and max(0, min(hit.end, probe.end) - max(hit.start, probe.start)) >= needed
        for hit in hits
    ]


def _score(probe: Probe, kind: str, hits: list[retrieve.Hit]) -> Scored:
    flags = _flags(probe, hits)
    rank = flags.index(True) + 1 if any(flags) else None
    read = hits[:rank] if rank else hits

    return Scored(
        probe=probe.text(kind),
        path=probe.path,
        hit_rank=rank,
        chars_to_hit=sum(hit.chars for hit in read),
        context_chars=sum(hit.chars for hit in hits),
        top=[{**hit.as_dict(), "hit": flag} for hit, flag in zip(hits, flags, strict=True)],
    )


def _metrics(scored: list[Scored]) -> dict[str, object]:
    total = len(scored)
    if not total:
        return {}

    found = [item for item in scored if item.hit_rank]
    return {
        "probes": total,
        **{
            f"recall@{rank}": round(
                sum(1 for item in scored if item.hit_rank and item.hit_rank <= rank) / total, 3
            )
            for rank in RANKS
        },
        "mrr@5": round(sum(1 / item.hit_rank for item in found) / total, 3),
        "context@5": round(sum(item.context_chars for item in scored) / total),
        "chars_to_hit": round(sum(item.chars_to_hit for item in scored) / total),
    }


def _by_source(scored: list[Scored], sources: dict[str, str]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Scored]] = {}
    for item in scored:
        grouped.setdefault(sources.get(item.path, "?"), []).append(item)

    return {source: _metrics(items) for source, items in sorted(grouped.items())}


def compare(kinds: tuple[str, ...] = KINDS) -> dict[str, object]:
    """Прогнать набор по каждому ретриверу каждым способом спросить."""
    loaded, notes = load()
    if not loaded:
        raise RuntimeError(notes[0] if notes else "Набор проб пуст.")

    info = index.require()
    sources = index.sources()

    results: dict[str, dict[str, object]] = {}
    for kind in kinds:
        by_retriever: dict[str, object] = {}

        for name in retrieve.RETRIEVERS:
            scored = [
                _score(probe, kind, retrieve.search(probe.text(kind), TOP_K, name).hits)
                for probe in loaded
            ]
            by_retriever[name] = {
                "metrics": _metrics(scored),
                "by_source": _by_source(scored, sources),
                "scored": [asdict(item) for item in scored],
            }

        results[kind] = by_retriever

    return {
        "index": info,
        "probes": [asdict(probe) for probe in loaded],
        "notes": notes,
        "kinds": list(kinds),
        "retrievers": list(retrieve.RETRIEVERS),
        "titles": retrieve.RETRIEVER_TITLES,
        "top_k": TOP_K,
        "overlap_share": OVERLAP_SHARE,
        "results": results,
    }
