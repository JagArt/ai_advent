"""Сравнение стратегий chunking: один набор эталонов, три способа спросить.

Главная трудность сравнения в том, что «правильного чанка» не существует. Три
стратегии режут текст по-разному, и номер чанка у них значит разное, поэтому
эталон задан там, где стратегий ещё нет — отрывком `[start, end)` в каноническом
тексте документа. Чанк засчитан, если он из того же файла и накрывает хотя бы
половину эталонного отрывка. Набор от этого получается один для всех стратегий и
ни к одной из них не подогнан.

Отрывки нарочно короткие — две-четыре фразы, 200–600 символов — и начинаются в
случайном месте документа. Если бы эталоном был раздел, границы эталона совпадали
бы с границами `structural`, и сравнение доказывало бы само себя.

Спросить про один и тот же отрывок можно по-разному, и это оказалось важнее, чем
казалось. Поэтому у каждого эталона три пробы:

- `query` — короткий поисковый запрос, 3–7 слов, как набирают в поле поиска;
- `question` — вопрос на естественном языке, как спросил бы человек у агента;
- `passage` — сам эталонный отрывок, модель для него не нужна вовсе.

Третья проба — не вопрос, а потолок: она измеряет индекс в отрыве от того, умеет
ли модель связать формулировку вопроса с текстом ответа. Разница между `passage` и
`question` — это ровно цена статических эмбеддингов, и без неё низкий `recall`
легко списать на разбиение, которое тут не при чём.

Пробы придумывает DeepSeek: сорок наборов руками — это день работы и вкусовщина, а
модель здесь не участвует ни в индексации, ни в поиске, поэтому повлиять на
результат в чью-то пользу не может. Набор кешируется в `probes.json` и коммитится:
сравнение должно воспроизводиться, а не пересчитываться каждый раз заново.

Метрик две группы. Первая — цена индекса: сколько чанков, какого размера, сколько
времени и байт. Вторая — качество поиска, и в ней рядом с `recall` обязательно
стоит `символов до ответа`: критерий попадания по перекрытию сам по себе выгоден
большим чанкам — накрыть эталон легче тому, кто тащит больше текста. Одно число
без другого читается неверно.
"""

import asyncio
import json
import random
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import chunking
import corpus
import embed
import index
import llm
from corpus import Document

PROBES_PATH = Path(__file__).resolve().parent / "probes.json"

# Сто двадцать, а не сорок: на сорока пробах разница в три попадания — это 0.07
# recall, то есть ровно тот порядок, в котором стратегии здесь и расходятся.
PROBES = 120
SEED = 21
CONCURRENCY = 8
TEMPERATURE = 0.3
MAX_TOKENS = 180

MIN_ANCHOR_CHARS = 200
MAX_ANCHOR_CHARS = 600
MIN_ANCHOR_WORDS = 20

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

SENTENCE = re.compile(r"(?<=[.!?…:])\s+")
REFUSAL = "—"

PROMPT = (
    "Ты готовишь пробы для проверки поиска по документации проекта AI Advent. "
    "На входе отрывок из файла проекта. Придумай по нему две пробы и выведи их "
    "ровно двумя строками, без пояснений:\n"
    "запрос: короткий поисковый запрос, 3–7 слов, как набирают в поле поиска — "
    "ключевые слова, без вопросительных слов и знака вопроса\n"
    "вопрос: один самодостаточный вопрос по-русски, на который отвечает именно "
    "этот отрывок; из вопроса должно быть понятно, о чём речь, без отрывка\n"
    "И запрос, и вопрос должны находить этот отрывок, а не любую страницу проекта. "
    "Не цитируй отрывок дословно длиннее трёх слов подряд.\n"
    f"Если отрывок неинформативен — оглавление, список импортов, набор символов — "
    f"ответь одним символом «{REFUSAL}»."
)

LINE = re.compile(r"^\s*(запрос|вопрос)\s*:\s*(.+)$", re.IGNORECASE)


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
    """Что стратегия ответила на одну пробу."""

    probe: str
    path: str
    hit_rank: int | None
    chars_to_hit: int
    context_chars: int
    top: list[dict[str, object]]


# --- набор проб --------------------------------------------------------------


def _units(document: Document) -> list[tuple[int, int]]:
    """Единицы, из которых собирается отрывок: фразы в тексте, строки в коде."""
    spans: list[tuple[int, int]] = []
    offset = 0

    if document.source == "code":
        for line in document.text.split("\n"):
            if line.strip():
                spans.append((offset, offset + len(line)))
            offset += len(line) + 1
        return spans

    for piece in SENTENCE.split(document.text):
        start = document.text.find(piece, offset)
        if start < 0:
            continue
        if piece.strip():
            spans.append((start, start + len(piece)))
        offset = start + len(piece)
    return spans


def _anchor(document: Document, generator: random.Random) -> tuple[int, int] | None:
    """Случайный отрывок в 200–600 символов, начинающийся на границе фразы или строки."""
    units = _units(document)
    if not units:
        return None

    for _ in range(12):
        first = generator.randrange(len(units))
        start = units[first][0]
        end = start

        for unit_start, unit_end in units[first:]:
            if unit_start - start >= MAX_ANCHOR_CHARS:
                break
            end = unit_end
            if end - start >= MIN_ANCHOR_CHARS:
                break

        passage = document.text[start:end]
        if MIN_ANCHOR_CHARS <= len(passage) <= MAX_ANCHOR_CHARS:
            if len(passage.split()) >= MIN_ANCHOR_WORDS:
                return start, end

    return None


def _anchors(documents: list[Document], count: int, seed: int) -> list[Probe]:
    """Отрывки по всему корпусу, поровну на каждый источник.

    Кода в корпусе вчетверо больше файлов, чем документации, и при случайной
    выборке пробы про markdown просто потерялись бы. Источникам выдаётся равная
    доля: разбиение по заголовкам и разбиение по `def` — это два разных сюжета, и
    сравнивать их надо на сопоставимом числе проб.
    """
    generator = random.Random(seed)
    pool = [document for document in documents if len(document.text) > MAX_ANCHOR_CHARS * 2]
    generator.shuffle(pool)

    by_source: dict[str, list[Document]] = {}
    for document in pool:
        by_source.setdefault(document.source, []).append(document)

    outlines = {document.path: chunking.outline(document) for document in pool}
    anchors: list[Probe] = []
    seen: set[tuple[str, int]] = set()
    sources = sorted(by_source)

    for position in range(count * 6):
        if len(anchors) >= count:
            break

        group = by_source[sources[position % len(sources)]]
        document = group[(position // len(sources)) % len(group)]
        span = _anchor(document, generator)
        if span is None or (document.path, span[0]) in seen:
            continue

        seen.add((document.path, span[0]))
        anchors.append(
            Probe(
                query="",
                question="",
                path=document.path,
                section=chunking.section_at(outlines[document.path], span[0]),
                start=span[0],
                end=span[1],
                passage=document.text[span[0] : span[1]],
                doc_sha256=document.sha256,
            )
        )

    return anchors


def _parse(text: str) -> tuple[str, str] | None:
    found: dict[str, str] = {}
    for line in text.splitlines():
        match = LINE.match(line)
        if match:
            found[match.group(1).lower()] = match.group(2).strip().strip('"«»')

    query, question = found.get("запрос", ""), found.get("вопрос", "")
    return (query, question) if len(query) >= 8 and len(question) >= 12 else None


async def _ask(anchor: Probe, title: str, gate: asyncio.Semaphore) -> Probe | None:
    task = (
        f"Файл: {anchor.path}\n"
        f"Документ: {title}\n"
        f"Раздел: {anchor.section}\n\n"
        f"Отрывок:\n{anchor.passage}"
    )

    async with gate:
        try:
            text = await llm.complete(
                [{"role": "system", "content": PROMPT}, {"role": "user", "content": task}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
        except Exception:
            return None

    parsed = _parse(text) if text and not text.startswith(REFUSAL) else None
    if parsed is None:
        return None

    return Probe(**{**asdict(anchor), "query": parsed[0], "question": parsed[1]})


async def generate(count: int = PROBES, seed: int = SEED) -> list[Probe]:
    """Пробы по случайным отрывкам корпуса. Неинформативные отрывки модель отклоняет."""
    documents = corpus.load()
    titles = {document.path: document.title for document in documents}
    # С запасом: часть отрывков модель отклонит, и добирать их вторым проходом незачем.
    anchors = _anchors(documents, round(count * 1.4), seed)

    gate = asyncio.Semaphore(CONCURRENCY)
    results = await asyncio.gather(*(_ask(anchor, titles[anchor.path], gate) for anchor in anchors))

    probes = [item for item in results if item is not None]
    if len(probes) < count // 2:
        raise RuntimeError(
            f"Модель вернула всего {len(probes)} проб из {len(anchors)} отрывков: "
            "сравнивать на таком наборе нечего."
        )

    return probes[:count]


def save(probes: list[Probe]) -> None:
    payload = {
        "model": llm.MODEL,
        "seed": SEED,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "count": len(probes),
        "probes": [asdict(probe) for probe in probes],
    }
    PROBES_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with index.db() as conn:
        conn.execute("DELETE FROM probes")
        conn.executemany(
            "INSERT INTO probes (query, question, path, section, start_char, end_char, passage,"
            " doc_sha256, model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (item.query, item.question, item.path, item.section, item.start, item.end,
                 item.passage, item.doc_sha256, payload["model"], payload["created_at"])
                for item in probes
            ],
        )


def load_probes() -> tuple[list[Probe], list[str]]:
    """Набор из кеша с привязкой к текущему тексту документов.

    Файл в репозитории живёт и меняется, а смещения эталона привязаны к его тексту.
    Поэтому у пробы хранится и sha документа, и сам отрывок: если файл изменился,
    отрывок ищется в новом тексте заново, а если не нашёлся — проба выбывает. Молча
    считать метрики по съехавшим эталонам хуже, чем потерять часть набора.
    """
    if not PROBES_PATH.is_file():
        return [], ["Набора проб нет: соберите его командой python day21/scenarios.py probes"]

    payload = json.loads(PROBES_PATH.read_text(encoding="utf-8"))
    documents = {document.path: document for document in corpus.load()}
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


def _flags(probe: Probe, hits: list[index.Hit]) -> list[bool]:
    """Накрыл ли чанк эталонный отрывок хотя бы наполовину."""
    needed = (probe.end - probe.start) * OVERLAP_SHARE
    return [
        hit.path == probe.path
        and max(0, min(hit.end, probe.end) - max(hit.start, probe.start)) >= needed
        for hit in hits
    ]


def _score(probe: Probe, kind: str, hits: list[index.Hit]) -> Scored:
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
        # Ожидаемая цена одного верного попадания: промахи оплачиваются тоже.
        "chars_per_hit": (
            round(sum(item.chars_to_hit for item in scored) / len(found)) if found else None
        ),
    }


def _by_source(scored: list[Scored], sources: dict[str, str]) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Scored]] = {}
    for item in scored:
        grouped.setdefault(sources.get(item.path, "?"), []).append(item)

    return {source: _metrics(items) for source, items in sorted(grouped.items())}


def compare(
    kinds: tuple[str, ...] = KINDS,
    probes: list[Probe] | None = None,
    strategies: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Прогнать набор по всем собранным индексам каждым способом спросить."""
    loaded, notes = (probes, []) if probes else load_probes()
    if not loaded:
        raise RuntimeError(notes[0] if notes else "Набор проб пуст.")

    built = index.built()
    order = strategies or tuple(strategy for strategy in chunking.STRATEGIES if strategy in built)
    if not order:
        raise RuntimeError("Ни один индекс не собран: python day21/scenarios.py build")

    sources = {
        document["path"]: document["source"]
        for document in (index.document(probe.path) or {} for probe in loaded)
        if document
    }

    results: dict[str, dict[str, object]] = {}
    for kind in kinds:
        # Вектор пробы один на все стратегии: сравнивается разбиение, а не эмбеддинги.
        vectors = embed.encode([probe.text(kind) for probe in loaded]).vectors
        by_strategy: dict[str, object] = {}

        for strategy in order:
            scored = [
                _score(probe, kind, index.search_vector(strategy, vector, TOP_K))
                for probe, vector in zip(loaded, vectors, strict=True)
            ]
            by_strategy[strategy] = {
                "index": built[strategy],
                "metrics": _metrics(scored),
                "by_source": _by_source(scored, sources),
                "scored": [asdict(item) for item in scored],
            }

        results[kind] = by_strategy

    return {
        "probes": [asdict(probe) for probe in loaded],
        "notes": notes,
        "kinds": list(kinds),
        "strategies": list(order),
        "top_k": TOP_K,
        "overlap_share": OVERLAP_SHARE,
        "results": results,
    }
