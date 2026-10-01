"""Три стратегии разбиения на чанки над одним и тем же каноническим текстом.

Чанк здесь — это не строка, а диапазон `[start, end)` в тексте документа. Из
этого следует всё остальное: стратегии можно сравнивать между собой, потому что
они говорят в одних координатах, а эталон вопроса не привязан ни к одной из них.

- `fixed` — окно фиксированного размера в токенах с перекрытием, структура не
  учитывается вовсе. Размер считается в токенах той же модели, что делает
  эмбеддинги: мерить чанки в символах, а бюджет модели в токенах — значит мерить
  двумя линейками.
- `structural` — по структуре, и под одним именем тут три разных парсера:
  markdown по заголовкам, Python по `def`/`class`, PDF по страницам. Границы
  разные, идея одна: резать там, где автор сам закончил мысль.
- `hybrid` — структура плюс страховка от её крайностей: раздел на две страницы
  режется внутри себя, раздел в три строки склеивается с соседом.

Ни одна стратегия не теряет текст: у markdown и кода соседние чанки стыкуются
встык, и каждый символ документа попадает ровно в один чанк (в `fixed` — минимум
в один, из-за перекрытия). Исключение одно — короткие огрызки структуры в
`structural`, и это не оплошность, а её честная цена, которую и должно показать
сравнение.
"""

import ast
import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

from corpus import Document

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
COMMENT = re.compile(r"^\s*#")

# Раздел короче — это заголовок с одной строкой под ним; в индексе от него шум.
MIN_SECTION_CHARS = 40

FIXED_TOKENS = 400
FIXED_OVERLAP = 80
HYBRID_MAX_TOKENS = 400
HYBRID_MIN_TOKENS = 60
HYBRID_OVERLAP = 40

STRATEGIES = ("fixed", "structural", "hybrid")
STRATEGY_TITLES = {
    "fixed": "фиксированный размер",
    "structural": "по структуре",
    "hybrid": "структура с потолком и полом",
}


class Tokenizer(Protocol):
    """То, что нужно от токенайзера: границы токенов в тексте и их число."""

    def offsets(self, text: str) -> list[tuple[int, int]]: ...

    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class Chunk:
    """Чанк вместе со всеми метаданными: их же потом видно в поиске."""

    strategy: str
    source: str
    path: str
    title: str
    section: str
    ordinal: int
    start: int
    end: int
    text: str
    tokens: int

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


@dataclass(frozen=True)
class Segment:
    """Промежуточный результат парсера: кусок текста и то, чем он назван."""

    start: int
    end: int
    section: str


def _line_starts(text: str) -> list[int]:
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n":
            starts.append(index + 1)
    return starts


def _lines(text: str) -> list[tuple[int, str]]:
    """Строки вместе со смещением начала каждой: без смещений резать нечем."""
    result: list[tuple[int, str]] = []
    offset = 0
    for line in text.split("\n"):
        result.append((offset, line))
        offset += len(line) + 1
    return result


# --- парсеры структуры -------------------------------------------------------


def _markdown_segments(document: Document) -> list[Segment]:
    """Разделы markdown по заголовкам `#`–`###`, с цепочкой заголовков в имени.

    Заголовки внутри ``` заголовками не считаются: в README этого проекта есть
    блоки с markdown внутри, и без учёта заборов они рвали бы разделы.
    """
    boundaries: list[tuple[int, int, str]] = []
    fence: str | None = None
    seen_title = False

    for offset, line in _lines(document.text):
        opening = FENCE.match(line)
        if opening:
            mark = opening.group(1)
            fence = None if fence == mark else (fence or mark)
            continue

        match = None if fence else HEADING.match(line)
        if match is None:
            continue

        level, name = len(match.group(1)), match.group(2).strip()
        if level == 1 and not seen_title:
            seen_title = True
            continue

        boundaries.append((offset, level, name))

    chain: list[str] = []
    segments: list[Segment] = []
    cuts = [offset for offset, _, _ in boundaries] + [len(document.text)]

    if cuts[0] > 0:
        segments.append(Segment(0, cuts[0], "вступление"))

    for index, (offset, level, name) in enumerate(boundaries):
        del chain[max(level - 2, 0) :]
        chain.append(name)
        segments.append(Segment(offset, cuts[index + 1], " / ".join(chain)))

    return segments


def _code_segments(document: Document) -> list[Segment]:
    """Код по верхнеуровневым `def` и `class`, вместе с декораторами и комментарием над ними.

    Границей объявления считается не строка `def`, а первая строка блока
    комментариев над ним: в этом проекте там лежит объяснение, зачем функция
    нужна, и отрывать его от кода — значит выбрасывать самое осмысленное.
    """
    try:
        tree = ast.parse(document.text)
    except SyntaxError:
        return [Segment(0, len(document.text), "модуль")]

    starts = _line_starts(document.text)
    lines = document.text.split("\n")
    cuts: list[tuple[int, str]] = []

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue

        first = min([node.lineno] + [item.lineno for item in node.decorator_list])
        index = first - 1
        while index > 0 and COMMENT.match(lines[index - 1]):
            index -= 1

        kind = "class" if isinstance(node, ast.ClassDef) else "def"
        cuts.append((starts[index], f"{kind} {node.name}"))

    if not cuts:
        return [Segment(0, len(document.text), "модуль")]

    segments: list[Segment] = []
    if cuts[0][0] > 0:
        segments.append(Segment(0, cuts[0][0], "модуль"))

    edges = [offset for offset, _ in cuts] + [len(document.text)]
    for index, (offset, name) in enumerate(cuts):
        segments.append(Segment(offset, edges[index + 1], name))

    return segments


def _pdf_segments(document: Document) -> list[Segment]:
    """Страницы PDF: единственная структура, которая у него точно есть."""
    if not document.pages:
        return [Segment(0, len(document.text), "весь документ")]

    return [
        Segment(page.start, page.end, f"страница {page.number}") for page in document.pages
    ]


PARSERS = {"docs": _markdown_segments, "code": _code_segments, "pdf": _pdf_segments}


def outline(document: Document) -> list[Segment]:
    """Структура документа как её видит парсер — до отсева и до склейки.

    Нужна не только стратегиям: генератор вопросов берёт отсюда имя раздела, в
    который попал отрывок, иначе вопрос выходит без понятия, о чём он.
    """
    return PARSERS[document.source](document)


def section_at(outline_: list[Segment], offset: int) -> str:
    for segment in outline_:
        if segment.start <= offset < segment.end:
            return segment.section
    return "вступление"


# --- стратегии ---------------------------------------------------------------


def _windows(
    offsets: list[tuple[int, int]], size: int, overlap: int
) -> list[tuple[int, int, int, int]]:
    """Окна по токенам: возвращает границы в символах и номера токенов."""
    if not offsets:
        return []

    step = max(size - overlap, 1)
    windows: list[tuple[int, int, int, int]] = []
    start = 0

    while start < len(offsets):
        window = offsets[start : start + size]
        windows.append((window[0][0], window[-1][1], start, start + len(window)))
        if start + size >= len(offsets):
            break
        start += step

    return windows


def _fixed(
    document: Document, tokenizer: Tokenizer, *, size: int, overlap: int
) -> list[Segment]:
    offsets = tokenizer.offsets(document.text)
    return [
        Segment(start, end, f"токены {first}–{last}")
        for start, end, first, last in _windows(offsets, size, overlap)
    ]


def _structural(document: Document) -> list[Segment]:
    parser = PARSERS[document.source]
    return [
        segment
        for segment in parser(document)
        if len(document.text[segment.start : segment.end].strip()) >= MIN_SECTION_CHARS
    ]


def _split_long(
    document: Document, segment: Segment, tokenizer: Tokenizer, *, size: int, overlap: int
) -> list[Segment]:
    """Длинный раздел режется внутри себя, но остаётся назван своим именем."""
    body = document.text[segment.start : segment.end]
    offsets = tokenizer.offsets(body)
    windows = _windows(offsets, size, overlap)

    if len(windows) <= 1:
        return [segment]

    return [
        Segment(
            segment.start + start,
            segment.start + end,
            f"{segment.section} · часть {number}",
        )
        for number, (start, end, _, _) in enumerate(windows, start=1)
    ]


def _hybrid(
    document: Document,
    tokenizer: Tokenizer,
    *,
    max_tokens: int,
    min_tokens: int,
    overlap: int,
) -> list[Segment]:
    """Структура, у которой подрезан потолок и поднят пол.

    Короткие разделы склеиваются с соседом, а не выбрасываются, поэтому пустых
    мест в тексте не остаётся — в отличие от `structural`, где огрызки отсеяны.
    """
    parser = PARSERS[document.source]
    segments: list[Segment] = []

    for segment in parser(document):
        segments.extend(
            _split_long(document, segment, tokenizer, size=max_tokens, overlap=overlap)
        )

    merged: list[Segment] = []
    for segment in segments:
        body = document.text[segment.start : segment.end]
        short = tokenizer.count(body) < min_tokens

        if short and merged:
            previous = merged[-1]
            combined = document.text[previous.start : segment.end]
            if tokenizer.count(combined) <= max_tokens:
                merged[-1] = Segment(
                    previous.start, segment.end, f"{previous.section} + {segment.section}"
                )
                continue

        merged.append(segment)

    # Первый раздел мог оказаться коротким, а склеивать его было не с кем назад.
    if len(merged) > 1 and tokenizer.count(document.text[merged[0].start : merged[0].end]) < min_tokens:
        head, following = merged[0], merged[1]
        merged[:2] = [
            Segment(head.start, following.end, f"{head.section} + {following.section}")
        ]

    return [
        segment
        for segment in merged
        if document.text[segment.start : segment.end].strip()
    ]


def build(
    documents: list[Document],
    strategy: str,
    tokenizer: Tokenizer,
    **params: int,
) -> list[Chunk]:
    """Чанки одной стратегии по всему корпусу, с непрерывной нумерацией."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Стратегии {strategy!r} нет. Есть: {', '.join(STRATEGIES)}.")

    chunks: list[Chunk] = []

    for document in documents:
        match strategy:
            case "fixed":
                segments = _fixed(
                    document,
                    tokenizer,
                    size=params.get("size", FIXED_TOKENS),
                    overlap=params.get("overlap", FIXED_OVERLAP),
                )
            case "structural":
                segments = _structural(document)
            case _:
                segments = _hybrid(
                    document,
                    tokenizer,
                    max_tokens=params.get("max_tokens", HYBRID_MAX_TOKENS),
                    min_tokens=params.get("min_tokens", HYBRID_MIN_TOKENS),
                    overlap=params.get("overlap", HYBRID_OVERLAP),
                )

        for ordinal, segment in enumerate(segments):
            text = document.text[segment.start : segment.end]
            if not text.strip():
                continue

            chunks.append(
                Chunk(
                    strategy=strategy,
                    source=document.source,
                    path=document.path,
                    title=document.title,
                    section=segment.section,
                    ordinal=ordinal,
                    start=segment.start,
                    end=segment.end,
                    text=text,
                    tokens=tokenizer.count(text),
                )
            )

    return chunks


def defaults(strategy: str) -> dict[str, int]:
    """Параметры стратегии — их же надо записать рядом с индексом."""
    match strategy:
        case "fixed":
            return {"size": FIXED_TOKENS, "overlap": FIXED_OVERLAP}
        case "structural":
            return {"min_section_chars": MIN_SECTION_CHARS}
        case _:
            return {
                "max_tokens": HYBRID_MAX_TOKENS,
                "min_tokens": HYBRID_MIN_TOKENS,
                "overlap": HYBRID_OVERLAP,
            }
