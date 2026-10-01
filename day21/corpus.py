"""Корпус: документация, код и PDF приводятся к одному каноническому тексту.

Три источника режутся по-разному, но сравнивать стратегии chunking можно только
тогда, когда у документа есть **один** текст, в котором смещения означают одно и
то же. Поэтому здесь каждый документ превращается в строку, и дальше весь
пайплайн говорит смещениями в ней: чанк — это `[start, end)`, эталон вопроса —
тоже `[start, end)`. Для markdown и кода канонический текст — это файл с
нормализованными переводами строк; для PDF — то, что вытащил `pypdf`, потому что
никакого «исходного текста» у PDF нет вовсе.

У PDF из этого следует ещё одно: границы страниц существуют только в канонической
строке, и запоминать их надо здесь. Структурная стратегия без них не отличит
страницу от страницы, а номер страницы — единственная осмысленная «секция»,
которая у PDF есть.
"""

import ast
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "corpus"

DAY_DIR = re.compile(r"^day(\d+)$")
DOC_NAMES = ("README.md", "TASK.md")
SKIP_DIRS = {"__pycache__", ".venv", ".git", ".idea", "static", "out"}

# Файл короче этого — не документ, а заглушка: в индексе от него только шум.
MIN_CHARS = 200

TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
EXTRA_BLANKS = re.compile(r"\n{3,}")
HEADING = re.compile(r"^#\s+(.+)$", re.MULTILINE)

SOURCE_TITLES = {"docs": "документация", "code": "код", "pdf": "PDF"}


@dataclass(frozen=True)
class Page:
    """Страница PDF: номер и её место в каноническом тексте документа."""

    number: int
    start: int
    end: int


@dataclass(frozen=True)
class Document:
    """Документ корпуса. `text` — единственная система координат для смещений."""

    path: str
    source: str
    title: str
    text: str
    pages: tuple[Page, ...] = field(default_factory=tuple)

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    def slice(self, start: int, end: int) -> str:
        return self.text[start:end]


def _normalize(text: str) -> str:
    """Переводы строк и хвостовые пробелы к одному виду, остальное не трогаем.

    Смещения обязаны указывать в этот текст, поэтому чистка делается один раз и
    до всего: каждая правка после неё расъехалась бы с эталонами вопросов.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = TRAILING_SPACE.sub("", text)
    return EXTRA_BLANKS.sub("\n\n", text).strip() + "\n"


def _days() -> list[Path]:
    return sorted(
        (path for path in ROOT.iterdir() if path.is_dir() and DAY_DIR.match(path.name)),
        key=lambda path: int(DAY_DIR.match(path.name).group(1)),
    )


def _markdown_title(text: str, path: Path) -> str:
    match = HEADING.search(text)
    return match.group(1).strip() if match else path.name


def _code_title(text: str, path: Path) -> str:
    """Заголовок модуля — первая строка его docstring: в этом проекте она есть везде."""
    try:
        docstring = ast.get_docstring(ast.parse(text))
    except SyntaxError:
        docstring = None

    if docstring:
        first = docstring.strip().splitlines()[0].strip()
        if first:
            return first

    return str(path.relative_to(ROOT))


# --- источники ---------------------------------------------------------------


def _doc_files() -> list[Path]:
    paths = [ROOT / "README.md"]
    paths.extend(day / name for day in _days() for name in DOC_NAMES)
    return [path for path in paths if path.is_file()]


def _code_files() -> list[Path]:
    paths: list[Path] = []
    for day in _days():
        for path in sorted(day.rglob("*.py")):
            if SKIP_DIRS.isdisjoint(part for part in path.relative_to(day).parts[:-1]):
                paths.append(path)
    return paths


def _pdf_files() -> list[Path]:
    """PDF кладёт сюда человек. Пустая папка — обычное состояние, не ошибка."""
    return sorted(PDF_DIR.glob("*.pdf")) if PDF_DIR.is_dir() else []


def _read_pdf(path: Path) -> tuple[str, tuple[Page, ...]]:
    """Текст PDF страница за страницей, с запомненными границами страниц."""
    import logging

    from pypdf import PdfReader

    # pypdf на уровне WARNING вываливает в вывод целые словари шрифтов с таблицами
    # ширин — на десятки строк на каждый нестандартный Type1. Читаемость прогона это
    # убивает, а узнать из этих предупреждений нечего: текст либо извлёкся, либо нет.
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    reader = PdfReader(path)
    parts: list[str] = []
    pages: list[Page] = []
    cursor = 0

    for number, page in enumerate(reader.pages, start=1):
        body = _normalize(page.extract_text() or "")
        if not body.strip():
            continue

        parts.append(body)
        pages.append(Page(number=number, start=cursor, end=cursor + len(body)))
        cursor += len(body) + 1

    return "\n".join(parts), tuple(pages)


def scan() -> tuple[list[Document], list[str]]:
    """Весь корпус и список того, что в него не попало.

    Заметки нужны из-за PDF. Отсканированный PDF — это картинки: `pypdf` достаёт из
    него ноль символов, и документ честно выпадает из корпуса. Промолчать тут нельзя:
    человек положил файл в папку и вправе знать, почему его нет в индексе.
    """
    documents: list[Document] = []
    notes: list[str] = []

    for path in _doc_files():
        text = _normalize(path.read_text(encoding="utf-8"))
        if len(text) >= MIN_CHARS:
            documents.append(
                Document(
                    path=str(path.relative_to(ROOT)),
                    source="docs",
                    title=_markdown_title(text, path),
                    text=text,
                )
            )

    for path in _code_files():
        text = _normalize(path.read_text(encoding="utf-8"))
        if len(text) >= MIN_CHARS:
            documents.append(
                Document(
                    path=str(path.relative_to(ROOT)),
                    source="code",
                    title=_code_title(text, path),
                    text=text,
                )
            )

    for path in _pdf_files():
        name = str(path.relative_to(ROOT))
        try:
            text, pages = _read_pdf(path)
        except Exception as exc:
            notes.append(f"{name}: не читается ({type(exc).__name__})")
            continue

        if len(text) < MIN_CHARS:
            notes.append(
                f"{name}: текстового слоя нет — похоже на скан, из него нужен OCR"
                if not text.strip()
                else f"{name}: текста всего {len(text)} символов, меньше порога {MIN_CHARS}"
            )
            continue

        documents.append(
            Document(
                path=name,
                source="pdf",
                title=path.stem,
                text=text,
                pages=pages,
            )
        )

    return documents, notes


def load() -> list[Document]:
    """Весь корпус одним списком. Порядок устойчив: day1 → day21, внутри — по имени."""
    return scan()[0]


def summary(documents: list[Document]) -> dict[str, object]:
    """Сводка корпуса: страницами, чтобы было видно, добрал ли он объём задания."""
    by_source: dict[str, dict[str, int]] = {}
    for document in documents:
        slot = by_source.setdefault(source := document.source, {"files": 0, "chars": 0})
        slot["files"] += 1
        slot["chars"] += document.chars
        by_source[source] = slot

    chars = sum(document.chars for document in documents)
    return {
        "files": len(documents),
        "chars": chars,
        "words": sum(len(document.text.split()) for document in documents),
        # 1800 символов — машинописная страница; в задании объём назван страницами.
        "pages": round(chars / 1800, 1),
        "by_source": by_source,
    }


if __name__ == "__main__":
    documents, notes = scan()
    stats = summary(documents)
    print(f"Документов: {stats['files']}, символов: {stats['chars']}, страниц: {stats['pages']}")
    for source, slot in stats["by_source"].items():
        pages = round(slot["chars"] / 1800, 1)
        print(f"  {SOURCE_TITLES[source]:14} {slot['files']:3} файлов, {pages:6} страниц")

    for note in notes:
        print(f"\nМимо корпуса: {note}")

    if not _pdf_files():
        print(f"\nPDF нет. Положите файлы в {PDF_DIR.relative_to(ROOT)}/ — они подхватятся сами.")
