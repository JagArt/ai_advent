"""RAG для ревью: документация и код проекта в одном индексе FTS5, поиск без модели.

Ревьюеру нужны два рода контекста. Документация говорит, как день задуман и
почему так — разделы README и TASK, те же, что у ассистента day31; разбор на
разделы импортируется прямо из `day31/docs.py`, как day28 брал поиск у day25. Код
говорит, что вокруг изменённого места на самом деле: функция, которую патч
зовёт, модуль, который он импортирует, соседний день с тем же решением.

Код режется не окнами, а по структуре. Для `.py` работает `ast`: чанк — это
функция, класс или метод с диапазоном строк; что между ними — импорты,
константы — отдельными чанками «модуль». Так замечание «в day31 эта же
функция делает иначе» опирается на целую функцию, а не на её половину.
Остальное — `.js`, `.html`, `.css`, `.yml` — окнами по 60 строк с перекрытием.

Индекс свой — `day32/index.db`, инкрементальный по sha файла; пересборка при
чистом дереве — миллисекунды. Токенайзер считает `_` частью слова: запрос
к коду идёт идентификаторами — `stream_chat`, `git_diff`, — и они должны
совпадать целиком, а не по половинкам.
"""

import ast
import hashlib
import importlib.util
import keyword
import re
import sqlite3
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DAY31 = ROOT / "day31"
DB_PATH = Path(__file__).resolve().parent / "index.db"

DOC_SUFFIXES = {".md"}
CODE_SUFFIXES = {".py", ".js", ".ts", ".html", ".css", ".yml", ".yaml", ".toml"}
EXCLUDE_PARTS = {".venv", "node_modules", "__pycache__", ".git"}
MAX_FILE_CHARS = 200_000
WINDOW_LINES = 60
WINDOW_OVERLAP = 10
MAX_CHUNK_CHARS = 4000
MIN_CHUNK_CHARS = 30
MAX_LIMIT = 10

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
STOPWORDS = set(keyword.kwlist) | {
    "self", "cls", "none", "true", "false", "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "print", "len", "range", "enumerate", "zip", "any", "all", "isinstance", "type", "object", "super",
    "const", "let", "var", "function", "document", "window", "this", "new", "null", "undefined", "await",
    "async", "return", "export", "default", "div", "span", "class", "style", "script", "href", "src",
    "the", "and", "for", "with", "from", "import", "not", "args", "kwargs", "value", "values", "item",
    "items", "key", "keys", "name", "text", "data", "result", "results", "path", "line", "lines",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path       TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    chunks     INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5 (
    path UNINDEXED,
    kind UNINDEXED,
    heading,
    body,
    line_start UNINDEXED,
    line_end UNINDEXED,
    tokenize = "unicode61 remove_diacritics 2 tokenchars '_'"
);
"""


def _load_day31_docs() -> ModuleType:
    """Модуль day31/docs.py под своим именем: простой `import docs` подхватил бы
    любой уже загруженный модуль с таким именем."""
    spec = importlib.util.spec_from_file_location("day31_docs", DAY31 / "docs.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Не найден {DAY31 / 'docs.py'}: индексу day32 нужен его разбор markdown на разделы.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_docs = _load_day31_docs()


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            conn.executescript(SCHEMA)
            yield conn


# --- корпус ------------------------------------------------------------------


def corpus() -> dict[str, str]:
    """Файлы под git и новые неигнорируемые, с родом: docs или code."""
    output = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    found: dict[str, str] = {}
    for name in sorted({line for line in output.splitlines() if line}):
        parts = Path(name).parts
        if EXCLUDE_PARTS.intersection(parts) or not (ROOT / name).is_file():
            continue
        suffix = Path(name).suffix.lower()
        if suffix in DOC_SUFFIXES:
            found[name] = "docs"
        elif suffix in CODE_SUFFIXES:
            found[name] = "code"
    return found


@dataclass(frozen=True)
class Chunk:
    heading: str
    line_start: int
    line_end: int
    body: str


def _windows(lines: list[str], offset: int = 0, label: str = "строки") -> list[Chunk]:
    chunks: list[Chunk] = []
    step = max(1, WINDOW_LINES - WINDOW_OVERLAP)
    for start in range(0, max(1, len(lines)), step):
        window = lines[start : start + WINDOW_LINES]
        body = "\n".join(window).strip()
        if len(body) >= MIN_CHUNK_CHARS:
            first, last = offset + start + 1, offset + start + len(window)
            chunks.append(Chunk(f"{label} {first}–{last}", first, last, body))
        if start + WINDOW_LINES >= len(lines):
            break
    return chunks


def _span(node: ast.AST) -> tuple[int, int]:
    start = min([node.lineno, *(item.lineno for item in getattr(node, "decorator_list", []))])
    return start, node.end_lineno or node.lineno


def python_chunks(text: str) -> list[Chunk]:
    """Функции, классы и методы как чанки; промежутки между ними — чанки «модуль»."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _windows(text.splitlines())

    lines = text.splitlines()
    spans: list[tuple[str, int, int]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        start, end = _span(node)
        if isinstance(node, ast.ClassDef):
            methods = [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
            whole = "\n".join(lines[start - 1 : end])
            if methods and len(whole) > MAX_CHUNK_CHARS:
                first_method = _span(methods[0])[0]
                spans.append((f"class {node.name}", start, first_method - 1))
                for method in methods:
                    m_start, m_end = _span(method)
                    spans.append((f"{node.name}.{method.name}", m_start, m_end))
                continue
            spans.append((f"class {node.name}", start, end))
        else:
            spans.append((f"def {node.name}", start, end))

    chunks: list[Chunk] = []

    def gap(first: int, last: int) -> None:
        if last < first:
            return
        body = "\n".join(lines[first - 1 : last]).strip()
        if len(body) >= MIN_CHUNK_CHARS:
            if len(body) > MAX_CHUNK_CHARS:
                chunks.extend(_windows(lines[first - 1 : last], offset=first - 1, label="модуль, строки"))
            else:
                chunks.append(Chunk(f"модуль, строки {first}–{last}", first, last, body))

    cursor = 1
    for heading, start, end in spans:
        gap(cursor, start - 1)
        body = "\n".join(lines[start - 1 : end]).strip()
        if len(body) > MAX_CHUNK_CHARS:
            chunks.extend(_windows(lines[start - 1 : end], offset=start - 1, label=f"{heading}, строки"))
        elif len(body) >= MIN_CHUNK_CHARS:
            chunks.append(Chunk(heading, start, end, body))
        cursor = max(cursor, end + 1)
    gap(cursor, len(lines))
    return chunks


def doc_chunks(text: str) -> list[Chunk]:
    """Разделы README/TASK сплиттером day31; номера строк — по первому вхождению тела."""
    chunks: list[Chunk] = []
    cursor = 0
    for _title, heading, body in _docs.sections(text):
        first_line = body.splitlines()[0] if body else ""
        position = text.find(first_line, cursor) if first_line else -1
        if position >= 0:
            start = text.count("\n", 0, position) + 1
            cursor = position
        else:
            start = 1
        end = start + body.count("\n")
        chunks.append(Chunk(heading, start, end, body))
    return chunks


def chunk_file(path: str, kind: str, text: str) -> list[Chunk]:
    if kind == "docs":
        return doc_chunks(text)
    if path.endswith(".py"):
        return python_chunks(text)
    return _windows(text.splitlines())


def reindex() -> dict[str, int]:
    """Инкрементально по sha: при чистом дереве пересчитывать нечего."""
    files = corpus()
    texts: dict[str, str] = {}
    for name in files:
        try:
            text = (ROOT / name).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) <= MAX_FILE_CHARS:
            texts[name] = text
    digests = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in texts.items()}
    changed = 0

    with _db() as conn:
        known = {row["path"]: row["sha256"] for row in conn.execute("SELECT path, sha256 FROM files")}
        for name, text in texts.items():
            if known.get(name) == digests[name]:
                continue
            parts = chunk_file(name, files[name], text)
            conn.execute("DELETE FROM chunks WHERE path = ?", (name,))
            conn.executemany(
                "INSERT INTO chunks (path, kind, heading, body, line_start, line_end) VALUES (?, ?, ?, ?, ?, ?)",
                [(name, files[name], c.heading, c.body, c.line_start, c.line_end) for c in parts],
            )
            conn.execute(
                "INSERT INTO files (path, kind, sha256, chunks, indexed_at) VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT (path) DO UPDATE SET kind = excluded.kind, sha256 = excluded.sha256,"
                " chunks = excluded.chunks, indexed_at = excluded.indexed_at",
                (name, files[name], digests[name], len(parts), datetime.now(UTC).isoformat()),
            )
            changed += 1
        for name in known.keys() - digests.keys():
            conn.execute("DELETE FROM chunks WHERE path = ?", (name,))
            conn.execute("DELETE FROM files WHERE path = ?", (name,))
        rows = conn.execute("SELECT kind, COUNT(*) AS files, SUM(chunks) AS chunks FROM files GROUP BY kind").fetchall()

    stats = {"files": 0, "chunks": 0, "docs_files": 0, "docs_chunks": 0, "code_files": 0, "code_chunks": 0, "reindexed": changed}
    for row in rows:
        stats["files"] += row["files"]
        stats["chunks"] += row["chunks"] or 0
        stats[f"{row['kind']}_files"] = row["files"]
        stats[f"{row['kind']}_chunks"] = row["chunks"] or 0
    return stats


# --- запрос и поиск ----------------------------------------------------------


def query_tokens(lines: Iterable[str], limit: int = 12) -> list[str]:
    """Идентификаторы из изменённых строк, по частоте: имена функций, модулей,
    полей. Ключевые слова и имена вроде `self` и `value` ничего не найдут —
    они есть в каждом файле."""
    counts: Counter[str] = Counter()
    order: dict[str, int] = {}
    for line in lines:
        for match in IDENTIFIER.finditer(line):
            token = match.group(0).lower()
            if token in STOPWORDS or token.isdigit():
                continue
            counts[token] += 1
            order.setdefault(token, len(order))
    ranked = sorted(counts, key=lambda token: (-counts[token], order[token]))
    return ranked[:limit]


def _match(tokens: list[str]) -> str:
    return " OR ".join(f'"{token}"' for token in tokens)


@dataclass(frozen=True)
class Hit:
    path: str
    kind: str
    heading: str
    line_start: int
    line_end: int
    score: float
    body: str

    @property
    def label(self) -> str:
        if self.kind == "docs":
            return f"{self.path} · «{self.heading}»"
        return f"{self.path}:{self.line_start}–{self.line_end} · {self.heading}"


def search(
    tokens: list[str],
    *,
    kind: str | None = None,
    paths: list[str] | None = None,
    exclude: str | None = None,
    limit: int = 4,
) -> list[Hit]:
    """Чанки по словам запроса: любое слово, порядок bm25 с весом заголовка 3."""
    tokens = [token for token in tokens if token]
    if not tokens:
        return []
    limit = max(1, min(int(limit), MAX_LIMIT))

    clauses: list[str] = []
    params: list[Any] = [_match(tokens)]
    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    if paths:
        clauses.append("(" + " OR ".join(["path LIKE ?"] * len(paths)) + ")")
        params.extend(f"{folder.strip('/')}/%" for folder in paths)
    if exclude:
        clauses.append("path != ?")
        params.append(exclude)
    where = (" AND " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    with _db() as conn:
        rows = conn.execute(
            "SELECT path, kind, heading, body, line_start, line_end,"
            " bm25(chunks, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0) AS rank"
            f" FROM chunks WHERE chunks MATCH ?{where} ORDER BY rank LIMIT ?",
            params,
        ).fetchall()

    return [
        Hit(row["path"], row["kind"], row["heading"], row["line_start"], row["line_end"], round(-row["rank"], 3), row["body"])
        for row in rows
    ]


if __name__ == "__main__":
    stats = reindex()
    print(
        f"Файлов: {stats['files']} (docs {stats['docs_files']}, code {stats['code_files']}), "
        f"чанков: {stats['chunks']} (docs {stats['docs_chunks']}, code {stats['code_chunks']}), "
        f"переиндексировано: {stats['reindexed']}"
    )
    if len(sys.argv) > 1:
        tokens = query_tokens(sys.argv[1:])
        print(f"Запрос: {', '.join(tokens)}")
        for hit in search(tokens, limit=8):
            print(f"  [{hit.kind}] {hit.label} ({hit.score})")
