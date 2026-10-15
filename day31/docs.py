"""RAG по документации проекта: разделы README и TASK в индексе FTS5, поиск без модели.

Папки `docs/` в репозитории нет: документация здесь — корневой `README.md` и
пары `README.md` / `TASK.md` в папке каждого дня. Все они идут в корпус, включая
папку самого day31: ассистент должен уметь рассказать и про себя.

Индекс — своя база `day31/docs.db`, а не `day25/index.db`. Индекс day25 собран
под замеры недели RAG: в нём код и PDF, из него выброшены четыре папки, и без
`model2vec` он не собирается. Ассистенту нужна документация целиком и
переиндексация при каждом запуске за миллисекунды — это FTS5 по разделам, как у
сервера docs в [day20](../day20/servers/docs.py), откуда и взят разбор на разделы.

Разделы режутся по заголовкам `#`–`###`, заголовки внутри ``` не считаются:
в README этого проекта есть блоки с markdown внутри. Переиндексация
инкрементальная — по sha файла, так что при запуске пересчитываются только
изменённые файлы.
"""

import hashlib
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(__file__).resolve().parent / "docs.db"

DAY_DIR = re.compile(r"^day(\d+)$")
CORPUS_NAMES = ("README.md", "TASK.md")
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
WORD = re.compile(r"[^\W_]{2,}", re.UNICODE)
FOLDER = re.compile(r"^[\w.-]+$", re.UNICODE)

MIN_SECTION_CHARS = 40
MAX_QUERY_TOKENS = 12
SNIPPET_TOKENS = 16
MAX_LIMIT = 10

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path       TEXT PRIMARY KEY,
    sha256     TEXT NOT NULL,
    sections   INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5 (
    path UNINDEXED,
    title,
    heading,
    body,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


class DocsError(ValueError):
    """Отказ поиска с причиной, которую можно показать модели и человеку."""


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            conn.executescript(SCHEMA)
            yield conn


# --- корпус ------------------------------------------------------------------


def corpus() -> list[Path]:
    """Корневой README и README/TASK каждого дня по порядку номеров."""
    days = sorted(
        (path for path in ROOT.iterdir() if path.is_dir() and DAY_DIR.match(path.name)),
        key=lambda path: int(DAY_DIR.match(path.name).group(1)),
    )
    paths = [ROOT / "README.md"]
    paths.extend(day / name for day in days for name in CORPUS_NAMES)
    return [path for path in paths if path.is_file()]


def sections(text: str) -> list[tuple[str, str, str]]:
    """Разделы документа: (заголовок документа, заголовок раздела, текст)."""
    title = ""
    heading = "вступление"
    lines: list[str] = []
    found: list[tuple[str, str, str]] = []
    fence: str | None = None

    def flush() -> None:
        body = "\n".join(lines).strip()
        if len(body) >= MIN_SECTION_CHARS:
            found.append((title, heading, body))

    for line in text.splitlines():
        opening = FENCE.match(line)
        if opening:
            mark = opening.group(1)
            fence = None if fence == mark else (fence or mark)
            lines.append(line)
            continue

        match = None if fence else HEADING.match(line)
        if match is None:
            lines.append(line)
            continue

        level, name = len(match.group(1)), match.group(2).strip()
        if level == 1 and not title:
            title = name
            continue

        flush()
        heading, lines = name, []

    flush()
    return found


def reindex() -> dict[str, int]:
    """Инкрементально: файл переиндексируется только если изменился его sha."""
    files = {path: path.read_text(encoding="utf-8") for path in corpus()}
    digests = {
        str(path.relative_to(ROOT)): hashlib.sha256(text.encode()).hexdigest()
        for path, text in files.items()
    }
    changed = 0

    with _db() as conn:
        known = {row["path"]: row["sha256"] for row in conn.execute("SELECT path, sha256 FROM files")}

        for path, text in files.items():
            name = str(path.relative_to(ROOT))
            if known.get(name) == digests[name]:
                continue

            parts = sections(text)
            conn.execute("DELETE FROM chunks WHERE path = ?", (name,))
            conn.executemany(
                "INSERT INTO chunks (path, title, heading, body) VALUES (?, ?, ?, ?)",
                [(name, title, heading, body) for title, heading, body in parts],
            )
            conn.execute(
                "INSERT INTO files (path, sha256, sections, indexed_at) VALUES (?, ?, ?, ?)"
                " ON CONFLICT (path) DO UPDATE SET sha256 = excluded.sha256,"
                " sections = excluded.sections, indexed_at = excluded.indexed_at",
                (name, digests[name], len(parts), datetime.now(UTC).isoformat()),
            )
            changed += 1

        for name in known.keys() - digests.keys():
            conn.execute("DELETE FROM chunks WHERE path = ?", (name,))
            conn.execute("DELETE FROM files WHERE path = ?", (name,))

        row = conn.execute("SELECT COUNT(*) AS files, SUM(sections) AS sections FROM files").fetchone()

    return {"files": row["files"], "sections": row["sections"] or 0, "reindexed": changed}


# --- поиск -------------------------------------------------------------------


def _match(query: str, strict: bool) -> str:
    """Запрос человека в выражение FTS5. Токены в кавычках: иначе «AND» из текста
    стало бы оператором, а дефис — отрицанием."""
    tokens = WORD.findall(query.lower())[:MAX_QUERY_TOKENS]
    if not tokens:
        raise DocsError(f"В запросе {query!r} нет слов для поиска.")
    return (" AND " if strict else " OR ").join(f'"{token}"*' for token in tokens)


def _filter(paths: list[str] | None) -> tuple[str, list[str]]:
    """Фильтр по папкам: только имена папок, никаких шаблонов в LIKE."""
    if not paths:
        return "", []

    folders = []
    for value in paths:
        folder = value.strip().strip("/")
        if not folder or not FOLDER.match(folder):
            raise DocsError(f"Непонятная папка: {value!r}. Ожидается имя вида «day19».")
        folders.append(folder)

    clause = " AND (" + " OR ".join(["path LIKE ?"] * len(folders)) + ")"
    return clause, [f"{folder}/%" for folder in folders]


def search(query: str, paths: list[str] | None = None, limit: int = 5) -> dict[str, Any]:
    """Разделы по запросу: сначала со всеми словами, потом с любым. Без модели."""
    limit = max(1, min(int(limit), MAX_LIMIT))
    index = reindex()
    clause, params = _filter(paths)

    # Сначала разделы со всеми словами запроса, потом выдача добирается до
    # лимита разделами с любым из слов: строгий поиск точнее, но часто короче.
    rows: list[sqlite3.Row] = []
    seen: set[tuple[str, str]] = set()
    strategy = "любое слово"
    with _db() as conn:
        for strict in (True, False):
            for row in conn.execute(
                "SELECT path, title, heading, body,"
                " bm25(chunks, 0.0, 2.0, 4.0, 1.0) AS rank,"
                f" snippet(chunks, 3, '', '', '…', {SNIPPET_TOKENS}) AS snippet"
                f" FROM chunks WHERE chunks MATCH ?{clause} ORDER BY rank LIMIT ?",
                (_match(query, strict), *params, limit),
            ):
                key = (row["path"], row["heading"])
                if key not in seen and len(rows) < limit:
                    seen.add(key)
                    rows.append(row)
            if strict and rows:
                strategy = "все слова" if len(rows) >= limit else "все слова, затем любое"
            if len(rows) >= limit:
                break

    if not rows:
        where = f" в папках {', '.join(paths)}" if paths else ""
        raise DocsError(
            f"По запросу {query!r}{where} в документации ничего не нашлось. "
            f"Разделов в индексе: {index['sections']}."
        )

    hits = [
        {
            "number": number,
            "path": row["path"],
            "title": row["title"],
            "heading": row["heading"],
            "score": round(-row["rank"], 3),
            "chars": len(row["body"]),
            "snippet": " ".join(row["snippet"].split()),
            "body": row["body"],
        }
        for number, row in enumerate(rows, start=1)
    ]

    return {
        "query": query,
        "strategy": strategy,
        "filter": paths or None,
        "index": index,
        "count": len(hits),
        "hits": hits,
    }


def read(path: str, heading: str) -> dict[str, Any]:
    """Один раздел целиком — по пути и заголовку из выдачи поиска."""
    reindex()
    with _db() as conn:
        row = conn.execute(
            "SELECT path, title, heading, body FROM chunks WHERE path = ? AND heading = ?",
            (path, heading),
        ).fetchone()
        if row is None:
            known = [item["heading"] for item in conn.execute("SELECT heading FROM chunks WHERE path = ?", (path,))]

    if row is None:
        if not known:
            raise DocsError(f"Файла {path!r} в индексе документации нет.")
        raise DocsError(f"В {path} нет раздела {heading!r}. Есть: {', '.join(repr(name) for name in known[:8])}.")

    return {
        "path": row["path"],
        "title": row["title"],
        "heading": row["heading"],
        "chars": len(row["body"]),
        "body": row["body"],
    }


if __name__ == "__main__":
    import sys

    stats = reindex()
    print(f"Файлов: {stats['files']}, разделов: {stats['sections']}, переиндексировано: {stats['reindexed']}")
    if len(sys.argv) > 1:
        found = search(" ".join(sys.argv[1:]))
        print(f"\nСтратегия: {found['strategy']}")
        for hit in found["hits"]:
            print(f"  [{hit['number']}] {hit['path']} — «{hit['heading']}» ({hit['score']})")
            print(f"      {hit['snippet']}")
