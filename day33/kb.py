"""База знаний Планёрки: статьи `kb/*.md` в индексе FTS5, поиск без модели.

Статья — markdown-файл с заголовком `#` и разделами `##`; в индекс идёт каждый
раздел отдельно, с названием статьи и заголовком раздела. Разбор и индекс взяты
у [day31/docs.py](../day31/docs.py): там это README и TASK дней проекта, здесь
— FAQ и документация продукта, устройство одно и то же.

Переиндексация инкрементальная, по sha файла, и выполняется при каждом поиске
за миллисекунды: статью можно править на лету, и ассистент увидит правку
следующим вопросом. Индекс `kb.db` не коммитится.
"""

import hashlib
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

KB_DIR = Path(__file__).resolve().parent / "kb"
DB_PATH = Path(__file__).resolve().parent / "kb.db"

FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
WORD = re.compile(r"[^\W_]{2,}", re.UNICODE)

MIN_SECTION_CHARS = 40
MAX_QUERY_TOKENS = 12
SNIPPET_TOKENS = 16
MAX_LIMIT = 10

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    slug       TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    sections   INTEGER NOT NULL,
    indexed_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5 (
    slug UNINDEXED,
    title,
    heading,
    body,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


class KbError(ValueError):
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
    return sorted(path for path in KB_DIR.glob("*.md") if path.is_file())


def sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Название статьи и её разделы: (заголовок раздела, текст). Текст до первого
    `##` — раздел «вступление». Заголовки внутри ``` не считаются."""
    title = ""
    heading = "вступление"
    lines: list[str] = []
    found: list[tuple[str, str]] = []
    fence: str | None = None

    def flush() -> None:
        body = "\n".join(lines).strip()
        if len(body) >= MIN_SECTION_CHARS:
            found.append((heading, body))

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
    return title, found


def reindex() -> dict[str, int]:
    """Инкрементально: статья переиндексируется только если изменился её sha."""
    files = {path.stem: path.read_text(encoding="utf-8") for path in corpus()}
    digests = {slug: hashlib.sha256(text.encode()).hexdigest() for slug, text in files.items()}
    changed = 0

    with _db() as conn:
        known = {row["slug"]: row["sha256"] for row in conn.execute("SELECT slug, sha256 FROM files")}

        for slug, text in files.items():
            if known.get(slug) == digests[slug]:
                continue
            title, parts = sections(text)
            title = title or slug
            conn.execute("DELETE FROM chunks WHERE slug = ?", (slug,))
            conn.executemany(
                "INSERT INTO chunks (slug, title, heading, body) VALUES (?, ?, ?, ?)",
                [(slug, title, heading, body) for heading, body in parts],
            )
            conn.execute(
                "INSERT INTO files (slug, title, sha256, sections, indexed_at) VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT (slug) DO UPDATE SET title = excluded.title, sha256 = excluded.sha256,"
                " sections = excluded.sections, indexed_at = excluded.indexed_at",
                (slug, title, digests[slug], len(parts), datetime.now(UTC).isoformat()),
            )
            changed += 1

        for slug in known.keys() - digests.keys():
            conn.execute("DELETE FROM chunks WHERE slug = ?", (slug,))
            conn.execute("DELETE FROM files WHERE slug = ?", (slug,))

        row = conn.execute("SELECT COUNT(*) AS articles, SUM(sections) AS sections FROM files").fetchone()

    return {"articles": row["articles"], "sections": row["sections"] or 0, "reindexed": changed}


def articles() -> list[dict[str, Any]]:
    reindex()
    with _db() as conn:
        return [dict(row) for row in conn.execute("SELECT slug, title, sections FROM files ORDER BY title")]


# --- поиск -------------------------------------------------------------------


def _match(query: str, strict: bool) -> str:
    """Запрос человека в выражение FTS5. Токены в кавычках: иначе «AND» из текста
    стало бы оператором, а дефис — отрицанием."""
    tokens = WORD.findall(query.lower())[:MAX_QUERY_TOKENS]
    if not tokens:
        raise KbError(f"В запросе {query!r} нет слов для поиска.")
    return (" AND " if strict else " OR ").join(f'"{token}"*' for token in tokens)


def search(query: str, limit: int = 5) -> dict[str, Any]:
    """Разделы по запросу: сначала со всеми словами, потом с любым. Без модели."""
    limit = max(1, min(int(limit), MAX_LIMIT))
    index = reindex()

    rows: list[sqlite3.Row] = []
    seen: set[tuple[str, str]] = set()
    strategy = "любое слово"
    with _db() as conn:
        for strict in (True, False):
            for row in conn.execute(
                "SELECT slug, title, heading, body,"
                " bm25(chunks, 0.0, 2.0, 4.0, 1.0) AS rank,"
                f" snippet(chunks, 3, '', '', '…', {SNIPPET_TOKENS}) AS snippet"
                " FROM chunks WHERE chunks MATCH ? ORDER BY rank LIMIT ?",
                (_match(query, strict), limit),
            ):
                key = (row["slug"], row["heading"])
                if key not in seen and len(rows) < limit:
                    seen.add(key)
                    rows.append(row)
            if strict and rows:
                strategy = "все слова" if len(rows) >= limit else "все слова, затем любое"
            if len(rows) >= limit:
                break

    if not rows:
        raise KbError(
            f"По запросу {query!r} в базе знаний ничего не нашлось. "
            f"Статей в индексе: {index['articles']}. Попробуйте другие слова."
        )

    hits = [
        {
            "number": number,
            "slug": row["slug"],
            "title": row["title"],
            "heading": row["heading"],
            "score": round(-row["rank"], 3),
            "chars": len(row["body"]),
            "snippet": " ".join(row["snippet"].split()),
            "body": row["body"],
        }
        for number, row in enumerate(rows, start=1)
    ]

    return {"query": query, "strategy": strategy, "index": index, "count": len(hits), "hits": hits}


def read(slug: str, heading: str | None = None) -> dict[str, Any]:
    """Раздел статьи по слагу и заголовку из выдачи поиска; без заголовка — статья целиком."""
    reindex()
    with _db() as conn:
        rows = [dict(row) for row in conn.execute(
            "SELECT slug, title, heading, body FROM chunks WHERE slug = ? ORDER BY rowid", (slug,)
        )]
    if not rows:
        known = [item["slug"] for item in articles()]
        raise KbError(f"Статьи {slug!r} в базе знаний нет. Есть: {', '.join(known)}.")

    if heading is None:
        body = "\n\n".join(f"## {row['heading']}\n{row['body']}" for row in rows)
        return {"slug": slug, "title": rows[0]["title"], "heading": None, "chars": len(body), "body": body}

    for row in rows:
        if row["heading"] == heading:
            return {**row, "chars": len(row["body"])}
    names = ", ".join(repr(row["heading"]) for row in rows)
    raise KbError(f"В статье «{rows[0]['title']}» нет раздела {heading!r}. Есть: {names}.")


if __name__ == "__main__":
    import sys

    stats = reindex()
    print(f"Статей: {stats['articles']}, разделов: {stats['sections']}, переиндексировано: {stats['reindexed']}")
    if len(sys.argv) > 1:
        found = search(" ".join(sys.argv[1:]))
        print(f"\nСтратегия: {found['strategy']}")
        for hit in found["hits"]:
            print(f"  [{hit['number']}] {hit['title']} — «{hit['heading']}» ({hit['score']})")
            print(f"      {hit['snippet']}")
