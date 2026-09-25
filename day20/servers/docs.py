"""MCP-сервер над документацией проекта: поиск по разделам, чтение раздела, сводка.

Второй сервер реестра day20, тоже stdio. Корпус и индекс FTS5 — как в day19,
но база своя (`day20/docs.db`) и общей с остальными серверами у неё нет: это
и есть причина, по которой артефакты в day20 держит реестр, а не сервер.

Два отличия от day19. Первое — у поиска появился фильтр `paths`: сервер git
говорит, какие папки трогали коммиты, и поиск идёт только по ним. Второе —
`summarize` принимает разделы аргументом, а не номером артефакта: про артефакты
знает реестр, а сервер остаётся самостоятельным.
"""

import hashlib
import re
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

# Сервер лежит в day20/servers, а llm.py — в day20: sys.path[0] у скрипта — его папка.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import llm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = Path(__file__).resolve().parent.parent / "docs.db"

DAY_DIR = re.compile(r"^day(\d+)$")
CORPUS_NAMES = ("README.md", "TASK.md")
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
WORD = re.compile(r"[^\W_]{2,}", re.UNICODE)
BULLET = re.compile(r"^\s*[-*•]\s+(.+)$")
REFS = re.compile(r"\[([\d,\s]+)\]\s*$")
FOLDER = re.compile(r"^[\w.-]+$", re.UNICODE)

MIN_SECTION_CHARS = 40
MAX_QUERY_TOKENS = 12
SNIPPET_TOKENS = 16
CHARS_PER_SECTION = 2000
TOTAL_CHARS = 14000
SUMMARY_TEMPERATURE = 0.2
SUMMARY_MAX_TOKENS = 700

SUMMARY_PROMPT = (
    "Ты — инструмент summarize сервера docs. На входе разделы документации проекта "
    "AI Advent, найденные по запросу, и иногда темы коммитов из репозитория. "
    "Сведи их в тезисы, не больше {max_bullets}. Каждый тезис — одна строка, "
    "начинается с «- », в конце в квадратных скобках номера разделов, откуда он "
    "взят: [1] или [2, 3]. Пиши по-русски, коротко и по делу, без заголовков и "
    "вступлений. Не добавляй ничего, чего нет во входных данных."
)

Query = Annotated[
    str,
    Field(
        description=(
            "Запрос на естественном языке, по нему ищутся разделы документации "
            "проекта, например «как считаются токены» или «гейты между этапами». "
            "Это поиск по текстам, а не по истории коммитов."
        ),
        min_length=2,
    ),
]
Paths = Annotated[
    list[str] | None,
    Field(
        description=(
            "Папки, которыми ограничить поиск, например [\"day19\", \"day18\"]. "
            "Без них ищется по всей документации."
        )
    ),
]
Limit = Annotated[
    int,
    Field(description="Сколько разделов вернуть, от 1 до 10.", ge=1, le=10),
]
Path_ = Annotated[
    str,
    Field(description="Путь файла из выдачи поиска, например «day19/README.md»."),
]
Heading = Annotated[
    str, Field(description="Заголовок раздела из выдачи поиска, как он там написан.")
]
MaxBullets = Annotated[
    int, Field(description="Сколько тезисов в сводке, от 3 до 10.", ge=3, le=10)
]
Context = Annotated[
    list[str] | None,
    Field(description="Дополнительные строки для сводки, например темы коммитов."),
]


class Section(BaseModel):
    """Раздел документации для сводки. Лишние поля выдачи поиска игнорируются."""

    number: int
    path: str
    heading: str
    body: str


Sections = Annotated[
    list[Section],
    Field(description="Разделы, найденные search: их текст и есть материал сводки.", min_length=1),
]

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

# Как и у git: логи stdio-сервера уходят в stderr клиента, а на INFO туда попадают
# и запросы к модели из `summarize`, и ожидаемые отказы инструментов.
mcp = MCPServer("AI Advent docs", version="20.0", log_level="WARNING")


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            conn.executescript(SCHEMA)
            yield conn


# --- корпус ------------------------------------------------------------------


def _corpus() -> list[Path]:
    days = sorted(
        (path for path in ROOT.iterdir() if path.is_dir() and DAY_DIR.match(path.name)),
        key=lambda path: int(DAY_DIR.match(path.name).group(1)),
    )
    paths = [ROOT / "README.md"]
    paths.extend(day / name for day in days for name in CORPUS_NAMES)
    return [path for path in paths if path.is_file()]


def _sections(text: str) -> list[tuple[str, str, str]]:
    """Разделы документа: заголовок и текст под ним.

    Заголовки внутри ``` не считаются заголовками: в README этого проекта
    есть блоки с markdown внутри, и без учёта заборов они рвали бы разделы.
    """
    title = ""
    heading = "вступление"
    lines: list[str] = []
    sections: list[tuple[str, str, str]] = []
    fence: str | None = None

    def flush() -> None:
        body = "\n".join(lines).strip()
        if len(body) >= MIN_SECTION_CHARS:
            sections.append((title, heading, body))

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
    return sections


def _reindex() -> dict[str, int]:
    """Инкрементально: файл переиндексируется только если изменился его sha."""
    files = {path: path.read_text(encoding="utf-8") for path in _corpus()}
    digests = {
        str(path.relative_to(ROOT)): hashlib.sha256(text.encode()).hexdigest()
        for path, text in files.items()
    }
    changed = 0

    with _db() as conn:
        known = {
            row["path"]: row["sha256"] for row in conn.execute("SELECT path, sha256 FROM files")
        }

        for path, text in files.items():
            name = str(path.relative_to(ROOT))
            if known.get(name) == digests[name]:
                continue

            sections = _sections(text)
            conn.execute("DELETE FROM chunks WHERE path = ?", (name,))
            conn.executemany(
                "INSERT INTO chunks (path, title, heading, body) VALUES (?, ?, ?, ?)",
                [(name, title, heading, body) for title, heading, body in sections],
            )
            conn.execute(
                "INSERT INTO files (path, sha256, sections, indexed_at) VALUES (?, ?, ?, ?)"
                " ON CONFLICT (path) DO UPDATE SET sha256 = excluded.sha256,"
                " sections = excluded.sections, indexed_at = excluded.indexed_at",
                (name, digests[name], len(sections), datetime.now(UTC).isoformat()),
            )
            changed += 1

        for name in known.keys() - digests.keys():
            conn.execute("DELETE FROM chunks WHERE path = ?", (name,))
            conn.execute("DELETE FROM files WHERE path = ?", (name,))

        row = conn.execute(
            "SELECT COUNT(*) AS files, SUM(sections) AS sections FROM files"
        ).fetchone()

    return {"files": row["files"], "sections": row["sections"] or 0, "reindexed": changed}


# --- поиск -------------------------------------------------------------------


def _match(query: str, strict: bool) -> str:
    """Запрос человека в выражение FTS5. Токены берутся в кавычки: иначе «AND» из
    текста стало бы оператором, а дефис — отрицанием."""
    tokens = WORD.findall(query.lower())[:MAX_QUERY_TOKENS]
    if not tokens:
        raise ToolError(f"В запросе {query!r} нет слов для поиска.")
    return (" AND " if strict else " OR ").join(f'"{token}"*' for token in tokens)


def _filter(paths: list[str] | None) -> tuple[str, list[str]]:
    """Фильтр по папкам: только имена папок, никаких шаблонов от модели в LIKE."""
    if not paths:
        return "", []

    folders = []
    for value in paths:
        folder = value.strip().strip("/")
        if not folder or not FOLDER.match(folder):
            raise ToolError(f"Непонятная папка: {value!r}. Ожидается имя вида «day19».")
        folders.append(folder)

    clause = " AND (" + " OR ".join(["path LIKE ?"] * len(folders)) + ")"
    return clause, [f"{folder}/%" for folder in folders]


def _find(
    conn: sqlite3.Connection, query: str, paths: list[str] | None, limit: int
) -> tuple[list[sqlite3.Row], str]:
    clause, params = _filter(paths)

    for strict in (True, False):
        rows = conn.execute(
            "SELECT path, title, heading, body,"
            " bm25(chunks, 0.0, 2.0, 4.0, 1.0) AS rank,"
            f" snippet(chunks, 3, '', '', '…', {SNIPPET_TOKENS}) AS snippet"
            f" FROM chunks WHERE chunks MATCH ?{clause} ORDER BY rank LIMIT ?",
            (_match(query, strict), *params, limit),
        ).fetchall()
        if rows:
            return rows, "все слова" if strict else "любое слово"

    return [], "любое слово"


@mcp.tool()
def search(query: Query, paths: Paths = None, limit: Limit = 5) -> dict[str, Any]:
    """Ищет разделы документации проекта по запросу, с фильтром по папкам.

    Возвращает разделы с их текстом — это материал для summarize.
    Если нужны коммиты, а не документация, инструмент с тем же именем есть у git.
    """
    index = _reindex()

    with _db() as conn:
        rows, strategy = _find(conn, query, paths, limit)

    if not rows:
        where = f" в папках {', '.join(paths)}" if paths else ""
        raise ToolError(
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
        "chars": sum(hit["chars"] for hit in hits),
        "hits": hits,
    }


@mcp.tool()
def read_section(path: Path_, heading: Heading) -> dict[str, Any]:
    """Один раздел документации целиком — по пути и заголовку из выдачи поиска."""
    with _db() as conn:
        row = conn.execute(
            "SELECT path, title, heading, body FROM chunks WHERE path = ? AND heading = ?",
            (path, heading),
        ).fetchone()

    if row is None:
        with _db() as conn:
            known = [
                item["heading"]
                for item in conn.execute("SELECT heading FROM chunks WHERE path = ?", (path,))
            ]
        if not known:
            raise ToolError(f"Файла {path!r} в индексе документации нет.")
        raise ToolError(
            f"В {path} нет раздела {heading!r}. Есть: {', '.join(repr(name) for name in known[:8])}."
        )

    return {
        "path": row["path"],
        "title": row["title"],
        "heading": row["heading"],
        "chars": len(row["body"]),
        "body": row["body"],
    }


# --- сводка ------------------------------------------------------------------


def _fragments(sections: list[Section]) -> str:
    """Разделы для модели: целиком, но с потолком по длине."""
    parts: list[str] = []
    budget = TOTAL_CHARS

    for section in sections:
        body = section.body[: min(CHARS_PER_SECTION, budget)]
        if not body:
            break
        budget -= len(body)
        tail = "\n[фрагмент обрезан]" if len(body) < len(section.body) else ""
        parts.append(f"[{section.number}] {section.path} — «{section.heading}»\n{body}{tail}")

    return "\n\n".join(parts)


def _bullets(text: str, sections: int, max_bullets: int) -> list[dict[str, Any]]:
    """Разбор ответа модели. Ссылки на несуществующие разделы отбрасываются."""
    bullets: list[dict[str, Any]] = []

    for line in text.splitlines():
        match = BULLET.match(line)
        if match is None:
            continue

        body = match.group(1).strip()
        refs: list[int] = []
        found = REFS.search(body)
        if found:
            body = body[: found.start()].strip()
            refs = sorted(
                {
                    number
                    for part in found.group(1).split(",")
                    if (number := int(part.strip() or 0)) and 1 <= number <= sections
                }
            )

        if body:
            bullets.append({"text": body, "refs": refs})

    return bullets[:max_bullets]


def _markdown(
    query: str,
    bullets: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    context: list[str] | None,
) -> str:
    """Готовый текст отчёта: сервер vault записывает его как есть, ничего не достраивая."""
    lines = [
        f"# {query}",
        "",
        f"Сводка собрана реестром day20 {datetime.now().astimezone().isoformat(timespec='seconds')},"
        f" модель {llm.MODEL}.",
        "",
        "## Тезисы",
        "",
    ]
    for bullet in bullets:
        refs = " " + ", ".join(f"[{ref}]" for ref in bullet["refs"]) if bullet["refs"] else ""
        lines.append(f"- {bullet['text']}{refs}")

    lines += ["", "## Разделы документации", ""]
    for source in sources:
        lines.append(f"{source['number']}. `{source['path']}` — «{source['heading']}»")

    if context:
        lines += ["", "## Коммиты", ""]
        lines.extend(f"- {item}" for item in context)

    lines += ["", "Отчёт записан инструментом `vault__save_file`.", ""]
    return "\n".join(lines)


@mcp.tool()
async def summarize(
    query: Query,
    sections: Sections,
    context: Context = None,
    max_bullets: MaxBullets = 5,
) -> dict[str, Any]:
    """Сводит разделы в тезисы со ссылками на источники и собирает текст отчёта.

    Разделы приходят от search целиком: пересказывать их в аргументы не нужно.
    Возвращает готовый markdown — его записывает save_to_file сервера vault.
    """
    task = f"Запрос: {query}\n\n{_fragments(sections)}"
    if context:
        task += "\n\nТемы коммитов:\n" + "\n".join(f"- {item}" for item in context)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SUMMARY_PROMPT.format(max_bullets=max_bullets)},
        {"role": "user", "content": task},
    ]

    try:
        text = await llm.complete(
            messages, temperature=SUMMARY_TEMPERATURE, max_tokens=SUMMARY_MAX_TOKENS
        )
    except Exception as exc:
        raise ToolError(f"Сводка не получилась: {exc}") from exc

    bullets = _bullets(text, len(sections), max_bullets)
    if not bullets:
        raise ToolError("Сводка не получилась: модель не вернула ни одного тезиса.")

    sources = [
        {"number": section.number, "path": section.path, "heading": section.heading}
        for section in sections
    ]

    return {
        "query": query,
        "model": llm.MODEL,
        "bullets": bullets,
        "sources": sources,
        "markdown": _markdown(query, bullets, sources, context),
    }


if __name__ == "__main__":
    mcp.run()
