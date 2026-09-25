"""MCP-сервер с композицией инструментов: поиск по документации, сводка, запись файла.

Транспорт снова stdio, как в day16 и day17: расписания здесь нет, сервер нужен
ровно на время запроса. Писать в stdout нельзя ничем, кроме протокола.

Главное решение дня — инструменты передают друг другу не данные, а ссылки на них.
Результат каждого шага ложится в таблицу `artifacts` и возвращается как handle:
номер, превью и метрики. Следующий инструмент принимает `artifact_id` и читает
payload из базы сам, поэтому текст не проходит через контекст модели и она не может
его сократить или переписать. Контракт держат три проверки: вид звена, sha256
payload и цепочка `parent_id`.

`run_pipeline` выполняет всю цепочку сам, одним вызовом, передавая номера в коде.
Те же три инструмента остаются доступны по отдельности — так видно разницу между
автоматической цепочкой и цепочкой, которую собирает модель.
"""

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

import llm

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "index.db"
OUT_DIR = HERE / "out"

DAY_DIR = re.compile(r"^day(\d+)$")
CORPUS_NAMES = ("README.md", "TASK.md")
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
WORD = re.compile(r"[^\W_]{2,}", re.UNICODE)
BULLET = re.compile(r"^\s*[-*•]\s+(.+)$")
REFS = re.compile(r"\[([\d,\s]+)\]\s*$")
SLUG_KEEP = re.compile(r"[^\w-]+", re.UNICODE)

MIN_SECTION_CHARS = 40
MAX_QUERY_TOKENS = 12
SNIPPET_TOKENS = 16
CHARS_PER_HIT = 2000
TOTAL_CHARS = 14000
PREVIEW_CHARS = 400
SLUG_CHARS = 60
SUMMARY_TEMPERATURE = 0.2
SUMMARY_MAX_TOKENS = 700
RUNS_SHOWN = 10

# Вид артефакта — это и есть тип звена: что можно подать на вход следующему шагу.
KINDS = {
    "found": ("результат поиска", "search"),
    "summary": ("сводка", "summarize"),
    "file": ("сохранённый файл", "save_to_file"),
}
PIPELINE = ("search", "summarize", "save_to_file")

SUMMARY_PROMPT = (
    "Ты — инструмент summarize в пайплайне MCP. На входе фрагменты документации "
    "проекта AI Advent, найденные по запросу пользователя. Сведи их в тезисы, "
    "не больше {max_bullets}. Каждый тезис — одна строка, начинается с «- », "
    "в конце в квадратных скобках номера фрагментов, откуда он взят: [1] или [2, 3]. "
    "Пиши по-русски, коротко и по делу, без заголовков и вступлений. "
    "Не добавляй ничего, чего нет во фрагментах."
)

Query = Annotated[
    str,
    Field(
        description=(
            "Запрос на естественном языке, по нему ищутся разделы документации "
            "проекта, например «как считаются токены» или «гейты между этапами»."
        ),
        min_length=2,
    ),
]
Limit = Annotated[
    int,
    Field(description="Сколько разделов взять в работу, от 1 до 10.", ge=1, le=10),
]
ArtifactId = Annotated[
    int,
    Field(description="Номер артефакта, который вернул предыдущий инструмент.", ge=1),
]
MaxBullets = Annotated[
    int, Field(description="Сколько тезисов в сводке, от 3 до 10.", ge=3, le=10)
]
Filename = Annotated[
    str | None,
    Field(
        description=(
            "Имя файла без папок. Приводится к безопасному виду, расширение .md "
            "добавляется само. Без имени оно собирается из запроса."
        )
    ),
]
RunsLimit = Annotated[
    int, Field(description="Сколько последних прогонов вернуть, от 1 до 50.", ge=1, le=50)
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
CREATE TABLE IF NOT EXISTS artifacts (
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL,
    parent_id  INTEGER REFERENCES artifacts (id),
    query      TEXT,
    payload    TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    bytes      INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    query       TEXT NOT NULL,
    mode        TEXT NOT NULL,
    status      TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    error       TEXT
);
CREATE TABLE IF NOT EXISTS steps (
    id          INTEGER PRIMARY KEY,
    run_id      INTEGER NOT NULL REFERENCES runs (id),
    position    INTEGER NOT NULL,
    tool        TEXT NOT NULL,
    artifact_in INTEGER,
    artifact_out INTEGER,
    status      TEXT NOT NULL,
    elapsed_ms  INTEGER NOT NULL,
    error       TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS steps_run ON steps (run_id);
CREATE INDEX IF NOT EXISTS steps_out ON steps (artifact_out);
"""

mcp = MCPServer("AI Advent", version="19.0")


# --- время и база ------------------------------------------------------------
# В базе UTC, наружу местное время: его читают человек и модель.


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _iso() -> str:
    return _now().isoformat()


def _local(value: str | None) -> str | None:
    if value is None:
        return None
    return datetime.fromisoformat(value).astimezone().isoformat()


def _ms(started: float) -> int:
    return round((perf_counter() - started) * 1000)


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
                (name, digests[name], len(sections), _iso()),
            )
            changed += 1

        for name in known.keys() - digests.keys():
            conn.execute("DELETE FROM chunks WHERE path = ?", (name,))
            conn.execute("DELETE FROM files WHERE path = ?", (name,))

        totals = conn.execute("SELECT COUNT(*) AS files, SUM(sections) AS sections FROM files")
        row = totals.fetchone()

    return {"files": row["files"], "sections": row["sections"] or 0, "reindexed": changed}


# --- артефакты ---------------------------------------------------------------


def _canonical(payload: dict[str, Any]) -> str:
    """Одна и та же форма JSON и для хранения, и для хеша: иначе хеш не сойдётся."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _save_artifact(
    conn: sqlite3.Connection,
    kind: str,
    payload: dict[str, Any],
    *,
    parent_id: int | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    text = _canonical(payload)
    digest = _digest(text)
    cursor = conn.execute(
        "INSERT INTO artifacts (kind, parent_id, query, payload, sha256, bytes, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (kind, parent_id, query, text, digest, len(text.encode()), _iso()),
    )
    return {
        "artifact_id": cursor.lastrowid,
        "kind": kind,
        "parent_id": parent_id,
        "sha256": digest,
        "bytes": len(text.encode()),
    }


def _row(conn: sqlite3.Connection, artifact_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
    if row is None:
        last = conn.execute("SELECT MAX(id) AS id FROM artifacts").fetchone()["id"]
        hint = f" Последний созданный артефакт — #{last}." if last else ""
        raise ToolError(f"Артефакта #{artifact_id} нет.{hint}")
    return row


def _check(row: sqlite3.Row) -> dict[str, Any]:
    """Целостность звена: payload в базе должен совпадать со своим хешем."""
    if _digest(row["payload"]) != row["sha256"]:
        raise ToolError(
            f"Артефакт #{row['id']} испорчен: sha256 payload не совпадает с записанным."
        )
    return json.loads(row["payload"])


def _load(
    conn: sqlite3.Connection, artifact_id: int, expected: str, tool: str
) -> tuple[sqlite3.Row, dict[str, Any]]:
    """Вид артефакта — контракт звена. Чужой вид не обрабатывается, а объясняется."""
    row = _row(conn, artifact_id)
    if row["kind"] != expected:
        actual, _ = KINDS[row["kind"]]
        title, producer = KINDS[expected]
        raise ToolError(
            f"Артефакт #{artifact_id} — {actual} (kind={row['kind']}). "
            f"Инструменту {tool} нужен артефакт kind={expected} — {title}, "
            f"его создаёт {producer}."
        )
    return row, _check(row)


def _chain(conn: sqlite3.Connection, artifact_id: int) -> list[dict[str, Any]]:
    """Происхождение артефакта от корня цепочки, с проверкой хеша каждого звена."""
    links: list[dict[str, Any]] = []
    current: int | None = artifact_id

    while current is not None:
        row = _row(conn, current)
        _check(row)
        links.append(
            {
                "artifact_id": row["id"],
                "kind": row["kind"],
                "tool": KINDS[row["kind"]][1],
                "sha256": row["sha256"],
                "created_at": _local(row["created_at"]),
            }
        )
        current = row["parent_id"]

    return list(reversed(links))


# --- прогоны -----------------------------------------------------------------


def _open_run(conn: sqlite3.Connection, query: str, mode: str) -> int:
    cursor = conn.execute(
        "INSERT INTO runs (query, mode, status, started_at) VALUES (?, ?, 'running', ?)",
        (query, mode, _iso()),
    )
    return cursor.lastrowid


def _run_of(conn: sqlite3.Connection, artifact_id: int) -> int | None:
    """Прогон, к которому относится артефакт: тот, где появился корень его цепочки."""
    root = _chain(conn, artifact_id)[0]["artifact_id"]
    row = conn.execute(
        "SELECT run_id FROM steps WHERE artifact_out = ? ORDER BY id LIMIT 1", (root,)
    ).fetchone()
    return row["run_id"] if row else None


def _add_step(
    conn: sqlite3.Connection,
    run_id: int | None,
    tool: str,
    *,
    artifact_in: int | None,
    artifact_out: int | None,
    status: str,
    elapsed_ms: int,
    error: str | None = None,
) -> None:
    if run_id is None:
        return

    position = conn.execute(
        "SELECT COUNT(*) AS taken FROM steps WHERE run_id = ?", (run_id,)
    ).fetchone()["taken"] + 1
    conn.execute(
        "INSERT INTO steps (run_id, position, tool, artifact_in, artifact_out, status,"
        " elapsed_ms, error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, position, tool, artifact_in, artifact_out, status, elapsed_ms, error, _iso()),
    )

    done = tool == PIPELINE[-1] and status == "ok"
    if done or status == "failed":
        conn.execute(
            "UPDATE runs SET status = ?, finished_at = ?, error = ? WHERE id = ?",
            ("done" if done else "failed", _iso(), error, run_id),
        )


def _failed(
    run_id: int | None, tool: str, artifact_in: int | None, error: str, elapsed_ms: int
) -> None:
    with _db() as conn:
        _add_step(
            conn,
            run_id,
            tool,
            artifact_in=artifact_in,
            artifact_out=None,
            status="failed",
            elapsed_ms=elapsed_ms,
            error=error,
        )


def _steps_of(conn: sqlite3.Connection, run_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM steps WHERE run_id = ? ORDER BY position", (run_id,)
    ).fetchall()
    return [
        {
            "position": row["position"],
            "tool": row["tool"],
            "artifact_in": row["artifact_in"],
            "artifact_out": row["artifact_out"],
            "status": row["status"],
            "elapsed_ms": row["elapsed_ms"],
            "error": row["error"],
        }
        for row in rows
    ]


def _run(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    return {
        "run_id": row["id"],
        "query": row["query"],
        "mode": row["mode"],
        "status": row["status"],
        "started_at": _local(row["started_at"]),
        "finished_at": _local(row["finished_at"]),
        "error": row["error"],
        "steps": _steps_of(conn, row["id"]),
    }


# --- шаг 1: поиск ------------------------------------------------------------


def _match(query: str, strict: bool) -> str:
    """Запрос человека в выражение FTS5. Токены берутся в кавычки: иначе «AND» из
    текста стало бы оператором, а дефис — отрицанием."""
    tokens = WORD.findall(query.lower())[:MAX_QUERY_TOKENS]
    if not tokens:
        raise ToolError(f"В запросе {query!r} нет слов для поиска.")
    return (" AND " if strict else " OR ").join(f'"{token}"*' for token in tokens)


def _find(conn: sqlite3.Connection, query: str, limit: int) -> tuple[list[sqlite3.Row], str]:
    """Сначала все слова, потом любое: строгий запрос точнее, но часто пуст."""
    for strict in (True, False):
        rows = conn.execute(
            "SELECT path, title, heading, body,"
            " bm25(chunks, 0.0, 2.0, 4.0, 1.0) AS rank,"
            f" snippet(chunks, 3, '', '', '…', {SNIPPET_TOKENS}) AS snippet"
            " FROM chunks WHERE chunks MATCH ? ORDER BY rank LIMIT ?",
            (_match(query, strict), limit),
        ).fetchall()
        if rows:
            return rows, "все слова" if strict else "любое слово"
    return [], "любое слово"


def _search(query: str, limit: int, run_id: int | None = None) -> dict[str, Any]:
    started = perf_counter()
    index = _reindex()

    # Прогон открывается своим соединением: ToolError ниже откатил бы транзакцию
    # вместе с записью о неудачном шаге, и пустой поиск не попал бы в историю.
    if run_id is None:
        with _db() as conn:
            run_id = _open_run(conn, query, "manual")

    with _db() as conn:
        rows, strategy = _find(conn, query, limit)
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

    if not hits:
        error = (
            f"По запросу {query!r} в документации проекта ничего не нашлось. "
            f"Разделов в индексе: {index['sections']}."
        )
        _failed(run_id, "search", None, error, _ms(started))
        raise ToolError(error)

    with _db() as conn:
        payload = {"query": query, "strategy": strategy, "hits": hits}
        artifact = _save_artifact(conn, "found", payload, query=query)
        _add_step(
            conn,
            run_id,
            "search",
            artifact_in=None,
            artifact_out=artifact["artifact_id"],
            status="ok",
            elapsed_ms=_ms(started),
        )

    return {
        **artifact,
        "run_id": run_id,
        "query": query,
        "strategy": strategy,
        "index": index,
        "total_chars": sum(hit["chars"] for hit in hits),
        # Наружу уходит превью, тексты разделов остаются в артефакте.
        "hits": [
            {key: hit[key] for key in ("number", "path", "heading", "score", "chars", "snippet")}
            for hit in hits
        ],
        "next": "summarize(artifact_id={})".format(artifact["artifact_id"]),
    }


# --- шаг 2: сводка -----------------------------------------------------------


def _fragments(hits: list[dict[str, Any]]) -> str:
    """Фрагменты для модели: целые разделы, а не сниппеты, но с потолком по длине."""
    parts: list[str] = []
    budget = TOTAL_CHARS

    for hit in hits:
        body = hit["body"][: min(CHARS_PER_HIT, budget)]
        if not body:
            break
        budget -= len(body)
        tail = "\n[фрагмент обрезан]" if len(body) < hit["chars"] else ""
        parts.append(f"[{hit['number']}] {hit['path']} — «{hit['heading']}»\n{body}{tail}")

    return "\n\n".join(parts)


def _bullets(text: str, hits: int, max_bullets: int) -> list[dict[str, Any]]:
    """Разбор ответа модели. Ссылки на несуществующие фрагменты отбрасываются."""
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
                    if (number := int(part.strip() or 0)) and 1 <= number <= hits
                }
            )

        if body:
            bullets.append({"text": body, "refs": refs})

    return bullets[:max_bullets]


async def _summarize(
    artifact_id: int, max_bullets: int, run_id: int | None = None
) -> dict[str, Any]:
    started = perf_counter()

    with _db() as conn:
        row, found = _load(conn, artifact_id, "found", "summarize")
        if run_id is None:
            run_id = _run_of(conn, artifact_id)

    hits = found["hits"]
    messages = [
        {"role": "system", "content": SUMMARY_PROMPT.format(max_bullets=max_bullets)},
        {
            "role": "user",
            "content": f"Запрос: {found['query']}\n\n{_fragments(hits)}",
        },
    ]

    try:
        text = await llm.complete(
            messages, temperature=SUMMARY_TEMPERATURE, max_tokens=SUMMARY_MAX_TOKENS
        )
        bullets = _bullets(text, len(hits), max_bullets)
        if not bullets:
            raise RuntimeError("модель не вернула ни одного тезиса")
    except Exception as exc:
        error = f"Сводка не получилась: {exc}"
        _failed(run_id, "summarize", artifact_id, error, _ms(started))
        raise ToolError(error) from exc

    payload = {
        "query": found["query"],
        "model": llm.MODEL,
        "bullets": bullets,
        "text": text,
        "sources": [
            {"number": hit["number"], "path": hit["path"], "heading": hit["heading"]}
            for hit in hits
        ],
    }

    with _db() as conn:
        artifact = _save_artifact(
            conn, "summary", payload, parent_id=row["id"], query=found["query"]
        )
        _add_step(
            conn,
            run_id,
            "summarize",
            artifact_in=artifact_id,
            artifact_out=artifact["artifact_id"],
            status="ok",
            elapsed_ms=_ms(started),
        )

    return {
        **artifact,
        "run_id": run_id,
        "query": found["query"],
        "model": llm.MODEL,
        "bullets": bullets,
        "sources": payload["sources"],
        "next": "save_to_file(artifact_id={})".format(artifact["artifact_id"]),
    }


# --- шаг 3: файл -------------------------------------------------------------


def _slug(raw: str) -> str:
    slug = SLUG_KEEP.sub("-", raw.lower().strip()).strip("-")
    return slug[:SLUG_CHARS].strip("-")


def _target(filename: str | None, query: str) -> Path:
    """Путь внутри `out/`: имя приводится к slug, папки из него выбрасываются."""
    raw = Path(filename).name if filename else query
    slug = _slug(raw.removesuffix(".md")) or "svodka"

    OUT_DIR.mkdir(exist_ok=True)
    path = (OUT_DIR / f"{slug}.md").resolve()
    # Slug не оставляет шанса выйти из папки, но проверка всё равно последняя линия.
    if path.parent != OUT_DIR.resolve():
        raise ToolError(f"Файл {filename!r} оказался бы вне out/.")

    number = 2
    while path.exists():
        path = OUT_DIR / f"{slug}-{number}.md"
        number += 1

    return path


def _markdown(summary: dict[str, Any], chain: list[dict[str, Any]]) -> str:
    lines = [
        f"# {summary['query']}",
        "",
        f"Сводка собрана пайплайном day19 {_local(_iso())}, модель {summary['model']}.",
        "",
        "## Тезисы",
        "",
    ]
    for bullet in summary["bullets"]:
        refs = " " + ", ".join(f"[{ref}]" for ref in bullet["refs"]) if bullet["refs"] else ""
        lines.append(f"- {bullet['text']}{refs}")

    lines += ["", "## Источники", ""]
    for source in summary["sources"]:
        lines.append(f"{source['number']}. `{source['path']}` — «{source['heading']}»")

    lines += ["", "## Цепочка", "", "| Шаг | Артефакт | sha256 payload |", "| --- | --- | --- |"]
    for link in chain:
        lines.append(f"| {link['tool']} | #{link['artifact_id']} | `{link['sha256'][:16]}…` |")

    lines += ["", "Файл записан инструментом `save_to_file`.", ""]
    return "\n".join(lines)


def _save(artifact_id: int, filename: str | None, run_id: int | None = None) -> dict[str, Any]:
    started = perf_counter()

    with _db() as conn:
        row, summary = _load(conn, artifact_id, "summary", "save_to_file")
        # Проверяется не только сводка, но и поиск под ней: цепочка целиком.
        chain = _chain(conn, artifact_id)
        if run_id is None:
            run_id = _run_of(conn, artifact_id)

    try:
        path = _target(filename, summary["query"])
        text = _markdown(summary, chain)
        path.write_text(text, encoding="utf-8")
    except ToolError:
        raise
    except OSError as exc:
        error = f"Файл не записался: {exc}"
        _failed(run_id, "save_to_file", artifact_id, error, _ms(started))
        raise ToolError(error) from exc

    payload = {
        "query": summary["query"],
        "path": str(path.relative_to(ROOT)),
        "filename": path.name,
        "bytes": len(text.encode()),
        "file_sha256": _digest(text),
        "chain": chain,
    }

    with _db() as conn:
        artifact = _save_artifact(
            conn, "file", payload, parent_id=row["id"], query=summary["query"]
        )
        _add_step(
            conn,
            run_id,
            "save_to_file",
            artifact_in=artifact_id,
            artifact_out=artifact["artifact_id"],
            status="ok",
            elapsed_ms=_ms(started),
        )

    return {
        **artifact,
        "run_id": run_id,
        "path": payload["path"],
        "filename": path.name,
        "file_bytes": payload["bytes"],
        "file_sha256": payload["file_sha256"],
        "chain": chain,
    }


# --- инструменты -------------------------------------------------------------


@mcp.tool()
def search(query: Query, limit: Limit = 5) -> dict[str, Any]:
    """Шаг 1: ищет разделы документации проекта по запросу.

    Возвращает не тексты, а артефакт с ними: номер `artifact_id` и превью.
    Этот номер подаётся в summarize — переписывать найденное не нужно.
    """
    return _search(query, limit)


@mcp.tool()
async def summarize(artifact_id: ArtifactId, max_bullets: MaxBullets = 5) -> dict[str, Any]:
    """Шаг 2: сводит найденные разделы в тезисы со ссылками на источники.

    На входе только номер артефакта от search: тексты берутся из него,
    а не из этого вызова. Возвращает артефакт сводки для save_to_file.
    """
    return await _summarize(artifact_id, max_bullets)


@mcp.tool()
def save_to_file(artifact_id: ArtifactId, filename: Filename = None) -> dict[str, Any]:
    """Шаг 3: записывает сводку в markdown-файл в day19/out.

    На входе только номер артефакта от summarize. В файл уходит цепочка
    артефактов с хешами — по нему видно, из какого поиска он вырос.
    """
    return _save(artifact_id, filename)


@mcp.tool()
async def run_pipeline(
    query: Query,
    limit: Limit = 5,
    filename: Filename = None,
    *,
    ctx: Context,
) -> dict[str, Any]:
    """Вся цепочка одним вызовом: search → summarize → save_to_file.

    Номера артефактов передаются между шагами в коде, поэтому данные не
    проходят через модель. Сбой шага останавливает цепочку: в ответе трасса
    с шагом, на котором всё встало, и причиной, а артефакты пройденных шагов
    остаются в базе.
    """
    started = perf_counter()
    with _db() as conn:
        run_id = _open_run(conn, query, "auto")

    trace: list[dict[str, Any]] = []
    artifacts: dict[str, int] = {}
    handles: dict[str, dict[str, Any]] = {}
    status, failed_at, error = "ok", None, None

    async def progress(position: int, frame: dict[str, Any]) -> None:
        # У progress-уведомления нет поля для данных, поэтому кадр едет в message.
        await ctx.report_progress(
            position, len(PIPELINE), json.dumps({"run_id": run_id, **frame}, ensure_ascii=False)
        )

    for position, tool in enumerate(PIPELINE, start=1):
        await progress(position - 1, {"tool": tool, "position": position, "status": "started"})
        step_started = perf_counter()

        try:
            if tool == "search":
                handle = _search(query, limit, run_id)
            elif tool == "summarize":
                handle = await _summarize(artifacts["found"], 5, run_id)
            else:
                handle = _save(artifacts["summary"], filename, run_id)
        except ToolError as exc:
            status, failed_at, error = "failed", tool, str(exc)
            trace.append(
                {
                    "position": position,
                    "tool": tool,
                    "status": "failed",
                    "elapsed_ms": _ms(step_started),
                    "error": error,
                }
            )
            await progress(position, {"tool": tool, "position": position, "status": "failed", "error": error})
            break

        artifacts[handle["kind"]] = handle["artifact_id"]
        handles[handle["kind"]] = handle
        step = {
            "position": position,
            "tool": tool,
            "status": "ok",
            "elapsed_ms": _ms(step_started),
            "artifact_in": handle.get("parent_id"),
            "artifact_out": handle["artifact_id"],
            "sha256": handle["sha256"],
            "handle": handle,
        }
        trace.append(step)
        await progress(position, {**{k: v for k, v in step.items() if k != "handle"}, "handle": handle})

    saved = handles.get("file")
    return {
        "run_id": run_id,
        "query": query,
        "status": status,
        "failed_at": failed_at,
        "error": error,
        "elapsed_ms": _ms(started),
        "artifacts": artifacts,
        "steps": trace,
        "file": (
            {
                "path": saved["path"],
                "bytes": saved["file_bytes"],
                "sha256": saved["file_sha256"],
            }
            if saved
            else None
        ),
    }


@mcp.tool()
def list_runs(limit: RunsLimit = RUNS_SHOWN) -> list[dict[str, Any]]:
    """Последние прогоны пайплайна с их шагами, от нового к старому.

    Режим `auto` — цепочка выполнена run_pipeline, `manual` — инструменты
    вызывались по одному.
    """
    with _db() as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [_run(conn, row) for row in rows]


@mcp.tool()
def get_artifact(artifact_id: ArtifactId) -> dict[str, Any]:
    """Содержимое артефакта с его цепочкой: чем именно шаги обменялись.

    Тексты найденных разделов обрезаются, целиком они нужны только summarize.
    """
    with _db() as conn:
        row = _row(conn, artifact_id)
        payload = _check(row)
        chain = _chain(conn, artifact_id)

    if row["kind"] == "found":
        payload["hits"] = [
            {**hit, "body": hit["body"][:PREVIEW_CHARS], "body_trimmed": hit["chars"] > PREVIEW_CHARS}
            for hit in payload["hits"]
        ]

    return {
        "artifact_id": row["id"],
        "kind": row["kind"],
        "parent_id": row["parent_id"],
        "sha256": row["sha256"],
        "bytes": row["bytes"],
        "created_at": _local(row["created_at"]),
        "chain": chain,
        "payload": payload,
    }


if __name__ == "__main__":
    mcp.run()
