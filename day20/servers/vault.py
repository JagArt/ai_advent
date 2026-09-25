"""MCP-сервер хранилища: файлы отчётов в day20/out и журнал в day20/vault.db.

Третий сервер реестра day20 и единственный постоянный: у него есть состояние,
которое должно переживать запросы, — журнал записанных отчётов. Поэтому транспорт
streamable HTTP и отдельный процесс, как в day18, а не подпроцесс на вызов.

Сервер ничего не знает ни про поиск, ни про артефакты реестра: ему дают имя и
готовый текст, он пишет файл и отдаёт его хеш. `verify` сверяет хеш с тем, что
лежит на диске, — этим замыкается длинный флоу.
"""

import hashlib
import logging
import re
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
DB_PATH = HERE / "vault.db"
OUT_DIR = HERE / "out"
HOST = "127.0.0.1"
PORT = 8770

SLUG_KEEP = re.compile(r"[^\w-]+", re.UNICODE)
SLUG_CHARS = 60
MAX_BODY_CHARS = 60000
PREVIEW_CHARS = 600

logger = logging.getLogger("vault")

Name = Annotated[
    str,
    Field(
        description=(
            "Имя файла без папок. Приводится к безопасному виду, расширение .md "
            "добавляется само; существующий файл не перезаписывается."
        ),
        min_length=1,
    ),
]
Body = Annotated[
    str,
    Field(
        description="Текст отчёта целиком — тот, что собрал summarize.",
        min_length=1,
        max_length=MAX_BODY_CHARS,
    ),
]
Filename = Annotated[
    str, Field(description="Имя файла из out, как его вернул save_file.", min_length=1)
]
Note = Annotated[
    str | None, Field(description="Одна строка от агента: чем этот отчёт был вызван.")
]
Limit = Annotated[
    int, Field(description="Сколько последних записей журнала вернуть, от 1 до 50.", ge=1, le=50)
]


class Entry(BaseModel):
    """Запись о файле — её отдаёт save_file и принимает journal_append."""

    path: str
    filename: str
    bytes: int
    sha256: str
    saved_at: str


class Check(BaseModel):
    """Что проверять: путь файла и ожидаемый хеш его содержимого."""

    path: str
    sha256: str


SCHEMA = """
CREATE TABLE IF NOT EXISTS journal (
    id         INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    filename   TEXT NOT NULL,
    path       TEXT NOT NULL,
    bytes      INTEGER NOT NULL,
    sha256     TEXT NOT NULL,
    note       TEXT
);
"""

mcp = MCPServer("AI Advent vault", version="20.0")


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _local(value: str) -> str:
    return datetime.fromisoformat(value).astimezone().isoformat()


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            conn.executescript(SCHEMA)
            yield conn


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _inside(filename: str) -> Path:
    """Путь внутри out/: имя берётся без папок, а результат проверяется после resolve."""
    path = (OUT_DIR / Path(filename).name).resolve()
    if path.parent != OUT_DIR.resolve():
        raise ToolError(f"Файл {filename!r} оказался бы вне out/.")
    return path


def _target(name: str) -> Path:
    slug = SLUG_KEEP.sub("-", name.removesuffix(".md").lower().strip()).strip("-")
    slug = slug[:SLUG_CHARS].strip("-") or "otchet"

    OUT_DIR.mkdir(exist_ok=True)
    path = _inside(f"{slug}.md")

    number = 2
    while path.exists():
        path = OUT_DIR / f"{slug}-{number}.md"
        number += 1

    return path


@mcp.tool()
def save_file(name: Name, body: Body) -> dict[str, Any]:
    """Записывает текст файлом в day20/out и отдаёт его размер и sha256.

    Текст пишется как есть: сервер ничего к нему не добавляет. В ответе есть
    готовая запись `entry` для journal_append и `check` для verify.
    """
    path = _target(name)
    try:
        path.write_text(body, encoding="utf-8")
    except OSError as exc:
        raise ToolError(f"Файл не записался: {exc}") from exc

    digest = _digest(body)
    entry = Entry(
        path=str(path.relative_to(ROOT)),
        filename=path.name,
        bytes=len(body.encode()),
        sha256=digest,
        saved_at=_now(),
    )
    logger.info("saved %s, %s байт", entry.filename, entry.bytes)

    return {
        **entry.model_dump(),
        "entry": entry.model_dump(),
        "check": {"path": entry.path, "sha256": digest},
        "preview": body[:PREVIEW_CHARS],
    }


@mcp.tool()
def read_file(filename: Filename) -> dict[str, Any]:
    """Читает файл из day20/out целиком, с его размером и sha256."""
    path = _inside(filename)
    if not path.is_file():
        known = sorted(item.name for item in OUT_DIR.glob("*.md")) if OUT_DIR.is_dir() else []
        hint = f" Есть: {', '.join(known[:8])}." if known else " Папка out пуста."
        raise ToolError(f"Файла {filename!r} в out нет.{hint}")

    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path.relative_to(ROOT)),
        "filename": path.name,
        "bytes": len(text.encode()),
        "sha256": _digest(text),
        "text": text,
    }


@mcp.tool()
def list_files() -> dict[str, Any]:
    """Что уже записано в day20/out: имена, размеры и время изменения."""
    if not OUT_DIR.is_dir():
        return {"count": 0, "files": []}

    files = [
        {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),
        }
        for path in sorted(OUT_DIR.glob("*.md"), key=lambda item: -item.stat().st_mtime)
    ]
    return {"count": len(files), "files": files}


@mcp.tool()
def journal_append(entry: Entry, note: Note = None) -> dict[str, Any]:
    """Заносит записанный файл в журнал хранилища — он переживает перезапуск.

    На входе запись `entry`, которую вернул save_file: путь, размер и хеш.
    """
    if not _inside(entry.filename).is_file():
        raise ToolError(f"Файла {entry.filename!r} в out нет: в журнал заносить нечего.")

    with _db() as conn:
        cursor = conn.execute(
            "INSERT INTO journal (created_at, filename, path, bytes, sha256, note)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (_now(), entry.filename, entry.path, entry.bytes, entry.sha256, note),
        )
        total = conn.execute("SELECT COUNT(*) AS count FROM journal").fetchone()["count"]

    logger.info("journal #%s %s", cursor.lastrowid, entry.filename)
    return {
        "id": cursor.lastrowid,
        "created_at": _local(_now()),
        "filename": entry.filename,
        "path": entry.path,
        "bytes": entry.bytes,
        "sha256": entry.sha256,
        "note": note,
        "total": total,
    }


@mcp.tool()
def journal_list(limit: Limit = 10) -> dict[str, Any]:
    """Последние записи журнала хранилища, от новой к старой."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM journal ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    return {
        "count": len(rows),
        "entries": [
            {
                "id": row["id"],
                "created_at": _local(row["created_at"]),
                "filename": row["filename"],
                "path": row["path"],
                "bytes": row["bytes"],
                "sha256": row["sha256"],
                "note": row["note"],
            }
            for row in rows
        ],
    }


@mcp.tool()
def verify(expect: Check) -> dict[str, Any]:
    """Сверяет sha256 файла на диске с ожидаемым — последний шаг длинного флоу.

    На входе `check` из ответа save_file. Несовпадение — не отказ инструмента,
    а результат: в ответе `ok: false` и оба хеша.
    """
    path = _inside(Path(expect.path).name)
    if not path.is_file():
        raise ToolError(f"Файла {expect.path!r} на диске нет: сверять нечего.")

    text = path.read_text(encoding="utf-8")
    actual = _digest(text)
    in_journal = False
    with _db() as conn:
        in_journal = bool(
            conn.execute(
                "SELECT 1 FROM journal WHERE filename = ? AND sha256 = ? LIMIT 1",
                (path.name, actual),
            ).fetchone()
        )

    return {
        "path": str(path.relative_to(ROOT)),
        "filename": path.name,
        "ok": actual == expect.sha256,
        "expected_sha256": expect.sha256,
        "sha256_on_disk": actual,
        "bytes": len(text.encode()),
        "in_journal": in_journal,
    }


if __name__ == "__main__":
    # Логи настраивает сам MCPServer, и здесь они кстати: у постоянного сервера
    # свой терминал, видно каждый запрос и каждую запись в хранилище.
    OUT_DIR.mkdir(exist_ok=True)
    with _db():
        pass
    logger.info("vault ready, out %s, db %s", OUT_DIR, DB_PATH)
    mcp.run(transport="streamable-http", host=HOST, port=PORT)
