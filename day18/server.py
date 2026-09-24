"""MCP-сервер с планировщиком: снимает активность git по расписанию и сводит её.

В отличие от day17 сервер — отдельный постоянный процесс на streamable HTTP,
а не подпроцесс клиента на один запрос. Фоновый цикл запускается в lifespan:
при HTTP-транспорте SDK входит в него один раз на процесс, а не на сессию,
поэтому цикл один, сколько бы клиентов ни подключалось.

Всё состояние — расписание, снимки, сводки — лежит в SQLite. После
перезапуска цикл читает `next_run_at` из базы и продолжает с того же места;
пропущенные за простой запуски не догоняются, задача выполняется один раз.
"""

import asyncio
import json
import logging
import re
import sqlite3
import subprocess
from collections import Counter
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(__file__).resolve().parent / "scheduler.db"
HOST = "127.0.0.1"
PORT = 8765

TICK_SECONDS = 1
DEFAULT_JOBS = (("collect", 60), ("summary", 600))
MAX_COMMITS_PER_SNAPSHOT = 50
MAX_COMMITS_IN_SUMMARY = 20
FINISHED_JOBS_SHOWN = 5

SEP = "\x1f"
RECORD = "\x1e"
SHORTSTAT = re.compile(r"(\d+) insertion|(\d+) deletion")

logger = logging.getLogger("scheduler")

Kind = Annotated[
    Literal["collect", "summary"],
    Field(
        description=(
            "collect — снимок git: новые коммиты, грязные файлы, объём изменений; "
            "summary — сводка снимков с прошлой сводки."
        )
    ),
]
EveryMinutes = Annotated[
    int | None,
    Field(
        description=(
            "Период в минутах, от 1 до 1440. Периодическая задача того же вида "
            "заменяется, а не дублируется. Без периода задача разовая."
        ),
        ge=1,
        le=1440,
    ),
]
DelayMinutes = Annotated[
    int | None,
    Field(
        description=(
            "Через сколько минут первый запуск, от 0 до 10080. "
            "Без задержки периодическая задача впервые сработает через период."
        ),
        ge=0,
        le=10080,
    ),
]
JobId = Annotated[int, Field(description="Номер задачи из list_jobs.", ge=1)]
Hours = Annotated[
    int, Field(description="Окно сводки в часах, от 1 до 168.", ge=1, le=168)
]
AfterId = Annotated[
    int, Field(description="Вернуть сводки с номером больше этого. 0 — с начала.", ge=0)
]
Limit = Annotated[
    int,
    Field(description="Сколько самых новых сводок вернуть, от 1 до 50.", ge=1, le=50),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            INTEGER PRIMARY KEY,
    kind          TEXT NOT NULL,
    every_seconds INTEGER,
    next_run_at   TEXT NOT NULL,
    last_run_at   TEXT,
    runs          INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT,
    enabled       INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    id          INTEGER PRIMARY KEY,
    taken_at    TEXT NOT NULL,
    head        TEXT NOT NULL,
    new_commits TEXT NOT NULL,
    dirty_files INTEGER NOT NULL,
    untracked   INTEGER NOT NULL,
    insertions  INTEGER NOT NULL,
    deletions   INTEGER NOT NULL,
    folders     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS summaries (
    id          INTEGER PRIMARY KEY,
    created_at  TEXT NOT NULL,
    period_from TEXT,
    period_to   TEXT NOT NULL,
    payload     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS snapshots_taken_at ON snapshots (taken_at);
"""


# --- время -------------------------------------------------------------------
# В базе — UTC: строки ISO с одинаковым смещением сравниваются как строки.
# Наружу — местное время, его читают человек и модель.


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def _local(value: str | None) -> str | None:
    if value is None:
        return None
    return datetime.fromisoformat(value).astimezone().isoformat()


# --- база --------------------------------------------------------------------


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    """Соединение на одну операцию: цикл и инструменты работают из разных потоков."""
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            yield conn


def _init_db() -> None:
    with _db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]:
            return

        now = _now()
        for kind, every in DEFAULT_JOBS:
            # Первый снимок сразу — это точка отсчёта; первая сводка — через период.
            first = now if kind == "collect" else now + timedelta(seconds=every)
            conn.execute(
                "INSERT INTO jobs (kind, every_seconds, next_run_at, created_at)"
                " VALUES (?, ?, ?, ?)",
                (kind, every, _iso(first), _iso(now)),
            )


def _job(row: sqlite3.Row) -> dict[str, Any]:
    every = row["every_seconds"]
    return {
        "id": row["id"],
        "kind": row["kind"],
        "schedule": f"каждые {every // 60} мин" if every else "разово",
        "every_minutes": every // 60 if every else None,
        "active": bool(row["enabled"]),
        "next_run_at": _local(row["next_run_at"]) if row["enabled"] else None,
        "last_run_at": _local(row["last_run_at"]),
        "runs": row["runs"],
        "last_error": row["last_error"],
    }


# --- git ---------------------------------------------------------------------


def _git(*args: str) -> str:
    """Возвращает stdout. Отказ git — это `ToolError`, чтобы модель прочитала причину."""
    result = subprocess.run(
        ["git", "-C", str(ROOT), "--no-pager", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "git вернул ошибку").strip()
        raise ToolError(message.splitlines()[0])

    return result.stdout


def _folder(path: str) -> str:
    head, sep, _ = path.strip('"').partition("/")
    return head if sep else "(корень)"


def _commits_since(previous: str | None, head: str) -> list[dict[str, Any]]:
    """Коммиты между прошлым снимком и текущим HEAD, с папками, которые они трогали."""
    if previous is None or previous == head:
        return []

    try:
        output = _git(
            "log",
            f"-n{MAX_COMMITS_PER_SNAPSHOT}",
            f"--format={RECORD}%H{SEP}%cI{SEP}%an{SEP}%s",
            "--name-only",
            f"{previous}..{head}",
        )
    except ToolError:
        # Прошлый HEAD исчез после rebase или reset: новых коммитов не видно, это не сбой.
        return []

    commits = []
    for record in output.split(RECORD)[1:]:
        header, *files = record.strip("\n").split("\n")
        hash_, date, author, subject = header.split(SEP, 3)
        folders = sorted({_folder(name) for name in files if name})
        commits.append(
            {"hash": hash_, "date": date, "author": author, "subject": subject, "folders": folders}
        )
    return commits


def _collect() -> dict[str, Any]:
    head = _git("rev-parse", "HEAD").strip()

    with _db() as conn:
        row = conn.execute("SELECT head FROM snapshots ORDER BY id DESC LIMIT 1").fetchone()
    commits = _commits_since(row["head"] if row else None, head)

    dirty = []
    untracked = 0
    for line in _git("status", "--porcelain=v1", "--untracked-files=all").splitlines():
        status, path = line[:2], line[3:]
        if status == "??":
            untracked += 1
        dirty.append(path.split(" -> ")[-1])

    shortstat = _git("diff", "--shortstat", "HEAD")
    insertions = deletions = 0
    for added, removed in SHORTSTAT.findall(shortstat):
        insertions += int(added or 0)
        deletions += int(removed or 0)

    snapshot = {
        "taken_at": _iso(_now()),
        "head": head,
        "new_commits": commits,
        "dirty_files": len(dirty),
        "untracked": untracked,
        "insertions": insertions,
        "deletions": deletions,
        "folders": dict(Counter(_folder(path) for path in dirty).most_common()),
    }

    with _db() as conn:
        conn.execute(
            "INSERT INTO snapshots (taken_at, head, new_commits, dirty_files, untracked,"
            " insertions, deletions, folders) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot["taken_at"],
                head,
                json.dumps(commits, ensure_ascii=False),
                snapshot["dirty_files"],
                untracked,
                insertions,
                deletions,
                json.dumps(snapshot["folders"], ensure_ascii=False),
            ),
        )
    return snapshot


# --- сводка ------------------------------------------------------------------


def _tree(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "at": _local(row["taken_at"]),
        "head": row["head"][:7],
        "dirty_files": row["dirty_files"],
        "untracked": row["untracked"],
        "insertions": row["insertions"],
        "deletions": row["deletions"],
    }


def _aggregate(period_from: str | None, period_to: str) -> dict[str, Any]:
    """Сводка снимков из полуинтервала (period_from, period_to]."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM snapshots WHERE taken_at > ? AND taken_at <= ? ORDER BY id",
            (period_from or "", period_to),
        ).fetchall()

    summary: dict[str, Any] = {
        "period": {"from": _local(period_from), "to": _local(period_to)},
        "snapshots": len(rows),
    }
    if not rows:
        summary["note"] = "За период нет ни одного снимка: сбор не работал."
        return summary

    commits: dict[str, dict[str, Any]] = {}
    for row in rows:
        for commit in json.loads(row["new_commits"]):
            commits.setdefault(commit["hash"], commit)

    ordered = sorted(commits.values(), key=lambda commit: commit["date"])
    authors = Counter(commit["author"] for commit in ordered)
    commit_folders = Counter(folder for commit in ordered for folder in commit["folders"])
    first, last = rows[0], rows[-1]

    summary.update(
        {
            "commits": {
                "count": len(ordered),
                "authors": dict(authors.most_common()),
                "folders": dict(commit_folders.most_common()),
                "items": [
                    {
                        "hash": commit["hash"][:7],
                        "date": commit["date"],
                        "author": commit["author"],
                        "subject": commit["subject"],
                    }
                    for commit in ordered[-MAX_COMMITS_IN_SUMMARY:]
                ],
            },
            "working_tree": {
                "first": _tree(first),
                "last": _tree(last),
                "peak_dirty_files": max(row["dirty_files"] for row in rows),
                "dirty_folders_now": json.loads(last["folders"]),
            },
        }
    )
    return summary


def _summarize() -> dict[str, Any]:
    with _db() as conn:
        row = conn.execute(
            "SELECT period_to FROM summaries ORDER BY id DESC LIMIT 1"
        ).fetchone()

    period_from = row["period_to"] if row else None
    period_to = _iso(_now())
    payload = _aggregate(period_from, period_to)

    with _db() as conn:
        cursor = conn.execute(
            "INSERT INTO summaries (created_at, period_from, period_to, payload)"
            " VALUES (?, ?, ?, ?)",
            (period_to, period_from, period_to, json.dumps(payload, ensure_ascii=False)),
        )
    return {"id": cursor.lastrowid, **payload}


# --- планировщик -------------------------------------------------------------

RUNNERS = {"collect": _collect, "summary": _summarize}


def _due_jobs() -> list[sqlite3.Row]:
    with _db() as conn:
        return conn.execute(
            "SELECT * FROM jobs WHERE enabled = 1 AND next_run_at <= ? ORDER BY next_run_at",
            (_iso(_now()),),
        ).fetchall()


def _execute(job: sqlite3.Row) -> None:
    """Сбой задачи записывается в неё же: цикл от одной ошибки не останавливается."""
    error = None
    try:
        result = RUNNERS[job["kind"]]()
        logger.info("job %s %s: ok %s", job["id"], job["kind"], _brief(job["kind"], result))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.warning("job %s %s: %s", job["id"], job["kind"], error)

    now = _now()
    every = job["every_seconds"]
    with _db() as conn:
        conn.execute(
            "UPDATE jobs SET runs = runs + 1, last_run_at = ?, last_error = ?,"
            " next_run_at = ?, enabled = ? WHERE id = ?",
            (
                _iso(now),
                error,
                _iso(now + timedelta(seconds=every or 0)),
                1 if every else 0,
                job["id"],
            ),
        )


def _brief(kind: str, result: dict[str, Any]) -> str:
    if kind == "collect":
        return (
            f"head={result['head'][:7]} new_commits={len(result['new_commits'])}"
            f" dirty={result['dirty_files']}"
        )
    return f"summary #{result['id']}, snapshots={result['snapshots']}"


async def _scheduler() -> None:
    while True:
        for job in await asyncio.to_thread(_due_jobs):
            await asyncio.to_thread(_execute, job)
        await asyncio.sleep(TICK_SECONDS)


@asynccontextmanager
async def lifespan(_: MCPServer) -> AsyncIterator[None]:
    _init_db()
    task = asyncio.create_task(_scheduler())
    logger.info("scheduler started, db %s", DB_PATH)
    try:
        yield
    finally:
        task.cancel()


mcp = MCPServer("AI Advent", version="18.0", lifespan=lifespan)


# --- инструменты -------------------------------------------------------------


@mcp.tool()
def schedule_job(
    kind: Kind,
    every_minutes: EveryMinutes = None,
    delay_minutes: DelayMinutes = None,
) -> dict[str, Any]:
    """Ставит задачу в расписание: периодическую, отложенную разовую или обе сразу."""
    if every_minutes is None and delay_minutes is None:
        raise ToolError("Нужен период every_minutes, задержка delay_minutes или оба.")

    now = _now()
    every = every_minutes * 60 if every_minutes else None
    first = now + timedelta(minutes=delay_minutes if delay_minutes is not None else every_minutes)

    with _db() as conn:
        existing = None
        if every:
            existing = conn.execute(
                "SELECT id FROM jobs WHERE kind = ? AND enabled = 1"
                " AND every_seconds IS NOT NULL ORDER BY id LIMIT 1",
                (kind,),
            ).fetchone()

        if existing:
            conn.execute(
                "UPDATE jobs SET every_seconds = ?, next_run_at = ? WHERE id = ?",
                (every, _iso(first), existing["id"]),
            )
            job_id = existing["id"]
        else:
            job_id = conn.execute(
                "INSERT INTO jobs (kind, every_seconds, next_run_at, created_at)"
                " VALUES (?, ?, ?, ?)",
                (kind, every, _iso(first), _iso(now)),
            ).lastrowid

        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    return {**_job(row), "replaced": existing is not None}


@mcp.tool()
def list_jobs() -> list[dict[str, Any]]:
    """Расписание: активные задачи и последние завершённые — период, запуски, ошибки."""
    with _db() as conn:
        active = conn.execute("SELECT * FROM jobs WHERE enabled = 1 ORDER BY id").fetchall()
        finished = conn.execute(
            "SELECT * FROM jobs WHERE enabled = 0 ORDER BY id DESC LIMIT ?",
            (FINISHED_JOBS_SHOWN,),
        ).fetchall()
    return [_job(row) for row in [*active, *reversed(finished)]]


@mcp.tool()
def cancel_job(job_id: JobId) -> dict[str, Any]:
    """Снимает задачу с расписания. История её запусков остаётся."""
    with _db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise ToolError(f"Задачи {job_id} нет.")
        if not row["enabled"]:
            raise ToolError(f"Задача {job_id} уже не активна.")

        conn.execute("UPDATE jobs SET enabled = 0 WHERE id = ?", (job_id,))
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _job(row)


@mcp.tool()
def get_summary(hours: Hours = 24) -> dict[str, Any]:
    """Сводка собранных снимков за последние N часов: коммиты, авторы, папки, рабочее дерево."""
    now = _now()
    return _aggregate(_iso(now - timedelta(hours=hours)), _iso(now))


@mcp.tool()
def list_summaries(after_id: AfterId = 0, limit: Limit = 10) -> list[dict[str, Any]]:
    """Сохранённые периодические сводки, от старой к новой. Для ленты: after_id — последний виденный."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM summaries WHERE id > ? ORDER BY id DESC LIMIT ?",
            (after_id, limit),
        ).fetchall()
    return [
        {"id": row["id"], "created_at": _local(row["created_at"]), **json.loads(row["payload"])}
        for row in reversed(rows)
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    mcp.run(transport="streamable-http", host=HOST, port=PORT)
