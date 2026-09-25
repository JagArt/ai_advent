"""Реестр MCP-серверов: три соединения, один каталог инструментов, один маршрут.

Серверов теперь несколько, и у каждого свой транспорт, своя база и свой
жизненный цикл. Знает обо всех только этот модуль: он держит сессии, собирает
объединённый каталог с квалифицированными именами вида `docs__search` и
разводит вызовы по серверам.

Главное решение дня отсюда же. В day19 инструменты передавали друг другу номера
артефактов через общую таблицу — это работало, потому что сервер был один. У трёх
серверов общей базы нет, и наивный путь один: результат git модель пересказывает
в аргументы docs, то есть данные снова идут через генератор текста. Поэтому
артефакты переехали на уровень оркестратора: каждый ответ инструмента целиком
ложится в `orchestration.db`, модель видит только handle с превью, а в аргументах
пишет ссылку `{"$from": 7}`. Реестр разворачивает её перед отправкой на сервер.

Серверы про артефакты не знают вовсе: `NEEDS` ниже — объявление реестра, а не
их инструментов. Из него же собирается контракт звена (какой вид артефакта
годится в какой аргумент) и схема, которую видит модель.
"""

import difflib
import hashlib
import json
import sqlite3
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import AsyncExitStack, asynccontextmanager, closing, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import anyio
import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DB_PATH = HERE / "orchestration.db"
SERVERS_DIR = HERE / "servers"

SEPARATOR = "__"
PROBE_TIMEOUT = 1.0
PREVIEW_STRING = 220
PREVIEW_ITEMS = 6
PREVIEW_BUDGET = 1500
TIGHT_STRING = 80
TIGHT_ITEMS = 3


@dataclass(frozen=True)
class Server:
    """Строка реестра: чем сервер занят, как до него добраться и сколько он живёт."""

    name: str
    title: str
    about: str
    transport: str
    lifetime: str
    script: str | None = None
    host: str | None = None
    port: int | None = None

    @property
    def url(self) -> str | None:
        return f"http://{self.host}:{self.port}/mcp" if self.transport == "http" else None

    @property
    def endpoint(self) -> str:
        if self.transport == "http":
            return self.url
        return f"{Path(sys.executable).name} servers/{self.script}"


SERVERS = (
    Server(
        name="git",
        title="git репозитория",
        about="история изменений: поиск по сообщениям коммитов, журнал, коммит, файлы",
        transport="stdio",
        lifetime="подпроцесс на время запроса",
        script="git.py",
    ),
    Server(
        name="docs",
        title="документация проекта",
        about="поиск по разделам README и TASK, чтение раздела, сводка тезисами",
        transport="stdio",
        lifetime="подпроцесс на время запроса",
        script="docs.py",
    ),
    Server(
        name="vault",
        title="хранилище отчётов",
        about="запись файлов в day20/out, журнал записанного, сверка хешей",
        transport="http",
        lifetime="постоянный процесс, запускается отдельно",
        host="127.0.0.1",
        port=8770,
    ),
)

# Вид артефакта — это тип звена: что годится на вход следующему инструменту.
KINDS = {
    "commits": "коммиты",
    "commit": "один коммит",
    "changed": "что трогали коммиты",
    "found": "разделы документации",
    "section": "раздел документации",
    "summary": "сводка",
    "file": "записанный файл",
    "text": "текст файла",
    "files": "список файлов out",
    "entry": "запись журнала",
    "journal": "журнал хранилища",
    "check": "проверка файла",
    "result": "ответ инструмента",
}

PRODUCES = {
    "git__search": "commits",
    "git__log": "commits",
    "git__show": "commit",
    "git__files": "changed",
    "docs__search": "found",
    "docs__read_section": "section",
    "docs__summarize": "summary",
    "vault__save_file": "file",
    "vault__read_file": "text",
    "vault__list_files": "files",
    "vault__journal_append": "entry",
    "vault__journal_list": "journal",
    "vault__verify": "check",
}


@dataclass(frozen=True)
class Need:
    """Аргумент, который принимает только ссылку на артефакт объявленного вида."""

    kind: str
    field: str
    about: str
    optional: bool = False


NEEDS: dict[str, dict[str, Need]] = {
    "git__files": {
        "commits": Need("commits", "hashes", "хеши коммитов из выдачи поиска или журнала"),
    },
    "docs__search": {
        "paths": Need("changed", "filter", "папки, которые трогали коммиты", optional=True),
    },
    "docs__summarize": {
        "sections": Need("found", "hits", "найденные разделы вместе с их текстом"),
        "context": Need("commits", "subjects", "темы коммитов", optional=True),
    },
    "vault__save_file": {
        "body": Need("summary", "markdown", "готовый текст отчёта"),
    },
    "vault__journal_append": {
        "entry": Need("file", "entry", "запись о файле от save_file"),
    },
    "vault__verify": {
        "expect": Need("file", "check", "путь и ожидаемый хеш файла"),
    },
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    query       TEXT NOT NULL,
    mode        TEXT NOT NULL,
    status      TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    error       TEXT,
    verdict     TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
    id         INTEGER PRIMARY KEY,
    run_id     INTEGER REFERENCES runs (id),
    kind       TEXT NOT NULL,
    server     TEXT NOT NULL,
    tool       TEXT NOT NULL,
    parents    TEXT NOT NULL,
    payload    TEXT NOT NULL,
    sha256     TEXT NOT NULL,
    bytes      INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS routes (
    id           INTEGER PRIMARY KEY,
    run_id       INTEGER REFERENCES runs (id),
    position     INTEGER NOT NULL,
    source       TEXT NOT NULL,
    requested    TEXT NOT NULL,
    server       TEXT,
    tool         TEXT,
    resolution   TEXT NOT NULL,
    status       TEXT NOT NULL,
    refs         TEXT NOT NULL,
    artifact_out INTEGER,
    elapsed_ms   INTEGER NOT NULL,
    error        TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS routes_run ON routes (run_id);
"""


# --- база --------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _local(value: str | None) -> str | None:
    return None if value is None else datetime.fromisoformat(value).astimezone().isoformat()


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            conn.executescript(SCHEMA)
            yield conn


def _canonical(payload: Any) -> str:
    """Одна и та же форма JSON и для хранения, и для хеша: иначе хеш не сойдётся."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def open_run(query: str, mode: str) -> int:
    with _db() as conn:
        return conn.execute(
            "INSERT INTO runs (query, mode, status, started_at) VALUES (?, ?, 'running', ?)",
            (query, mode, _now()),
        ).lastrowid


def finish_run(
    run_id: int, status: str, *, error: str | None = None, verdict: dict[str, Any] | None = None
) -> None:
    with _db() as conn:
        conn.execute(
            "UPDATE runs SET status = ?, finished_at = ?, error = ?, verdict = ? WHERE id = ?",
            (
                status,
                _now(),
                error,
                json.dumps(verdict, ensure_ascii=False) if verdict else None,
                run_id,
            ),
        )


def _run(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    routes = conn.execute(
        "SELECT * FROM routes WHERE run_id = ? ORDER BY position", (row["id"],)
    ).fetchall()
    return {
        "run_id": row["id"],
        "query": row["query"],
        "mode": row["mode"],
        "status": row["status"],
        "started_at": _local(row["started_at"]),
        "finished_at": _local(row["finished_at"]),
        "error": row["error"],
        "verdict": json.loads(row["verdict"]) if row["verdict"] else None,
        "routes": [_route(item) for item in routes],
    }


def _route(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "position": row["position"],
        "source": row["source"],
        "requested": row["requested"],
        "server": row["server"],
        "tool": row["tool"],
        "qualified": (
            f"{row['server']}{SEPARATOR}{row['tool']}" if row["server"] and row["tool"] else None
        ),
        "resolution": row["resolution"],
        "status": row["status"],
        "refs": json.loads(row["refs"]),
        "artifact_out": row["artifact_out"],
        "elapsed_ms": row["elapsed_ms"],
        "error": row["error"],
        "at": _local(row["created_at"]),
    }


def list_runs(limit: int = 10) -> list[dict[str, Any]]:
    with _db() as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [_run(conn, row) for row in rows]


def routes_of(run_id: int) -> list[dict[str, Any]]:
    """Маршруты одного прогона по порядку — материал для проверки порядка вызовов."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM routes WHERE run_id = ? ORDER BY position", (run_id,)
        ).fetchall()
    return [_route(row) for row in rows]


def list_routes(limit: int = 30) -> list[dict[str, Any]]:
    """Журнал маршрутизации целиком, от нового к старому: кто, куда и как разрешилось."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM routes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{**_route(row), "run_id": row["run_id"]} for row in rows]


# --- артефакты ---------------------------------------------------------------


def _artifact_row(conn: sqlite3.Connection, artifact_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
    if row is None:
        last = conn.execute("SELECT MAX(id) AS id FROM artifacts").fetchone()["id"]
        hint = f" Последний созданный артефакт — #{last}." if last else ""
        raise Refused(f"Артефакта #{artifact_id} нет.{hint}", "no_artifact")
    return row


def _check(row: sqlite3.Row) -> Any:
    """Целостность звена: payload в базе должен совпадать со своим хешем."""
    if _digest(row["payload"]) != row["sha256"]:
        raise Refused(
            f"Артефакт #{row['id']} испорчен: sha256 payload не совпадает с записанным.",
            "corrupt",
        )
    return json.loads(row["payload"])


def _lineage(conn: sqlite3.Connection, artifact_id: int) -> list[dict[str, Any]]:
    """Происхождение артефакта. Родителей теперь может быть несколько: сводка
    растёт и из разделов документации, и из тем коммитов — это уже не цепочка, а граф."""
    seen: set[int] = set()
    queue = [artifact_id]

    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        row = _artifact_row(conn, current)
        _check(row)
        queue.extend(json.loads(row["parents"]))

    links = []
    # Номера растут по времени создания, поэтому родитель всегда идёт раньше ребёнка.
    for current in sorted(seen):
        row = _artifact_row(conn, current)
        links.append(
            {
                "artifact_id": row["id"],
                "kind": row["kind"],
                "server": row["server"],
                "tool": row["tool"],
                "parents": json.loads(row["parents"]),
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "at": _local(row["created_at"]),
            }
        )
    return links


def get_artifact(artifact_id: int) -> dict[str, Any]:
    """Payload артефакта целиком и его происхождение — чем именно обменялись серверы."""
    with _db() as conn:
        row = _artifact_row(conn, artifact_id)
        payload = _check(row)
        lineage = _lineage(conn, artifact_id)

    return {
        "artifact_id": row["id"],
        "run_id": row["run_id"],
        "kind": row["kind"],
        "kind_title": KINDS.get(row["kind"], row["kind"]),
        "server": row["server"],
        "tool": row["tool"],
        "sha256": row["sha256"],
        "bytes": row["bytes"],
        "at": _local(row["created_at"]),
        "lineage": lineage,
        "payload": payload,
    }


def _trim(value: Any, *, chars: int, items: int) -> tuple[Any, bool]:
    """Превью для модели: реестр не знает смысла полей, поэтому режет по бюджету."""
    if isinstance(value, str):
        return (value[:chars] + "…", True) if len(value) > chars else (value, False)

    if isinstance(value, list):
        trimmed = len(value) > items
        result = []
        for item in value[:items]:
            shorter, cut = _trim(item, chars=chars, items=items)
            result.append(shorter)
            trimmed = trimmed or cut
        if len(value) > items:
            result.append(f"…ещё {len(value) - items}")
        return result, trimmed

    if isinstance(value, dict):
        trimmed = False
        result = {}
        for key, item in value.items():
            shorter, cut = _trim(item, chars=chars, items=items)
            result[key] = shorter
            trimmed = trimmed or cut
        return result, trimmed

    return value, False


def _preview(payload: Any) -> tuple[Any, bool]:
    preview, trimmed = _trim(payload, chars=PREVIEW_STRING, items=PREVIEW_ITEMS)
    if len(_canonical(preview)) <= PREVIEW_BUDGET:
        return preview, trimmed

    preview, _ = _trim(payload, chars=TIGHT_STRING, items=TIGHT_ITEMS)
    if len(_canonical(preview)) <= PREVIEW_BUDGET:
        return preview, True

    # Осталось только перечислить, что в артефакте есть: целиком он лежит в базе.
    if isinstance(payload, dict):
        return {"поля": sorted(payload)}, True
    return {"элементов": len(payload) if isinstance(payload, list) else 1}, True


def _consumers(kind: str) -> list[str]:
    """Кому годится артефакт этого вида — для подсказки `next` в handle."""
    return [
        f"{tool}({name})"
        for tool, needs in NEEDS.items()
        for name, need in needs.items()
        if need.kind == kind
    ]


def save_artifact(
    *,
    run_id: int | None,
    kind: str,
    server: str,
    tool: str,
    payload: Any,
    parents: list[int],
) -> dict[str, Any]:
    text = _canonical(payload)
    digest = _digest(text)

    with _db() as conn:
        artifact_id = conn.execute(
            "INSERT INTO artifacts (run_id, kind, server, tool, parents, payload, sha256,"
            " bytes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                kind,
                server,
                tool,
                json.dumps(parents),
                text,
                digest,
                len(text.encode()),
                _now(),
            ),
        ).lastrowid

    preview, trimmed = _preview(payload)
    handle: dict[str, Any] = {
        "artifact_id": artifact_id,
        "kind": kind,
        "kind_title": KINDS.get(kind, kind),
        "server": server,
        "tool": tool,
        "parents": parents,
        "sha256": digest,
        "bytes": len(text.encode()),
        "preview": preview,
        "preview_trimmed": trimmed,
    }
    if consumers := _consumers(kind):
        handle["next"] = (
            f"передай {{\"$from\": {artifact_id}}} в " + " или ".join(consumers)
        )
    return handle


# --- отказы ------------------------------------------------------------------


class Refused(Exception):
    """Отказ маршрутизатора: до сервера вызов не дошёл, а модель прочитает причину.

    У отказа есть код причины: он ложится в журнал маршрутов вместо способа
    разрешения имени, и по нему потом считается, на чём именно спотыкалась модель.
    """

    def __init__(self, message: str, cause: str) -> None:
        super().__init__(message)
        self.cause = cause


CAUSES = {
    "collision": "одноимённые инструменты",
    "unknown_server": "нет такого сервера",
    "unknown_tool": "нет такого инструмента",
    "offline": "сервер не на связи",
    "inline": "данные вместо ссылки",
    "wrong_kind": "чужой вид артефакта",
    "no_artifact": "нет артефакта",
    "no_field": "нет поля в артефакте",
    "missing_arg": "нет обязательной ссылки",
    "corrupt": "артефакт испорчен",
    "bad_arguments": "аргументы не разобрались",
}


def _last_of_kind(kind: str) -> int | None:
    with _db() as conn:
        row = conn.execute(
            "SELECT MAX(id) AS id FROM artifacts WHERE kind = ?", (kind,)
        ).fetchone()
    return row["id"]


def _producers(kind: str) -> str:
    tools = [tool for tool, produced in PRODUCES.items() if produced == kind]
    return " или ".join(tools) if tools else "никто"


def _needs_hint(kind: str) -> str:
    hint = f"артефакт вида {kind} — {KINDS.get(kind, kind)}, его создаёт {_producers(kind)}"
    if last := _last_of_kind(kind):
        hint += f". Последний такой артефакт — #{last}"
    return hint


# --- маршрут -----------------------------------------------------------------


@dataclass(frozen=True)
class Dispatch:
    """Один вызов: куда ушёл, что подставили, что получилось."""

    position: int
    requested: str
    resolution: str
    status: str
    elapsed_ms: int
    server: str | None = None
    tool: str | None = None
    refs: list[dict[str, Any]] = field(default_factory=list)
    arguments: dict[str, Any] = field(default_factory=dict)
    handle: dict[str, Any] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def qualified(self) -> str | None:
        return f"{self.server}{SEPARATOR}{self.tool}" if self.server else None

    def text(self) -> str:
        """Что уходит модели сообщением role=tool: handle или причина отказа."""
        if self.error:
            return self.error
        return json.dumps(self.handle, ensure_ascii=False)

    def frame(self) -> dict[str, Any]:
        """Кадр для страницы и CLI: без payload, только маршрут и handle."""
        return {
            "position": self.position,
            "requested": self.requested,
            "server": self.server,
            "tool": self.tool,
            "qualified": self.qualified,
            "resolution": self.resolution,
            "status": self.status,
            "elapsed_ms": self.elapsed_ms,
            "refs": self.refs,
            "arguments": _trim(self.arguments, chars=TIGHT_STRING, items=TIGHT_ITEMS)[0],
            "handle": self.handle,
            "error": self.error,
        }


class Registry:
    """Сессии всех серверов, объединённый каталог и маршрутизация вызовов."""

    def __init__(self) -> None:
        self.servers: list[dict[str, Any]] = []
        self.sessions: dict[str, ClientSession] = {}
        self.tools: dict[str, types.Tool] = {}
        self.bare: dict[str, list[str]] = {}
        self._positions: dict[int, int] = {}

    # --- каталог ---

    def _add(self, server: Server, tools: list[types.Tool]) -> None:
        for tool in tools:
            qualified = f"{server.name}{SEPARATOR}{tool.name}"
            self.tools[qualified] = tool
            self.bare.setdefault(tool.name, []).append(qualified)

    @property
    def online(self) -> list[dict[str, Any]]:
        return [server for server in self.servers if server["online"]]

    def info(self) -> dict[str, Any]:
        """Реестр для страницы: серверы, каталог и коллизии имён."""
        return {
            "servers": self.servers,
            "catalog": self.catalog(),
            "collisions": {
                name: names for name, names in sorted(self.bare.items()) if len(names) > 1
            },
            "causes": CAUSES,
            "kinds": KINDS,
            "tools": len(self.tools),
        }

    def catalog(self) -> list[dict[str, Any]]:
        entries = []
        for qualified, tool in self.tools.items():
            server, name = qualified.split(SEPARATOR, 1)
            needs = NEEDS.get(qualified, {})
            entries.append(
                {
                    "qualified": qualified,
                    "server": server,
                    "tool": name,
                    "description": (tool.description or "").strip(),
                    "produces": PRODUCES.get(qualified, "result"),
                    "produces_title": KINDS.get(PRODUCES.get(qualified, "result")),
                    "collides": len(self.bare.get(name, [])) > 1,
                    "needs": [
                        {
                            "arg": arg,
                            "kind": need.kind,
                            "from": _producers(need.kind),
                            "optional": need.optional,
                            "about": need.about,
                        }
                        for arg, need in needs.items()
                    ],
                    "input_schema": self._schema(qualified, tool),
                }
            )
        return entries

    def _schema(self, qualified: str, tool: types.Tool) -> dict[str, Any]:
        """Схема для модели: аргументы-ссылки подменяются на форму {"$from": N}.

        Схему правит реестр, а не сервер: сервер про артефакты не знает и ждёт
        настоящие данные — их подставит маршрутизатор.
        """
        schema = json.loads(json.dumps(tool.input_schema))
        needs = NEEDS.get(qualified, {})
        if not needs:
            return schema

        properties = schema.setdefault("properties", {})
        for arg, need in needs.items():
            if arg not in properties:
                continue
            properties[arg] = {
                "type": "object",
                "properties": {"$from": {"type": "integer", "minimum": 1}},
                "required": ["$from"],
                "additionalProperties": False,
                "description": (
                    f"Ссылка на артефакт вида {need.kind} ({KINDS.get(need.kind, need.kind)}):"
                    f" {{\"$from\": N}}. {need.about.capitalize()}."
                    f" Артефакт создаёт {_producers(need.kind)}."
                    " Сами данные сюда подставит реестр — переписывать их не нужно."
                ),
            }
        return schema

    def openai_tools(self) -> list[dict[str, Any]]:
        """Объединённый каталог для модели: имена с префиксом сервера."""
        tools = []
        for entry in self.catalog():
            about = next(
                server["about"] for server in self.servers if server["name"] == entry["server"]
            )
            description = (
                f"{entry['description']}\n"
                f"Сервер {entry['server']} — {about}.\n"
                f"Возвращает артефакт вида {entry['produces']}"
                f" ({entry['produces_title']}): номер, превью и хеш."
            )
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": entry["qualified"],
                        "description": description,
                        "parameters": entry["input_schema"],
                    },
                }
            )
        return tools

    # --- разрешение имени ---

    def resolve(self, name: str) -> tuple[str, str, str]:
        """Имя от модели в (сервер, инструмент, как разрешилось). Иначе — `Refused`."""
        requested = (name or "").strip()

        if requested in self.tools:
            server, tool = requested.split(SEPARATOR, 1)
            return server, tool, "qualified"

        if SEPARATOR in requested:
            server, tool = requested.split(SEPARATOR, 1)
            known = {item["name"] for item in self.servers}
            if server not in known:
                raise Refused(
                    f"Сервера {server!r} в реестре нет. Есть: {', '.join(sorted(known))}.",
                    "unknown_server",
                )
            if not any(item["name"] == server and item["online"] for item in self.servers):
                raise Refused(
                    f"Сервер {server} не на связи, вызвать {requested} нечем.", "offline"
                )
            raise Refused(
                f"У сервера {server} нет инструмента {tool!r}. Есть: "
                + ", ".join(sorted(self._tools_of(server)))
                + ".",
                "unknown_tool",
            )

        # Голое имя: однозначное маршрутизируется, одноимённое у двух серверов — нет.
        candidates = self.bare.get(requested, [])
        if len(candidates) == 1:
            server, tool = candidates[0].split(SEPARATOR, 1)
            return server, tool, "resolved"

        if len(candidates) > 1:
            raise Refused(
                f"Инструмент {requested!r} есть у нескольких серверов: "
                + "; ".join(
                    f"{qualified} — {(self.tools[qualified].description or '').strip().splitlines()[0]}"
                    for qualified in candidates
                )
                + ". Позови нужный полным именем.",
                "collision",
            )

        close = difflib.get_close_matches(requested, self.tools, n=3, cutoff=0.5)
        hint = f" Похоже на: {', '.join(close)}." if close else ""
        raise Refused(
            f"Инструмента {requested!r} в реестре нет. Серверы: "
            + ", ".join(
                f"{item['name']} ({len(self._tools_of(item['name']))})" for item in self.online
            )
            + f".{hint}",
            "unknown_tool",
        )

    def _tools_of(self, server: str) -> list[str]:
        prefix = f"{server}{SEPARATOR}"
        return [name.removeprefix(prefix) for name in self.tools if name.startswith(prefix)]

    # --- ссылки на артефакты ---

    def _substitute(
        self, qualified: str, arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], list[dict[str, Any]], list[int]]:
        """Ссылки `{"$from": N}` в настоящие данные. Инлайн данных в ref-аргумент — отказ."""
        needs = NEEDS.get(qualified, {})
        prepared = dict(arguments)
        refs: list[dict[str, Any]] = []
        parents: list[int] = []

        for arg, value in arguments.items():
            reference = value.get("$from") if isinstance(value, dict) else None

            if arg not in needs:
                if reference is not None:
                    raise Refused(
                        f"У инструмента {qualified} аргумент {arg!r} принимает данные, "
                        "а не ссылку на артефакт.",
                        "inline",
                    )
                continue

            need = needs[arg]
            if reference is None:
                if value is None and need.optional:
                    prepared.pop(arg)
                    continue
                raise Refused(
                    f"Аргумент {arg!r} инструмента {qualified} принимает только ссылку "
                    f"вида {{\"$from\": N}} на {_needs_hint(need.kind)}. "
                    "Данные через себя не переписывай — их подставит реестр.",
                    "inline",
                )

            with _db() as conn:
                row = _artifact_row(conn, int(reference))
                payload = _check(row)

            if row["kind"] != need.kind:
                raise Refused(
                    f"Артефакт #{row['id']} — {KINDS.get(row['kind'], row['kind'])} "
                    f"(kind={row['kind']}), а аргументу {arg!r} инструмента {qualified} "
                    f"нужен {_needs_hint(need.kind)}.",
                    "wrong_kind",
                )

            if not isinstance(payload, dict) or need.field not in payload:
                raise Refused(
                    f"В артефакте #{row['id']} нет поля {need.field!r}: подставить в "
                    f"{arg!r} нечего.",
                    "no_field",
                )

            data = payload[need.field]
            prepared[arg] = data
            parents.append(row["id"])
            refs.append(
                {
                    "arg": arg,
                    "artifact_id": row["id"],
                    "kind": row["kind"],
                    "field": need.field,
                    "sha256": _digest(_canonical(data)),
                    "bytes": len(_canonical(data).encode()),
                }
            )

        for arg, need in needs.items():
            if arg not in prepared and not need.optional:
                raise Refused(
                    f"Инструменту {qualified} нужен аргумент {arg!r} — ссылка "
                    f"{{\"$from\": N}} на {_needs_hint(need.kind)}.",
                    "missing_arg",
                )

        return prepared, refs, parents

    # --- вызов ---

    def _position(self, run_id: int | None) -> int:
        self._positions[run_id] = self._positions.get(run_id, 0) + 1
        return self._positions[run_id]

    async def dispatch(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        run_id: int | None = None,
        source: str = "agent",
    ) -> Dispatch:
        """Единственная дверь к серверам: разрешение имени, подстановка ссылок, вызов.

        Отказ не поднимается исключением — он возвращается кадром и уходит модели
        текстом: перепутанный сервер должен быть читаемой ошибкой, а не обрывом.
        """
        started = perf_counter()
        position = self._position(run_id)
        arguments = arguments or {}

        def elapsed() -> int:
            return round((perf_counter() - started) * 1000)

        def refused(exc: Refused, *, server: str | None = None, tool: str | None = None) -> Dispatch:
            # Причина отказа встаёт на место способа разрешения имени: до сервера
            # вызов не дошёл, разрешать было нечего.
            record = Dispatch(
                position=position,
                requested=name,
                resolution=exc.cause,
                status="refused",
                elapsed_ms=elapsed(),
                server=server,
                tool=tool,
                arguments=arguments,
                error=str(exc),
            )
            self._log(run_id, source, record)
            return record

        try:
            server, tool, resolution = self.resolve(name)
        except Refused as exc:
            return refused(exc)

        qualified = f"{server}{SEPARATOR}{tool}"
        try:
            prepared, refs, parents = self._substitute(qualified, arguments)
        except Refused as exc:
            return refused(exc, server=server, tool=tool)

        try:
            result = await self.sessions[server].call_tool(tool, prepared)
        except Exception as exc:
            record = Dispatch(
                position=position,
                requested=name,
                resolution=resolution,
                status="failed",
                elapsed_ms=elapsed(),
                server=server,
                tool=tool,
                refs=refs,
                arguments=arguments,
                error=f"Сервер {server} не ответил: {reason(exc)}",
            )
            self._log(run_id, source, record)
            return record

        if result.is_error:
            record = Dispatch(
                position=position,
                requested=name,
                resolution=resolution,
                status="failed",
                elapsed_ms=elapsed(),
                server=server,
                tool=tool,
                refs=refs,
                arguments=arguments,
                error=_tool_error(result, tool),
            )
            self._log(run_id, source, record)
            return record

        handle = save_artifact(
            run_id=run_id,
            kind=PRODUCES.get(qualified, "result"),
            server=server,
            tool=qualified,
            payload=_payload_of(result),
            parents=parents,
        )
        record = Dispatch(
            position=position,
            requested=name,
            resolution=resolution,
            status="ok",
            elapsed_ms=elapsed(),
            server=server,
            tool=tool,
            refs=refs,
            arguments=arguments,
            handle=handle,
        )
        self._log(run_id, source, record)
        return record

    def refuse(
        self,
        name: str,
        error: str,
        cause: str,
        *,
        run_id: int | None = None,
        source: str = "agent",
    ) -> Dispatch:
        """Отказ до маршрутизации — например, аргументы не разобрались как JSON.

        Такой вызов тоже занимает позицию в журнале: попытка была, и в проверке
        порядка она должна быть видна.
        """
        record = Dispatch(
            position=self._position(run_id),
            requested=name,
            resolution=cause,
            status="refused",
            elapsed_ms=0,
            error=error,
        )
        self._log(run_id, source, record)
        return record

    def _log(self, run_id: int | None, source: str, record: Dispatch) -> None:
        with _db() as conn:
            conn.execute(
                "INSERT INTO routes (run_id, position, source, requested, server, tool,"
                " resolution, status, refs, artifact_out, elapsed_ms, error, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    record.position,
                    source,
                    record.requested,
                    record.server,
                    record.tool,
                    record.resolution,
                    record.status,
                    json.dumps(record.refs, ensure_ascii=False),
                    record.handle["artifact_id"] if record.handle else None,
                    record.elapsed_ms,
                    record.error,
                    _now(),
                ),
            )


# --- ответы серверов ---------------------------------------------------------


def _result_text(result: types.CallToolResult) -> str:
    text = "\n".join(
        block.text for block in result.content if isinstance(block, types.TextContent)
    )
    if not text and result.structured_content is not None:
        text = json.dumps(result.structured_content, ensure_ascii=False)
    return text or "(инструмент вернул пустой ответ)"


def _tool_error(result: types.CallToolResult, tool: str) -> str:
    """Причина отказа инструмента без служебной приставки SDK: имя уже есть в маршруте."""
    return _result_text(result).removeprefix(f"Error executing tool {tool}: ")


def _payload_of(result: types.CallToolResult) -> Any:
    payload = result.structured_content
    if payload is None:
        try:
            return json.loads(_result_text(result))
        except json.JSONDecodeError:
            return {"text": _result_text(result)}
    # Список SDK заворачивает в {"result": [...]}: у structuredContent корень — объект.
    if set(payload) == {"result"}:
        return payload["result"]
    return payload


def reason(exc: BaseException) -> str:
    """Сбой транспорта приезжает группой исключений; человеку нужна первая причина."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


# --- соединения --------------------------------------------------------------


async def _reachable(server: Server) -> str | None:
    """Постоянный сервер проверяется до транспорта: иначе сбой соединения вылезет
    группой исключений уже при закрытии стека, посреди чужого ответа."""
    try:
        with anyio.fail_after(PROBE_TIMEOUT):
            stream = await anyio.connect_tcp(server.host, server.port)
            await stream.aclose()
    except (OSError, TimeoutError) as exc:
        return (
            f"{server.url} не отвечает ({type(exc).__name__}). "
            f"Запустите python day20/servers/{server.name}.py"
        )
    return None


@asynccontextmanager
async def connect() -> AsyncIterator[Registry]:
    """Соединения со всеми серверами реестра на время блока.

    Серверы подключаются по очереди и в одной задаче: транспорты MCP держат
    внутри себя группы задач anyio, а те привязаны к задаче, которая их открыла.
    Недоступный сервер не мешает остальным — он остаётся в реестре с причиной.
    """
    registry = Registry()
    stack = AsyncExitStack()
    await stack.__aenter__()

    try:
        for server in SERVERS:
            started = perf_counter()
            record: dict[str, Any] = {
                "name": server.name,
                "title": server.title,
                "about": server.about,
                "transport": server.transport,
                "lifetime": server.lifetime,
                "endpoint": server.endpoint,
                "online": False,
                "error": None,
                "tools": [],
            }

            try:
                if server.transport == "http":
                    if error := await _reachable(server):
                        raise ConnectionError(error)
                    read, write = await stack.enter_async_context(
                        streamable_http_client(server.url)
                    )
                else:
                    read, write = await stack.enter_async_context(
                        stdio_client(
                            StdioServerParameters(
                                command=sys.executable,
                                args=[str(SERVERS_DIR / server.script)],
                            )
                        )
                    )

                session = await stack.enter_async_context(ClientSession(read, write))
                init = await session.initialize()
                listed = await session.list_tools()
            except Exception as exc:
                record["error"] = reason(exc)
                registry.servers.append(record)
                continue

            registry.sessions[server.name] = session
            registry._add(server, listed.tools)
            record.update(
                {
                    "online": True,
                    "server": f"{init.server_info.name} {init.server_info.version}".strip(),
                    "protocol": init.protocol_version,
                    "capabilities": sorted(init.capabilities.model_dump(exclude_none=True)),
                    "elapsed_ms": round((perf_counter() - started) * 1000),
                    "tools": [tool.name for tool in listed.tools],
                }
            )
            registry.servers.append(record)

        yield registry
    finally:
        try:
            await stack.aclose()
        except BaseException as exc:  # noqa: BLE001
            # Обрыв на закрытии транспорта — не ошибка запроса: ответ уже отдан.
            print(f"реестр: транспорт закрылся с ошибкой — {reason(exc)}", file=sys.stderr)
