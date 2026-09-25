"""Длинный флоу: контракт из семи шагов по трём серверам и проверка порядка вызовов.

Контракт — это список шагов: какой инструмент какого сервера, что он должен
произвести и зачем он в этом месте. Он служит двум делам сразу: по нему флоу
выполняется в режиме `flow` и по нему же проверяется прогон в режиме `agent`,
где инструменты выбирает модель. Сравнивать есть с чем, потому что оба режима
ходят через один и тот же `registry.dispatch` и пишут один и тот же журнал.

Шаги идут не в произвольном порядке: `git__files` нужны хеши от `git__search`,
`docs__search` — папки от `git__files`, и только потом появляется то, что можно
сводить и записывать. Инструмент `search` при этом встречается дважды и на разных
серверах — сначала у git, потом у docs.
"""

import difflib
from collections import Counter
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import registry as reg


@dataclass
class Task:
    """Запрос и номера артефактов, которые флоу собрал по ходу."""

    query: str
    limit: int = 5
    bullets: int = 5
    name: str | None = None
    artifacts: dict[str, int] = field(default_factory=dict)

    def ref(self, kind: str) -> dict[str, int]:
        """Ссылка на артефакт нужного вида — ровно то же, что пишет модель."""
        return {"$from": self.artifacts[kind]}


@dataclass(frozen=True)
class Step:
    tool: str
    produces: str
    why: str
    arguments: Callable[[Task], dict[str, Any]]

    @property
    def server(self) -> str:
        return self.tool.split(reg.SEPARATOR, 1)[0]


STEPS: tuple[Step, ...] = (
    Step(
        tool="git__search",
        produces="commits",
        why="что в истории репозитория делали по этой теме",
        arguments=lambda task: {"query": task.query, "limit": task.limit * 2},
    ),
    Step(
        tool="git__files",
        produces="changed",
        why="какие папки трогали эти коммиты",
        arguments=lambda task: {"commits": task.ref("commits")},
    ),
    Step(
        tool="docs__search",
        produces="found",
        why="разделы документации в этих же папках",
        arguments=lambda task: {
            "query": task.query,
            "paths": task.ref("changed"),
            "limit": task.limit,
        },
    ),
    Step(
        tool="docs__summarize",
        produces="summary",
        why="тезисы по разделам с темами коммитов рядом",
        arguments=lambda task: {
            "query": task.query,
            "sections": task.ref("found"),
            "context": task.ref("commits"),
            "max_bullets": task.bullets,
        },
    ),
    Step(
        tool="vault__save_file",
        produces="file",
        why="отчёт файлом в day20/out",
        arguments=lambda task: {"name": task.name or task.query, "body": task.ref("summary")},
    ),
    Step(
        tool="vault__journal_append",
        produces="entry",
        why="запись в журнал хранилища, он переживает перезапуск",
        arguments=lambda task: {
            "entry": task.ref("file"),
            "note": f"длинный флоу day20: {task.query}",
        },
    ),
    Step(
        tool="vault__verify",
        produces="check",
        why="sha256 файла на диске против артефакта — флоу замыкается",
        arguments=lambda task: {"expect": task.ref("file")},
    ),
)

CONTRACT = tuple(step.tool for step in STEPS)
SERVERS_IN_ORDER = tuple(dict.fromkeys(step.server for step in STEPS))

PROMPT = (
    "Собери отчёт по теме «{query}» и сохрани его.\n"
    "Порядок такой: найди коммиты по теме у сервера git, узнай, какие папки они "
    "трогали, найди разделы документации в этих папках у сервера docs, сведи их "
    "в тезисы, запиши отчёт файлом в хранилище, занеси файл в журнал и сверь его "
    "sha256 на диске. Разделов документации возьми {limit}, тезисов {bullets}."
)


def contract() -> list[dict[str, Any]]:
    """Контракт для страницы и CLI: чем занят каждый шаг и что оставляет после себя."""
    return [
        {
            "position": position,
            "tool": step.tool,
            "server": step.server,
            "produces": step.produces,
            "produces_title": reg.KINDS.get(step.produces, step.produces),
            "why": step.why,
        }
        for position, step in enumerate(STEPS, start=1)
    ]


def prompt(task: Task) -> str:
    return PROMPT.format(query=task.query, limit=task.limit, bullets=task.bullets)


def verdict(run_id: int) -> dict[str, Any]:
    """Проверка прогона по контракту: те же инструменты, те же серверы, тот же порядок.

    Считается по журналу маршрутов, поэтому режимы сравнимы: код ли держал номера
    артефактов или модель, проверка одна и та же.
    """
    routes = reg.routes_of(run_id)
    done = [route["qualified"] for route in routes if route["status"] == "ok"]

    matched, missing, extra = 0, [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        a=list(CONTRACT), b=done, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            matched += i2 - i1
        else:
            missing.extend(CONTRACT[i1:i2])
            extra.extend(done[j1:j2])

    refused = [route for route in routes if route["status"] == "refused"]
    failed = [route for route in routes if route["status"] == "failed"]

    return {
        "contract": list(CONTRACT),
        "actual": done,
        "servers": list(dict.fromkeys(route["server"] for route in routes if route["server"])),
        "order_ok": not missing and not extra,
        "complete": not missing,
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "calls": len(routes),
        "refused": len(refused),
        "failed": len(failed),
        "resolved_bare": sum(1 for route in routes if route["resolution"] == "resolved"),
        "causes": dict(Counter(route["resolution"] for route in refused)),
        "tools_ms": sum(route["elapsed_ms"] for route in routes),
    }


def _artifact(task: Task, kind: str) -> dict[str, Any] | None:
    """Payload артефакта целиком — для итога нужны путь и хеши, а не превью."""
    if kind not in task.artifacts:
        return None
    return reg.get_artifact(task.artifacts[kind])["payload"]


async def run(
    registry: reg.Registry, task: Task, *, run_id: int, source: str = "flow"
) -> AsyncIterator[dict[str, Any]]:
    """Проходит контракт сам, подставляя номера артефактов в ссылки `{"$from": N}`.

    Кадры отдаются по ходу: шаг начался, шаг закончился, флоу закончился. Ждать
    конца не нужно — шагов семь и три из них ходят по сети.
    """
    started = perf_counter()
    status, failed_at, error = "ok", None, None
    trace: list[dict[str, Any]] = []

    for position, step in enumerate(STEPS, start=1):
        yield {
            "event": "step",
            "stage": "started",
            "position": position,
            "total": len(STEPS),
            "tool": step.tool,
            "server": step.server,
            "produces": step.produces,
            "why": step.why,
        }

        record = await registry.dispatch(
            step.tool, step.arguments(task), run_id=run_id, source=source
        )
        frame = {
            "event": "step",
            "stage": "done",
            **record.frame(),
            "position": position,
            "total": len(STEPS),
            "route": record.position,
            "why": step.why,
            "produces": step.produces,
        }
        trace.append(frame)
        yield frame

        if not record.ok:
            status, failed_at, error = "failed", step.tool, record.error
            break

        task.artifacts[record.handle["kind"]] = record.handle["artifact_id"]

    elapsed_ms = round((perf_counter() - started) * 1000)
    checked = verdict(run_id)
    reg.finish_run(run_id, "done" if status == "ok" else "failed", error=error, verdict=checked)

    yield {
        "event": "flow",
        "run_id": run_id,
        "query": task.query,
        "status": status,
        "failed_at": failed_at,
        "error": error,
        "elapsed_ms": elapsed_ms,
        "artifacts": task.artifacts,
        "steps": trace,
        "file": _artifact(task, "file"),
        "check": _artifact(task, "check"),
        "verdict": checked,
    }
