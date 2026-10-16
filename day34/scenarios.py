"""Проверки песочницы без модели и живой прогон двух целей через агента.

    python day34/scenarios.py            # jail, поиск, запись, сброс — без LLM
    python day34/scenarios.py ask        # оба пресета: reset → цель → markdown
    python day34/scenarios.py ask --cloud
    python day34/scenarios.py ask 1      # только первый пресет (нумерация с 1)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import agent
import llm
import workspace

CHECKS = (
    (
        "docs",
        "в README есть archive_note и аргумент body",
        lambda: "archive_note" in workspace.read_plain("README.md") and "body" in workspace.read_plain("README.md"),
    ),
    (
        "audit",
        "появился AUDIT.md",
        lambda: (workspace.WORKSPACE / "AUDIT.md").is_file(),
    ),
)


def _ok(condition: bool, message: str) -> None:
    mark = "ok" if condition else "FAIL"
    print(f"  {mark}  {message}")
    if not condition:
        raise SystemExit(1)


def check_workspace() -> None:
    """Детерминированные операции с файлами: то, что агент делает через MCP."""
    print("## песочница, без модели\n")
    info = workspace.reset()
    _ok(info["files"] == 6, f"reset копирует эталон: {info['files']} файлов")
    _ok("src/notes/api.py" in info["paths"], "в дереве есть src/notes/api.py")
    _ok("archive_note" not in workspace.read_plain("README.md"), "эталонный README устарел: нет archive_note")

    for bad in ("..", "../README.md", "/etc/passwd"):
        try:
            workspace.resolve(bad)
        except workspace.WorkspaceError:
            continue
        _ok(False, f"jail пропустил {bad!r}")
    _ok(True, "jail отвергает .. и абсолютные пути")

    found = workspace.search("create_note")
    paths = {hit["path"] for hit in found["hits"]}
    _ok({"src/notes/api.py", "src/notes/store.py", "README.md"} <= paths, f"search create_note: {sorted(paths)}")

    workspace.write("AUDIT.md", "# аудит\nREADME не знает archive_note\n")
    delta = workspace.diff()
    _ok("AUDIT.md" in delta["changed"] and "AUDIT.md" in delta["diff"], "write + diff показывают новый файл")

    workspace.reset()
    _ok(not (workspace.WORKSPACE / "AUDIT.md").exists(), "повторный reset стирает правки")
    _ok(workspace.diff()["changed"] == [], "после reset отличий от эталона нет")
    print()


def _format_args(arguments: dict) -> str:
    parts = []
    for key, value in arguments.items():
        shown = value if not isinstance(value, str) or len(value) <= 60 else f"{value[:60]}…"
        parts.append(f"{key}={shown!r}")
    return ", ".join(parts)


async def run_preset(index: int, provider: str) -> agent.Done:
    question = agent.PRESETS[index]
    print(f"## пресет {index + 1}\n")
    print(f"Цель: {question}\n")
    workspace.reset()
    llm.check(provider, await llm.probe())
    async with agent.connect() as connection:
        done: agent.Done | None = None
        async for event in agent.answer(connection, question=question, provider=provider):
            if isinstance(event, agent.ToolCalled):
                args = _format_args(event.result.arguments)
                mark = "" if event.result.ok else " · отказ"
                print(f"  ⚙ r{event.round} {event.result.name}({args}){mark}")
            elif isinstance(event, agent.Nudged):
                print(f"  ↻ раунд {event.round} без вызовов — толчок")
            elif isinstance(event, agent.Done):
                done = event
        if done is None:
            raise RuntimeError("агент не вернул Done")
    print()
    print(done.text or "(модель не дала текста)")
    print(
        f"— {done.model} · раундов {done.rounds} · вызовов {done.calls}"
        f"{' · толчок' if done.nudged else ''} · {done.total_ms / 1000:.1f} с"
    )
    if done.files_touched:
        print(f"файлы: {', '.join(done.files_touched)}")
    if done.changed:
        print(f"изменены: {', '.join(done.changed)}")
        snippet = "\n".join(done.diff.splitlines()[:40])
        print()
        print("```")
        print(snippet)
        print("```")
    key, title, predicate = CHECKS[index]
    passed = predicate()
    print()
    print(f"проверка {key}: {'ok' if passed else 'FAIL'} — {title}")
    print()
    if not passed:
        raise SystemExit(1)
    return done


async def run_live(provider: str, only: int | None) -> None:
    indexes = [only - 1] if only else list(range(len(agent.PRESETS)))
    for index in indexes:
        if index < 0 or index >= len(agent.PRESETS):
            raise SystemExit(f"пресета {index + 1} нет, есть 1–{len(agent.PRESETS)}")
        await run_preset(index, provider)
        if index != indexes[-1]:
            workspace.reset()


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверки day34: песочница и живые цели.")
    parser.add_argument("command", nargs="?", default="check", choices=("check", "ask"))
    parser.add_argument("preset", nargs="?", type=int)
    parser.add_argument("--local", action="store_const", const="local", dest="provider")
    parser.add_argument("--cloud", action="store_const", const="cloud", dest="provider")
    args = parser.parse_args()
    provider = args.provider or "cloud"

    if args.command == "check":
        check_workspace()
        return
    asyncio.run(run_live(provider, args.preset))


if __name__ == "__main__":
    main()
