"""MCP-сервер над этим же репозиторием: дни, их задания и файлы.

Запускается клиентом как подпроцесс и говорит по stdio, поэтому писать в stdout
здесь нельзя ничем, кроме протокола: одна лишняя строка ломает соединение.
"""

import re
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

ROOT = Path(__file__).resolve().parent.parent
DAY_DIR = re.compile(r"^day(\d+)$")
DAY_ROW = re.compile(r"^\|\s*\[(day\d+)/\]\([^)]+\)\s*\|\s*(.+?)\s*\|$")
SKIP_SUFFIXES = {".db", ".pyc"}
SKIP_DIRS = {"__pycache__"}

Day = Annotated[str, Field(description="Номер или папка дня, например «13» или «day13».")]

mcp = MCPServer("AI Advent", version="16.0")


def _day_dir(day: str) -> Path:
    """Принимает и «13», и «day13»: модель зовёт инструмент и так, и так.

    Отказ — это `ToolError`, а не `ValueError`: первый доезжает до модели текстом,
    и у неё остаётся ход на исправление, второй превращается в «Error executing tool».
    """
    name = day.strip().lower()
    if name.isdigit():
        name = f"day{name}"

    if not DAY_DIR.match(name):
        raise ToolError(f"Непонятный день: {day!r}. Ожидается «day13» или «13».")

    path = ROOT / name
    if not path.is_dir():
        raise ToolError(f"Папки {name} в проекте нет.")

    return path


def _titles() -> dict[str, str]:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    rows = (DAY_ROW.match(line) for line in readme.splitlines())
    return {row.group(1): row.group(2) for row in rows if row}


@mcp.tool()
def list_days() -> list[dict[str, str]]:
    """Список итераций проекта: папка дня и чем этот день занят."""
    titles = _titles()
    days = sorted(
        (path for path in ROOT.iterdir() if path.is_dir() and DAY_DIR.match(path.name)),
        key=lambda path: int(DAY_DIR.match(path.name).group(1)),
    )
    return [{"day": path.name, "title": titles.get(path.name, "")} for path in days]


@mcp.tool()
def day_task(day: Day) -> str:
    """Текст задания дня — то, что требовалось сделать, из его TASK.md."""
    root = _day_dir(day)
    path = root / "TASK.md"
    if not path.is_file():
        raise ToolError(f"У {root.name} нет TASK.md.")

    return path.read_text(encoding="utf-8")


@mcp.tool()
def day_files(day: Day) -> list[dict[str, str]]:
    """Из чего состоит день: файлы с размерами в байтах, без кэшей и баз."""
    root = _day_dir(day)
    files = (
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix not in SKIP_SUFFIXES
        and not SKIP_DIRS.intersection(path.relative_to(root).parts)
    )
    return [
        {"path": str(path.relative_to(root)), "bytes": str(path.stat().st_size)}
        for path in files
    ]


if __name__ == "__main__":
    mcp.run()
