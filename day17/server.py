"""MCP-сервер над git этого репозитория: журнал, коммит, статус.

Запускается клиентом как подпроцесс и говорит по stdio, поэтому писать в stdout
здесь нельзя ничем, кроме протокола: одна лишняя строка ломает соединение.

Git вызывается списком аргументов, не через shell: модель передаёт только
поля инструмента, а не произвольную команду.
"""

import re
import subprocess
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

ROOT = Path(__file__).resolve().parent.parent
# Хеш, ветка, тег, HEAD, HEAD~2, origin/main. Не диапазон и не флаг.
REF = re.compile(r"^(?:HEAD(?:~\d+|\^[0-9]*)?|[A-Za-z0-9][A-Za-z0-9._/-]*)$")
SEP = "\x1f"

Limit = Annotated[
    int,
    Field(description="Сколько последних коммитов вернуть, от 1 до 30.", ge=1, le=30),
]
Ref = Annotated[
    str,
    Field(description="Ревизия: хеш, ветка, тег или HEAD, например «HEAD» или «abc1234»."),
]

mcp = MCPServer("AI Advent", version="17.0")


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


def _ref(value: str) -> str:
    ref = value.strip()
    if not ref or ref.startswith("-") or ".." in ref or not REF.match(ref):
        raise ToolError(
            f"Непонятная ревизия: {value!r}. Ожидается хеш, ветка, тег или HEAD."
        )
    return ref


def _commit(line: str) -> dict[str, str]:
    hash_, date, author, subject = line.split(SEP, 3)
    return {"hash": hash_, "date": date, "author": author, "subject": subject}


@mcp.tool()
def git_log(limit: Limit = 10) -> list[dict[str, str]]:
    """Последние коммиты репозитория: хеш, дата, автор и тема."""
    output = _git("log", f"-n{limit}", f"--format=%H{SEP}%cI{SEP}%an{SEP}%s")
    lines = [line for line in output.splitlines() if line]
    if not lines:
        raise ToolError("В репозитории нет коммитов.")
    return [_commit(line) for line in lines]


@mcp.tool()
def git_show(ref: Ref) -> dict[str, Any]:
    """Один коммит: сообщение и список изменённых файлов со статистикой."""
    name = _ref(ref)
    output = _git("log", "-1", f"--format=%H{SEP}%cI{SEP}%an{SEP}%s{SEP}%b", name)
    if not output.strip():
        raise ToolError(f"Ревизии {name} в репозитории нет.")

    hash_, date, author, subject, body = output.strip("\n").split(SEP, 4)
    files = []
    for line in _git("diff-tree", "--no-commit-id", "--numstat", "--root", "-r", name).splitlines():
        insertions, deletions, path = line.split("\t", 2)
        files.append({"path": path, "insertions": insertions, "deletions": deletions})

    return {
        "hash": hash_,
        "date": date,
        "author": author,
        "subject": subject,
        "body": body.strip(),
        "files": files,
    }


@mcp.tool()
def git_status() -> dict[str, Any]:
    """Текущая ветка и грязные файлы рабочего дерева."""
    output = _git("status", "--porcelain=v1", "-b")
    lines = output.splitlines()
    if not lines:
        raise ToolError("git status ничего не вернул.")

    heading = lines[0]
    if not heading.startswith("## "):
        raise ToolError(f"Непонятный статус: {heading!r}.")

    branch = heading[3:].split("...", 1)[0]
    files = []
    for line in lines[1:]:
        files.append({"status": line[:2].strip(), "path": line[3:]})

    return {"branch": branch, "files": files}


if __name__ == "__main__":
    mcp.run()
