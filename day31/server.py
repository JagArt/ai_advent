"""MCP-сервер над проектом: git этого репозитория и его документация.

Один сервер, stdio, поднимается подпроцессом ассистента и живёт, пока открыт
он. Писать в stdout нельзя ничем, кроме протокола, поэтому логи — на WARNING.

Инструменты двух родов под одной крышей — сознательно. Ассистент берёт через
этот же сервер и то, что подкладывает в промпт сам (`git_branch`, `git_status`,
`docs_search`), и то, что модель может позвать по ходу ответа (`git_diff`,
`git_files`, `docs_read`). Путь один, и в журнале хода видно, что из контекста
дала предзагрузка, а что модель запросила сама.

Git вызывается списком аргументов, не через shell. Ревизия проверяется
регуляркой, путь — на `..` и ведущий дефис: модель передаёт поля инструмента, а
не произвольную команду. Разделы документации ищет [docs.py](docs.py).
"""

import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

import docs

ROOT = Path(__file__).resolve().parent.parent
REF = re.compile(r"^(?:HEAD(?:~\d+|\^[0-9]*)?|[A-Za-z0-9][A-Za-z0-9._/-]*)$")
PATH = re.compile(r"^[\w./-]+$", re.UNICODE)
SEP = "\x1f"

MAX_FILES = 500
MAX_DIFF_CHARS = 20000
MAX_DIFF_FILES = 60

Folder = Annotated[
    str | None,
    Field(
        description="Папка, которой ограничить список, например «day30» или «day20/servers». Без неё — весь репозиторий."
    ),
]
FilesLimit = Annotated[int, Field(description="Сколько файлов вернуть, от 1 до 500.", ge=1, le=MAX_FILES)]
Ref = Annotated[
    str | None,
    Field(
        description=(
            "С чем сравнивать рабочее дерево: ревизия вида «HEAD», «HEAD~1», хеш или ветка. "
            "Без неё — незакоммиченные изменения относительно HEAD."
        )
    ),
]
DiffPath = Annotated[
    str | None,
    Field(description="Файл или папка, которыми ограничить diff, например «day30/server.py» или «day31»."),
]
Staged = Annotated[bool, Field(description="Показать только то, что уже в индексе (git add), а не всё рабочее дерево.")]
MaxChars = Annotated[
    int,
    Field(description="Потолок длины патча в символах, от 500 до 20000. Остальное обрезается с пометкой.", ge=500, le=MAX_DIFF_CHARS),
]
Query = Annotated[
    str,
    Field(
        description=(
            "Запрос на естественном языке к документации проекта, например «как считаются токены» "
            "или «порт хранилища day20». Ищутся разделы README и TASK всех дней."
        ),
        min_length=2,
    ),
]
Paths = Annotated[
    list[str] | None,
    Field(description="Папки, которыми ограничить поиск, например [\"day20\", \"day18\"]. Без них — вся документация."),
]
SearchLimit = Annotated[int, Field(description="Сколько разделов вернуть, от 1 до 10.", ge=1, le=10)]
DocPath = Annotated[str, Field(description="Путь файла из выдачи поиска, например «day20/README.md».")]
Heading = Annotated[str, Field(description="Заголовок раздела из выдачи поиска, как он там написан.")]

mcp = MCPServer("AI Advent project", version="31.0", log_level="WARNING")


# --- git ---------------------------------------------------------------------


def _git(*args: str) -> str:
    """stdout git. Отказ git — `ToolError`, чтобы модель прочитала причину."""
    result = subprocess.run(
        ["git", "-C", str(ROOT), "--no-pager", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "git вернул ошибку").strip()
        raise ToolError(message.splitlines()[0])
    return result.stdout


def _ref(value: str) -> str:
    ref = value.strip()
    if not ref or ref.startswith("-") or ".." in ref or not REF.match(ref):
        raise ToolError(f"Непонятная ревизия: {value!r}. Ожидается хеш, ветка, тег или HEAD.")
    return ref


def _path(value: str) -> str:
    path = value.strip().strip("/")
    if not path or path.startswith("-") or ".." in path.split("/") or not PATH.match(path):
        raise ToolError(f"Непонятный путь: {value!r}. Ожидается относительный путь вида «day30/server.py».")
    return path


def _head() -> dict[str, str] | None:
    output = _git("log", "-1", f"--format=%H{SEP}%cI{SEP}%an{SEP}%s").strip()
    if not output:
        return None
    hash_, date, author, subject = output.split(SEP, 3)
    return {"hash": hash_, "short": hash_[:7], "date": date, "author": author, "subject": subject}


@mcp.tool()
def git_branch() -> dict[str, Any]:
    """Текущая ветка репозитория: имя, upstream, отставание и последний коммит.

    Это первое, что стоит знать о состоянии проекта; без ветки нельзя сказать,
    над чем сейчас идёт работа.
    """
    name = _git("rev-parse", "--abbrev-ref", "HEAD").strip()
    detached = name == "HEAD"
    payload: dict[str, Any] = {
        "branch": _git("rev-parse", "--short", "HEAD").strip() if detached else name,
        "detached": detached,
        "upstream": None,
        "ahead": None,
        "behind": None,
        "head": _head(),
    }

    if not detached:
        upstream = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if upstream.returncode == 0 and upstream.stdout.strip():
            payload["upstream"] = upstream.stdout.strip()
            counts = _git("rev-list", "--left-right", "--count", f"{payload['upstream']}...HEAD").split()
            payload["behind"], payload["ahead"] = int(counts[0]), int(counts[1])

    return payload


def _porcelain() -> dict[str, list[str]]:
    staged: list[str] = []
    modified: list[str] = []
    untracked: list[str] = []
    for line in _git("status", "--porcelain=v1", "--untracked-files=all").splitlines():
        if len(line) < 4:
            continue
        index, tree, path = line[0], line[1], line[3:]
        if index == "?" and tree == "?":
            untracked.append(path)
            continue
        if index not in " ?":
            staged.append(path)
        if tree not in " ?":
            modified.append(path)
    return {"staged": staged, "modified": modified, "untracked": untracked}


@mcp.tool()
def git_status() -> dict[str, Any]:
    """Состояние рабочего дерева: ветка, чисто ли, что в индексе, что изменено, что не отслеживается."""
    changes = _porcelain()
    return {
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "clean": not any(changes.values()),
        **changes,
        "count": sum(len(items) for items in changes.values()),
    }


@mcp.tool()
def git_files(folder: Folder = None, limit: FilesLimit = 200) -> dict[str, Any]:
    """Файлы под контролем git — весь репозиторий или одна папка, со счётчиками по папкам верхнего уровня.

    Годится, чтобы понять структуру проекта: какие папки есть, что в них лежит.
    Новые файлы, ещё не добавленные в git, тоже в списке; игнорируемые — нет.
    """
    args = ["ls-files", "--cached", "--others", "--exclude-standard", "--"]
    if folder:
        args.append(_path(folder))
    paths = sorted({line for line in _git(*args).splitlines() if line})
    if not paths:
        raise ToolError(f"В папке {folder!r} нет файлов." if folder else "В репозитории нет файлов.")

    folders = Counter(path.partition("/")[0] if "/" in path else "(корень)" for path in paths)
    return {
        "folder": folder,
        "total": len(paths),
        "folders": [{"folder": name, "files": count} for name, count in folders.most_common()],
        "files": paths[:limit],
        "truncated": len(paths) > limit,
    }


def _untracked_patch(name: str) -> str:
    """Новый файл как патч «из ничего». `--no-index` возвращает 1 при различиях — это не ошибка."""
    result = subprocess.run(
        ["git", "-C", str(ROOT), "--no-pager", "diff", "--no-index", "--", "/dev/null", name],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode not in (0, 1):
        return f"diff --git a/{name} b/{name}\n[новый файл: не читается]\n"
    return result.stdout


@mcp.tool()
def git_diff(ref: Ref = None, path: DiffPath = None, staged: Staged = False, max_chars: MaxChars = 6000) -> dict[str, Any]:
    """Что изменилось: статистика по файлам и патч — незакоммиченное или относительно ревизии.

    Без аргументов — все незакоммиченные изменения рабочего дерева. С `ref` —
    разница между этой ревизией и рабочим деревом. Патч обрезается по `max_chars`.
    """
    args = ["diff"]
    if staged:
        args.append("--cached")
    if ref:
        args.append(_ref(ref))
    scope = ["--", _path(path)] if path else []

    stat_lines = _git(*args, "--numstat", *scope).splitlines()
    files = []
    for line in stat_lines:
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        added, removed, name = parts
        files.append(
            {
                "path": name,
                "insertions": int(added) if added.isdigit() else 0,
                "deletions": int(removed) if removed.isdigit() else 0,
                "binary": not added.isdigit(),
            }
        )

    untracked = []
    if not ref and not staged:
        untracked = _porcelain()["untracked"]
        if path:
            prefix = _path(path)
            untracked = [item for item in untracked if item == prefix or item.startswith(prefix + "/")]

    if not files and not untracked:
        where = f" в {path}" if path else ""
        against = f" относительно {ref}" if ref else (" в индексе" if staged else "")
        raise ToolError(f"Изменений{where}{against} нет: рабочее дерево совпадает.")

    patch = _git(*args, *scope)
    # У нового файла diff относительно HEAD пуст — git его ещё не знает. Чтобы на
    # вопрос «что нового в дереве» можно было ответить содержанием, а не именами,
    # новые файлы показываются целиком как добавленные, пока хватает бюджета.
    for name in untracked:
        if len(patch) >= max_chars:
            break
        patch += _untracked_patch(name)
    truncated = len(patch) > max_chars
    if truncated:
        patch = patch[:max_chars] + "\n[патч обрезан]"

    return {
        "ref": ref,
        "path": path,
        "staged": staged,
        "files": files[:MAX_DIFF_FILES],
        "insertions": sum(item["insertions"] for item in files),
        "deletions": sum(item["deletions"] for item in files),
        "untracked": untracked[:MAX_DIFF_FILES],
        "patch": patch,
        "patch_truncated": truncated,
    }


# --- документация ------------------------------------------------------------


@mcp.tool()
def docs_search(query: Query, paths: Paths = None, limit: SearchLimit = 5) -> dict[str, Any]:
    """Ищет разделы документации проекта — README и TASK всех дней — по словам запроса.

    Возвращает разделы с текстом целиком, пронумерованные. Это документация, а не
    история git: про коммиты и изменения спрашивай git_diff и git_branch.
    """
    try:
        return docs.search(query, paths, limit)
    except docs.DocsError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def docs_read(path: DocPath, heading: Heading) -> dict[str, Any]:
    """Один раздел документации целиком — по пути и заголовку из выдачи docs_search."""
    try:
        return docs.read(path, heading)
    except docs.DocsError as exc:
        raise ToolError(str(exc)) from exc


if __name__ == "__main__":
    mcp.run()
