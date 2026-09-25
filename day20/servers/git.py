"""MCP-сервер над git этого репозитория: поиск по коммитам, журнал, коммит, файлы.

Один из трёх серверов реестра day20. Транспорт — stdio: состояния между
запросами у него нет, поднимается подпроцессом на время вызова. Писать в stdout
нельзя ничем, кроме протокола.

Инструмент `search` назван так же, как у сервера docs, и это не случайность:
реестр обязан развести одноимённые инструменты разных серверов, а модель —
выбрать нужный. Здесь поиск идёт по сообщениям коммитов, там — по документации.

`files` принимает список хешей от `search` или `log` и сводит, что эти коммиты
трогали: файлы, папки верхнего уровня и папки дней.
"""

import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

ROOT = Path(__file__).resolve().parent.parent.parent
# Хеш, ветка, тег, HEAD, HEAD~2, origin/main. Не диапазон и не флаг.
REF = re.compile(r"^(?:HEAD(?:~\d+|\^[0-9]*)?|[A-Za-z0-9][A-Za-z0-9._/-]*)$")
DAY_DIR = re.compile(r"^day(\d+)$")
WORD = re.compile(r"[^\W_]{2,}", re.UNICODE)
SEP = "\x1f"

MAX_QUERY_WORDS = 6
MAX_COMMITS_IN_FILES = 20
SUBJECT_CHARS = 90

Query = Annotated[
    str,
    Field(
        description=(
            "Слова, которые ищутся в сообщениях коммитов, например «пайплайн MCP» "
            "или «память агента». Это поиск по истории репозитория, а не по документации."
        ),
        min_length=2,
    ),
]
Limit = Annotated[
    int,
    Field(description="Сколько коммитов вернуть, от 1 до 30.", ge=1, le=30),
]
Ref = Annotated[
    str,
    Field(description="Ревизия: хеш, ветка, тег или HEAD, например «HEAD» или «abc1234»."),
]
Hashes = Annotated[
    list[str],
    Field(
        description="Хеши коммитов — те, что вернули search или log.",
        min_length=1,
        max_length=MAX_COMMITS_IN_FILES,
    ),
]

# stdio-сервер пишет логи в stderr клиента: на уровне INFO там оказывались и
# ожидаемые отказы инструментов, вперемешку с выводом флоу.
mcp = MCPServer("AI Advent git", version="20.0", log_level="WARNING")


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
        raise ToolError(f"Непонятная ревизия: {value!r}. Ожидается хеш, ветка, тег или HEAD.")
    return ref


def _commit(line: str) -> dict[str, str]:
    hash_, date, author, subject = line.split(SEP, 3)
    return {
        "hash": hash_,
        "short": hash_[:7],
        "date": date,
        "author": author,
        "subject": subject,
    }


def _commits(*args: str) -> list[dict[str, str]]:
    output = _git("log", f"--format=%H{SEP}%cI{SEP}%an{SEP}%s", *args)
    return [_commit(line) for line in output.splitlines() if line]


def _folder(path: str) -> str:
    head, sep, _ = path.strip('"').partition("/")
    return head if sep else "(корень)"


def _payload(commits: list[dict[str, str]], **extra: Any) -> dict[str, Any]:
    """Хеши и темы отдельными списками: их подставляет реестр в следующий вызов."""
    return {
        **extra,
        "count": len(commits),
        "commits": commits,
        "hashes": [commit["hash"] for commit in commits],
        "subjects": [f"{commit['short']} {commit['subject'][:SUBJECT_CHARS]}" for commit in commits],
    }


@mcp.tool()
def search(query: Query, limit: Limit = 10) -> dict[str, Any]:
    """Ищет коммиты по словам в их сообщениях: хеш, дата, автор, тема.

    Это история изменений репозитория. Если нужен текст документации проекта,
    инструмент называется так же, но у сервера docs.
    """
    words = WORD.findall(query.lower())[:MAX_QUERY_WORDS]
    if not words:
        raise ToolError(f"В запросе {query!r} нет слов для поиска по коммитам.")

    # Сначала коммиты со всеми словами, потом с любым: строгий поиск точнее, но часто пуст.
    for strict in (True, False):
        greps = [f"--grep={word}" for word in words]
        found = _commits(
            f"-n{limit}", "--regexp-ignore-case", *(["--all-match"] if strict else []), *greps
        )
        if found:
            return _payload(
                found,
                query=query,
                words=words,
                strategy="все слова" if strict else "любое слово",
            )

    # Сообщения коммитов в этом репозитории английские, а запрос приходит русским:
    # поэтому в отказе перечислены темы последних коммитов — по ним видно, какими
    # словами тут вообще можно искать, и что вместо поиска есть log.
    total = _git("rev-list", "--count", "HEAD").strip()
    recent = ", ".join(commit["subject"][:40] for commit in _commits("-n3"))
    raise ToolError(
        f"Коммитов со словами {', '.join(words)} в истории нет. Всего коммитов: {total}, "
        f"последние темы: {recent}. Если тема запроса по-русски, ищи по латинским словам "
        "из неё или возьми последние коммиты инструментом git__log."
    )


@mcp.tool()
def log(limit: Limit = 10) -> dict[str, Any]:
    """Последние коммиты репозитория по порядку, без фильтра по словам."""
    found = _commits(f"-n{limit}")
    if not found:
        raise ToolError("В репозитории нет коммитов.")
    return _payload(found, query=None, strategy="последние")


@mcp.tool()
def show(ref: Ref = "HEAD") -> dict[str, Any]:
    """Один коммит целиком: сообщение и изменённые файлы со статистикой."""
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
        "short": hash_[:7],
        "date": date,
        "author": author,
        "subject": subject,
        "body": body.strip(),
        "files": files,
    }


@mcp.tool()
def files(commits: Hashes) -> dict[str, Any]:
    """Что трогали эти коммиты: файлы со статистикой, папки и папки дней.

    На входе хеши от search или log. Папки дней отсюда годятся в фильтр
    поиска по документации: искать разделы там, где менялся код.
    """
    stats: dict[str, dict[str, int]] = {}
    seen: list[str] = []

    for value in commits[:MAX_COMMITS_IN_FILES]:
        name = _ref(value)
        seen.append(name[:7])
        for line in _git(
            "diff-tree", "--no-commit-id", "--numstat", "--root", "-r", name
        ).splitlines():
            added, removed, path = line.split("\t", 2)
            slot = stats.setdefault(path, {"commits": 0, "insertions": 0, "deletions": 0})
            slot["commits"] += 1
            # Бинарный файл git показывает дефисами вместо чисел.
            slot["insertions"] += int(added) if added.isdigit() else 0
            slot["deletions"] += int(removed) if removed.isdigit() else 0

    if not stats:
        raise ToolError(f"Коммиты {', '.join(seen)} не меняли ни одного файла.")

    folders = Counter()
    for path, slot in stats.items():
        folders[_folder(path)] += slot["commits"]

    days = sorted(
        (folder for folder in folders if DAY_DIR.match(folder)),
        key=lambda folder: int(DAY_DIR.match(folder).group(1)),
    )

    # `filter` — готовый список папок для поиска по документации: дни, а если
    # коммиты их не трогали — всё, кроме корня. Решает сервер, а не реестр.
    named = [folder for folder, _ in folders.most_common() if folder != "(корень)"]

    return {
        "commits": seen,
        "files": [
            {"path": path, **slot}
            for path, slot in sorted(stats.items(), key=lambda item: -item[1]["commits"])
        ],
        "folders": [folder for folder, _ in folders.most_common()],
        "days": days,
        "filter": days or named,
        "insertions": sum(slot["insertions"] for slot in stats.values()),
        "deletions": sum(slot["deletions"] for slot in stats.values()),
    }


if __name__ == "__main__":
    mcp.run()
