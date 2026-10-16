"""Песочница файлового ассистента: эталон, рабочая копия, jail и diff.

Ассистент видит только `day34/workspace/`. Эталон — `day34/fixture/`. Сброс
стирает рабочую копию и копирует эталон заново. Все пути резолвятся
относительно корня workspace; `..` и абсолютные пути — ошибка.
"""

from __future__ import annotations

import re
import shutil
from difflib import unified_diff
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixture"
WORKSPACE = HERE / "workspace"

MAX_BODY_CHARS = 60000
MAX_READ_CHARS = 12000
MAX_DIFF_CHARS = 12000
SEARCH_CONTEXT = 1
MAX_HITS = 40
TEXT_SUFFIXES = {".md", ".py", ".txt", ".toml", ".json", ".cfg", ".ini"}
SKIP_DIR_NAMES = {"__pycache__"}
SKIP_FILE_NAMES = {".DS_Store"}


class WorkspaceError(Exception):
    """Ошибка, которую MCP отдаёт модели как ToolError."""


def _is_text(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def _skip(path: Path, root: Path) -> bool:
    parts = path.relative_to(root).parts
    if any(part in SKIP_DIR_NAMES for part in parts[:-1]):
        return True
    return path.name in SKIP_FILE_NAMES


def _collect(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _skip(path, root) or not _is_text(path):
            continue
        files.append(path)
    return files


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def fixture_names() -> list[str]:
    return [_rel(path, FIXTURE) for path in _collect(FIXTURE)]


def ensure() -> Path:
    """Рабочая копия есть: если папки нет или она пуста — сбросить из эталона."""
    if not WORKSPACE.is_dir() or not any(path for path in WORKSPACE.iterdir() if path.name != ".DS_Store"):
        reset()
    return WORKSPACE


def reset() -> dict[str, Any]:
    if not FIXTURE.is_dir():
        raise WorkspaceError(f"Эталона нет: {FIXTURE}")
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    shutil.copytree(
        FIXTURE,
        WORKSPACE,
        ignore=shutil.ignore_patterns("__pycache__", ".DS_Store", "*.pyc"),
    )
    files = list_files()
    return {"reset": True, "files": len(files), "paths": [item["path"] for item in files]}


def resolve(rel: str) -> Path:
    raw = (rel or "").strip().replace("\\", "/")
    if not raw or raw.startswith("/") or raw.startswith("~"):
        raise WorkspaceError(f"Путь {rel!r} должен быть относительным внутри workspace.")
    parts = [part for part in raw.split("/") if part and part != "."]
    if not parts or any(part == ".." for part in parts):
        raise WorkspaceError(f"Путь {rel!r} выходит за workspace.")
    root = WORKSPACE.resolve()
    path = root.joinpath(*parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise WorkspaceError(f"Путь {rel!r} выходит за workspace.") from exc
    return path


def list_files() -> list[dict[str, Any]]:
    ensure()
    return [{"path": _rel(path, WORKSPACE), "bytes": path.stat().st_size} for path in _collect(WORKSPACE)]


def names() -> list[str]:
    return [item["path"] for item in list_files()]


def tree() -> dict[str, Any]:
    files = list_files()
    return {"root": "workspace", "count": len(files), "files": files}


def read(rel: str, max_chars: int = MAX_READ_CHARS) -> dict[str, Any]:
    ensure()
    path = resolve(rel)
    if not path.is_file():
        known = ", ".join(names()) or "(пусто)"
        raise WorkspaceError(f"Файла {rel!r} нет. Есть: {known}.")
    if not _is_text(path):
        raise WorkspaceError(f"Файл {rel!r} не текстовый.")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise WorkspaceError(f"Файл {rel!r} не utf-8.") from exc
    numbered = "\n".join(f"{index:>4}|{line}" for index, line in enumerate(text.splitlines(), 1))
    truncated = len(numbered) > max_chars
    if truncated:
        numbered = numbered[:max_chars] + "\n[обрезано]"
    lines = 0 if not text else text.count("\n") + (0 if text.endswith("\n") else 1)
    return {
        "path": _rel(path, WORKSPACE),
        "bytes": path.stat().st_size,
        "lines": lines,
        "truncated": truncated,
        "content": numbered,
    }


def write(rel: str, content: str) -> dict[str, Any]:
    ensure()
    if len(content) > MAX_BODY_CHARS:
        raise WorkspaceError(f"Текст длиннее {MAX_BODY_CHARS} символов.")
    if "\x00" in content:
        raise WorkspaceError("Бинарный текст писать нельзя.")
    path = resolve(rel)
    if path.suffix.lower() not in TEXT_SUFFIXES:
        allowed = ", ".join(sorted(TEXT_SUFFIXES))
        raise WorkspaceError(f"Писать можно только текстовые файлы: {allowed}.")
    created = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    patch = file_diff(_rel(path, WORKSPACE))
    return {
        "path": _rel(path, WORKSPACE),
        "created": created,
        "bytes": len(content.encode()),
        "changed": patch["changed"],
        "diff": patch["diff"],
    }


def search(
    query: str,
    path: str | None = None,
    regex: bool = False,
    limit: int = MAX_HITS,
) -> dict[str, Any]:
    ensure()
    compiled: re.Pattern[str] | None = None
    if regex:
        try:
            compiled = re.compile(query)
        except re.error as exc:
            raise WorkspaceError(f"Некорректное регулярное выражение: {exc}") from exc

    if path:
        target = resolve(path)
        if target.is_file():
            scope = [target]
        elif target.is_dir():
            scope = [item for item in _collect(target)]
        else:
            raise WorkspaceError(f"Пути {path!r} нет.")
    else:
        scope = _collect(WORKSPACE)

    hits: list[dict[str, Any]] = []
    for file in scope:
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(lines, 1):
            found = compiled.search(line) is not None if compiled else query in line
            if not found:
                continue
            start = max(1, number - SEARCH_CONTEXT)
            end = min(len(lines), number + SEARCH_CONTEXT)
            hits.append(
                {
                    "path": _rel(file, WORKSPACE),
                    "line": number,
                    "text": line,
                    "context": [{"line": index, "text": lines[index - 1]} for index in range(start, end + 1)],
                }
            )
            if len(hits) >= limit:
                return {"query": query, "regex": regex, "count": len(hits), "truncated": True, "hits": hits}
    return {"query": query, "regex": regex, "count": len(hits), "truncated": False, "hits": hits}


def _text_or_none(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def file_diff(rel: str) -> dict[str, Any]:
    return diff(rel)


def diff(rel: str | None = None, max_chars: int = MAX_DIFF_CHARS) -> dict[str, Any]:
    ensure()
    if rel:
        names_to_compare = [rel]
    else:
        names_to_compare = sorted(set(names()) | set(fixture_names()))

    chunks: list[str] = []
    changed: list[str] = []
    for name in names_to_compare:
        live = _text_or_none(WORKSPACE / name)
        gold = _text_or_none(FIXTURE / name)
        if live == gold:
            continue
        changed.append(name)
        before = gold.splitlines() if gold is not None else []
        after = live.splitlines() if live is not None else []
        fromfile = f"a/{name}" if gold is not None else "/dev/null"
        tofile = f"b/{name}" if live is not None else "/dev/null"
        lines = list(unified_diff(before, after, fromfile=fromfile, tofile=tofile, lineterm=""))
        chunks.append("\n".join(lines))

    text = "\n".join(chunk for chunk in chunks if chunk)
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars] + "\n[diff обрезан]"
    return {
        "changed": changed,
        "count": len(changed),
        "truncated": truncated,
        "diff": text or "(нет отличий от эталона)",
    }


def stats() -> dict[str, Any]:
    ensure()
    files = list_files()
    delta = diff()
    return {
        "files": len(files),
        "paths": [item["path"] for item in files],
        "changed": delta["changed"],
        "dirty": bool(delta["changed"]),
    }


def read_plain(rel: str) -> str:
    """Текст без номеров строк — для проверок сценариев, не для модели."""
    ensure()
    path = resolve(rel)
    if not path.is_file():
        raise WorkspaceError(f"Файла {rel!r} нет.")
    return path.read_text(encoding="utf-8")
