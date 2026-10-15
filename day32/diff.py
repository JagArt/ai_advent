"""Что изменилось в PR: diff между базой и головой, разобранный по файлам и ханкам.

Ревью смотрит на патч, а не на весь репозиторий, и для этого патч нужен в
структуре: какие файлы, с каким статусом, какие ханки и на каких строках
новой версии. Номера строк нужны модели, чтобы замечание указывало на место, а
не на «где-то в файле», и нужны рендеру, чтобы собрать `path:line`.

Сравнение всегда идёт от merge-base: `git diff base...head` — то, что принёс
PR, а не то, чем база успела уйти вперёд. Без `head` сравнивается рабочее
дерево — так пайплайн прогоняется локально до коммита, и новые файлы, ещё не
добавленные в git, показываются целиком как добавленные, как в
[day31/server.py](../day31/server.py).

Git вызывается списком аргументов, не через shell; ревизии проверяются
регуляркой — они приходят из аргументов командной строки и из события PR.
"""

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REF = re.compile(r"^(?:HEAD(?:~\d+|\^\d*)?|[A-Za-z0-9][A-Za-z0-9._/-]*)$")
DIFF_HEADER = re.compile(r"^diff --git a/(.+?) b/(.+)$")
HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")
DAY_DIR = re.compile(r"^day\d+$")

SKIP_SUFFIXES = {".db", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".pyc", ".woff", ".woff2", ".zip"}
SKIP_NAMES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock", "Pipfile.lock"}
LARGE_JSON_CHARS = 20_000
MAX_FILES = 20
MAX_FILE_CHARS = 200_000

# Файлы с кодом идут в ревью первыми: когда PR больше лимита, лучше потерять
# markdown, чем питон.
PRIORITY = {".py": 0, ".js": 1, ".ts": 1, ".html": 2, ".css": 3, ".yml": 4, ".yaml": 4, ".toml": 4, ".txt": 5, ".md": 6}


class DiffError(RuntimeError):
    """Отказ с причиной: непонятная ревизия, git вернул ошибку, изменений нет."""


def _git(*args: str, ok_codes: tuple[int, ...] = (0,)) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "--no-pager", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode not in ok_codes:
        message = (result.stderr or result.stdout or "git вернул ошибку").strip()
        raise DiffError(message.splitlines()[0] if message else "git вернул ошибку")
    return result.stdout


def _ref(value: str) -> str:
    ref = value.strip()
    if not ref or ref.startswith("-") or ".." in ref or not REF.match(ref):
        raise DiffError(f"Непонятная ревизия: {value!r}. Ожидается хеш, ветка, тег или HEAD.")
    return ref


def short(rev: str) -> str:
    return _git("rev-parse", "--short", rev).strip()


# --- структура патча ---------------------------------------------------------


@dataclass(frozen=True)
class Hunk:
    """Один `@@`-блок: диапазоны старой и новой версии и строки с префиксами."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section: str
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        header = f"@@ -{self.old_start},{self.old_count} +{self.new_start},{self.new_count} @@"
        if self.section:
            header += f" {self.section}"
        return "\n".join((header, *self.lines))

    def numbered(self) -> str:
        """Тот же ханк, но у строк новой версии проставлен номер — модели так проще
        назвать строку, чем считать от заголовка."""
        out = [f"@@ -{self.old_start},{self.old_count} +{self.new_start},{self.new_count} @@" + (f" {self.section}" if self.section else "")]
        number = self.new_start
        for line in self.lines:
            if line.startswith("-"):
                out.append(f"      {line}")
            elif line.startswith("\\"):
                out.append(f"      {line}")
            else:
                out.append(f"{number:>5} {line}")
                number += 1
        return "\n".join(out)

    @property
    def added(self) -> list[str]:
        return [line[1:] for line in self.lines if line.startswith("+")]

    @property
    def removed(self) -> list[str]:
        return [line[1:] for line in self.lines if line.startswith("-")]


@dataclass
class ChangedFile:
    path: str
    status: str  # added | modified | deleted | renamed
    old_path: str | None = None
    binary: bool = False
    insertions: int = 0
    deletions: int = 0
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def suffix(self) -> str:
        return Path(self.path).suffix.lower()

    @property
    def day(self) -> str | None:
        top = self.path.partition("/")[0]
        return top if DAY_DIR.match(top) else None

    @property
    def changed_lines(self) -> list[str]:
        return [line for hunk in self.hunks for line in (*hunk.added, *hunk.removed)]

    def patch(self, max_chars: int | None = None, numbered: bool = True) -> tuple[str, int]:
        """Патч файла в бюджет: ханки целиком, пока хватает места. Возвращает текст и
        число ханков, которые не вошли."""
        parts: list[str] = []
        used = 0
        for index, hunk in enumerate(self.hunks):
            text = hunk.numbered() if numbered else hunk.text
            if max_chars is not None and parts and used + len(text) > max_chars:
                return "\n".join(parts), len(self.hunks) - index
            if max_chars is not None and len(text) > max_chars:
                text = text[:max_chars] + "\n[ханк обрезан]"
            parts.append(text)
            used += len(text) + 1
        return "\n".join(parts), 0


@dataclass(frozen=True)
class Skipped:
    path: str
    reason: str


@dataclass
class Diff:
    base: str
    head: str | None
    merge_base: str
    files: list[ChangedFile]
    skipped: list[Skipped]

    @property
    def head_label(self) -> str:
        return "рабочее дерево" if self.head is None else self.head[:7]

    @property
    def insertions(self) -> int:
        return sum(item.insertions for item in self.files)

    @property
    def deletions(self) -> int:
        return sum(item.deletions for item in self.files)


# --- разбор ------------------------------------------------------------------


def parse(patch: str) -> list[ChangedFile]:
    """Unified diff в список файлов. Заголовки файла читаются до первого `@@`."""
    files: list[ChangedFile] = []
    current: ChangedFile | None = None
    hunk: dict | None = None

    def close_hunk() -> None:
        nonlocal hunk
        if hunk is not None and current is not None:
            current.hunks.append(Hunk(**{**hunk, "lines": tuple(hunk["lines"])}))
        hunk = None

    for line in patch.splitlines():
        header = DIFF_HEADER.match(line)
        if header:
            close_hunk()
            old, new = header.group(1), header.group(2)
            current = ChangedFile(path=new, status="modified", old_path=old if old != new else None)
            files.append(current)
            continue
        if current is None:
            continue

        if hunk is None:
            if line.startswith("new file mode"):
                current.status = "added"
            elif line.startswith("deleted file mode"):
                current.status = "deleted"
                current.path = current.old_path or current.path
                current.old_path = None
            elif line.startswith("rename from "):
                current.status = "renamed"
                current.old_path = line[len("rename from "):]
            elif line.startswith("rename to "):
                current.path = line[len("rename to "):]
            elif line.startswith("Binary files") or line.startswith("GIT binary patch"):
                current.binary = True

        match = HUNK_HEADER.match(line)
        if match:
            close_hunk()
            hunk = {
                "old_start": int(match.group(1)),
                "old_count": int(match.group(2) or 1),
                "new_start": int(match.group(3)),
                "new_count": int(match.group(4) or 1),
                "section": match.group(5).strip(),
                "lines": [],
            }
            continue
        if hunk is not None and (line == "" or line[:1] in (" ", "+", "-", "\\")):
            # Пустая строка контекста у git — один пробел, но некоторые патчи
            # приходят с голой пустой строкой: без неё поедет нумерация.
            hunk["lines"].append(line or " ")
            if line.startswith("+"):
                current.insertions += 1
            elif line.startswith("-"):
                current.deletions += 1

    close_hunk()
    return files


def _untracked() -> list[str]:
    return [line for line in _git("ls-files", "--others", "--exclude-standard").splitlines() if line]


def _untracked_patch(name: str) -> str:
    """Новый файл как патч «из ничего». `--no-index` возвращает 1 при различиях."""
    return _git("diff", "--no-color", "--no-index", "--", "/dev/null", name, ok_codes=(0, 1))


def file_after(path: str, head: str | None) -> str | None:
    """Файл после изменения: из ревизии `head` или с диска. Нет файла — None."""
    if head is None:
        target = ROOT / path
        if not target.is_file():
            return None
        try:
            return target.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    try:
        return _git("show", f"{head}:{path}")
    except DiffError:
        return None


def _skip_reason(item: ChangedFile, head: str | None) -> str | None:
    name = Path(item.path).name
    if item.binary or item.suffix in SKIP_SUFFIXES:
        return "бинарный файл"
    if name in SKIP_NAMES:
        return "lock-файл"
    if not item.hunks:
        return "без изменений в тексте" if item.status != "renamed" else "переименование без правок"
    if item.suffix == ".json" and item.status != "deleted":
        content = file_after(item.path, head)
        if content is not None and len(content) > LARGE_JSON_CHARS:
            return f"JSON больше {LARGE_JSON_CHARS // 1000} КБ — данные, не код"
    return None


def collect(base: str, head: str | None = None, max_files: int = MAX_FILES) -> Diff:
    """Diff PR: от merge-base базы и головы до головы или до рабочего дерева."""
    base = _ref(base)
    head = _ref(head) if head else None
    merge_base = _git("merge-base", base, head or "HEAD").strip()

    args = ["diff", "--no-color", "--no-ext-diff", "-M", "-U3", merge_base]
    if head:
        args.append(head)
    patch = _git(*args)
    if head is None:
        for name in _untracked():
            patch += _untracked_patch(name)

    files = parse(patch)
    if not files:
        where = "рабочее дерево совпадает" if head is None else f"{head[:7]} совпадает"
        raise DiffError(f"Изменений относительно {base} нет: {where} с merge-base {merge_base[:7]}.")

    files.sort(key=lambda item: (PRIORITY.get(item.suffix, 5), item.path))
    kept: list[ChangedFile] = []
    skipped: list[Skipped] = []
    for item in files:
        reason = _skip_reason(item, head)
        if reason:
            skipped.append(Skipped(item.path, reason))
        elif len(kept) >= max_files:
            skipped.append(Skipped(item.path, f"сверх лимита {max_files} файлов на ревью"))
        else:
            kept.append(item)

    return Diff(base=base, head=head, merge_base=merge_base, files=kept, skipped=skipped)


if __name__ == "__main__":
    import sys

    found = collect(sys.argv[1] if len(sys.argv) > 1 else "HEAD", sys.argv[2] if len(sys.argv) > 2 else None)
    print(f"merge-base {found.merge_base[:7]} → {found.head_label}: файлов {len(found.files)}, +{found.insertions} −{found.deletions}")
    for item in found.files:
        print(f"  {item.status:8} {item.path}  ханков {len(item.hunks)}  +{item.insertions} −{item.deletions}")
    for item in found.skipped:
        print(f"  пропущен {item.path} — {item.reason}")
