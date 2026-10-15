"""Ревью PR: diff по файлам, контекст из документации и кода, модель, текст ревью.

    python day32/review.py                                  # незакоммиченное относительно HEAD
    python day32/review.py --base main                      # ветка целиком, рабочее дерево
    python day32/review.py --base HEAD~1 --head HEAD        # один коммит
    python day32/review.py --base <sha> --head <sha> --post # в CI: ещё и комментарий в PR
    python day32/review.py --base main --cloud --out review.md --json review.json

Флаги `--local` и `--cloud` перекрывают выбор модели по доступности.

Пайплайн детерминированный: код сам решает, что модель увидит, и зовёт её по
разу на файл плюс один раз на общий вердикт. Инструментов у модели нет —
[day31](../day31/README.md) показал, что 4B-модель на них не надеется, а в CI
важнее предсказуемая цена, чем свобода манёвра.

На каждый файл модель получает четыре блока: разделы документации той папки,
в которой файл лежит; функции из других файлов, которые патч зовёт или
повторяет (поиск по идентификаторам из изменённых строк); сам файл после
изменения с номерами строк — окнами вокруг ханков, если целиком не лезет; и
патч, где строки новой версии тоже пронумерованы. Ответ — JSON по схеме:
`summary` и `findings` с родом (`bug`, `architecture`, `recommendation`),
серьёзностью, строкой, заголовком, объяснением и предложением. Markdown из
этого собирает код, а не модель — чтобы три раздела из задания были на месте
всегда, даже когда модель нашла ноль замечаний.

Бюджеты — по провайдеру: у `qwen3-4b` окно 8192 токена, у DeepSeek — на
порядок больше. Прогресс печатается в stderr, ревью — в stdout, так что
`> review.md` тоже работает.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import diff as diffmod
import index
import llm
import publish

MARKER = "<!-- ai-review -->"

PATCH_CHARS = {"local": 5000, "cloud": 14000}
FILE_CHARS = {"local": 4500, "cloud": 16000}
FILE_CONTEXT_LINES = 25
DOCS_LIMIT = {"local": 2, "cloud": 3}
DOCS_CHARS = {"local": 900, "cloud": 2000}
CODE_LIMIT = {"local": 3, "cloud": 4}
CODE_CHARS = {"local": 800, "cloud": 1800}
MAX_TOKENS = {"local": 1000, "cloud": 1600}
VERDICT_TOKENS = 300
MAX_FINDINGS = 6

KINDS = ("bug", "architecture", "recommendation")
KIND_TITLES = {"bug": "Потенциальные баги", "architecture": "Архитектурные проблемы", "recommendation": "Рекомендации"}
SEVERITIES = ("high", "medium", "low")
SEVERITY_MARKS = {"high": "🔴", "medium": "🟡", "low": "🟢"}

FINDINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(KINDS)},
                    "severity": {"type": "string", "enum": list(SEVERITIES)},
                    "line": {"type": "integer"},
                    "quote": {"type": "string"},
                    "title": {"type": "string"},
                    "detail": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": ["kind", "severity", "line", "quote", "title", "detail", "suggestion"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "findings"],
    "additionalProperties": False,
}
VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}},
    "required": ["verdict"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "Ты — ревьюер кода в проекте AI Advent: репозиторий-дневник, каждая итерация лежит в папке dayN "
    "со своим README.md и TASK.md, код дня — в его папке. Ты смотришь один файл из pull request и "
    "получаешь: разделы документации этой папки, код из других файлов, который патч использует или "
    "повторяет, сам файл после изменения и патч. Отвечай по-русски, коротко и по делу. "
    "Ответ — только JSON по схеме, без текста вокруг."
)

RULES = (
    "Правила ревью:\n"
    "1. bug — ошибка, которая проявится при выполнении: неверная логика, необработанный случай, вызов не "
    "по сигнатуре из контекста, утечка ресурса, небезопасная обработка ввода, расхождение с тем, что "
    "обещает документация.\n"
    "2. architecture — дублирование того, что уже есть в коде рядом, нарушение решения из документации, "
    "смешение ответственностей, зависимость не в ту сторону.\n"
    "3. recommendation — что сделает код надёжнее или понятнее; не стиль ради стиля.\n"
    "4. Каждое замечание указывает на строку новой версии файла: line — число слева от строки в патче "
    "или в файле, quote — дословный фрагмент этой строки (до 120 символов, без номера). Код проверит, "
    "что фрагмент в файле есть. Если замечание ко всему файлу — line = 0 и quote пустая.\n"
    "5. Замечание называет конкретное имя из патча — функцию, переменную, вызов — и говорит, что именно "
    "с ним не так. Общие советы без привязки к коду («добавить проверки», «улучшить документацию») не "
    "нужны. Если код совпадает с документацией — это не замечание.\n"
    "6. Пиши только то, что видно в патче и контексте. Не предполагай код, которого не показано. "
    "Одна проблема — одно замечание, не повторяй его для соседних строк. Не более {max_findings} "
    "замечаний, важные первыми. Если проблем нет — findings пустой: это нормальный ответ.\n"
    "7. summary — одно-два предложения: что меняет этот файл.\n"
    "8. Для файлов документации (.md) проверяй только согласованность с кодом и командами из контекста.\n"
    "Схема ответа: {{\"summary\": \"...\", \"findings\": [{{\"kind\": \"bug|architecture|recommendation\", "
    "\"severity\": \"high|medium|low\", \"line\": 0, \"quote\": \"...\", \"title\": \"...\", \"detail\": \"...\", "
    "\"suggestion\": \"...\"}}]}}"
)

VERDICT_PROMPT = (
    "Ниже — что делает каждый файл pull request и какие замечания к нему нашлись. Напиши общий вердикт "
    "в два-три предложения: что PR меняет, главный риск, если он есть, и готов ли PR к слиянию. "
    "Не перечисляй файлы. Ответ — JSON {{\"verdict\": \"...\"}}.\n\n{body}"
)


# --- результат ---------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    kind: str
    severity: str
    path: str
    line: int
    quote: str
    verified: bool
    title: str
    detail: str
    suggestion: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class FileReview:
    file: diffmod.ChangedFile
    summary: str = ""
    findings: list[Finding] = field(default_factory=list)
    docs: list[index.Hit] = field(default_factory=list)
    code: list[index.Hit] = field(default_factory=list)
    prompt_chars: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    ms: int = 0
    attempts: int = 0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.file.path,
            "status": self.file.status,
            "insertions": self.file.insertions,
            "deletions": self.file.deletions,
            "summary": self.summary,
            "findings": [finding.as_dict() for finding in self.findings],
            "context": {
                "docs": [hit.label for hit in self.docs],
                "code": [hit.label for hit in self.code],
            },
            "prompt_chars": self.prompt_chars,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "ms": self.ms,
            "attempts": self.attempts,
            "error": self.error,
        }


@dataclass
class Review:
    route: llm.Route
    diff: diffmod.Diff
    files: list[FileReview]
    verdict: str
    verdict_source: str
    index_stats: dict[str, int]
    calls: int
    total_ms: int

    @property
    def findings(self) -> list[Finding]:
        return [finding for item in self.files for finding in item.findings]

    @property
    def prompt_tokens(self) -> int:
        return sum(item.prompt_tokens or 0 for item in self.files)

    @property
    def completion_tokens(self) -> int:
        return sum(item.completion_tokens or 0 for item in self.files)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.route.model,
            "provider": self.route.provider,
            "reason": self.route.reason,
            "base": self.diff.base,
            "head": self.diff.head,
            "merge_base": self.diff.merge_base,
            "verdict": self.verdict,
            "verdict_source": self.verdict_source,
            "files": [item.as_dict() for item in self.files],
            "skipped": [{"path": item.path, "reason": item.reason} for item in self.diff.skipped],
            "index": self.index_stats,
            "calls": self.calls,
            "total_ms": self.total_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


# --- контекст файла ----------------------------------------------------------


def _numbered(lines: list[str], start: int = 1) -> str:
    return "\n".join(f"{number:>5} {line}" for number, line in enumerate(lines, start))


def file_excerpt(text: str, hunks: list[diffmod.Hunk], max_chars: int) -> str:
    """Файл с номерами строк: целиком, если лезет, иначе окнами вокруг ханков."""
    lines = text.splitlines()
    whole = _numbered(lines)
    if len(whole) <= max_chars:
        return whole

    ranges: list[list[int]] = []
    for hunk in hunks:
        first = max(1, hunk.new_start - FILE_CONTEXT_LINES)
        last = min(len(lines), hunk.new_start + max(hunk.new_count, 1) - 1 + FILE_CONTEXT_LINES)
        if ranges and first <= ranges[-1][1] + 1:
            ranges[-1][1] = max(ranges[-1][1], last)
        else:
            ranges.append([first, last])

    parts: list[str] = []
    used = 0
    for first, last in ranges:
        part = _numbered(lines[first - 1 : last], first)
        if parts and used + len(part) > max_chars:
            break
        if len(part) > max_chars:
            part = part[:max_chars] + "\n[окно обрезано]"
        parts.append(part)
        used += len(part)
    shown = len(parts)
    note = f"\n[показаны {shown} из {len(ranges)} окон вокруг изменений; всего строк в файле {len(lines)}]"
    return "\n      ⋯\n".join(parts) + note


def _hits_block(hits: list[index.Hit], prefix: str, max_chars: int) -> str:
    parts = []
    for number, hit in enumerate(hits, 1):
        body = hit.body if len(hit.body) <= max_chars else hit.body[:max_chars] + "\n[обрезано]"
        parts.append(f"[{prefix}{number}] {hit.label}\n{body}")
    return "\n\n".join(parts)


def context_for(
    item: diffmod.ChangedFile, head: str | None, provider: str
) -> tuple[list[index.Hit], list[index.Hit], str | None, list[str] | None]:
    """Документация папки, код рядом, выдержка файла после изменения для модели и его
    строки для сверки цитат."""
    tokens = index.query_tokens(item.changed_lines)
    name_tokens = index.query_tokens([Path(item.path).stem.replace("-", "_")], limit=2)
    docs_hits: list[index.Hit] = []
    if tokens or name_tokens:
        query = [*name_tokens, *tokens][:12]
        if item.day:
            docs_hits = index.search(query, kind="docs", paths=[item.day], limit=DOCS_LIMIT[provider])
        if not docs_hits:
            docs_hits = index.search(query, kind="docs", limit=DOCS_LIMIT[provider])

    code_hits = index.search(tokens, kind="code", exclude=item.path, limit=CODE_LIMIT[provider]) if tokens else []

    excerpt = None
    lines = None
    if item.status != "deleted":
        text = diffmod.file_after(item.path, head)
        if text is not None:
            lines = text.splitlines()
            # Новому файлу выдержка не нужна: патч ниже — и есть весь файл.
            if item.status != "added":
                excerpt = file_excerpt(text, item.hunks, FILE_CHARS[provider])
    return docs_hits, code_hits, excerpt, lines


def messages_for(item: diffmod.ChangedFile, docs_hits: list[index.Hit], code_hits: list[index.Hit], excerpt: str | None, provider: str) -> list[dict[str, str]]:
    status = {"added": "новый файл", "modified": "изменён", "deleted": "удалён", "renamed": "переименован"}[item.status]
    renamed = f" (был {item.old_path})" if item.old_path else ""
    header = f"Файл: {item.path} — {status}{renamed}, +{item.insertions} −{item.deletions}" + (f", папка {item.day}" if item.day else "")

    blocks = [header]
    if docs_hits:
        blocks.append("Документация этой папки:\n\n" + _hits_block(docs_hits, "D", DOCS_CHARS[provider]))
    else:
        blocks.append("Документации к этой папке не нашлось.")
    if code_hits:
        blocks.append("Код рядом — что патч использует или повторяет:\n\n" + _hits_block(code_hits, "C", CODE_CHARS[provider]))
    if excerpt is not None:
        blocks.append("Файл после изменения, строки пронумерованы:\n" + excerpt)
    elif item.status == "added":
        blocks.append("Файл новый: патч ниже — это весь файл.")
    elif item.status == "deleted":
        blocks.append("Файл удалён: патч ниже — всё, что из него ушло.")

    patch, dropped = item.patch(PATCH_CHARS[provider])
    tail = f"\n[не показано ханков: {dropped}]" if dropped else ""
    blocks.append("Патч (слева — номер строки новой версии):\n" + patch + tail)
    blocks.append(RULES.format(max_findings=MAX_FINDINGS))
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "\n\n".join(blocks)}]


# --- ревью -------------------------------------------------------------------


def _clean(value: Any, limit: int = 600) -> str:
    return " ".join(str(value or "").split())[:limit]


def _squash(text: str) -> str:
    return " ".join(text.split())


def locate(quote: str, lines: list[str], hint: int) -> int | None:
    """Строка файла, где лежит цитата модели; при нескольких — ближайшая к названной.

    Сверка та же, что у цитат в [day24](../day24/README.md): пробелы не в счёт,
    остальное — дословно. Цитата короче четырёх знаков ничего не доказывает.
    """
    needle = _squash(quote)
    if len(needle) < 4:
        return None
    found = [number for number, line in enumerate(lines, 1) if needle in _squash(line)]
    if not found:
        return None
    return min(found, key=lambda number: abs(number - hint)) if hint else found[0]


def parse_findings(data: dict[str, Any], path: str, lines: list[str] | None) -> list[Finding]:
    """Ответ модели в находки: род и серьёзность приводятся к словарю, цитата сверяется с
    файлом и уточняет номер строки, повторы и пустые — отбрасываются."""
    findings: list[Finding] = []
    seen: set[str] = set()
    for raw in data.get("findings") or []:
        if not isinstance(raw, dict):
            continue
        title, detail = _clean(raw.get("title"), 160), _clean(raw.get("detail"))
        if not (title or detail):
            continue
        # Слабая модель, разогнавшись, повторяет одно замечание для соседних строк —
        # остаётся первое с таким заголовком.
        key = (title or detail[:80]).lower()
        if key in seen:
            continue
        seen.add(key)
        kind = str(raw.get("kind", "")).lower()
        severity = str(raw.get("severity", "")).lower()
        try:
            line = max(0, int(raw.get("line") or 0))
        except (TypeError, ValueError):
            line = 0
        quote = _clean(raw.get("quote"), 160)
        verified = False
        if quote and lines is not None:
            located = locate(quote, lines, line)
            if located is not None:
                line, verified = located, True
        findings.append(
            Finding(
                kind=kind if kind in KINDS else "recommendation",
                severity=severity if severity in SEVERITIES else "medium",
                path=path,
                line=line,
                quote=quote,
                verified=verified,
                title=title or detail[:80],
                detail=detail,
                suggestion=_clean(raw.get("suggestion")),
            )
        )
    order = {name: number for number, name in enumerate(SEVERITIES)}
    findings.sort(key=lambda finding: order[finding.severity])
    return findings[:MAX_FINDINGS]


def review_file(item: diffmod.ChangedFile, head: str | None, route: llm.Route) -> FileReview:
    result = FileReview(item)
    result.docs, result.code, excerpt, lines = context_for(item, head, route.provider)
    messages = messages_for(item, result.docs, result.code, excerpt, route.provider)
    result.prompt_chars = sum(len(message["content"]) for message in messages)
    try:
        completion = llm.complete_json(
            messages, provider=route.provider, schema=FINDINGS_SCHEMA, schema_name="review", max_tokens=MAX_TOKENS[route.provider]
        )
    except llm.ModelError as exc:
        result.error = str(exc)
        return result
    result.summary = _clean(completion.data.get("summary"), 400)
    result.findings = parse_findings(completion.data, item.path, lines)
    result.prompt_tokens, result.completion_tokens = completion.prompt_tokens, completion.completion_tokens
    result.ms, result.attempts = completion.ms, completion.attempts
    return result


def _fallback_verdict(files: list[FileReview], skipped: int) -> str:
    findings = [finding for item in files for finding in item.findings]
    high = sum(finding.severity == "high" for finding in findings)
    bugs = sum(finding.kind == "bug" for finding in findings)
    parts = [f"Файлов в ревью {len(files)}" + (f", пропущено {skipped}" if skipped else "") + f"; замечаний {len(findings)}"]
    if findings:
        parts.append(f"из них серьёзных {high}, потенциальных багов {bugs}")
    failed = [item.file.path for item in files if item.error]
    if failed:
        parts.append(f"без ответа модели: {', '.join(failed)}")
    return "; ".join(parts) + "."


def verdict_for(files: list[FileReview], route: llm.Route, skipped: int = 0) -> tuple[str, str]:
    """Общий вердикт одним коротким вызовом; если модель молчит — сводка кодом."""
    lines = []
    for item in files:
        lines.append(f"- {item.file.path} ({item.file.status}, +{item.file.insertions} −{item.file.deletions}): {item.summary or item.error or 'без описания'}")
        for finding in item.findings:
            lines.append(f"    [{finding.severity} {finding.kind}] {finding.title}")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": VERDICT_PROMPT.format(body="\n".join(lines)[:12000])},
    ]
    try:
        completion = llm.complete_json(messages, provider=route.provider, schema=VERDICT_SCHEMA, schema_name="verdict", max_tokens=VERDICT_TOKENS)
        text = _clean(completion.data.get("verdict"), 800)
        if text:
            return text, "model"
    except llm.ModelError:
        pass
    return _fallback_verdict(files, skipped), "code"


def say(text: str = "") -> None:
    print(text, file=sys.stderr, flush=True)


def run(base: str, head: str | None, forced: str | None, max_files: int) -> Review:
    started = time.perf_counter()
    availability = llm.probe()
    route = llm.route(availability, forced)
    context = f" · окно {availability.local_context} токенов" if route.provider == "local" and availability.local_context else ""
    say(f"→ модель: {route.model} · {route.title} — {route.reason}{context}")

    changes = diffmod.collect(base, head, max_files=max_files)
    say(
        f"→ diff: merge-base {changes.merge_base[:7]} → {changes.head_label} · файлов {len(changes.files)}, "
        f"+{changes.insertions} −{changes.deletions}" + (f" · пропущено {len(changes.skipped)}" if changes.skipped else "")
    )
    stats = index.reindex()
    say(
        f"→ индекс: документация {stats['docs_files']} файлов / {stats['docs_chunks']} разделов, "
        f"код {stats['code_files']} файлов / {stats['code_chunks']} чанков" + (f" · переиндексировано {stats['reindexed']}" if stats["reindexed"] else "")
    )

    files: list[FileReview] = []
    calls = 0
    for number, item in enumerate(changes.files, 1):
        say(f"  [{number}/{len(changes.files)}] {item.path} …")
        result = review_file(item, changes.head, route)
        calls += max(result.attempts, 1)
        files.append(result)
        if result.error:
            say(f"      ✗ {result.error}")
        else:
            tokens = f" · токены {result.prompt_tokens} → {result.completion_tokens}" if result.prompt_tokens else ""
            retry = " · повтор" if result.attempts > 1 else ""
            verified = sum(1 for finding in result.findings if finding.verified)
            say(
                f"      замечаний {len(result.findings)} (цитата сошлась у {verified}) · docs {len(result.docs)} · "
                f"code {len(result.code)} · {result.ms / 1000:.1f} с{tokens}{retry}"
            )

    verdict, source = verdict_for(files, route, len(changes.skipped))
    calls += 1
    say(f"→ вердикт: {'модель' if source == 'model' else 'сводка кодом'} · всего вызовов {calls}")

    return Review(route, changes, files, verdict, source, stats, calls, round((time.perf_counter() - started) * 1000))


# --- markdown ----------------------------------------------------------------


def _blob_url(review: Review) -> str | None:
    repo = os.environ.get("GITHUB_REPOSITORY")
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    if repo and review.diff.head:
        return f"{server}/{repo}/blob/{review.diff.head}"
    return None


def _place(finding: Finding, blob: str | None) -> str:
    where = f"{finding.path}:{finding.line}" if finding.line else finding.path
    if blob:
        anchor = f"#L{finding.line}" if finding.line else ""
        return f"[`{where}`]({blob}/{finding.path}{anchor})"
    return f"`{where}`"


def render(review: Review) -> str:
    blob = _blob_url(review)
    changes = review.diff
    out = [MARKER, f"## AI-ревью · {review.route.model}", ""]
    head = f"`{changes.head[:7]}`" if changes.head else "рабочее дерево"
    out.append(
        f"`{changes.base}` (merge-base `{changes.merge_base[:7]}`) → {head} · файлов {len(changes.files)}, "
        f"+{changes.insertions} −{changes.deletions} · замечаний {len(review.findings)}"
    )
    out += ["", f"**Вердикт.** {review.verdict}", ""]

    order = {name: number for number, name in enumerate(SEVERITIES)}
    for kind in KINDS:
        found = sorted((finding for finding in review.findings if finding.kind == kind), key=lambda finding: order[finding.severity])
        out.append(f"### {KIND_TITLES[kind]} ({len(found)})")
        out.append("")
        if not found:
            out.append("— ничего не найдено.")
        for finding in found:
            line = f"- {SEVERITY_MARKS[finding.severity]} {_place(finding, blob)} — **{finding.title}**"
            if finding.detail and finding.detail != finding.title:
                line += f". {finding.detail}"
            if finding.suggestion:
                line += f" → {finding.suggestion}"
            if finding.quote and not finding.verified:
                line += " <sub>цитата в файле не найдена</sub>"
            out.append(line)
        out.append("")

    out += ["### Файлы", "", "| Файл | Изменение | Что делает | Замечаний | Контекст |", "| --- | --- | --- | --- | --- |"]
    for item in review.files:
        what = item.summary or (f"✗ {item.error[:120]}" if item.error else "—")
        what = what.replace("|", "\\|")
        change = {"added": "новый", "deleted": "удалён", "renamed": "переименован", "modified": "изменён"}[item.file.status]
        out.append(
            f"| `{item.file.path}` | {change}, +{item.file.insertions} −{item.file.deletions} | {what} | {len(item.findings)} | "
            f"docs {len(item.docs)} · code {len(item.code)} |"
        )
    if changes.skipped:
        out += ["", "Не ревьюились: " + "; ".join(f"`{item.path}` — {item.reason}" for item in changes.skipped) + "."]

    tokens = f" · токены {review.prompt_tokens} → {review.completion_tokens}" if review.prompt_tokens else ""
    quoted = sum(1 for finding in review.findings if finding.quote)
    verified = sum(1 for finding in review.findings if finding.verified)
    out += [
        "",
        f"<sub>{review.route.model} · {review.route.title} — {review.route.reason} · вызовов {review.calls} · "
        f"{review.total_ms / 1000:.0f} с{tokens} · цитат подтверждено {verified} из {quoted} · "
        f"индекс: документация {review.index_stats['docs_files']} файлов, код {review.index_stats['code_files']} файлов</sub>",
        "",
    ]
    return "\n".join(out)


# --- запуск ------------------------------------------------------------------


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="AI-ревью изменений: diff, RAG по документации и коду, текст ревью.")
    parser.add_argument("--base", default="HEAD", help="база сравнения: ветка, тег или хеш (по умолчанию HEAD)")
    parser.add_argument("--head", default=None, help="голова: ревизия PR; без неё — рабочее дерево")
    parser.add_argument("--out", default=None, help="куда записать markdown ревью")
    parser.add_argument("--json", dest="json_path", default=None, help="куда записать ревью в JSON")
    parser.add_argument("--post", action="store_true", help="опубликовать комментарием в PR (GITHUB_TOKEN, событие pull_request)")
    parser.add_argument("--max-files", type=int, default=diffmod.MAX_FILES, help=f"сколько файлов ревьюить, по умолчанию {diffmod.MAX_FILES}")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--local", action="store_const", const="local", dest="forced", help="только LM Studio")
    group.add_argument("--cloud", action="store_const", const="cloud", dest="forced", help="только DeepSeek")
    args = parser.parse_args(argv)

    try:
        review = run(args.base, args.head, args.forced, args.max_files)
    except (llm.ModelError, diffmod.DiffError) as exc:
        say(f"✗ {exc}")
        return 2

    text = render(review)
    print(text, flush=True)

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        say(f"→ записано: {args.out}")
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(review.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        say(f"→ записано: {args.json_path}")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")
        say("→ записано в step summary")
    if args.post:
        try:
            url = publish.comment(text, marker=MARKER)
        except publish.PublishError as exc:
            say(f"✗ комментарий не опубликован: {exc}")
        else:
            say(f"→ комментарий в PR: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
