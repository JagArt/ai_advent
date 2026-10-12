"""Контрольный набор обоим профилям одной модели, по три прогона на вопрос, вывод в markdown.

Оси те же, что в day28, — качество, скорость, стабильность, — но сравниваются не
две модели, а две настройки одной: `base` (ровно day28) и `optimized`. Поиск
общий и детерминированный; разница в контексте — только бюджет оптимизированного
профиля, и он сам часть того, что сравнивается.

Качество — доля обязательных подстрок `facts` в ответе, отказ на вопросах «вне
базы», ложный отказ на вопросах «в базе», доля ответов со ссылкой [n] и — новое
для этого дня — доля ответов с латиницей вне кода и имён: это ловит английские
слова посреди русской фразы, которые 4B вставляла в day28.

Стабильность — разброс доли фактов и времени между прогонами, вопросы, где
решение «ответить или отказаться» разошлось, обрывы по `max_tokens`, ошибки.

    python day29/scenarios.py                       # оба профиля, 10 вопросов × 3
    python day29/scenarios.py --profiles optimized  # только оптимизированный
    python day29/scenarios.py --runs 1              # по одному прогону
    python day29/scenarios.py ask "вопрос"          # один вопрос обоим профилям, с контекстом
    python day29/scenarios.py load --context 4096   # перезагрузить модель с этим окном
    python day29/scenarios.py load --context 4096 --parallel 1 --checkpoints 0   # один слот, без снимков кэша

Скорость у локальной модели зависит от кэша промптов LM Studio сильнее, чем от
чего бы то ни было: уже виденный промпт читается за доли секунды вместо
секунд, и кэш переживает и смену вопроса, и перезапуск этого скрипта — его
сбрасывает только перезагрузка модели. Поэтому таблица скорости показывает
первый токен отдельно для первого прогона и для повторов, а честные «холодные»
числа получаются, если перед прогоном выполнить `load`. Профили чередуются на
каждом вопросе, чтобы оба были в равных условиях.
"""

import argparse
import asyncio
import re
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

import llm
import lmstudio
import profiles
import questions
import search
from profiles import Profile
from questions import Question

CITATION = re.compile(r"\[(\d+)\]")
LATIN_WORD = re.compile(r"(?<![\w`/.-])[A-Za-z]{3,}(?![\w`/.-])")
CODE_SPAN = re.compile(r"`[^`]*`")


def latin_outside_names(text: str, block: str) -> list[str]:
    """Латинские слова ответа, которых нет ни в коде, ни в выдержках: это и есть «whereas»."""
    plain = CODE_SPAN.sub(" ", text)
    allowed = set(word.lower() for word in re.findall(r"[A-Za-z]{3,}", block))
    return [word for word in LATIN_WORD.findall(plain) if word.lower() not in allowed]


@dataclass(frozen=True)
class Run:
    question: Question
    profile: str
    done: llm.Done | None
    block: str = ""
    error: str | None = None
    attempt: int = 0

    @property
    def text(self) -> str:
        return self.done.text if self.done else ""

    @property
    def facts_share(self) -> float | None:
        if self.question.out_of_base or self.done is None:
            return None
        if not self.question.facts:
            return None
        lowered = self.text.lower()
        found = sum(fact.lower() in lowered for fact in self.question.facts)
        return found / len(self.question.facts)

    @property
    def refused(self) -> bool:
        return bool(self.done and self.done.refused)

    @property
    def cited(self) -> bool:
        return bool(CITATION.search(self.text))

    @property
    def truncated(self) -> bool:
        return bool(self.done and self.done.finish_reason == "length")

    @property
    def foreign(self) -> list[str]:
        return latin_outside_names(self.text, self.block) if self.done else []


@dataclass
class Summary:
    profile: str
    runs: list[Run] = field(default_factory=list)

    def _ok(self) -> list[Run]:
        return [run for run in self.runs if run.done is not None]

    @property
    def errors(self) -> int:
        return sum(run.done is None for run in self.runs)

    @property
    def facts(self) -> float | None:
        shares = [run.facts_share for run in self._ok() if run.facts_share is not None]
        return statistics.mean(shares) if shares else None

    @property
    def right_refusals(self) -> tuple[int, int]:
        rows = [run for run in self._ok() if run.question.out_of_base]
        return sum(run.refused for run in rows), len(rows)

    @property
    def false_refusals(self) -> tuple[int, int]:
        rows = [run for run in self._ok() if not run.question.out_of_base]
        return sum(run.refused for run in rows), len(rows)

    @property
    def cited_share(self) -> float | None:
        rows = [run for run in self._ok() if not run.refused]
        return statistics.mean(run.cited for run in rows) if rows else None

    @property
    def foreign_runs(self) -> tuple[int, int]:
        rows = [run for run in self._ok() if not run.refused]
        return sum(bool(run.foreign) for run in rows), len(rows)

    @property
    def truncated(self) -> int:
        return sum(run.truncated for run in self._ok())

    def _median(self, attr: str, attempts: Callable[[int], bool] = lambda _: True) -> float | None:
        values = [
            getattr(run.done, attr)
            for run in self._ok()
            if attempts(run.attempt) and getattr(run.done, attr) is not None
        ]
        return statistics.median(values) if values else None

    @property
    def ttft_ms(self) -> float | None:
        return self._median("ttft_ms")

    @property
    def ttft_cold_ms(self) -> float | None:
        """Первый прогон: промпт модель видит впервые, кэш LM Studio не помогает."""
        return self._median("ttft_ms", lambda attempt: attempt == 0)

    @property
    def ttft_warm_ms(self) -> float | None:
        """Повторы: тот же промпт уже в кэше, и первый токен приходит за доли секунды."""
        return self._median("ttft_ms", lambda attempt: attempt > 0)

    @property
    def total_ms(self) -> float | None:
        return self._median("total_ms")

    @property
    def tokens_per_sec(self) -> float | None:
        return self._median("tokens_per_sec")

    @property
    def completion_tokens(self) -> float | None:
        return self._median("completion_tokens")

    @property
    def prompt_tokens(self) -> float | None:
        return self._median("prompt_tokens")

    def by_question(self) -> dict[int, list[Run]]:
        grouped: dict[int, list[Run]] = {}
        for run in self.runs:
            grouped.setdefault(run.question.id, []).append(run)
        return grouped

    @property
    def facts_spread(self) -> float | None:
        """Средний размах доли фактов между прогонами одного вопроса."""
        spreads = []
        for runs in self.by_question().values():
            shares = [run.facts_share for run in runs if run.facts_share is not None]
            if len(shares) > 1:
                spreads.append(max(shares) - min(shares))
        return statistics.mean(spreads) if spreads else None

    @property
    def time_spread(self) -> float | None:
        """Средний размах полного времени между прогонами, в долях от медианы."""
        spreads = []
        for runs in self.by_question().values():
            times = [run.done.total_ms for run in runs if run.done]
            if len(times) > 1 and statistics.median(times):
                spreads.append((max(times) - min(times)) / statistics.median(times))
        return statistics.mean(spreads) if spreads else None

    @property
    def flipped(self) -> tuple[int, int]:
        """Вопросы, где решение «ответить / отказаться» разошлось между прогонами."""
        groups = [[run.refused for run in runs if run.done] for runs in self.by_question().values()]
        groups = [group for group in groups if len(group) > 1]
        return sum(len(set(group)) > 1 for group in groups), len(groups)


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def ms(value: float | None) -> str:
    return "—" if value is None else f"{value / 1000:.1f} с"


def whole(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}"


def ratio(pair: tuple[int, int]) -> str:
    done, total = pair
    return f"{done} из {total}" if total else "—"


def one_line(text: str, limit: int = 160) -> str:
    body = " ".join(text.split())
    return body if len(body) <= limit else body[:limit] + "…"


async def ensure_model() -> None:
    try:
        models = await llm.local_models()
    except Exception as exc:
        sys.exit(f"LM Studio на {llm.BASE_URL} не отвечает: {exc}. Запустите `lms server start`.")
    if llm.MODEL not in models:
        sys.exit(f"Модели {llm.MODEL} нет на сервере: {', '.join(models)}")


async def model_line() -> str:
    try:
        info = await lmstudio.model(llm.MODEL)
    except Exception:
        return f"`{llm.MODEL}`"
    if info is None:
        return f"`{llm.MODEL}`"
    window = f"контекст {info.loaded_context_length} · {info.cache}" if info.loaded else "не загружена"
    return f"`{llm.MODEL}` · {info.quantization} · {window}"


async def run_one(question: Question, context: search.Context, profile: Profile, think: bool, attempt: int) -> Run:
    try:
        done = await llm.answer(question.question, context.block, profile=profile, think=think)
    except Exception as exc:
        return Run(question=question, profile=profile.name, done=None, block=context.block, error=str(exc), attempt=attempt)
    return Run(question=question, profile=profile.name, done=done, block=context.block, attempt=attempt)


async def questions_report(chosen: list[Profile], runs: int, think: bool) -> None:
    items = questions.load()
    summaries = {profile.name: Summary(profile.name) for profile in chosen}
    found = {item.id: search.prepare(item.question) for item in items}
    contexts = {(item.id, profile.name): profile.context(found[item.id]) for item in items for profile in chosen}

    print(f"Индекс day25: `{search.INDEX_PATH.name}`, поиск `{search.RETRIEVER}`, top-{search.TOP_K}. ", end="")
    print(f"Модель: {await model_line()}. Прогонов на вопрос: {runs}.\n")
    for profile in chosen:
        print(f"- `{profile.name}` — {profile.summary}")
    print()

    for attempt in range(runs):
        for item in items:
            for profile in chosen:
                run = await run_one(item, contexts[(item.id, profile.name)], profile, think, attempt)
                summaries[profile.name].runs.append(run)
                status = run.error or ("отказ" if run.refused else f"факты {pct(run.facts_share)}")
                print(f"<!-- прогон {attempt + 1}, #{item.id}, {profile.name}: {status} -->", file=sys.stderr)

    print("## Качество\n")
    print("| Профиль | Факты найдены | Верных отказов «вне базы» | Ложных отказов | Ответов со ссылкой [n] | Ответов с английскими словами | Оборвано по max_tokens | Ошибок |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for summary in summaries.values():
        print(
            f"| `{summary.profile}` | {pct(summary.facts)} | {ratio(summary.right_refusals)} "
            f"| {ratio(summary.false_refusals)} | {pct(summary.cited_share)} | {ratio(summary.foreign_runs)} "
            f"| {summary.truncated} | {summary.errors} |"
        )

    print("\n## Скорость\n")
    print("| Профиль | Токенов промпта, медиана | Первый токен, прогон 1 | Первый токен, повторы | Всего, медиана | Токенов ответа, медиана | Ток/с, медиана |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for summary in summaries.values():
        print(
            f"| `{summary.profile}` | {whole(summary.prompt_tokens)} | {ms(summary.ttft_cold_ms)} | {ms(summary.ttft_warm_ms)} "
            f"| {ms(summary.total_ms)} | {whole(summary.completion_tokens)} | {pct(summary.tokens_per_sec)} |"
        )

    print("\n## Стабильность\n")
    print("| Профиль | Разброс доли фактов | Разброс времени, доля от медианы | Вопросов, где решение разошлось |")
    print("| --- | --- | --- | --- |")
    for summary in summaries.values():
        print(
            f"| `{summary.profile}` | {pct(summary.facts_spread)} | {pct(summary.time_spread)} "
            f"| {ratio(summary.flipped)} |"
        )

    print("\n## По вопросам\n")
    print("| # | Вид | Контекст, ток. " + "".join(f"| `{p.name}`: факты · время " for p in chosen) + "|")
    print("| --- | --- | --- " + "| --- " * len(chosen) + "|")
    for item in items:
        cells = []
        for profile in chosen:
            runs_here = summaries[profile.name].by_question().get(item.id, [])
            marks = []
            for run in runs_here:
                if run.done is None:
                    marks.append("ошибка")
                elif run.refused:
                    marks.append("отказ")
                else:
                    marks.append(pct(run.facts_share))
            times = [run.done.total_ms for run in runs_here if run.done]
            cells.append(f"{' / '.join(marks)} · {ms(statistics.median(times)) if times else '—'}")
        tokens = " → ".join(str(contexts[(item.id, profile.name)].tokens) for profile in chosen)
        print(f"| {item.id} | {item.kind} | {tokens} | " + " | ".join(cells) + " |")

    print("\n## Ответы первого прогона\n")
    for item in items:
        print(f"**#{item.id} · {item.kind}.** {item.question}\n")
        print(f"Ожидается: {item.expect}\n")
        for profile in chosen:
            first = next(run for run in summaries[profile.name].runs if run.question.id == item.id)
            text = first.error or first.text
            foreign = f" — латиница: {', '.join(first.foreign)}" if first.foreign else ""
            print(f"- `{profile.name}`: {one_line(text, 400)}{foreign}")
        print()


async def ask_report(question: str, chosen: list[Profile], think: bool) -> None:
    found = search.prepare(question)
    print(f"**Вопрос.** {question}\n")
    print(f"Поиск `{search.RETRIEVER}` за {found.seconds * 1000:.0f} мс, {found.tokens} токенов контекста, файлы: {', '.join(found.paths)}\n")
    for source in found.sources:
        print(f"- [{source.number}] `{source.path}` · {source.section} · bm25 {source.score:.2f} · {source.tokens} ток.")
    print()
    for profile in chosen:
        context = profile.context(found)
        done = await llm.answer(question, context.block, profile=profile, think=think)
        print(f"### {profile.title}\n")
        cut = ""
        if context.budget is not None:
            cut = f" · контекст {context.tokens} из {found.tokens} ток., обрезано {sum(s.trimmed for s in context.sources)}, отброшено {context.dropped}"
        print(
            f"первый токен {done.ttft_ms} мс · всего {done.total_ms / 1000:.1f} с · токенов "
            f"{done.prompt_tokens} → {done.completion_tokens} · {done.tokens_per_sec} ток/с"
            + (" · отказ" if done.refused else "")
            + cut
            + "\n"
        )
        print(done.text + "\n")


async def load_report(context_length: int, parallel: int | None, checkpoints: int | None) -> None:
    before = await lmstudio.model(llm.MODEL)
    was = f"контекст {before.loaded_context_length} · {before.cache}" if before and before.loaded else "не загружена"
    print(f"`{llm.MODEL}` до: {was}; загружаю с контекстом {context_length}...", file=sys.stderr)
    result = await lmstudio.load(llm.MODEL, context_length=context_length, parallel=parallel, context_checkpoints=checkpoints)
    after = await lmstudio.model(llm.MODEL)
    print(f"{await model_line()} · загрузка {result.get('load_time_seconds', '?')} с")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="questions", choices=("questions", "ask", "load"))
    parser.add_argument("question", nargs="?", help="вопрос для команды ask")
    parser.add_argument("--runs", type=int, default=3, help="прогонов на вопрос (по умолчанию 3)")
    parser.add_argument("--profiles", default="base,optimized", help="через запятую: base, optimized")
    parser.add_argument("--think", action="store_true", help="включить рассуждения там, где профиль их допускает")
    parser.add_argument("--context", type=int, default=4096, help="контекстное окно для команды load")
    parser.add_argument("--parallel", type=int, help="слотов KV-кэша для команды load (LM Studio по умолчанию 4)")
    parser.add_argument("--checkpoints", type=int, help="снимков KV-кэша для команды load (по умолчанию 32; 0 — холодные замеры)")
    args = parser.parse_args()

    if args.command == "load":
        await load_report(args.context, args.parallel, args.checkpoints)
        return

    try:
        search.require_index()
    except search.IndexMissing as exc:
        sys.exit(str(exc))

    wanted = [name.strip() for name in args.profiles.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in profiles.PROFILES]
    if unknown:
        sys.exit(f"Неизвестные профили: {', '.join(unknown)}. Есть: {', '.join(profiles.PROFILES)}.")
    chosen = [profiles.get(name) for name in wanted]

    await ensure_model()

    if args.command == "ask":
        if not args.question:
            sys.exit("Команде ask нужен вопрос.")
        await ask_report(args.question, chosen, args.think)
    else:
        await questions_report(chosen, args.runs, args.think)


if __name__ == "__main__":
    asyncio.run(main())
