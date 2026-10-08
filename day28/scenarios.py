"""Контрольный набор обеим моделям, по три прогона на вопрос, вывод в markdown.

Три оси задания — качество, скорость, стабильность — считаются на одном и том
же контексте: поиск у обеих моделей общий и детерминированный, поэтому всё, что
различается в таблицах, — заслуга или вина модели, а не выдачи.

Качество — доля обязательных подстрок `facts` в ответе, как в day22–day24, плюс
отказ на вопросах «вне базы» и ложный отказ на вопросах «в базе». Судьи-модели
здесь нет сознательно: судить ответы локальной модели облачной значило бы
встроить облако в оценку локальной системы, а судить локальной — доверять оценку
той, кого оцениваем.

Стабильность — то, что видно только на повторах: разброс доли фактов и времени
между тремя прогонами одного вопроса, доля ответов, где решение «ответить или
отказаться» разошлось между прогонами, обрывы по `max_tokens` и ошибки
транспорта. Три прогона — мало для десятых долей, но достаточно, чтобы отличить
«всегда так» от «как повезёт».

    python day28/scenarios.py                 # обе доступные модели, 10 вопросов × 3
    python day28/scenarios.py --runs 1        # по одному прогону
    python day28/scenarios.py --models local  # только локальная
    python day28/scenarios.py ask "вопрос"    # один вопрос обеим, с контекстом
"""

import argparse
import asyncio
import re
import statistics
import sys
from dataclasses import dataclass, field

import llm
import questions
import search
from questions import Question

CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Run:
    question: Question
    provider: str
    done: llm.Done | None
    error: str | None = None

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


@dataclass
class Summary:
    provider: str
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
    def truncated(self) -> int:
        return sum(run.truncated for run in self._ok())

    def _median(self, attr: str) -> float | None:
        values = [getattr(run.done, attr) for run in self._ok() if getattr(run.done, attr) is not None]
        return statistics.median(values) if values else None

    @property
    def ttft_ms(self) -> float | None:
        return self._median("ttft_ms")

    @property
    def total_ms(self) -> float | None:
        return self._median("total_ms")

    @property
    def tokens_per_sec(self) -> float | None:
        return self._median("tokens_per_sec")

    @property
    def completion_tokens(self) -> float | None:
        return self._median("completion_tokens")

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


def ratio(pair: tuple[int, int]) -> str:
    done, total = pair
    return f"{done} из {total}" if total else "—"


def one_line(text: str, limit: int = 160) -> str:
    body = " ".join(text.split())
    return body if len(body) <= limit else body[:limit] + "…"


async def available(wanted: list[str]) -> list[str]:
    providers = []
    for provider in wanted:
        if provider == "local":
            try:
                models = await llm.local_models()
            except Exception as exc:
                print(f"<!-- LM Studio на {llm.LOCAL_BASE_URL} не отвечает: {exc} -->", file=sys.stderr)
                continue
            if llm.LOCAL_MODEL not in models:
                print(f"<!-- модели {llm.LOCAL_MODEL} нет на сервере: {', '.join(models)} -->", file=sys.stderr)
                continue
        elif not llm.cloud_configured():
            print("<!-- DEEPSEEK_API_KEY не задан: облачная модель пропущена -->", file=sys.stderr)
            continue
        providers.append(provider)
    return providers


async def run_one(question: Question, block: str, provider: str, think: bool) -> Run:
    try:
        done = await llm.answer(question.question, block, provider=provider, think=think)
    except Exception as exc:
        return Run(question=question, provider=provider, done=None, error=str(exc))
    return Run(question=question, provider=provider, done=done)


async def questions_report(providers: list[str], runs: int, think: bool) -> None:
    items = questions.load()
    summaries = {provider: Summary(provider) for provider in providers}
    contexts = {item.id: search.prepare(item.question) for item in items}

    print(f"Индекс day25: `{search.INDEX_PATH.name}`, поиск `{search.RETRIEVER}`, top-{search.TOP_K}. ", end="")
    print(f"Модели: {', '.join(f'`{llm.MODELS[p]}`' for p in providers)}. Прогонов на вопрос: {runs}.\n")

    for attempt in range(runs):
        for item in items:
            for provider in providers:
                run = await run_one(item, contexts[item.id].block, provider, think)
                summaries[provider].runs.append(run)
                status = run.error or ("отказ" if run.refused else f"факты {pct(run.facts_share)}")
                print(f"<!-- прогон {attempt + 1}, #{item.id}, {provider}: {status} -->", file=sys.stderr)

    print("## Качество\n")
    print("| Модель | Факты найдены | Верных отказов «вне базы» | Ложных отказов | Ответов со ссылкой [n] | Оборвано по max_tokens | Ошибок |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for summary in summaries.values():
        print(
            f"| `{llm.MODELS[summary.provider]}` | {pct(summary.facts)} | {ratio(summary.right_refusals)} "
            f"| {ratio(summary.false_refusals)} | {pct(summary.cited_share)} | {summary.truncated} | {summary.errors} |"
        )

    print("\n## Скорость\n")
    print("| Модель | Первый токен, медиана | Всего, медиана | Токенов ответа, медиана | Ток/с, медиана |")
    print("| --- | --- | --- | --- | --- |")
    for summary in summaries.values():
        print(
            f"| `{llm.MODELS[summary.provider]}` | {ms(summary.ttft_ms)} | {ms(summary.total_ms)} "
            f"| {pct(summary.completion_tokens).replace('.00', '')} | {pct(summary.tokens_per_sec)} |"
        )

    print("\n## Стабильность\n")
    print("| Модель | Разброс доли фактов | Разброс времени, доля от медианы | Вопросов, где решение разошлось |")
    print("| --- | --- | --- | --- |")
    for summary in summaries.values():
        print(
            f"| `{llm.MODELS[summary.provider]}` | {pct(summary.facts_spread)} | {pct(summary.time_spread)} "
            f"| {ratio(summary.flipped)} |"
        )

    print("\n## По вопросам\n")
    print("| # | Вид | Файлы в контексте | " + " | ".join(f"`{llm.MODELS[p]}`: факты · время" for p in providers) + " |")
    print("| --- | --- | --- | " + " | ".join("---" for _ in providers) + " |")
    for item in items:
        cells = []
        for provider in providers:
            runs_here = summaries[provider].by_question().get(item.id, [])
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
        paths = ", ".join(contexts[item.id].paths[:3])
        print(f"| {item.id} | {item.kind} | {paths} | " + " | ".join(cells) + " |")

    print("\n## Ответы первого прогона\n")
    for item in items:
        print(f"**#{item.id} · {item.kind}.** {item.question}\n")
        print(f"Ожидается: {item.expect}\n")
        for provider in providers:
            first = next(run for run in summaries[provider].runs if run.question.id == item.id)
            text = first.error or first.text
            print(f"- `{llm.MODELS[provider]}`: {one_line(text, 400)}")
        print()


async def ask_report(question: str, providers: list[str], think: bool) -> None:
    context = search.prepare(question)
    print(f"**Вопрос.** {question}\n")
    print(f"Поиск `{search.RETRIEVER}` за {context.seconds * 1000:.0f} мс, {context.tokens} токенов контекста, файлы: {', '.join(context.paths)}\n")
    for source in context.sources:
        print(f"- [{source.number}] `{source.path}` · {source.section} · bm25 {source.score:.2f}")
    print()
    for provider in providers:
        done = await llm.answer(question, context.block, provider=provider, think=think)
        print(f"### {llm.PROVIDER_TITLES[provider]} · `{done.model}`\n")
        print(
            f"первый токен {done.ttft_ms} мс · всего {done.total_ms / 1000:.1f} с · токенов "
            f"{done.prompt_tokens} → {done.completion_tokens} · {done.tokens_per_sec} ток/с"
            + (" · отказ" if done.refused else "")
            + "\n"
        )
        print(done.text + "\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="questions", choices=("questions", "ask"))
    parser.add_argument("question", nargs="?", help="вопрос для команды ask")
    parser.add_argument("--runs", type=int, default=3, help="прогонов на вопрос (по умолчанию 3)")
    parser.add_argument("--models", default="local,cloud", help="через запятую: local, cloud")
    parser.add_argument("--think", action="store_true", help="включить рассуждения у локальной модели")
    args = parser.parse_args()

    try:
        search.require_index()
    except search.IndexMissing as exc:
        sys.exit(str(exc))

    wanted = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in llm.PROVIDERS]
    if unknown:
        sys.exit(f"Неизвестные модели: {', '.join(unknown)}. Есть: {', '.join(llm.PROVIDERS)}.")

    providers = await available(wanted)
    if not providers:
        sys.exit("Ни одна модель недоступна.")

    if args.command == "ask":
        if not args.question:
            sys.exit("Команде ask нужен вопрос.")
        await ask_report(args.question, providers, args.think)
    else:
        await questions_report(providers, args.runs, args.think)


if __name__ == "__main__":
    asyncio.run(main())
