"""Замеры для README: один и тот же диалог со сжатием истории и без него.

    python day9/scenarios.py            # все сценарии
    python day9/scenarios.py summary    # только эволюция конспектов

Диалог идёт через того же агента, что и веб-чат, но в отдельной базе: замер не
попадает в историю приложения. По ходу разговора в него подсаживаются факты, а в
конце агента спрашивают о них — ответы сверяет модель-судья.
"""

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

import memory
import tokens
from agent import SYSTEM_PROMPT, Agent, AgentCompaction, AgentPlan, AgentTurn
from llm import MODEL, complete
from storage import Storage

# Ответы в замере короткие: сравнивать надо память агента, а не его словоохотливость.
# Заодно история растёт предсказуемо, и прогон стоит десятки центов, а не единицы долларов.
SCENARIO_SYSTEM = SYSTEM_PROMPT + "Отвечай по делу и не длиннее трёх предложений.\n"
SCENARIO_MAX_TOKENS = 400

# Бюджет заведомо больше любого прогона: в режиме без сжатия история не должна
# обрезаться ни разу, иначе сравнивались бы два разных способа забывать.
SCENARIO_BUDGET = 64_000

# Разговор об одном проекте: факты появляются в первых ходах и к концу диалога
# уезжают из хвоста — как раз в ту часть, которую заменяет конспект.
DIALOG = (
    "Привет! Меня зовут Игорь, я тимлид в сервисе доставки еды «Тарелка». "
    "Бэкенд у нас на Python и FastAPI. С чего начать проектирование платёжного модуля?",
    "База у нас PostgreSQL 16, очереди на RabbitMQ. Что из этого пригодится платежам?",
    "Провайдер — Stripe, но подключаем ещё ЮKassa. Как развести двух провайдеров в коде?",
    "Договорились: делаем адаптер PaymentGateway с методами charge и refund. "
    "Какие поля нужны в запросе charge?",
    "Таблицу назовём payments. Какие колонки в ней нужны?",
    "Историю платежей храним пять лет — это требование наших юристов. Где такое хранить?",
    "Как сделать ретраи, если провайдер отвечает таймаутом?",
    "Что такое ключ идемпотентности и где его держать?",
    "Мы решили брать UUID v7 как ключ идемпотентности. Это хорошая идея?",
    "Как тестировать интеграцию с провайдером без реальных платежей?",
    "Какие метрики собирать по платежам?",
    "У нас Grafana и Prometheus. Что вывести на дашборд?",
    "Как принимать вебхуки от провайдера?",
    "Как защитить вебхук от подделки?",
    "Что писать в логи, а что писать нельзя?",
    "Выносить платежи в отдельный сервис или оставить в монолите?",
    "Мы решили оставить платежи в монолите, но отдельным модулем. Как нарезать границы?",
    "Какой таймаут ставить на запрос к провайдеру?",
    "Как сделать частичные возвраты?",
    "Подведи итог в трёх пунктах.",
)

# Контрольные вопросы задаются после диалога: факты из первых ходов к этому
# моменту либо пересказаны конспектом, либо тащатся дословно.
CHECKS = (
    (
        "Напомни: как меня зовут и в каком сервисе я работаю?",
        "Игорь, тимлид сервиса доставки еды «Тарелка»",
    ),
    (
        "Перечисли наш стек: язык, фреймворк, база, очереди.",
        "Python, FastAPI, PostgreSQL 16, RabbitMQ",
    ),
    (
        "Каких провайдеров мы подключаем и как назвали адаптер и его методы?",
        "Stripe и ЮKassa; адаптер PaymentGateway с методами charge и refund",
    ),
    (
        "Сколько лет мы храним историю платежей и почему именно столько?",
        "пять лет, требование юристов",
    ),
    (
        "Что мы решили про ключ идемпотентности и про место платежей в архитектуре?",
        "ключ — UUID v7; платежи остаются в монолите отдельным модулем",
    ),
)

JUDGE_PROMPT = """\
Ты проверяешь, помнит ли ассистент факты из давнего разговора.
Тебе дают вопрос, эталонные факты и ответ ассистента.
Оценивай только совпадение фактов, а не стиль и не полноту рассуждений:
2 — все факты названы верно;
1 — часть фактов верна, часть потеряна или искажена;
0 — фактов нет, они неверны или ассистент отвечает, что не помнит.
Верни строго JSON без пояснений и без markdown: {"score": 2, "comment": "коротко, что не так"}
"""

JUDGE_MAX_TOKENS = 200

MODES = {True: "со сжатием", False: "без сжатия"}


@dataclass
class Row:
    """Один ход диалога: что уходило в модель и что пришло в счёте."""

    prompt: str
    answer: str
    estimated: int
    context_messages: int
    summary_messages: int
    summary_tokens: int
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    cost: float
    compaction: AgentCompaction | None = None


@dataclass
class Run:
    """Прогон диалога в одном режиме: ходы, контрольные ответы и весь счёт."""

    compression: bool
    rows: list[Row] = field(default_factory=list)
    checks: list[Row] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    summaries: list[dict[str, Any]] = field(default_factory=list)

    @property
    def dialog(self) -> list[Row]:
        return [*self.rows, *self.checks]

    @property
    def input_tokens(self) -> int:
        return sum(row.prompt_tokens for row in self.dialog)

    @property
    def cached_tokens(self) -> int:
        return sum(row.cached_tokens for row in self.dialog)

    @property
    def output_tokens(self) -> int:
        return sum(row.completion_tokens for row in self.dialog)

    @property
    def dialog_cost(self) -> float:
        return sum(row.cost for row in self.dialog)

    def service_cost(self, kind: str) -> float:
        return sum(turn["cost_usd"] for turn in self.turns if turn["kind"] == kind)

    def service_tokens(self, kind: str) -> int:
        return sum(
            turn["prompt_tokens"] + turn["completion_tokens"]
            for turn in self.turns
            if turn["kind"] == kind
        )

    @property
    def total_cost(self) -> float:
        return sum(turn["cost_usd"] for turn in self.turns)


def cell(value: Any) -> str:
    # Таблица идёт прямиком в README, а в ответах модели и судьи попадаются
    # вертикальные черты: неэкранированная сломала бы разметку.
    return str(value).replace("|", "\\|")


def table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    print(f"| {' | '.join(headers)} |")
    print(f"| {' | '.join('---' for _ in headers)} |")
    for row in rows:
        print(f"| {' | '.join(cell(value) for value in row)} |")
    print()


def money(value: float) -> str:
    return f"${value:.6f}"


def percent(before: int | float, after: int | float) -> str:
    if not before:
        return "—"
    return f"{(after - before) / before * 100:+.1f}%"


def oneline(text: str) -> str:
    return " ".join(text.split())


def miss_price(input_tokens: int) -> float:
    """Во сколько встал бы вход, если бы кэша префикса не существовало.

    Кэш и сжатие тянут счёт в разные стороны: сжатие убирает токены, но меняет
    начало запроса и тем самым выбивает его из кэша. Эта цена показывает эффект
    одного сжатия, без второго.
    """
    return tokens.cost_usd(0, input_tokens, 0)


async def ask(agent: Agent, prompt: str, compression: bool) -> Row:
    """Один ход через того же агента, что и в веб-чате: события те же."""
    plan: AgentPlan | None = None
    turn: AgentTurn | None = None
    compaction: AgentCompaction | None = None
    parts: list[str] = []

    async for event in agent.ask(prompt, SCENARIO_BUDGET, compression):
        if isinstance(event, AgentPlan):
            plan = event
        elif isinstance(event, AgentTurn):
            turn = event
        elif isinstance(event, AgentCompaction):
            compaction = event
        elif event.content:
            parts.append(event.content)

    if plan is None or turn is None:
        raise RuntimeError("Ход не дошёл до конца")

    return Row(
        prompt=prompt,
        answer="".join(parts),
        estimated=plan.request_tokens,
        context_messages=plan.context_messages,
        summary_messages=plan.summary_messages,
        summary_tokens=plan.summary_tokens,
        prompt_tokens=turn.usage.prompt_tokens,
        cached_tokens=turn.usage.cached_tokens,
        completion_tokens=turn.usage.completion_tokens,
        cost=turn.usage.cost_usd,
        compaction=compaction,
    )


async def play(compression: bool) -> Run:
    """Диалог и контрольные вопросы в одном режиме, в своей временной базе."""
    run = Run(compression=compression)

    with tempfile.TemporaryDirectory() as directory:
        storage = Storage(Path(directory) / "scenarios.db")
        await storage.init()
        session_id = await storage.create_session()
        agent = Agent(
            session_id,
            storage,
            system_prompt=SCENARIO_SYSTEM,
            max_tokens=SCENARIO_MAX_TOKENS,
            context_budget=SCENARIO_BUDGET,
            compression=compression,
        )

        for prompt in DIALOG:
            run.rows.append(await ask(agent, prompt, compression))
        for question, _ in CHECKS:
            run.checks.append(await ask(agent, question, compression))

        run.turns = await agent.turns()
        run.summaries = await agent.summaries()

    return run


async def judge(question: str, reference: str, answer: str) -> tuple[int, str, float]:
    """Совпадение с эталоном оценивает модель: у неё перед глазами оба текста."""
    request: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=JUDGE_PROMPT),
        ChatCompletionUserMessageParam(
            role="user",
            content=(
                f"Вопрос: {question}\n"
                f"Эталонные факты: {reference}\n"
                f"Ответ ассистента: {oneline(answer)}"
            ),
        ),
    ]
    verdict = await complete(request, temperature=0, max_tokens=JUDGE_MAX_TOKENS)
    cost = tokens.usage_from(verdict.usage).cost_usd

    text = verdict.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        parsed = json.loads(text)
        return int(parsed["score"]), oneline(str(parsed.get("comment", ""))), cost
    except (ValueError, KeyError, TypeError):
        # Судья ответил не по форме: замер от этого не рушится, но и оценки нет.
        return -1, f"ответ судьи не разобран: {oneline(verdict.text)[:80]}", cost


def report_tokens(full: Run, packed: Run) -> None:
    """Главная таблица дня: один и тот же ход в двух режимах, вход против входа."""
    print("### Расход токенов по ходам\n")

    rows: list[tuple[Any, ...]] = []
    for number, (before, after) in enumerate(zip(full.dialog, packed.dialog), start=1):
        context = f"{after.context_messages} сообщ."
        if after.summary_messages:
            context += f" + конспект {after.summary_tokens} ток. ({after.summary_messages} сообщ.)"
        rows.append(
            (
                number if number <= len(DIALOG) else f"вопрос {number - len(DIALOG)}",
                before.context_messages,
                context,
                before.prompt_tokens,
                after.prompt_tokens,
                percent(before.prompt_tokens, after.prompt_tokens),
                money(before.cost),
                money(after.cost),
            ),
        )

    table(
        (
            "Ход",
            "Контекст без сжатия",
            "Контекст со сжатием",
            "Вход без сжатия",
            "Вход со сжатием",
            "Разница",
            "Стоимость без",
            "Стоимость со",
        ),
        rows,
    )


def report_totals(full: Run, packed: Run) -> None:
    print("### Итоги прогона\n")

    rows = [
        ("Входные токены", full.input_tokens, packed.input_tokens, percent(full.input_tokens, packed.input_tokens)),
        ("из них из кэша", full.cached_tokens, packed.cached_tokens, percent(full.cached_tokens, packed.cached_tokens)),
        ("Выходные токены", full.output_tokens, packed.output_tokens, percent(full.output_tokens, packed.output_tokens)),
        (
            "Токены свёрток",
            full.service_tokens("summary"),
            packed.service_tokens("summary"),
            "—",
        ),
        (
            "Вход по цене без кэша",
            money(miss_price(full.input_tokens)),
            money(miss_price(packed.input_tokens)),
            percent(miss_price(full.input_tokens), miss_price(packed.input_tokens)),
        ),
        (
            "Стоимость диалога",
            money(full.dialog_cost),
            money(packed.dialog_cost),
            percent(full.dialog_cost, packed.dialog_cost),
        ),
        (
            "Стоимость свёрток",
            money(full.service_cost("summary")),
            money(packed.service_cost("summary")),
            "—",
        ),
        (
            "Всего со служебными запросами",
            money(full.total_cost),
            money(packed.total_cost),
            percent(full.total_cost, packed.total_cost),
        ),
    ]
    table(("Показатель", "Без сжатия", "Со сжатием", "Разница"), rows)

    compactions = [row.compaction for row in packed.dialog if row.compaction is not None]
    replaced = sum(compaction.replaced_tokens for compaction in compactions)
    print(f"Свёрток: {len(compactions)}, ими заменено {replaced} токенов дословной истории.")
    print(
        f"Ходов: {len(packed.dialog)}, из них контрольных вопросов: {len(CHECKS)}. "
        f"Бюджет контекста {SCENARIO_BUDGET} токенов — в режиме без сжатия история не обрезалась.\n",
    )


async def report_quality(full: Run, packed: Run) -> float:
    """Качество — не впечатление, а оценка судьи по каждому контрольному факту."""
    print("### Качество ответов на контрольные вопросы\n")

    rows: list[tuple[Any, ...]] = []
    scores = {True: 0, False: 0}
    spent = 0.0

    for (question, reference), before, after in zip(CHECKS, full.checks, packed.checks):
        for run, row in ((full, before), (packed, after)):
            score, comment, cost = await judge(question, reference, row.answer)
            spent += cost
            if score >= 0:
                scores[run.compression] += score
            rows.append(
                (
                    oneline(question),
                    MODES[run.compression],
                    score if score >= 0 else "—",
                    oneline(row.answer)[:160],
                    comment[:80],
                ),
            )

    table(("Вопрос", "Режим", "Оценка", "Ответ агента", "Судья"), rows)

    top = 2 * len(CHECKS)
    print(f"Сумма оценок: без сжатия {scores[False]} из {top}, со сжатием {scores[True]} из {top}.")
    print(f"Работа судьи: {money(spent)}\n")
    return spent


def report_summaries(run: Run) -> None:
    print("### Что агент помнит вместо истории\n")

    for number, summary in enumerate(run.summaries, start=1):
        print(f"**Свёртка {number}** — {summary['messages']} сообщ., {summary['tokens']} ток.\n")
        print(f"> {oneline(summary['text'])}\n")

    if not run.summaries:
        print("Свёрток не было: диалог не дорос до первого блока.\n")


async def compare() -> None:
    print("## Сжатие против полной истории\n")
    print(
        f"Диалог из {len(DIALOG)} ходов плюс {len(CHECKS)} контрольных вопросов проигран дважды: "
        f"с полной историей в запросе и со сжатием (хвост {memory.TAIL_MESSAGES} сообщ. дословно, "
        f"свёртка каждые {memory.COMPRESS_EVERY}).\n",
    )

    full = await play(compression=False)
    packed = await play(compression=True)

    report_tokens(full, packed)
    report_totals(full, packed)
    await report_quality(full, packed)
    report_summaries(packed)


async def summary() -> None:
    """Только сжатый прогон: промпт свёртки можно править, не платя за второй."""
    print("## Конспекты сжатого прогона\n")
    packed = await play(compression=True)
    report_summaries(packed)

    rows = [
        (
            number,
            turn["context_messages"],
            turn["prompt_tokens"],
            turn["completion_tokens"],
            money(turn["cost_usd"]),
        )
        for number, turn in enumerate(
            (turn for turn in packed.turns if turn["kind"] == "summary"),
            start=1,
        )
    ]
    table(("Свёртка", "Свёрнуто сообщ.", "Вход", "Ответ", "Стоимость"), rows)
    print(f"Диалог: {money(packed.dialog_cost)}, свёртки: {money(packed.service_cost('summary'))}\n")


SCENARIOS = {
    "compare": compare,
    "summary": summary,
}


async def main(names: list[str]) -> None:
    print(f"Модель: {MODEL}, тариф {'пиковый' if tokens.is_peak_hour() else 'непиковый'}\n")
    for name in names:
        await SCENARIOS[name]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="*", choices=sorted(SCENARIOS), default=sorted(SCENARIOS))
    arguments = parser.parse_args()
    asyncio.run(main(list(arguments.scenarios)))
    sys.exit(0)
