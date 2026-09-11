"""Замеры для README: один и тот же диалог на разных стратегиях контекста.

    python day10/scenarios.py            # все сценарии
    python day10/scenarios.py facts      # только эволюция картотеки
    python day10/scenarios.py branches   # только ветки

Диалог идёт через того же агента, что и веб-чат, но в отдельной базе: замер не
попадает в историю приложения. По ходу разговора в него подсаживаются факты, а в
конце агента спрашивают о них — ответы сверяет модель-судья.
"""

import argparse
import asyncio
import json
import sys
import tempfile
from collections.abc import Awaitable, Callable
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
from agent import SYSTEM_PROMPT, Agent, AgentFactsUpdate, AgentPlan, AgentTurn
from llm import MODEL, complete
from storage import Storage

# Ответы в замере короткие: сравнивать надо память агента, а не его словоохотливость.
# Заодно история растёт предсказуемо, и прогон стоит десятки центов, а не единицы долларов.
SCENARIO_SYSTEM = SYSTEM_PROMPT + "Отвечай по делу и не длиннее трёх предложений.\n"
SCENARIO_MAX_TOKENS = 400

# Бюджет заведомо больше любого окна: обрезать контекст должна стратегия, а не
# бюджет, иначе сравнивались бы два разных способа забывать.
SCENARIO_BUDGET = 64_000

# Окно то же, что в интерфейсе по умолчанию: три последних хода дословно.
SCENARIO_WINDOW = memory.WINDOW_MESSAGES

# Разговор об одном проекте: факты появляются в первых ходах и к концу диалога
# уезжают из окна — как раз в ту часть, которую держит картотека.
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
# моменту либо лежат в картотеке, либо забыты вместе с репликами.
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

# Сколько ходов общего начала проигрывается до ветвления: на десятом ходу решение
# об архитектуре ещё не принято, и от этого места разговор уходит в две стороны.
BRANCH_POINT = 10

BRANCH_DIALOGS = (
    (
        "отдельный сервис",
        (
            "Мы решили вынести платежи в отдельный сервис. Как разрезать данные между ним и монолитом?",
            "Как монолит узнает, что платёж прошёл?",
            "Что делать, если сервис платежей недоступен?",
        ),
    ),
    (
        "монолит",
        (
            "Мы решили оставить платежи в монолите отдельным модулем. Как обозначить границы модуля?",
            "Как не дать другим модулям лазить в таблицы платежей напрямую?",
            "Что делать, если модуль всё-таки придётся вынести?",
        ),
    ),
)

# Один и тот же вопрос в обеих ветках: первый проверяет изоляцию (решения разные),
# второй — память о общем начале, которое из окна уже выпало.
BRANCH_CHECKS = (
    (
        "Что мы решили про место платежей в архитектуре?",
        ("платежи выносим в отдельный сервис", "платежи остаются в монолите отдельным модулем"),
    ),
    (
        "Напомни: как меня зовут и какой у нас стек?",
        ("Игорь; Python, FastAPI, PostgreSQL 16, RabbitMQ",) * 2,
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

MODES = {
    memory.WINDOW: "скользящее окно",
    memory.FACTS: "факты и окно",
    memory.BRANCHING: "ветки",
}


@dataclass
class Row:
    """Один ход диалога: что уходило в модель и что пришло в счёте."""

    prompt: str
    answer: str
    estimated: int
    context_messages: int
    facts_count: int
    facts_tokens: int
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    cost: float
    branch: str = ""
    update: AgentFactsUpdate | None = None


@dataclass
class Run:
    """Прогон диалога в одной стратегии: ходы, контрольные ответы и весь счёт."""

    strategy: str
    rows: list[Row] = field(default_factory=list)
    checks: list[Row] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)

    @property
    def label(self) -> str:
        return MODES[self.strategy]

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


def note(text: str) -> None:
    """Ход работы идёт в stderr: в stdout лежит отчёт, его перенаправляют в файл.

    Прогон молчит минуты: два диалога по 25 ходов печатаются таблицами только в
    самом конце. Без этих строк непонятно, считает он или повис — и в терминале,
    и на странице сценариев, которая читает те же два потока.
    """
    print(text, file=sys.stderr, flush=True)


# Реплики идут в тот же stderr, что и ход работы, но с пометкой, кто говорит:
# страница рисует их чатом, а в терминале видно сам разговор, а не только счётчик.
SPEAKERS = {"user": "вы", "assistant": "агент"}


def say(role: str, text: str) -> None:
    # В одну строку: поток разбирается построчно, и перевод строки внутри ответа
    # выглядел бы как начало новой реплики.
    note(f"{SPEAKERS[role]}: {oneline(text)}")


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

    Кэш и стратегия тянут счёт в разные стороны: картотека добавляет токенов, но
    стоит в начале запроса и потому попадает в кэш. Эта цена показывает эффект
    одной стратегии, без второго.
    """
    return tokens.cost_usd(0, input_tokens, 0)


def context_note(row: Row) -> str:
    note = f"{row.context_messages} сообщ."
    if row.facts_count:
        note += f" + картотека {row.facts_tokens} ток. ({row.facts_count} фактов)"
    return note


async def ask(agent: Agent, prompt: str, strategy: str, branch: str = "") -> Row:
    """Один ход через того же агента, что и в веб-чате: события те же."""
    plan: AgentPlan | None = None
    turn: AgentTurn | None = None
    update: AgentFactsUpdate | None = None
    parts: list[str] = []

    say("user", prompt)
    async for event in agent.ask(prompt, SCENARIO_BUDGET, strategy, SCENARIO_WINDOW):
        if isinstance(event, AgentPlan):
            plan = event
        elif isinstance(event, AgentTurn):
            turn = event
        elif isinstance(event, AgentFactsUpdate):
            update = event
        elif event.content:
            parts.append(event.content)

    if plan is None or turn is None:
        raise RuntimeError("Ход не дошёл до конца")

    answer = "".join(parts)
    say("assistant", answer)
    if update is not None and (update.added or update.changed or update.removed):
        # Служебный ход стратегии фактов виден отдельной строкой: иначе в чате
        # непонятно, откуда у агента память о начале разговора.
        note(
            f"картотека: добавлено {update.added}, изменено {update.changed}, "
            f"удалено {update.removed} — всего {len(update.items)}",
        )

    return Row(
        prompt=prompt,
        answer=answer,
        estimated=plan.request_tokens,
        context_messages=plan.context_messages,
        facts_count=plan.facts_count,
        facts_tokens=plan.facts_tokens,
        prompt_tokens=turn.usage.prompt_tokens,
        cached_tokens=turn.usage.cached_tokens,
        completion_tokens=turn.usage.completion_tokens,
        cost=turn.usage.cost_usd,
        branch=branch,
        update=update,
    )


def scenario_agent(storage: Storage, session_id: str, strategy: str) -> Agent:
    return Agent(
        session_id,
        storage,
        system_prompt=SCENARIO_SYSTEM,
        max_tokens=SCENARIO_MAX_TOKENS,
        context_budget=SCENARIO_BUDGET,
        strategy=strategy,
        window_messages=SCENARIO_WINDOW,
    )


async def play(strategy: str) -> Run:
    """Диалог и контрольные вопросы в одной стратегии, в своей временной базе."""
    run = Run(strategy=strategy)

    with tempfile.TemporaryDirectory() as directory:
        storage = Storage(Path(directory) / "scenarios.db")
        await storage.init()
        session_id = await storage.create_session()
        agent = scenario_agent(storage, session_id, strategy)

        for number, prompt in enumerate(DIALOG, start=1):
            note(f"{MODES[strategy]}: ход {number} из {len(DIALOG)}")
            run.rows.append(await ask(agent, prompt, strategy))
        for number, (question, _) in enumerate(CHECKS, start=1):
            note(f"{MODES[strategy]}: контрольный вопрос {number} из {len(CHECKS)}")
            run.checks.append(await ask(agent, question, strategy))

        run.turns = await agent.turns()

    return run


async def play_branches() -> Run:
    """Общее начало, две ветки от одного места и контрольные вопросы в каждой."""
    run = Run(strategy=memory.BRANCHING)
    strategy = memory.BRANCHING

    with tempfile.TemporaryDirectory() as directory:
        storage = Storage(Path(directory) / "scenarios.db")
        await storage.init()
        session_id = await storage.create_session()
        agent = scenario_agent(storage, session_id, strategy)

        for number, prompt in enumerate(DIALOG[:BRANCH_POINT], start=1):
            note(f"общее начало: ход {number} из {BRANCH_POINT}")
            run.rows.append(await ask(agent, prompt, strategy, "общее начало"))

        # Checkpoint — последняя реплика общего начала: обе ветки продолжают
        # разговор от неё и друг о друге не знают.
        checkpoint = (await agent.transcript())[-1]["id"]
        for name, prompts in BRANCH_DIALOGS:
            await agent.fork(checkpoint, name)
            for number, prompt in enumerate(prompts, start=1):
                note(f"ветка «{name}»: ход {number} из {len(prompts)}")
                run.rows.append(await ask(agent, prompt, strategy, name))

        # Контрольные вопросы задаются в каждой ветке отдельно: вопросы одни, а
        # верные ответы разные — этим и проверяется, что ветки не смешались.
        for name, _ in BRANCH_DIALOGS:
            branch = next(item for item in await agent.branches() if item["name"] == name)
            await agent.switch(branch["id"])
            for number, (question, _) in enumerate(BRANCH_CHECKS, start=1):
                note(f"ветка «{name}»: контрольный вопрос {number} из {len(BRANCH_CHECKS)}")
                run.checks.append(await ask(agent, question, strategy, name))

        run.turns = await agent.turns()

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


def report_tokens(window: Run, facts: Run) -> None:
    """Главная таблица дня: один и тот же ход в двух стратегиях, вход против входа."""
    print("### Расход токенов по ходам\n")

    rows: list[tuple[Any, ...]] = []
    for number, (before, after) in enumerate(zip(window.dialog, facts.dialog), start=1):
        rows.append(
            (
                number if number <= len(DIALOG) else f"вопрос {number - len(DIALOG)}",
                context_note(before),
                context_note(after),
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
            "Контекст окна",
            "Контекст фактов",
            "Вход окна",
            "Вход фактов",
            "Разница",
            "Стоимость окна",
            "Стоимость фактов",
        ),
        rows,
    )


def report_totals(window: Run, facts: Run) -> None:
    print("### Итоги прогона\n")

    rows = [
        (
            "Входные токены",
            window.input_tokens,
            facts.input_tokens,
            percent(window.input_tokens, facts.input_tokens),
        ),
        (
            "из них из кэша",
            window.cached_tokens,
            facts.cached_tokens,
            percent(window.cached_tokens, facts.cached_tokens),
        ),
        (
            "Выходные токены",
            window.output_tokens,
            facts.output_tokens,
            percent(window.output_tokens, facts.output_tokens),
        ),
        (
            "Токены картотеки",
            window.service_tokens("facts"),
            facts.service_tokens("facts"),
            "—",
        ),
        (
            "Вход по цене без кэша",
            money(miss_price(window.input_tokens)),
            money(miss_price(facts.input_tokens)),
            percent(miss_price(window.input_tokens), miss_price(facts.input_tokens)),
        ),
        (
            "Стоимость диалога",
            money(window.dialog_cost),
            money(facts.dialog_cost),
            percent(window.dialog_cost, facts.dialog_cost),
        ),
        (
            "Стоимость картотеки",
            money(window.service_cost("facts")),
            money(facts.service_cost("facts")),
            "—",
        ),
        (
            "Всего со служебными запросами",
            money(window.total_cost),
            money(facts.total_cost),
            percent(window.total_cost, facts.total_cost),
        ),
    ]
    table(("Показатель", "Окно", "Факты", "Разница"), rows)

    updates = [row.update for row in facts.dialog if row.update is not None]
    written = sum(update.added for update in updates)
    print(
        f"Обновлений картотеки: {len(updates)}, записано фактов: {written}, "
        f"уточнено: {sum(update.changed for update in updates)}, "
        f"вычеркнуто: {sum(update.removed for update in updates)}.",
    )
    print(
        f"Ходов: {len(facts.dialog)}, из них контрольных вопросов: {len(CHECKS)}. "
        f"Окно {SCENARIO_WINDOW} сообщ., бюджет {SCENARIO_BUDGET} токенов — "
        f"контекст обрезала стратегия, а не бюджет.\n",
    )


async def report_quality(runs: tuple[Run, ...]) -> float:
    """Качество — не впечатление, а оценка судьи по каждому контрольному факту."""
    print("### Качество ответов на контрольные вопросы\n")

    rows: list[tuple[Any, ...]] = []
    scores = {run.strategy: 0 for run in runs}
    spent = 0.0

    for index, (question, reference) in enumerate(CHECKS):
        note(f"судья: вопрос {index + 1} из {len(CHECKS)}")
        for run in runs:
            row = run.checks[index]
            score, comment, cost = await judge(question, reference, row.answer)
            spent += cost
            if score >= 0:
                scores[run.strategy] += score
            rows.append(
                (
                    oneline(question),
                    run.label,
                    score if score >= 0 else "—",
                    oneline(row.answer)[:160],
                    comment[:80],
                ),
            )

    table(("Вопрос", "Стратегия", "Оценка", "Ответ агента", "Судья"), rows)

    top = 2 * len(CHECKS)
    totals = ", ".join(f"{MODES[strategy]} {score} из {top}" for strategy, score in scores.items())
    print(f"Сумма оценок: {totals}.")
    print(f"Работа судьи: {money(spent)}\n")
    return spent


def report_facts(run: Run) -> None:
    print("### Что агент помнит вместо истории\n")

    rows: list[tuple[Any, ...]] = []
    for number, row in enumerate(run.dialog, start=1):
        if row.update is None:
            continue
        rows.append(
            (
                number,
                row.update.added or "—",
                row.update.changed or "—",
                row.update.removed or "—",
                len(row.update.items),
                row.update.tokens,
                money(row.update.usage.cost_usd),
            ),
        )
    table(("Ход", "Записано", "Уточнено", "Вычеркнуто", "Фактов", "Токенов", "Стоимость"), rows)

    final = next((row.update for row in reversed(run.dialog) if row.update is not None), None)
    if final is None:
        print("Картотека осталась пустой: модель не нашла в диалоге фактов.\n")
        return

    print("Картотека в конце диалога:\n")
    table(("Ключ", "Значение"), [(key, value) for key, value in final.items])


async def report_branches(run: Run) -> float:
    """Ветки: одно начало, два продолжения, и они друг о друге не знают."""
    print("### Ветки от одного checkpoint\n")

    prefix = [row for row in run.rows if row.branch == "общее начало"]
    print(
        f"Общее начало — {len(prefix)} ходов, checkpoint на последней реплике. "
        f"От него отходят две ветки по {len(BRANCH_DIALOGS[0][1])} хода, "
        f"в каждой те же {len(BRANCH_CHECKS)} контрольных вопроса.\n",
    )

    rows: list[tuple[Any, ...]] = []
    for number, row in enumerate(run.rows, start=1):
        rows.append(
            (
                number,
                row.branch,
                oneline(row.prompt)[:60],
                context_note(row),
                row.prompt_tokens,
                money(row.cost),
            ),
        )
    table(("Ход", "Ветка", "Вопрос", "Контекст", "Вход", "Стоимость"), rows)

    spent = 0.0
    checks: list[tuple[Any, ...]] = []
    for position, (name, _) in enumerate(BRANCH_DIALOGS):
        for index, (question, references) in enumerate(BRANCH_CHECKS):
            note(f"судья: ветка «{name}», вопрос {index + 1} из {len(BRANCH_CHECKS)}")
            row = run.checks[position * len(BRANCH_CHECKS) + index]
            score, comment, cost = await judge(question, references[position], row.answer)
            spent += cost
            checks.append(
                (
                    oneline(question),
                    name,
                    score if score >= 0 else "—",
                    oneline(row.answer)[:160],
                    comment[:80],
                ),
            )
    table(("Вопрос", "Ветка", "Оценка", "Ответ агента", "Судья"), checks)

    branch_cost = sum(row.cost for row in run.rows if row.branch != "общее начало")
    prefix_cost = sum(row.cost for row in prefix)
    print(
        f"Общее начало стоило {money(prefix_cost)} и проиграно один раз на две ветки. "
        f"Продолжения обошлись в {money(branch_cost)}, весь прогон со служебными "
        f"запросами — {money(run.total_cost)}. Двумя отдельными диалогами то же "
        f"сравнение стоило бы примерно {money(prefix_cost + run.total_cost)}.",
    )
    print(f"Работа судьи: {money(spent)}\n")
    return spent


async def compare() -> None:
    print("## Окно против фактов\n")
    print(
        f"Диалог из {len(DIALOG)} ходов плюс {len(CHECKS)} контрольных вопросов проигран дважды: "
        f"на скользящем окне из {SCENARIO_WINDOW} сообщений и на нём же с картотекой фактов, "
        f"которую агент обновляет после каждого хода.\n",
    )

    window = await play(memory.WINDOW)
    facts = await play(memory.FACTS)

    report_tokens(window, facts)
    report_totals(window, facts)
    await report_quality((window, facts))
    report_facts(facts)


async def facts() -> None:
    """Только прогон с фактами: промпт картотеки можно править, не платя за второй."""
    print("## Картотека фактов\n")
    run = await play(memory.FACTS)
    report_facts(run)
    updates = len([row for row in run.dialog if row.update is not None])
    print(
        f"Диалог: {money(run.dialog_cost)}, картотека: {money(run.service_cost('facts'))}, "
        f"обновлений: {updates}.\n",
    )


async def branches() -> None:
    print("## Ветвление диалога\n")
    run = await play_branches()
    await report_branches(run)


@dataclass(frozen=True)
class Scenario:
    """Сценарий вместе с ценником: страница показывает его до нажатия кнопки."""

    run: Callable[[], Awaitable[None]]
    title: str
    about: str
    # Сколько запросов уйдёт в модель: прогон платный, и знать это лучше заранее.
    requests: int


# Ходов в прогоне: диалог плюс контрольные вопросы. В стратегии фактов каждый ход
# тянет за собой ещё один запрос — разбор реплики на факты.
TURNS = len(DIALOG) + len(CHECKS)
BRANCH_TURNS = BRANCH_POINT + sum(len(prompts) for _, prompts in BRANCH_DIALOGS)
BRANCH_QUESTIONS = len(BRANCH_DIALOGS) * len(BRANCH_CHECKS)

SCENARIOS = {
    "compare": Scenario(
        compare,
        "Окно против фактов",
        "один диалог в двух стратегиях, таблицы токенов и оценки судьи",
        # Два прогона, картотека во втором, две оценки судьи на вопрос, два заголовка.
        requests=2 * TURNS + TURNS + 2 * len(CHECKS) + 2,
    ),
    "facts": Scenario(
        facts,
        "Картотека фактов",
        "только прогон с фактами: промпт извлечения можно править дешевле",
        requests=2 * TURNS + 1,
    ),
    "branches": Scenario(
        branches,
        "Ветвление диалога",
        "общее начало и две ветки от одного места с контрольными вопросами",
        # Контрольный вопрос в ветке — это ход агента и следом оценка судьи.
        requests=BRANCH_TURNS + 2 * BRANCH_QUESTIONS + 1,
    ),
}


async def main(names: list[str]) -> None:
    print(f"Модель: {MODEL}, тариф {'пиковый' if tokens.is_peak_hour() else 'непиковый'}\n")
    for name in names:
        await SCENARIOS[name].run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="*", choices=sorted(SCENARIOS), default=sorted(SCENARIOS))
    arguments = parser.parse_args()
    asyncio.run(main(list(arguments.scenarios)))
    sys.exit(0)
