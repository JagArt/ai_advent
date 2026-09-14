"""Замеры для README: один и тот же диалог, прогнанный без стратегии и с каждой из трёх.

    python day10/scenarios.py            # все прогоны по очереди
    python day10/scenarios.py clean      # без управления контекстом: вся история в запросе
    python day10/scenarios.py facts      # только картотека
    python day10/scenarios.py branches   # только ветки

Каждый сценарий — один самостоятельный прогон: диалог, контрольные вопросы, оценки
судьи и свой счёт. Сравнивать прогоны сценарий не умеет и не должен: результат
уходит в JSON (`--result`), а сводные таблицы собирает `compare.py` из тех
прогонов, что лежат в сессии страницы.

Диалог идёт через того же агента, что и веб-чат, но в отдельной базе: замер не
попадает в историю приложения.
"""

import argparse
import asyncio
import hashlib
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

import compare
import memory
import report
import tokens
from agent import SYSTEM_PROMPT, Agent, AgentFactsUpdate, AgentPlan, AgentTurn
from llm import MODEL, complete
from report import money, oneline
from storage import Storage

# Ответы в замере короткие: сравнивать надо память агента, а не его словоохотливость.
# Заодно история растёт предсказуемо, и прогон стоит десятки центов, а не единицы долларов.
SCENARIO_SYSTEM = SYSTEM_PROMPT + "Отвечай по делу и не длиннее трёх предложений.\n"
SCENARIO_MAX_TOKENS = 400

# Бюджет заведомо больше любого окна: обрезать контекст должна стратегия, а не
# бюджет, иначе сравнивались бы два разных способа забывать. Чистому прогону тот же
# бюджет нужен, чтобы вся история влезала в запрос до самого конца диалога.
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

PREFIX = "общее начало"

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
    memory.CLEAN: "Без стратегии",
    memory.WINDOW: "Sliding Window",
    memory.FACTS: "Sticky Facts",
    memory.BRANCHING: "Branching",
}


@dataclass
class Row:
    """Один ход диалога: что уходило в модель, что пришло в счёте и что сказал судья."""

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
    # Заполняется только у контрольных вопросов: остальные ходы судья не смотрит.
    reference: str = ""
    score: int = -1
    comment: str = ""


@dataclass
class Run:
    """Прогон в одной стратегии: ходы, контрольные ответы, оценки и весь счёт."""

    strategy: str
    scenario: str
    rows: list[Row] = field(default_factory=list)
    checks: list[Row] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    judge_cost: float = 0.0

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

    @property
    def score(self) -> int:
        return sum(row.score for row in self.checks if row.score > 0)

    @property
    def max_score(self) -> int:
        return compare.MAX_SCORE * len(self.checks)

    @property
    def service_turns(self) -> list[dict[str, Any]]:
        """Строки счёта, которых в диалоге не видно: заголовок и разбор на факты."""
        return [turn for turn in self.turns if turn["kind"] != "turn"]

    @property
    def service_cost(self) -> float:
        return sum(turn["cost_usd"] for turn in self.service_turns)

    @property
    def service_tokens(self) -> int:
        return sum(turn["prompt_tokens"] + turn["completion_tokens"] for turn in self.service_turns)

    @property
    def total_cost(self) -> float:
        """Весь счёт прогона: ходы, служебные запросы агента и работа судьи."""
        return sum(turn["cost_usd"] for turn in self.turns) + self.judge_cost

    def result(self) -> compare.RunResult:
        """Машинный результат: то же, что в отчёте, но пригодное для сравнения."""
        # Диалог и контрольные вопросы нумеруются каждый со своей единицы: в
        # сравнении ходы сводятся по номеру внутри своего вида.
        turns = [turn(number, "dialog", row) for number, row in enumerate(self.rows, start=1)]
        turns += [turn(number, "check", row) for number, row in enumerate(self.checks, start=1)]

        return compare.RunResult(
            scenario=self.scenario,
            label=self.label,
            strategy=self.strategy,
            model=MODEL,
            peak=tokens.is_peak_hour(),
            fingerprint=fingerprint(),
            window_messages=SCENARIO_WINDOW,
            budget=SCENARIO_BUDGET,
            turns=turns,
            checks=[
                compare.RunCheck(
                    question=row.prompt,
                    reference=row.reference,
                    answer=row.answer,
                    score=row.score,
                    comment=row.comment,
                    branch=row.branch,
                )
                for row in self.checks
            ],
            input_tokens=self.input_tokens,
            cached_tokens=self.cached_tokens,
            output_tokens=self.output_tokens,
            service_tokens=self.service_tokens,
            dialog_cost=self.dialog_cost,
            service_cost=self.service_cost,
            judge_cost=self.judge_cost,
            total_cost=self.total_cost,
        )


def turn(number: int, kind: str, row: Row) -> compare.RunTurn:
    return compare.RunTurn(
        number=number,
        kind=kind,
        branch=row.branch,
        context_messages=row.context_messages,
        facts_tokens=row.facts_tokens,
        prompt_tokens=row.prompt_tokens,
        cached_tokens=row.cached_tokens,
        completion_tokens=row.completion_tokens,
        cost_usd=row.cost,
    )


def fingerprint() -> str:
    """Отпечаток замера: модель, диалог, вопросы и настройки контекста.

    Чистый прогон переносится между сессиями страницы и может пережить правку
    диалога. По отпечатку видно, что колонки в сравнении посчитаны про разное.
    """
    material = json.dumps(
        [MODEL, DIALOG, CHECKS, BRANCH_DIALOGS, BRANCH_CHECKS, SCENARIO_WINDOW, SCENARIO_BUDGET],
        ensure_ascii=False,
    )
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def note(text: str) -> None:
    """Ход работы идёт в stderr: в stdout лежит отчёт, его перенаправляют в файл.

    Прогон молчит минуты: диалог из 25 ходов печатается таблицами только в самом
    конце. Без этих строк непонятно, считает он или повис — и в терминале, и на
    странице сценариев, которая читает те же два потока.
    """
    print(text, file=sys.stderr, flush=True)


def out(*lines: str) -> None:
    """Отчёт в stdout: строки собраны хелперами из `report`, здесь только печать."""
    for line in lines:
        print(line)


# Реплики идут в тот же stderr, что и ход работы, но с пометкой, кто говорит:
# страница рисует их чатом, а в терминале видно сам разговор, а не только счётчик.
SPEAKERS = {"user": "вы", "assistant": "агент"}


def say(role: str, text: str) -> None:
    # В одну строку: поток разбирается построчно, и перевод строки внутри ответа
    # выглядел бы как начало новой реплики.
    note(f"{SPEAKERS[role]}: {oneline(text)}")


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


async def play(scenario: str, strategy: str) -> Run:
    """Диалог, контрольные вопросы и оценки судьи в одной стратегии."""
    run = Run(strategy=strategy, scenario=scenario)

    with tempfile.TemporaryDirectory() as directory:
        storage = Storage(Path(directory) / "scenarios.db")
        await storage.init()
        session_id = await storage.create_session()
        agent = scenario_agent(storage, session_id, strategy)

        for number, prompt in enumerate(DIALOG, start=1):
            note(f"{MODES[strategy]}: ход {number} из {len(DIALOG)}")
            run.rows.append(await ask(agent, prompt, strategy))
        for number, (question, reference) in enumerate(CHECKS, start=1):
            note(f"{MODES[strategy]}: контрольный вопрос {number} из {len(CHECKS)}")
            row = await ask(agent, question, strategy)
            row.reference = reference
            run.checks.append(row)

        run.turns = await agent.turns()

    await score(run)
    return run


async def play_branches() -> Run:
    """Общее начало, две ветки от одного места и контрольные вопросы в каждой."""
    strategy = memory.BRANCHING
    run = Run(strategy=strategy, scenario="branches")

    with tempfile.TemporaryDirectory() as directory:
        storage = Storage(Path(directory) / "scenarios.db")
        await storage.init()
        session_id = await storage.create_session()
        agent = scenario_agent(storage, session_id, strategy)

        for number, prompt in enumerate(DIALOG[:BRANCH_POINT], start=1):
            note(f"{PREFIX}: ход {number} из {BRANCH_POINT}")
            run.rows.append(await ask(agent, prompt, strategy, PREFIX))

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
        for position, (name, _) in enumerate(BRANCH_DIALOGS):
            branch = next(item for item in await agent.branches() if item["name"] == name)
            await agent.switch(branch["id"])
            for number, (question, references) in enumerate(BRANCH_CHECKS, start=1):
                note(f"ветка «{name}»: контрольный вопрос {number} из {len(BRANCH_CHECKS)}")
                row = await ask(agent, question, strategy, name)
                row.reference = references[position]
                run.checks.append(row)

        run.turns = await agent.turns()

    await score(run)
    return run


async def score(run: Run) -> None:
    """Оценки судьи ставятся внутри прогона: сравнение потом только сводит их.

    Иначе сравнить два прогона было бы нельзя, не переспросив модель заново: в
    сессии лежит результат, а не сам разговор с судьёй.
    """
    for number, row in enumerate(run.checks, start=1):
        note(f"судья: контрольный вопрос {number} из {len(run.checks)}")
        row.score, row.comment, cost = await judge(row.prompt, row.reference, row.answer)
        run.judge_cost += cost


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


def report_turns(run: Run) -> None:
    """Ходы прогона: во что собрался контекст и во сколько встал вход."""
    out("### Расход токенов по ходам", "")

    rows: list[tuple[Any, ...]] = []
    for number, row in enumerate(run.rows, start=1):
        rows.append(
            (
                number,
                *([row.branch] if run.strategy == memory.BRANCHING else []),
                oneline(row.prompt)[:60],
                context_note(row),
                row.prompt_tokens,
                row.cached_tokens,
                money(row.cost),
            ),
        )
    for number, row in enumerate(run.checks, start=1):
        rows.append(
            (
                f"вопрос {number}",
                *([row.branch] if run.strategy == memory.BRANCHING else []),
                oneline(row.prompt)[:60],
                context_note(row),
                row.prompt_tokens,
                row.cached_tokens,
                money(row.cost),
            ),
        )

    headers = (
        "Ход",
        *(("Ветка",) if run.strategy == memory.BRANCHING else ()),
        "Вопрос",
        "Контекст",
        "Вход",
        "из кэша",
        "Стоимость",
    )
    out(*report.table(headers, rows))


def report_totals(run: Run) -> None:
    out("### Итоги прогона", "")

    rows: list[tuple[Any, ...]] = [
        ("Ходов", len(run.dialog)),
        ("из них контрольных вопросов", len(run.checks)),
        ("Входные токены", run.input_tokens),
        ("из них из кэша", run.cached_tokens),
        ("Выходные токены", run.output_tokens),
        ("Токены служебных запросов", run.service_tokens),
        ("Вход по цене без кэша", money(compare.miss_price(run.input_tokens))),
        ("Стоимость диалога", money(run.dialog_cost)),
        ("Стоимость служебных запросов", money(run.service_cost)),
        ("Работа судьи", money(run.judge_cost)),
        ("Всего", money(run.total_cost)),
    ]
    out(*report.table(("Показатель", "Значение"), rows))

    if run.strategy == memory.CLEAN:
        out(
            f"Стратегия: {run.label}. Историю никто не обрезал, а бюджет "
            f"{SCENARIO_BUDGET} токенов выбран так, чтобы весь диалог в запрос влезал.",
            "",
        )
    else:
        out(
            f"Стратегия: {run.label}. Окно {SCENARIO_WINDOW} сообщ., бюджет "
            f"{SCENARIO_BUDGET} токенов — контекст обрезала стратегия, а не бюджет.",
            "",
        )


def report_quality(run: Run) -> None:
    """Качество — не впечатление, а оценка судьи по каждому контрольному факту."""
    out("### Качество ответов на контрольные вопросы", "")

    rows: list[tuple[Any, ...]] = []
    for row in run.checks:
        rows.append(
            (
                oneline(row.prompt),
                *([row.branch] if run.strategy == memory.BRANCHING else []),
                row.score if row.score >= 0 else report.DASH,
                oneline(row.answer)[:160],
                row.comment[:80],
            ),
        )

    headers = (
        "Вопрос",
        *(("Ветка",) if run.strategy == memory.BRANCHING else ()),
        "Оценка",
        "Ответ агента",
        "Судья",
    )
    out(*report.table(headers, rows))
    out(f"Сумма оценок: {run.score} из {run.max_score}. Работа судьи: {money(run.judge_cost)}", "")


def report_facts(run: Run) -> None:
    out("### Что агент помнит вместо истории", "")

    rows: list[tuple[Any, ...]] = []
    for number, row in enumerate(run.dialog, start=1):
        if row.update is None:
            continue
        rows.append(
            (
                number,
                row.update.added or report.DASH,
                row.update.changed or report.DASH,
                row.update.removed or report.DASH,
                len(row.update.items),
                row.update.tokens,
                money(row.update.usage.cost_usd),
            ),
        )
    out(*report.table(("Ход", "Записано", "Уточнено", "Вычеркнуто", "Фактов", "Токенов", "Стоимость"), rows))

    final = next((row.update for row in reversed(run.dialog) if row.update is not None), None)
    if final is None:
        out("Картотека осталась пустой: модель не нашла в диалоге фактов.", "")
        return

    out("Картотека в конце диалога:", "")
    out(*report.table(("Ключ", "Значение"), [(key, value) for key, value in final.items]))


def report_branches(run: Run) -> None:
    """Branching: одно начало, два продолжения, и они друг о друге не знают."""
    out("### Ветки от одного checkpoint", "")

    prefix = [row for row in run.rows if row.branch == PREFIX]
    out(
        f"Общее начало — {len(prefix)} ходов, checkpoint на последней реплике. "
        f"От него отходят две ветки по {len(BRANCH_DIALOGS[0][1])} хода, "
        f"в каждой те же {len(BRANCH_CHECKS)} контрольных вопроса.",
        "",
    )

    branch_cost = sum(row.cost for row in run.rows if row.branch != PREFIX)
    prefix_cost = sum(row.cost for row in prefix)
    out(
        f"Общее начало стоило {money(prefix_cost)} и проиграно один раз на две ветки. "
        f"Продолжения обошлись в {money(branch_cost)}, весь прогон со служебными "
        f"запросами и судьёй — {money(run.total_cost)}. Двумя отдельными диалогами то же "
        f"сравнение стоило бы примерно {money(prefix_cost + run.total_cost)}.",
        "",
    )


async def clean() -> Run:
    """Точка отсчёта: вся история уходит в модель, ничего не забывается."""
    out(f"## {MODES[memory.CLEAN]}", "")
    out(
        f"Диалог из {len(DIALOG)} ходов плюс {len(CHECKS)} контрольных вопросов без всякого "
        "управления контекстом: в каждый запрос уходит вся история целиком. "
        "Это верхняя граница качества и расхода — с ней и сравниваются стратегии.",
        "",
    )
    run = await play("clean", memory.CLEAN)
    report_turns(run)
    report_totals(run)
    report_quality(run)
    return run


async def window() -> Run:
    out(f"## {MODES[memory.WINDOW]}", "")
    out(
        f"Тот же диалог, но в запрос идут только последние {SCENARIO_WINDOW} сообщений — "
        "три последних хода. Всё, что старше, агент не видит.",
        "",
    )
    run = await play("window", memory.WINDOW)
    report_turns(run)
    report_totals(run)
    report_quality(run)
    return run


async def facts() -> Run:
    out(f"## {MODES[memory.FACTS]} / Key-Value Memory", "")
    out(
        f"То же окно из {SCENARIO_WINDOW} сообщений плюс картотека, которую агент "
        "обновляет после каждого хода: за это платится один служебный запрос на ход.",
        "",
    )
    run = await play("facts", memory.FACTS)
    report_turns(run)
    report_totals(run)
    report_quality(run)
    report_facts(run)
    return run


async def branches() -> Run:
    out(f"## {MODES[memory.BRANCHING]}", "")
    out(
        f"{BRANCH_POINT} ходов общего начала, checkpoint на последней реплике и две ветки "
        f"по {len(BRANCH_DIALOGS[0][1])} хода от одного места. Контрольные вопросы задаются "
        "в каждой ветке: у них одни формулировки и разные верные ответы.",
        "",
    )
    run = await play_branches()
    report_branches(run)
    report_turns(run)
    report_totals(run)
    report_quality(run)
    return run


@dataclass(frozen=True)
class Scenario:
    """Сценарий вместе с ценником: страница показывает его до нажатия кнопки."""

    run: Callable[[], Awaitable[Run]]
    title: str
    about: str
    # Сколько запросов уйдёт в модель: прогон платный, и знать это лучше заранее.
    requests: int


# Ходов в прогоне: диалог плюс контрольные вопросы. В стратегии фактов каждый ход
# тянет за собой ещё один запрос — разбор реплики на факты. Плюс заголовок диалога
# и по одному вызову судьи на контрольный вопрос.
TURNS = len(DIALOG) + len(CHECKS)
BRANCH_TURNS = BRANCH_POINT + sum(len(prompts) for _, prompts in BRANCH_DIALOGS)
BRANCH_QUESTIONS = len(BRANCH_DIALOGS) * len(BRANCH_CHECKS)

SCENARIOS = {
    "clean": Scenario(
        clean,
        MODES[memory.CLEAN],
        "точка отсчёта: вся история в каждом запросе, ничего не забыто",
        requests=TURNS + 1 + len(CHECKS),
    ),
    "window": Scenario(
        window,
        MODES[memory.WINDOW],
        f"только последние {SCENARIO_WINDOW} сообщений в запросе",
        requests=TURNS + 1 + len(CHECKS),
    ),
    "facts": Scenario(
        facts,
        f"{MODES[memory.FACTS]} / Key-Value Memory",
        "окно плюс картотека, которую агент обновляет после каждого хода",
        requests=2 * TURNS + 1 + len(CHECKS),
    ),
    "branches": Scenario(
        branches,
        MODES[memory.BRANCHING],
        "общее начало и две ветки от одного места с контрольными вопросами",
        requests=BRANCH_TURNS + BRANCH_QUESTIONS + 1 + BRANCH_QUESTIONS,
    ),
}


async def main(names: list[str], result: Path | None) -> None:
    out(f"Модель: {MODEL}, тариф {'пиковый' if tokens.is_peak_hour() else 'непиковый'}", "")

    runs = [await SCENARIOS[name].run() for name in names]

    # Результат нужен только тому, кто собирается сравнивать прогоны, — странице.
    # Из терминала сценарий запускают за отчётом, и файла тогда никто не просит.
    if result is not None and runs:
        result.write_text(runs[-1].result().model_dump_json(), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="*", choices=sorted(SCENARIOS), default=sorted(SCENARIOS))
    parser.add_argument(
        "--result",
        type=Path,
        default=None,
        help="куда сложить машинный результат прогона (JSON) для сравнения",
    )
    arguments = parser.parse_args()
    asyncio.run(main(list(arguments.scenarios), arguments.result))
    sys.exit(0)
