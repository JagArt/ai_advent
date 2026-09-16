"""Замер для README: сбор ТЗ за 12 ходов и три уровня памяти по ходу разговора.

    python day11/scenarios.py

Диалог идёт через того же агента, что и веб-чат, но в отдельной временной базе:
замер не попадает в историю приложения. Уровень для каждой находки выбирает сам
агент — в браузере это делает пользователь кнопкой, но прогон идти к кнопке не
может, поэтому включено автосохранение.

В конце те же контрольные вопросы задаются дважды: агенту с памятью и агенту с
той же лентой, но с пустыми блоками. Второй диалог заново не проходит — реплики
первого прогона кладутся ему в базу как есть, без обращения к модели.
"""

import asyncio
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from agent import SYSTEM_PROMPT, Agent, AgentPlan, AgentProposals, AgentTurn
from demo import DIALOG
from llm import MODEL, complete
from memory import LONGTERM, SECTIONS, WINDOW_MESSAGES, WORKING
from storage import Storage

# Ответы в замере короткие: сравнивать надо память агента, а не его словоохотливость.
# Заодно лента растёт предсказуемо, и таблица роста читается построчно.
SCENARIO_SYSTEM = SYSTEM_PROMPT + "Отвечай по делу и не длиннее трёх предложений.\n"
SCENARIO_MAX_TOKENS = 400

# Окно то же, что в чате: шесть реплик, три последних хода. Всё, что сказано
# раньше, к концу диалога живёт только в блоках памяти — или не живёт нигде.
SCENARIO_WINDOW = WINDOW_MESSAGES

# Контрольные вопросы задаются после диалога. Ответы на первые два лежат в ходах
# 2, 5, 6, 7 и 10 — к этому моменту они давно за окном, и взять их можно только из
# блоков памяти. Третий проверяет seed: его в разговоре не было вовсе.
CHECKS = (
    (
        "Напомни: в каких форматах нужен отчёт и сколько строк максимум за один запрос?",
        "CSV и XLSX, не больше 100 тысяч строк",
    ),
    (
        "Что мы решили про хранение готовых файлов и про доступ к выгрузке?",
        "S3-совместимое хранилище, подписанная ссылка на час, хранить семь дней, "
        "доступ только роли warehouse_manager",
    ),
    (
        "Какой у меня стек и как мы выкатываем релизы?",
        "Python 3.12, FastAPI, PostgreSQL; релизы по вторникам через GitLab CI в Kubernetes",
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


@dataclass
class Row:
    """Один ход: что уходило в модель и что агент из него запомнил."""

    prompt: str
    answer: str
    history_messages: int
    context_messages: int
    longterm_count: int
    working_count: int
    found: tuple[tuple[str, str, str], ...] = ()


@dataclass
class Run:
    """Прогон целиком: ходы, контрольные ответы и память на выходе."""

    rows: list[Row] = field(default_factory=list)
    checks: list[Row] = field(default_factory=list)
    longterm: list[dict[str, Any]] = field(default_factory=list)
    working: list[dict[str, Any]] = field(default_factory=list)

    @property
    def dialog(self) -> list[Row]:
        return [*self.rows, *self.checks]


def cell(value: Any) -> str:
    # Таблица идёт прямиком в README, а в ответах модели попадаются вертикальные
    # черты: неэкранированная сломала бы разметку.
    return str(value).replace("|", "\\|")


def table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    print(f"| {' | '.join(headers)} |")
    print(f"| {' | '.join('---' for _ in headers)} |")
    for row in rows:
        print(f"| {' | '.join(cell(value) for value in row)} |")
    print()


def oneline(text: str) -> str:
    return " ".join(text.split())


async def ask(agent: Agent, prompt: str, autosave: bool = True) -> Row:
    """Один ход через того же агента, что и в веб-чате: события те же."""
    plan: AgentPlan | None = None
    turn: AgentTurn | None = None
    proposals: AgentProposals | None = None
    parts: list[str] = []

    async for event in agent.ask(prompt, SCENARIO_WINDOW, autosave):
        if isinstance(event, AgentPlan):
            plan = event
        elif isinstance(event, AgentTurn):
            turn = event
        elif isinstance(event, AgentProposals):
            proposals = event
        elif event.content:
            parts.append(event.content)

    if plan is None or turn is None:
        raise RuntimeError("Ход не дошёл до конца")

    return Row(
        prompt=prompt,
        answer="".join(parts),
        history_messages=plan.history_messages,
        context_messages=plan.context_messages,
        longterm_count=plan.longterm_count,
        working_count=plan.working_count,
        found=tuple(
            (item.text, item.tier, item.section)
            for item in (proposals.items if proposals else ())
        ),
    )


async def play(directory: Path) -> tuple[Run, Storage, str]:
    """Диалог и контрольные вопросы с полной памятью."""
    run = Run()
    storage = Storage(directory / "memory.db")
    await storage.init()
    session_id = await storage.create_session()
    agent = Agent(
        session_id,
        storage,
        system_prompt=SCENARIO_SYSTEM,
        max_tokens=SCENARIO_MAX_TOKENS,
        window_messages=SCENARIO_WINDOW,
    )

    for prompt in DIALOG:
        run.rows.append(await ask(agent, prompt))
    for question, _ in CHECKS:
        run.checks.append(await ask(agent, question))

    run.longterm = await storage.load_longterm()
    run.working = await storage.load_working(session_id)
    return run, storage, session_id


async def replay(directory: Path, rows: list[Row]) -> Run:
    """Тот же диалог, но у агента только окно: блоки памяти пусты.

    Реплики кладутся в базу как есть — отвечать на них второй раз незачем. К
    модели этот прогон обращается только за контрольными вопросами, и в этом весь
    смысл: лента одна и та же, память разная.
    """
    run = Run()
    storage = Storage(directory / "window.db")
    await storage.init()

    # Долговременная память заливается при создании базы, и для этого прогона её
    # нужно убрать: сравниваем агента с памятью и агента без неё.
    for item in await storage.load_longterm():
        await storage.delete_longterm(item["id"])

    session_id = await storage.create_session()
    for row in rows:
        await storage.save_turn(session_id, row.prompt, row.answer)

    agent = Agent(
        session_id,
        storage,
        system_prompt=SCENARIO_SYSTEM,
        max_tokens=SCENARIO_MAX_TOKENS,
        window_messages=SCENARIO_WINDOW,
    )
    for question, _ in CHECKS:
        # autosave=False: записывать этому агенту всё равно некуда — вопрос в том,
        # что он помнит, а не в том, что он успеет запомнить.
        run.checks.append(await ask(agent, question, autosave=False))

    return run


async def judge(question: str, reference: str, answer: str) -> tuple[int, str]:
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

    text = verdict.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        parsed = json.loads(text)
        return int(parsed["score"]), oneline(str(parsed.get("comment", "")))
    except (ValueError, KeyError, TypeError):
        # Судья ответил не по форме: замер от этого не рушится, но и оценки нет.
        return -1, f"ответ судьи не разобран: {oneline(verdict.text)[:80]}"


def report_growth(run: Run) -> None:
    """Главная таблица дня: три уровня по ходам, каждый со своим ростом."""
    print("### Как растут три уровня\n")

    rows: list[tuple[Any, ...]] = []
    for number, row in enumerate(run.dialog, start=1):
        label = number if number <= len(DIALOG) else f"вопрос {number - len(DIALOG)}"
        rows.append(
            (
                label,
                f"{row.history_messages} сообщ.",
                f"{row.context_messages} сообщ.",
                f"{row.working_count} п.",
                f"{row.longterm_count} п.",
            )
        )

    table(("Ход", "Лента до хода", "Окно в запросе", "Рабочая", "Долговременная"), rows)
    print(
        f"Лента растёт на два сообщения за ход, окно упирается в {SCENARIO_WINDOW} и дальше не "
        "двигается, а блоки памяти растут только тогда, когда в ходе было что записать.\n"
    )


def report_found(run: Run) -> None:
    """Что агент выудил из каждого хода и на какой уровень отправил."""
    print("### Что агент предложил запомнить\n")

    rows: list[tuple[Any, ...]] = []
    for number, row in enumerate(run.dialog, start=1):
        label = number if number <= len(DIALOG) else f"вопрос {number - len(DIALOG)}"
        if not row.found:
            rows.append((label, "—", "—"))
            continue
        for index, (text, tier, section) in enumerate(row.found):
            rows.append((label if index == 0 else "", text, f"{tier} · {section}"))

    table(("Ход", "Формулировка", "Уровень"), rows)


def render_memory(title: str, tier: str, items: list[dict[str, Any]]) -> None:
    print(f"### {title}\n")
    if not items:
        print("Пусто.\n")
        return

    for section in SECTIONS[tier]:
        chosen = [item for item in items if item["section"] == section]
        if not chosen:
            continue
        print(f"**{section}**\n")
        for item in chosen:
            mark = " *(до разговора)*" if item["origin"] == "seed" else ""
            print(f"- {item['text']}{mark}")
        print()


def report_checks(
    memory_run: Run,
    window_run: Run,
    scores: list[tuple[tuple[int, str], tuple[int, str]]],
) -> None:
    """Один и тот же вопрос двум агентам: лента общая, память разная."""
    print("### Контрольные вопросы\n")

    rows: list[tuple[Any, ...]] = []
    for index, (question, reference) in enumerate(CHECKS):
        (with_score, with_comment), (without_score, without_comment) = scores[index]
        rows.append(
            (
                question,
                reference,
                f"{with_score}/2 — {with_comment}" if with_comment else f"{with_score}/2",
                f"{without_score}/2 — {without_comment}" if without_comment else f"{without_score}/2",
            )
        )

    table(("Вопрос", "Эталон", "С памятью", "Только окно"), rows)

    print("Ответы целиком:\n")
    for index, (question, _) in enumerate(CHECKS):
        print(f"**{question}**\n")
        print(f"- с памятью: {oneline(memory_run.checks[index].answer)}")
        print(f"- только окно: {oneline(window_run.checks[index].answer)}\n")


async def main() -> None:
    print("# Прогон day11 — модель памяти агента\n")
    print(
        f"Модель {MODEL}, окно {SCENARIO_WINDOW} сообщ., ответы не длиннее "
        f"{SCENARIO_MAX_TOKENS} ток. "
        f"Уровень для находок выбирает агент: прогон идти к кнопке не может.\n"
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        memory_run, _, _ = await play(path)
        window_run = await replay(path, memory_run.rows)

    scores = []
    for index, (question, reference) in enumerate(CHECKS):
        with_memory = await judge(question, reference, memory_run.checks[index].answer)
        without = await judge(question, reference, window_run.checks[index].answer)
        scores.append((with_memory, without))

    report_growth(memory_run)
    report_found(memory_run)
    render_memory("Рабочая память после прогона — готовое ТЗ", WORKING, memory_run.working)
    render_memory("Долговременная память после прогона", LONGTERM, memory_run.longterm)
    report_checks(memory_run, window_run, scores)


if __name__ == "__main__":
    asyncio.run(main())
