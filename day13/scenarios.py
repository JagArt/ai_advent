"""Замер для README: как задача проходит автомат и что даёт блок состояния.

    python day13/scenarios.py

Два прогона, оба в отдельной временной базе — замер не попадает в историю приложения.
Решения в обоих принимает агент: прогон не может пойти и нажать кнопку в карточке,
поэтому автосохранение включено и переходы применяются сразу.

Первый: один и тот же разговор дважды — с блоком состояния в запросе и без него.
Автомат ведётся в обоих: он в базе, а не в промпте, и разница между прогонами
объясняется только тем, видит модель своё состояние или нет. Меряется трижды.
Объективно — сколько переходов агент предложил сам и сколько предложений отбросил
guard. Судьёй — спрашивает ли агент про ожидаемое действие текущего шага и не
забегает ли вперёд этапа. И журналом переходов, который показывает путь целиком.

После разговора задача доводится до конца кнопкой, как это сделал бы пользователь.
Это не подпорка к замеру, а вторая его половина: guard разрешает переход, а не
приказывает его сделать, и разница между «можно» и «пора» остаётся за человеком.

Второй: отклонённые предложения из обоих прогонов в одной таблице. Это ответ на
вопрос, зачем нужен guard, если модель и так видит своё состояние.
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

import task as task_state
from agent import COMPARE_MAX_TOKENS, Agent, AgentPlan, AgentProposals, AgentTurn
from demo import DIALOG, RUN_CHECK, RUN_PROFILE, STAGE_TURNS
from llm import MODEL, complete
from memory import WINDOW_MESSAGES
from remember import SCALE
from storage import Storage

# Окно то же, что в чате: шесть реплик, три последних хода.
SCENARIO_WINDOW = WINDOW_MESSAGES

# Лимит ответа тот же, что в сравнении профилей: обрезка портила бы не только текст,
# но и разбор хода — по половине ответа переход не оценить.
SCENARIO_MAX_TOKENS = COMPARE_MAX_TOKENS

JUDGE_PROMPT = """\
Ты проверяешь, держится ли ассистент состояния задачи.
Тебе дают этап, текущий шаг, ожидаемое от пользователя действие и ответ ассистента.
Оценивай только работу на шаге, а не правильность ответа по существу:
2 — ответ работает на текущем шаге: добивается именно ожидаемого действия и не
    забегает вперёд этапа;
1 — работает частично: сползает на соседние темы или касается будущего этапа мимоходом;
0 — ответ не про текущий шаг: занят другим этапом или сам объявляет переход.
Считай нарушением, если ассистент сам объявляет смену этапа или называет задачу
готовой: этап двигается переходом, а не его словами.
Верни строго JSON без пояснений и без markdown: {"score": 2, "comment": "коротко, что не так"}
"""

JUDGE_MAX_TOKENS = 200


@dataclass
class Turn:
    """Ход разговора: где стояла задача, что ответил агент и куда автомат двинулся."""

    prompt: str
    answer: str
    # Состояние до хода: именно оно уходило в запрос вместе с вопросом.
    stage: str
    step: str
    expected: str
    # Состояние после разбора хода: пусто, если не двигались.
    moved: str = ""
    # Предложение, которое не прошло по графу или по условию.
    rejected: str = ""
    found: tuple[str, ...] = ()
    score: int = -1
    comment: str = ""

    @property
    def line(self) -> str:
        return f"{self.stage} · {self.step}" if self.step else self.stage


@dataclass
class Run:
    """Прогон разговора: ходы, журнал переходов и состояние на выходе."""

    stateful: bool
    turns: list[Turn] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    # Переходы, которые после разговора остались открытыми, но так и не были
    # предложены: их доводит пользователь. Строгий guard разрешает переход, а не
    # требует его, и разница между «можно» и «пора» — это как раз они.
    manual: list[str] = field(default_factory=list)
    final: str = ""
    check: Turn | None = None

    @property
    def name(self) -> str:
        return "с блоком состояния" if self.stateful else "без блока состояния"

    @property
    def moves(self) -> int:
        return sum(1 for turn in self.turns if turn.moved)

    @property
    def rejected(self) -> list[Turn]:
        return [turn for turn in self.turns if turn.rejected]

    @property
    def score(self) -> int:
        return sum(turn.score for turn in self.turns if turn.score >= 0)

    @property
    def scored(self) -> int:
        return sum(1 for turn in self.turns if turn.score >= 0)


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


def plural(count: int, one: str, few: str, many: str) -> str:
    if count % 100 in range(11, 15):
        return many
    if count % 10 == 1:
        return one
    return few if count % 10 in (2, 3, 4) else many


def quote(text: str) -> None:
    for line in text.strip().splitlines():
        print(f"> {line}" if line.strip() else ">")
    print()


def short(text: str, limit: int = 60) -> str:
    body = oneline(text)
    return body if len(body) <= limit else f"{body[:limit - 1]}…"


def stage_name(key: str) -> str:
    stage = task_state.BY_KEY.get(key)
    return stage.name if stage else key


def place(stage: str, step: str) -> str:
    """Пара «этап · шаг». У готового шага нет, и приписывать ему прочерк незачем."""
    name = stage_name(stage)
    return f"{name} · {step}" if step else name


def where(plan: AgentPlan) -> tuple[str, str, str]:
    return stage_name(plan.stage), plan.step, plan.expected


async def judge(turn: Turn) -> tuple[int, str]:
    """Работу на шаге оценивает модель: у неё перед глазами и шаг, и ответ."""
    request: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=JUDGE_PROMPT),
        ChatCompletionUserMessageParam(
            role="user",
            content=(
                f"Этап: {turn.stage}\n"
                f"Текущий шаг: {turn.step or '—'}\n"
                f"Ожидаемое действие: {turn.expected}\n\n"
                f"Реплика пользователя:\n{turn.prompt}\n\n"
                f"Ответ ассистента:\n{turn.answer}"
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


async def ask(agent: Agent, prompt: str) -> Turn:
    """Один ход через того же агента, что и в веб-чате: события те же."""
    plan: AgentPlan | None = None
    turn: AgentTurn | None = None
    proposals: AgentProposals | None = None
    parts: list[str] = []

    async for event in agent.ask(prompt, SCENARIO_WINDOW, autosave=True):
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

    found = []
    for item in proposals.items if proposals else ():
        if item.kind == SCALE:
            found.append(f"профиль · шкала {item.scale}: {item.value}")
        else:
            found.append(f"{item.tier} · {item.section}: {item.text}")

    move = proposals.move if proposals else None
    stage, step, expected = where(plan)
    return Turn(
        prompt=prompt,
        answer="".join(parts),
        stage=stage,
        step=step,
        expected=expected,
        moved=move.name if move else "",
        rejected=proposals.rejected if proposals else "",
        found=tuple(found),
    )


async def finish(agent: Agent) -> list[str]:
    """Довести задачу до конца так, как это сделал бы пользователь кнопкой.

    Здесь и видно, что guard разрешает переход, а не приказывает его сделать: если
    условия выполнены, а модель двинуться не предложила, дальше идёт человек. Ходит
    он по тому же графу и через ту же проверку — прогон не умеет обойти правила,
    даже когда ему это удобно.
    """
    applied = []
    # Потолок на всякий случай: у графа семь рёбер вперёд, и цикл здесь означал бы
    # ошибку в самом графе, а не долгую задачу.
    for _ in range(len(task_state.STAGES) * 4):
        state = (await agent.memory())["task"]
        forward = [
            move for move in state["moves"] if not move["blocked"] and not move["back"]
        ]
        if not forward:
            return applied

        move = forward[0]
        if move["kind"] == task_state.STAGE:
            await agent.move(stage=move["stage"])
        else:
            await agent.move(step=move["step"])
        applied.append(move["name"])
    return applied


async def walk(storage: Storage, stateful: bool) -> Run:
    """Разговор, который проходит автомат: одни и те же реплики, один и тот же профиль."""
    run = Run(stateful=stateful)
    session_id = await storage.create_session(RUN_PROFILE)
    agent = Agent(
        session_id,
        storage,
        max_tokens=SCENARIO_MAX_TOKENS,
        window_messages=SCENARIO_WINDOW,
        stateful=stateful,
    )

    for prompt in DIALOG:
        run.turns.append(await ask(agent, prompt))

    # Контрольный ход — в том же диалоге, после того как разговор закончился: агент
    # не должен начинать новый круг требований сам, что бы ни стояло в состоянии.
    run.check = await ask(agent, RUN_CHECK)

    run.manual = await finish(agent)

    state = await storage.load_task(session_id)
    run.final = task_state.build(state["stage"], state["step"]).line
    run.transitions = await storage.load_transitions(session_id)

    verdicts = await asyncio.gather(*(judge(turn) for turn in run.turns))
    for turn, (score, comment) in zip(run.turns, verdicts):
        turn.score, turn.comment = score, comment
    return run


def report_walk(run: Run) -> None:
    print(f"### Прогон {run.name}\n")
    rows: list[tuple[Any, ...]] = []
    for number, turn in enumerate(run.turns, start=1):
        rows.append(
            (
                number,
                short(turn.prompt, 52),
                turn.line,
                short(turn.expected, 40),
                turn.moved or (f"отброшен: {short(turn.rejected, 34)}" if turn.rejected else "—"),
                f"{turn.score}/2",
            )
        )
    if run.check is not None:
        rows.append(
            (
                "контрольный",
                short(run.check.prompt, 52),
                run.check.line,
                short(run.check.expected, 40),
                run.check.moved or "—",
                "—",
            )
        )
    table(("Ход", "Реплика", "Этап · шаг до хода", "Ожидаемое действие", "Переход", "Судья"), rows)

    sentences = [
        f"Итог: за разговор {run.moves}"
        f" {plural(run.moves, 'переход', 'перехода', 'переходов')} по предложению агента,"
        f" отброшено предложений {len(run.rejected)}, судья {run.score}/{2 * run.scored}."
    ]
    if run.manual:
        sentences.append(
            f"После разговора условия остались выполненными ещё для {len(run.manual)}"
            f" {plural(len(run.manual), 'перехода', 'переходов', 'переходов')},"
            f" и до конца задачу довёл пользователь: {'; '.join(run.manual)}."
        )
    sentences.append(f"Задача дошла до **{run.final}**.")
    print(f"{' '.join(sentences)}\n")


def report_transitions(run: Run) -> None:
    print(f"### Журнал переходов, {run.name}\n")
    if not run.transitions:
        print("Автомат не двинулся ни разу.\n")
        return
    table(
        ("№", "Откуда", "Куда", "Кто", "Почему"),
        [
            (
                number,
                place(row["from_stage"], row["from_step"]),
                place(row["to_stage"], row["to_step"]),
                "агент" if row["origin"] == "agent" else "пользователь",
                short(str(row["why"] or ""), 54) or "—",
            )
            for number, row in enumerate(run.transitions, start=1)
        ],
    )


def report_compare(runs: list[Run]) -> None:
    print("## Блок состояния в запросе: с ним и без него\n")
    print(
        f"Один и тот же разговор из {len(DIALOG)} реплик, один и тот же профиль,"
        " одна и та же долговременная память. Автомат ведётся в обоих прогонах —"
        " различается только то, видит ли модель своё состояние в запросе.\n"
    )
    table(
        ("Прогон", "Переходов агентом", "Осталось пользователю", "Отброшено guard'ом", "Дошли до", "Судья"),
        [
            (
                run.name,
                run.moves,
                len(run.manual),
                len(run.rejected),
                run.final,
                f"{run.score}/{2 * run.scored}",
            )
            for run in runs
        ],
    )

    print("### Замечания судьи\n")
    remarks = [
        (run, turn)
        for run in runs
        for turn in run.turns
        if turn.score < 2 and turn.comment
    ]
    if not remarks:
        print("Судья не нашёл нарушений ни в одном ходу.\n")
    else:
        table(
            ("Прогон", "Ход", "Этап · шаг", "Оценка", "Что не так"),
            [
                (
                    run.name,
                    run.turns.index(turn) + 1,
                    turn.line,
                    f"{turn.score}/2",
                    turn.comment,
                )
                for run, turn in remarks
            ],
        )


def report_rejected(runs: list[Run]) -> None:
    print("## Отклонённые переходы\n")
    print(
        "Предложения, которые не стали карточкой: перехода нет в графе или его условие"
        " не выполнено. Отбрасываются они целиком — состояние, которое двинулось не по"
        " правилам, хуже состояния, которое стоит.\n"
    )
    rows = [
        (run.name, run.turns.index(turn) + 1, turn.line, turn.rejected)
        for run in runs
        for turn in run.rejected
    ]
    if not rows:
        print("Ни одного: модель не предложила ни одного недопустимого перехода.\n")
        return
    table(("Прогон", "Ход", "Этап · шаг", "Почему отброшено"), rows)


def report_answers(run: Run) -> None:
    """Три ответа целиком: границы этапов — то место, где состояние видно в тексте."""
    print(f"## Ответы на границах этапов, {run.name}\n")
    for number in STAGE_TURNS:
        if number > len(run.turns):
            continue
        turn = run.turns[number - 1]
        print(f"**Ход {number}** — {turn.line}, ожидалось: {turn.expected}\n")
        print(f"> {oneline(turn.prompt)}\n")
        quote(turn.answer)
        print(f"Переход: {turn.moved or 'нет'}\n")

    if run.check is not None:
        print("### Контрольный ход: новое требование в конце разговора\n")
        print(f"> {oneline(run.check.prompt)}\n")
        quote(run.check.answer)


async def main() -> None:
    print("# Прогон day13 — состояние задачи\n")
    print(
        f"Модель {MODEL}, ответы не длиннее {SCENARIO_MAX_TOKENS} ток., окно"
        f" {SCENARIO_WINDOW} сообщ., профиль «{RUN_PROFILE}». Во время разговора находки"
        " и переходы применяет агент: к кнопке в карточке прогон не пойдёт. После"
        " разговора остаток пути прогон проходит кнопкой — той же ручкой и с той же"
        " проверкой, что и человек.\n"
    )

    with tempfile.TemporaryDirectory() as directory:
        storage = Storage(Path(directory) / "state.db")
        await storage.init()

        # Прогоны идут по очереди, а не одновременно: у них общая долговременная
        # память, и параллельный ход дописывал бы в неё то, чего второй ещё не видел.
        runs = [await walk(storage, stateful=True), await walk(storage, stateful=False)]

    report_compare(runs)
    for run in runs:
        report_walk(run)
        report_transitions(run)
    report_rejected(runs)
    report_answers(runs[0])


if __name__ == "__main__":
    asyncio.run(main())
