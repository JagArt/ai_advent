"""Замер для README: одинаковые вопросы разным профилям и профиль, который учится.

    python day12/scenarios.py

Два прогона, оба в отдельной временной базе — замер не попадает в историю приложения.

Первый: три вопроса задаются каждому профилю. Вопрос один и тот же, долговременная
память одна и та же, диалога нет вовсе — отличается только блок профиля, поэтому
разницу в ответах больше нечем объяснить. Форму ответа считает shape.py, соблюдение
профиля оценивает модель-судья: у неё перед глазами и требования, и ответ.

Второй: разговор, в котором пользователь на третьем ходу просит отвечать короче, а
на четвёртом — без списков. Обе просьбы уходят в профиль правкой шкал, и дальше
видно, что агент держит новую форму сам. Контрольный вопрос задаётся в новом
диалоге того же профиля: правка не должна остаться внутри того разговора, где её
сделали. Уровень для находок выбирает сам агент — прогон не может пойти и нажать
кнопку в карточке, поэтому включено автосохранение.
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

import shape
from agent import (
    COMPARE_MAX_TOKENS,
    Agent,
    AgentPlan,
    AgentProposals,
    AgentTurn,
    answer_as,
)
from demo import COMPARE, DIALOG, LESSON_CHECK, LESSON_PROFILE, LESSON_TURNS
from llm import MODEL, complete
from memory import LONGTERM, WINDOW_MESSAGES, block
from profile import BY_KEY, SCALE_KEYS, Profile
from profile import build as build_profile
from remember import SCALE
from storage import Storage

# Окно то же, что в чате: шесть реплик, три последних хода.
SCENARIO_WINDOW = WINDOW_MESSAGES

# Лимит ответа высокий намеренно: обрезка испортила бы главную мерку — длину, — и
# профиль «подробно» проиграл бы не по своей вине. Системный промпт тоже не тронут:
# любая приписка вроде «отвечай коротко» спорила бы с профилем, а проверяем мы его.
SCENARIO_MAX_TOKENS = COMPARE_MAX_TOKENS

JUDGE_PROMPT = """\
Ты проверяешь, отвечает ли ассистент так, как требует профиль пользователя.
Тебе дают требования профиля и ответ ассистента.
Оценивай только соблюдение требований — тон, длину, формат, уровень объяснений,
обращение и прямые запреты, — а не правильность ответа по существу:
2 — требования соблюдены;
1 — часть соблюдена, часть нарушена;
0 — ответ написан вопреки профилю.
Верни строго JSON без пояснений и без markdown: {"score": 2, "comment": "коротко, что не так"}
"""

JUDGE_MAX_TOKENS = 200


@dataclass
class Answer:
    """Ответ одного профиля на один вопрос."""

    question: int
    profile: Profile
    text: str
    score: int = -1
    comment: str = ""

    @property
    def shape(self) -> shape.Shape:
        return shape.measure(self.text)


@dataclass
class Turn:
    """Ход разговора: что ответил агент и что после него ушло в профиль и в память."""

    prompt: str
    answer: str
    found: tuple[str, ...] = ()

    @property
    def shape(self) -> shape.Shape:
        return shape.measure(self.answer)


@dataclass
class Lesson:
    """Прогон с обучением: ходы, профиль до и после, контрольный вопрос."""

    turns: list[Turn] = field(default_factory=list)
    before: dict[str, str] = field(default_factory=dict)
    after: dict[str, str] = field(default_factory=dict)
    check: Turn | None = None


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


def quote(text: str) -> None:
    """Ответ цитатой, а не строкой: в этом дне важна форма, и списки нельзя схлопывать."""
    for line in text.strip().splitlines():
        print(f"> {line}" if line.strip() else ">")
    print()


def scales_line(profile: Profile) -> str:
    return " · ".join(profile.value(key) for key in SCALE_KEYS)


async def judge(profile: Profile, answer: str) -> tuple[int, str]:
    """Соблюдение профиля оценивает модель: у неё перед глазами и требования, и ответ."""
    request: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=JUDGE_PROMPT),
        ChatCompletionUserMessageParam(
            role="user",
            content=f"Требования профиля:\n{profile.text}\n\nОтвет ассистента:\n{answer}",
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


async def load_profiles(storage: Storage) -> list[Profile]:
    rows = await storage.list_profiles()
    loaded = [await storage.load_profile(str(row["id"])) for row in rows]
    return [build_profile(row) for row in loaded if row]


async def compare(storage: Storage) -> list[Answer]:
    """Один вопрос — все профили. Ни диалога, ни ТЗ: сравнивать можно только равное."""
    longterm = block(LONGTERM, await storage.load_longterm())
    profiles = await load_profiles(storage)

    answers = []
    for number, question in enumerate(COMPARE, start=1):
        # Профили спрашиваются одновременно: ответы независимы, а ждать их по
        # очереди на трёх профилях и трёх вопросах — девять запросов подряд.
        texts = await asyncio.gather(
            *(
                answer_as(person, longterm, question, max_tokens=SCENARIO_MAX_TOKENS)
                for person in profiles
            )
        )
        answers.extend(
            Answer(question=number, profile=person, text=text)
            for person, text in zip(profiles, texts)
        )

    verdicts = await asyncio.gather(*(judge(item.profile, item.text) for item in answers))
    for item, (score, comment) in zip(answers, verdicts):
        item.score, item.comment = score, comment
    return answers


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

    return Turn(prompt=prompt, answer="".join(parts), found=tuple(found))


async def lesson(storage: Storage) -> Lesson:
    """Разговор, в котором пользователь меняет форму ответа, а профиль это запоминает."""
    run = Lesson()
    loaded = await storage.load_profile(LESSON_PROFILE)
    if loaded is None:
        raise RuntimeError(f"Профиля {LESSON_PROFILE} нет в базе")
    run.before = {key: build_profile(loaded).value(key) for key in SCALE_KEYS}

    session_id = await storage.create_session(LESSON_PROFILE)
    agent = Agent(
        session_id,
        storage,
        max_tokens=SCENARIO_MAX_TOKENS,
        window_messages=SCENARIO_WINDOW,
    )
    for prompt in DIALOG:
        run.turns.append(await ask(agent, prompt))

    after = await storage.load_profile(LESSON_PROFILE)
    run.after = {key: build_profile(after).value(key) for key in SCALE_KEYS} if after else {}

    # Контрольный вопрос — в новом диалоге того же профиля: правка предпочтения не
    # принадлежит разговору, в котором её сделали, и это надо показать, а не обещать.
    fresh_id = await storage.create_session(LESSON_PROFILE)
    fresh = Agent(
        fresh_id,
        storage,
        max_tokens=SCENARIO_MAX_TOKENS,
        window_messages=SCENARIO_WINDOW,
    )
    run.check = await ask(fresh, LESSON_CHECK)
    return run


def report_compare(answers: list[Answer]) -> None:
    print("## Один вопрос — три профиля\n")
    print("Вопросы:\n")
    for number, question in enumerate(COMPARE, start=1):
        print(f"{number}. {question}")
    print()

    print("### Форма ответов\n")
    rows: list[tuple[Any, ...]] = []
    for item in answers:
        measured = item.shape
        rows.append(
            (
                item.question,
                item.profile.name,
                scales_line(item.profile),
                measured.words,
                measured.bullets or "—",
                measured.tables or "—",
                measured.address,
                f"{item.score}/2",
            )
        )
    table(
        ("№", "Профиль", "Шкалы", "Слов", "Пунктов", "Строк табл.", "Обращение", "Судья"),
        rows,
    )

    print("### Итог по профилям\n")
    totals: list[tuple[Any, ...]] = []
    for profile in dict.fromkeys(item.profile.id for item in answers):
        chosen = [item for item in answers if item.profile.id == profile]
        shapes = [item.shape for item in chosen]
        scores = [item.score for item in chosen if item.score >= 0]
        totals.append(
            (
                chosen[0].profile.name,
                scales_line(chosen[0].profile),
                round(sum(one.words for one in shapes) / len(shapes)),
                sum(one.bullets for one in shapes),
                sum(one.tables for one in shapes),
                sum(one.emoji for one in shapes),
                f"{sum(scores)}/{2 * len(chosen)}",
            )
        )
    table(
        ("Профиль", "Шкалы", "Слов в среднем", "Пунктов", "Строк табл.", "Эмодзи", "Судья"),
        totals,
    )

    print("### Замечания судьи\n")
    remarks = [item for item in answers if item.score < 2 and item.comment]
    if not remarks:
        print("Судья не нашёл нарушений ни в одном ответе.\n")
    else:
        table(
            ("№", "Профиль", "Оценка", "Что не так"),
            [(item.question, item.profile.name, f"{item.score}/2", item.comment) for item in remarks],
        )

    # Целиком печатается один вопрос: девять ответов подряд в README не читает никто,
    # а разницу в форме видно и на одном. Второй вопрос для этого лучший — он про
    # термин, и шкала «уровень» на нём расходится сильнее всего.
    number = 2
    print(f"### Ответы на вопрос {number} целиком\n")
    print(f"{COMPARE[number - 1]}\n")
    for item in answers:
        if item.question != number:
            continue
        print(f"**{item.profile.name}** — {scales_line(item.profile)}, {item.shape.line}\n")
        quote(item.text)


def report_lesson(run: Lesson) -> None:
    print("## Профиль учится на ходу\n")
    print(
        f"Диалог от лица профиля «{LESSON_PROFILE}»: {len(DIALOG)} ходов, "
        f"на ходах {' и '.join(str(number) for number in LESSON_TURNS)} "
        "пользователь просит другую форму ответа.\n"
    )

    rows: list[tuple[Any, ...]] = []
    for number, turn in enumerate(run.turns, start=1):
        measured = turn.shape
        rows.append(
            (
                number,
                oneline(turn.prompt),
                measured.words,
                measured.bullets or "—",
                "; ".join(turn.found) or "—",
            )
        )
    if run.check is not None:
        measured = run.check.shape
        rows.append(
            (
                "новый диалог",
                oneline(run.check.prompt),
                measured.words,
                measured.bullets or "—",
                "; ".join(run.check.found) or "—",
            )
        )
    table(("Ход", "Реплика", "Слов в ответе", "Пунктов списком", "Что ушло в профиль и в память"), rows)

    print("### Шкалы до и после\n")
    changed = [key for key in SCALE_KEYS if run.before.get(key) != run.after.get(key)]
    table(
        ("Шкала", "До прогона", "После прогона"),
        [
            (
                f"{key} ({BY_KEY[key].about})",
                run.before.get(key, "—"),
                run.after.get(key, "—") + (" ← правка" if key in changed else ""),
            )
            for key in SCALE_KEYS
        ],
    )

    if run.check is not None:
        print("### Контрольный вопрос в новом диалоге\n")
        print(f"{LESSON_CHECK}\n")
        print(f"Форма ответа: {run.check.shape.line}\n")
        quote(run.check.answer)


async def main() -> None:
    print("# Прогон day12 — персонализация ассистента\n")
    print(
        f"Модель {MODEL}, ответы не длиннее {SCENARIO_MAX_TOKENS} ток., окно "
        f"{SCENARIO_WINDOW} сообщ. Системный промпт не тронут: длину и формат ответа "
        "задаёт только профиль.\n"
    )

    with tempfile.TemporaryDirectory() as directory:
        storage = Storage(Path(directory) / "personal.db")
        await storage.init()

        answers = await compare(storage)
        teaching = await lesson(storage)

    report_compare(answers)
    report_lesson(teaching)


if __name__ == "__main__":
    asyncio.run(main())
