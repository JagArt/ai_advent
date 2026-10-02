"""Кто ведёт память задачи: отдельный вызов после хода и запрет на подмену цели.

В [day11](../day11/remember.py) и [day13](../day13/progress.py) находку приносил
агент, а записывал её пользователь карточкой. Здесь карточки нет: день собирает
чат production-like, а в таком чате никто не подтверждает каждую строку памяти
руками, да и сценарий на пятнадцать реплик с подтверждениями не прогнать.

Отсюда вопрос, которого в day11 не было: что мешает модели молча переписать цель?
Ничего — если полагаться на промпт. Попросить «не меняй цель» можно, но проверить
исполнение нечем, а цена ошибки ровно та, которую задание и просит измерить:
ассистент теряет цель.

Поэтому разрешение выдаёт код, а не промпт. Модель может **предложить** смену,
но `apply` применит её только при двух условиях сразу: явный флаг `goal_changed`
и непустое обоснование `why`. Нет обоснования — цель остаётся прежней, а попытка
попадает в `Applied.kept_goal` и доезжает до замера. Это тот же приём, что в
[day14](../day14/invariants.py): рамку снимает не тот, кто в ней работает.

## Почему отдельный вызов, а не поле в ответе

Напрашивалось добавить `state` в схему [cite.py](cite.py) и обойтись одним
вызовом. Отвергнуто по двум причинам.

Первая: ответ обязан быть подкреплён выдержками, а память задачи берётся из
реплик пользователя, которых в выдержках нет вовсе. Одна схема заставляла бы
модель либо цитировать то, что процитировать нечем, либо считать правило
«утверждение без цитаты писать нельзя» необязательным — и тогда оно перестало бы
работать и для ответа.

Вторая: ход с отказом по порогу до модели не доходит, а память обновить надо и
на нём — пользователь что-то уточнил, даже если ответить было нечем. Отдельный
вызов видит реплику независимо от того, дошёл ли ход до ответа.

Цена известна и считается: один лишний вызов на ход, короткий запрос без
выдержек, порядка трёхсот токенов.
"""

import json
import re
from dataclasses import dataclass

import llm
import state
from state import Applied, Fact, Refused, TaskState

TEMPERATURE = 0.0
MAX_TOKENS = 500

# Сколько последних реплик показывать. Память обновляется по текущему ходу, но
# «а второй вариант?» без предыдущей реплики не разобрать.
TAIL = 4

SCHEMA = """{
  "goal": "",
  "goal_changed": false,
  "why": "",
  "add": [{"section": "ограничения", "text": "одно короткое утверждение"}],
  "drop": []
}"""

SYSTEM = (
    "Ты ведёшь память задачи в диалоге про проект AI Advent.\n"
    "Память — это не пересказ разговора, а короткий список того, что в нём "
    "зафиксировано и что обязательно учитывать дальше.\n"
    "Ответ возвращается строго в формате json по схеме:\n"
    f"{SCHEMA}\n"
    "Поля значат вот что.\n"
    "`goal` — цель диалога одной строкой: чего пользователь добивается этим разговором. "
    "Если цель уже стоит и пользователь её не менял, повтори её дословно.\n"
    "`goal_changed` — true только тогда, когда пользователь прямо попросил заняться "
    "другим. Новый вопрос по той же задаче цель не меняет.\n"
    "`why` — при `goal_changed` цитата или пересказ той реплики, где пользователь "
    "попросил сменить цель; иначе пустая строка.\n"
    "`add` — что добавить в память: ноль-два пункта за ход, каждый одним предложением. "
    f"Раздел — одно из: {', '.join(state.SECTIONS)}.\n"
    "`drop` — номера пунктов, которые пользователь отменил или заменил.\n"
    "Пустые `add` и `drop` — нормальный ответ: на большинстве ходов фиксировать нечего."
)

RULES = (
    "Что считать пунктом памяти:\n"
    f"`{state.CLARIFIED}` — пользователь сузил вопрос или выбрал из вариантов;\n"
    f"`{state.LIMITS}` — рамка, которую ответ обязан соблюдать;\n"
    f"`{state.TERMS}` — слово, которому пользователь придал в этом разговоре "
    "конкретный смысл.\n"
    "Берётся только то, что сказал пользователь. Что сказал ассистент — это ответ, "
    "а не решение по задаче, и в память не идёт.\n"
    "Не записывай факты о проекте и подробности из выдержек: память держит условия "
    "задачи, а не содержание базы.\n"
    "Если в этом ходе ничего не зафиксировано — верни пустые `add` и `drop`."
)

FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")

KEPT_NO_WHY = "смена цели без обоснования"
KEPT_SAME = "та же цель другими словами"


@dataclass(frozen=True)
class Proposal:
    """Что модель предложила сделать с памятью. Ещё не применено."""

    goal: str = ""
    goal_changed: bool = False
    why: str = ""
    add: tuple[tuple[str, str], ...] = ()
    drop: tuple[int, ...] = ()
    reply: llm.Reply | None = None
    broken: str = ""
    raw: str = ""

    @property
    def tokens(self) -> int:
        return (self.reply.prompt_tokens + self.reply.completion_tokens) if self.reply else 0

    @property
    def seconds(self) -> float:
        return self.reply.seconds if self.reply else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "goal": self.goal,
            "goal_changed": self.goal_changed,
            "why": self.why,
            "add": [{"section": section, "text": text} for section, text in self.add],
            "drop": list(self.drop),
            "broken": self.broken,
            "tokens": self.tokens,
        }


@dataclass(frozen=True)
class Tracked:
    """Итог обновления: новый снимок, что изменилось и чего это стоило."""

    state: TaskState
    applied: Applied
    proposal: Proposal

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state.as_dict(),
            "applied": self.applied.as_dict(),
            "proposal": self.proposal.as_dict(),
        }


# --- разбор ------------------------------------------------------------------


def _numbers(payload: object) -> tuple[int, ...]:
    if not isinstance(payload, list):
        return ()

    found: list[int] = []
    for item in payload:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            found.append(item)
        elif isinstance(item, str) and (digits := re.search(r"\d+", item)):
            found.append(int(digits.group()))
    return tuple(dict.fromkeys(found))


def _items(payload: object) -> tuple[tuple[str, str], ...]:
    """Пары «раздел, текст». Чужие разделы не отбрасываются: их считает `add`."""
    if not isinstance(payload, list):
        return ()

    found: list[tuple[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        section = str(item.get("section") or "").strip().lower()
        text = str(item.get("text") or "").strip()
        if text:
            found.append((section, text))
    return tuple(found)


def parse(text: str) -> Proposal:
    """Ответ модели в `Proposal`. Сорванный json не чинится, а признаётся пустым.

    В [cite.py](cite.py) обрыв json спасают по кускам: там на кону ответ, за
    который уже заплачено контекстом. Здесь на кону обновление памяти, и
    спасать нечего — пропущенный пункт вернётся на следующем ходе, а собранный
    из обрывков может оказаться половиной ограничения, то есть ограничением
    с другим смыслом.
    """
    body = FENCE.sub("", text.strip())

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return Proposal(broken="ответ не разобрался как json", raw=body)

    if not isinstance(payload, dict):
        return Proposal(broken="ответ не объект", raw=body)

    return Proposal(
        goal=str(payload.get("goal") or "").strip(),
        goal_changed=bool(payload.get("goal_changed")),
        why=str(payload.get("why") or "").strip(),
        add=_items(payload.get("add")),
        drop=_numbers(payload.get("drop")),
        raw=body,
    )


# --- спросить ----------------------------------------------------------------


def messages(
    question: str, answer: str, current: TaskState, history: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Запрос трекера: состояние, хвост разговора и текущий ход."""
    known = current.block() or "пока пусто"
    numbered = "\n".join(f"[{fact.id}] {fact.section}: {fact.text}" for fact in current.facts)

    tail = "\n".join(
        f"{'Пользователь' if item['role'] == 'user' else 'Ассистент'}: {item['content']}"
        for item in history[-TAIL:]
    )

    parts = [f"Память задачи сейчас:\n{known}"]
    if numbered:
        parts.append(f"Пункты с номерами (для `drop`):\n{numbered}")
    if tail:
        parts.append(f"Предыдущие реплики:\n{tail}")
    parts.append(f"Новая реплика пользователя:\n{question}")
    if answer:
        parts.append(f"Ответ ассистента на неё:\n{answer}")

    return [
        {"role": "system", "content": f"{SYSTEM}\n\n{RULES}"},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


async def propose(
    question: str,
    answer: str,
    current: TaskState,
    history: list[dict[str, str]] | None = None,
) -> Proposal:
    """Спросить модель, что изменилось в условиях задачи."""
    reply = await llm.complete(
        messages(question, answer, current, history or []),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        json=True,
    )
    parsed = parse(reply.text)
    return Proposal(
        goal=parsed.goal,
        goal_changed=parsed.goal_changed,
        why=parsed.why,
        add=parsed.add,
        drop=parsed.drop,
        reply=reply,
        broken=parsed.broken,
        raw=parsed.raw,
    )


def apply(current: TaskState, proposal: Proposal, turn: int) -> Tracked:
    """Применить предложение. Разрешение на смену цели выдаётся здесь, в коде.

    Порядок обязателен: сначала `drop`, потом `add`. Иначе замена пункта
    («не SQLite, а Postgres») упирается в потолок раздела, который сам же и
    освобождала.
    """
    goal_was = current.goal
    kept = ""
    notes: list[str] = []

    if proposal.broken:
        notes.append(proposal.broken)

    after, dropped = current.drop(list(proposal.drop))

    added: list[Fact] = []
    refused: list[Refused] = []
    for section, text in proposal.add:
        after, outcome = after.add(section, text, turn)
        if isinstance(outcome, Fact):
            added.append(outcome)
        else:
            refused.append(outcome)

    goal_set = None
    if proposal.goal:
        if goal_was is None:
            # Первая постановка цели — не смена, и обоснования не требует: до неё
            # цели не было вовсе, подменять нечего.
            after = after.retarget(proposal.goal, turn)
            goal_set = after.goal
        elif state.fold(proposal.goal) == state.fold(goal_was.text):
            pass
        elif not proposal.goal_changed:
            kept = KEPT_SAME
        elif not proposal.why:
            kept = KEPT_NO_WHY
        else:
            after = after.retarget(proposal.goal, turn, proposal.why)
            goal_set = after.goal

    return Tracked(
        state=after,
        applied=Applied(
            added=tuple(added),
            dropped=dropped,
            refused=tuple(refused),
            goal_set=goal_set,
            goal_was=goal_was,
            kept_goal=kept,
            notes=notes,
        ),
        proposal=proposal,
    )


async def update(
    current: TaskState,
    question: str,
    answer: str,
    turn: int,
    history: list[dict[str, str]] | None = None,
) -> Tracked:
    """Ход памяти задачи целиком: спросить модель и применить разрешённое."""
    return apply(current, await propose(question, answer, current, history), turn)


def skipped(current: TaskState) -> Tracked:
    """Ход без трекера: режимы без памяти задачи платить за него не должны."""
    return Tracked(state=current, applied=Applied(), proposal=Proposal())
