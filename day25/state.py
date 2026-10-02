"""Память задачи: цель диалога и то, что в нём уже зафиксировано.

Диалоговая история отвечает на вопрос «что только что говорили». На вопрос «чего
мы вообще добиваемся» она не отвечает: к пятнадцатой реплике первая уже вышла из
хвоста, а цель была названа именно в ней. Отсюда второй уровень памяти, который
живёт не репликами, а утверждениями о задаче.

Устройство взято у [day11](../day11/memory.py): фиксированные разделы и отдельный
системный блок в запросе. Но набор разделов другой, и выбран он по заданию дня —
что пользователь уточнил, какие ограничения и термины зафиксированы, что является
целью диалога:

    цель          одна строка, отдельное поле, меняется только явным решением
    уточнения     что пользователь сузил или выбрал по ходу разговора
    ограничения   рамки, в которых ответ обязан держаться
    термины       слова, которым в этом разговоре придан конкретный смысл

## Почему цель — поле, а не раздел

Соблазн был сделать `цель` четвёртым разделом и не плодить сущностей. Отвергнуто:
разделы — это списки, а список целей — это отсутствие цели. Пока цель лежит в
списке, «добавить» и «сменить» выглядят одинаково, и к десятому ходу там три
формулировки, из которых модель выбирает удобную. Отдельное поле делает смену
цели событием, у которого есть обоснование и запись в журнале, — а [track.py](track.py)
может эту смену не разрешить.

## Потолок отказывает, а не вытесняет

У раздела есть потолок, и при переполнении новый пункт **не записывается**.
Напрашивалось вытеснение по старшинству — так делают кэши, — но здесь это ровно
худший выбор: ограничение называют в начале разговора, а переполняется раздел в
конце, и вытеснение по старшинству выбрасывало бы в первую очередь то, что
держит рамку. Поэтому место освобождается только через `drop`, то есть явным
решением модели, а отказ потолка считается в замере: если он срабатывает часто,
значит потолок мал, и это видно числом, а не на глаз.

## Приведение, а не сравнение строк

Один и тот же факт модель формулирует каждый ход чуть иначе: «только SQLite»,
«хранить в SQLite», «SQLite, без внешних сервисов». Сравнение строк тут бесполезно,
поэтому дедуп идёт по приведённому виду — нижний регистр, выброшенная пунктуация,
схлопнутые пробелы. Это грубо и ловит не всё, зато не требует ни модели, ни сети:
платить вызовом за то, чтобы не записать дубль, дороже самого дубля.
"""

import re
from dataclasses import dataclass, field, replace

from openai.types.chat import ChatCompletionSystemMessageParam

CLARIFIED = "уточнения"
LIMITS = "ограничения"
TERMS = "термины"

SECTIONS = (CLARIFIED, LIMITS, TERMS)

SECTION_ABOUT = {
    CLARIFIED: "что пользователь уточнил или выбрал",
    LIMITS: "рамки, которые ответ обязан соблюдать",
    TERMS: "слова, которым в этом разговоре придан конкретный смысл",
}

# Пунктов на раздел. Шесть — потолок, за которым блок состояния перестаёт быть
# справкой и становится пересказом разговора: на пятнадцати репликах это заметно
# сразу. При переполнении новый пункт отвергается, см. докстринг модуля.
CAP = 6

# Потолок на пункт. Длиннее — это уже не зафиксированный факт, а абзац ответа,
# который приехал в память по недосмотру модели.
TEXT_CHARS = 180

# Потолок на цель. Цель в три строки нельзя ни удержать, ни проверить.
GOAL_CHARS = 200

INTRO = (
    "Память задачи — то, что в этом разговоре уже зафиксировано.\n"
    "Она собрана автоматически из предыдущих ходов и обязательна к соблюдению:\n"
)

# Обязательство едет вместе с блоком, а не в общих правилах: без него блок
# читается моделью как справка «к сведению», и термин из него она подменяет
# своим уже на третьем ходе.
OBLIGE = (
    "\nОтвечай так, чтобы ответ служил этой цели, даже если вопрос задан про частность.\n"
    "Зафиксированные термины используй в том смысле, который здесь указан, "
    "а ограничения не нарушай и не предлагай обойти.\n"
    "Цель диалога не меняй по своей воле: если вопрос ведёт в сторону — ответь на него, "
    "но держи цель."
)

PUNCTUATION = re.compile(r"[^\w\s]+", re.UNICODE)
SPACES = re.compile(r"\s+")

# Причины, по которым пункт не записался. Нужны замеру: «состояние не выросло»
# и «состояние отказалось расти» — разные события.
DUPLICATE = "дубль"
CROWDED = "потолок раздела"
EMPTY = "пустой текст"
UNKNOWN_SECTION = "нет такого раздела"


def fold(text: str) -> str:
    """Приведённый вид пункта: по нему ловятся переформулировки одного и того же."""
    return SPACES.sub(" ", PUNCTUATION.sub(" ", text.lower())).strip()


@dataclass(frozen=True)
class Goal:
    """Цель диалога: формулировка, ход постановки и чем обоснована смена."""

    text: str
    turn: int
    why: str = ""

    def as_dict(self) -> dict[str, object]:
        return {"text": self.text, "turn": self.turn, "why": self.why}


@dataclass(frozen=True)
class Fact:
    """Пункт памяти задачи: раздел, формулировка и ход, на котором записан."""

    id: int
    section: str
    text: str
    turn: int

    def as_dict(self) -> dict[str, object]:
        return {"id": self.id, "section": self.section, "text": self.text, "turn": self.turn}


@dataclass(frozen=True)
class Refused:
    """Пункт, который записать не дали. Причина важнее самого пункта."""

    section: str
    text: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {"section": self.section, "text": self.text, "reason": self.reason}


@dataclass(frozen=True)
class TaskState:
    """Снимок памяти задачи. Неизменяемый: каждый ход даёт новый снимок.

    Неизменяемость тут не стилистика. Замер показывает состояние после каждого
    хода рядом с ответом этого хода, и если бы снимок правился на месте, в
    журнале лежала бы пятнадцать раз одна и та же последняя версия.
    """

    goal: Goal | None = None
    facts: tuple[Fact, ...] = ()

    @property
    def empty(self) -> bool:
        return self.goal is None and not self.facts

    @property
    def size(self) -> int:
        return len(self.facts) + (1 if self.goal else 0)

    def of(self, section: str) -> tuple[Fact, ...]:
        return tuple(fact for fact in self.facts if fact.section == section)

    def next_id(self) -> int:
        return max((fact.id for fact in self.facts), default=0) + 1

    # --- изменения -----------------------------------------------------------

    def add(self, section: str, text: str, turn: int) -> tuple["TaskState", Fact | Refused]:
        """Записать пункт. Возвращает новый снимок и то, чем кончилась попытка."""
        body = " ".join(text.split())[:TEXT_CHARS].strip()

        if section not in SECTIONS:
            return self, Refused(section, body, UNKNOWN_SECTION)
        if not body:
            return self, Refused(section, body, EMPTY)

        folded = fold(body)
        if any(fold(fact.text) == folded for fact in self.of(section)):
            return self, Refused(section, body, DUPLICATE)
        if len(self.of(section)) >= CAP:
            return self, Refused(section, body, CROWDED)

        fact = Fact(id=self.next_id(), section=section, text=body, turn=turn)
        return replace(self, facts=(*self.facts, fact)), fact

    def drop(self, ids: list[int]) -> tuple["TaskState", tuple[Fact, ...]]:
        """Убрать пункты по номерам. Неизвестные номера молча пропускаются."""
        wanted = set(ids)
        gone = tuple(fact for fact in self.facts if fact.id in wanted)
        if not gone:
            return self, ()
        kept = tuple(fact for fact in self.facts if fact.id not in wanted)
        return replace(self, facts=kept), gone

    def retarget(self, text: str, turn: int, why: str = "") -> "TaskState":
        """Поставить или сменить цель. Разрешение на смену выдаётся в `track.apply`."""
        body = " ".join(text.split())[:GOAL_CHARS].strip()
        if not body:
            return self
        return replace(self, goal=Goal(text=body, turn=turn, why=why))

    # --- в запрос ------------------------------------------------------------

    def lines(self) -> list[str]:
        """Блок состояния по строкам. Пустые разделы не печатаются вовсе."""
        out: list[str] = []
        if self.goal:
            out.append(f"Цель диалога: {self.goal.text}")

        for section in SECTIONS:
            items = self.of(section)
            if not items:
                continue
            out.append(f"{section.capitalize()} ({SECTION_ABOUT[section]}):")
            out.extend(f"— {fact.text}" for fact in items)

        return out

    def block(self) -> str:
        return "\n".join(self.lines())

    def as_param(self) -> ChatCompletionSystemMessageParam | None:
        """Системное сообщение с памятью задачи. Пустое состояние блока не даёт."""
        if self.empty:
            return None
        return {"role": "system", "content": f"{INTRO}{self.block()}{OBLIGE}"}

    def query_hint(self) -> str:
        """Что из состояния помогает искать: цель и зафиксированные термины.

        Ограничения и уточнения сюда не идут, и это не упущение. «Только SQLite»
        описывает, каким должен быть ответ, а не где его искать: в поисковом
        запросе такое слово тянет выдачу на чанки про хранение и вытесняет из
        пула то, о чём спросили.
        """
        parts: list[str] = []
        if self.goal:
            parts.append(self.goal.text)
        parts.extend(fact.text for fact in self.of(TERMS))
        return "\n".join(parts)

    def as_dict(self) -> dict[str, object]:
        return {
            "goal": self.goal.as_dict() if self.goal else None,
            "size": self.size,
            "sections": {
                section: [fact.as_dict() for fact in self.of(section)] for section in SECTIONS
            },
        }


@dataclass(frozen=True)
class Applied:
    """Что ход сделал с памятью задачи. По нему страница рисует подсветку.

    `kept_goal` — цель, которую модель предложила сменить, а код не дал. Это
    главное число дня, и теряться в логах оно не должно.
    """

    added: tuple[Fact, ...] = ()
    dropped: tuple[Fact, ...] = ()
    refused: tuple[Refused, ...] = ()
    goal_set: Goal | None = None
    goal_was: Goal | None = None
    kept_goal: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.dropped or self.goal_set)

    def as_dict(self) -> dict[str, object]:
        return {
            "changed": self.changed,
            "added": [fact.as_dict() for fact in self.added],
            "dropped": [fact.as_dict() for fact in self.dropped],
            "refused": [item.as_dict() for item in self.refused],
            "goal_set": self.goal_set.as_dict() if self.goal_set else None,
            "goal_was": self.goal_was.as_dict() if self.goal_was else None,
            "kept_goal": self.kept_goal,
            "notes": self.notes,
        }
