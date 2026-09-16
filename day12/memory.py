"""Модель памяти агента: три уровня, три разных правила.

Уровни различаются не содержимым, а тремя признаками сразу — охватом, сроком
жизни и тем, как в них попадают записи:

    долговременная  всё приложение   бессрочно, есть до первого слова   явным решением
    рабочая         одна задача      живёт с диалогом                   явным решением
    краткосрочная   один диалог      последние N реплик                 сама, каждый ход

Поэтому один уровень не сводится к другому: краткосрочная помнит формулировки и
теряет их, рабочая держит ТЗ, пока задача не закрыта, а долговременная переживает
и задачу, и диалог. Модуль ничего не знает ни про базу, ни про сеть: он хранит
уровни раздельно и собирает из них кадр запроса. Сами запросы делает агент.

Профиль пользователя в этих уровнях не участвует: он живёт слоем над ними, в
profile.py. Память — про содержание разговора, профиль — про его форму, и в
запросе они стоят рядом, но собираются отдельно.
"""

from dataclasses import dataclass

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

LONGTERM = "longterm"
WORKING = "working"
TIERS = (LONGTERM, WORKING)

# Разделы фиксированы: иначе модель заводит новый раздел на каждый ход, и блок
# памяти расползается в пересказ разговора.
#
# Раздела «профиль» здесь нет: всё про человека живёт в profile.py, отдельным
# слоем. Память отвечает на вопрос «о чём договорились», и решения с знаниями в
# ней общие — они не меняются от того, кто сейчас спрашивает.
SECTIONS = {
    LONGTERM: ("решения", "знания"),
    WORKING: ("цель", "требования", "ограничения", "решения", "вопросы"),
}

# Раздел «решения» есть на обоих уровнях, и это не дубль: «новые сервисы на
# FastAPI» верно всегда, «очередь на Redis» — только в этой задаче. Куда попадёт
# решение, отличает общее от частного, и выбирает это пользователь.

TIER_NAMES = {
    LONGTERM: "долговременная",
    WORKING: "рабочая",
}

# Блоки уходят в модель системными сообщениями: это не чьи-то реплики, а справка
# о команде и о задаче. Каждый уровень — своё сообщение: разделение памяти
# видно и в том, что уходит в запрос.
INTROS = {
    LONGTERM: (
        "Долговременная память — то, что верно независимо от текущей задачи.\n"
        "Решения команды и знания о её окружении, общие для всех собеседников:\n"
    ),
    WORKING: (
        "Рабочая память — техническое задание текущей задачи.\n"
        "Собрано из этого же разговора и подтверждено пользователем:\n"
    ),
}

# Сколько последних реплик уходит в модель дословно. Шесть — три последних хода,
# та часть разговора, где важны формулировки: «а подробнее?» относится к ним.
WINDOW_MESSAGES = 6

# Значения для переключателя: на двух сообщениях агент забывает прошлый ход, на
# двадцати в запрос идёт почти весь диалог — разницу видно вживую.
WINDOW_OPTIONS = (2, 6, 10, 20)


@dataclass(frozen=True)
class Item:
    """Пункт памяти: раздел, формулировка и то, как он сюда попал."""

    id: int
    section: str
    text: str
    # seed — было до начала общения, user — отправил пользователь кнопкой,
    # agent — записал сам при включённом автосохранении.
    origin: str = "user"

    def as_dict(self) -> dict[str, object]:
        return {"id": self.id, "section": self.section, "text": self.text, "origin": self.origin}


@dataclass(frozen=True)
class Message:
    """Реплика диалога: краткосрочная память хранится ровно такими строками."""

    role: str
    content: str
    # id строки в базе: по нему страница отмечает границу окна в ленте.
    id: int = 0

    def as_param(self) -> ChatCompletionMessageParam:
        if self.role == "user":
            return ChatCompletionUserMessageParam(role="user", content=self.content)
        return ChatCompletionAssistantMessageParam(role="assistant", content=self.content)


@dataclass(frozen=True)
class Store:
    """Один уровень памяти: пункты по разделам и то, как они уходят в запрос."""

    tier: str
    items: tuple[Item, ...]

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def text(self) -> str:
        return render(self.tier, self.items)

    def as_param(self) -> ChatCompletionMessageParam:
        return ChatCompletionSystemMessageParam(role="system", content=INTROS[self.tier] + self.text)


def render(tier: str, items: tuple[Item, ...]) -> str:
    """Память разделами, а не сплошным списком: так её читает и модель, и человек."""
    lines = []
    for section in SECTIONS[tier]:
        chosen = [item for item in items if item.section == section]
        if not chosen:
            continue
        lines.append(f"{section}:")
        lines.extend(f"- {item.text}" for item in chosen)
    return "\n".join(lines)


def block(tier: str, rows: list[dict[str, object]]) -> Store | None:
    """Пустой уровень в запрос не идёт: врезка без пунктов только сбивает модель."""
    items = tuple(
        Item(
            id=int(row["id"]),  # type: ignore[arg-type]
            section=str(row["section"]),
            text=str(row["text"]),
            origin=str(row["origin"]),
        )
        for row in rows
        # Раздел вне списка означает рассинхрон кода и базы: такой пункт не
        # отрисуется в блоке, и в запрос его пускать тоже незачем.
        if row["section"] in SECTIONS[tier]
    )
    if not items:
        return None
    return Store(tier=tier, items=items)


@dataclass(frozen=True)
class Window:
    """Что именно уйдёт в модель: два блока памяти и дословные реплики."""

    longterm: Store | None
    working: Store | None
    messages: list[Message]

    @property
    def stores(self) -> list[Store]:
        return [store for store in (self.longterm, self.working) if store is not None]

    def params(self) -> list[ChatCompletionMessageParam]:
        return [
            *(store.as_param() for store in self.stores),
            *(message.as_param() for message in self.messages),
        ]


class Memory:
    """Три уровня рядом: реплики диалога, ТЗ задачи и долговременная память.

    Объект живёт в процессе и пересобирается из базы при первом обращении. Хранятся
    уровни раздельно, и здесь это видно буквально: три поля, которые меняются по
    разным правилам и в разные моменты.
    """

    def __init__(self) -> None:
        # Краткосрочная: пополняется сама после каждого хода.
        self.messages: list[Message] = []
        # Долговременная и рабочая: меняются только явным решением.
        self.longterm: Store | None = None
        self.working: Store | None = None

    @property
    def size(self) -> int:
        return len(self.messages)

    def add(self, *messages: Message) -> None:
        self.messages.extend(messages)

    def select(self, window: int) -> Window:
        """Кадр запроса: оба блока памяти плюс последние N реплик.

        Блоки идут целиком: они уже выжимки, и ужимать выжимку нечем. Из ленты
        берётся только хвост — сколько именно, задаёт окно, и это единственная
        граница краткосрочной памяти.
        """
        chosen = list(self.messages[-window:]) if window > 0 else []

        # Ответ без своего вопроса — обрывок: если пара не влезла в окно целиком,
        # уходит и вторая её половина.
        if chosen and chosen[0].role == "assistant":
            del chosen[0]

        return Window(longterm=self.longterm, working=self.working, messages=chosen)
