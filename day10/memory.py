"""Память агента: что из диалога уходит в модель и по какому правилу.

Три стратегии решают одну задачу — собрать запрос из истории, которая в него
целиком не влезает, — и решают её по-разному: окно оставляет последние реплики,
факты держат рядом с окном картотеку, ветки меняют саму историю.

Модуль ничего не знает ни про базу, ни про сеть: он держит реплики ветки, считает
их цену в токенах и отбирает окно. Сами запросы к модели делает агент.
"""

from dataclasses import dataclass

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)

import tokens
from facts import Facts

# Стратегии управления контекстом — значения переключателя в шапке страницы.
WINDOW = "window"
FACTS = "facts"
BRANCHING = "branching"

STRATEGIES = (WINDOW, FACTS, BRANCHING)
DEFAULT_STRATEGY = WINDOW

# Сколько последних реплик уходит в модель дословно. Шесть — три последних хода,
# та часть разговора, где важны формулировки: «а подробнее?» относится к ним.
WINDOW_MESSAGES = 6

# Значения для переключателя: на двух сообщениях агент забывает прошлый ход, на
# двадцати в запрос идёт почти весь диалог — разницу видно вживую.
WINDOW_OPTIONS = (2, 6, 10, 20)


@dataclass(frozen=True)
class Counted:
    """Реплика вместе со своей ценой в токенах: считаем её один раз."""

    role: str
    content: str
    tokens: int
    # id строки в базе: по нему проходит граница ветки и считается окно.
    id: int = 0

    @property
    def cost(self) -> int:
        return tokens.MESSAGE_OVERHEAD + self.tokens

    def as_param(self) -> ChatCompletionMessageParam:
        if self.role == "user":
            return ChatCompletionUserMessageParam(role="user", content=self.content)
        return ChatCompletionAssistantMessageParam(role="assistant", content=self.content)


def counted(role: str, content: str, message_id: int = 0) -> Counted:
    return Counted(role=role, content=content, tokens=tokens.count_text(content), id=message_id)


@dataclass(frozen=True)
class Window:
    """Что именно уйдёт в модель из памяти: картотека и дословные реплики."""

    facts: Facts | None
    messages: list[Counted]

    @property
    def cost(self) -> int:
        facts = self.facts.cost if self.facts else 0
        return facts + sum(message.cost for message in self.messages)

    def params(self) -> list[ChatCompletionMessageParam]:
        facts = [self.facts.as_param()] if self.facts else []
        return [*facts, *(message.as_param() for message in self.messages)]


class Memory:
    """Реплики текущей ветки и её картотека: живут в процессе, хранятся в базе.

    В стратегии веток объект пересобирается при переключении: у каждой ветки своя
    история и своя картотека, и одна ничего не знает о другой.
    """

    def __init__(self) -> None:
        self.messages: list[Counted] = []
        self.facts: Facts | None = None

    @property
    def size(self) -> int:
        return len(self.messages)

    @property
    def cost(self) -> int:
        """Во сколько встала бы вся история ветки, если тащить её целиком."""
        return sum(message.cost for message in self.messages)

    def add(self, *messages: Counted) -> None:
        self.messages.extend(messages)

    def select(self, budget: int, base: int, *, strategy: str, window: int) -> Window:
        """Окно под стратегию: последние N реплик, при фактах — ещё и картотека.

        Число сообщений — главное ограничение, бюджет — второе: длинный ход может
        не влезть в оставшееся место, и тогда окно короче заявленного.
        """
        # Ветки не меняют способ набрать контекст, они меняют историю, из которой
        # он набирается: внутри ветки работает то же скользящее окно.
        facts = self.facts if strategy == FACTS else None
        used = base + (facts.cost if facts else 0)

        chosen: list[Counted] = []
        # Окно набирается с конца: свежие реплики важнее, старые уходят первыми.
        for message in reversed(self.messages[-window:] if window > 0 else []):
            if used + message.cost > budget:
                break
            used += message.cost
            chosen.append(message)
        chosen.reverse()

        # Ответ без своего вопроса — обрывок: если пара не влезла целиком,
        # уходит и вторая её половина.
        if chosen and chosen[0].role == "assistant":
            del chosen[0]

        return Window(facts=facts, messages=chosen)
