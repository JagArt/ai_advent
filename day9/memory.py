"""Память агента: хвост диалога дословно, всё, что старше — одной свёрткой.

Модуль ничего не знает ни про базу, ни про сеть: он держит реплики сессии,
считает их цену в токенах и решает, что уйдёт в запрос и что пора пересказать.
Сами запросы к модели делает агент.
"""

from dataclasses import dataclass

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

import tokens

# Сколько последних реплик всегда уходит в модель дословно. Три хода — та часть
# разговора, где важны формулировки: «а подробнее?» относится именно к ним.
TAIL_MESSAGES = 6

# Свёртка — платный запрос, поэтому она делается не на каждом ходу, а когда за
# хвостом накопился блок из десяти реплик, то есть раз в пять ходов.
COMPRESS_EVERY = 10

# Конспект не должен разрастаться: он пересказывает сам себя от свёртки к свёртке,
# и без потолка растёт вместе с диалогом — ровно то, от чего мы уходим. Размер
# задаётся числом тезисов, а не слов: счёт строк модель выдерживает, счёт слов —
# нет. max_tokens при этом только страхует от обрыва на полуслове.
SUMMARY_MAX_TOKENS = 700
SUMMARY_POINTS = 18
SUMMARY_POINT_WORDS = 12

# Конспект — не разговор, поэтому температура низкая: нужен пересказ, а не сочинение.
SUMMARY_TEMPERATURE = 0.2

SUMMARY_PROMPT = f"""\
Ты ведёшь конспект диалога пользователя с ассистентом.
На входе — предыдущий конспект, если он есть, и новые реплики.
Верни один обновлённый конспект, который заменит и то и другое: он остаётся
единственной памятью о свёрнутой части разговора.

Конспект состоит из четырёх разделов, каждый — список коротких тезисов:
Факты: кто пользователь, над чем работает, названия, числа, сроки, требования.
Решения: что решили и почему, теми же словами, которыми это называли.
Обсуждалось: темы, без подробностей и советов.
Открыто: вопросы, оставшиеся без ответа.

Всего не больше {SUMMARY_POINTS} тезисов, каждый не длиннее {SUMMARY_POINT_WORDS} слов.
Факты и решения из предыдущего конспекта переноси все — они не устаревают, даже
если новые реплики их не касаются, — но сжимай: одно решение — один тезис, без
перечисления полей, колонок и вариантов.
Решения пиши от свежих к старым, близкие объединяй в одно.
Если тезисов выходит больше, выбрасывай подробности из «Обсуждалось», но не
факты и не решения.
Пиши на языке диалога, без вступления.
Не обращайся к пользователю и не отвечай на его вопросы: только конспектируй.
"""

# Конспект уходит в модель системным сообщением: это не чья-то реплика, а справка
# о том, что было до начала видимой части диалога.
SUMMARY_INTRO = "Краткое содержание предыдущей части диалога:\n"

SPEAKERS = {"user": "Пользователь", "assistant": "Ассистент"}


@dataclass(frozen=True)
class Counted:
    """Реплика вместе со своей ценой в токенах: считаем её один раз."""

    role: str
    content: str
    tokens: int
    # id строки в базе: по нему проходит граница свёрнутой части истории.
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
class Summary:
    """Свёрнутая часть диалога: что агент помнит вместо самих реплик."""

    text: str
    # Последняя реплика, попавшая в конспект: всё, что старше, в запрос не идёт.
    covered_id: int
    messages: int
    # Цена конспекта в запросе, вместе со служебной врезкой, — она и стоит денег.
    tokens: int

    @property
    def cost(self) -> int:
        return tokens.MESSAGE_OVERHEAD + self.tokens

    def as_param(self) -> ChatCompletionMessageParam:
        return ChatCompletionSystemMessageParam(role="system", content=SUMMARY_INTRO + self.text)


def tidy(text: str) -> str:
    """Уборка за лимитом ответа: длинный конспект обрывается на полуслове.

    Оборванный тезис в памяти хуже отсутствующего: он выглядит как факт, но
    договорён до середины. Целые тезисы остаются, последний неполный уходит.
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) > 1 and not lines[-1].rstrip().endswith((".", "!", "?", ";", ":")):
        del lines[-1]
    return "\n".join(lines)


def summarized(text: str, covered_id: int, messages: int) -> Summary:
    return Summary(
        text=text,
        covered_id=covered_id,
        messages=messages,
        tokens=tokens.count_text(SUMMARY_INTRO + text),
    )


@dataclass(frozen=True)
class Window:
    """Что именно уйдёт в модель из памяти: конспект и дословные реплики."""

    summary: Summary | None
    messages: list[Counted]

    @property
    def cost(self) -> int:
        summary = self.summary.cost if self.summary else 0
        return summary + sum(message.cost for message in self.messages)

    def params(self) -> list[ChatCompletionMessageParam]:
        summary = [self.summary.as_param()] if self.summary else []
        return [*summary, *(message.as_param() for message in self.messages)]


class Memory:
    """Реплики сессии и её конспект: живут в процессе, хранятся в базе."""

    def __init__(
        self,
        *,
        tail_messages: int = TAIL_MESSAGES,
        compress_every: int = COMPRESS_EVERY,
    ) -> None:
        self.tail_messages = tail_messages
        self.compress_every = compress_every
        self.messages: list[Counted] = []
        self.summary: Summary | None = None

    @property
    def size(self) -> int:
        return len(self.messages)

    @property
    def cost(self) -> int:
        """Во сколько встала бы вся история диалога, если тащить её целиком."""
        return sum(message.cost for message in self.messages)

    def add(self, *messages: Counted) -> None:
        self.messages.extend(messages)

    def live(self) -> list[Counted]:
        """Реплики, которых нет в конспекте: только они идут в запрос дословно."""
        if self.summary is None:
            return list(self.messages)
        return [message for message in self.messages if message.id > self.summary.covered_id]

    def block(self) -> list[Counted]:
        """Кандидаты на свёртку: всё, что вышло за хвост и ещё не пересказано."""
        live = self.live()
        return live[: max(len(live) - self.tail_messages, 0)]

    def crowded(self) -> bool:
        return len(self.block()) >= self.compress_every

    def select(self, budget: int, base: int, *, compression: bool) -> Window:
        """Окно под бюджет: со сжатием — конспект и хвост, без — вся история.

        Бюджет здесь работает страховкой: свёртка уже убрала основной объём, но
        один длинный ход в хвосте может не влезть и в оставшееся место.
        """
        summary = self.summary if compression else None
        source = self.live() if compression else self.messages

        # Конспект неснимаем: он и есть сжатая история, без него от неё ничего
        # не остаётся. Место под дословные реплики считается уже после него.
        used = base + (summary.cost if summary else 0)

        window: list[Counted] = []
        # Окно набирается с конца: свежие реплики важнее, старые уходят первыми.
        for message in reversed(source):
            if used + message.cost > budget:
                break
            used += message.cost
            window.append(message)
        window.reverse()

        # Ответ без своего вопроса — обрывок: если пара не влезла целиком,
        # уходит и вторая её половина.
        if window and window[0].role == "assistant":
            del window[0]

        return Window(summary=summary, messages=window)


def summary_request(
    summary: Summary | None,
    block: list[Counted],
) -> list[ChatCompletionMessageParam]:
    """Реплики уходят на свёртку расшифровкой, а не диалогом.

    Роли в таком запросе сбили бы модель с задачи: получив чат, она продолжает
    разговор, а нам нужен текст о разговоре.
    """
    parts = []
    if summary is not None:
        parts.append(f"Предыдущий конспект:\n{summary.text}")

    lines = "\n".join(f"{SPEAKERS[message.role]}: {message.content}" for message in block)
    parts.append(f"Новые реплики:\n{lines}")

    return [
        ChatCompletionSystemMessageParam(role="system", content=SUMMARY_PROMPT),
        ChatCompletionUserMessageParam(role="user", content="\n\n".join(parts)),
    ]
