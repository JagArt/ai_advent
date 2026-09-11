"""Факты диалога: короткая память «ключ — значение» вместо самих реплик.

Модуль знает, как факты выглядят в запросе, как их просят у модели и как разбирать
её ответ. Сам запрос делает агент, хранит картотеку база.
"""

import json
from dataclasses import dataclass

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

import tokens

# Потолок картотеки: без него блок фактов растёт вместе с диалогом — ровно та
# история, от которой мы уходим. Значения короткие по той же причине: факт нужен
# как справка, а не как пересказ реплики.
FACTS_LIMIT = 24
FACTS_VALUE_WORDS = 12

# Извлечение фактов — не разговор: нужен разбор реплики, а не сочинение по ней.
FACTS_TEMPERATURE = 0.0
FACTS_MAX_TOKENS = 500

FACTS_PROMPT = f"""\
Ты ведёшь картотеку фактов о задаче пользователя.
На входе — текущая картотека и последний обмен репликами.
Верни только изменения к картотеке, а не её целиком.

Записывай то, что не устаревает к следующему ходу: цель, ограничения,
предпочтения, принятые решения, договорённости, названия, числа, сроки.
Не записывай советы ассистента, его рассуждения и темы, которые просто обсуждались.

Ключ — 1–3 слова на языке диалога, значение — не длиннее {FACTS_VALUE_WORDS} слов.
Уточняя известный факт, повторяй его ключ буква в букву: одноимённый факт
заменяется, а не добавляется вторым.
Ключ, который перестал быть верным, отправляй в delete.
Всего в картотеке не больше {FACTS_LIMIT} фактов, поэтому близкие объединяй в один.

Верни строго JSON без markdown и пояснений:
{{"set": {{"ключ": "значение"}}, "delete": ["ключ"]}}
Если ход ничего не добавил, верни {{"set": {{}}, "delete": []}}.
"""

# Картотека уходит в модель системным сообщением: это не чья-то реплика, а справка
# о разговоре, часть которого в запрос уже не попадает.
FACTS_INTRO = "Известные факты о задаче:\n"

EMPTY = "(пусто)"


@dataclass(frozen=True)
class Facts:
    """Картотека вместе со своей ценой в запросе: считаем её один раз."""

    items: tuple[tuple[str, str], ...]
    tokens: int

    @property
    def count(self) -> int:
        return len(self.items)

    @property
    def cost(self) -> int:
        return tokens.MESSAGE_OVERHEAD + self.tokens

    @property
    def text(self) -> str:
        return render(self.items)

    def as_param(self) -> ChatCompletionMessageParam:
        return ChatCompletionSystemMessageParam(role="system", content=FACTS_INTRO + self.text)


@dataclass(frozen=True)
class Update:
    """Изменения картотеки за один ход: что записать и что вычеркнуть."""

    assign: tuple[tuple[str, str], ...] = ()
    remove: tuple[str, ...] = ()


def render(items: tuple[tuple[str, str], ...]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in items)


def block(items: tuple[tuple[str, str], ...]) -> Facts | None:
    """Пустая картотека в запрос не идёт: врезка без фактов только сбивает модель."""
    if not items:
        return None
    return Facts(items=items, tokens=tokens.count_text(FACTS_INTRO + render(items)))


def clean(text: str) -> str:
    return " ".join(str(text).split())


def parse(text: str) -> Update:
    """Разбор ответа модели: JSON с двумя списками, всё остальное — ошибка."""
    body = clean_json(text)
    try:
        parsed = json.loads(body)
    except ValueError as error:
        raise ValueError(f"ответ не разобран как JSON: {clean(text)[:80]}") from error

    if not isinstance(parsed, dict):
        raise ValueError("ожидался объект с полями set и delete")

    # Модель иногда отдаёт саму картотеку вместо операций над ней: плоский словарь
    # читается как «записать всё это» — терять ход из-за формы ответа незачем.
    raw_assign = parsed.get("set", parsed if "delete" not in parsed else {})
    raw_remove = parsed.get("delete", ())
    if not isinstance(raw_assign, dict) or not isinstance(raw_remove, (list, tuple)):
        raise ValueError("поле set должно быть объектом, delete — списком")

    # Вложенный объект вместо значения — не факт, а ещё одна картотека: такой
    # пункт пропускается, остальные записываются.
    assign = tuple(
        (clean(key), clean(value))
        for key, value in raw_assign.items()
        if not isinstance(value, (dict, list)) and clean(key) and clean(value)
    )
    remove = tuple(clean(key) for key in raw_remove if clean(key))
    return Update(assign=assign, remove=remove)


def clean_json(text: str) -> str:
    body = text.strip()
    if body.startswith("```"):
        body = body.removeprefix("```json").removeprefix("```").removesuffix("```")
    return body.strip()


def apply(
    items: tuple[tuple[str, str], ...],
    update: Update,
) -> tuple[tuple[tuple[str, str], ...], int, int, int]:
    """Новая картотека и что в ней изменилось: добавлено, уточнено, вычеркнуто."""
    current = dict(items)

    removed = sum(1 for key in update.remove if current.pop(key, None) is not None)
    added = changed = 0
    for key, value in update.assign:
        if key not in current:
            added += 1
        elif current[key] != value:
            changed += 1
        # Уточнение известного факта не двигает его в конец: порядок картотеки —
        # порядок появления, и по нему видно, что агент узнал раньше остального.
        current[key] = value

    # Потолок картотеки держится за устоявшиеся факты: имя и стек из первых ходов
    # не должны вытесняться подробностями последнего. Освобождать место —
    # работа модели, у неё для этого есть delete.
    return tuple(current.items())[:FACTS_LIMIT], added, changed, removed


def update_request(
    facts: Facts | None,
    prompt: str,
    answer: str,
) -> list[ChatCompletionMessageParam]:
    """Ход уходит на разбор расшифровкой, а не диалогом.

    Роли в таком запросе сбили бы модель с задачи: получив чат, она продолжает
    разговор, а нам нужны факты о нём.
    """
    return [
        ChatCompletionSystemMessageParam(role="system", content=FACTS_PROMPT),
        ChatCompletionUserMessageParam(
            role="user",
            content=(
                f"Картотека:\n{facts.text if facts else EMPTY}\n\n"
                f"Пользователь: {prompt}\nАссистент: {answer}"
            ),
        ),
    ]
