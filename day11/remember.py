"""Маршрутизация: что из разговора стоит запомнить и на каком уровне.

Модуль знает, как спросить у модели кандидатов на запись и как разобрать её
ответ. Решение он не принимает: кандидат — это предложение, а не запись. Куда
пункт ляжет и ляжет ли вообще, решает пользователь кнопкой (или, если он этого
попросил, агент по своей же рекомендации). Сам запрос делает агент, хранит
память база.
"""

import json
from dataclasses import dataclass

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from memory import LONGTERM, SECTIONS, TIERS, WORKING, Store

# Разбор хода — не разговор: нужна классификация сказанного, а не сочинение по нему.
TEMPERATURE = 0.0
MAX_TOKENS = 700

# Потолок формулировки: пункт памяти нужен как справка, а не как пересказ реплики.
ITEM_WORDS = 14

# Сколько кандидатов агент показывает за ход. Больше трёх карточек за раз — это
# уже не предложение, а анкета: пользователь перестаёт их читать и жмёт подряд.
ITEM_LIMIT = 3

# Куда падает пункт, если модель назвала раздел, которого в списке нет. Терять
# находку из-за синонима («требование» вместо «требования») незачем.
FALLBACK = {LONGTERM: "знания", WORKING: "требования"}

EMPTY = "(пусто)"

PROMPT = f"""\
Ты ведёшь память агента и раскладываешь по ней то, что всплыло в разговоре.

Уровней два, и отличаются они сроком годности.

longterm — долговременная память: верно независимо от текущей задачи и останется
верным в следующей. Кто такой пользователь, его постоянные предпочтения, принятые
им принципы работы, устройство его команды и окружения.
Разделы: {", ".join(SECTIONS[LONGTERM])}.

working — рабочая память: техническое задание текущей задачи. Верно только здесь
и потеряет смысл, когда задача закончится.
Разделы: {", ".join(SECTIONS[WORKING])}.

На входе — обе памяти целиком и последний обмен репликами. Верни то, что стоит
записать по итогам этого обмена.

Правила:
- Формулировка законченная и самостоятельная: её прочитают без разговора вокруг.
- Не длиннее {ITEM_WORDS} слов, на языке диалога, без «пользователь сказал».
- Только то, что сказал или подтвердил пользователь. Вариант, который предложил
  ассистент и никто не принял, — не факт.
- Что уже есть в памяти, не повторяй, даже другими словами.
- Общее правило и частное требование различай по сроку: «пишем на Python» верно
  всегда, «здесь нужен Redis» — только в этой задаче.
- Сомневаешься между уровнями — выбирай working: перенести в долговременную
  пользователь может и сам.
- Не больше {ITEM_LIMIT} пунктов за ход, самых важных. Ход ни о чём — пустой список.

Поле why — одна короткая строка, зачем это помнить: её увидит пользователь рядом
с кнопками выбора.

Верни строго JSON без markdown и пояснений:
{{"items": [{{"text": "...", "target": "working", "section": "требования", "why": "..."}}]}}
Записывать нечего — верни {{"items": []}}.
"""


@dataclass(frozen=True)
class Candidate:
    """Предложение записать: формулировка, уровень, раздел и зачем это помнить."""

    text: str
    tier: str
    section: str
    why: str = ""


def clean(text: object) -> str:
    return " ".join(str(text).split())


def key(text: str) -> str:
    """Ключ для сравнения формулировок: регистр и пунктуация здесь не различают."""
    return "".join(character for character in clean(text).lower() if character.isalnum() or character == " ")


def request(
    longterm: Store | None,
    working: Store | None,
    prompt: str,
    answer: str,
) -> list[ChatCompletionMessageParam]:
    """Ход уходит на разбор расшифровкой, а не диалогом.

    Роли в таком запросе сбили бы модель с задачи: получив чат, она продолжает
    разговор, а нам нужна классификация сказанного в нём.
    """
    return [
        ChatCompletionSystemMessageParam(role="system", content=PROMPT),
        ChatCompletionUserMessageParam(
            role="user",
            content=(
                f"Долговременная память:\n{longterm.text if longterm else EMPTY}\n\n"
                f"Рабочая память:\n{working.text if working else EMPTY}\n\n"
                f"Последний обмен:\nПользователь: {prompt}\nАссистент: {answer}"
            ),
        ),
    ]


def clean_json(text: str) -> str:
    body = text.strip()
    if body.startswith("```"):
        body = body.removeprefix("```json").removeprefix("```").removesuffix("```")
    return body.strip()


def parse(text: str) -> tuple[Candidate, ...]:
    """Разбор ответа модели: список кандидатов, всё остальное — ошибка."""
    body = clean_json(text)
    try:
        parsed = json.loads(body)
    except ValueError as error:
        raise ValueError(f"ответ не разобран как JSON: {clean(text)[:80]}") from error

    # Модель иногда отдаёт голый список вместо объекта с полем items: форма ответа
    # не та, но содержимое то самое — терять ход из-за обёртки незачем.
    raw = parsed if isinstance(parsed, list) else parsed.get("items") if isinstance(parsed, dict) else None
    if not isinstance(raw, (list, tuple)):
        raise ValueError("ожидался объект с полем items")

    candidates = []
    for entry in raw:
        candidate = _candidate(entry)
        if candidate is not None:
            candidates.append(candidate)
    return tuple(candidates[:ITEM_LIMIT])


def _candidate(entry: object) -> Candidate | None:
    if not isinstance(entry, dict):
        return None

    text = clean(entry.get("text", ""))
    if not text:
        return None

    # Уровень по умолчанию рабочий: то же правило, что и в промпте, — общее
    # пользователь поднимет сам, а частное в долговременной памяти мешает всем.
    tier = clean(entry.get("target", "")).lower()
    if tier not in TIERS:
        tier = WORKING

    return Candidate(text=text, tier=tier, section=section_of(tier, entry.get("section", "")), why=clean(entry.get("why", "")))


def section_of(tier: str, raw: object) -> str:
    """Раздел из ответа модели — к списку разделов уровня."""
    name = clean(raw).lower()
    sections = SECTIONS[tier]
    if name in sections:
        return name
    # Синонимы отличаются окончанием: «требование», «ограничение», «вопрос».
    for section in sections:
        if name and (section.startswith(name) or name.startswith(section)):
            return section
    return FALLBACK[tier]


def fresh(
    candidates: tuple[Candidate, ...],
    longterm: Store | None,
    working: Store | None,
) -> tuple[Candidate, ...]:
    """Кандидаты, которых в памяти ещё нет.

    Промпт просит не повторяться, но просьба — не гарантия: модель охотно
    предлагает записать то, что уже лежит в блоке у неё перед глазами.
    """
    known = {key(item.text) for store in (longterm, working) if store for item in store.items}
    chosen = []
    for candidate in candidates:
        marker = key(candidate.text)
        if marker in known:
            continue
        known.add(marker)
        chosen.append(candidate)
    return tuple(chosen)
