"""Маршрутизация: что из разговора стоит запомнить и куда это положить.

Адресов три: два уровня памяти и профиль. Уровни памяти отличаются друг от друга
сроком годности — что переживёт задачу, а что уйдёт вместе с ней. Профиль стоит
особняком: он не про то, что сказано, а про то, как об этом говорить, и попадает
туда не факт, а требование к ответу. «Лимит 100 тысяч строк» — рабочая память,
«асинхронные выгрузки у нас везде» — долговременная, «отвечай короче» — профиль.

Находки бывают двух видов. Обычная — формулировка в раздел: её можно положить на
любой из трёх адресов и перенести между ними. Правка шкалы — только про профиль:
у неё нет текста, есть шкала и новое значение из её списка.

Решение модуль не принимает: кандидат — это предложение, а не запись. Куда пункт
ляжет и ляжет ли вообще, решает пользователь кнопкой (или, если он этого попросил,
агент по своей же рекомендации). Сам запрос делает агент, хранит всё база.
"""

import json
from dataclasses import dataclass

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

import profile as user_profile
from memory import LONGTERM, SECTIONS, TIERS, WORKING, Store
from profile import PROFILE, SCALES, Profile

# Отказ — такое же решение, как и выбор адреса, и обрабатывается там же.
SKIP = "skip"

# Полный словарь ответов на вопрос «куда это»: два уровня памяти, профиль и «никуда».
# Он живёт здесь, а не в memory.py: память не знает ни про профиль, ни про то, что
# у находки бывает четвёртая судьба.
TARGETS = (*TIERS, PROFILE, SKIP)

# Вид находки: формулировка в раздел или правка шкалы профиля.
ITEM = "item"
SCALE = "scale"

# Разделы всех трёх адресов в одном месте: карточке нужен список по адресу, и
# память с профилем здесь равноправны.
SECTIONS_BY_TARGET = {**SECTIONS, PROFILE: user_profile.SECTIONS}

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


def _scales_help() -> str:
    return "\n".join(
        f"- {scale.key} ({scale.about}): {' / '.join(scale.options)}" for scale in SCALES
    )


PROMPT = f"""\
Ты ведёшь память и профиль агента и раскладываешь по ним то, что всплыло в разговоре.

Адресов три.

longterm — долговременная память: верно независимо от текущей задачи и останется
верным в следующей. Принятые принципы работы, устройство команды и её окружения.
Разделы: {", ".join(SECTIONS[LONGTERM])}.

working — рабочая память: техническое задание текущей задачи. Верно только здесь
и потеряет смысл, когда задача закончится.
Разделы: {", ".join(SECTIONS[WORKING])}.

profile — профиль пользователя: не факт о задаче, а требование к форме ответа или
сведение о самом собеседнике.
Разделы: {", ".join(user_profile.SECTIONS)}.

У профиля есть шкалы с готовыми значениями. Если пользователь просит изменить саму
форму ответа, верни правку шкалы, а не текст:
{_scales_help()}

На входе — профиль, обе памяти целиком и последний обмен репликами. Верни то, что
стоит записать по итогам этого обмена.

Правила:
- Формулировка законченная и самостоятельная: её прочитают без разговора вокруг.
- Не длиннее {ITEM_WORDS} слов, на языке диалога, без «пользователь сказал».
- Только то, что сказал или подтвердил пользователь. Вариант, который предложил
  ассистент и никто не принял, — не факт.
- Что уже есть в памяти или в профиле, не повторяй, даже другими словами.
- Общее правило и частное требование различай по сроку: «пишем на Python» верно
  всегда, «здесь нужен Redis» — только в этой задаче.
- В профиль отправляй только про манеру ответа и про самого человека: «не надо
  списков», «объясняй термины», «я перешёл в другую команду». Решение по задаче,
  даже сказанное как пожелание, — это память, а не профиль.
- Просьбу изменить форму ответа отдавай шкалой, если она среди шкал есть, и
  свободным пунктом, если такой шкалы нет.
- Сомневаешься между уровнями памяти — выбирай working: перенести в долговременную
  пользователь может и сам.
- Не больше {ITEM_LIMIT} пунктов за ход, самых важных. Ход ни о чём — пустой список.

Поле why — одна короткая строка, зачем это помнить: её увидит пользователь рядом
с кнопками выбора.

Верни строго JSON без markdown и пояснений:
{{"items": [{{"text": "...", "target": "working", "section": "требования", "why": "..."}},
 {{"target": "profile", "scale": "длина", "value": "коротко", "why": "..."}}]}}
Записывать нечего — верни {{"items": []}}.
"""


@dataclass(frozen=True)
class Candidate:
    """Предложение записать: адрес, формулировка или правка шкалы, и зачем это помнить."""

    text: str
    tier: str
    section: str
    why: str = ""
    kind: str = ITEM
    # Только у правки шкалы: имя шкалы и новое значение из её списка.
    scale: str = ""
    value: str = ""


def clean(text: object) -> str:
    return " ".join(str(text).split())


def key(text: str) -> str:
    """Ключ для сравнения формулировок: регистр и пунктуация здесь не различают."""
    return "".join(character for character in clean(text).lower() if character.isalnum() or character == " ")


def request(
    profile: Profile | None,
    longterm: Store | None,
    working: Store | None,
    prompt: str,
    answer: str,
) -> list[ChatCompletionMessageParam]:
    """Ход уходит на разбор расшифровкой, а не диалогом.

    Роли в таком запросе сбили бы модель с задачи: получив чат, она продолжает
    разговор, а нам нужна классификация сказанного в нём. Профиль едет вместе с
    памятью и по той же причине: иначе модель раз в три хода предлагает записать
    предпочтение, которое в профиле уже стоит.
    """
    return [
        ChatCompletionSystemMessageParam(role="system", content=PROMPT),
        ChatCompletionUserMessageParam(
            role="user",
            content=(
                f"Профиль:\n{profile.text if profile else EMPTY}\n\n"
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

    if entry.get("scale"):
        return _scale_candidate(entry)

    text = clean(entry.get("text", ""))
    if not text:
        return None

    # Уровень по умолчанию рабочий: то же правило, что и в промпте, — общее
    # пользователь поднимет сам, а частное в долговременной памяти мешает всем.
    tier = clean(entry.get("target", "")).lower()
    if tier == PROFILE:
        return Candidate(
            text=text,
            tier=PROFILE,
            section=user_profile.section_of(entry.get("section", "")),
            why=clean(entry.get("why", "")),
        )
    if tier not in TIERS:
        tier = WORKING

    return Candidate(text=text, tier=tier, section=section_of(tier, entry.get("section", "")), why=clean(entry.get("why", "")))


def _scale_candidate(entry: dict[str, object]) -> Candidate | None:
    """Правка шкалы: имя и значение должны быть из списка, иначе это не находка.

    Шкала с выдуманным значением («длина: телеграфно») хуже, чем её отсутствие:
    в промпт она уйдёт строкой, которой у шкалы нет, и профиль перестанет быть
    предсказуемым. Такой кандидат отбрасывается целиком.
    """
    scale = user_profile.scale_of(entry.get("scale", ""))
    if scale is None:
        return None

    value = clean(entry.get("value", "")).lower()
    if value not in scale.values:
        return None

    return Candidate(
        # Текст у правки шкалы служебный: он нужен, чтобы карточку можно было
        # свернуть в такую же строку, как остальные.
        text=f"{scale.key}: {value}",
        tier=PROFILE,
        section="",
        why=clean(entry.get("why", "")),
        kind=SCALE,
        scale=scale.key,
        value=value,
    )


def section_of(tier: str, raw: object) -> str:
    """Раздел из ответа модели — к списку разделов уровня."""
    if tier == PROFILE:
        return user_profile.section_of(raw)

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
    profile: Profile | None = None,
) -> tuple[Candidate, ...]:
    """Кандидаты, которых в памяти и в профиле ещё нет.

    Промпт просит не повторяться, но просьба — не гарантия: модель охотно
    предлагает записать то, что уже лежит в блоке у неё перед глазами. У шкалы
    свой способ повториться — предложить значение, которое и так стоит.
    """
    known = {key(item.text) for store in (longterm, working) if store for item in store.items}
    known.update(key(item.text) for item in (profile.items if profile else ()))

    chosen = []
    for candidate in candidates:
        if candidate.kind == SCALE:
            if profile is not None and profile.value(candidate.scale) == candidate.value:
                continue
            chosen.append(candidate)
            continue

        marker = key(candidate.text)
        if marker in known:
            continue
        known.add(marker)
        chosen.append(candidate)
    return tuple(chosen)
