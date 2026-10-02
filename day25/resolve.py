"""Из реплики — самостоятельный вопрос: то, что вообще можно искать.

В [day22](../day22/answer.py)–[day24](../day24/answer.py) вопрос приходил готовым:
набор написан руками, каждая строка самодостаточна, искать можно её саму. В чате
это неправда начиная со второй реплики. «А в нём как?» в индексе не найдётся
ничего: слова «нём» в корпусе нет, а то, к чему оно относится, лежало в прошлом
ходе.

Поэтому между репликой и конвейером [day24](../day24/pipeline.py) появляется шаг,
который превращает реплику в вопрос, понятный без разговора. Дальше конвейер
работает ровно как в day24 — переписывание `keyword`, поиск, реранкер, — и этого
достаточно: его входом снова становится самодостаточная строка.

## Почему не новый режим в rewrite.py

Соблазн был добавить четвёртый режим в [rewrite.py](rewrite.py) и переиспользовать
его кэш. Отвергнуто по сигнатуре: `rewrite.apply(mode, question)` видит только
вопрос, и протащить туда историю значило бы поменять интерфейс у всех трёх
прежних режимов, которые в истории не нуждаются. Хуже того, поехал бы кэш:
подпись в `rewrites.json` считается по тексту вопроса, а здесь один и тот же
вопрос в двух разговорах обязан дать два разных запроса.

Так что шаг стоит снаружи и отдельно, а внутрь конвейера уходит его результат.
Два переписывания подряд — да, и это сознательно: разрешение ссылок и выжимка
ключевых слов решают разные задачи, и склеивать их в один промпт значило бы
получить шаг, у которого нельзя проверить ни одну из двух работ.

## Дешёвый отказ от работы

Большинство реплик самодостаточны: «Что такое sticky facts?» переписывать нечем.
Поэтому до модели стоит проверка в коде — есть ли в реплике за что цепляться:
местоимение, указание на прошлый ход, слишком короткая фраза. Нет — реплика едет
в поиск как есть, и ход не платит ни токена. Проверка грубая и работает в одну
сторону: пропустить лишнюю реплику к модели не страшно, а вот не пропустить
ссылочную — значит искать по слову «нём».
"""

import re
from dataclasses import dataclass

import llm
from state import TaskState

TEMPERATURE = 0.0
MAX_TOKENS = 120

# Сколько последних реплик показывать. Шесть — три хода, та часть разговора, где
# ссылка ещё может на что-то указывать: «а в предыдущем?» дальше трёх ходов не
# достаёт, а длинный хвост начинает тянуть запрос на старую тему.
TAIL = 6

# Короче этого реплика почти наверняка неполна: «а дальше?», «почему?», «а там?».
SHORT_WORDS = 4

# За что цепляется ссылка. Список грубый и нужен только чтобы не ходить к модели
# на каждой самодостаточной реплике; ошибка в его пользу бесплатна.
HOOKS = re.compile(
    r"\b("
    r"он|она|оно|они|его|её|их|нём|ней|них|ему|ей|им|там|туда|тут|это|этот|эта|эти|"
    r"тот|та|те|такой|такая|такие|так|тогда|оба|первый|второй|третий|последний|"
    r"предыдущ\w*|выше|раньше|он[аи]?же"
    r")\b",
    re.IGNORECASE,
)

# Начало реплики, которое само по себе означает продолжение разговора.
OPENERS = re.compile(r"^\s*(а|и|но|ещё|еще|тогда|значит|ок|хорошо|ладно)\b", re.IGNORECASE)

SYSTEM = (
    "Ты готовишь поисковый запрос к базе проекта AI Advent по реплике из диалога.\n"
    "Верни одну строку — вопрос, который понятен без разговора: подставь вместо "
    "местоимений и ссылок на предыдущие ходы то, к чему они относятся.\n"
    "Сохрани смысл и объём вопроса. Не отвечай на него, не добавляй подробностей, "
    "которых в реплике нет, и не расширяй его до соседних тем.\n"
    "Если реплика и так понятна без разговора, повтори её без изменений.\n"
    "Никаких пояснений, кавычек и префиксов — только сам вопрос."
)


@dataclass(frozen=True)
class Resolved:
    """Реплика и то, что из неё поехало в поиск."""

    question: str
    standalone: str
    changed: bool = False
    used_state: bool = False
    asked: bool = False
    reply: llm.Reply | None = None

    @property
    def tokens(self) -> int:
        return (self.reply.prompt_tokens + self.reply.completion_tokens) if self.reply else 0

    @property
    def seconds(self) -> float:
        return self.reply.seconds if self.reply else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "standalone": self.standalone,
            "changed": self.changed,
            "used_state": self.used_state,
            "asked": self.asked,
            "tokens": self.tokens,
            "seconds": round(self.seconds, 2),
        }


def plain(question: str) -> Resolved:
    """Реплика как есть: режим без памяти и самодостаточные вопросы."""
    body = question.strip()
    return Resolved(question=body, standalone=body)


def dependent(question: str) -> bool:
    """Похоже ли, что реплику без разговора не понять."""
    body = question.strip()
    if not body:
        return False
    if len(body.split()) <= SHORT_WORDS:
        return True
    return bool(HOOKS.search(body) or OPENERS.match(body))


def messages(
    question: str, history: list[dict[str, str]], current: TaskState | None
) -> list[dict[str, str]]:
    """Запрос шага: цель с терминами, хвост разговора и сама реплика."""
    parts: list[str] = []

    hint = current.query_hint() if current else ""
    if hint:
        parts.append(f"О чём разговор и что в нём значат слова:\n{hint}")

    tail = "\n".join(
        f"{'Пользователь' if item['role'] == 'user' else 'Ассистент'}: {item['content']}"
        for item in history[-TAIL:]
    )
    if tail:
        parts.append(f"Предыдущие реплики:\n{tail}")

    parts.append(f"Реплика, которую нужно сделать самостоятельной:\n{question}")

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _clean(text: str, fallback: str) -> str:
    """Одна строка без кавычек и префиксов. Пустой ответ — это отказ от правки."""
    body = " ".join(text.split()).strip().strip('"«»')
    body = re.sub(r"^(запрос|вопрос|поисковый запрос)\s*[:—-]\s*", "", body, flags=re.IGNORECASE)
    return body or fallback


async def apply(
    question: str,
    history: list[dict[str, str]] | None = None,
    current: TaskState | None = None,
) -> Resolved:
    """Разрешить ссылки в реплике. Самодостаточную реплику к модели не несёт."""
    body = question.strip()
    if not body:
        raise ValueError("Пустую реплику искать нечем.")

    history = history or []
    hint = current.query_hint() if current else ""

    # Без разговора и без состояния разрешать нечем: ссылке некуда указывать.
    if not history and not hint:
        return Resolved(question=body, standalone=body)

    if not dependent(body):
        return Resolved(question=body, standalone=body, used_state=bool(hint))

    reply = await llm.complete(
        messages(body, history, current),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    standalone = _clean(reply.text, body)

    return Resolved(
        question=body,
        standalone=standalone,
        changed=standalone.lower() != body.lower(),
        used_state=bool(hint),
        asked=True,
        reply=reply,
    )
