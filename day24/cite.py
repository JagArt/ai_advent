"""Форма ответа: утверждение, его источник и дословная цитата — и ничего больше.

В [day23](../day23/answer.py) ответ был свободным текстом, а подкрепление —
номером в квадратных скобках: «Постоянный MCP-сервер слушает на порту 8770 [1]».
Проверить такую ссылку можно ровно одним способом — поверить, что в выдержке [1]
действительно написано про 8770. Номер говорит *откуда*, но не *что именно*
оттуда взято, и между выдержкой и утверждением остаётся зазор шириной в чанк.

Здесь зазор закрывается цитатой. Модель обязана вернуть JSON, и в нём у каждого
утверждения три поля: текст, номер выдержки и кусок её текста, из которого
утверждение следует. Цитату уже можно искать в чанке посимвольно — чем и
занимается [verify.py](verify.py).

## Почему отдельного поля «ответ» нет

Напрашивалась схема «ответ целиком плюс список цитат рядом». Она отвергнута, и
причина не стилистическая: поле со свободным текстом — это место, куда можно
записать неподкреплённое предложение, и ни схема, ни сверка его не поймают.
Цитаты лежали бы рядом с ответом, а не под ним, и «обязательность цитат»
означала бы «цитаты где-то приложены».

Поэтому ответа как отдельной сущности нет вовсе: **ответ — это и есть список
утверждений**, а его текст собирается склейкой. Написать предложение без цитаты
структурно негде. Цена известна и принята: ответ получается рубленее, чем
связный абзац day23, — зато каждое его предложение показывает своё основание.

## Отказ — такое же значение схемы, а не особый случай

`unknown: true` вместе с `clarify` — это не ошибка разбора и не пустой ответ, а
полноправная ветка той же схемы. Модель, которой нечем подкрепить ни одного
утверждения, обязана ею воспользоваться: список утверждений тогда пуст, а в
`clarify` лежит вопрос, который вернёт разговор к тому, что в базе есть.

## Чинить разбор, а не переспрашивать

JSON приходит сорванным двумя способами. Первый — обёртка в ```` ```json ````,
хотя режим `json_object` её запрещает. Второй опаснее: ответ упёрся в
`max_tokens` и обрывается посреди цитаты. Повторный вызов тут стоил бы ещё
одного полного запроса с контекстом, а спасти можно и так: утверждения
независимы, и те, что дописаны до обрыва, годны целиком. Поэтому `parse`
вытаскивает все закрытые объекты и отмечает в `broken`, что ответ был починен, —
замер обязан видеть такие случаи, а не считать их нормальными.
"""

import json
import re
from dataclasses import dataclass, field

REFUSAL = "В базе ответа нет"

# Схема печатается в промпт дословно: модель повторяет форму надёжнее, чем
# следует её описанию прозой. Слово «json» в промпте обязательно — без него
# DeepSeek отклоняет `response_format={"type": "json_object"}`.
SCHEMA = """{
  "unknown": false,
  "clarify": "",
  "claims": [
    {"text": "одно утверждение", "source": 1, "quote": "дословный кусок выдержки [1]"}
  ]
}"""

SYSTEM = (
    "Ты отвечаешь на вопросы о проекте AI Advent — репозитории с дневником итераций, "
    "где есть документация, код на Python и PDF.\n"
    "Ответ возвращается строго в формате json по схеме:\n"
    f"{SCHEMA}\n"
    "Поля значат вот что.\n"
    "`claims` — ответ, разложенный на отдельные утверждения, два-четыре штуки. "
    "Другого места для ответа в схеме нет: что не попало в `claims`, того ты не сказал.\n"
    "`text` — само утверждение, одно предложение по-русски, без вступлений.\n"
    "`source` — номер выдержки, из которой утверждение взято.\n"
    "`quote` — кусок текста этой выдержки, скопированный дословно, от десяти слов до "
    "двух предложений. Его будут искать в выдержке посимвольно: пересказ, перевод, "
    "исправленная опечатка и собранная из разных мест фраза не найдутся.\n"
    "`unknown` — true, если ни одного утверждения подкрепить нечем.\n"
    "`clarify` — при `unknown` вопрос, который поможет спросить о том, что в базе есть; "
    "иначе пустая строка."
)

RULES = (
    "Ниже выдержки из базы проекта, каждая под своим номером.\n"
    "Отвечай только по ним, не добавляя ничего от себя.\n"
    "Числа, имена файлов и названия приводи ровно так, как они стоят в выдержках.\n"
    "Утверждение без дословной цитаты из выдержки писать нельзя: нечего процитировать — "
    "значит, нечего и утверждать.\n"
    f"Если выдержки на вопрос не отвечают, верни `unknown: true`, пустой `claims` "
    f"и уточняющий вопрос в `clarify` — это и есть ответ «{REFUSAL}»."
)

FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")

# Поля отказа вытаскиваются отдельными регулярками: в сорванном ответе объект
# верхнего уровня может не закрыться, и `json.loads` до них уже не доберётся.
UNKNOWN = re.compile(r'"unknown"\s*:\s*(true|false)')
CLARIFY = re.compile(r'"clarify"\s*:\s*"((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class Claim:
    """Одно утверждение ответа: что сказано, откуда взято и чем подкреплено.

    `source` хранится ровно таким, каким его назвала модель, — включая номера,
    которых в контексте нет. Выдуманная ссылка должна доехать до замера, а не
    исчезнуть при разборе.
    """

    text: str
    source: int | None
    quote: str

    def as_dict(self) -> dict[str, object]:
        return {"text": self.text, "source": self.source, "quote": self.quote}


@dataclass(frozen=True)
class Cited:
    """Разобранный ответ модели. Отказ — такое же его состояние, как и ответ."""

    unknown: bool = False
    clarify: str = ""
    claims: list[Claim] = field(default_factory=list)
    broken: str = ""
    raw: str = ""

    @property
    def text(self) -> str:
        """Ответ прозой: склейка утверждений, а при отказе — отказ с уточнением."""
        if self.unknown or not self.claims:
            clarify = f" {self.clarify.strip()}" if self.clarify.strip() else ""
            return f"{REFUSAL}.{clarify}".strip()
        return " ".join(claim.text.strip() for claim in self.claims if claim.text.strip())

    @property
    def refused(self) -> bool:
        return self.unknown or not self.claims

    def as_dict(self) -> dict[str, object]:
        return {
            "unknown": self.unknown,
            "refused": self.refused,
            "clarify": self.clarify,
            "claims": [claim.as_dict() for claim in self.claims],
            "broken": self.broken,
        }


# --- разбор ------------------------------------------------------------------


def _objects(text: str) -> list[str]:
    """Закрытые объекты `{...}` на верхнем уровне массива `claims`.

    Скобки считаются вручную, потому что обрыв по `max_tokens` оставляет массив
    незакрытым, и любой разбор целого документа на таком ответе падает. Строки
    пропускаются вместе с экранированием, иначе `{` внутри цитаты сбил бы счёт.
    """
    start = text.find('"claims"')
    if start < 0:
        return []

    opened = text.find("[", start)
    if opened < 0:
        return []

    found: list[str] = []
    depth = 0
    begin = 0
    inside = False
    escaped = False

    for position in range(opened, len(text)):
        symbol = text[position]

        if inside:
            if escaped:
                escaped = False
            elif symbol == "\\":
                escaped = True
            elif symbol == '"':
                inside = False
            continue

        if symbol == '"':
            inside = True
        elif symbol == "{":
            if depth == 0:
                begin = position
            depth += 1
        elif symbol == "}":
            depth -= 1
            if depth == 0:
                found.append(text[begin : position + 1])
        elif symbol == "]" and depth == 0:
            break

    return found


def _claim(payload: object) -> Claim | None:
    if not isinstance(payload, dict):
        return None

    source = payload.get("source")
    if isinstance(source, str):
        digits = re.search(r"\d+", source)
        source = int(digits.group()) if digits else None
    elif isinstance(source, bool) or not isinstance(source, int):
        source = None

    text = str(payload.get("text") or "").strip()
    quote = str(payload.get("quote") or "").strip()

    # Утверждение без текста — это не утверждение. Пустую цитату, наоборот,
    # выбрасывать нельзя: ровно её отсутствие колонка «цитаты есть» и считает.
    return Claim(text=text, source=source, quote=quote) if text else None


def _salvage(text: str) -> Cited:
    """Собрать, что уцелело: закрытые утверждения и поля отказа по отдельности."""
    claims = [claim for raw in _objects(text) if (claim := _claim(_loose(raw))) is not None]

    unknown = UNKNOWN.search(text)
    clarify = CLARIFY.search(text)

    return Cited(
        unknown=bool(unknown and unknown.group(1) == "true") or not claims,
        clarify=_unescape(clarify.group(1)) if clarify else "",
        claims=claims,
        broken="ответ не разобрался как json, собран по кускам",
        raw=text,
    )


def _loose(raw: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _unescape(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value


def parse(text: str) -> Cited:
    """Ответ модели в `Cited`. Сорванный json чинится локально, а не переспрашивается."""
    body = FENCE.sub("", text.strip())

    payload = _loose(body)
    if not isinstance(payload, dict):
        return _salvage(body)

    claims = [
        claim
        for item in (payload.get("claims") or [])
        if (claim := _claim(item)) is not None
    ]

    # `unknown: false` при пустом списке — противоречие, и разрешается оно в
    # пользу отказа: утверждений нет, значит ответа нет.
    return Cited(
        unknown=bool(payload.get("unknown")) or not claims,
        clarify=str(payload.get("clarify") or "").strip(),
        claims=claims,
        raw=body,
    )
