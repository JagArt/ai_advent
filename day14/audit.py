"""Проверка ответа по инвариантам: то, что не поймать паттерном.

Второй слой guard'а и второй способ узнать о нарушении. Первый живёт в
invariants.scan: он ищет запрещённую формулировку, и найденное им — не мнение, а
факт, поэтому оно обрывает ответ на полуслове. Здесь всё наоборот: аудитор читает
уже отданный ответ и оценивает его по правилам, у которых детектора нет и быть не
может — «наружу смотрит только шлюз» словами не перечислить.

Отсюда и разница в силе. Детектор переписывает ответ, аудитор приносит карточку:
задним числом переписать отданное нельзя, а знать о нарушении полезнее, чем не знать.

Модуль решений не принимает и правил не толкует: он собирает запрос и читает ответ.
Названного инварианта нет среди действующих или цитата пустая — нарушение
отбрасывается целиком, как выдуманный переход в progress.py: нарушение, придуманное
моделью, хуже пропущенного, потому что чинить по нему нечего.

Здесь же лежит инструкция второго прохода. Она про то же самое — что делать с
найденным нарушением, — и жить ей рядом с промптом аудита логичнее, чем в агенте.
"""

import json
from dataclasses import dataclass

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from invariants import Hit, Rule, clean

# Проверка — не разговор: нужна оценка написанного, а не рассуждение о правилах.
TEMPERATURE = 0.0
MAX_TOKENS = 400

# Кто поймал нарушение: код по детектору или модель. Строка уходит в журнал и в
# карточку — по ней видно, доказано нарушение или оценено.
SCAN = "scan"
AUDIT = "audit"

# Где наткнулись: в ответе агента или в пункте, который просили записать в память.
ANSWER = "answer"
ITEM = "item"

PROMPT = """\
Ты проверяешь ответ ассистента по инвариантам — правилам, которые он не имеет права
нарушать. На входе список инвариантов с номерами и ответ ассистента.

Нарушение — это когда ассистент предлагает, советует или соглашается сделать то, что
инвариант запрещает. Не нарушение:
- упоминание запрещённого варианта в отказе от него;
- объяснение, почему путь закрыт, и предложение альтернативы;
- пересказ того, что просил пользователь, без согласия это делать;
- обсуждение того, что уже сделано и здесь не решается.

Правила:
- Ссылайся только на номера из списка. Своих инвариантов не придумывай.
- quote — дословный фрагмент ответа, в котором видно нарушение. Не пересказывай его
  и не сочиняй: цитата без опоры в тексте будет отброшена.
- Сомневаешься — нарушения нет. Ложное нарушение дороже пропущенного: по нему
  пользователю нечего исправлять.
- Одно нарушение на инвариант, даже если оно повторяется в ответе несколько раз.

Поле why — одна короткая строка, что именно нарушено: её увидит пользователь в карточке.

Верни строго JSON без markdown и пояснений:
{"violations": [{"id": 3, "quote": "...", "why": "..."}]}
Ответ в рамках инвариантов — верни {"violations": []}.
"""

# Второй проход. Инвариант называется целиком, вместе с альтернативой и с тем куском,
# на котором ответ оборвался: без цитаты модель переписывает наугад и обрывается снова.
#
# Про «скажи прямо» здесь не вежливость, а суть задания: отказ, о котором пользователь
# не узнал, — это не отказ, а тихая подмена его решения.
RETRY = """\
Предыдущий ответ был отброшен: он нарушал инвариант, и пользователь его не увидел.

{report}

Ответь на тот же вопрос заново, оставаясь в рамках инварианта. Прямо скажи, что
предложенный путь закрыт и почему, и назови то, что делать вместо него. Не пересказывай
эту инструкцию, не упоминай отброшенный ответ и не извиняйся.
"""


@dataclass(frozen=True)
class Verdict:
    """Что назвала модель. Ещё не нарушение: номер, цитата и причина, больше ничего."""

    id: int
    quote: str
    why: str = ""


def rules_text(rules: tuple[Rule, ...]) -> str:
    """Инварианты списком с номерами: по этим номерам аудитор и отвечает."""
    return "\n".join(rule.line for rule in rules)


def request(
    rules: tuple[Rule, ...],
    prompt: str,
    answer: str,
) -> list[ChatCompletionMessageParam]:
    """Ответ уходит на проверку расшифровкой, а не диалогом — как и разбор находок.

    Реплика пользователя едет вместе с ответом: без неё «сделаем на Flask» и «на Flask
    нельзя, потому что вы просили именно его» неотличимы, а различать их обязательно.
    """
    return [
        ChatCompletionSystemMessageParam(role="system", content=PROMPT),
        ChatCompletionUserMessageParam(
            role="user",
            content=(
                f"Инварианты:\n{rules_text(rules)}\n\n"
                f"Реплика пользователя:\n{prompt}\n\n"
                f"Ответ ассистента:\n{answer}"
            ),
        ),
    ]


def clean_json(text: str) -> str:
    body = text.strip()
    if body.startswith("```"):
        body = body.removeprefix("```json").removeprefix("```").removesuffix("```")
    return body.strip()


def parse(text: str) -> tuple[Verdict, ...]:
    """Разбор ответа аудитора: список нарушений, всё остальное — ошибка.

    Пустой список и «нарушений нет» — одно и то же и ошибкой не считаются: ход, на
    котором агент никого не нарушил, — обычный ход, а не сбой разбора.
    """
    body = clean_json(text)
    try:
        parsed = json.loads(body)
    except ValueError as error:
        raise ValueError(f"ответ не разобран как JSON: {clean(text)[:80]}") from error

    # Модель иногда отдаёт голый список вместо объекта с полем violations: форма не
    # та, содержимое то самое.
    raw = parsed if isinstance(parsed, list) else parsed.get("violations") if isinstance(parsed, dict) else None
    if not isinstance(raw, (list, tuple)):
        raise ValueError("ожидался объект с полем violations")

    verdicts = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            number = int(str(entry.get("id", "")).lstrip("#"))
        except ValueError:
            continue
        verdicts.append(
            Verdict(id=number, quote=clean(entry.get("quote", "")), why=clean(entry.get("why", ""))),
        )
    return tuple(verdicts)


def confirm(rules: tuple[Rule, ...], verdicts: tuple[Verdict, ...]) -> tuple[list[Hit], list[str]]:
    """Вердикты — в нарушения и в отказы.

    Проверяется ровно два условия, и оба про то, можно ли по нарушению что-то сделать:
    инвариант с таким номером должен действовать, а цитата — быть. Нарушение без
    номера нечего показывать в карточке, нарушение без цитаты нечем подтвердить.
    """
    known = {rule.id: rule for rule in rules if rule.enabled}
    hits: list[Hit] = []
    rejected: list[str] = []
    seen: set[int] = set()

    for verdict in verdicts:
        rule = known.get(verdict.id)
        if rule is None:
            rejected.append(f"инварианта #{verdict.id} нет среди действующих")
            continue
        if not verdict.quote:
            rejected.append(f"нарушение {rule.label} без цитаты")
            continue
        if verdict.id in seen:
            continue
        seen.add(verdict.id)
        hits.append(Hit(rule=rule, banned="", quote=verdict.quote))
    return hits, rejected


def report(hits: tuple[Hit, ...] | list[Hit]) -> str:
    """Нарушения словами: то же, что уходит во второй проход и в строку отказа."""
    lines = []
    for hit in hits:
        lines.append(f"Нарушен {hit.rule.summary}")
        if hit.rule.instead:
            lines.append(f"В рамках инварианта: {hit.rule.instead}")
        if hit.quote:
            lines.append(f"Оборванный ответ дошёл до: «{hit.quote}»")
    return "\n".join(lines)


def refusal(hits: tuple[Hit, ...] | list[Hit]) -> str:
    """Отказ, написанный кодом: последнее слово, когда модель дважды не справилась.

    Такой ответ хуже любого, который написала бы модель: он не отвечает на вопрос и не
    продолжает разговор. Но он ровно то, чем этот день заканчивается по определению —
    ассистент не предложит решения, нарушающего инвариант, даже если больше ему
    предложить нечего. Альтернатива в нём есть всегда: она взята из самого инварианта.
    """
    lines = ["Этот путь закрыт — он нарушает инвариант, а инварианты я не обхожу."]
    for hit in hits:
        lines.append(f"- {hit.rule.summary}")
        if hit.rule.instead:
            lines.append(f"  Вместо этого: {hit.rule.instead}")
    lines.append("Снять инвариант можно только в панели — это ваше решение, не моё.")
    return "\n".join(lines)


def retry_param(hits: tuple[Hit, ...] | list[Hit]) -> ChatCompletionMessageParam:
    """Системное сообщение второго прохода: инвариант назван, альтернатива названа."""
    return ChatCompletionSystemMessageParam(
        role="system",
        content=RETRY.format(report=report(hits)),
    )
