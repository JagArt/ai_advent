"""Область этапа: что работа текущего этапа, а что — работа за гейтом.

Гейты из gates.py держат состояние: пока план не утверждён, задача не окажется на
исполнении. Само по себе это не мешает агенту делать работу исполнения — отвечать
схемой таблиц и списком эндпоинтов, сидя на планировании. Состояние при этом стоит
на месте, и формально всё честно: этап не перепрыгнут, он просто не имеет значения.

Этот модуль закрывает ровно эту дырку. У каждого этапа есть область: что он
отпускает и какая работа принадлежит этапу за гейтом. Работа за гейтом ловится тем
же двухслойным guard'ом, которым в day14 ловились инварианты:

    детектор   доказывает паттерном   обрывает ответ раньше, чем он дошёл до экрана
    аудитор    оценивает моделью      приносит карточку и строку в журнал

Механика детектора взята из invariants.py целиком — окно вокруг совпадения, оборот
для отрицания, суждение только о законченных предложениях. Отличается словарь и одно
правило: глагол предложения здесь не требуется. Инвариант нарушают, когда запрещённое
*предлагают*; область этапа нарушают тем, что работу *делают*, и «схема: таблица
reports с колонками…» — это уже сделанная работа, без всякого «давай».

Зато добавлен второй список снятий. Отказ от работы вперёд почти всегда выглядит как
отсрочка — «схему таблиц спроектируем после утверждения плана», — и без слов отсрочки
детектор обрывал бы ровно те ответы, которые делают то, что от них требуется.

Модуль ничего не знает ни про базу, ни про сеть: он хранит области, ищет забеги
вперёд и собирает промпт аудитора. Запросы делает агент, гейты считает gates.py.
"""

import json
import re
from dataclasses import dataclass

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

import gates
import invariants
from invariants import clean
from task import BY_KEY, DONE, EXECUTION, PLANNING, VALIDATION, Task

# Проверка — не разговор: нужна оценка написанного, а не рассуждение об этапах.
TEMPERATURE = 0.0
MAX_TOKENS = 400

# Кто поймал забег: код по детектору или модель-аудитор. Строка уходит в журнал и в
# карточку — по ней видно, доказан забег или оценен.
SCAN = "scan"
AUDIT = "audit"

# Слова отсрочки. Снимают совпадение так же, как отрицание, и по той же причине: в
# обороте «схему таблиц нарисуем после утверждения плана» работа не сделана, а
# названа — и назвать её приходится, потому что блок гейтов прямо требует объяснить,
# что держит путь дальше. Без этого списка guard наказывал бы за исполнение задания.
DEFERRING = (
    r"\bпозже\b",
    r"\bпотом\b",
    r"\bпока\b",
    r"\bсначала\b",
    r"\bрано\b",
    r"на этапе",
    r"на следующем этапе",
    r"после утвержд\w*",
    r"после гейта",
    r"когда утверд\w*",
    r"гейт",
    r"утверд\w*",
    r"этап\w* (?:исполнени|проверк|готово)",
)

_NEGATION = re.compile("|".join(invariants.NEGATIONS))
_DEFER = re.compile("|".join(DEFERRING))

# Объявление готовности. Список общий для всех этапов до финала: «ТЗ принято» —
# решение пользователя, и агент не объявляет его ни на планировании, ни на проверке.
FINAL = (
    r"т[зэ] (?:готов\w*|принят\w*|утвержден\w*|утверждён\w*)",
    r"задача (?:готов\w*|закрыт\w*|завершен\w*|завершён\w*)",
    r"можно отдавать в работу",
    r"отда(?:ём|ем|дим) в работу",
    r"считаю (?:т[зэ]|задачу) (?:готов\w*|принят\w*|закрыт\w*)",
    r"переход(?:им|ит) (?:к|в) (?:готово|финал\w*)",
    r"объявля\w+ (?:т[зэ]|задачу) готов\w*",
)


@dataclass(frozen=True)
class Reach:
    """Работа этапа впереди: чем называется, чем держится и чем её заменить.

    gate — ключ гейта, который эту работу держит, и он же делает область живой, а не
    нарисованной: открылся гейт — работа стала твоей, и детектор про неё забывает.
    Закрытая область, которая не привязана ни к одному гейту, была бы просто запретом
    на слова.
    """

    stage: str
    work: str
    gate: str
    instead: str
    banned: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return BY_KEY[self.stage].name

    @property
    def summary(self) -> str:
        """Одна строка для отказа и журнала: чья это работа и что её держит."""
        held = gates.by_key(self.gate)
        label = held.label if held else "гейт"
        return f"работа этапа «{self.name}»: {self.work} — её держит {label}"


@dataclass(frozen=True)
class Scope:
    """Область этапа: что он отпускает и что принадлежит этапам за гейтом."""

    stage: str
    allows: tuple[str, ...]
    reaches: tuple[Reach, ...] = ()


# Области этапов. Списка «чего нельзя» на каждый этап нет и быть не может: этапов
# впереди бывает несколько, а работа за гейтом различается не темой, а тем, какой
# гейт её держит. Поэтому область собрана из Reach — по одной записи на «работу,
# которая ждёт гейта».
#
# Забеги описаны только те, что стоят за гейтом подтверждения, и это не экономия:
# работа, которую пускает авто-гейт, не требует чужого решения, и делать её раньше
# времени — неаккуратность, а не подмена решения человека. Ровно два примера из
# задания — реализация до утверждённого плана и финал без валидации — как раз и есть
# два гейта подтверждения этого приложения.
IMPLEMENTATION = Reach(
    stage=EXECUTION,
    work="проектировать реализацию — схема таблиц, эндпоинты, код",
    gate="plan_approved",
    instead="довести требования и ограничения до формулировок",
    banned=(
        r"create table",
        r"\bddl\b",
        r"схем[ауы] (?:таблиц|базы|данных)",
        r"таблиц\w+ \w+ (?:с колонк|с пол)",
        r"колонк\w+ \w+ (?:text|integer|uuid|timestamp|numeric|bool)",
        r"\bэндпоинт\w*",
        r"\bendpoint\w*",
        r"\b(?:get|post|put|patch|delete) /",
        r"\bапи-метод\w*",
        r"структур\w+ (?:таблиц|индекс)\w*",
    ),
)

ACCEPTANCE = Reach(
    stage=DONE,
    work="объявлять ТЗ принятым и задачу закрытой",
    gate="spec_accepted",
    instead="назвать гейт и попросить утверждение у пользователя",
    banned=FINAL,
)

SCOPES = {
    PLANNING: Scope(
        stage=PLANNING,
        allows=(
            "выяснять цель задачи",
            "собирать требования",
            "называть сроки, объёмы и запреты",
        ),
        reaches=(IMPLEMENTATION, ACCEPTANCE),
    ),
    EXECUTION: Scope(
        stage=EXECUTION,
        allows=(
            "разбирать развилки реализации",
            "подтверждать решения по реализации",
        ),
        reaches=(ACCEPTANCE,),
    ),
    VALIDATION: Scope(
        stage=VALIDATION,
        allows=(
            "искать пробелы и противоречия в ТЗ",
            "закрывать открытые вопросы",
        ),
        reaches=(ACCEPTANCE,),
    ),
    DONE: Scope(
        stage=DONE,
        allows=("отвечать по собранному ТЗ", "принимать новую задачу"),
    ),
}


@dataclass(frozen=True)
class Jump:
    """Доказанный забег вперёд: работа, паттерн и фрагмент текста.

    Устроен как invariants.Hit и по той же причине: цитата берётся куском вокруг
    совпадения, уходит в журнал как улика и во второй проход как то, что переписывают.
    """

    reach: Reach
    banned: str
    quote: str

    @property
    def reason(self) -> str:
        return f"{self.reach.summary} — «{self.quote}»"


def scope(stage: str) -> Scope | None:
    return SCOPES.get(stage)


def reaches(task: Task, tally: dict[str, int], approved: frozenset[str]) -> tuple[Reach, ...]:
    """Работа впереди, которую держит закрытый гейт.

    Гейт открыт — работа больше не забег: план утверждён, значит проектировать
    реализацию можно, даже если автомат ещё стоит на планировании. Иначе область
    превратилась бы в запрет ради запрета, а гейт — в формальность.
    """
    found = scope(task.at.key)
    if found is None:
        return ()

    live = []
    for reach in found.reaches:
        gate = gates.by_key(reach.gate)
        if gate is not None and gate.blocked(task, tally, approved):
            live.append(reach)
    return tuple(live)


def scope_lines(task: Task, tally: dict[str, int], approved: frozenset[str]) -> list[str]:
    """Область этапа словами: строки для блока гейтов."""
    found = scope(task.at.key)
    if found is None:
        return []

    rows = [f"этап отпускает: {'; '.join(found.allows)}"]
    for reach in reaches(task, tally, approved):
        gate = gates.by_key(reach.gate)
        name = f"гейта «{gate.name}»" if gate else "гейта"
        rows.append(f"до {name} закрыто: {reach.work}. Вместо этого: {reach.instead}")
    return rows


def mentions(live: tuple[Reach, ...], text: str) -> bool:
    """Есть ли в тексте работа за гейтом — без разбора, забег это или отсрочка.

    Дешёвая половина детектора и единственная, которую можно спросить о недописанном
    предложении: агент по ней решает, отдавать кусок ответа сразу или задержать его до
    точки. Устроено как invariants.mentions и нужно для того же.
    """
    if not text.strip():
        return False

    body = text.lower()
    for reach in live:
        for banned in reach.banned:
            try:
                if re.search(banned, body):
                    return True
            except re.error:
                continue
    return False


def scan(live: tuple[Reach, ...], text: str, *, partial: bool = False) -> tuple[Jump, ...]:
    """Забеги вперёд в ответе агента: работа за гейтом, названная без отсрочки.

    Глагола предложения, в отличие от invariants.scan, не требуется: область этапа
    нарушают не предложением, а исполнением. Зато снятий больше — к отрицанию
    добавлена отсрочка, и без неё обрывался бы любой ответ, который честно объясняет,
    какой гейт держит путь дальше.
    """
    body = invariants.settled(text) if partial else text
    if not body.strip():
        return ()

    lowered = body.lower()
    found: list[Jump] = []
    for reach in live:
        hit = _first(reach, body, lowered)
        # По работе довольно одного забега: отвечать на него будут один раз.
        if hit is not None:
            found.append(hit)
    return tuple(found)


def _first(reach: Reach, text: str, body: str) -> Jump | None:
    for banned in reach.banned:
        try:
            pattern = re.compile(banned)
        except re.error:
            # Сломанный паттерн — опечатка в области, а не разрешение её нарушать:
            # работа остаётся названной в блоке запроса, детектора у неё просто нет.
            continue

        for match in pattern.finditer(body):
            start = max(0, match.start() - invariants.WINDOW)
            window = body[start : match.end() + invariants.WINDOW]
            # Отрицание ищется в обороте, отсрочка — в окне, и мерки разные не для
            # симметрии с day14, а по смыслу. Отрицание привязано к одному слову: «схему
            # таблиц не рисуем» — здесь и только здесь. Отсрочка относится к фразе
            # целиком и обычно стоит в соседнем обороте: «чтобы ТЗ отдавать в работу,
            # нужно ваше утверждение» — отсрочка во второй половине, работа в первой.
            if _NEGATION.search(invariants.clause(body, match.start(), match.end())):
                continue
            if _DEFER.search(window):
                continue
            return Jump(
                reach=reach,
                banned=banned,
                quote=clean(text[start : match.end() + invariants.WINDOW]),
            )
    return None


PROMPT = """\
Ты проверяешь, не сделал ли ассистент работу этапа, до которого задача ещё не дошла.

Задача идёт по этапам, между этапами стоят гейты. Гейт подтверждения проходит только
пользователь, и до его решения работа следующего этапа закрыта. На входе — текущий
этап, что он отпускает, какая работа закрыта и каким гейтом, реплика пользователя и
ответ ассистента.

Забег вперёд — это когда ассистент делает закрытую работу: выдаёт её результат,
описывает её по существу или объявляет решение, которое принимает пользователь. Не
забег:
- назвать гейт и сказать, что до него эта работа закрыта;
- пообещать сделать её после утверждения;
- попросить у пользователя утверждение;
- пересказать просьбу пользователя без её выполнения;
- уточняющий вопрос по текущему этапу, даже если он задан о будущей работе.

Правила:
- Ссылайся только на этапы из списка закрытой работы. Своих не придумывай.
- quote — дословный фрагмент ответа, в котором видна сделанная работа. Не
  пересказывай его и не сочиняй: цитата без опоры в тексте будет отброшена.
- Сомневаешься — забега нет. Ложный забег дороже пропущенного: по нему пользователю
  нечего исправлять.
- Один забег на этап, даже если он повторяется в ответе несколько раз.

Поле why — одна короткая строка, какая работа сделана раньше времени: её увидит
пользователь в карточке.

Верни строго JSON без markdown и пояснений:
{"jumps": [{"stage": "execution", "quote": "...", "why": "..."}]}
Ответ в области своего этапа — верни {"jumps": []}.
"""

# Второй проход. Называется всё то же, что в day14: чья это работа, какой гейт её
# держит, чем заняться вместо и на каком куске ответ оборвался. Без цитаты модель
# переписывает наугад и обрывается снова, без гейта — извиняется, не понимая за что.
RETRY = """\
Предыдущий ответ был отброшен: он делал работу этапа, до которого задача не дошла, и
пользователь его не увидел.

{report}

Ответь на тот же вопрос заново, оставаясь в области текущего этапа. Прямо скажи, какой
гейт держит эту работу и кто его проходит, и займись тем, что этап отпускает. Не
пересказывай эту инструкцию, не упоминай отброшенный ответ и не извиняйся.
"""


@dataclass(frozen=True)
class Verdict:
    """Что назвал аудитор. Ещё не забег: этап, цитата и причина, больше ничего."""

    stage: str
    quote: str
    why: str = ""


def state(task: Task, live: tuple[Reach, ...]) -> str:
    """Этап и его область словами: то же, что уходит в блок запроса."""
    found = scope(task.at.key)
    rows = [
        f"этап: {task.at.name} — {task.at.about}",
        f"шаг: {task.step or '—'}",
        f"этап отпускает: {'; '.join(found.allows) if found else '—'}",
        "",
        "Закрытая работа:",
    ]
    if not live:
        rows.append("- нет: все гейты впереди открыты")
    for reach in live:
        gate = gates.by_key(reach.gate)
        rows.append(f"- {reach.stage} ({reach.name}): {reach.work} — держит {gate.label if gate else 'гейт'}")
    return "\n".join(rows)


def request(
    task: Task,
    live: tuple[Reach, ...],
    prompt: str,
    answer: str,
) -> list[ChatCompletionMessageParam]:
    """Ответ уходит на проверку расшифровкой, а не диалогом — как и аудит инвариантов.

    Реплика пользователя едет вместе с ответом по той же причине, что в audit.py: без
    неё «вот схема таблиц» и «схему таблиц нельзя, пока план не утверждён» неотличимы,
    а различать их обязательно.
    """
    return [
        ChatCompletionSystemMessageParam(role="system", content=PROMPT),
        ChatCompletionUserMessageParam(
            role="user",
            content=(
                f"Состояние задачи:\n{state(task, live)}\n\n"
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
    """Разбор ответа аудитора: список забегов, всё остальное — ошибка.

    Пустой список и «забегов нет» — одно и то же и ошибкой не считаются: ход, на
    котором агент остался в своём этапе, — обычный ход, а не сбой разбора.
    """
    body = clean_json(text)
    try:
        parsed = json.loads(body)
    except ValueError as error:
        raise ValueError(f"ответ не разобран как JSON: {clean(text)[:80]}") from error

    # Модель иногда отдаёт голый список вместо объекта с полем jumps: форма не та,
    # содержимое то самое.
    raw = parsed if isinstance(parsed, list) else parsed.get("jumps") if isinstance(parsed, dict) else None
    if not isinstance(raw, (list, tuple)):
        raise ValueError("ожидался объект с полем jumps")

    verdicts = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        verdicts.append(
            Verdict(
                stage=clean(entry.get("stage", "")).lower(),
                quote=clean(entry.get("quote", "")),
                why=clean(entry.get("why", "")),
            ),
        )
    return tuple(verdicts)


def confirm(live: tuple[Reach, ...], verdicts: tuple[Verdict, ...]) -> tuple[list[Jump], list[str]]:
    """Вердикты — в забеги и в отказы.

    Проверяется то же, что у аудитора инвариантов, и в том же смысле: по забегу должно
    быть понятно, чья это работа и чем она подтверждается. Названного этапа нет среди
    закрытых — забег отбрасывается целиком: работа открытого этапа не забег, а работа.
    """
    known = {reach.stage: reach for reach in live}
    found: list[Jump] = []
    rejected: list[str] = []
    seen: set[str] = set()

    for verdict in verdicts:
        reach = known.get(verdict.stage)
        if reach is None:
            rejected.append(f"этапа «{verdict.stage}» нет среди закрытых")
            continue
        if not verdict.quote:
            rejected.append(f"забег в этап «{verdict.stage}» без цитаты")
            continue
        if verdict.stage in seen:
            continue
        seen.add(verdict.stage)
        found.append(Jump(reach=reach, banned="", quote=verdict.quote))
    return found, rejected


def report(jumps: tuple[Jump, ...] | list[Jump]) -> str:
    """Забеги словами: то же, что уходит во второй проход и в строку отказа."""
    rows = []
    for jump in jumps:
        rows.append(f"Сделана {jump.reach.summary}")
        rows.append(f"Вместо этого: {jump.reach.instead}")
        if jump.quote:
            rows.append(f"Оборванный ответ дошёл до: «{jump.quote}»")
    return "\n".join(rows)


def refusal(task: Task, jumps: tuple[Jump, ...] | list[Jump]) -> str:
    """Отказ, написанный кодом: последнее слово, когда модель дважды не справилась.

    Такой ответ хуже любого, который написала бы модель, и он ровно то, чем этот день
    заканчивается по определению: ассистент не сделает работу следующего этапа, даже
    если больше ему сделать нечего. Выход в нём назван всегда — гейт и тот, кто его
    проходит.
    """
    rows = [f"Это работа следующего этапа, а задача на этапе «{task.at.name}»."]
    for jump in jumps:
        rows.append(f"- {jump.reach.summary}")
        rows.append(f"  Вместо этого: {jump.reach.instead}")
    rows.append("Гейт подтверждения проходите вы — кнопкой в панели состояния, не я.")
    return "\n".join(rows)


def retry_param(jumps: tuple[Jump, ...] | list[Jump]) -> ChatCompletionMessageParam:
    """Системное сообщение второго прохода: этап назван, гейт назван, выход назван."""
    return ChatCompletionSystemMessageParam(
        role="system",
        content=RETRY.format(report=report(jumps)),
    )


def defaults() -> list[dict[str, object]]:
    """Области этапов для /api/defaults: описание процесса, а не состояния задачи."""
    return [
        {
            "stage": stage,
            "name": BY_KEY[stage].name,
            "allows": list(found.allows),
            "closed": [
                {
                    "stage": reach.stage,
                    "name": reach.name,
                    "work": reach.work,
                    "gate": reach.gate,
                    "instead": reach.instead,
                    "banned": list(reach.banned),
                }
                for reach in found.reaches
            ],
        }
        for stage, found in SCOPES.items()
    ]
