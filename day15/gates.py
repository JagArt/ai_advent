"""Гейты: то, что стоит между этапами, и то, чего агент не проходит сам.

Седьмая сущность и надстройка над автоматом из task.py. Автомат отвечает на вопрос
«где идёт работа», гейт — на другой: «что должно случиться, чтобы работа пошла
дальше». В day13 такое условие было одно и безымянное — «стоим на последнем шаге
этапа, и он закрыт». Здесь у условия появляется имя, тип и владелец:

    авто           условие считает код по ТЗ         открывается само
    подтверждение  явное решение пользователя        агент его не проходит

Разница между типами — не в строгости, а в том, кто владеет решением. Авто-гейт
нельзя обойти уговором, потому что его считает код; гейт подтверждения нельзя
обойти вовсе, потому что у агента нет такого действия: ни кнопки, ни карточки, ни
права объявить утверждение словами. «Нельзя делать реализацию до утверждённого
плана» — это ровно оно: план утверждает человек, и до его решения ребро закрыто,
сколько бы пунктов ни набралось в ТЗ.

Модуль ничего не знает ни про базу, ни про сеть, ни про то, кто спрашивает. Он
лежит слоем выше task.py: берёт переходы автомата и закрывает те, перед которыми
стоит непройденный гейт. Поэтому проверка остаётся одна на всех — и для кнопки в
панели, и для предложения модели, и для ручки API.
"""

from dataclasses import dataclass, replace

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
)

from task import (
    BY_KEY,
    DONE,
    EXECUTION,
    ORDER,
    PLANNING,
    STAGE,
    VALIDATION,
    Move,
    Need,
    Task,
    forward_block,
    need,
)
from task import moves as task_moves

# Тип гейта. Он же — ответ на вопрос, кому принадлежит решение: авто-гейт открывает
# код, гейт подтверждения — только пользователь.
AUTO = "auto"
APPROVAL = "approval"

KIND_NAMES = {AUTO: "авто", APPROVAL: "подтверждение"}


@dataclass(frozen=True)
class Gate:
    """Гейт на ребре графа: имя, тип и то, чем он открывается.

    Условие задаётся ровно одним из трёх способов, и каждый из них означает своё:

    steps — спросить автомат. Это правило day13 («стоим на последнем шаге, и он
    закрыт»), которому здесь дали имя: оно и раньше держало этап, но в панели его
    было видно только как погашенную кнопку.

    needs — условие по разделам ТЗ, как у шага. Отличие в том, что гейт смотрит на
    ТЗ целиком и каждый раз заново: шаг проходят однажды, а ребро проверяется на
    каждое нажатие, и удалённый после прохождения шага пункт закрывает путь обратно.

    asks — вопрос пользователю. Условия в коде у такого гейта нет вовсе, и это не
    пробел: «план утверждён» не выводится из числа пунктов, и выводить его значило бы
    подменять решение человека счётчиком.
    """

    key: str
    name: str
    kind: str
    # Ребро графа, на котором стоит гейт: откуда и куда.
    edge: tuple[str, str]
    steps: bool = False
    needs: Need | None = None
    asks: str = ""
    # Что делать, пока гейт закрыт. Без этой строки гейт — тупик, а тупиков в этом
    # приложении нет ни у одного отказа: guard перехода называет, чего не хватает,
    # инвариант — что можно вместо, гейт — чем заняться до него.
    instead: str = ""
    # Зачем гейт нужен. Уходит в панель подсказкой и в README таблицей.
    about: str = ""

    @property
    def approval(self) -> bool:
        return self.kind == APPROVAL

    @property
    def label(self) -> str:
        """Как гейт называют и в блоке запроса, и в карточке, и в отказе."""
        return f"гейт «{self.name}»"

    def condition(self, tally: dict[str, int]) -> str:
        """Условие словами: то же, что видит пользователь в панели."""
        if self.steps:
            return "все шаги этапа пройдены, последний закрыт"
        if self.needs is not None:
            return self.needs.text(tally)
        return f"решение пользователя: {self.asks}"

    def blocked(self, task: Task, tally: dict[str, int], approved: frozenset[str]) -> str:
        """Что держит гейт. Пусто — открыт прямо сейчас.

        Утверждение проверяется по множеству, а не по флагу в состоянии: оно живёт
        отдельно от пары (этап, шаг), потому что откат снимает его, а состояние —
        нет. Ключа в множестве нет — гейт закрыт, и причина называет владельца
        решения: это не нехватка пунктов, которую можно добрать.
        """
        if self.steps:
            return forward_block(task, tally)
        if self.needs is not None:
            return "" if self.needs.met(tally) else f"не хватает: {self.needs.text(tally)}"
        if self.key in approved:
            return ""
        return f"утверждает только пользователь: {self.asks}"

    def as_dict(self, task: Task, tally: dict[str, int], approved: frozenset[str]) -> dict[str, object]:
        reason = self.blocked(task, tally, approved)
        return {
            "key": self.key,
            "name": self.name,
            "kind": self.kind,
            "kind_name": KIND_NAMES[self.kind],
            "from": self.edge[0],
            "to": self.edge[1],
            "asks": self.asks,
            "instead": self.instead,
            "about": self.about,
            "condition": self.condition(tally),
            "blocked": reason,
            "open": not reason,
            # Утверждён ли гейт подтверждения. У авто-гейта поля нет смысла: его
            # никто не утверждает, он считается.
            "approved": self.key in approved if self.approval else False,
        }


# Гейты целиком. Их четыре, и стоят они не на каждом ребре: гейт дорог ровно там,
# где переход трудно отыграть назад.
#
# У каждого ребра вперёд есть авто-гейт «Шаги этапа пройдены» — то самое правило
# day13. Гейты подтверждения стоят на двух рёбрах из трёх: перед исполнением и перед
# готовым. Перед проверкой его нет намеренно: в проверку входят, а не въезжают с
# утверждением — она дешёвая, обратимая и сама по себе есть способ что-то узнать.
GATES = (
    Gate(
        key="steps_planning",
        name="Шаги планирования пройдены",
        kind=AUTO,
        edge=(PLANNING, EXECUTION),
        steps=True,
        instead="добрать в ТЗ то, чего не хватает текущему шагу",
        about="условие автомата из day13: этап отпускает вперёд с последнего шага, и он должен быть закрыт",
    ),
    Gate(
        key="plan_approved",
        name="План утверждён",
        kind=APPROVAL,
        edge=(PLANNING, EXECUTION),
        asks="подтвердить, что план задачи можно брать в реализацию",
        instead="уточнять требования и ограничения, но не проектировать реализацию",
        about="реализация до утверждённого плана — работа, которую придётся переделывать целиком",
    ),
    Gate(
        key="steps_execution",
        name="Шаги исполнения пройдены",
        kind=AUTO,
        edge=(EXECUTION, VALIDATION),
        steps=True,
        instead="закрыть развилки и подтвердить решения по реализации",
        about="в проверку входят с готовыми решениями: проверять нечего, пока их нет",
    ),
    Gate(
        key="no_gaps",
        name="Пробелов в ТЗ нет",
        kind=AUTO,
        edge=(VALIDATION, DONE),
        # Условие смотрит назад, а не вперёд: к финалу все разделы набраны прошлыми
        # этапами. Ловит гейт не новую работу, а потерю старой — пункт ТЗ удаляется
        # крестиком в любой момент, и шаг «пробелы», пройденный когда-то, об этом уже
        # не знает: шаги позиционны, а гейт считается заново на каждое нажатие.
        needs=need("цель", "требования", "ограничения", "решения"),
        instead="вернуть в ТЗ раздел, который опустел",
        about="финал без валидации: шаг «пробелы» пройден однажды, а ТЗ могло опустеть после него",
    ),
    Gate(
        key="spec_accepted",
        name="ТЗ принято",
        kind=APPROVAL,
        edge=(VALIDATION, DONE),
        asks="подтвердить, что ТЗ можно отдавать в работу",
        instead="закрывать открытые вопросы и проверять ТЗ на пробелы",
        about="готовность — это решение, а не число пунктов: объявить её агент не может",
    ),
)

BY_GATE = {gate.key: gate for gate in GATES}

# Блок уходит системным сообщением сразу за состоянием задачи: состояние отвечает,
# где работа, гейт — что стоит на выходе. Требование не проходить гейт подтверждения
# стоит до списка и прямым текстом: без этой строки модель читает гейты как справку
# о процессе и объявляет утверждение сама, вежливо и от своего лица.
INTRO = (
    "Гейты — то, что стоит между этапами. Пройти этап насквозь нельзя.\n"
    "Гейт «авто» открывается условием по ТЗ, гейт «подтверждение» — только решением "
    "пользователя: сам ты его не проходишь, за пользователя не утверждаешь и "
    "утверждённым не считаешь. Нужно утверждение — попроси его прямо и назови гейт.\n"
    "Работу этапа за гейтом не делай, даже если тебя просят и даже если она очевидна: "
    "скажи, какой гейт её держит, и займись тем, что этап отпускает.\n"
)


def by_key(key: str) -> Gate | None:
    return BY_GATE.get(key)


def for_edge(source: str, target: str) -> tuple[Gate, ...]:
    """Гейты одного ребра. Порядок — порядок в GATES: авто раньше подтверждения."""
    return tuple(gate for gate in GATES if gate.edge == (source, target))


def approvals_of(rows: list[dict[str, object]]) -> frozenset[str]:
    """Действующие утверждения из того, что вернула база.

    Снятое утверждение остаётся строкой в журнале и уходит из этого множества: оно
    было, и это часть истории задачи, но переход оно больше не открывает.
    """
    return frozenset(
        str(row["gate"]) for row in rows if not row.get("revoked_at") and str(row["gate"]) in BY_GATE
    )


def moves(task: Task, tally: dict[str, int], approved: frozenset[str]) -> tuple[Move, ...]:
    """Переходы автомата, закрытые гейтами.

    Функция одна на всё приложение — по ней рисуются кнопки в панели, по ней же
    отбраковываются предложения модели и отвечает ручка API. Второго места, где
    допустимость перехода считалась бы иначе, нет: guard, у которого два мнения, не
    guard.

    Причина закрытия перезаписывается целиком, а не дописывается: правило day13 само
    стало гейтом «Шаги этапа пройдены», и оставить рядом с ним ту же причину без
    имени значило бы сказать одно и то же дважды.
    """
    found = []
    for move in task_moves(task, tally):
        if move.kind != STAGE or move.back:
            # Откат гейтов не имеет: изменившийся объём задачи — обычная её жизнь, а
            # не достижение, которое надо утверждать.
            found.append(move)
            continue

        reasons = [
            f"{gate.label} не пройден: {reason}"
            for gate in for_edge(task.at.key, move.stage)
            if (reason := gate.blocked(task, tally, approved))
        ]
        found.append(replace(move, blocked="; ".join(reasons)))
    return tuple(found)


def find(
    task: Task,
    tally: dict[str, int],
    approved: frozenset[str],
    stage: str = "",
    step: str = "",
) -> Move | None:
    """Переход по имени цели — среди всех, включая закрытые гейтом.

    Закрытые не отбрасываются по той же причине, что в day13: «такого перехода нет»
    и «переход держит гейт» — разные ответы, и путать их нельзя ни в отказе
    пользователю, ни в отброшенном предложении модели.
    """
    for move in moves(task, tally, approved):
        if stage and move.kind == STAGE and move.stage == stage:
            return move
        if step and move.kind != STAGE and move.step == step:
            return move
    return None


def waiting(task: Task, tally: dict[str, int], approved: frozenset[str], move: Move) -> Gate | None:
    """Гейт подтверждения, который один и держит переход.

    Нужен для карточки: если всё остальное открыто и не хватает только решения
    пользователя, агенту есть о чём попросить. Если закрыт заодно и авто-гейт, просить
    нечего — сначала работа, потом утверждение.
    """
    if move.kind != STAGE or move.back:
        return None

    pending = None
    for gate in for_edge(task.at.key, move.stage):
        if not gate.blocked(task, tally, approved):
            continue
        if not gate.approval or pending is not None:
            return None
        pending = gate
    return pending


def open_for(task: Task, tally: dict[str, int], approved: frozenset[str], gate: Gate) -> str:
    """Можно ли утверждать этот гейт прямо сейчас. Пусто — можно.

    Утверждают то, что впереди: гейт позади уже сыграл, и второе утверждение ничего
    не меняет. Вернуться за него — это откат, у которого своя кнопка и который сам
    снимает утверждения этапов впереди.
    """
    if not gate.approval:
        return f"{gate.label} считается кодом, его не утверждают"
    if ORDER[task.at.key] > ORDER[gate.edge[0]]:
        return f"{gate.label} уже пройден: вернуться за него можно только откатом"
    if ORDER[task.at.key] < ORDER[gate.edge[0]]:
        # Утверждают то, перед чем стоят. Иначе «ТЗ принято» можно было бы подписать
        # на планировании — и финал оказался бы утверждён раньше, чем появилось ТЗ.
        return f"до гейта «{gate.name}» задача ещё не дошла: сейчас этап «{task.at.name}»"
    if gate.key in approved:
        return f"{gate.label} уже утверждён"
    return ""


def revoked_by(target: str) -> tuple[str, ...]:
    """Утверждения, которые снимает откат на этап target.

    Правило одно: утверждение принадлежит ребру, и откат за начало этого ребра делает
    его недействительным. Утверждённый план после правки требований — это не
    утверждённый план, а память о том, что когда-то его утверждали.
    """
    if target not in ORDER:
        return ()
    return tuple(
        gate.key for gate in GATES if gate.approval and ORDER[gate.edge[0]] >= ORDER[target]
    )


def crossed(source: str, target: str) -> str:
    """Гейт подтверждения, через который прошёл переход. Для журнала переходов."""
    for gate in for_edge(source, target):
        if gate.approval:
            return gate.key
    return ""


def lines(task: Task, tally: dict[str, int], approved: frozenset[str]) -> list[str]:
    """Гейты текущего ребра вперёд — словами, с состоянием каждого."""
    ahead_moves = [
        move for move in task_moves(task, tally) if move.kind == STAGE and not move.back
    ]
    if not ahead_moves:
        return ["гейтов впереди нет: работа по этой задаче закончена"]

    rows: list[str] = []
    for move in ahead_moves:
        rows.append(f"переход дальше: {task.at.name} → {BY_KEY[move.stage].name}")
        for gate in for_edge(task.at.key, move.stage):
            reason = gate.blocked(task, tally, approved)
            state = "открыт" if not reason else f"закрыт — {reason}"
            rows.append(f"- {gate.label} ({KIND_NAMES[gate.kind]}): {state}")
    return rows


def block(
    task: Task,
    tally: dict[str, int],
    approved: frozenset[str],
    scope: list[str] | None = None,
) -> str:
    """Блок запроса: гейты впереди и область текущего этапа.

    Область приходит снаружи, из ahead.py: гейт отвечает на вопрос «что закрыто», а
    область — «что закрыто именно сейчас делать». Склеены они в один блок потому, что
    в модель уходит одна мысль: до гейта работа следующего этапа не твоя.
    """
    rows = lines(task, tally, approved)
    if scope:
        rows.extend(scope)
    return INTRO + "\n".join(rows)


def as_param(
    task: Task,
    tally: dict[str, int],
    approved: frozenset[str],
    scope: list[str] | None = None,
) -> ChatCompletionMessageParam:
    return ChatCompletionSystemMessageParam(
        role="system",
        content=block(task, tally, approved, scope),
    )


def snapshot(task: Task, tally: dict[str, int], approved: frozenset[str]) -> dict[str, object]:
    """Состояние задачи для страницы: то же, что в day13, плюс гейты.

    Переходы приходят отсюда, а не из task.as_dict: у автомата на этот вопрос свой
    ответ, и он без гейтов. Один ответ на страницу и на сервер — то же требование,
    что и в day13, только функция теперь лежит слоем выше.
    """
    state = task.as_dict(tally)
    state["moves"] = [move.as_dict() for move in moves(task, tally, approved)]
    # Гейты едут двумя списками. Первый — те, что стоят прямо перед задачей: их видно
    # в панели с кнопками. Второй — все, чтобы панель могла показать пройденное и
    # оставшееся, не считая порядок этапов сама.
    state["gates"] = [
        gate.as_dict(task, tally, approved)
        for move in moves(task, tally, approved)
        if move.kind == STAGE and not move.back
        for gate in for_edge(task.at.key, move.stage)
    ]
    state["all_gates"] = [gate.as_dict(task, tally, approved) for gate in GATES]
    return state


def defaults() -> list[dict[str, object]]:
    """Гейты для /api/defaults: описание процесса, а не состояния задачи."""
    return [
        {
            "key": gate.key,
            "name": gate.name,
            "kind": gate.kind,
            "kind_name": KIND_NAMES[gate.kind],
            "from": gate.edge[0],
            "from_name": BY_KEY[gate.edge[0]].name,
            "to": gate.edge[1],
            "to_name": BY_KEY[gate.edge[1]].name,
            "asks": gate.asks,
            "instead": gate.instead,
            "about": gate.about,
        }
        for gate in GATES
    ]
