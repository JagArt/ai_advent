"""Состояние задачи как конечный автомат: этап, шаг и ожидаемое действие.

Пятый блок запроса и третья сущность рядом с памятью и профилем. Отвечает она на
третий вопрос: память знает, о чём договорились, профиль — кому и как это сказать,
а состояние — где сейчас идёт работа и чего агент ждёт от пользователя.

    этап    planning → execution → validation → done   состояние автомата
    шаг     позиция внутри этапа                       по шагу на ожидаемое действие
    действие что должен сделать пользователь           берётся у текущего шага

Автомат формальный, а не описательный: переходы перечислены в EDGES, и того, чего
в графе нет, не случится ни по просьбе модели, ни по кнопке. Условие перехода тоже
не на совести модели — оно выражено через рабочую память: шаг закрыт, когда его
раздел ТЗ набрал нужное число пунктов. Поэтому «мы всё обсудили» ничего не двигает,
а подтверждённое требование двигает.

Модуль ничего не знает ни про базу, ни про сеть, ни про то, кто предложил переход:
он хранит пару (этап, шаг), считает допустимые переходы и собирает системный блок.
Предлагает переходы progress.py, применяет агент, решает пользователь.
"""

from dataclasses import dataclass

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
)

from memory import Store

PLANNING = "planning"
EXECUTION = "execution"
VALIDATION = "validation"
DONE = "done"

# Вид перехода: внутри этапа или между этапами. Различие не косметическое — у них
# разные условия и разная цена ошибки, и в карточке они выглядят по-разному.
STAGE = "stage"
STEP = "step"


def _items(count: int) -> str:
    """Условие читает и человек, и модель: «не меньше 1 пункта», «не меньше 2 пунктов»."""
    return "пункта" if count % 10 == 1 and count % 100 != 11 else "пунктов"


@dataclass(frozen=True)
class Need:
    """Условие шага, выраженное через рабочую память.

    Условие — единственное, что отделяет автомат от подписи под ответом. Считается
    оно по разделам ТЗ, а ТЗ наполняется только подтверждёнными пунктами, поэтому
    закрыть шаг разговором нельзя: нужно, чтобы пользователь что-то принял.

    Разделов может быть несколько, и тогда условие требуется от каждого. Так устроен
    шаг «пробелы»: там проверяется не новая работа, а то, что за время задачи из ТЗ
    ничего не пропало.
    """

    sections: tuple[str, ...]
    at_least: int = 1

    def missing(self, counts: dict[str, int]) -> list[str]:
        return [name for name in self.sections if counts.get(name, 0) < self.at_least]

    def met(self, counts: dict[str, int]) -> bool:
        return not self.missing(counts)

    def text(self, counts: dict[str, int]) -> str:
        amount = f"не меньше {self.at_least} {_items(self.at_least)}"
        if len(self.sections) == 1:
            name = self.sections[0]
            return f"в разделе «{name}» {amount} (сейчас {counts.get(name, 0)})"

        listed = ", ".join(self.sections)
        gaps = self.missing(counts)
        tail = f" (пусто: {', '.join(gaps)})" if gaps else " (все набраны)"
        return f"в каждом из разделов {listed} — {amount}{tail}"


def need(*sections: str, at_least: int = 1) -> Need:
    """Условие шага: раздел ТЗ или несколько сразу."""
    return Need(sections=sections, at_least=at_least)


@dataclass(frozen=True)
class Step:
    """Шаг этапа: одно ожидаемое действие и одно условие, чтобы пройти дальше.

    needs = None — шаг, который из памяти не выводится: его закрывает согласие
    пользователя, а не пункт ТЗ. Такой шаг стоит последним в этапе, и стоять на
    нём и есть подтверждение.
    """

    key: str
    expects: str
    needs: Need | None = None

    def met(self, counts: dict[str, int]) -> bool:
        return self.needs is None or self.needs.met(counts)

    def condition(self, counts: dict[str, int]) -> str:
        if self.needs is None:
            return "подтверждения пользователя — из памяти этот шаг не выводится"
        return self.needs.text(counts)


@dataclass(frozen=True)
class Stage:
    """Этап — состояние автомата. Шаги внутри него проходятся по порядку."""

    key: str
    name: str
    about: str
    steps: tuple[Step, ...] = ()

    def step(self, key: str) -> Step | None:
        """Шаг по имени или первый: пустого шага у этапа с шагами не бывает."""
        for step in self.steps:
            if step.key == key:
                return step
        return self.steps[0] if self.steps else None

    def number(self, key: str) -> int:
        step = self.step(key)
        return self.steps.index(step) + 1 if step else 0


# Задача этого приложения — довести разговор до технического задания, поэтому этапы
# описывают жизнь ТЗ, а не абстрактную работу. Разделы в условиях — те же, что у
# рабочей памяти в memory.SECTIONS: автомат считает по тому самому ТЗ, которое
# собирается карточками, и другого источника правды у него нет.
STAGES = (
    Stage(
        key=PLANNING,
        name="планирование",
        about="о чём задача и что от неё требуется",
        steps=(
            Step("цель", "назвать, что за сервис и зачем он нужен", need("цель")),
            Step("требования", "перечислить, что сервис должен делать", need("требования", at_least=2)),
            Step("ограничения", "назвать сроки, объёмы и запреты", need("ограничения")),
        ),
    ),
    Stage(
        key=EXECUTION,
        name="исполнение",
        about="как это будет сделано",
        steps=(
            Step("развилки", "выбрать вариант на каждой развилке", need("решения")),
            Step("детали", "подтвердить решения по реализации", need("решения", at_least=3)),
        ),
    ),
    Stage(
        key=VALIDATION,
        name="проверка",
        about="что ТЗ полное и непротиворечивое",
        steps=(
            # Условие пробелов смотрит назад, а не вперёд: к проверке все четыре
            # раздела уже набраны прошлыми этапами, и шаг ловит не новую работу, а
            # потерю старой — пункт ТЗ можно удалить крестиком в любой момент, и
            # тогда путь к готовому закрывается обратно.
            Step(
                "пробелы",
                "убедиться, что в ТЗ не осталось пустых разделов",
                need("цель", "требования", "ограничения", "решения"),
            ),
            # Условия у приёмки нет намеренно: «ТЗ можно отдавать в работу» — это
            # решение, а не число пунктов, и выводить его из памяти было бы подлогом.
            Step("приёмка", "подтвердить, что ТЗ можно отдавать в работу"),
        ),
    ),
    # У готового этапа шагов нет: ждать от пользователя больше нечего.
    Stage(key=DONE, name="готово", about="ТЗ собрано, задача закрыта"),
)

BY_KEY = {stage.key: stage for stage in STAGES}
ORDER = {stage.key: index for index, stage in enumerate(STAGES)}

# Граф переходов целиком. Вперёд — по одному ребру, назад — тоже: и откат из
# проверки в исполнение (нашли пробел), и откат из исполнения в планирование
# (изменился объём) — обычная жизнь задачи, а не поломка. Готово не тупик: из него
# можно вернуться в проверку, если ТЗ пришлось переоткрыть.
EDGES = {
    PLANNING: (EXECUTION,),
    EXECUTION: (VALIDATION, PLANNING),
    VALIDATION: (DONE, EXECUTION),
    DONE: (VALIDATION,),
}

# Начальное состояние. Новая задача открывается на первом шаге планирования: чего
# от пользователя ждут, известно ещё до его первой реплики.
START_STAGE = STAGES[0].key
START_STEP = STAGES[0].steps[0].key

# Блок уходит системным сообщением и стоит последним из блоков, прямо перед
# репликами: профиль задаёт форму всего, память — что известно, а состояние — где
# работа сейчас, и это единственный блок, который меняется каждый ход.
INTRO = (
    "Состояние задачи — где идёт работа и чего ты ждёшь от пользователя.\n"
    "Этап и шаг двигаются переходами, а не твоим решением по ходу ответа: "
    "работай на текущем шаге и добивайся ожидаемого действия.\n"
)

DONE_EXPECTED = "начать новую задачу или вернуться к правкам"


@dataclass(frozen=True)
class Move:
    """Переход: куда двинемся и что этому мешает.

    Ход всегда описан целиком парой (этап, шаг): у перехода между этапами шаг —
    первый шаг нового этапа, у перехода внутри этапа этап не меняется. Пустое
    blocked значит «можно прямо сейчас», непустое — готовая причина для интерфейса
    и для отказа, одна и та же и агенту, и пользователю.
    """

    kind: str
    stage: str
    step: str
    name: str
    blocked: str = ""
    back: bool = False

    @property
    def allowed(self) -> bool:
        return not self.blocked

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "stage": self.stage,
            "step": self.step,
            "name": self.name,
            "blocked": self.blocked,
            "back": self.back,
        }


@dataclass(frozen=True)
class Task:
    """Состояние задачи: этап и шаг внутри него.

    Пара нормализуется на чтении, а не на записи: у этапа из будущей версии кода
    может не оказаться шага, который лежит в базе, и в этом случае состояние должно
    выродиться в первый шаг этапа, а не в исключение посреди хода.
    """

    stage: str = START_STAGE
    step: str = START_STEP

    @property
    def at(self) -> Stage:
        return BY_KEY.get(self.stage) or STAGES[0]

    @property
    def on(self) -> Step | None:
        return self.at.step(self.step)

    @property
    def stage_number(self) -> int:
        return ORDER[self.at.key] + 1

    @property
    def step_number(self) -> int:
        return self.at.number(self.step)

    @property
    def expected(self) -> str:
        step = self.on
        return step.expects if step else DONE_EXPECTED

    @property
    def last(self) -> bool:
        """Стоим на последнем шаге этапа: только отсюда этап отпускает вперёд."""
        return not self.at.steps or self.step_number == len(self.at.steps)

    @property
    def line(self) -> str:
        step = self.on
        return f"{self.at.name} · {step.key}" if step else self.at.name

    def lines(self, counts: dict[str, int]) -> list[str]:
        stage = self.at
        lines = [
            f"этап: {stage.name} ({self.stage_number} из {len(STAGES)}) — {stage.about}",
        ]

        step = self.on
        if step is None:
            lines.append("шага нет: работа по этой задаче закончена")
        else:
            lines.append(f"шаг: {step.key} ({self.step_number} из {len(stage.steps)})")
        lines.append(f"ожидаемое действие: {self.expected}")

        # Условие перехода уходит в модель словами, а не только числом: иначе она
        # предлагает двинуться вперёд, не понимая, чего для этого не хватает.
        if step is not None:
            lines.append(f"шаг закроется, когда: {step.condition(counts)}")

        ahead = [
            BY_KEY[key] for key in EDGES.get(stage.key, ()) if ORDER[key] > ORDER[stage.key]
        ]
        for target in ahead:
            lines.append(f"следующий этап: {target.name} — {target.about}")
        return lines

    def block(self, counts: dict[str, int]) -> str:
        return INTRO + "\n".join(self.lines(counts))

    def as_param(self, counts: dict[str, int]) -> ChatCompletionMessageParam:
        return ChatCompletionSystemMessageParam(role="system", content=self.block(counts))

    def as_dict(self, counts: dict[str, int]) -> dict[str, object]:
        stage = self.at
        step = self.on
        return {
            "stage": stage.key,
            "stage_name": stage.name,
            "stage_about": stage.about,
            "stage_number": self.stage_number,
            "stage_total": len(STAGES),
            "step": step.key if step else "",
            "step_number": self.step_number,
            "step_total": len(stage.steps),
            "expected": self.expected,
            "line": self.line,
            # Шаги этапа с отметкой, какие уже пройдены: по ним видно не только где
            # мы, но и сколько осталось до перехода.
            "steps": [
                {
                    "key": one.key,
                    "expects": one.expects,
                    "condition": one.condition(counts),
                    "met": one.met(counts),
                    "passed": index < self.step_number,
                    "current": one.key == (step.key if step else ""),
                }
                for index, one in enumerate(stage.steps, start=1)
            ],
            # Переходы приходят вместе с состоянием: кнопки в панели и проверка на
            # сервере считаются одной и той же функцией.
            "moves": [move.as_dict() for move in moves(self, counts)],
        }


def build(stage: object, step: object) -> Task:
    """Состояние из того, что вернула база: неизвестное имя — начало автомата."""
    key = str(stage)
    if key not in BY_KEY:
        return Task()
    found = BY_KEY[key].step(str(step))
    return Task(stage=key, step=found.key if found else "")


def counts(working: Store | None) -> dict[str, int]:
    """Разделы ТЗ в числах: единственное, что автомат знает о рабочей памяти."""
    if working is None:
        return {}
    tally: dict[str, int] = {}
    for item in working.items:
        tally[item.section] = tally.get(item.section, 0) + 1
    return tally


def moves(task: Task, tally: dict[str, int]) -> tuple[Move, ...]:
    """Все переходы из текущего состояния — и открытые, и закрытые с причиной.

    Закрытые не отбрасываются: причина, по которой переход не состоится, нужна и
    панели (погашенная кнопка с объяснением), и отказу на попытку — иначе «нельзя»
    выглядит как поломка. Кто спрашивает, функция не знает: правило одно для
    модели, для кнопки и для прогона.
    """
    stage = task.at
    step = task.on
    found = []

    # Переход внутри этапа: на следующий шаг, если текущий закрыт. Дальше чем на
    # один шаг не прыгают — иначе ожидаемое действие можно было бы пропустить.
    if step is not None and task.step_number < len(stage.steps):
        nxt = stage.steps[task.step_number]
        found.append(
            Move(
                kind=STEP,
                stage=stage.key,
                step=nxt.key,
                name=f"шаг: {step.key} → {nxt.key}",
                blocked="" if step.met(tally) else f"не хватает: {step.condition(tally)}",
            )
        )

    for key in EDGES.get(stage.key, ()):
        target = BY_KEY[key]
        back = ORDER[key] < ORDER[stage.key]
        found.append(
            Move(
                kind=STAGE,
                stage=key,
                # Этап всегда начинается со своего первого шага, в том числе на
                # откате: вернулись именно за тем, чтобы пройти его заново.
                step=target.steps[0].key if target.steps else "",
                name=f"этап: {stage.name} → {target.name}",
                blocked="" if back else forward_block(task, tally),
                back=back,
            )
        )
    return tuple(found)


def forward_block(task: Task, tally: dict[str, int]) -> str:
    """Что держит этап: незакрытый шаг или то, что шаги ещё не пройдены до конца.

    Условие перехода между этапами — не сумма условий, а место: стоять надо на
    последнем шаге, и он должен быть закрыт. Так каждое ожидаемое действие
    оказывается пройденным, а не просто выполнимым по стечению обстоятельств.
    """
    step = task.on
    if step is None:
        return ""
    if not task.last:
        remaining = len(task.at.steps) - task.step_number
        return f"шаг {step.key} не последний в этапе, впереди ещё {remaining}"
    return "" if step.met(tally) else f"не хватает: {step.condition(tally)}"


def find(task: Task, tally: dict[str, int], stage: str = "", step: str = "") -> Move | None:
    """Переход по имени цели: чего в списке нет, того не бывает.

    Ищется среди всех переходов, включая закрытые: «такого перехода нет» и «этот
    переход пока закрыт» — разные ответы, и путать их нельзя ни в отказе
    пользователю, ни в замере отклонённых предложений модели.
    """
    for move in moves(task, tally):
        if stage and move.kind == STAGE and move.stage == stage:
            return move
        if step and move.kind == STEP and move.step == step:
            return move
    return None


def apply(move: Move) -> Task:
    """Новое состояние после перехода. Проверок здесь нет — их делает вызывающий."""
    return Task(stage=move.stage, step=move.step)
