"""Замер двух сценариев в трёх режимах: детекторы в коде и судья там, где иначе никак.

Порядок тот же, что в [day14](../day14/invariants.py) и [day24](../day24/evaluate.py):
сначала считается всё, что можно посчитать без модели, и только оставшееся отдаётся
судье. Здесь это деление проходит особенно чётко, потому что разметка сценариев
сделана под него.

Детекторами ловятся четыре вещи:

    источники есть      ход сослался хотя бы на один файл базы
    цитата дословна     verify.py нашёл цитату в том чанке, который видела модель
    ссылка разрешена    ход рода `referential` процитировал ожидаемый файл
    рамка соблюдена     в ответе есть обязательное слово и нет запрещённого

Судья нужен ровно для того, что словами не ловится: служит ли ответ цели разговора.
«Раз у меня только SQLite — где day11 держит рабочую память?» можно ответить верно
по фактам и мимо задачи, и отличить одно от другого может только читающий.

## Судья видит цель и рамку, но не память агента

Судье показывают цель сценария и ограничения, зафиксированные пользователем до
этого хода, — то есть то, что взято из [dialogs.py](dialogs.py), а не из памяти
агента. Это важно: если бы судья получал блок состояния, он судил бы режимы по
разным эталонам и у `clean`, где состояния нет вовсе, эталона не было бы совсем.
Эталон один для всех трёх режимов, и разница в оценках — это разница ответов.

## Цель проверяется в памяти, а не в ответе

Отдельный вопрос «не потерялась ли цель» решается без судьи-ответчика: исходная
формулировка сверяется с той, что лежит в памяти задачи после последнего хода.
Сверка нужна мягкая — модель переписывает цель своими словами, и сравнение строк
тут показало бы расхождение там, где его нет, — поэтому сверяет модель, одним
вызовом на прогон. Плюс считается `kept_goal`: сколько раз код не дал подменить
цель. Второе число дешевле первого и говорит больше.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field

import chat
import dialogs
import gate
import llm
import state
import storage
from chat import Turn
from dialogs import Dialogue, Say
from state import TaskState

# Сколько прогонов идёт одновременно. Внутри прогона ходы строго по одному: ход N
# зависит от памяти, собранной ходом N-1, и параллелить их значит мерить не то.
CONCURRENCY = 6

# Пороги отказа для развертки. Шкала реранкера: 1.0 — «выдержка содержит прямой
# ответ», 2/3 — «отвечает частично», 1/3 — «рядом с темой».
THRESHOLDS = (0.0, 1 / 3, 2 / 3, 1.0)

JUDGE_TEMPERATURE = 0.0
JUDGE_TOKENS = 400

SAME = "та же"
NARROWED = "сузилась"
DRIFTED = "другая"
LOST = "потеряна"
GOAL_VERDICTS = (SAME, NARROWED, DRIFTED, LOST)

FRAME_OK = "соблюдена"
FRAME_BROKEN = "нарушена"
FRAME_NONE = "—"

JUDGE = (
    "Ты оцениваешь один ход диалога технического ассистента.\n"
    "Тебе дают цель разговора, ограничения, которые пользователь зафиксировал ранее, "
    "род реплики, вопрос этого хода и ответ ассистента.\n"
    "Верни json: {\"goal\": 0|1|2, \"frame\": \"соблюдена\"|\"нарушена\"|\"—\", \"why\": \"одно предложение\"}\n"
    "`goal` — служит ли ответ цели разговора:\n"
    "2 — ответ работает на цель: он про то, что человеку нужно для его задачи;\n"
    "1 — ответ верен по фактам, но цель в нём не видна: справка вместо помощи;\n"
    "0 — ответ уводит от цели или отвечает не о том.\n"
    "Важное исключение. Если род реплики — «отвлечение», значит пользователь сам "
    "спросил про другое, и отвечать ему — правильно. Оценивай тогда так: 2 — ответил "
    "на заданный вопрос и не объявил новую цель разговора; 1 — ответил, но заявил или "
    "подразумевает, что разговор теперь о другом; 0 — не ответил на заданный вопрос.\n"
    "`frame` — нарушает ли ответ зафиксированные ограничения и термины: "
    "«нарушена», если ответ предлагает то, что пользователь исключил, или называет "
    "вещи не тем словом, которое он закрепил; «—», если ограничений не было.\n"
    "Отказ «в базе ответа нет» вместе с уточняющим вопросом — это 1, если уточнение "
    "держит цель, и 0, если нет. Полноту фактов не оценивай: это делают другие проверки."
)

MATCH = (
    "Ты сверяешь две формулировки цели одного и того же разговора.\n"
    "Первая — исходная, как её задал пользователь. Вторая — та, что ассистент "
    "держит в памяти задачи сейчас.\n"
    "Верни json: {\"verdict\": \"та же\"|\"сузилась\"|\"другая\"|\"потеряна\", \"why\": \"одно предложение\"}\n"
    "«та же» — про то же самое, пусть и другими словами;\n"
    "«сузилась» — та же задача, но сведённая к одной её части;\n"
    "«другая» — речь о другой задаче;\n"
    "«потеряна» — второй формулировки нет вовсе."
)

FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


def _loose(text: str) -> dict[str, object]:
    try:
        payload = json.loads(FENCE.sub("", text.strip()))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


# --- детекторы ---------------------------------------------------------------


def fold(text: str) -> str:
    return " ".join(text.lower().split())


def share(values: list[bool | None]) -> float | None:
    """Доля по тем ходам, где проверка вообще применима.

    `None` означает «эта проверка к этому ходу не относится», и считать такие
    ходы провалами нельзя: рамки до третьей реплики не существует, и делить на
    все четырнадцать значило бы наказывать режим за первые два хода.
    """
    known = [value for value in values if value is not None]
    return sum(known) / len(known) if known else None


def share_of(values: list[float | None]) -> float | None:
    """То же среднее, но по числам: балл лучшей выдержки есть не у каждого хода."""
    known = [value for value in values if value is not None]
    return sum(known) / len(known) if known else None


def touched(paths: list[str], wanted: tuple[str, ...]) -> bool:
    """Попал ли ход хотя бы в один из ожидаемых файлов. Сверка по подстроке пути."""
    return any(any(part in path for path in paths) for part in wanted)


def frame_of(dialogue: Dialogue, number: int) -> str:
    """Ограничения и термины, зафиксированные до этого хода включительно.

    Берутся из сценария, а не из памяти агента: эталон обязан быть один для всех
    трёх режимов, иначе `clean` сравнивать не с чем.
    """
    lines = [
        f"— {say.text}"
        for position, say in enumerate(dialogue.says[:number], start=1)
        if say.kind in (dialogs.CONSTRAINT, dialogs.TERM)
    ]
    return "\n".join(lines)


@dataclass(frozen=True)
class Scored:
    """Один ход после замера: что посчитали детекторы и что сказал судья."""

    number: int
    kind: str
    question: str
    standalone: str
    changed: bool
    text: str
    paths: list[str]
    has_sources: bool
    claims: int
    exact: int
    grounded: int
    fabricated: int
    refused: bool
    refusal: str
    # Балл лучшей выдержки от реранкера — то самое число, по которому слой в коде
    # решает, отвечать ли. Нужен в разрезе режимов: переписанная реплика — другой
    # поисковый запрос, и попасть по нему в одну выдержку может быть труднее.
    best: float | None = None
    resolved_ok: bool | None = None
    framed_ok: bool | None = None
    missing: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    goal: int | None = None
    frame: str = FRAME_NONE
    why: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stage_tokens: int = 0
    memory_tokens: int = 0
    total_tokens: int = 0
    seconds: float = 0.0
    kept_goal: str = ""
    added: int = 0
    refused_adds: tuple[str, ...] = ()

    @property
    def exact_share(self) -> float | None:
        return self.exact / self.claims if self.claims else None

    def as_dict(self) -> dict[str, object]:
        return {
            "number": self.number,
            "kind": self.kind,
            "kind_title": dialogs.KIND_TITLES[self.kind],
            "question": self.question,
            "standalone": self.standalone,
            "changed": self.changed,
            "text": self.text,
            "paths": self.paths,
            "has_sources": self.has_sources,
            "claims": self.claims,
            "exact": self.exact,
            "grounded": self.grounded,
            "fabricated": self.fabricated,
            "exact_share": self.exact_share,
            "refused": self.refused,
            "refusal": self.refusal,
            "best": self.best,
            "resolved_ok": self.resolved_ok,
            "framed_ok": self.framed_ok,
            "missing": list(self.missing),
            "forbidden": list(self.forbidden),
            "goal": self.goal,
            "frame": self.frame,
            "why": self.why,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "stage_tokens": self.stage_tokens,
            "memory_tokens": self.memory_tokens,
            "total_tokens": self.total_tokens,
            "seconds": round(self.seconds, 2),
            "kept_goal": self.kept_goal,
            "added": self.added,
            "refused_adds": list(self.refused_adds),
        }


def detect(say: Say, turn: Turn) -> dict[str, object]:
    """Всё, что считается без модели: ссылки, рамка, источники, цитаты."""
    body = fold(turn.text)

    resolved_ok = None
    if say.kind == dialogs.REFERENTIAL and say.expect_paths:
        resolved_ok = touched(turn.paths, say.expect_paths)

    framed_ok = None
    missing: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    if say.framed:
        missing = tuple(word for word in say.must_say if fold(word) not in body)
        forbidden = tuple(word for word in say.must_not_say if fold(word) in body)
        framed_ok = not missing and not forbidden

    report = turn.report
    return {
        "paths": turn.paths,
        "has_sources": turn.has_sources,
        "claims": report.claims if report else 0,
        "exact": report.exact if report else 0,
        "grounded": report.grounded if report else 0,
        "fabricated": report.fabricated if report else 0,
        "resolved_ok": resolved_ok,
        "framed_ok": framed_ok,
        "missing": missing,
        "forbidden": forbidden,
    }


# --- судья -------------------------------------------------------------------


async def judge(dialogue: Dialogue, number: int, say: Say, turn: Turn) -> dict[str, object]:
    """Оценка одного хода: служит ли цели и не нарушена ли рамка."""
    frame = frame_of(dialogue, number)
    parts = [
        f"Цель разговора: {dialogue.goal}",
        f"Род реплики: {dialogs.KIND_TITLES[say.kind]}",
        f"Вопрос хода: {say.text}",
        f"Ответ ассистента: {turn.text}",
    ]
    if frame:
        parts.insert(2, f"Пользователь ранее зафиксировал:\n{frame}")

    reply = await llm.complete(
        [
            {"role": "system", "content": JUDGE},
            {"role": "user", "content": "\n\n".join(parts)},
        ],
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_TOKENS,
        json=True,
    )
    payload = _loose(reply.text)

    score = payload.get("goal")
    verdict = str(payload.get("frame") or FRAME_NONE).strip()
    return {
        "goal": score if isinstance(score, int) and 0 <= score <= 2 else None,
        "frame": verdict if verdict in (FRAME_OK, FRAME_BROKEN, FRAME_NONE) else FRAME_NONE,
        "why": str(payload.get("why") or "").strip(),
    }


async def goal_match(reference: str, current: TaskState) -> dict[str, str]:
    """Сверить исходную цель с той, что лежит в памяти задачи."""
    if current.goal is None:
        return {"verdict": LOST, "why": "в памяти задачи цели нет"}

    reply = await llm.complete(
        [
            {"role": "system", "content": MATCH},
            {
                "role": "user",
                "content": (
                    f"Исходная цель: {reference}\n"
                    f"Цель в памяти задачи: {current.goal.text}"
                ),
            },
        ],
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_TOKENS,
        json=True,
    )
    payload = _loose(reply.text)
    verdict = str(payload.get("verdict") or "").strip()
    return {
        "verdict": verdict if verdict in GOAL_VERDICTS else DRIFTED,
        "why": str(payload.get("why") or "").strip(),
    }


# --- прогон ------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """Сценарий в одном режиме: ходы, итоговая память и сводка."""

    dialogue: str
    title: str
    mode: str
    turns: list[Scored]
    state: TaskState = field(default_factory=TaskState)
    goal_verdict: str = FRAME_NONE
    goal_why: str = ""
    session_id: str = ""

    @property
    def length(self) -> int:
        return len(self.turns)

    def _of(self, kinds: tuple[str, ...]) -> list[Scored]:
        return [turn for turn in self.turns if turn.kind in kinds]

    def summary(self) -> dict[str, object]:
        claims = sum(turn.claims for turn in self.turns)
        exact = sum(turn.exact for turn in self.turns)
        fabricated = sum(turn.fabricated for turn in self.turns)
        scored = [turn.goal for turn in self.turns if turn.goal is not None]
        probes = [turn.goal for turn in self._of((dialogs.PROBE,)) if turn.goal is not None]

        return {
            "dialogue": self.dialogue,
            "title": self.title,
            "mode": self.mode,
            "mode_title": chat.MODE_TITLES[self.mode],
            "turns": self.length,
            "sources": share([turn.has_sources for turn in self.turns]),
            "claims": claims,
            "exact": exact / claims if claims else None,
            "fabricated": fabricated,
            "resolved": share([turn.resolved_ok for turn in self.turns]),
            "framed": share([turn.framed_ok for turn in self.turns]),
            "frame_broken": sum(1 for turn in self.turns if turn.frame == FRAME_BROKEN),
            "goal": sum(scored) / len(scored) if scored else None,
            "goal_two": sum(1 for score in scored if score == 2),
            "probe_goal": sum(probes) / len(probes) if probes else None,
            "probes": len(probes),
            "refused": sum(1 for turn in self.turns if turn.refused),
            "rewritten": sum(1 for turn in self.turns if turn.changed),
            "kept_goal": sum(1 for turn in self.turns if turn.kept_goal),
            "facts": len(self.state.facts),
            "crowded": sum(
                1 for turn in self.turns for reason in turn.refused_adds if reason == state.CROWDED
            ),
            "duplicates": sum(
                1 for turn in self.turns for reason in turn.refused_adds if reason == state.DUPLICATE
            ),
            "goal_verdict": self.goal_verdict,
            "goal_why": self.goal_why,
            "goal_text": self.state.goal.text if self.state.goal else "",
            "prompt_tokens": self._mean("prompt_tokens"),
            "completion_tokens": self._mean("completion_tokens"),
            "stage_tokens": self._mean("stage_tokens"),
            "memory_tokens": self._mean("memory_tokens"),
            "total_tokens": self._mean("total_tokens"),
            "seconds": self._mean("seconds"),
        }

    def _mean(self, name: str) -> float:
        values = [getattr(turn, name) for turn in self.turns]
        return sum(values) / len(values) if values else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary(),
            "state": self.state.as_dict(),
            "session_id": self.session_id,
            "turns": [turn.as_dict() for turn in self.turns],
        }


async def one(
    dialogue: Dialogue,
    mode: str,
    store: storage.Storage,
    threshold: float | None = None,
    judged: bool = True,
) -> Run:
    """Прогнать сценарий в одном режиме и оценить каждый ход.

    Судья спрашивается после всего разговора, а не по ходу: оценка на ход не
    влияет, а вызовы судьи, пущенные вперемешку с ходами, растянули бы прогон на
    время, которое к самому чату отношения не имеет.
    """
    dialog = await chat.Dialog.open(store, mode, f"{dialogue.title} — {chat.MODE_TITLES[mode]}")

    turns: list[Turn] = []
    for say in dialogue.says:
        turns.append(await dialog.ask(say.text, threshold))

    verdicts: list[dict[str, object]] = [{} for _ in turns]
    if judged:
        verdicts = list(
            await asyncio.gather(
                *(
                    judge(dialogue, number, say, turn)
                    for number, (say, turn) in enumerate(zip(dialogue.says, turns, strict=True), start=1)
                )
            )
        )

    scored = [
        Scored(
            number=number,
            kind=say.kind,
            question=say.text,
            standalone=turn.resolved.standalone,
            changed=turn.resolved.changed,
            text=turn.text,
            refused=turn.refused,
            refusal=turn.refusal,
            best=turn.verdict.best if turn.verdict else None,
            prompt_tokens=turn.reply.prompt_tokens,
            completion_tokens=turn.reply.completion_tokens,
            stage_tokens=turn.stage_tokens,
            memory_tokens=turn.memory_tokens,
            total_tokens=turn.total_tokens,
            seconds=turn.seconds,
            kept_goal=turn.tracked.applied.kept_goal if turn.tracked else "",
            added=len(turn.tracked.applied.added) if turn.tracked else 0,
            refused_adds=tuple(
                item.reason for item in (turn.tracked.applied.refused if turn.tracked else ())
            ),
            **detect(say, turn),  # type: ignore[arg-type]
            **verdict,  # type: ignore[arg-type]
        )
        for number, (say, turn, verdict) in enumerate(
            zip(dialogue.says, turns, verdicts, strict=True), start=1
        )
    ]

    matched = {"verdict": FRAME_NONE, "why": ""}
    if chat.memory_of(mode).tracked and judged:
        matched = await goal_match(dialogue.goal, dialog.state)

    return Run(
        dialogue=dialogue.key,
        title=dialogue.title,
        mode=mode,
        turns=scored,
        state=dialog.state,
        goal_verdict=str(matched["verdict"]),
        goal_why=str(matched["why"]),
        session_id=dialog.session_id,
    )


async def every(
    keys: tuple[str, ...] = dialogs.KEYS,
    modes: tuple[str, ...] = chat.MODES,
    store: storage.Storage | None = None,
    threshold: float | None = None,
    judged: bool = True,
) -> list[Run]:
    """Все сценарии во всех режимах. Прогоны идут параллельно, ходы внутри — нет."""
    store = store or storage.Storage()
    await store.init()

    pairs = [(dialogs.load(key), mode) for key in keys for mode in modes]
    gate = asyncio.Semaphore(CONCURRENCY)

    async def guarded(dialogue: Dialogue, mode: str) -> Run:
        async with gate:
            return await one(dialogue, mode, store, threshold, judged)

    return list(await asyncio.gather(*(guarded(dialogue, mode) for dialogue, mode in pairs)))


# --- сводки ------------------------------------------------------------------


def by_mode(runs: list[Run]) -> dict[str, dict[str, object]]:
    """Сложить сценарии в одну строку на режим: сравнивают именно режимы."""
    folded: dict[str, dict[str, object]] = {}

    for mode in chat.MODES:
        picked = [run for run in runs if run.mode == mode]
        if not picked:
            continue

        turns = [turn for run in picked for turn in run.turns]
        if not turns:
            continue

        claims = sum(turn.claims for turn in turns)
        scored = [turn.goal for turn in turns if turn.goal is not None]
        probes = [
            turn.goal for turn in turns if turn.kind == dialogs.PROBE and turn.goal is not None
        ]
        referential = [turn for turn in turns if turn.kind == dialogs.REFERENTIAL]

        folded[mode] = {
            "mode": mode,
            "title": chat.MODE_TITLES[mode],
            "runs": len(picked),
            "turns": len(turns),
            "sources": share([turn.has_sources for turn in turns]),
            # Главное число задания: «всегда выводит источники». Считать его по всем
            # ходам нельзя — отказ источников не имеет и не должен иметь. Вопрос в
            # другом: бывает ли ответ без источников. Знаменатель — отвеченные ходы.
            "answered": sum(1 for turn in turns if not turn.refused),
            "answered_sources": share(
                [turn.has_sources for turn in turns if not turn.refused]
            ),
            "claims": claims,
            "exact": sum(turn.exact for turn in turns) / claims if claims else None,
            "fabricated": sum(turn.fabricated for turn in turns),
            "resolved": share([turn.resolved_ok for turn in turns]),
            "resolved_count": sum(1 for turn in referential if turn.resolved_ok),
            "referential": len(referential),
            "framed": share([turn.framed_ok for turn in turns]),
            "frame_broken": sum(1 for turn in turns if turn.frame == FRAME_BROKEN),
            "goal": sum(scored) / len(scored) if scored else None,
            "goal_two": sum(1 for score in scored if score == 2),
            "goal_scored": len(scored),
            "probe_goal": sum(probes) / len(probes) if probes else None,
            "probes": len(probes),
            "refused": sum(1 for turn in turns if turn.refused),
            "best": share_of([turn.best for turn in turns]),
            "rewritten": sum(1 for turn in turns if turn.changed),
            "kept_goal": sum(1 for turn in turns if turn.kept_goal),
            "goal_verdicts": [run.goal_verdict for run in picked],
            "prompt_tokens": sum(turn.prompt_tokens for turn in turns) / len(turns),
            "completion_tokens": sum(turn.completion_tokens for turn in turns) / len(turns),
            "stage_tokens": sum(turn.stage_tokens for turn in turns) / len(turns),
            "memory_tokens": sum(turn.memory_tokens for turn in turns) / len(turns),
            "total_tokens": sum(turn.total_tokens for turn in turns) / len(turns),
            "seconds": sum(turn.seconds for turn in turns) / len(turns),
        }

    return folded


async def sweep(
    thresholds: tuple[float, ...] = THRESHOLDS,
    key: str = dialogs.KEYS[0],
    mode: str = chat.MODES[-1],
    store: storage.Storage | None = None,
) -> list[dict[str, object]]:
    """Один сценарий при нескольких порогах отказа.

    Порог 1.0 снят в day24 на одиночных вопросах, и там он был бесплатным: на
    всех контрольных вопросах хотя бы одна выдержка получала от реранкера 1.0.
    В диалоге так не выходит, и причина не в пороге, а в вопросах: «что мне
    взять из двух» прямого ответа в базе не имеет ни в одном чанке — ответ
    собирается из нескольких выдержек, каждая из которых отвечает частично.
    Развертка нужна, чтобы это стало числом, а не догадкой.

    Прогоны идут без судьи: сравниваются отказы и источники, а их считают
    детекторы. Платить за судью шесть раз ради колонок, которых в сравнении
    порогов нет, незачем.
    """
    store = store or storage.Storage()
    await store.init()
    dialogue = dialogs.load(key)

    runs = await asyncio.gather(
        *(one(dialogue, mode, store, threshold, judged=False) for threshold in thresholds)
    )

    rows: list[dict[str, object]] = []
    for threshold, run in zip(thresholds, runs, strict=True):
        claims = sum(turn.claims for turn in run.turns)
        rows.append(
            {
                "threshold": threshold,
                "turns": run.length,
                "refused": sum(1 for turn in run.turns if turn.refused),
                "by_code": sum(1 for turn in run.turns if turn.refusal in (gate.EMPTY, gate.BELOW)),
                "by_model": sum(1 for turn in run.turns if turn.refusal == gate.MODEL),
                "sources": sum(turn.has_sources for turn in run.turns) / run.length,
                "claims": claims,
                "exact": sum(turn.exact for turn in run.turns) / claims if claims else None,
                "fabricated": sum(turn.fabricated for turn in run.turns),
                "framed": share([turn.framed_ok for turn in run.turns]),
                "resolved": share([turn.resolved_ok for turn in run.turns]),
            }
        )
    return rows


def by_kind(runs: list[Run]) -> dict[str, dict[str, dict[str, object]]]:
    """Те же режимы, но разложенные по роду реплики: видно, где память работает."""
    out: dict[str, dict[str, dict[str, object]]] = {}

    for kind in dialogs.KINDS:
        row: dict[str, dict[str, object]] = {}
        for mode in chat.MODES:
            turns = [
                turn
                for run in runs
                if run.mode == mode
                for turn in run.turns
                if turn.kind == kind
            ]
            if not turns:
                continue

            scored = [turn.goal for turn in turns if turn.goal is not None]
            row[mode] = {
                "turns": len(turns),
                "sources": sum(turn.has_sources for turn in turns) / len(turns),
                "goal": sum(scored) / len(scored) if scored else None,
                "refused": sum(1 for turn in turns if turn.refused),
            }
        if row:
            out[kind] = row

    return out
