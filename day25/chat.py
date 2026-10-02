"""Ход чата целиком и три режима памяти, между которыми его сравнивают.

[day24](../day24/answer.py) держал четыре режима и менял между ними форму ответа.
Здесь форма одна — json со списком утверждений, у каждого источник и дословная
цитата, — а меняется то, что агент помнит:

| Режим | История | Память задачи | Что уходит в поиск |
| --- | --- | --- | --- |
| `clean` | нет | нет | реплика как есть |
| `history` | хвост 6 реплик | нет | реплика, разрешённая по истории |
| `tracked` | хвост 6 реплик | есть | реплика, разрешённая по истории и цели |

`clean` — точка отсчёта, и это буквально day24: каждая реплика как отдельный
вопрос, без связи с предыдущими. Он нужен не как «плохой режим», а как
единственный способ сказать, сколько стоит и что даёт память: разница между
`clean` и `history` — вклад разговора, между `history` и `tracked` — вклад
состояния задачи.

## Порядок блоков в запросе

    1. system  роль и схема ответа из cite.py
    2. system  память задачи: цель, уточнения, ограничения, термины
    3. system  выдержки из базы под номерами и то, по чему их искали
    4.         хвост разговора дословно, по реплике на сообщение
    5. user    текущая реплика

Порядок повторяет [day13](../day13/agent.py) и [day14](../day14/agent.py): рамка
раньше разговора. Причина та же — блок, стоящий после истории, читается моделью
как последняя просьба и действует один ход, а стоящий до неё — как условия, в
которых происходит всё остальное.

Выдержки живут в системном сообщении, а не в пользовательском, как было в day24.
Разница появилась именно из-за истории: пользовательское сообщение с пятью
чанками в чате становится одной из реплик разговора, и на следующем ходе модель
видит, что «пользователь присылал выдержки», — а на третьем начинает на них
ссылаться, хотя контекст давно другой.

## История не конспектируется

В [day9](../day9/memory.py) история старше хвоста уходила в конспект, который
агент писал себе сам. Здесь конспекта нет, и это не упущение: **память задачи и
есть конспект**, только не пересказывающий разговор, а хранящий его условия.
Разница в том, что конспект day9 отвечал на вопрос «о чём говорили», а состояние
отвечает на «что решили», и именно второе нужно, чтобы не потерять цель.

Проверяется это тем же замером: если бы конспект был нужен, `tracked` терял бы на
длинных сценариях факты, названные в начале, — а он их как раз держит.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace

import cite
import embed
import gate
import llm
import pipeline
import resolve
import track
import verify
from cite import Cited
from gate import Verdict
from pipeline import Context, Plan
from resolve import Resolved
from state import TaskState
from storage import Storage
from track import Tracked

MODES = ("clean", "history", "tracked")
MODE_TITLES = {
    "clean": "без памяти",
    "history": "только история",
    "tracked": "история + память задачи",
}

# Конвейер взят у day24 целиком и в день не входит: `keyword` + `llm` — лучшая
# клетка матрицы day23, recall@5 0.81 на 119 пробах. Один план на все режимы,
# иначе замер мерил бы разный поиск, а не разную память.
REWRITER = "keyword"
RERANKER = "llm"
PLAN = Plan.of(rewriter=REWRITER, reranker=RERANKER)

# Сколько последних сообщений уходит в модель дословно. Шесть — три хода, как в
# day9 и day11: та часть разговора, где важны формулировки, а не смысл.
TAIL_MESSAGES = 6

TEMPERATURE = 0.2
MAX_TOKENS = 1000

PASSAGES_INTRO = "Выдержки из базы проекта, каждая под своим номером:"
SEARCHED_AS = "Искали по запросу: "


@dataclass(frozen=True)
class Memory:
    """Что режиму разрешено помнить. Ветвлений по имени режима в коде нет."""

    history: bool
    tracked: bool


MEMORY = {
    "clean": Memory(history=False, tracked=False),
    "history": Memory(history=True, tracked=False),
    "tracked": Memory(history=True, tracked=True),
}


def memory_of(mode: str) -> Memory:
    if mode not in MODES:
        raise ValueError(f"Режима {mode!r} нет. Есть: {', '.join(MODES)}.")
    return MEMORY[mode]


# --- запрос ------------------------------------------------------------------


def passages_block(context: Context, resolved: Resolved) -> str:
    """Блок выдержек вместе с запросом, по которому их нашли.

    Запрос печатается только когда реплику переписали. Это не отладочный вывод:
    без него модель, получив на «а в нём как?» пять чанков про day13, не видит,
    что «он» уже разрешён в day13, и либо переспрашивает, либо угадывает заново.
    """
    head = PASSAGES_INTRO
    if resolved.changed:
        head = f"{SEARCHED_AS}{resolved.standalone}\n{head}"
    return f"{head}\n\n{context.block}"


def messages(
    question: str,
    context: Context | None,
    current: TaskState | None,
    history: list[dict[str, str]],
    resolved: Resolved | None = None,
) -> list[dict[str, str]]:
    """Запрос к модели. Рамка раньше разговора, выдержки — системным блоком."""
    built: list[dict[str, str]] = [
        {"role": "system", "content": f"{cite.SYSTEM}\n\n{cite.RULES}"}
    ]

    if current is not None and (block := current.as_param()) is not None:
        built.append(dict(block))

    if context is not None and context.sources:
        built.append(
            {
                "role": "system",
                "content": passages_block(context, resolved or resolve.plain(question)),
            }
        )

    built.extend(
        {"role": item["role"], "content": item["content"]} for item in history[-TAIL_MESSAGES:]
    )
    built.append({"role": "user", "content": question})
    return built


def sources_of(turn: "Turn") -> list[dict[str, object]]:
    """Источники хода по утверждениям — в том виде, в котором их хранит база.

    Вердикт берётся по месту в списке, а не поиском по тексту утверждения:
    `verify.review` возвращает сверки в том же порядке, а два утверждения в
    ответе бывают дословно одинаковыми — и тогда поиск по тексту приписал бы
    обоим вердикт первого.
    """
    if turn.context is None or turn.cited is None:
        return []

    by_number = {source.number: source for source in turn.context.sources}
    checked = turn.report.checked if turn.report else []

    rows: list[dict[str, object]] = []
    for position, claim in enumerate(turn.cited.claims):
        source = by_number.get(claim.source or 0)
        rows.append(
            {
                "claim": claim.text,
                "number": claim.source,
                "chunk_id": source.chunk_id if source else None,
                "path": source.path if source else "",
                "section": source.section if source else "",
                "quote": claim.quote,
                "verdict": checked[position].verdict if position < len(checked) else "",
            }
        )
    return rows


# --- ход ---------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    """Один ход чата со всем, по чему его можно проверить и оплатить."""

    mode: str
    number: int
    question: str
    text: str
    resolved: Resolved
    reply: llm.Reply
    context: Context | None = None
    cited: Cited | None = None
    report: verify.Report | None = None
    verdict: Verdict | None = None
    tracked: Tracked | None = None
    state: TaskState = field(default_factory=TaskState)
    memory_tokens: int = 0

    @property
    def claims(self) -> list[cite.Claim]:
        return self.cited.claims if self.cited else []

    @property
    def refused(self) -> bool:
        if self.verdict is not None and not self.verdict.passed:
            return True
        return self.cited.refused if self.cited else False

    @property
    def refusal(self) -> str:
        """Какой слой сказал «не знаю». Пустая строка — значит ответ дан."""
        if self.verdict is not None and not self.verdict.passed:
            return self.verdict.reason
        if self.cited is not None and self.cited.refused:
            return gate.MODEL
        return ""

    @property
    def clarify(self) -> str:
        if self.verdict is not None and not self.verdict.passed:
            return self.verdict.clarify
        return self.cited.clarify if self.cited else ""

    @property
    def numbers(self) -> list[int]:
        """Номера выдержек, на которые ход сослался. Выдуманные отброшены."""
        if self.context is None or self.cited is None:
            return []
        known = {source.number for source in self.context.sources}
        return sorted({claim.source for claim in self.cited.claims if claim.source in known})

    @property
    def paths(self) -> list[str]:
        if self.context is None:
            return []
        by_number = {source.number: source.path for source in self.context.sources}
        seen: list[str] = []
        for number in self.numbers:
            path = by_number.get(number)
            if path and path not in seen:
                seen.append(path)
        return seen

    @property
    def has_sources(self) -> bool:
        return bool(self.paths)

    @property
    def stage_tokens(self) -> int:
        """Всё, за что ход заплатил помимо самого ответа."""
        return (
            self.resolved.tokens
            + (self.context.stage_tokens if self.context else 0)
            + (self.tracked.proposal.tokens if self.tracked else 0)
        )

    @property
    def total_tokens(self) -> int:
        return self.reply.prompt_tokens + self.reply.completion_tokens + self.stage_tokens

    @property
    def seconds(self) -> float:
        return (
            self.resolved.seconds
            + (self.context.stage_seconds if self.context else 0.0)
            + self.reply.seconds
            + (self.tracked.proposal.seconds if self.tracked else 0.0)
        )

    def row(self) -> dict[str, object]:
        """Строка для таблицы `turns`."""
        return {
            "mode": self.mode,
            "question": self.question,
            "standalone": self.resolved.standalone,
            "answer": self.text,
            "refused": int(self.refused),
            "gate_reason": self.refusal,
            "clarify": self.clarify,
            "kept_goal": self.tracked.applied.kept_goal if self.tracked else "",
            "claims": len(self.claims),
            "prompt_tokens": self.reply.prompt_tokens,
            "completion_tokens": self.reply.completion_tokens,
            "stage_tokens": self.stage_tokens,
            "memory_tokens": self.memory_tokens,
            "seconds": round(self.seconds, 3),
        }

    def as_dict(self) -> dict[str, object]:
        # Поля `reply` перечислены, а не расплющены словарём, как в day24. Там это
        # сходило: ключи не пересекались ни с чем важным. Здесь пересеклись бы оба —
        # `text` затёрся бы сырым json вместо собранного ответа, а `seconds` временем
        # одного вызова вместо времени всего хода.
        return {
            "mode": self.mode,
            "title": MODE_TITLES[self.mode],
            "number": self.number,
            "question": self.question,
            "text": self.text,
            "refused": self.refused,
            "refusal": self.refusal,
            "clarify": self.clarify,
            "numbers": self.numbers,
            "paths": self.paths,
            "has_sources": self.has_sources,
            "resolve": self.resolved.as_dict(),
            "structured": self.cited.as_dict() if self.cited else None,
            "verify": self.report.as_dict() if self.report else None,
            "gate": self.verdict.as_dict() if self.verdict else None,
            "track": self.tracked.as_dict() if self.tracked else None,
            "state": self.state.as_dict(),
            "memory_tokens": self.memory_tokens,
            "stage_tokens": self.stage_tokens,
            "total_tokens": self.total_tokens,
            "seconds": round(self.seconds, 2),
            "answer_seconds": round(self.reply.seconds, 2),
            "prompt_tokens": self.reply.prompt_tokens,
            "completion_tokens": self.reply.completion_tokens,
            "context": self.context.as_dict() if self.context else None,
        }


def passages(context: Context) -> dict[int, verify.Passage]:
    """Выдержки глазами сверки: номер, адрес и текст, который видела модель."""
    return {
        source.number: verify.Passage(
            number=source.number,
            chunk_id=source.chunk_id,
            path=source.path,
            section=source.section,
            text=context.texts.get(source.number, ""),
        )
        for source in context.sources
    }


def cost_of(parts: list[str]) -> int:
    """Во что обходятся блоки памяти — локальным токенайзером, не счётчиком DeepSeek.

    Точного числа провайдер по частям запроса не даёт: `usage` приходит на запрос
    целиком. Поэтому цена памяти в README снимается разницей между режимами, а
    это число — оценка для страницы, чтобы блок состояния можно было смотреть
    рядом с его размером.
    """
    body = "\n".join(part for part in parts if part)
    return embed.ModelTokenizer().count(body) if body else 0


# --- диалог ------------------------------------------------------------------


class Dialog:
    """Один разговор: история и память задачи в базе, режим — на всю сессию.

    Снимки держатся и в процессе тоже, хотя источник правды — база. Читать
    историю и состояние из SQLite перед каждым ходом было бы вернее по букве, но
    в прогоне на пятнадцать реплик это пятнадцать лишних чтений ради данных,
    которые только что сами же и записали.
    """

    def __init__(self, storage: Storage, session_id: str, mode: str) -> None:
        self.storage = storage
        self.session_id = session_id
        self.mode = mode
        self.memory = memory_of(mode)
        self._history: list[dict[str, str]] = []
        self._state = TaskState()
        self._turn = 0

    @classmethod
    async def open(cls, storage: Storage, mode: str, title: str = "") -> "Dialog":
        memory_of(mode)
        session_id = await storage.open(mode, title)
        return cls(storage, session_id, mode)

    @classmethod
    async def resume(cls, storage: Storage, session_id: str) -> "Dialog":
        row = await storage.session(session_id)
        if row is None:
            raise ValueError(f"Диалога {session_id!r} нет.")

        dialog = cls(storage, session_id, str(row["mode"]))
        dialog._history = [
            {"role": str(item["role"]), "content": str(item["content"])}
            for item in await storage.history(session_id)
        ]
        dialog._state = await storage.load_state(session_id)
        dialog._turn = await storage.turns_done(session_id)
        return dialog

    @property
    def state(self) -> TaskState:
        return self._state

    @property
    def history(self) -> list[dict[str, str]]:
        return list(self._history)

    @property
    def turns(self) -> int:
        return self._turn

    def _seen(self) -> list[dict[str, str]]:
        """История, которую видит этот режим. У `clean` её нет вовсе."""
        return self._history if self.memory.history else []

    def _known(self) -> TaskState | None:
        return self._state if self.memory.tracked else None

    # --- этапы хода ----------------------------------------------------------

    async def prepare(self, question: str) -> tuple[Resolved, Context]:
        """Разрешить ссылки и собрать контекст. Один шаг на два вызова модели."""
        resolved = (
            await resolve.apply(question, self._seen(), self._known())
            if self.memory.history or self.memory.tracked
            else resolve.plain(question)
        )
        context = await pipeline.prepare(resolved.standalone, PLAN)
        return resolved, context

    def inspect(self, context: Context, threshold: float | None = None) -> Verdict:
        return gate.inspect(
            context,
            gate.THRESHOLD if threshold is None else threshold,
            self._known(),
        )

    def request(self, question: str, context: Context, resolved: Resolved) -> list[dict[str, str]]:
        return messages(question, context, self._known(), self._seen(), resolved)

    def weight(self) -> int:
        """Оценка цены памяти этого хода в токенах: блок состояния и хвост истории."""
        parts: list[str] = []
        if self._known() is not None and not self._state.empty:
            parts.append(self._state.block())
        parts.extend(item["content"] for item in self._seen()[-TAIL_MESSAGES:])
        return cost_of(parts)

    def assemble(
        self,
        question: str,
        resolved: Resolved,
        context: Context | None,
        reply: llm.Reply,
        verdict: Verdict | None,
        memory_tokens: int,
    ) -> Turn:
        """Собрать ход из текста модели: разбор, сверка и номер источника — здесь."""
        parsed = cite.parse(reply.text)
        report = verify.review(parsed.claims, passages(context)) if context else None

        return Turn(
            mode=self.mode,
            number=self._turn + 1,
            question=question,
            text=parsed.text,
            resolved=resolved,
            reply=reply,
            context=context,
            cited=parsed,
            report=report,
            verdict=verdict,
            state=self._state,
            memory_tokens=memory_tokens,
        )

    def stopped(
        self,
        question: str,
        resolved: Resolved,
        context: Context | None,
        verdict: Verdict,
        memory_tokens: int,
    ) -> Turn:
        """Отказ слоя в коде: модель не вызывалась, и цена ответа нулевая."""
        return Turn(
            mode=self.mode,
            number=self._turn + 1,
            question=question,
            text=verdict.text,
            resolved=resolved,
            reply=llm.Reply(text=verdict.text, prompt_tokens=0, completion_tokens=0, seconds=0.0),
            context=context,
            cited=Cited(unknown=True, clarify=verdict.clarify),
            report=verify.Report(),
            verdict=verdict,
            state=self._state,
            memory_tokens=memory_tokens,
        )

    async def remember(self, turn: Turn) -> Turn:
        """Обновить память задачи и записать ход. Возвращает ход со свежим снимком.

        Трекер спрашивается до записи, но после ответа: ему нужен и вопрос, и то,
        чем на него ответили. Режимы без памяти задачи сюда не платят вовсе — их
        `Tracked` пустой, см. `track.skipped`.
        """
        tracked = (
            await track.update(
                self._state, turn.question, turn.text, turn.number, self._seen()
            )
            if self.memory.tracked
            else track.skipped(self._state)
        )

        self._state = tracked.state
        self._turn = turn.number
        self._history.append({"role": "user", "content": turn.question})
        self._history.append({"role": "assistant", "content": turn.text})

        done = replace(turn, tracked=tracked, state=tracked.state)

        await self.storage.save_state(
            self.session_id,
            done.number,
            tracked.applied.goal_set,
            tracked.applied.added,
            tracked.applied.dropped,
        )
        await self.storage.save_turn(
            self.session_id,
            done.number,
            done.row(),
            sources_of(done),
            done.question,
            done.text,
        )
        if done.number == 1:
            await self.storage.rename(self.session_id, done.question)

        return done

    # --- целиком -------------------------------------------------------------

    async def ask(self, question: str, threshold: float | None = None) -> Turn:
        """Ход без стрима: для прогонов сценариев и для терминала."""
        question = question.strip()
        if not question:
            raise ValueError("Пустую реплику задавать нечего.")

        resolved, context = await self.prepare(question)
        weight = self.weight()
        verdict = self.inspect(context, threshold)

        if not verdict.passed:
            return await self.remember(
                self.stopped(question, resolved, context, verdict, weight)
            )

        reply = await llm.complete(
            self.request(question, context, resolved),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            json=True,
        )
        return await self.remember(
            self.assemble(question, resolved, context, reply, verdict, weight)
        )

    async def stream(
        self, question: str, threshold: float | None = None
    ) -> AsyncIterator[tuple[str, object]]:
        """Тот же ход кадрами для страницы: контекст, текст по кускам, итог.

        Стрим здесь показывает не столько скорость, сколько порядок: сначала
        видно, по чему искали и что нашлось, и только потом — что из этого
        собрался ответ. На отказе второй половины не будет вовсе.
        """
        question = question.strip()
        if not question:
            raise ValueError("Пустую реплику задавать нечего.")

        resolved, context = await self.prepare(question)
        weight = self.weight()
        verdict = self.inspect(context, threshold)

        yield "context", {
            "resolve": resolved.as_dict(),
            "gate": verdict.as_dict(),
            "context": context.as_dict(),
        }

        if not verdict.passed:
            done = await self.remember(
                self.stopped(question, resolved, context, verdict, weight)
            )
            yield "turn", done.as_dict()
            return

        reply: llm.Reply | None = None
        async for piece in llm.stream(
            self.request(question, context, resolved),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            json=True,
        ):
            if piece.reply is not None:
                reply = piece.reply
            elif piece.content:
                yield "delta", piece.content

        if reply is None:
            raise RuntimeError("Поток кончился без итогового кадра: токены не посчитаны.")

        done = await self.remember(
            self.assemble(question, resolved, context, reply, verdict, weight)
        )
        yield "turn", done.as_dict()


async def run(
    storage: Storage,
    mode: str,
    replies: list[str],
    threshold: float | None = None,
    title: str = "",
) -> tuple[Dialog, list[Turn]]:
    """Сценарий целиком, ход за ходом. Параллелить нельзя: ходы зависят друг от друга."""
    dialog = await Dialog.open(storage, mode, title)
    turns = [await dialog.ask(question, threshold) for question in replies]
    return dialog, turns
