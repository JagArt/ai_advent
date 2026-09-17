import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

import invariants
import remember
import shape
from agent import (
    DEFAULT_AUTOSAVE,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    SYSTEM_PROMPT,
    Agent,
    AgentBreak,
    AgentDelta,
    AgentEvent,
    AgentPlan,
    AgentProposals,
    AgentTurn,
    answer_as,
)
from demo import COMPARE, DIALOG, RUN_PROFILE
from invariants import GLOBAL, KINDS, SCOPES, TASK
from llm import MODEL
from memory import (
    LONGTERM,
    TIERS,
    WINDOW_MESSAGES,
    WINDOW_OPTIONS,
    block,
)
from profile import PROFILE, SCALES, build as build_profile
from remember import SECTIONS_BY_TARGET, TARGETS
from seed import DEFAULT_PROFILE
from storage import Storage
from task import STAGES

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
NO_CACHE = {"Cache-Control": "no-cache"}
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
MAX_SESSIONS = 100

# Окно шире этого числа реплик — уже не окно, а вся лента: шкала на странице
# кончается на двадцати, а потолок нужен, чтобы запрос не собирался вслепую.
MAX_WINDOW = 200


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "no-cache"
        return response


class AgentRegistry:
    """Кэш живых агентов: переписка принадлежит базе, объект можно создать заново."""

    def __init__(self, storage: Storage, max_sessions: int = MAX_SESSIONS) -> None:
        self._storage = storage
        self._agents: dict[str, Agent] = {}
        self._max_sessions = max_sessions

    async def create(self, profile_id: str = DEFAULT_PROFILE) -> str:
        session_id = await self._storage.create_session(profile_id)
        self._remember(Agent(session_id, self._storage))
        return session_id

    async def get(self, session_id: str) -> Agent:
        agent = self._agents.get(session_id)
        if agent is not None:
            # Порядок ключей — очередь на вытеснение, обращение возвращает сессию в конец.
            self._agents[session_id] = self._agents.pop(session_id)
            return agent

        # Процесс мог перезапуститься или сессию вытеснили из кэша: агент собирается
        # заново вокруг той же строки в базе и поднимает память из неё. Нерешённые
        # карточки при этом теряются — они и были краткосрочной памятью.
        if not await self._storage.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Сессия не найдена")

        agent = Agent(session_id, self._storage)
        self._remember(agent)
        return agent

    def forget(self, session_id: str) -> None:
        self._agents.pop(session_id, None)

    async def reset(self) -> str:
        """База к seed, живые агенты — тоже: у них в памяти старые блоки и карточки."""
        await self._storage.reset()
        self._agents.clear()
        return await self.create()

    def _remember(self, agent: Agent) -> None:
        self._agents[agent.session_id] = agent
        while len(self._agents) > self._max_sessions:
            self._agents.pop(next(iter(self._agents)))


storage = Storage()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await storage.init()
    yield


app = FastAPI(title="DeepSeek Web", lifespan=lifespan)
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")
registry = AgentRegistry(storage)


class AskRequest(BaseModel):
    session_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    window_messages: int = Field(default=WINDOW_MESSAGES, ge=1, le=MAX_WINDOW)
    # Кто решает, куда класть находки: по умолчанию пользователь кнопкой.
    autosave: bool = DEFAULT_AUTOSAVE


class CommitRequest(BaseModel):
    """Решение по карточке: куда её, в каком разделе и с какой формулировкой."""

    proposal_id: int = Field(ge=1)
    target: str
    section: str = ""
    text: str = ""


class ScaleRequest(BaseModel):
    """Правка шкалы профиля: имя шкалы и значение из её списка."""

    scale: str
    value: str


class MoveRequest(BaseModel):
    """Перевод задачи вручную: либо этап, либо шаг — но не оба сразу."""

    stage: str = ""
    step: str = ""


class DecisionRequest(BaseModel):
    """Решение по карточке перехода: применить или откатить. Третьего у перехода нет."""

    proposal_id: int = Field(ge=1)
    apply: bool


class InvariantRequest(BaseModel):
    """Новый инвариант: своими словами или из пункта ТЗ.

    banned приходит списком строк, а не одной строкой: паттерн — не текст правила, и
    склеивать их в одно поле значило бы разбирать его обратно на сервере.
    """

    text: str = ""
    kind: str = ""
    scope: str = TASK
    instead: str = ""
    banned: list[str] = Field(default_factory=list)
    # Пункт рабочей памяти, из которого растёт инвариант задачи.
    item_id: int = Field(default=0, ge=0)


class RelaxRequest(BaseModel):
    """Снять инвариант или вернуть его в силу. Решение только пользователя."""

    enabled: bool


class CompareRequest(BaseModel):
    prompt: str = Field(min_length=1)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/api/defaults")
async def defaults() -> dict[str, Any]:
    return {
        "system_prompt": SYSTEM_PROMPT,
        "prompt": DEFAULT_PROMPT,
        "model": MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "window_messages": WINDOW_MESSAGES,
        "window_options": list(WINDOW_OPTIONS),
        "autosave": DEFAULT_AUTOSAVE,
        # Разделы задаёт сервер: по ним раскладывается память и профиль, и из них
        # собраны выпадающие списки в карточках. Профиль здесь равноправен с
        # уровнями памяти — карточка не знает, чем они отличаются.
        "sections": {target: list(sections) for target, sections in SECTIONS_BY_TARGET.items()},
        # Шкалы со всеми значениями: страница рисует по селектору на шкалу, а
        # список значений задаёт сервер — своё значение в шкалу не придумать.
        "scales": [
            {"key": scale.key, "about": scale.about, "options": list(scale.options)}
            for scale in SCALES
        ],
        # Профили для селектора. Приходят вместе с остальными настройками: они есть
        # до первого диалога, и отдельного запроса за ними идти незачем.
        "profiles": await storage.list_profiles(),
        # Автомат целиком: этапы по порядку и шаги внутри них. Страница рисует по
        # нему полосу этапов и список шагов, но состояние в него не подставляет —
        # где мы сейчас, приходит снимком сессии. Так граф остаётся описанием, а не
        # ещё одним местом, где хранится текущий этап.
        "stages": [
            {
                "key": stage.key,
                "name": stage.name,
                "about": stage.about,
                "steps": [
                    {"key": step.key, "expects": step.expects} for step in stage.steps
                ],
            }
            for stage in STAGES
        ],
        # Виды и уровни инвариантов: по ним страница рисует форму и группирует панель.
        # Список задаёт сервер по той же причине, что и разделы памяти: своего вида в
        # инвариант не придумать, иначе список перестаёт читаться сверху вниз.
        "kinds": list(KINDS),
        "scopes": [{"key": scope, "name": invariants.SCOPE_NAMES[scope]} for scope in SCOPES],
        "proposal_limit": remember.ITEM_LIMIT,
        # Реплики автопрогона: те же, что в замере. Страница печатает их за
        # пользователя, поэтому ничего, кроме готового списка, ей не нужно.
        "autorun": list(DIALOG),
        # Профиль, на который прогон переключается: выбран по длине ответов, чтобы за
        # ходом автомата было видно сам ход, а не полотно текста.
        "autorun_profile": RUN_PROFILE,
        # Вопросы для сравнения профилей: тоже те же, что в замере.
        "compare": list(COMPARE),
    }


@app.post("/api/session")
async def create_session(profile_id: str = Query(default=DEFAULT_PROFILE)) -> dict[str, str]:
    """Новый диалог всегда от чьего-то лица: профиль выбирается до первой реплики."""
    if await storage.load_profile(profile_id) is None:
        raise HTTPException(status_code=422, detail=f"Профиля {profile_id} нет")
    return {"session_id": await registry.create(profile_id)}


@app.get("/api/sessions")
async def sessions() -> dict[str, list[dict[str, Any]]]:
    return {"sessions": await storage.list_sessions()}


@app.get("/api/session/{session_id}")
async def session(
    session_id: str,
    window_messages: int = Query(default=WINDOW_MESSAGES, ge=1, le=MAX_WINDOW),
) -> dict[str, Any]:
    agent = await registry.get(session_id)
    return await snapshot(agent, await agent.plan(window=window_messages))


@app.post("/api/reset")
async def reset() -> dict[str, str]:
    """Сброс к исходному: seed памяти и профилей, диалогов нет, страница получит новый."""
    return {"session_id": await registry.reset()}


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    if not await storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    # Уходит диалог и его рабочая память; долговременная остаётся — она не
    # принадлежала этой задаче.
    await storage.delete_session(session_id)
    registry.forget(session_id)
    return {"session_id": session_id}


@app.post("/api/session/{session_id}/memory")
async def commit(
    session_id: str,
    request: CommitRequest,
    window_messages: int = Query(default=WINDOW_MESSAGES, ge=1, le=MAX_WINDOW),
) -> dict[str, Any]:
    """Ответ на карточку: сохранить по адресу, перенести на другой или отклонить."""
    if request.target not in TARGETS:
        raise HTTPException(status_code=422, detail=f"Неизвестный адрес записи: {request.target}")

    agent = await registry.get(session_id)
    try:
        memory = await agent.commit(request.proposal_id, request.target, request.section, request.text)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    # Профиль и блоки памяти уходят в каждый запрос, поэтому вместе с ними
    # пересобирается и состав контекста: страница сразу видит, что теперь знает агент.
    return {**memory, **context(await agent.plan(window=window_messages))}


@app.delete("/api/session/{session_id}/memory/{tier}/{item_id}")
async def forget(
    session_id: str,
    tier: str,
    item_id: int,
    window_messages: int = Query(default=WINDOW_MESSAGES, ge=1, le=MAX_WINDOW),
) -> dict[str, Any]:
    """Удаление пункта памяти. Уровень стоит в пути: стереть можно только адресно."""
    if tier not in TIERS:
        raise HTTPException(status_code=422, detail=f"Неизвестный уровень памяти: {tier}")

    agent = await registry.get(session_id)
    try:
        memory = await agent.forget(tier, item_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return {**memory, **context(await agent.plan(window=window_messages))}


@app.patch("/api/session/{session_id}/profile")
async def switch_profile(
    session_id: str,
    profile_id: str = Query(min_length=1),
    window_messages: int = Query(default=WINDOW_MESSAGES, ge=1, le=MAX_WINDOW),
) -> dict[str, Any]:
    """Сменить собеседника у диалога: память остаётся, форма ответов меняется."""
    agent = await registry.get(session_id)
    try:
        memory = await agent.switch(profile_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return {**memory, **context(await agent.plan(window=window_messages))}


@app.put("/api/session/{session_id}/profile/scale")
async def tune_profile(
    session_id: str,
    request: ScaleRequest,
    window_messages: int = Query(default=WINDOW_MESSAGES, ge=1, le=MAX_WINDOW),
) -> dict[str, Any]:
    """Правка шкалы: у профиля одно значение на шкалу, поэтому это PUT, а не POST."""
    agent = await registry.get(session_id)
    try:
        memory = await agent.tune(request.scale, request.value)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {**memory, **context(await agent.plan(window=window_messages))}


@app.delete("/api/session/{session_id}/profile/item/{item_id}")
async def forget_preference(
    session_id: str,
    item_id: int,
    window_messages: int = Query(default=WINDOW_MESSAGES, ge=1, le=MAX_WINDOW),
) -> dict[str, Any]:
    """Удаление свободного пункта профиля. Шкала так не удаляется: она есть всегда."""
    agent = await registry.get(session_id)
    try:
        memory = await agent.forget(PROFILE, item_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return {**memory, **context(await agent.plan(window=window_messages))}


@app.put("/api/session/{session_id}/task")
async def move_task(
    session_id: str,
    request: MoveRequest,
    window_messages: int = Query(default=WINDOW_MESSAGES, ge=1, le=MAX_WINDOW),
) -> dict[str, Any]:
    """Перевести задачу вручную. PUT, а не POST: у диалога одно состояние, не список.

    Условие перехода проверяется и здесь, хотя пришёл сюда живой человек: автомат
    формальный, и невалидным его состояние не сделает никто. Отказ приходит 422 с
    причиной — той же строкой, которая стоит в панели у погашенной кнопки.
    """
    if bool(request.stage) == bool(request.step):
        raise HTTPException(status_code=422, detail="Нужен ровно один переход: этап или шаг")

    agent = await registry.get(session_id)
    try:
        state = await agent.move(stage=request.stage, step=request.step)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {**state, **context(await agent.plan(window=window_messages))}


@app.post("/api/session/{session_id}/task/decision")
async def decide_move(
    session_id: str,
    request: DecisionRequest,
    window_messages: int = Query(default=WINDOW_MESSAGES, ge=1, le=MAX_WINDOW),
) -> dict[str, Any]:
    """Ответ на карточку перехода: применить или откатить применённый."""
    agent = await registry.get(session_id)
    try:
        state = await agent.decide(request.proposal_id, request.apply)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {**state, **context(await agent.plan(window=window_messages))}


@app.get("/api/session/{session_id}/transitions")
async def transitions(session_id: str) -> dict[str, Any]:
    """Журнал переходов: не «где задача», а «как она сюда пришла», включая откаты."""
    if not await storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return {"transitions": await storage.load_transitions(session_id)}


@app.post("/api/session/{session_id}/invariants")
async def constrain(
    session_id: str,
    request: InvariantRequest,
    window_messages: int = Query(default=WINDOW_MESSAGES, ge=1, le=MAX_WINDOW),
) -> dict[str, Any]:
    """Завести инвариант: своими словами или зафиксировав пункт ТЗ.

    Путь идёт через сессию, хотя общий инвариант ей не принадлежит: заводят его всегда
    изнутри разговора, а живёт он потом отдельно от него — и в базе это видно по
    отсутствию session_id у общего уровня.
    """
    if request.scope not in SCOPES:
        raise HTTPException(status_code=422, detail=f"Неизвестный уровень инварианта: {request.scope}")
    if not request.text.strip() and not request.item_id:
        raise HTTPException(status_code=422, detail="Нужна формулировка или пункт ТЗ")

    agent = await registry.get(session_id)
    try:
        state = await agent.constrain(
            text=request.text,
            kind=request.kind,
            scope=request.scope,
            instead=request.instead,
            banned=request.banned,
            item_id=request.item_id,
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {**state, **context(await agent.plan(window=window_messages))}


@app.patch("/api/session/{session_id}/invariants/{invariant_id}")
async def relax(
    session_id: str,
    invariant_id: int,
    request: RelaxRequest,
    window_messages: int = Query(default=WINDOW_MESSAGES, ge=1, le=MAX_WINDOW),
) -> dict[str, Any]:
    """Отключить инвариант или вернуть его в силу. PATCH: меняется одно поле из многих."""
    agent = await registry.get(session_id)
    try:
        state = await agent.relax(invariant_id, request.enabled)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return {**state, **context(await agent.plan(window=window_messages))}


@app.delete("/api/session/{session_id}/invariants/{invariant_id}")
async def release(
    session_id: str,
    invariant_id: int,
    window_messages: int = Query(default=WINDOW_MESSAGES, ge=1, le=MAX_WINDOW),
) -> dict[str, Any]:
    """Удалить инвариант. Нарушения его в журнале остаются: они были."""
    agent = await registry.get(session_id)
    try:
        state = await agent.release(invariant_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return {**state, **context(await agent.plan(window=window_messages))}


@app.get("/api/session/{session_id}/violations")
async def violations(session_id: str) -> dict[str, Any]:
    """Журнал нарушений: не «что запрещено», а «где на это наткнулись и чем поймали»."""
    if not await storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return {"violations": await storage.load_violations(session_id)}


@app.post("/api/compare")
async def compare(request: CompareRequest) -> dict[str, Any]:
    """Один вопрос всем профилям сразу: ответы рядом, без диалога и без ТЗ.

    Проверка «ответы для разных профилей» в один клик. Запросы уходят тем же
    сборщиком, что и ход чата, но без ленты и рабочей памяти: разница между
    ответами должна объясняться профилем, а не тем, что у кого-то была история.
    Долговременная память общая и остаётся — она одинакова для всех профилей.

    Общие инварианты остаются тоже, и это не мелочь: вопрос про очередь задан ровно
    так, чтобы наткнуться на них, и три отказа рядом показывают, что рамки одни, а
    форма отказа у каждого профиля своя.
    """
    prompt = request.prompt.strip()
    longterm = block(LONGTERM, await storage.load_longterm())
    limits = invariants.block(GLOBAL, await storage.load_invariants(GLOBAL))
    profiles = await storage.list_profiles()

    async def ask_one(row: dict[str, Any]) -> dict[str, Any]:
        loaded = await storage.load_profile(str(row["id"]))
        person = build_profile(loaded) if loaded else None
        try:
            answer = await answer_as(person, longterm, limits, prompt)
        except Exception as error:  # noqa: BLE001 — ответ одного профиля не должен рушить сравнение
            return {"id": row["id"], "name": row["name"], "error": str(error)}

        return {
            "id": row["id"],
            "name": row["name"],
            "scales": person.as_dict()["scales"] if person else {},
            "answer": answer,
            # Мерки считает сервер: в таблице README и в панели браузера должны
            # стоять одни и те же числа.
            "shape": shape.measure(answer).as_dict(),
        }

    # Профили спрашиваются одновременно: ждать три ответа по очереди незачем, а
    # порядок в ответе сохраняет gather.
    return {"prompt": prompt, "answers": list(await asyncio.gather(*(ask_one(row) for row in profiles)))}


async def snapshot(agent: Agent, plan: AgentPlan) -> dict[str, Any]:
    """Всё, что нужно странице: лента, профиль, состояние, память и обе очереди карточек."""
    return {
        "history": await agent.transcript(),
        **await agent.memory(),
        **context(plan),
    }


def context(plan: AgentPlan) -> dict[str, Any]:
    """Что уходит в запрос: профиль и состояние целиком, память — по счётчику на уровень."""
    return {
        "window_messages": plan.window_messages,
        "history_size": plan.history_messages,
        "context_size": plan.context_messages,
        "longterm_size": plan.longterm_count,
        "working_size": plan.working_count,
        # Профиль в счётчиках не участвует: он уходит в запрос целиком и всегда,
        # поэтому в шапке стоит его имя, а не число пунктов.
        "profile_id": plan.profile_id,
        "profile_name": plan.profile_name,
        "profile_size": plan.profile_count,
        # Состояние — тем же способом и по той же причине: в шапке стоит пара
        # «этап · шаг», а не число, и рядом с ней то, чего агент ждёт от собеседника.
        "stage": plan.stage,
        "stage_name": plan.stage_name,
        "step": plan.step,
        "expected": plan.expected,
        # Инварианты — двумя числами: сколько рамок действует общих и сколько поставила
        # сама задача. Отключённые не считаются, потому что в запрос они не идут.
        "global_rules": plan.global_rules,
        "task_rules": plan.task_rules,
    }


def sse_frame(data: Any, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def event_stream(
    agent: Agent,
    events: AsyncIterator[AgentEvent],
    window: int,
) -> AsyncIterator[str]:
    chunks: list[str] = []
    finish_reason: str | None = None
    turn: AgentTurn | None = None

    try:
        async for event in events:
            if isinstance(event, AgentPlan):
                # Кадр до первого токена: состав запроса известен раньше ответа.
                yield sse_frame(
                    {
                        "dropped_messages": event.dropped_messages,
                        **context(event),
                    },
                    event="prompt",
                )
            elif isinstance(event, AgentDelta):
                if event.content:
                    chunks.append(event.content)
                    yield sse_frame(event.content)
                if event.finish_reason:
                    finish_reason = event.finish_reason
            elif isinstance(event, AgentBreak):
                # Ответ оборван детектором. Страница снимает показанный текст: то, что
                # уже на экране, принадлежит отброшенному ответу, и оставить его значило
                # бы показать половину нарушения. Молчание на месте кадра выглядело бы
                # как обрыв сети, поэтому кадр приходит вместе с причиной.
                chunks.clear()
                yield sse_frame(
                    {
                        "shown": event.shown,
                        "retry": event.retry,
                        "violations": [violation.as_dict() for violation in event.violations],
                    },
                    event="broken",
                )
            elif isinstance(event, AgentProposals):
                # Разбор хода идёт после ответа: он уже на экране, а отдельный кадр
                # показывает находки этого хода и предложенный переход.
                yield sse_frame(
                    {
                        # Находки этого хода и очередь целиком — разные числа:
                        # карточка прошлого хода могла так и остаться без ответа.
                        "found": len(event.items),
                        "saved": event.saved,
                        # Переход этого хода: id, чтобы страница нашла его карточку
                        # среди остальных, и причина отказа, если предложение не
                        # прошло по графу или по условию.
                        "moved": event.move.id if event.move else 0,
                        "applied": bool(event.move and event.move.applied),
                        "rejected": event.rejected,
                        # Нарушения, которые нашёл аудитор в уже отданном ответе, и
                        # находки, которые не стали карточками: пункт нарушал инвариант.
                        "caught": [violation.id for violation in event.violations],
                        "blocked": list(event.blocked),
                        # Снимок пришёл с событием: ask() ещё не отпустил агента,
                        # и спрашивать его о памяти прямо сейчас нельзя.
                        **event.memory,
                    },
                    event="memory",
                )
            else:
                turn = event
                # id реплик уходят сразу, а не в конце хода: карточки встают в
                # ленте под своим ответом, и к разбору хода страница уже должна
                # знать, какое сообщение чем оказалось.
                yield sse_frame(
                    {"question_id": turn.question_id, "answer_id": turn.answer_id},
                    event="turn",
                )
    except Exception as exc:
        yield sse_frame({"message": str(exc), "kind": "api"}, event="error")
        return

    if turn is None:
        yield sse_frame({"message": "Модель не вернула ответ", "kind": "empty"}, event="error")
        return

    yield sse_frame(
        {
            "finish_reason": finish_reason,
            "word_count": len("".join(chunks).split()),
            # id реплик хода: по ним страница отмечает границу окна в ленте.
            "question_id": turn.question_id,
            "answer_id": turn.answer_id,
            **await agent.memory(),
            **context(await agent.plan(window=window)),
        },
        event="done",
    )


@app.post("/api/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    agent = await registry.get(request.session_id)
    return StreamingResponse(
        event_stream(
            agent,
            agent.ask(
                request.prompt.strip(),
                request.window_messages,
                request.autosave,
            ),
            request.window_messages,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
