import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

import remember
from agent import (
    DEFAULT_AUTOSAVE,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    SYSTEM_PROMPT,
    Agent,
    AgentDelta,
    AgentEvent,
    AgentPlan,
    AgentProposals,
    AgentTurn,
)
from demo import DIALOG
from llm import MODEL
from memory import (
    SECTIONS,
    TARGETS,
    TIERS,
    WINDOW_MESSAGES,
    WINDOW_OPTIONS,
)
from storage import Storage

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

    async def create(self) -> str:
        session_id = await self._storage.create_session()
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
        # Разделы задаёт сервер: по ним раскладывается память и из них собраны
        # выпадающие списки в карточках.
        "sections": {tier: list(sections) for tier, sections in SECTIONS.items()},
        "proposal_limit": remember.ITEM_LIMIT,
        # Реплики автопрогона: те же, что в замере. Страница печатает их за
        # пользователя, поэтому ничего, кроме готового списка, ей не нужно.
        "autorun": list(DIALOG),
    }


@app.post("/api/session")
async def create_session() -> dict[str, str]:
    return {"session_id": await registry.create()}


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
    """Сброс к исходному: девять пунктов seed, диалогов нет, страница получит новый."""
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
    """Ответ на карточку: сохранить на уровне, перенести на другой или отклонить."""
    if request.target not in TARGETS:
        raise HTTPException(status_code=422, detail=f"Неизвестный уровень памяти: {request.target}")

    agent = await registry.get(session_id)
    try:
        memory = await agent.commit(request.proposal_id, request.target, request.section, request.text)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    # Блок памяти уходит в каждый запрос, поэтому вместе с ним пересобирается и
    # состав контекста: страница сразу видит, что теперь знает агент.
    return {**memory, **context(await agent.plan(window=window_messages))}


@app.delete("/api/session/{session_id}/memory/{tier}/{item_id}")
async def forget(
    session_id: str,
    tier: str,
    item_id: int,
    window_messages: int = Query(default=WINDOW_MESSAGES, ge=1, le=MAX_WINDOW),
) -> dict[str, Any]:
    """Удаление пункта. Уровень стоит в пути: стереть можно только адресно."""
    if tier not in TIERS:
        raise HTTPException(status_code=422, detail=f"Неизвестный уровень памяти: {tier}")

    agent = await registry.get(session_id)
    try:
        memory = await agent.forget(tier, item_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return {**memory, **context(await agent.plan(window=window_messages))}


async def snapshot(agent: Agent, plan: AgentPlan) -> dict[str, Any]:
    """Всё, что нужно странице: лента, оба блока памяти и очередь карточек."""
    return {
        "history": await agent.transcript(),
        **await agent.memory(),
        **context(plan),
    }


def context(plan: AgentPlan) -> dict[str, Any]:
    """Состояние памяти по уровням: что из неё попало в этот запрос."""
    return {
        "window_messages": plan.window_messages,
        "history_size": plan.history_messages,
        "context_size": plan.context_messages,
        "longterm_size": plan.longterm_count,
        "working_size": plan.working_count,
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
            elif isinstance(event, AgentProposals):
                # Разбор хода идёт после ответа: он уже на экране, а отдельный кадр
                # показывает находки этого хода.
                yield sse_frame(
                    {
                        # Находки этого хода и очередь целиком — разные числа:
                        # карточка прошлого хода могла так и остаться без ответа.
                        "found": len(event.items),
                        "saved": event.saved,
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
