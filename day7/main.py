import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from agent import (
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    HISTORY_LIMIT,
    MAX_TOKENS,
    SYSTEM_PROMPT,
    Agent,
    AgentDelta,
)
from llm import MODEL
from storage import Storage

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
NO_CACHE = {"Cache-Control": "no-cache"}
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
MAX_SESSIONS = 100


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
        # заново вокруг той же строки в базе и поднимает контекст из неё.
        if not await self._storage.session_exists(session_id):
            raise HTTPException(status_code=404, detail="Сессия не найдена")

        agent = Agent(session_id, self._storage)
        self._remember(agent)
        return agent

    def forget(self, session_id: str) -> None:
        self._agents.pop(session_id, None)

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
        "max_tokens": MAX_TOKENS,
        "history_limit": HISTORY_LIMIT,
    }


@app.post("/api/session")
async def create_session() -> dict[str, str]:
    return {"session_id": await registry.create()}


@app.get("/api/sessions")
async def sessions() -> dict[str, list[dict[str, Any]]]:
    return {"sessions": await storage.list_sessions()}


@app.get("/api/session/{session_id}")
async def session(session_id: str) -> dict[str, Any]:
    agent = await registry.get(session_id)
    return {
        "history": await agent.transcript(),
        "history_size": await agent.history_size(),
        "context_size": await agent.context_size(),
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    if not await storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    await storage.delete_session(session_id)
    registry.forget(session_id)
    return {"session_id": session_id}


def sse_frame(data: str, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n"


async def event_stream(agent: Agent, deltas: AsyncIterator[AgentDelta]) -> AsyncIterator[str]:
    chunks: list[str] = []
    finish_reason: str | None = None

    try:
        async for delta in deltas:
            if delta.content:
                chunks.append(delta.content)
                yield sse_frame(json.dumps(delta.content, ensure_ascii=False))
            if delta.finish_reason:
                finish_reason = delta.finish_reason
    except Exception as exc:
        yield sse_frame(json.dumps(str(exc), ensure_ascii=False), event="error")
        return

    yield sse_frame(
        json.dumps(
            {
                "finish_reason": finish_reason,
                "word_count": len("".join(chunks).split()),
                "history_size": await agent.history_size(),
                "context_size": await agent.context_size(),
            },
            ensure_ascii=False,
        ),
        event="done",
    )


@app.post("/api/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    agent = await registry.get(request.session_id)
    return StreamingResponse(
        event_stream(agent, agent.ask(request.prompt.strip())),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
