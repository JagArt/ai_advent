import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import uuid4

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
    """Агенты живут в процессе: память диалога принадлежит агенту, а не браузеру."""

    def __init__(self, max_sessions: int = MAX_SESSIONS) -> None:
        self._agents: dict[str, Agent] = {}
        self._max_sessions = max_sessions

    def create(self) -> str:
        session_id = uuid4().hex
        self._agents[session_id] = Agent()
        while len(self._agents) > self._max_sessions:
            self._agents.pop(next(iter(self._agents)))
        return session_id

    def get(self, session_id: str) -> Agent:
        agent = self._agents.get(session_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Сессия не найдена")
        # Порядок ключей — очередь на вытеснение, обращение возвращает сессию в конец.
        self._agents[session_id] = self._agents.pop(session_id)
        return agent


app = FastAPI(title="DeepSeek Web")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")
registry = AgentRegistry()


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
    return {"session_id": registry.create()}


@app.get("/api/session/{session_id}")
async def session(session_id: str) -> dict[str, Any]:
    agent = registry.get(session_id)
    return {"history": agent.transcript(), "history_size": agent.history_size}


@app.post("/api/session/{session_id}/reset")
async def reset_session(session_id: str) -> dict[str, int]:
    agent = registry.get(session_id)
    agent.reset()
    return {"history_size": agent.history_size}


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
                "history_size": agent.history_size,
            },
            ensure_ascii=False,
        ),
        event="done",
    )


@app.post("/api/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    agent = registry.get(request.session_id)
    return StreamingResponse(
        event_stream(agent, agent.ask(request.prompt.strip())),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
