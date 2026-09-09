import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

import tokens
from agent import (
    BUDGET_OPTIONS,
    CONTEXT_BUDGET,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    MAX_TOKENS,
    MIN_BUDGET,
    SYSTEM_PROMPT,
    Agent,
    AgentDelta,
    AgentEvent,
    AgentPlan,
    AgentTurn,
    ContextOverflow,
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
        # заново вокруг той же строки в базе и поднимает историю из неё.
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
    context_budget: int = Field(default=CONTEXT_BUDGET, ge=MIN_BUDGET, le=tokens.MODEL_CONTEXT_LIMIT)


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
        "context_budget": CONTEXT_BUDGET,
        "budget_options": list(BUDGET_OPTIONS),
        "model_context_limit": tokens.MODEL_CONTEXT_LIMIT,
        "encoding": tokens.ENCODING,
        "peak": tokens.is_peak_hour(),
        "pricing": {
            "input_miss": tokens.PRICING.input_miss,
            "input_hit": tokens.PRICING.input_hit,
            "output": tokens.PRICING.output,
            "multiplier": tokens.price_multiplier(),
        },
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
    context_budget: int = Query(
        default=CONTEXT_BUDGET,
        ge=MIN_BUDGET,
        le=tokens.MODEL_CONTEXT_LIMIT,
    ),
) -> dict[str, Any]:
    agent = await registry.get(session_id)
    # Бюджет приходит со страницы: сколько сообщений влезет в окно, зависит от него,
    # поэтому окно считается на текущей настройке, а не на дефолтной.
    plan = await agent.plan(budget=context_budget)
    return {
        "history": await agent.transcript(),
        "turns": await agent.turns(),
        **memory(plan),
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    if not await storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    await storage.delete_session(session_id)
    registry.forget(session_id)
    return {"session_id": session_id}


def memory(plan: AgentPlan) -> dict[str, Any]:
    """Состояние памяти: вся история — одно число, окно контекста — другое."""
    return {
        "history_size": plan.history_messages,
        "history_tokens": plan.history_tokens,
        "context_size": plan.context_messages,
        "context_tokens": plan.context_tokens,
        "context_budget": plan.budget,
    }


def sse_frame(data: Any, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def event_stream(
    agent: Agent,
    events: AsyncIterator[AgentEvent],
    budget: int,
) -> AsyncIterator[str]:
    chunks: list[str] = []
    finish_reason: str | None = None
    turn: AgentTurn | None = None

    try:
        async for event in events:
            if isinstance(event, AgentPlan):
                # Кадр до первого токена: цена запроса известна раньше ответа.
                yield sse_frame(
                    {
                        "question_tokens": event.question_tokens,
                        "request_tokens": event.request_tokens,
                        "dropped_messages": event.dropped_messages,
                        "over_budget": event.over_budget,
                        **memory(event),
                    },
                    event="prompt",
                )
            elif isinstance(event, AgentDelta):
                if event.content:
                    chunks.append(event.content)
                    yield sse_frame(event.content)
                if event.finish_reason:
                    finish_reason = event.finish_reason
            else:
                turn = event
    except ContextOverflow as exc:
        yield sse_frame(
            {
                "message": str(exc),
                "kind": "overflow",
                "request_tokens": exc.request_tokens,
                "max_tokens": exc.max_tokens,
                "limit": exc.limit,
            },
            event="error",
        )
        return
    except Exception as exc:
        yield sse_frame({"message": str(exc), "kind": "api"}, event="error")
        return

    if turn is None:
        yield sse_frame({"message": "Модель не вернула ответ", "kind": "empty"}, event="error")
        return

    # Заголовок диалога — тоже запрос к модели, поэтому таблица ходов и итоги
    # берутся из базы: там уже лежат обе строки этого хода.
    turns = await agent.turns()
    yield sse_frame(
        {
            "finish_reason": finish_reason,
            "word_count": len("".join(chunks).split()),
            "estimated_tokens": turn.estimated_tokens,
            "prompt_tokens": turn.usage.prompt_tokens,
            "cached_tokens": turn.usage.cached_tokens,
            "completion_tokens": turn.usage.completion_tokens,
            "total_tokens": turn.usage.total_tokens,
            "cost_usd": turn.usage.cost_usd,
            "drift_percent": turn.drift_percent,
            "turns": turns,
            **memory(await agent.plan(budget=budget)),
        },
        event="done",
    )


@app.post("/api/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    agent = await registry.get(request.session_id)
    return StreamingResponse(
        event_stream(
            agent,
            agent.ask(request.prompt.strip(), request.context_budget),
            request.context_budget,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
