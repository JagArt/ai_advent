import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

import facts as facts_policy
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
    AgentFactsUpdate,
    AgentPlan,
    AgentTurn,
    ContextOverflow,
)
from llm import MODEL
from memory import (
    DEFAULT_STRATEGY,
    STRATEGIES,
    WINDOW_MESSAGES,
    WINDOW_OPTIONS,
)
from runner import Finished, Line, Progress, ScenarioBusy, ScenarioRunner
from scenarios import SCENARIOS
from storage import Storage

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
NO_CACHE = {"Cache-Control": "no-cache"}
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
MAX_SESSIONS = 100

# Окно шире этого числа реплик — уже не стратегия, а вся история: шкала на странице
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
runner = ScenarioRunner()


class AskRequest(BaseModel):
    session_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    context_budget: int = Field(default=CONTEXT_BUDGET, ge=MIN_BUDGET, le=tokens.MODEL_CONTEXT_LIMIT)
    strategy: str = DEFAULT_STRATEGY
    window_messages: int = Field(default=WINDOW_MESSAGES, ge=1, le=MAX_WINDOW)


class ForkRequest(BaseModel):
    """Checkpoint — реплика в ленте: от неё и пойдёт новая ветка."""

    message_id: int = Field(ge=1)
    name: str = ""


def checked(strategy: str) -> str:
    if strategy not in STRATEGIES:
        raise HTTPException(status_code=422, detail=f"Неизвестная стратегия: {strategy}")
    return strategy


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/scenarios")
async def scenarios_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "scenarios.html", headers=NO_CACHE)


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
        "strategy": DEFAULT_STRATEGY,
        "strategies": list(STRATEGIES),
        "window_messages": WINDOW_MESSAGES,
        "window_options": list(WINDOW_OPTIONS),
        "facts_limit": facts_policy.FACTS_LIMIT,
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
    strategy: str = Query(default=DEFAULT_STRATEGY),
    window_messages: int = Query(default=WINDOW_MESSAGES, ge=1, le=MAX_WINDOW),
) -> dict[str, Any]:
    agent = await registry.get(session_id)
    # Стратегия и её настройки приходят со страницы: что попадёт в окно — только
    # последние реплики или ещё и картотека, — зависит от них, а не от умолчаний.
    plan = await agent.plan(
        budget=context_budget,
        strategy=checked(strategy),
        window=window_messages,
    )
    return await snapshot(agent, plan)


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str) -> dict[str, str]:
    if not await storage.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    await storage.delete_session(session_id)
    registry.forget(session_id)
    return {"session_id": session_id}


@app.post("/api/session/{session_id}/branch")
async def fork(
    session_id: str,
    request: ForkRequest,
    context_budget: int = Query(
        default=CONTEXT_BUDGET,
        ge=MIN_BUDGET,
        le=tokens.MODEL_CONTEXT_LIMIT,
    ),
    strategy: str = Query(default=DEFAULT_STRATEGY),
    window_messages: int = Query(default=WINDOW_MESSAGES, ge=1, le=MAX_WINDOW),
) -> dict[str, Any]:
    agent = await registry.get(session_id)
    try:
        await agent.fork(request.message_id, request.name)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    # Ветка создана и уже активна: страница получает её ленту тем же ответом,
    # чтобы не запрашивать состояние вторым запросом.
    return await snapshot(
        agent,
        await agent.plan(budget=context_budget, strategy=checked(strategy), window=window_messages),
    )


@app.post("/api/session/{session_id}/branch/{branch_id}")
async def switch_branch(
    session_id: str,
    branch_id: int,
    context_budget: int = Query(
        default=CONTEXT_BUDGET,
        ge=MIN_BUDGET,
        le=tokens.MODEL_CONTEXT_LIMIT,
    ),
    strategy: str = Query(default=DEFAULT_STRATEGY),
    window_messages: int = Query(default=WINDOW_MESSAGES, ge=1, le=MAX_WINDOW),
) -> dict[str, Any]:
    agent = await registry.get(session_id)
    try:
        await agent.switch(branch_id)
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return await snapshot(
        agent,
        await agent.plan(budget=context_budget, strategy=checked(strategy), window=window_messages),
    )


async def snapshot(agent: Agent, plan: AgentPlan) -> dict[str, Any]:
    """Всё, что нужно странице для показа ветки: лента, счёт, картотека, ветки."""
    return {
        "branch_id": agent.branch_id,
        "history": await agent.transcript(),
        "turns": await agent.turns(),
        "facts": await agent.facts(),
        "branches": await agent.branches(),
        **context(plan),
    }


def context(plan: AgentPlan) -> dict[str, Any]:
    """Состояние памяти: история ветки, окно запроса и картотека рядом с ним."""
    return {
        "strategy": plan.strategy,
        "window_messages": plan.window_messages,
        "history_size": plan.history_messages,
        "history_tokens": plan.history_tokens,
        "context_size": plan.context_messages,
        "context_tokens": plan.context_tokens,
        "context_budget": plan.budget,
        "facts_size": plan.facts_count,
        "facts_tokens": plan.facts_tokens,
    }


def sse_frame(data: Any, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def event_stream(
    agent: Agent,
    events: AsyncIterator[AgentEvent],
    budget: int,
    strategy: str,
    window: int,
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
            elif isinstance(event, AgentFactsUpdate):
                # Разбор реплики на факты идёт после ответа: он уже на экране, а
                # отдельный кадр показывает, за что списаны лишние токены.
                yield sse_frame(
                    {
                        "added": event.added,
                        "changed": event.changed,
                        "removed": event.removed,
                        "facts": [{"key": key, "value": value} for key, value in event.items],
                        "facts_size": len(event.items),
                        "facts_tokens": event.tokens,
                        "prompt_tokens": event.usage.prompt_tokens,
                        "completion_tokens": event.usage.completion_tokens,
                        "cost_usd": event.usage.cost_usd,
                    },
                    event="facts",
                )
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

    # Заголовок диалога и разбор фактов — тоже запросы к модели, поэтому таблица
    # ходов и итоги берутся из базы: там уже лежат все строки этого хода.
    yield sse_frame(
        {
            "finish_reason": finish_reason,
            "facts": await agent.facts(),
            "branches": await agent.branches(),
            "word_count": len("".join(chunks).split()),
            "estimated_tokens": turn.estimated_tokens,
            "prompt_tokens": turn.usage.prompt_tokens,
            "cached_tokens": turn.usage.cached_tokens,
            "completion_tokens": turn.usage.completion_tokens,
            "total_tokens": turn.usage.total_tokens,
            "cost_usd": turn.usage.cost_usd,
            "drift_percent": turn.drift_percent,
            # id реплик: только что сказанное тоже может стать checkpoint.
            "question_id": turn.question_id,
            "answer_id": turn.answer_id,
            "turns": await agent.turns(),
            **context(await agent.plan(budget=budget, strategy=strategy, window=window)),
        },
        event="done",
    )


@app.post("/api/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    agent = await registry.get(request.session_id)
    strategy = checked(request.strategy)
    return StreamingResponse(
        event_stream(
            agent,
            agent.ask(request.prompt.strip(), request.context_budget, strategy, request.window_messages),
            request.context_budget,
            strategy,
            request.window_messages,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.get("/api/scenarios")
async def scenarios() -> dict[str, Any]:
    """Что можно прогнать со страницы и во сколько запросов это встанет."""
    return {
        "busy": runner.busy,
        "scenarios": [
            {
                "name": name,
                "title": scenario.title,
                "about": scenario.about,
                "requests": scenario.requests,
            }
            for name, scenario in SCENARIOS.items()
        ],
    }


async def scenario_stream(name: str) -> AsyncIterator[str]:
    """Вывод сценария кадрами: отчёт строками, ход работы — отдельным событием."""
    lines = 0
    try:
        async for output in runner.stream():
            if isinstance(output, Line):
                lines += 1
                # Строка отчёта уходит как есть: собирать markdown будет страница,
                # ей же он нужен целиком для копирования в README.
                yield sse_frame(output.text)
            elif isinstance(output, Progress):
                yield sse_frame({"text": output.text}, event="progress")
            elif isinstance(output, Finished):
                yield sse_frame(
                    {
                        "exit_code": output.exit_code,
                        "seconds": round(output.seconds, 1),
                        "lines": lines,
                        "scenario": name,
                    },
                    event="done",
                )
    except Exception as exc:
        yield sse_frame({"message": str(exc), "kind": "runner"}, event="error")


@app.post("/api/scenario/{name}")
async def run_scenario(name: str) -> StreamingResponse:
    if name not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Неизвестный сценарий: {name}")

    # Процесс поднимается до ответа: занятый сервер должен вернуть статус, а не
    # ошибку посреди потока, который страница уже начала читать.
    try:
        await runner.start(name)
    except ScenarioBusy as error:
        raise HTTPException(status_code=409, detail=str(error)) from error

    return StreamingResponse(
        scenario_stream(name),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
