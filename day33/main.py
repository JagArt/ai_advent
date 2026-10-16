"""Веб-слой ассистента поддержки: список тикетов, карточка, чат с агентом потоком.

MCP-сессия одна на процесс: `lifespan` поднимает сервер подпроцессом при старте
и закрывает при остановке. Через эту же сессию страница получает список тикетов
и карточку — тем же `agent.call`, что и модель, так что у оператора и у модели
один источник данных и один путь к нему.

Кадры потока `/api/ask` идут в порядке, в котором происходит работа:

    tool      вызов инструмента: имя, аргументы, миллисекунды, отказ
    nudge     первый раунд прошёл без вызовов — черновик отброшен, модели дан толчок
    reasoning рассуждения локальной модели, если пришли
    data      текст ответа по кускам
    done      итог: текст, источники, раунды, вызовы, толчок, замеры
    error     модель недоступна или сервер не ответил

Сервер без состояния: историю диалога по тикету держит страница и присылает с
каждым вопросом — только вопросы оператора и тексты ответов.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

import agent
import crm
import kb
import llm

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
NO_CACHE = {"Cache-Control": "no-cache"}
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

PRESETS = (
    "Почему не работает авторизация?",
    "В чём причина обращения и что делать оператору?",
    "Составь ответ клиенту",
    "Были ли похожие обращения и чем они кончились?",
)


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "no-cache"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with agent.connect() as connection:
        app.state.connection = connection
        yield


app = FastAPI(title="Планёрка · ассистент поддержки", lifespan=lifespan)
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    ticket_id: str = Field(min_length=3)
    question: str = Field(min_length=1)
    provider: str = Field(default="local", pattern="^(local|cloud)$")
    history: list[Turn] = Field(default_factory=list)


def connection() -> agent.Connection:
    return app.state.connection


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/api/status")
async def status() -> dict:
    availability = await llm.probe()
    return {
        **availability.as_dict(),
        "mcp": connection().as_dict(),
        "kb": await run_in_threadpool(kb.reindex),
        "crm": await run_in_threadpool(crm.stats),
        "presets": PRESETS,
    }


@app.get("/api/tickets")
async def tickets() -> dict:
    result = await agent.call(connection().session, "crm_list_tickets", {"limit": crm.MAX_LIMIT})
    if not result.ok:
        raise HTTPException(502, result.text)
    users = {user["id"]: user for user in crm.load()["users"]}
    for ticket in result.payload["tickets"]:
        user = users.get(ticket["user_id"], {})
        ticket["user_name"] = user.get("name")
        ticket["user_plan"] = user.get("plan")
    return result.payload


@app.get("/api/tickets/{ticket_id}")
async def ticket(ticket_id: str) -> dict:
    session = connection().session
    found = await agent.call(session, "crm_get_ticket", {"ticket_id": ticket_id})
    if not found.ok:
        raise HTTPException(404, found.text)
    user = await agent.call(session, "crm_get_user", {"user": found.payload["user_id"]})
    history = await agent.call(session, "crm_list_tickets", {"user": found.payload["user_id"], "limit": 10})
    return {
        "ticket": found.payload,
        "user": user.payload if user.ok else {"error": user.text},
        "history": history.payload["tickets"] if history.ok else [],
    }


def sse_frame(data: object, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def event_stream(request: AskRequest) -> AsyncIterator[str]:
    try:
        llm.check(request.provider, await llm.probe())
        found = await agent.call(connection().session, "crm_get_ticket", {"ticket_id": request.ticket_id})
        if not found.ok:
            raise RuntimeError(found.text)

        events = agent.answer(
            connection(),
            question=request.question,
            ticket_id=found.payload["id"],
            user_id=found.payload["user_id"],
            provider=request.provider,
            history=[turn.model_dump() for turn in request.history],
        )
        async for event in events:
            if isinstance(event, agent.Chunk):
                yield sse_frame(event.text)
            elif isinstance(event, agent.Reasoning):
                yield sse_frame(event.text, event="reasoning")
            elif isinstance(event, agent.ToolCalled):
                yield sse_frame({"round": event.round, **event.result.as_dict()}, event="tool")
            elif isinstance(event, agent.Nudged):
                yield sse_frame({"round": event.round, "draft": event.draft}, event="nudge")
            elif isinstance(event, agent.Cleared):
                yield sse_frame({"round": event.round}, event="clear")
            elif isinstance(event, agent.Done):
                yield sse_frame(event.as_dict(), event="done")
    except Exception as exc:  # noqa: BLE001
        yield sse_frame(f"{type(exc).__name__}: {exc}" if not isinstance(exc, RuntimeError) else str(exc), event="error")


@app.post("/api/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    request.question = request.question.strip()
    return StreamingResponse(event_stream(request), media_type="text/event-stream", headers=SSE_HEADERS)
