"""Веб-слой файлового ассистента: дерево песочницы, diff, чат с агентом потоком.

MCP-сессия одна на процесс: `lifespan` поднимает сервер подпроцессом при старте
и закрывает при остановке. Через эту же сессию страница не читает файлы —
дерево и превью идут напрямую из `workspace.py`, тем же jail, что и инструменты.

Кадры потока `/api/ask` идут в порядке, в котором происходит работа:

    tool      вызов инструмента: имя, аргументы, миллисекунды, отказ
    nudge     первый раунд прошёл без вызовов — черновик отброшен, модели дан толчок
    reasoning рассуждения локальной модели, если пришли
    data      текст ответа по кускам
    done      итог: текст, файлы, diff, раунды, вызовы, толчок, замеры
    error     модель недоступна или сервер не ответил

Сервер без состояния: историю диалога держит страница и присылает с каждым
вопросом — только цели пользователя и тексты ответов.
"""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

import agent
import llm
import workspace

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
NO_CACHE = {"Cache-Control": "no-cache"}
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "no-cache"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    workspace.ensure()
    async with agent.connect() as connection:
        app.state.connection = connection
        yield


app = FastAPI(title="notes · файловый ассистент", lifespan=lifespan)
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
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
        "workspace": workspace.stats(),
        "presets": list(agent.PRESETS),
    }


@app.get("/api/files")
async def files() -> dict:
    tree = workspace.tree()
    delta = workspace.diff()
    return {**tree, "changed": delta["changed"]}


@app.get("/api/files/{path:path}")
async def file(path: str) -> dict:
    try:
        return workspace.read(path, max_chars=20000)
    except workspace.WorkspaceError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/diff")
async def current_diff() -> dict:
    return workspace.diff()


@app.post("/api/reset")
async def reset() -> dict:
    return workspace.reset()


def sse_frame(data: object, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def event_stream(request: AskRequest) -> AsyncIterator[str]:
    try:
        llm.check(request.provider, await llm.probe())
        events = agent.answer(
            connection(),
            question=request.question,
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
