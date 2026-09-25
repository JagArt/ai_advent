"""Веб-слой: один запрос, два способа пройти цепочку, и всё это потоком.

`/api/pipeline` в режиме `auto` делает один вызов `run_pipeline`, а кадры шагов
берёт из progress-уведомлений: вызов идёт задачей, уведомления копятся в очереди,
а генератор SSE отдаёт их странице по мере поступления. В режиме `manual` тот же
запрос уходит агенту, и цепочку собирает модель — кадры те же, что в чате.

Остальные ручки читают то, что цепочка уже оставила в базе и в папке out.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from agent import Chunk, Connected, Done, ToolRun, answer
from client import connect

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
OUT_DIR = BASE_DIR / "out"
NO_CACHE = {"Cache-Control": "no-cache"}
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

QUEUE_TICK = 0.2


def _reason(exc: BaseException) -> str:
    """Сбой транспорта приезжает группой исключений; человеку нужна первая причина."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(title="MCP — композиция инструментов")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


class PipelineRequest(BaseModel):
    query: str = Field(min_length=2)
    limit: int = Field(default=5, ge=1, le=10)
    filename: str | None = None
    mode: str = Field(default="auto", pattern="^(auto|manual)$")


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1)


def sse_frame(data: object, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.post("/api/connect")
async def api_connect() -> dict[str, object]:
    """Только соединение и список инструментов, без модели."""
    try:
        async with connect() as connection:
            return connection.info()
    except Exception as exc:
        return {"error": _reason(exc)}


@app.get("/api/runs")
async def api_runs(limit: int = 10) -> dict[str, object]:
    try:
        async with connect() as connection:
            return {"runs": await connection.call("list_runs", {"limit": limit})}
    except Exception as exc:
        return {"error": _reason(exc)}


@app.get("/api/artifacts/{artifact_id}")
async def api_artifact(artifact_id: int) -> dict[str, object]:
    try:
        async with connect() as connection:
            return await connection.call("get_artifact", {"artifact_id": artifact_id})
    except Exception as exc:
        return {"error": _reason(exc)}


@app.get("/api/files/{filename}")
async def api_file(filename: str) -> PlainTextResponse:
    """Файл, который записал save_to_file. Имя берётся без папок, путь проверяется."""
    path = (OUT_DIR / Path(filename).name).resolve()
    if path.parent != OUT_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Файла {filename} в out/ нет.")

    return PlainTextResponse(path.read_text(encoding="utf-8"), headers=NO_CACHE)


async def auto_stream(request: PipelineRequest) -> AsyncIterator[str]:
    """Цепочка одним вызовом: шаги приезжают уведомлениями, пока вызов идёт."""
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async with connect() as connection:
        yield sse_frame(connection.info(), event="mcp")

        arguments = {"query": request.query, "limit": request.limit}
        if request.filename:
            arguments["filename"] = request.filename

        call = asyncio.create_task(
            connection.call("run_pipeline", arguments, on_step=queue.put)
        )

        while not call.done() or not queue.empty():
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=QUEUE_TICK)
            except TimeoutError:
                continue
            yield sse_frame(frame, event="step")

        yield sse_frame(await call, event="pipeline")


MANUAL_PROMPT = (
    "Собери сводку по запросу «{query}» и сохрани её файлом. "
    "Возьми {limit} разделов документации. Пройди цепочку по шагам, "
    "не вызывая run_pipeline."
)


async def manual_stream(request: PipelineRequest) -> AsyncIterator[str]:
    """Ту же цепочку собирает модель: кадры такие же, как в чате."""
    prompt = MANUAL_PROMPT.format(query=request.query, limit=request.limit)
    if request.filename:
        prompt += f" Имя файла: {request.filename}."

    async for frame in agent_stream(prompt):
        yield frame


async def agent_stream(prompt: str) -> AsyncIterator[str]:
    async for event in answer(prompt):
        match event:
            case Chunk():
                yield sse_frame(event.text)
            case Connected():
                yield sse_frame(event.info, event="mcp")
            case ToolRun():
                yield sse_frame(asdict(event), event="tool")
            case Done():
                yield sse_frame(asdict(event), event="done")


async def guarded(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """Ошибка любого шага должна доехать до страницы кадром, а не обрывом потока."""
    try:
        async for frame in stream:
            yield frame
    except Exception as exc:
        yield sse_frame(_reason(exc), event="error")


@app.post("/api/pipeline")
async def pipeline(request: PipelineRequest) -> StreamingResponse:
    stream = auto_stream(request) if request.mode == "auto" else manual_stream(request)
    return StreamingResponse(
        guarded(stream), media_type="text/event-stream", headers=SSE_HEADERS
    )


@app.post("/api/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    return StreamingResponse(
        guarded(agent_stream(request.prompt.strip())),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
