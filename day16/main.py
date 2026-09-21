import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from agent import Chunk, Connected, Done, ToolRun, answer
from client import connect

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
NO_CACHE = {"Cache-Control": "no-cache"}


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(title="MCP — подключение и инструменты")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1)
    mcp: bool = True


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


def sse_frame(data: object, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/connect")
async def api_connect() -> dict[str, object]:
    """Только соединение и список инструментов, без модели."""
    try:
        async with connect() as connection:
            return connection.info()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


async def event_stream(prompt: str, use_mcp: bool) -> AsyncIterator[str]:
    try:
        async for event in answer(prompt, use_mcp=use_mcp):
            match event:
                case Chunk():
                    yield sse_frame(event.text)
                case Connected():
                    yield sse_frame(event.info, event="mcp")
                case ToolRun():
                    yield sse_frame(asdict(event), event="tool")
                case Done():
                    yield sse_frame(asdict(event), event="done")
    except Exception as exc:
        yield sse_frame(f"{type(exc).__name__}: {exc}", event="error")


@app.post("/api/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    return StreamingResponse(
        event_stream(request.prompt.strip(), request.mcp),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
