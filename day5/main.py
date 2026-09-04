import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from llm import (
    DEFAULT_PROMPT,
    MAX_TOKENS,
    MODELS,
    TEMPERATURE,
    Chunk,
    Metrics,
    ModelSpec,
    is_peak_hour,
    stream_model,
)

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


app = FastAPI(title="DeepSeek Web")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/api/models")
async def models() -> dict[str, Any]:
    return {
        "prompt": DEFAULT_PROMPT,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "peak": is_peak_hour(),
        "models": [
            {
                "key": spec.key,
                "tier": spec.tier,
                "model": spec.model,
                "provider": spec.provider,
                "link": spec.link,
                "note": spec.note,
                "paid": spec.pricing is not None,
            }
            for spec in MODELS
        ],
    }


def sse_frame(payload: Any, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _run_model(spec: ModelSpec, prompt: str, queue: asyncio.Queue[str | None]) -> None:
    try:
        async for item in stream_model(spec, prompt):
            if isinstance(item, Chunk):
                await queue.put(
                    sse_frame({"model": spec.key, "kind": item.kind, "text": item.text}, event="chunk"),
                )
            elif isinstance(item, Metrics):
                await queue.put(sse_frame({"model": spec.key, **asdict(item)}, event="done"))
    except Exception as exc:
        await queue.put(sse_frame({"model": spec.key, "message": str(exc)}, event="error"))
    finally:
        await queue.put(None)


async def event_stream(prompt: str) -> AsyncIterator[str]:
    """Все три модели стартуют от одной точки отсчёта и пишут в общую очередь."""
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    tasks = [asyncio.create_task(_run_model(spec, prompt, queue)) for spec in MODELS]
    pending = len(tasks)

    try:
        while pending:
            frame = await queue.get()
            if frame is None:
                pending -= 1
                continue
            yield frame
        yield sse_frame({}, event="end")
    finally:
        for task in tasks:
            task.cancel()


@app.post("/api/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    return StreamingResponse(
        event_stream(request.prompt.strip()),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
