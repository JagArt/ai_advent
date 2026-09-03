import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from llm import (
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    TEMP_MAX,
    TEMP_MIN,
    TEMP_STEP,
    TEMPERATURE_BANDS,
    AnswerDelta,
    stream_answer,
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
    temperature: float = Field(ge=TEMP_MIN, le=TEMP_MAX)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/api/defaults")
async def defaults() -> dict[str, Any]:
    return {
        "prompt": DEFAULT_PROMPT,
        "temperature": DEFAULT_TEMPERATURE,
        "temp_min": TEMP_MIN,
        "temp_max": TEMP_MAX,
        "temp_step": TEMP_STEP,
        "bands": [{"max": upper, "label": label} for upper, label in TEMPERATURE_BANDS],
    }


def sse_frame(data: str, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n"


async def event_stream(deltas: AsyncIterator[AnswerDelta]) -> AsyncIterator[str]:
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
            },
            ensure_ascii=False,
        ),
        event="done",
    )


@app.post("/api/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    deltas = stream_answer(request.prompt.strip(), temperature=request.temperature)
    return StreamingResponse(
        event_stream(deltas),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
