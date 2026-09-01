import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from constraints import Constraints, constraints
from llm import stream_answer

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="DeepSeek Web")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1)
    constrained: bool = False


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/constraints")
async def get_constraints() -> Constraints:
    return constraints


def sse_frame(data: str, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n"


async def event_stream(prompt: str, constrained: bool) -> AsyncIterator[str]:
    chunks: list[str] = []
    finish_reason: str | None = None

    try:
        async for delta in stream_answer(prompt, constrained=constrained):
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
    return StreamingResponse(
        event_stream(request.prompt.strip(), request.constrained),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
