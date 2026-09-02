import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator
from starlette.staticfiles import StaticFiles

from llm import DEFAULT_EXPERTS, DEFAULT_PROMPT, stream_answer

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
NO_CACHE = {"Cache-Control": "no-cache"}


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
    step_by_step: bool = False
    meta_prompt: bool = False
    experts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_experts(self) -> "AskRequest":
        self.experts = [name.strip() for name in self.experts if name.strip()]
        return self


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/api/defaults")
async def defaults() -> dict[str, str | list[str]]:
    return {"prompt": DEFAULT_PROMPT, "experts": list(DEFAULT_EXPERTS)}


def sse_frame(data: str, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n"


async def event_stream(
    prompt: str,
    *,
    step_by_step: bool,
    meta_prompt: bool,
    experts: list[str],
) -> AsyncIterator[str]:
    chunks: list[str] = []
    finish_reason: str | None = None
    stage = "answer"

    try:
        async for delta in stream_answer(
            prompt,
            step_by_step=step_by_step,
            meta_prompt=meta_prompt,
            experts=experts,
        ):
            if delta.stage:
                stage = delta.stage
                yield sse_frame(json.dumps(delta.stage, ensure_ascii=False), event="stage")
            if delta.content:
                if stage == "answer":
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


SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@app.post("/api/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    return StreamingResponse(
        event_stream(
            request.prompt.strip(),
            step_by_step=request.step_by_step,
            meta_prompt=request.meta_prompt,
            experts=request.experts,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
