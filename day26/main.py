import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from llm import BASE_URL, MODEL, Delta, list_models, stream_answer
from prompts import as_json

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
NO_CACHE = {"Cache-Control": "no-cache"}


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(title="Local LLM")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1)
    think: bool = True


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/api/status")
async def status() -> dict:
    try:
        models = await list_models()
    except Exception as exc:
        return {
            "online": False,
            "base_url": BASE_URL,
            "model": MODEL,
            "error": f"LM Studio не отвечает: {exc}. Запустите `lms server start`.",
        }

    return {
        "online": True,
        "base_url": BASE_URL,
        "model": MODEL,
        "models": models,
        "model_available": MODEL in models,
    }


@app.get("/api/prompts")
async def prompts() -> list[dict]:
    return as_json()


def sse_frame(data: str, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n"


async def event_stream(prompt: str, think: bool) -> AsyncIterator[str]:
    try:
        async for item in stream_answer(prompt, think=think):
            if isinstance(item, Delta):
                if item.reasoning:
                    yield sse_frame(json.dumps(item.reasoning, ensure_ascii=False), event="reasoning")
                if item.content:
                    yield sse_frame(json.dumps(item.content, ensure_ascii=False))
            else:
                yield sse_frame(json.dumps(asdict(item), ensure_ascii=False), event="done")
    except Exception as exc:
        yield sse_frame(json.dumps(str(exc), ensure_ascii=False), event="error")


@app.post("/api/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    return StreamingResponse(
        event_stream(request.prompt.strip(), request.think),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
