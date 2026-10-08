"""Веб-слой: один вопрос, локальный поиск, ответ выбранной модели потоком.

Ручка `/api/ask` отвечает за одну модель за раз, а «обе сразу» делает страница
двумя параллельными запросами. Так колонки независимы по-настоящему: облако без
ключа или без сети присылает `error` в свою колонку, а локальная колонка этого
не замечает. Поиск при этом выполняется дважды, и это ничего не стоит — лексика
по FTS5 детерминирована и укладывается в десятки миллисекунд, так что обе
колонки получают один и тот же контекст.

Кадры потока идут в порядке, в котором происходит работа:

    context   что нашлось в индексе day25 — до единого токена модели
    reasoning рассуждения локальной модели, если включены
    data      текст ответа по кускам
    done      итог с замерами: первый токен, всего, токены, ток/с
    error     если модель недоступна или индекс не собран
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

import llm
import questions
import search

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


app = FastAPI(title="Локальная LLM + RAG")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    provider: str = Field(default="local", pattern="^(local|cloud)$")
    think: bool = False


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/api/status")
async def status() -> dict:
    local: dict[str, object] = {"base_url": llm.LOCAL_BASE_URL, "model": llm.LOCAL_MODEL}
    try:
        models = await llm.local_models()
    except Exception as exc:
        local.update(online=False, error=f"LM Studio не отвечает: {exc}. Запустите `lms server start`.")
    else:
        local.update(online=True, models=models, model_available=llm.LOCAL_MODEL in models)

    return {
        "local": local,
        "cloud": {"configured": llm.cloud_configured(), "model": llm.CLOUD_MODEL},
        "index": await run_in_threadpool(search.index_status),
    }


@app.get("/api/questions")
async def control_questions() -> list[dict]:
    return [item.as_dict() for item in questions.load()]


def sse_frame(data: object, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def event_stream(request: AskRequest) -> AsyncIterator[str]:
    try:
        context = await run_in_threadpool(search.prepare, request.question)
        yield sse_frame(context.as_dict(), event="context")

        async for item in llm.stream_answer(
            request.question, context.block, provider=request.provider, think=request.think
        ):
            if isinstance(item, llm.Delta):
                if item.reasoning:
                    yield sse_frame(item.reasoning, event="reasoning")
                if item.content:
                    yield sse_frame(item.content)
            else:
                yield sse_frame(item.as_dict(), event="done")
    except Exception as exc:
        yield sse_frame(str(exc), event="error")


@app.post("/api/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    request.question = request.question.strip()
    return StreamingResponse(event_stream(request), media_type="text/event-stream", headers=SSE_HEADERS)
