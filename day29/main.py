"""Веб-слой: один вопрос, локальный поиск, ответ локальной модели по выбранному профилю.

Ручка `/api/ask` отвечает за один профиль за раз; режим «сравнить» на странице —
два запроса подряд, базовый и оптимизированный. Подряд, а не параллельно, как
«обе» в day28: там колонки шли к разным серверам, здесь — к одной модели на
одной машине, и два одновременных запроса делили бы между собой и GPU, и
KV-кэш, а замеры времени показывали бы не профиль, а соседа.

Поиск у профилей общий, бюджет контекста — нет: `event: context` поэтому
приходит в каждом ответе свой, с тем набором выдержек, который модель и читала.
Страница показывает выдержки последней колонки и помечает обрезанные.

Кадры потока идут в порядке, в котором происходит работа:

    context   выдержки для этого профиля — до единого токена модели
    reasoning рассуждения, если профиль их допускает и галочка стоит
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
import lmstudio
import profiles
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


app = FastAPI(title="Оптимизация локальной LLM")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    optimized: bool = False
    think: bool = False


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/api/status")
async def status() -> dict:
    local: dict[str, object] = {"base_url": llm.BASE_URL, "model": llm.MODEL}
    try:
        models = await llm.local_models()
    except Exception as exc:
        local.update(online=False, error=f"LM Studio не отвечает: {exc}. Запустите `lms server start`.")
    else:
        local.update(online=True, models=models, model_available=llm.MODEL in models)
        try:
            info = await lmstudio.model(llm.MODEL)
        except Exception as exc:
            local["info_error"] = f"REST LM Studio не ответил: {exc}"
        else:
            local["info"] = info.as_dict() if info else None

    return {
        "local": local,
        "index": await run_in_threadpool(search.index_status),
        "profiles": {name: profile.as_dict() for name, profile in profiles.PROFILES.items()},
    }


@app.get("/api/questions")
async def control_questions() -> list[dict]:
    return [item.as_dict() for item in questions.load()]


def sse_frame(data: object, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def event_stream(request: AskRequest) -> AsyncIterator[str]:
    profile = profiles.for_flag(request.optimized)
    try:
        found = await run_in_threadpool(search.prepare, request.question)
        context = await run_in_threadpool(profile.context, found)
        yield sse_frame({**context.as_dict(), "profile": profile.name}, event="context")

        async for item in llm.stream_answer(request.question, context.block, profile=profile, think=request.think):
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
