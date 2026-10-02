"""Веб-слой: чат с источниками слева, память задачи справа, замер по кнопке.

В [day24](../day24/main.py) ручка `/api/ask` считала один вопрос в четырёх
режимах сразу и была полностью без состояния: запрос приходил со всем, что нужно
для ответа. Здесь это невозможно — у хода есть предыстория, — и структура ручек
меняется ровно на это: сначала заводится диалог, дальше реплики едут в него.

Режим памяти задаётся при создании диалога и потом не меняется. Переключатель в
шапке создаёт новый диалог, а не правит текущий: разговор, половина которого
собрана с памятью задачи, а половина без, нельзя ни понять, ни замерить.

Потоком отдаются две вещи. Сборка индекса — как в day21–day24. И ход чата, но
кадров у него больше, чем в day24, и порядок кадров здесь — часть ответа на
вопрос «как это работает»:

    resolved  по какому запросу пошли искать и переписали ли реплику
    context   что нашлось и что сказал порог отказа
    delta     json ответа по кускам, как он набирается
    turn      разобранный ход со сверкой цитат и обновлённой памятью задачи

Между `context` и `delta` стоит решение порога: если он отказал, `delta` не будет
вовсе, и `turn` приходит сразу — без единого токена, как и в day24.

Кадр `turn` последний не случайно: память задачи обновляется **после** ответа, и
показать её раньше нельзя, не соврав о порядке.
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

import chat
import chunking
import corpus
import dialogs
import evaluate
import gate
import index
import llm
import pipeline
import state
import storage
import verify

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


app = FastAPI(title="Мини-чат с RAG и памятью задачи")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")

store = storage.Storage()

# Диалоги живут в процессе по одному на сессию: снимок истории и памяти у них
# свой, а источник правды — база. Перечитывать её на каждый ход можно, но тогда
# у страницы и у прогонов сценариев получились бы разные пути в одном коде.
_dialogs: dict[str, chat.Dialog] = {}


@app.on_event("startup")
async def startup() -> None:
    await store.init()


async def dialog_of(session_id: str) -> chat.Dialog:
    """Диалог из процесса, а при промахе — поднятый из базы."""
    if session_id not in _dialogs:
        try:
            _dialogs[session_id] = await chat.Dialog.resume(store, session_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _dialogs[session_id]


class OpenRequest(BaseModel):
    mode: str = chat.MODES[-1]
    title: str = ""


class SayRequest(BaseModel):
    question: str = Field(min_length=2)
    gate_threshold: float | None = Field(default=None, ge=0, le=1)


class RunRequest(BaseModel):
    keys: list[str] | None = None
    modes: list[str] | None = None
    gate_threshold: float | None = Field(default=None, ge=0, le=1)
    judged: bool = True


def sse_frame(data: object, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def guarded(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """Сбой посреди потока должен доехать до страницы кадром, а не обрывом."""
    try:
        async for frame in stream:
            yield frame
    except Exception as exc:
        yield sse_frame(str(exc), event="error")


@app.get("/")
async def page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/api/state")
async def api_state() -> dict[str, object]:
    """Что уже есть: корпус, индекс, режимы памяти, разделы состояния и сценарии."""
    documents, skipped = await run_in_threadpool(corpus.scan)

    return {
        "corpus": corpus.summary(documents),
        "skipped": skipped,
        "source_titles": corpus.SOURCE_TITLES,
        "strategy": {
            "name": chunking.STRATEGY,
            "title": chunking.STRATEGY_TITLE,
            "params": chunking.defaults(),
        },
        "built": index.built(),
        "model": llm.MODEL,
        "modes": [{"name": mode, "title": chat.MODE_TITLES[mode]} for mode in chat.MODES],
        "mode_titles": chat.MODE_TITLES,
        "memory": {
            mode: {"history": chat.MEMORY[mode].history, "tracked": chat.MEMORY[mode].tracked}
            for mode in chat.MODES
        },
        "sections": list(state.SECTIONS),
        "section_about": state.SECTION_ABOUT,
        "verdicts": list(verify.VERDICTS),
        "kinds": list(dialogs.KINDS),
        "kind_titles": dialogs.KIND_TITLES,
        "defaults": {
            "mode": chat.MODES[-1],
            "rewriter": chat.REWRITER,
            "reranker": chat.RERANKER,
            "pool": pipeline.POOL,
            "top_k": pipeline.TOP_K,
            "tail": chat.TAIL_MESSAGES,
            "cap": state.CAP,
            "gate_threshold": gate.THRESHOLD,
        },
        "dialogues": [dialogue.as_dict() for dialogue in dialogs.every()],
        "sessions": await store.sessions(),
    }


# --- сборка ------------------------------------------------------------------


async def build_stream() -> AsyncIterator[str]:
    documents = await run_in_threadpool(corpus.load)
    yield sse_frame({"corpus": corpus.summary(documents)}, event="start")
    row = await run_in_threadpool(index.build, documents)
    yield sse_frame(row, event="built")


@app.post("/api/build")
async def api_build() -> StreamingResponse:
    return StreamingResponse(
        guarded(build_stream()), media_type="text/event-stream", headers=SSE_HEADERS
    )


# --- диалоги -----------------------------------------------------------------


@app.post("/api/session")
async def api_open(request: OpenRequest) -> dict[str, object]:
    """Новый диалог. Режим памяти задаётся здесь и дальше не меняется."""
    if request.mode not in chat.MODES:
        raise HTTPException(status_code=404, detail=f"Режима {request.mode!r} нет.")

    dialog = await chat.Dialog.open(store, request.mode, request.title)
    _dialogs[dialog.session_id] = dialog
    return {
        "session_id": dialog.session_id,
        "mode": dialog.mode,
        "title": chat.MODE_TITLES[dialog.mode],
        "state": dialog.state.as_dict(),
        "turns": [],
    }


@app.get("/api/session/{session_id}")
async def api_session(session_id: str) -> dict[str, object]:
    """Диалог целиком: ходы с источниками, журнал цели и текущая память задачи."""
    dialog = await dialog_of(session_id)
    return {
        "session_id": session_id,
        "mode": dialog.mode,
        "title": chat.MODE_TITLES[dialog.mode],
        "state": dialog.state.as_dict(),
        "turns": await store.transcript(session_id),
        "goals": await store.goal_log(session_id),
        "stats": await store.stats(session_id),
    }


@app.delete("/api/session/{session_id}")
async def api_drop(session_id: str) -> dict[str, object]:
    await store.drop(session_id)
    _dialogs.pop(session_id, None)
    return {"dropped": session_id}


async def say_stream(dialog: chat.Dialog, request: SayRequest) -> AsyncIterator[str]:
    async for event, payload in dialog.stream(request.question, request.gate_threshold):
        yield sse_frame(payload, event=event)
    yield sse_frame({"session_id": dialog.session_id}, event="done")


@app.post("/api/session/{session_id}/say")
async def api_say(session_id: str, request: SayRequest) -> StreamingResponse:
    """Один ход чата потоком. Память задачи обновляется последним кадром."""
    dialog = await dialog_of(session_id)
    return StreamingResponse(
        guarded(say_stream(dialog, request)),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.get("/api/chunk/{chunk_id}")
async def api_chunk(chunk_id: int) -> dict[str, object]:
    """Чанк целиком. По смещениям цитаты страница подсвечивает её в этом тексте."""
    found = await run_in_threadpool(index.chunk, chunk_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"Чанка #{chunk_id} в индексе нет.")
    return found


# --- замер -------------------------------------------------------------------


@app.post("/api/run")
async def api_run(request: RunRequest) -> dict[str, object]:
    """Оба сценария во всех трёх режимах. Самая дорогая ручка на странице."""
    keys = tuple(request.keys or dialogs.KEYS)
    modes = tuple(request.modes or chat.MODES)

    for key in keys:
        try:
            dialogs.load(key)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    for mode in modes:
        if mode not in chat.MODES:
            raise HTTPException(status_code=404, detail=f"Режима {mode!r} нет.")

    try:
        runs = await evaluate.every(
            keys=keys,
            modes=modes,
            store=store,
            threshold=request.gate_threshold,
            judged=request.judged,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "runs": [run.as_dict() for run in runs],
        "by_mode": evaluate.by_mode(runs),
        "by_kind": evaluate.by_kind(runs),
    }


@app.post("/api/sweep")
async def api_sweep(request: RunRequest) -> dict[str, object]:
    """Развертка по порогу отказа на одном сценарии, без судьи."""
    key = (request.keys or list(dialogs.KEYS))[0]
    try:
        dialogs.load(key)
        rows = await evaluate.sweep(key=key, store=store)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {"key": key, "thresholds": list(evaluate.THRESHOLDS), "rows": rows}
