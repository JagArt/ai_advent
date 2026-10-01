"""Веб-слой: один вопрос в двух режимах рядом, поиск под ними и контрольный набор.

Потоком отдаются две вещи, и обе по одной причине — показывать их одним ответом
в конце значило бы спрятать самое интересное.

Сборка индекса считается секунды, и всё это время страница была бы пустой.

Ответы — важнее. Оба режима идут к модели одновременно и пишут в один поток,
помечая кадры своим именем. Ответ без RAG начинает появляться раньше: ему нечего
читать, у него сто двадцать токенов запроса против почти двух тысяч. Эта разница
и есть цена контекста, и в потоке она видна прямо, а в готовом ответе — только
числом в подписи.

Остальное — обычные JSON-ручки. Поиск, сравнение ретриверов и состояние работают
без ключа: эмбеддинги локальные, лексика в SQLite, набор проб лежит в файле.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

import answer
import chunking
import corpus
import evaluate
import index
import llm
import probes
import retrieve

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


app = FastAPI(title="Первый RAG-запрос: ответ с базой и без неё")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    retriever: str = Field(default=retrieve.DEFAULT)
    limit: int = Field(default=answer.TOP_K, ge=1, le=12)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    limit: int = Field(default=5, ge=1, le=20)


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
    """Что уже есть: корпус на диске, индекс в базе, набор проб и контрольный набор."""
    documents, skipped = await run_in_threadpool(corpus.scan)
    loaded, notes = await run_in_threadpool(probes.load)
    questions = await run_in_threadpool(evaluate.load)

    return {
        "corpus": corpus.summary(documents),
        "skipped": skipped,
        "source_titles": corpus.SOURCE_TITLES,
        "strategy": {"name": chunking.STRATEGY, "title": chunking.STRATEGY_TITLE,
                     "params": chunking.defaults()},
        "built": index.built(),
        "model": llm.MODEL,
        "retrievers": [
            {"name": name, "title": retrieve.RETRIEVER_TITLES[name], "default": name == retrieve.DEFAULT}
            for name in retrieve.RETRIEVERS
        ],
        "modes": [{"name": mode, "title": answer.MODE_TITLES[mode]} for mode in answer.MODES],
        "probes": {"count": len(loaded), "notes": notes, "kinds": probes.KIND_TITLES},
        "questions": {
            "count": len(questions),
            "kinds": sorted({item.kind for item in questions}),
            "items": [
                {"id": item.id, "kind": item.kind, "question": item.question,
                 "expect": item.expect, "facts": item.facts, "sources": item.sources}
                for item in questions
            ],
        },
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


# --- вопрос ------------------------------------------------------------------


async def ask_stream(request: AskRequest) -> AsyncIterator[str]:
    """Оба режима одновременно в один поток, кадры помечены режимом."""
    question = request.question.strip()
    context = await run_in_threadpool(
        answer.prepare, question, retriever=request.retriever, limit=request.limit
    )
    yield sse_frame({"question": question, **context.as_dict()}, event="context")

    queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()

    async def pump(mode: str, used: answer.Context | None) -> None:
        try:
            async for piece in llm.stream(
                answer.messages(question, used),
                temperature=answer.TEMPERATURE,
                max_tokens=answer.MAX_TOKENS,
            ):
                if piece.reply is None:
                    await queue.put(("delta", {"mode": mode, "content": piece.content}))
                else:
                    done = answer.finish(question, mode, used, piece.reply)
                    await queue.put(("answer", done.as_dict()))
        except Exception as exc:
            await queue.put(("error", f"{answer.MODE_TITLES[mode]}: {exc}"))
        finally:
            await queue.put(("closed", mode))

    tasks = [
        asyncio.create_task(pump("plain", None)),
        asyncio.create_task(pump("rag", context)),
    ]

    try:
        live = len(tasks)
        while live:
            event, payload = await queue.get()
            if event == "closed":
                live -= 1
                continue
            yield sse_frame(payload, event=event)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    yield sse_frame({"question": question}, event="done")


@app.post("/api/ask")
async def api_ask(request: AskRequest) -> StreamingResponse:
    if request.retriever not in retrieve.RETRIEVERS:
        raise HTTPException(status_code=404, detail=f"Ретривера {request.retriever!r} нет.")

    return StreamingResponse(
        guarded(ask_stream(request)), media_type="text/event-stream", headers=SSE_HEADERS
    )


# --- поиск и чанки -----------------------------------------------------------


@app.post("/api/search")
async def api_search(request: SearchRequest) -> dict[str, object]:
    """Один запрос всеми тремя ретриверами: разницу видно рядом."""
    try:
        found = await run_in_threadpool(retrieve.search_all, request.query.strip(), request.limit)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "query": request.query.strip(),
        "results": {name: item.as_dict() for name, item in found.items()},
    }


@app.get("/api/chunk/{chunk_id}")
async def api_chunk(chunk_id: int) -> dict[str, object]:
    found = await run_in_threadpool(index.chunk, chunk_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"Чанка #{chunk_id} в индексе нет.")
    return found


# --- замеры ------------------------------------------------------------------


@app.post("/api/retrieval")
async def api_retrieval() -> dict[str, object]:
    """Сравнение ретриверов на пробах day21. Без модели и без ключа."""
    try:
        return await run_in_threadpool(probes.compare)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/questions")
async def api_questions(retriever: str = retrieve.DEFAULT) -> dict[str, object]:
    """Контрольный набор в обоих режимах. Единственная ручка, которой нужен ключ."""
    try:
        return await evaluate.run(retriever)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
