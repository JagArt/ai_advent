"""Веб-слой: корпус, сборка индексов, поиск по трём стратегиям рядом и сравнение.

Модели в запросах этой страницы нет ни одной: эмбеддинги считает локальная
`model2vec`, поиск — numpy, сравнение — готовый набор проб из `probes.json`. Ключ
DeepSeek нужен только шагу, который придумывает пробы, и он живёт в CLI.

Потоком отдаётся ровно одна вещь — сборка индексов: три стратегии по полторы
тысячи чанков считаются секунды, и показывать их одним ответом в конце значило бы
держать страницу пустой всё это время. Остальное — обычные JSON-ручки.
"""

import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

import chunking
import corpus
import evaluate
import index

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


app = FastAPI(title="Индексация документов: чанки, эмбеддинги, сравнение стратегий")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    limit: int = Field(default=5, ge=1, le=20)


def sse_frame(data: object, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/")
async def page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/api/state")
async def api_state() -> dict[str, object]:
    """Что уже собрано: корпус на диске, индексы в базе, набор проб."""
    documents, skipped = await run_in_threadpool(corpus.scan)
    probes, notes = await run_in_threadpool(evaluate.load_probes)

    return {
        "corpus": corpus.summary(documents),
        "skipped": skipped,
        "source_titles": corpus.SOURCE_TITLES,
        "strategies": [
            {
                "strategy": strategy,
                "title": chunking.STRATEGY_TITLES[strategy],
                "params": chunking.defaults(strategy),
            }
            for strategy in chunking.STRATEGIES
        ],
        "built": index.built(),
        "probes": {"count": len(probes), "notes": notes, "kinds": evaluate.KIND_TITLES},
    }


async def build_stream() -> AsyncIterator[str]:
    documents = await run_in_threadpool(corpus.load)
    yield sse_frame(
        {"corpus": corpus.summary(documents), "strategies": list(chunking.STRATEGIES)},
        event="start",
    )

    for strategy in chunking.STRATEGIES:
        yield sse_frame({"strategy": strategy}, event="building")
        row = await run_in_threadpool(index.build, strategy, documents)
        yield sse_frame(row, event="built")

    yield sse_frame({"built": index.built()}, event="done")


async def guarded(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """Сбой посреди сборки должен доехать до страницы кадром, а не обрывом потока."""
    try:
        async for frame in stream:
            yield frame
    except Exception as exc:
        yield sse_frame(str(exc), event="error")


@app.post("/api/build")
async def api_build() -> StreamingResponse:
    return StreamingResponse(
        guarded(build_stream()), media_type="text/event-stream", headers=SSE_HEADERS
    )


@app.post("/api/search")
async def api_search(request: SearchRequest) -> dict[str, object]:
    """Один запрос по всем собранным стратегиям: вектор считается один раз."""
    try:
        found = await run_in_threadpool(index.search_all, request.query.strip(), request.limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "query": request.query.strip(),
        "results": {
            strategy: [hit.as_dict() for hit in hits] for strategy, hits in found.items()
        },
    }


@app.get("/api/chunk/{chunk_id}")
async def api_chunk(chunk_id: int) -> dict[str, object]:
    """Чанк целиком: текст, все метаданные и соседи — чтобы видеть, где прошла граница."""
    found = await run_in_threadpool(index.chunk, chunk_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"Чанка #{chunk_id} в индексе нет.")
    return found


@app.get("/api/documents")
async def api_documents(strategy: str = "structural") -> dict[str, object]:
    if strategy not in chunking.STRATEGIES:
        raise HTTPException(status_code=404, detail=f"Стратегии {strategy!r} нет.")
    return {"strategy": strategy, "documents": await run_in_threadpool(index.paths, strategy)}


@app.get("/api/document")
async def api_document(path: str) -> dict[str, object]:
    """Один документ и границы чанков в нём по каждой стратегии — разбиения рядом."""
    document = await run_in_threadpool(index.document, path)
    if document is None:
        raise HTTPException(status_code=404, detail=f"Документа {path!r} в индексе нет.")

    built = index.built()
    return {
        "document": {key: value for key, value in document.items() if key != "text"},
        "text": document["text"],
        "cuts": {
            strategy: await run_in_threadpool(index.chunks_of, strategy, path)
            for strategy in chunking.STRATEGIES
            if strategy in built
        },
    }


@app.post("/api/compare")
async def api_compare() -> dict[str, object]:
    """Сравнение по готовому набору проб. Без модели: пробы уже придуманы и лежат в файле."""
    try:
        return await run_in_threadpool(evaluate.compare)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
