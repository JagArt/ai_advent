"""Веб-слой: четыре режима одним потоком, цитаты со сверкой и два набора.

Потоком отдаются две вещи, и обе по одной причине — показывать их одним ответом
в конце значило бы спрятать самое интересное.

Сборка индекса считается секунды, и всё это время страница была бы пустой.

Ответы — важнее, и в day24 к прежнему доводу добавился новый. Два режима из
четырёх пишут json, и в потоке видно, как он набирается: сначала утверждение,
потом его цитата, потом следующее. Разбор и сверка случаются на последнем
кадре, и до него страница честно показывает сырой текст — в том числе когда
ответ обрывается на половине цитаты.

Отдельный кадр есть у отказа слоя в коде. Он приходит мгновенно и без единого
токена: порог решил, что отвечать нечем, и к модели запрос не ушёл. Это не
ошибка и не пустой ответ, поэтому и кадр у него свой.
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
import gate
import index
import llm
import pipeline
import rerank
import retrieve
import rewrite
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


app = FastAPI(title="RAG: цитаты, источники и анти-галлюцинации")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    retriever: str | None = None
    rewriter: str | None = None
    reranker: str | None = None
    pool: int | None = Field(default=None, ge=5, le=40)
    top_k: int | None = Field(default=None, ge=1, le=12)
    threshold: float | None = Field(default=None, ge=0, le=1)
    gate_threshold: float | None = Field(default=None, ge=0, le=1)


class RunRequest(BaseModel):
    rewriter: str | None = None
    reranker: str | None = None
    threshold: float | None = None
    gate_threshold: float | None = Field(default=None, ge=0, le=1)


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
    """Что уже есть: корпус, индекс, ручки конвейера и оба набора вопросов."""
    documents, skipped = await run_in_threadpool(corpus.scan)
    questions = await run_in_threadpool(evaluate.load)
    weak = await run_in_threadpool(evaluate.load_weak)

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
        "rewriters": [
            {"name": name, "title": rewrite.MODE_TITLES[name], "default": name == answer.REWRITER}
            for name in rewrite.MODES
        ],
        "rerankers": [
            {"name": name, "title": rerank.RERANKER_TITLES[name], "default": name == answer.RERANKER}
            for name in rerank.RERANKERS
        ],
        "modes": [{"name": mode, "title": answer.MODE_TITLES[mode]} for mode in answer.MODES],
        "mode_titles": answer.MODE_TITLES,
        "structured_modes": list(answer.STRUCTURED),
        "gated_modes": list(answer.GATED),
        "verdicts": list(verify.VERDICTS),
        "defaults": {
            "rewriter": answer.REWRITER,
            "reranker": answer.RERANKER,
            "retriever": retrieve.DEFAULT,
            "pool": pipeline.POOL,
            "top_k": pipeline.TOP_K,
            "threshold": rerank.THRESHOLDS[answer.RERANKER],
            "per_path": rerank.PER_PATH,
            "gate_threshold": gate.THRESHOLD,
        },
        "questions": {
            "count": len(questions),
            "kinds": sorted({item.kind for item in questions}),
            "items": [
                {"id": item.id, "kind": item.kind, "question": item.question,
                 "expect": item.expect, "facts": item.facts, "sources": item.sources}
                for item in questions
            ],
        },
        "weak": {
            "count": len(weak),
            "kinds": list(evaluate.WEAK_KINDS),
            "items": [
                {"id": item.id, "kind": item.kind, "question": item.question,
                 "expect": item.expect}
                for item in weak
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


def _overrides(request: AskRequest) -> dict[str, object]:
    return {
        "retriever": request.retriever,
        "rewriter": request.rewriter,
        "reranker": request.reranker,
        "pool": request.pool,
        "top_k": request.top_k,
        "threshold": request.threshold,
    }


async def ask_stream(request: AskRequest) -> AsyncIterator[str]:
    """Все четыре режима одновременно в один поток, кадры помечены режимом."""
    question = request.question.strip()
    plans = {mode: answer.plan_of(mode, **_overrides(request)) for mode in answer.MODES}
    prepared = await answer.contexts(question, plans=plans)

    verdicts = {
        mode: answer.checkpoint(mode, prepared[mode], request.gate_threshold)
        for mode in answer.MODES
    }

    yield sse_frame(
        {
            "question": question,
            "contexts": {
                mode: context.as_dict() if context else None
                for mode, context in prepared.items()
            },
            "plans": {mode: plan.as_dict() if plan else None for mode, plan in plans.items()},
            "gates": {
                mode: verdict.as_dict() if verdict else None
                for mode, verdict in verdicts.items()
            },
        },
        event="context",
    )

    queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()

    async def pump(mode: str) -> None:
        used = prepared[mode]
        verdict = verdicts[mode]

        # Отказ слоя в коде: модель не вызывается вовсе, и кадр уходит сразу.
        if verdict is not None and not verdict.passed:
            await queue.put(("answer", answer.stopped(question, mode, used, verdict).as_dict()))
            await queue.put(("closed", mode))
            return

        try:
            async for piece in llm.stream(
                answer.messages(question, used, mode),
                temperature=answer.TEMPERATURE,
                max_tokens=(
                    answer.STRUCTURED_MAX_TOKENS
                    if answer.structured(mode)
                    else answer.MAX_TOKENS
                ),
                json=answer.structured(mode),
            ):
                if piece.reply is None:
                    await queue.put(("delta", {"mode": mode, "content": piece.content}))
                else:
                    done = answer.finish(question, mode, used, piece.reply, verdict)
                    await queue.put(("answer", done.as_dict()))
        except Exception as exc:
            await queue.put(("error", f"{answer.MODE_TITLES[mode]}: {exc}"))
        finally:
            await queue.put(("closed", mode))

    tasks = [asyncio.create_task(pump(mode)) for mode in answer.MODES]

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
    if request.retriever is not None and request.retriever not in retrieve.RETRIEVERS:
        raise HTTPException(status_code=404, detail=f"Ретривера {request.retriever!r} нет.")
    if request.rewriter is not None and request.rewriter not in rewrite.MODES:
        raise HTTPException(status_code=404, detail=f"Режима переписывания {request.rewriter!r} нет.")
    if request.reranker is not None and request.reranker not in rerank.RERANKERS:
        raise HTTPException(status_code=404, detail=f"Реранкера {request.reranker!r} нет.")

    return StreamingResponse(
        guarded(ask_stream(request)), media_type="text/event-stream", headers=SSE_HEADERS
    )


@app.get("/api/chunk/{chunk_id}")
async def api_chunk(chunk_id: int) -> dict[str, object]:
    """Чанк целиком. По смещениям цитаты страница подсвечивает её в этом тексте."""
    found = await run_in_threadpool(index.chunk, chunk_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"Чанка #{chunk_id} в индексе нет.")
    return found


# --- замеры ------------------------------------------------------------------


@app.post("/api/questions")
async def api_questions(request: RunRequest) -> dict[str, object]:
    """Контрольный набор во всех четырёх режимах. Самая дорогая ручка на странице."""
    try:
        return await evaluate.run(
            rewriter=request.rewriter,
            reranker=request.reranker,
            threshold=request.threshold,
            gate_threshold=request.gate_threshold,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/weak")
async def api_weak(request: RunRequest) -> dict[str, object]:
    """Набор слабого контекста: восемь вопросов, где верный ответ — отказ."""
    try:
        return await evaluate.run_weak(
            rewriter=request.rewriter,
            reranker=request.reranker,
            threshold=request.threshold,
            gate_threshold=request.gate_threshold,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/gate")
async def api_gate() -> dict[str, object]:
    """Развертка по порогу отказа. После прогрева кэша — без единого вызова модели."""
    try:
        return await evaluate.sweep()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
