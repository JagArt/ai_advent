"""Веб-слой: пять режимов одним потоком, контекст с отсеянным и замеры.

Потоком отдаются две вещи, и обе по одной причине — показывать их одним ответом
в конце значило бы спрятать самое интересное.

Сборка индекса считается секунды, и всё это время страница была бы пустой.

Ответы — важнее. Все пять режимов идут к модели одновременно и пишут в один
поток, помечая кадры своим именем. Ответ без RAG начинает появляться раньше:
ему нечего читать. Эта разница и есть цена контекста, и в потоке она видна
прямо, а в готовом ответе — только числом в подписи.

Остальное — обычные JSON-ручки. Матрица и развертка после прогрева кэшей
считаются на машине; ключ нужен вопросу, контрольному набору и первому
прогреву переписывания и LLM-реранкера.
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
import pipeline
import probes
import rerank
import retrieve
import rewrite
from pipeline import Plan

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


app = FastAPI(title="RAG: переписывание, реранкинг и сравнение режимов")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    retriever: str | None = None
    rewriter: str | None = None
    reranker: str | None = None
    pool: int | None = Field(default=None, ge=5, le=40)
    top_k: int | None = Field(default=None, ge=1, le=12)
    threshold: float | None = Field(default=None, ge=0, le=1)


class RewriteRequest(BaseModel):
    question: str = Field(min_length=3)


class RerankRequest(BaseModel):
    question: str = Field(min_length=3)
    retriever: str = Field(default=retrieve.DEFAULT)
    pool: int = Field(default=pipeline.POOL, ge=5, le=40)
    top_k: int = Field(default=pipeline.TOP_K, ge=1, le=12)
    threshold: float | None = None


class SweepRequest(BaseModel):
    rewriter: str = Field(default="none")


class QuestionsRequest(BaseModel):
    rewriter: str | None = None
    reranker: str | None = None
    threshold: float | None = None


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


def _overrides(request: AskRequest | QuestionsRequest) -> dict[str, object]:
    return {
        "retriever": getattr(request, "retriever", None),
        "rewriter": request.rewriter,
        "reranker": request.reranker,
        "pool": getattr(request, "pool", None),
        "top_k": getattr(request, "top_k", None),
        "threshold": request.threshold,
    }


@app.get("/")
async def page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/api/state")
async def api_state() -> dict[str, object]:
    """Что уже есть: корпус, индекс, ручки конвейера, набор проб и контрольный набор."""
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
        "defaults": {
            "rewriter": answer.REWRITER,
            "reranker": answer.RERANKER,
            "retriever": retrieve.DEFAULT,
            "pool": pipeline.POOL,
            "top_k": pipeline.TOP_K,
            "threshold": rerank.THRESHOLDS[answer.RERANKER],
            "per_path": rerank.PER_PATH,
            "thresholds": rerank.THRESHOLDS,
        },
        "probes": {"count": len(loaded), "notes": notes},
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
    """Все пять режимов одновременно в один поток, кадры помечены режимом."""
    question = request.question.strip()
    plans = {mode: answer.plan_of(mode, **_overrides(request)) for mode in answer.MODES}
    prepared = await answer.contexts(question, plans=plans)
    yield sse_frame(
        {
            "question": question,
            "contexts": {
                mode: context.as_dict() if context else None
                for mode, context in prepared.items()
            },
            "plans": {mode: plan.as_dict() if plan else None for mode, plan in plans.items()},
        },
        event="context",
    )

    queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()

    async def pump(mode: str, used: pipeline.Context | None) -> None:
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
        asyncio.create_task(pump(mode, prepared[mode]))
        for mode in answer.MODES
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
    if request.retriever is not None and request.retriever not in retrieve.RETRIEVERS:
        raise HTTPException(status_code=404, detail=f"Ретривера {request.retriever!r} нет.")
    if request.rewriter is not None and request.rewriter not in rewrite.MODES:
        raise HTTPException(status_code=404, detail=f"Режима переписывания {request.rewriter!r} нет.")
    if request.reranker is not None and request.reranker not in rerank.RERANKERS:
        raise HTTPException(status_code=404, detail=f"Реранкера {request.reranker!r} нет.")

    return StreamingResponse(
        guarded(ask_stream(request)), media_type="text/event-stream", headers=SSE_HEADERS
    )


# --- этапы и чанки -----------------------------------------------------------


@app.post("/api/rewrite")
async def api_rewrite(request: RewriteRequest) -> dict[str, object]:
    """Один вопрос всеми тремя режимами переписывания."""
    try:
        results = {
            mode: (await rewrite.apply(mode, request.question.strip())).as_dict()
            for mode in rewrite.MODES
        }
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {"question": request.question.strip(), "rewrites": results}


@app.post("/api/rerank")
async def api_rerank(request: RerankRequest) -> dict[str, object]:
    """Один вопрос всеми реранкерами: пул один, судьба кандидатов разная."""
    if request.retriever not in retrieve.RETRIEVERS:
        raise HTTPException(status_code=404, detail=f"Ретривера {request.retriever!r} нет.")

    try:
        results = {}
        for name in rerank.RERANKERS:
            plan = Plan.of(
                reranker=name,
                retriever=request.retriever,
                pool=request.pool,
                top_k=request.top_k,
                threshold=request.threshold,
            )
            context = await pipeline.prepare(request.question.strip(), plan)
            results[name] = context.as_dict()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {"question": request.question.strip(), "results": results}


@app.get("/api/chunk/{chunk_id}")
async def api_chunk(chunk_id: int) -> dict[str, object]:
    found = await run_in_threadpool(index.chunk, chunk_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"Чанка #{chunk_id} в индексе нет.")
    return found


# --- замеры ------------------------------------------------------------------


@app.post("/api/matrix")
async def api_matrix() -> dict[str, object]:
    """Матрица переписывание × реранкер на пробах day21."""
    try:
        return await probes.compare()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/sweep")
async def api_sweep(request: SweepRequest) -> dict[str, object]:
    """Развертка по порогу отсечения. После прогрева кэшей — без сети."""
    if request.rewriter not in rewrite.MODES:
        raise HTTPException(status_code=404, detail=f"Режима переписывания {request.rewriter!r} нет.")

    try:
        return await probes.sweep(rewriter=request.rewriter)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/questions")
async def api_questions(request: QuestionsRequest) -> dict[str, object]:
    """Контрольный набор во всех пяти режимах. Единственная тяжёлая ручка с ключом."""
    try:
        return await evaluate.run(
            rewriter=request.rewriter,
            reranker=request.reranker,
            threshold=request.threshold,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
