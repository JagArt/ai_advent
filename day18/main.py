"""Веб-агент: фоновый наблюдатель за сводками плюс чат с инструментами.

Наблюдатель работает, пока жив процесс, и страница ему не нужна: раз в
`POLL_SECONDS` он забирает новые сводки с MCP-сервера, отдаёт каждую модели
на пересказ и дописывает в `feed.json`. Открытая страница только подписывается
на ленту — закрыть вкладку не значит остановить агента.

Соединение с сервером — на каждый опрос, а не одно на всё время: сервер можно
перезапустить, и следующий опрос просто подключится заново.
"""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

from agent import Chunk, Connected, Done, ToolRun, answer, narrate
from client import URL, connect

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
FEED_PATH = BASE_DIR / "feed.json"
NO_CACHE = {"Cache-Control": "no-cache"}
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

POLL_SECONDS = 15
# Первый старт с пустой лентой: пересказываются только последние сводки, а не вся история.
BOOTSTRAP_SUMMARIES = 3
BATCH_SUMMARIES = 10
FEED_KEPT = 200
FEED_SENT = 50
PING_SECONDS = 20


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _reason(exc: BaseException) -> str:
    """Сбой транспорта приезжает группой исключений; человеку нужна первая настоящая причина."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return f"{type(exc).__name__}: {exc}"


class Feed:
    """Лента пересказов: файл на диске и подписчики страницы."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.items: list[dict[str, Any]] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        )
        self.status: dict[str, Any] = {"state": "starting", "detail": "", "checked_at": None}
        self.subscribers: set[asyncio.Queue[tuple[str, Any]]] = set()

    @property
    def last_id(self) -> int | None:
        return self.items[-1]["summary_id"] if self.items else None

    def add(self, item: dict[str, Any]) -> None:
        self.items = [*self.items, item][-FEED_KEPT:]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
        self._publish("item", item)

    def set_status(self, state: str, detail: str = "") -> None:
        changed = (state, detail) != (self.status["state"], self.status["detail"])
        self.status = {"state": state, "detail": detail, "checked_at": _now()}
        if changed:
            self._publish("status", self.status)

    def _publish(self, event: str, data: Any) -> None:
        for queue in self.subscribers:
            queue.put_nowait((event, data))


feed = Feed(FEED_PATH)


async def _poll_once() -> None:
    async with connect() as connection:
        if feed.last_id is None:
            arguments = {"after_id": 0, "limit": BOOTSTRAP_SUMMARIES}
        else:
            arguments = {"after_id": feed.last_id, "limit": BATCH_SUMMARIES}
        summaries = await connection.call("list_summaries", arguments)

    feed.set_status("online", f"сводок на пересказ: {len(summaries)}" if summaries else "")

    for summary in summaries:
        text, error = None, None
        try:
            text = await narrate(summary)
        except Exception as exc:
            # Сводка всё равно попадает в ленту: без пересказа, но с агрегатом и причиной.
            error = _reason(exc)

        feed.add(
            {
                "summary_id": summary["id"],
                "created_at": summary["created_at"],
                "period": summary["period"],
                "narrated_at": _now(),
                "text": text,
                "error": error,
                "summary": summary,
            }
        )
    if summaries:
        feed.set_status("online")


async def watch() -> None:
    while True:
        try:
            await _poll_once()
        except Exception as exc:
            feed.set_status("offline", f"{URL} недоступен — {_reason(exc)}")
        await asyncio.sleep(POLL_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    task = asyncio.create_task(watch())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(title="MCP — планировщик и сводки", lifespan=lifespan)
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


def sse_frame(data: object, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/api/connect")
async def api_connect() -> dict[str, object]:
    """Только соединение и список инструментов, без модели."""
    try:
        async with connect() as connection:
            return connection.info()
    except Exception as exc:
        return {"error": _reason(exc)}


@app.get("/api/jobs")
async def api_jobs() -> dict[str, object]:
    try:
        async with connect() as connection:
            return {"jobs": await connection.call("list_jobs")}
    except Exception as exc:
        return {"error": _reason(exc)}


async def feed_stream() -> AsyncIterator[str]:
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    feed.subscribers.add(queue)
    try:
        yield sse_frame(feed.status, event="status")
        for item in feed.items[-FEED_SENT:]:
            yield sse_frame(item, event="item")

        while True:
            try:
                event, data = await asyncio.wait_for(queue.get(), timeout=PING_SECONDS)
            except TimeoutError:
                yield ": ping\n\n"
                continue
            yield sse_frame(data, event=event)
    finally:
        feed.subscribers.discard(queue)


@app.get("/api/feed")
async def api_feed() -> StreamingResponse:
    return StreamingResponse(feed_stream(), media_type="text/event-stream", headers=SSE_HEADERS)


async def event_stream(prompt: str) -> AsyncIterator[str]:
    try:
        async for event in answer(prompt):
            match event:
                case Chunk():
                    yield sse_frame(event.text)
                case Connected():
                    yield sse_frame(event.info, event="mcp")
                case ToolRun():
                    yield sse_frame(asdict(event), event="tool")
                case Done():
                    yield sse_frame(asdict(event), event="done")
    except Exception as exc:
        yield sse_frame(_reason(exc), event="error")


@app.post("/api/ask")
async def ask(request: AskRequest) -> StreamingResponse:
    return StreamingResponse(
        event_stream(request.prompt.strip()),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
