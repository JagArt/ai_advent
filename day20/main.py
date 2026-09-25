"""Веб-слой: один запрос, два способа пройти длинный флоу по трём серверам.

`/api/flow` в режиме `flow` проходит контракт из [flow.py](flow.py) сам и отдаёт
кадры шагов по ходу. В режиме `agent` тот же запрос уходит модели, и она выбирает
инструменты из объединённого каталога сама — кадры те же, что в чате, а в конце
приезжает проверка порядка вызовов.

Соединения со всеми серверами живут ровно столько, сколько идёт ответ: реестр
открывается внутри генератора SSE и закрывается вместе с ним.
"""

import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

import flow
import registry as reg
from agent import Call, Chunk, Connected, Done, answer

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
OUT_DIR = BASE_DIR / "out"
NO_CACHE = {"Cache-Control": "no-cache"}
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "no-cache"
        return response


app = FastAPI(title="MCP — реестр серверов и маршрутизация")
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


class FlowRequest(BaseModel):
    query: str = Field(min_length=2)
    limit: int = Field(default=4, ge=1, le=10)
    bullets: int = Field(default=5, ge=3, le=10)
    name: str | None = None
    mode: str = Field(default="flow", pattern="^(flow|agent)$")

    def task(self) -> flow.Task:
        return flow.Task(
            query=self.query, limit=self.limit, bullets=self.bullets, name=self.name or None
        )


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1)


def sse_frame(data: object, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/api/contract")
async def api_contract() -> dict[str, object]:
    """Контракт длинного флоу: страница рисует шаги до того, как что-то запущено."""
    return {"steps": flow.contract(), "servers": list(flow.SERVERS_IN_ORDER)}


@app.get("/api/servers")
async def api_servers() -> dict[str, object]:
    """Рукопожатие со всеми серверами реестра и объединённый каталог, без модели."""
    try:
        async with reg.connect() as registry:
            return registry.info()
    except Exception as exc:
        return {"error": reg.reason(exc)}


@app.get("/api/runs")
async def api_runs(limit: int = 8) -> dict[str, object]:
    return {"runs": reg.list_runs(limit)}


@app.get("/api/routes")
async def api_routes(limit: int = 25) -> dict[str, object]:
    """Журнал маршрутизации: каждая попытка вызова, включая отказы реестра."""
    return {"routes": reg.list_routes(limit), "causes": reg.CAUSES}


@app.get("/api/artifacts/{artifact_id}")
async def api_artifact(artifact_id: int) -> dict[str, object]:
    try:
        return reg.get_artifact(artifact_id)
    except reg.Refused as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/files/{filename}")
async def api_file(filename: str) -> PlainTextResponse:
    """Файл, который записал vault__save_file. Имя берётся без папок, путь проверяется."""
    path = (OUT_DIR / Path(filename).name).resolve()
    if path.parent != OUT_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Файла {filename} в out/ нет.")

    return PlainTextResponse(path.read_text(encoding="utf-8"), headers=NO_CACHE)


async def flow_stream(request: FlowRequest) -> AsyncIterator[str]:
    """Контракт проходит реестр: номера артефактов в ссылках подставляет код."""
    task = request.task()
    run_id = reg.open_run(task.query, "flow")

    async with reg.connect() as registry:
        yield sse_frame({**registry.info(), "run_id": run_id}, event="mcp")

        async for frame in flow.run(registry, task, run_id=run_id):
            yield sse_frame(frame, event=frame["event"])


async def agent_stream(prompt: str, *, mode: str = "chat", query: str | None = None) -> AsyncIterator[str]:
    """Инструменты выбирает модель: кадры те же, что в чате, плюс проверка порядка."""
    async for event in answer(prompt, mode=mode, query=query):
        match event:
            case Chunk():
                yield sse_frame(event.text)
            case Connected():
                yield sse_frame({**event.info, "run_id": event.run_id}, event="mcp")
            case Call():
                yield sse_frame({"round": event.round, **event.frame}, event="call")
            case Done():
                yield sse_frame(asdict(event), event="done")


async def guarded(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """Ошибка любого шага должна доехать до страницы кадром, а не обрывом потока."""
    try:
        async for frame in stream:
            yield frame
    except Exception as exc:
        yield sse_frame(reg.reason(exc), event="error")


@app.post("/api/flow")
async def api_flow(request: FlowRequest) -> StreamingResponse:
    task = request.task()
    stream: AsyncIterator[str]
    if request.mode == "flow":
        stream = flow_stream(request)
    else:
        stream = agent_stream(flow.prompt(task), mode="agent", query=task.query)

    return StreamingResponse(
        guarded(stream), media_type="text/event-stream", headers=SSE_HEADERS
    )


@app.post("/api/ask")
async def api_ask(request: AskRequest) -> StreamingResponse:
    return StreamingResponse(
        guarded(agent_stream(request.prompt.strip())),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
