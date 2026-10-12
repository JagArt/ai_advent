"""Приватный HTTP-сервис перед локальной моделью — API и страница чата на одном порту.

LM Studio остаётся на 127.0.0.1:1234 и в сеть не смотрит. Наружу — этот процесс.
В браузере вводят `http://<ip-машины>:8780`, получают страницу чата, и она ходит
на тот же адрес. Внешним клиентам — OpenAI-совместимый `/v1` под ключом.

Два входа, один путь допуска (`admit`): частота по идентификатору клиента, обрезка
истории под потолок контекста, очередь в один слот. Для `/v1` идентификатор —
ключ, для страницы — IP, с которого она открыта. Запрос, не прошедший частоту,
слот не занимает. Слот держится всё время генерации и отпускается в `finally`,
даже если клиент оборвал стрим. Диалоги на диск не пишутся.

Слушать не-loopback с ключом `day30-local` процесс отказывается — и из
`python day30/server.py`, и из `uvicorn server:app`.
"""

import json
import secrets
import socket
import subprocess
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from starlette.staticfiles import StaticFiles

import config
from limits import QueueFull, RateLimiter, SlotQueue, Trimmed, trim

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
NO_CACHE = {"Cache-Control": "no-cache"}
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

PAGE_SYSTEM = "Ты локальный помощник. Отвечай по-русски, коротко и по делу."
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

_upstream = AsyncOpenAI(api_key="lm-studio", base_url=config.LMSTUDIO_BASE_URL, timeout=180)
_models_cache: tuple[float, list[str] | None, str | None] | None = None

limiter = RateLimiter(config.RATE_LIMIT, config.RATE_WINDOW)
queue = SlotQueue(config.SLOTS, config.QUEUE_DEPTH)


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "no-cache"
        return response


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0, le=2)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class PageRequest(BaseModel):
    messages: list[Turn] = Field(min_length=1)
    think: bool = False


def ensure_private() -> None:
    if config.refuses_public_default():
        raise SystemExit(
            "SERVICE_HOST открыт наружу, а SERVICE_API_KEY всё ещё day30-local. "
            "Задайте свой ключ, прежде чем слушать сеть."
        )


def require_key(authorization: str | None = Header(default=None)) -> str:
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token or not secrets.compare_digest(token, config.SERVICE_API_KEY):
        raise HTTPException(status_code=401, detail="Нужен заголовок Authorization: Bearer и верный ключ.")
    return token


def client_id(request: Request) -> str:
    """Идентификатор страницы для окна частоты: IP, с которого она открыта."""
    host = request.client.host if request.client else "unknown"
    return f"ip:{host}"


def rate_headers(remaining: int) -> dict[str, str]:
    return {
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Limit": str(config.RATE_LIMIT),
    }


def lan_addresses() -> list[str]:
    """Адреса этой машины в локальной сети, самые вероятные первыми.

    Маршрут по умолчанию может уходить в VPN, и тогда UDP-проба показала бы туннель,
    а не Wi-Fi. Поэтому сначала читаются интерфейсы (`ifconfig` на macOS и BSD,
    `ip -4 -o addr` на Linux), туннели `a --> b` и loopback отбрасываются, домашние
    диапазоны 192.168/16 и 172.16/12 идут раньше 10/8. Проба — только запасной путь.
    """
    found: list[str] = []
    for command in (("ifconfig",), ("ip", "-4", "-o", "addr")):
        try:
            output = subprocess.run(command, capture_output=True, text=True, timeout=2, check=False).stdout
        except (OSError, subprocess.TimeoutExpired):
            continue
        for line in output.splitlines():
            if "-->" in line or "inet " not in line:
                continue
            address = line.split("inet ", 1)[1].split()[0].split("/", 1)[0]
            if address.startswith("127.") or address.startswith("169.254."):
                continue
            if address not in found:
                found.append(address)
        if found:
            break

    if not found:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.connect(("10.255.255.255", 1))
                found.append(probe.getsockname()[0])
        except OSError:
            pass

    def rank(address: str) -> int:
        if address.startswith("192.168."):
            return 0
        second = int(address.split(".")[1]) if address.count(".") == 3 and address.split(".")[1].isdigit() else -1
        if address.startswith("172.") and 16 <= second <= 31:
            return 1
        if address.startswith("10."):
            return 2
        return 3

    return sorted(found, key=rank)


async def upstream_models() -> list[str]:
    """Список моделей LM Studio. Удачный ответ кэшируется на три секунды, ошибка — на одну:
    плашка статуса опрашивает сервис каждые несколько секунд, а список за это время не меняется.
    """
    global _models_cache
    now = time.monotonic()
    if _models_cache is not None:
        moment, cached, error = _models_cache
        fresh = 3 if cached is not None else 1
        if now - moment < fresh and cached is not None:
            return cached
        if now - moment < fresh and cached is None:
            raise RuntimeError(error)

    try:
        page = await _upstream.with_options(timeout=3).models.list()
    except Exception as exc:
        _models_cache = (now, None, str(exc))
        raise RuntimeError(str(exc)) from exc
    names = [model.id for model in page.data]
    _models_cache = (now, names, None)
    return names


@dataclass(frozen=True)
class Admission:
    """Запрос прошёл частоту, уложен в бюджет и занял слот. Слот отдаёт вызывающий."""

    trimmed: Trimmed
    max_tokens: int
    waited_s: float
    remaining: int

    def headers(self) -> dict[str, str]:
        return {
            "X-Queue-Wait-Ms": str(round(self.waited_s * 1000)),
            "X-Context-Dropped": str(self.trimmed.dropped),
            "X-Context-Truncated": "1" if self.trimmed.truncated else "0",
            "X-Context-Tokens": str(self.trimmed.tokens),
            **rate_headers(self.remaining),
        }

    def meta(self) -> dict[str, int]:
        return {
            "queue_wait_ms": round(self.waited_s * 1000),
            "context_dropped": self.trimmed.dropped,
            "context_truncated": 1 if self.trimmed.truncated else 0,
            "context_tokens": self.trimmed.tokens,
            "rate_remaining": self.remaining,
            "rate_limit": config.RATE_LIMIT,
        }


async def admit(client: str, messages: list[dict[str, str]], max_tokens: int | None) -> Admission:
    """Частота → обрезка → очередь. Любой отказ — `HTTPException`, слот при этом не занят."""
    decision = limiter.check(client)
    if not decision.allowed:
        retry = max(1, int(decision.retry_after + 0.999))
        raise HTTPException(
            status_code=429,
            detail=f"Слишком много запросов. Повторите через {retry} с.",
            headers={"Retry-After": str(retry), **rate_headers(0)},
        )

    ceiling = min(max_tokens or config.MAX_TOKENS, config.MAX_TOKENS)
    try:
        trimmed = trim(messages, max_context=config.MAX_CONTEXT, max_tokens=ceiling)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        waited = await queue.enter()
    except QueueFull:
        raise HTTPException(
            status_code=503,
            detail=f"Очередь заполнена: уже ждут {config.QUEUE_DEPTH}. Повторите чуть позже.",
            headers=rate_headers(decision.remaining),
        ) from None

    return Admission(trimmed=trimmed, max_tokens=ceiling, waited_s=waited, remaining=decision.remaining)


def service_status() -> dict[str, object]:
    return {
        "host": config.SERVICE_HOST,
        "port": config.SERVICE_PORT,
        "model": config.MODEL,
        "slots": config.SLOTS,
        "waiting": queue.waiting,
        "queue_limit": config.QUEUE_DEPTH,
        "rate_limit": config.RATE_LIMIT,
        "rate_window_s": config.RATE_WINDOW,
        "max_context": config.MAX_CONTEXT,
        "max_tokens": config.MAX_TOKENS,
        "default_key": config.SERVICE_API_KEY == config.DEFAULT_API_KEY,
    }


async def upstream_status() -> dict[str, object]:
    try:
        names = await upstream_models()
    except Exception as exc:
        return {"online": False, "error": f"LM Studio не отвечает: {exc}"}
    return {
        "online": True,
        "base_url": config.LMSTUDIO_BASE_URL,
        "models": names,
        "model_available": config.MODEL in names,
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_private()
    yield


app = FastAPI(title="Приватный сервис локальной LLM", lifespan=lifespan)
app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


# --- Страница чата: тот же порт, без ключа, частота по IP ---------------------


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=NO_CACHE)


@app.get("/api/status")
async def page_status(request: Request) -> dict[str, object]:
    upstream = await upstream_status()
    return {
        "service": {
            **service_status(),
            "online": True,
            "client": request.client.host if request.client else None,
            "rate_remaining": limiter.peek(client_id(request)),
            "upstream_online": bool(upstream.get("online")),
            "model_available": bool(upstream.get("model_available")),
            "upstream_error": upstream.get("error"),
        }
    }


def sse(data: object, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data, ensure_ascii=False)}\n\n"


def page_messages(turns: list[Turn], think: bool) -> list[dict[str, str]]:
    """Системный промпт страницы плюс переписка; без галочки «думать» — `/no_think` к последней реплике."""
    copied = [turn.model_dump() for turn in turns]
    if not think:
        for message in reversed(copied):
            if message["role"] == "user":
                text = message["content"].rstrip()
                if not text.endswith("/no_think"):
                    message["content"] = f"{text} /no_think"
                break
    return [{"role": "system", "content": PAGE_SYSTEM}, *copied]


class _ThinkSplitter:
    """Разводит `<think>` и ответ, если рассуждения пришли внутри content."""

    def __init__(self) -> None:
        self.inside = False

    def feed(self, text: str) -> tuple[str, str]:
        reasoning: list[str] = []
        content: list[str] = []
        while text:
            tag = THINK_CLOSE if self.inside else THINK_OPEN
            head, found, text = text.partition(tag)
            (reasoning if self.inside else content).append(head)
            if found:
                self.inside = not self.inside
        return "".join(reasoning), "".join(content)


async def page_stream(admission: Admission) -> AsyncIterator[str]:
    """Кадры страницы: meta, reasoning, data, done. Слот отпускается здесь, в `finally`."""
    started = time.perf_counter()
    first_token: float | None = None
    parts: list[str] = []
    splitter = _ThinkSplitter()
    usage: dict[str, int] | None = None
    finish: str | None = None

    try:
        yield sse(admission.meta(), event="meta")
        upstream = await _upstream.chat.completions.create(
            model=config.MODEL,
            messages=admission.trimmed.messages,
            max_tokens=admission.max_tokens,
            temperature=0.7,
            stream=True,
            stream_options={"include_usage": True},
        )
        async with upstream:
            async for chunk in upstream:
                if chunk.usage:
                    usage = _usage(chunk.usage)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    finish = choice.finish_reason
                reasoning, content = splitter.feed(choice.delta.content or "")
                reasoning = (getattr(choice.delta, "reasoning_content", None) or "") + reasoning
                if reasoning:
                    yield sse(reasoning, event="reasoning")
                if content:
                    first_token = first_token or time.perf_counter()
                    parts.append(content)
                    yield sse(content)
    except Exception as exc:
        yield sse(f"LM Studio не ответил: {exc}", event="error")
        return
    finally:
        await queue.leave()

    finished = time.perf_counter()
    yield sse(
        {
            **admission.meta(),
            "model": config.MODEL,
            "text": "".join(parts).strip(),
            "finish_reason": finish,
            "ttft_ms": None if first_token is None else round((first_token - started) * 1000),
            "total_ms": round((finished - started) * 1000),
            "prompt_tokens": None if usage is None else usage.get("prompt_tokens"),
            "completion_tokens": None if usage is None else usage.get("completion_tokens"),
        },
        event="done",
    )


@app.post("/api/chat")
async def page_chat(request: PageRequest, http: Request) -> StreamingResponse:
    try:
        admission = await admit(client_id(http), page_messages(request.messages, request.think), None)
    except HTTPException as exc:
        # Отказ — тоже кадр потока: страница читает один формат, а не два.
        return StreamingResponse(
            iter([sse(str(exc.detail), event="error")]),
            media_type="text/event-stream",
            headers={**SSE_HEADERS, **(exc.headers or {})},
        )
    return StreamingResponse(page_stream(admission), media_type="text/event-stream", headers=SSE_HEADERS)


# --- Внешний API: OpenAI-совместимый, под ключом -----------------------------


@app.get("/health")
async def health() -> dict[str, object]:
    return {"ok": True, **service_status(), "upstream": await upstream_status()}


@app.get("/v1/models")
async def models(token: str = Depends(require_key)) -> JSONResponse:
    headers = rate_headers(limiter.peek(token))
    try:
        names = await upstream_models()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LM Studio не отвечает: {exc}", headers=headers) from exc
    return JSONResponse(
        {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "local"} for name in names]},
        headers=headers,
    )


def _usage(raw: object) -> dict[str, int] | None:
    if raw is None:
        return None
    prompt = getattr(raw, "prompt_tokens", None)
    completion = getattr(raw, "completion_tokens", None)
    total = getattr(raw, "total_tokens", None)
    if prompt is None and completion is None:
        return None
    return {
        "prompt_tokens": prompt or 0,
        "completion_tokens": completion or 0,
        "total_tokens": total if total is not None else (prompt or 0) + (completion or 0),
    }


def _chunk(
    completion_id: str,
    created: int,
    delta: dict[str, str],
    finish: str | None,
    usage: dict[str, int] | None,
) -> str:
    payload: dict[str, object] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": config.MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream(completion_id: str, created: int, upstream: object) -> AsyncIterator[str]:
    try:
        usage: dict[str, int] | None = None
        finish: str | None = None
        async with upstream:  # type: ignore[attr-defined]
            async for chunk in upstream:  # type: ignore[attr-defined]
                if chunk.usage:
                    usage = _usage(chunk.usage)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta_raw = choice.delta
                delta: dict[str, str] = {}
                if delta_raw.role:
                    delta["role"] = delta_raw.role
                if delta_raw.content:
                    delta["content"] = delta_raw.content
                reasoning = getattr(delta_raw, "reasoning_content", None) or ""
                if reasoning:
                    delta["reasoning_content"] = reasoning
                if choice.finish_reason:
                    finish = choice.finish_reason
                if delta or choice.finish_reason:
                    yield _chunk(completion_id, created, delta, choice.finish_reason, None)
        yield _chunk(completion_id, created, {}, finish, usage)
        yield "data: [DONE]\n\n"
    finally:
        await queue.leave()


def _completion(
    completion_id: str,
    created: int,
    text: str,
    finish: str | None,
    usage: dict[str, int] | None,
) -> dict[str, object]:
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": config.MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }
        ],
        "usage": usage,
    }


@app.post("/v1/chat/completions", response_model=None)
async def chat(request: ChatRequest, token: str = Depends(require_key)) -> StreamingResponse | JSONResponse:
    admission = await admit(token, [message.model_dump() for message in request.messages], request.max_tokens)
    headers = admission.headers()
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    payload = {
        "model": config.MODEL,
        "messages": admission.trimmed.messages,
        "max_tokens": admission.max_tokens,
        "temperature": 0.7 if request.temperature is None else request.temperature,
    }

    try:
        if request.stream:
            upstream = await _upstream.chat.completions.create(
                stream=True,
                stream_options={"include_usage": True},
                **payload,
            )
            return StreamingResponse(
                _stream(completion_id, created, upstream),
                media_type="text/event-stream",
                headers={**headers, **SSE_HEADERS},
            )
        completion = await _upstream.chat.completions.create(**payload)
        choice = completion.choices[0]
        body = _completion(
            completion_id,
            created,
            choice.message.content or "",
            choice.finish_reason,
            _usage(completion.usage),
        )
    except Exception as exc:
        await queue.leave()
        raise HTTPException(status_code=502, detail=f"LM Studio не ответил: {exc}") from exc

    await queue.leave()
    return JSONResponse(body, headers=headers)


def main() -> None:
    ensure_private()
    print(f"Страница и API: http://127.0.0.1:{config.SERVICE_PORT}", flush=True)
    if config.SERVICE_HOST not in config.LOOPBACK_HOSTS:
        for address in lan_addresses():
            print(f"В сети:         http://{address}:{config.SERVICE_PORT}", flush=True)
    else:
        print("Только эта машина. Для сети: SERVICE_HOST=0.0.0.0 и свой SERVICE_API_KEY.", flush=True)
    uvicorn.run(app, host=config.SERVICE_HOST, port=config.SERVICE_PORT)


if __name__ == "__main__":
    main()
