"""Ход файлового ассистента: чистый агент — всё, что он знает, он берёт инструментами.

Код не кладёт в промпт содержимого файлов. Системное сообщение несёт роль и
указатели: какие имена сейчас есть в workspace, и что текст надо взять через
`fs_list`, `fs_search`, `fs_read`. Дальше модель решает сама, что открыть и
что переписать через `fs_write`. Это тот же приём, что в
[day33](../day33/agent.py): указатели есть, данных нет.

Если первый раунд кончился без единого вызова, ответ отбрасывается, в диалог
добавляется толчок — «ты не читал файлы», и даётся ещё раунд. Толчок один на
ход, он виден в ленте и считается в итоге.

В `Done` код сам кладёт текущий diff относительно эталона и список путей,
которые модель открывала или меняла, — странице не нужно делать второй запрос.
"""

import json
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai.types.chat import ChatCompletionMessageParam

import llm
import workspace

SERVER_PATH = Path(__file__).resolve().parent / "server.py"

TEMPERATURE = 0.2
MAX_TOKENS = 1800
HISTORY_TURNS = 6

MAX_ROUNDS = {"local": 5, "cloud": 6}
TOOL_RESULT_CHARS = {"local": 3500, "cloud": 9000}

PRESETS = (
    "Найди все места, где используется create_note, и обнови README так, чтобы раздел API совпадал с кодом",
    "Проверь проект на правила из RULES.md и запиши отчёт в AUDIT.md",
)

SYSTEM_PROMPT = (
    "Ты — файловый ассистент. Тебе дали цель по мини-проекту в песочнице workspace/. "
    "Содержимого файлов ты не знаешь. Имена, которые сейчас есть: {files}. "
    "Список даёт fs_list, поиск по нескольким файлам — fs_search, чтение — fs_read, "
    "создание и полная перезапись — fs_write, проверка правок — fs_diff. "
    "Не выдумывай файлы и не отвечай по догадке. Пиши и читай только через инструменты. "
    "Отвечай по-русски. Сейчас {now}."
)

RULES = (
    "Правила работы:\n"
    "1. Сначала инструменты: посмотри дерево (fs_list) или сразу ищи (fs_search), "
    "прочитай нужные файлы (fs_read). Без них не отвечай.\n"
    "2. Цель дана словами, не путями. Сам реши, какие файлы открыть.\n"
    "3. Если цель требует изменить или создать файл — сделай это через fs_write "
    "(полная новая версия файла целиком), затем проверь fs_diff.\n"
    "4. Не выдумывай содержимое. Пути и фрагменты бери только из ответов инструментов.\n"
    "5. В конце кратко скажи, что нашёл или что изменил, и перечисли пути."
)

NUDGE = (
    "Ты ответил, не вызвав ни одного инструмента, — значит, не читал файлы проекта. "
    "Этот ответ отброшен. Вызови fs_list или fs_search, прочитай нужные файлы через "
    "fs_read, и если цель требует правок — fs_write, затем ответь заново."
)


@dataclass(frozen=True)
class Connection:
    """Сессия с файловым сервером и то, что он о себе рассказал при рукопожатии."""

    session: ClientSession
    server: str
    protocol: str
    tools: list[types.Tool]
    elapsed_ms: int

    @property
    def command(self) -> str:
        return f"{Path(sys.executable).name} day34/{SERVER_PATH.name}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "server": self.server,
            "protocol": self.protocol,
            "command": self.command,
            "elapsed_ms": self.elapsed_ms,
            "tools": [
                {"name": tool.name, "description": (tool.description or "").strip().splitlines()[0]}
                for tool in self.tools
            ],
        }

    def openai_tools(self, provider: str) -> list[dict[str, Any]]:
        """`inputSchema` MCP кладётся в `parameters` как есть: это один и тот же JSON Schema.

        Локальной модели — описания в одну строку и без `title` у полей: с полным
        каталогом qwen3-4b в day31 «залипала».
        """
        tools = []
        for tool in self.tools:
            description = (tool.description or "").strip()
            schema = json.loads(json.dumps(tool.input_schema))
            if provider == "local":
                description = description.splitlines()[0]
                for prop in schema.get("properties", {}).values():
                    prop.pop("title", None)
                    if "description" in prop:
                        prop["description"] = prop["description"].split(". ")[0].rstrip(".")[:90]
            tools.append(
                {
                    "type": "function",
                    "function": {"name": tool.name, "description": description, "parameters": schema},
                }
            )
        return tools


@asynccontextmanager
async def connect() -> AsyncIterator[Connection]:
    """Подпроцесс сервера живёт ровно столько, сколько открыт этот блок."""
    workspace.ensure()
    started = time.perf_counter()
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER_PATH)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            listed = await session.list_tools()
            yield Connection(
                session=session,
                server=f"{init.server_info.name} {init.server_info.version}".strip(),
                protocol=init.protocol_version,
                tools=listed.tools,
                elapsed_ms=round((time.perf_counter() - started) * 1000),
            )


@dataclass(frozen=True)
class ToolResult:
    """Один вызов инструмента: что просили, что вернулось, сколько заняло."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    payload: Any
    text: str
    ms: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "ms": self.ms,
            "error": None if self.ok else self.text,
        }


def _result_text(result: types.CallToolResult) -> str:
    text = "\n".join(block.text for block in result.content if isinstance(block, types.TextContent))
    if not text and result.structured_content is not None:
        text = json.dumps(result.structured_content, ensure_ascii=False)
    return text or "(инструмент вернул пустой ответ)"


def _payload(result: types.CallToolResult) -> Any:
    payload = result.structured_content
    if payload is None:
        try:
            return json.loads(_result_text(result))
        except json.JSONDecodeError:
            return {"text": _result_text(result)}
    if isinstance(payload, dict) and set(payload) == {"result"}:
        return payload["result"]
    return payload


async def call(session: ClientSession, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
    """Единственная дверь к серверу. Отказ инструмента — не исключение, а результат с `ok=False`."""
    arguments = arguments or {}
    started = time.perf_counter()
    try:
        result = await session.call_tool(name, arguments)
    except Exception as exc:  # noqa: BLE001
        text = f"Сервер не ответил: {type(exc).__name__}: {exc}"
        return ToolResult(name, arguments, False, None, text, round((time.perf_counter() - started) * 1000))

    ms = round((time.perf_counter() - started) * 1000)
    if result.is_error:
        return ToolResult(name, arguments, False, None, _result_text(result).removeprefix(f"Error executing tool {name}: "), ms)
    return ToolResult(name, arguments, True, _payload(result), _result_text(result), ms)


def _remember(result: ToolResult, touched: list[str]) -> None:
    """Собирает пути, которые модель открывала, искала или меняла."""
    if not result.ok or not isinstance(result.payload, dict):
        return

    def add(path: str) -> None:
        if path and path not in touched:
            touched.append(path)

    if result.name == "fs_list":
        return
    if isinstance(result.payload.get("path"), str):
        add(result.payload["path"])
    for hit in result.payload.get("hits") or []:
        if isinstance(hit, dict) and isinstance(hit.get("path"), str):
            add(hit["path"])
    for path in result.payload.get("changed") or []:
        if isinstance(path, str):
            add(path)


def messages_for(question: str, history: list[dict[str, str]]) -> list[ChatCompletionMessageParam]:
    """Системное сообщение с указателями, история прошлых ходов, цель с правилами."""
    files = ", ".join(workspace.names()) or "(пусто — вызови fs_list после сброса)"
    system = SYSTEM_PROMPT.format(files=files, now=datetime.now().astimezone().isoformat(timespec="minutes"))
    messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]
    for turn in history[-HISTORY_TURNS * 2 :]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})  # type: ignore[arg-type]
    messages.append({"role": "user", "content": f"Цель: {question}\n\n{RULES}\n\nВыполни цель: {question}"})
    return messages


@dataclass(frozen=True)
class Chunk:
    text: str


@dataclass(frozen=True)
class Reasoning:
    text: str


@dataclass(frozen=True)
class ToolCalled:
    round: int
    result: ToolResult


@dataclass(frozen=True)
class Nudged:
    """Первый раунд прошёл без вызовов: черновик отброшен, модели дан толчок."""

    round: int
    draft: str


@dataclass(frozen=True)
class Cleared:
    """Раунд кончился вызовами: текст до них — комментарий к инструментам, не ответ."""

    round: int


@dataclass(frozen=True)
class Done:
    provider: str
    model: str
    text: str
    files_touched: list[str]
    changed: list[str]
    diff: str
    rounds: int
    calls: int
    tools_used: list[str]
    nudged: bool
    finish_reason: str | None
    ttft_ms: int | None
    total_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "title": llm.TITLES[self.provider],
            "text": self.text,
            "files_touched": self.files_touched,
            "changed": self.changed,
            "diff": self.diff,
            "rounds": self.rounds,
            "calls": self.calls,
            "tools_used": self.tools_used,
            "nudged": self.nudged,
            "finish_reason": self.finish_reason,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


Event = Chunk | Reasoning | ToolCalled | Nudged | Cleared | Done


def _assistant_message(text: str, calls: tuple[llm.ToolCall, ...]) -> ChatCompletionMessageParam:
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
            for call in calls
        ]
    return message  # type: ignore[return-value]


def _tool_message(call: llm.ToolCall, result: ToolResult, provider: str) -> ChatCompletionMessageParam:
    text = result.text
    limit = TOOL_RESULT_CHARS[provider]
    if len(text) > limit:
        text = text[:limit] + "\n[ответ инструмента обрезан]"
    return {"role": "tool", "tool_call_id": call.id, "content": text}


async def answer(
    connection: Connection,
    *,
    question: str,
    provider: str,
    history: list[dict[str, str]] | None = None,
) -> AsyncIterator[Event]:
    """Ход целиком: промпт с указателями, раунды с инструментами, толчок, итог с diff."""
    started = time.perf_counter()
    messages = messages_for(question, history or [])
    tools = connection.openai_tools(provider)
    first_token: float | None = None
    calls_made = 0
    tools_used: list[str] = []
    touched: list[str] = []
    nudged = False
    prompt_tokens: int | None = None
    completion_tokens = 0
    text_parts: list[str] = []
    finish: str | None = None

    for number in range(1, MAX_ROUNDS[provider] + 1):
        round_text: list[str] = []
        tool_calls: tuple[llm.ToolCall, ...] = ()

        async for delta in llm.stream_chat(
            messages, provider=provider, tools=tools, temperature=TEMPERATURE, max_tokens=MAX_TOKENS
        ):
            if delta.reasoning:
                yield Reasoning(delta.reasoning)
            if delta.content:
                first_token = first_token or time.perf_counter()
                round_text.append(delta.content)
                yield Chunk(delta.content)
            if delta.finish_reason:
                finish = delta.finish_reason
                tool_calls = delta.tool_calls
                prompt_tokens = delta.prompt_tokens if delta.prompt_tokens is not None else prompt_tokens
                completion_tokens += delta.completion_tokens or 0

        if not tool_calls and calls_made == 0 and not nudged:
            nudged = True
            draft = "".join(round_text).strip()
            yield Nudged(number, draft)
            messages.append(_assistant_message(draft, ()))
            messages.append({"role": "user", "content": NUDGE})
            first_token = None
            continue

        if tool_calls:
            if "".join(round_text).strip():
                yield Cleared(number)
            first_token = None
        else:
            text_parts.extend(round_text)
            break

        messages.append(_assistant_message("".join(round_text), tool_calls))
        for tool_call in tool_calls:
            try:
                arguments = json.loads(tool_call.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("аргументы должны быть объектом")
            except (json.JSONDecodeError, ValueError) as exc:
                result = ToolResult(tool_call.name, {}, False, None, f"Аргументы не разобрались как JSON: {exc}", 0)
            else:
                result = await call(connection.session, tool_call.name, arguments)
            calls_made += 1
            tools_used.append(result.name)
            _remember(result, touched)
            yield ToolCalled(number, result)
            messages.append(_tool_message(tool_call, result, provider))
    else:
        finish = "max_rounds"
        number = MAX_ROUNDS[provider]

    delta = workspace.diff()
    yield Done(
        provider=provider,
        model=llm.MODELS[provider],
        text="".join(text_parts).strip(),
        files_touched=touched,
        changed=delta["changed"],
        diff=delta["diff"],
        rounds=number,
        calls=calls_made,
        tools_used=tools_used,
        nudged=nudged,
        finish_reason=finish,
        ttft_ms=None if first_token is None else round((first_token - started) * 1000),
        total_ms=round((time.perf_counter() - started) * 1000),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens or None,
    )
