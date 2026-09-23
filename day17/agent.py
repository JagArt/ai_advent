"""Ход с включённым и выключенным MCP: разница только в инструментах.

Системный промпт, температура и лимит у обоих режимов одни и те же — иначе
сравнивать было бы нечего. Выключенный MCP не поднимает сервер вовсе: «выключен»
должно значить «сервера нет», а не «есть, но им не пользуются».
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from openai.types.chat import ChatCompletionMessageParam

from client import as_openai_tools, connect, result_to_text
from llm import ToolCall, stream_chat

SYSTEM_PROMPT = (
    "Ты отвечаешь на вопросы про git-историю репозитория AI Advent. "
    "Отвечай по-русски, коротко и по делу. "
    "Не выдумывай коммиты, авторов и файлы: если не знаешь, так и скажи."
)
TEMPERATURE = 0.3
MAX_TOKENS = 1200
MAX_ROUNDS = 4


@dataclass(frozen=True)
class Connected:
    """Рукопожатие состоялось: что за сервер и что он умеет."""

    info: dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    text: str


@dataclass(frozen=True)
class ToolRun:
    name: str
    arguments: dict[str, Any]
    result: str
    is_error: bool
    elapsed_ms: int


@dataclass(frozen=True)
class Done:
    rounds: int
    calls: int
    finish_reason: str | None


Event = Connected | Chunk | ToolRun | Done


@dataclass
class Round:
    """Что модель наговорила за один проход."""

    text: str = ""
    finish_reason: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)


async def _run_tool(session: Any, call: ToolCall) -> ToolRun:
    """Неудача вызова — тоже ответ: модель получит её текстом и сможет исправиться."""
    started = perf_counter()

    def elapsed() -> int:
        return round((perf_counter() - started) * 1000)

    try:
        arguments = json.loads(call.arguments or "{}")
    except json.JSONDecodeError as exc:
        return ToolRun(call.name, {}, f"Аргументы не разобрались как JSON: {exc}", True, elapsed())

    try:
        result = await session.call_tool(call.name, arguments)
    except Exception as exc:
        return ToolRun(call.name, arguments, f"Вызов не прошёл: {exc}", True, elapsed())

    return ToolRun(
        name=call.name,
        arguments=arguments,
        result=result_to_text(result),
        is_error=bool(result.is_error),
        elapsed_ms=elapsed(),
    )


def _assistant_message(round_: Round) -> ChatCompletionMessageParam:
    return {
        "role": "assistant",
        "content": round_.text,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in round_.tool_calls
        ],
    }


async def answer(prompt: str, *, use_mcp: bool) -> AsyncIterator[Event]:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    if not use_mcp:
        round_ = Round()
        async for delta in _pass(messages, tools=None):
            if isinstance(delta, Chunk):
                yield delta
            else:
                round_ = delta
        yield Done(rounds=1, calls=0, finish_reason=round_.finish_reason)
        return

    async with connect() as connection:
        yield Connected(connection.info())

        tools = as_openai_tools(connection.tools)
        calls = 0

        for number in range(1, MAX_ROUNDS + 1):
            round_ = Round()
            async for delta in _pass(messages, tools=tools):
                if isinstance(delta, Chunk):
                    yield delta
                else:
                    round_ = delta

            if not round_.tool_calls:
                yield Done(rounds=number, calls=calls, finish_reason=round_.finish_reason)
                return

            messages.append(_assistant_message(round_))

            for call in round_.tool_calls:
                run = await _run_tool(connection.session, call)
                calls += 1
                yield run
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": run.result}
                )

        # Потолок раундов: лучше честный итог, чем бесконечное хождение за данными.
        yield Done(rounds=MAX_ROUNDS, calls=calls, finish_reason="max_rounds")


async def _pass(
    messages: list[ChatCompletionMessageParam],
    *,
    tools: list[dict[str, Any]] | None,
) -> AsyncIterator[Chunk | Round]:
    """Отдаёт куски текста по мере их появления, а последним — итог прохода."""
    round_ = Round()

    async for delta in stream_chat(
        messages,
        tools=tools,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    ):
        if delta.content:
            round_.text += delta.content
            yield Chunk(delta.content)
        if delta.finish_reason:
            round_.finish_reason = delta.finish_reason
            round_.tool_calls = delta.tool_calls

    yield round_
