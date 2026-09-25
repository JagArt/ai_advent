"""Ход агента с инструментами пайплайна: тот же цикл, что в day17, другой промпт.

Здесь агент не только отвечает на вопросы, но и собирает цепочку сам, когда его
об этом просят. Вся разница с `run_pipeline` — в том, кто держит номера артефактов:
там их передаёт код, тут модель, и ей про это сказано прямо.

Потолок раундов выше, чем в day17: цепочка из трёх инструментов — это минимум
четыре прохода, и упираться в потолок на последнем шаге было бы обидно.
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Any

from openai.types.chat import ChatCompletionMessageParam

from client import as_openai_tools, connect, result_to_text
from llm import ToolCall, stream_chat

SYSTEM_PROMPT = (
    "Ты — агент над MCP-сервером, который умеет искать по документации проекта "
    "AI Advent, сводить найденное в тезисы и сохранять сводку файлом.\n"
    "Инструменты передают данные через артефакты: search возвращает artifact_id, "
    "его номер подаётся в summarize, номер сводки — в save_to_file. "
    "Никогда не переписывай найденный текст в аргументы: передавай только номера, "
    "иначе данные пойдут через тебя и изменятся.\n"
    "Если просят собрать сводку и сохранить её, пройди цепочку целиком: "
    "search → summarize → save_to_file, по одному вызову за раунд. "
    "Инструмент run_pipeline делает то же самое одним вызовом — зови его, "
    "только если об этом просят явно.\n"
    "Отвечай по-русски, коротко. Цифры, пути и номера артефактов бери из ответов "
    "инструментов, не придумывай их. Сейчас {now}."
)
TEMPERATURE = 0.3
MAX_TOKENS = 1200
MAX_ROUNDS = 6


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
    structured: Any | None = None


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
        structured=result.structured_content,
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


async def answer(prompt: str) -> AsyncIterator[Event]:
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(now=now)},
        {"role": "user", "content": prompt},
    ]

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
