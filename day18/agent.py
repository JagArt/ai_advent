"""Два режима агента: ход в чате и пересказ сводки, пришедшей по расписанию.

Ход в чате — цикл вызовов как в day17, только без тумблера: без сервера агенту
нечего сказать ни про расписание, ни про собранные снимки. Пересказ — один проход
без инструментов: агрегат уже посчитан сервером, модель только переводит его
с JSON на человеческий и не имеет права дописать то, чего в нём нет.
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
    "Ты — агент, который круглосуточно следит за git-репозиторием AI Advent. "
    "Фоновый планировщик на MCP-сервере по расписанию снимает состояние репозитория "
    "и сохраняет сводки; ты читаешь их инструментами и управляешь расписанием. "
    "Отвечай по-русски, коротко и по делу. "
    "Не выдумывай коммиты, авторов и цифры: бери их только из ответов инструментов. "
    "Сейчас {now}."
)
NARRATE_PROMPT = (
    "Тебе приходит JSON периодической сводки активности git-репозитория AI Advent. "
    "Перескажи его по-русски в 3–5 коротких строках: что закоммичено и кем, "
    "в каких папках идёт работа, как изменилось рабочее дерево за период. "
    "Если за период ничего не менялось, скажи это одной строкой. "
    "Бери цифры только из JSON, ничего не добавляй. Без заголовков и markdown."
)
TEMPERATURE = 0.3
MAX_TOKENS = 1200
NARRATE_MAX_TOKENS = 400
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
            async for delta in _pass(messages, tools=tools, max_tokens=MAX_TOKENS):
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


async def narrate(summary: dict[str, Any]) -> str:
    """Пересказ одной сводки. Без инструментов: всё, что можно сказать, уже в JSON."""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": NARRATE_PROMPT},
        {"role": "user", "content": json.dumps(summary, ensure_ascii=False)},
    ]

    round_ = Round()
    async for delta in _pass(messages, tools=None, max_tokens=NARRATE_MAX_TOKENS):
        if isinstance(delta, Round):
            round_ = delta
    return round_.text.strip()


async def _pass(
    messages: list[ChatCompletionMessageParam],
    *,
    tools: list[dict[str, Any]] | None,
    max_tokens: int,
) -> AsyncIterator[Chunk | Round]:
    """Отдаёт куски текста по мере их появления, а последним — итог прохода."""
    round_ = Round()

    async for delta in stream_chat(
        messages,
        tools=tools,
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
    ):
        if delta.content:
            round_.text += delta.content
            yield Chunk(delta.content)
        if delta.finish_reason:
            round_.finish_reason = delta.finish_reason
            round_.tool_calls = delta.tool_calls

    yield round_
