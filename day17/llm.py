"""Транспорт до модели: тот же DeepSeek, что и в остальных днях, плюс вызовы инструментов.

Отличие от `llm.py` прошлых дней одно: кроме текста из дельт копятся `tool_calls`.
В потоке они приезжают кусками — `id` и имя в первой дельте, аргументы строкой
по частям, — и собрать их обратно должен тот, кто читает поток.
"""

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class AnswerDelta:
    content: str = ""
    finish_reason: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)


async def stream_chat(
    messages: list[ChatCompletionMessageParam],
    *,
    tools: list[dict[str, Any]] | None,
    temperature: float,
    max_tokens: int,
) -> AsyncIterator[AnswerDelta]:
    """Один проход к модели. Вызовы инструментов отдаются вместе с `finish_reason`."""
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
        **({"tools": tools, "tool_choice": "auto"} if tools else {}),
    )

    pending: dict[int, dict[str, str]] = {}

    async with stream as chunks:
        async for chunk in chunks:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]

            if choice.delta.content:
                yield AnswerDelta(content=choice.delta.content)

            for call in choice.delta.tool_calls or []:
                slot = pending.setdefault(call.index, {"id": "", "name": "", "arguments": ""})
                if call.id:
                    slot["id"] = call.id
                if call.function and call.function.name:
                    slot["name"] = call.function.name
                if call.function and call.function.arguments:
                    slot["arguments"] += call.function.arguments

            if choice.finish_reason:
                yield AnswerDelta(
                    finish_reason=choice.finish_reason,
                    tool_calls=tuple(
                        ToolCall(**pending[index]) for index in sorted(pending)
                    ),
                )
