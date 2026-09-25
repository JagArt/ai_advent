"""Транспорт до модели: стрим для чата и один нестримовый вызов для инструмента.

Отличие от `llm.py` day18 — `complete()` и ленивый клиент. `complete()` нужен
`summarize`: инструмент отдаёт результат целиком, показывать его по буквам некому.
Клиент создаётся при первом обращении, потому что этот модуль импортирует и
MCP-сервер: без ключа должен отказать инструмент, а не упасть весь процесс,
который к тому моменту уже держит соединение по stdio.
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
BASE_URL = "https://api.deepseek.com"

_client: AsyncOpenAI | None = None


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("В окружении нет DEEPSEEK_API_KEY: сводку делать нечем.")
        _client = AsyncOpenAI(api_key=key, base_url=BASE_URL)

    return _client


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


async def complete(
    messages: list[ChatCompletionMessageParam],
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    """Один вызов без стрима: ответ нужен целиком, куски здесь никому не нужны."""
    response = await client().chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return (response.choices[0].message.content or "").strip()


async def stream_chat(
    messages: list[ChatCompletionMessageParam],
    *,
    tools: list[dict[str, Any]] | None,
    temperature: float,
    max_tokens: int,
) -> AsyncIterator[AnswerDelta]:
    """Один проход к модели. Вызовы инструментов отдаются вместе с `finish_reason`."""
    stream = await client().chat.completions.create(
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
