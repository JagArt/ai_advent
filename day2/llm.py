import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from constraints import constraints

load_dotenv()

MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
FREE_SYSTEM_PROMPT = "You are a helpful assistant."

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


@dataclass(frozen=True)
class AnswerDelta:
    content: str = ""
    finish_reason: str | None = None


async def stream_answer(prompt: str, *, constrained: bool = False) -> AsyncIterator[AnswerDelta]:
    system_prompt = constraints.build_system_prompt() if constrained else FREE_SYSTEM_PROMPT
    params = dict(constraints.params) if constrained else {}

    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=system_prompt,
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=prompt,
        ),
    ]

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        **params,
    )

    async for chunk in stream:
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        text = choice.delta.content
        if text:
            yield AnswerDelta(content=text)
        if choice.finish_reason:
            yield AnswerDelta(finish_reason=choice.finish_reason)
