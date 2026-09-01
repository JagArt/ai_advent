import os
from collections.abc import AsyncIterator

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

load_dotenv()

MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
SYSTEM_PROMPT = "You are a helpful assistant."

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


async def stream_answer(prompt: str) -> AsyncIterator[str]:
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=SYSTEM_PROMPT,
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
    )

    async for chunk in stream:
        if not chunk.choices:
            continue

        text = chunk.choices[0].delta.content

        if text:
            yield text
