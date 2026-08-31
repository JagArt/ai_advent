import os
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

load_dotenv()

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


async def ask_deepseek(prompt: str) -> None:
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content="You are a helpful assistant.",
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=prompt,
        ),
    ]

    stream = await client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        stream=True,
    )

    async for chunk in stream:
        if not chunk.choices:
            continue

        text = chunk.choices[0].delta.content

        if text:
            print(text, end="", flush=True)


async def main():
    prompt = input("Введите запрос: ").strip()
    if not prompt:
        print("Пустой запрос, выход.")
        return

    await ask_deepseek(prompt)


asyncio.run(main())