"""Транспорт до модели: один нестримовый вызов, больше здесь ничего не нужно.

В этом дне модель делает ровно одну работу — придумывает вопросы для сравнения
стратегий chunking. Ответ нужен целиком и сразу, показывать его по буквам некому,
поэтому от `llm.py` предыдущих дней остался только `complete()`.

Клиент создаётся при первом обращении: индексация, поиск и вся страница работают
без ключа вовсе, и падать на импорте из-за отсутствующего `DEEPSEEK_API_KEY`
модулю, который к эмбеддингам отношения не имеет, незачем.
"""

import os

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
            raise RuntimeError("В окружении нет DEEPSEEK_API_KEY: вопросы придумывать нечем.")
        _client = AsyncOpenAI(api_key=key, base_url=BASE_URL)

    return _client


async def complete(
    messages: list[ChatCompletionMessageParam],
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    response = await client().chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return (response.choices[0].message.content or "").strip()
