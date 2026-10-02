"""Транспорт до модели: стрим для страницы, один вызов для прогонов и судьи.

От [day21](../day21/llm.py), где модель только придумывала пробы, здесь добавлены
две вещи, и обе нужны сравнению режимов.

Первая — токены. Весь смысл RAG в том, что к вопросу пристёгивается контекст, и
счёт за это приходит в токенах запроса. Сказать «с RAG дороже» без чисел нельзя,
поэтому `usage` возвращается наружу вместе с текстом, а не выбрасывается.

Вторая — стрим. На странице два ответа считаются одновременно, и ответ без
контекста приходит раньше: ему нечего читать. Эта разница во времени — часть
того, что день показывает, и видна она только в потоке.

Клиент создаётся при первом обращении: индексация и поиск работают без ключа
вовсе, и падать на импорте модулю, который к эмбеддингам отношения не имеет,
незачем.

В day24 к обоим входам добавился `json`. Просить структуру одним промптом
недостаточно: модель охотно оборачивает ответ в ```` ```json ```` или
предваряет его фразой «Вот ответ:», и разбор превращается в угадайку.
`response_format={"type": "json_object"}` снимает это на стороне провайдера.
Стрим при этом остаётся общим для всех режимов: цена контекста видна только в
том, когда пошли первые символы, и терять эту разницу ради удобства разбора
было бы обменом не в ту сторону.
"""

import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

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
            raise RuntimeError("В окружении нет DEEPSEEK_API_KEY: спрашивать нечем.")
        _client = AsyncOpenAI(api_key=key, base_url=BASE_URL)

    return _client


@dataclass(frozen=True)
class Reply:
    """Ответ модели вместе с его ценой: без токенов режимы несравнимы."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "seconds": round(self.seconds, 2),
        }


@dataclass(frozen=True)
class Piece:
    """Кусок потока: либо текст по мере готовности, либо итог последним кадром."""

    content: str = ""
    reply: Reply | None = None


def _format(json: bool) -> dict[str, object]:
    return {"response_format": {"type": "json_object"}} if json else {}


async def complete(
    messages: list[ChatCompletionMessageParam],
    *,
    temperature: float,
    max_tokens: int,
    json: bool = False,
) -> Reply:
    """Один вызов без стрима: для прогонов и для судьи ответ нужен целиком."""
    started = time.perf_counter()
    response = await client().chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
        **_format(json),
    )

    usage = response.usage
    return Reply(
        text=(response.choices[0].message.content or "").strip(),
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        seconds=time.perf_counter() - started,
    )


async def stream(
    messages: list[ChatCompletionMessageParam],
    *,
    temperature: float,
    max_tokens: int,
    json: bool = False,
) -> AsyncIterator[Piece]:
    """Текст по кускам, последним кадром — собранный `Reply` с токенами.

    `include_usage` обязателен: без него стримовый ответ приходит вообще без
    счётчиков, и считать токены пришлось бы самим — то есть неверно.
    """
    started = time.perf_counter()
    response = await client().chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
        stream_options={"include_usage": True},
        extra_body={"thinking": {"type": "disabled"}},
        **_format(json),
    )

    parts: list[str] = []
    prompt_tokens = completion_tokens = 0

    async with response as chunks:
        async for chunk in chunks:
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                parts.append(content)
                yield Piece(content=content)

    yield Piece(
        reply=Reply(
            text="".join(parts).strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            seconds=time.perf_counter() - started,
        )
    )
