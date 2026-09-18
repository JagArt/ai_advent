import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

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
class AnswerDelta:
    content: str = ""
    finish_reason: str | None = None


@dataclass(frozen=True)
class Answer:
    text: str


async def complete(
    messages: list[ChatCompletionMessageParam],
    *,
    temperature: float,
    max_tokens: int,
) -> Answer:
    """Короткий ответ целиком: стримить нечего, когда ждут пару слов."""
    completion = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return Answer(text=completion.choices[0].message.content or "")


async def stream_chat(
    messages: list[ChatCompletionMessageParam],
    *,
    temperature: float,
    max_tokens: int,
) -> AsyncIterator[AnswerDelta]:
    """Транспорт до модели: что за сообщения пришли и откуда — решает агент.

    Поток закрывается через `async with`, а не сам собой, и с этого дня это важно:
    guard обрывает чтение на середине ответа, и брошенное соединение достаётся сборщику
    мусора — он закрывает его когда придётся и жалуется на это в stderr. Обрыв здесь
    штатный, поэтому и закрытие должно быть штатным.
    """
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}},
    )

    async with stream as chunks:
        async for chunk in chunks:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            text = choice.delta.content
            if text:
                yield AnswerDelta(content=text)
            if choice.finish_reason:
                yield AnswerDelta(finish_reason=choice.finish_reason)
