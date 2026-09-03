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

load_dotenv()

MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
FREE_SYSTEM_PROMPT = "You are a helpful assistant."

TEMP_MIN = 0.0
TEMP_MAX = 2.0
TEMP_STEP = 0.1
DEFAULT_TEMPERATURE = 0.7

# (верхняя граница диапазона, описание) — подпись под слайдером на странице.
TEMPERATURE_BANDS = (
    (0.2, "Максимально детерминированно: факты, код, классификация."),
    (0.6, "Баланс точности и разнообразия; хороший default для большинства задач."),
    (1.0, "Более творческие и разнообразные ответы."),
    (1.4, "Высокая вариативность, больше неожиданных формулировок."),
    (TEMP_MAX, "Очень высокая случайность; выше риск бессвязности и ошибок."),
)

DEFAULT_PROMPT = """\
Объясни в 3–4 предложениях, что такое кэш в программировании: зачем он нужен и в чём его главный риск.
Затем придумай одну неожиданную аналогию из повседневной жизни и одно название для сервиса кэширования.
"""

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


@dataclass(frozen=True)
class AnswerDelta:
    content: str = ""
    finish_reason: str | None = None


def _messages(system: str, user: str) -> list[ChatCompletionMessageParam]:
    return [
        ChatCompletionSystemMessageParam(role="system", content=system),
        ChatCompletionUserMessageParam(role="user", content=user),
    ]


async def _stream_chat(
    messages: list[ChatCompletionMessageParam],
    *,
    temperature: float,
) -> AsyncIterator[AnswerDelta]:
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        temperature=temperature,
        extra_body={"thinking": {"type": "disabled"}},
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


async def stream_answer(prompt: str, *, temperature: float) -> AsyncIterator[AnswerDelta]:
    async for delta in _stream_chat(_messages(FREE_SYSTEM_PROMPT, prompt), temperature=temperature):
        yield delta
