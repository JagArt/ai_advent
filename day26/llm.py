"""Транспорт до локальной модели в LM Studio.

LM Studio поднимает на `localhost:1234` сервер с API, совместимым с OpenAI, поэтому
клиент тот же `AsyncOpenAI`, что и в прошлых днях, — меняется только адрес. Ключ
сервер не проверяет, но SDK без него не создаётся, отсюда строка-заглушка.

Qwen3 по умолчанию сначала рассуждает и только потом отвечает. LM Studio отдаёт
рассуждения отдельным полем `reasoning_content`; если в настройках модели это
разделение выключено, они приходят в `content` внутри `<think>...</think>`.
Оба случая сводятся к одному потоку событий: рассуждение отдельно, ответ отдельно.
Выключается рассуждение штатным для Qwen3 маркером `/no_think` в конце промпта.
"""

import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen3-4b")
SYSTEM_PROMPT = "Ты полезный ассистент. Отвечай на языке вопроса."

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

client = AsyncOpenAI(api_key="lm-studio", base_url=BASE_URL)


async def list_models() -> list[str]:
    page = await client.models.list()
    return [model.id for model in page.data]


@dataclass(frozen=True)
class Delta:
    reasoning: str = ""
    content: str = ""


@dataclass(frozen=True)
class Done:
    finish_reason: str | None
    ttft_ms: int | None
    answer_ms: int | None
    total_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    tokens_per_sec: float | None


class _ThinkSplitter:
    """Разводит `<think>`-блок и ответ, если рассуждения пришли внутри `content`."""

    def __init__(self) -> None:
        self.inside = False

    def feed(self, text: str) -> Delta:
        reasoning: list[str] = []
        content: list[str] = []
        while text:
            tag = THINK_CLOSE if self.inside else THINK_OPEN
            head, found, text = text.partition(tag)
            (reasoning if self.inside else content).append(head)
            if found:
                self.inside = not self.inside
        return Delta(reasoning="".join(reasoning), content="".join(content))


def _ms(start: float, moment: float | None) -> int | None:
    return None if moment is None else round((moment - start) * 1000)


async def stream_answer(prompt: str, *, think: bool = True) -> AsyncIterator[Delta | Done]:
    user_text = prompt if think else f"{prompt} /no_think"
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    started = time.perf_counter()
    first_token: float | None = None
    first_answer: float | None = None
    finish_reason: str | None = None
    usage = None
    splitter = _ThinkSplitter()

    stream = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )

    async with stream:
        async for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            raw_reasoning = getattr(choice.delta, "reasoning_content", None) or ""
            split = splitter.feed(choice.delta.content or "")
            delta = Delta(reasoning=raw_reasoning + split.reasoning, content=split.content)

            if delta.reasoning or delta.content:
                now = time.perf_counter()
                first_token = first_token or now
                if delta.content.strip():
                    first_answer = first_answer or now
                yield delta
            if choice.finish_reason:
                finish_reason = choice.finish_reason

    finished = time.perf_counter()
    completion = usage.completion_tokens if usage else None
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    generating = finished - first_token if first_token else None

    yield Done(
        finish_reason=finish_reason,
        ttft_ms=_ms(started, first_token),
        answer_ms=_ms(started, first_answer),
        total_ms=round((finished - started) * 1000),
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=completion,
        reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
        tokens_per_sec=round(completion / generating, 1) if completion and generating else None,
    )
