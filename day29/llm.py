"""Одна модель — `qwen3-4b` в LM Studio — и два способа её спросить, заданные профилем.

В day28 здесь жили два провайдера, локальный и облачный, за одним промптом. Облако
ушло: день про то, как настроить локальную модель под задачу, и сравнивать её
нужно с ней же в базовой настройке, а не с DeepSeek. Остался один клиент
`AsyncOpenAI` на `localhost:1234`, а всё, что раньше было константами —
температура, потолок ответа, промпт, — переехало в [profiles.py](profiles.py).
Этот модуль профиль только исполняет: собирает сообщения, передаёт параметры,
стримит ответ и считает замеры.

Замеры те же, что в day26–day28: время до первого токена, до первого токена
ответа (если были рассуждения), всего, токены промпта и ответа, ток/с по
`usage` из последнего кадра стрима. Они и есть ось «скорость» в сравнении
профилей: оптимизация режет промпт, и это должно быть видно в `prompt_tokens`
и `ttft_ms`, а не только на слух.
"""

import os
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass

from dotenv import load_dotenv
from openai import AsyncOpenAI

from profiles import REFUSAL, Profile

load_dotenv()

BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen3-4b")

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

_client = AsyncOpenAI(api_key="lm-studio", base_url=BASE_URL)


async def local_models() -> list[str]:
    page = await _client.models.list()
    return [model.id for model in page.data]


@dataclass(frozen=True)
class Delta:
    reasoning: str = ""
    content: str = ""


@dataclass(frozen=True)
class Done:
    """Итог стрима: текст целиком и цена — время до первого токена, всего, токены."""

    profile: str
    model: str
    text: str
    finish_reason: str | None
    ttft_ms: int | None
    answer_ms: int | None
    total_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    tokens_per_sec: float | None

    @property
    def refused(self) -> bool:
        """Отказ — по первой строке: «[1] В выдержках этого нет.» тоже отказ."""
        first = self.text.strip().split("\n", 1)[0].lower()
        return REFUSAL.rstrip(".").lower() in first

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "refused": self.refused}


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


async def stream_answer(
    question: str,
    block: str,
    *,
    profile: Profile,
    think: bool = False,
) -> AsyncIterator[Delta | Done]:
    """Ответ модели по профилю, по кускам; последним кадром — `Done` с замерами."""
    messages = profile.messages(question, block, think=think)

    started = time.perf_counter()
    first_token: float | None = None
    first_answer: float | None = None
    finish_reason: str | None = None
    usage = None
    parts: list[str] = []
    splitter = _ThinkSplitter()

    stream = await _client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        **profile.sampling(),
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
                parts.append(delta.content)
                yield delta
            if choice.finish_reason:
                finish_reason = choice.finish_reason

    finished = time.perf_counter()
    completion = usage.completion_tokens if usage else None
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    generating = finished - first_token if first_token else None

    yield Done(
        profile=profile.name,
        model=MODEL,
        text="".join(parts).strip(),
        finish_reason=finish_reason,
        ttft_ms=_ms(started, first_token),
        answer_ms=_ms(started, first_answer),
        total_ms=round((finished - started) * 1000),
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=completion,
        reasoning_tokens=getattr(details, "reasoning_tokens", None) if details else None,
        tokens_per_sec=round(completion / generating, 1) if completion and generating else None,
    )


async def answer(question: str, block: str, *, profile: Profile, think: bool = False) -> Done:
    """Ответ целиком: для прогонов стрим не нужен, а замеры — те же."""
    async for item in stream_answer(question, block, profile=profile, think=think):
        if isinstance(item, Done):
            return item
    raise RuntimeError("Стрим закончился без итога.")
