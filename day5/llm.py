import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

load_dotenv()

SYSTEM_PROMPT = "You are a helpful assistant."

# Одинаковые для всех трёх моделей: сравниваем модели, а не настройки запроса.
TEMPERATURE = 0.3
# Общий потолок с запасом: на самых длинных ответах ни одна из трёх в него не упирается.
MAX_TOKENS = 4000

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Пиковые окна DeepSeek по UTC, пн–пт; в остальные часы цены вдвое ниже.
PEAK_WINDOWS_UTC = ((1, 4), (6, 10))


@dataclass(frozen=True)
class Pricing:
    """Цены за 1M токенов в непиковые часы; в пиковые — вдвое дороже."""

    input_miss: float
    input_hit: float
    output: float


@dataclass(frozen=True)
class ModelSpec:
    key: str
    tier: str
    model: str
    provider: str
    base_url: str
    api_key_env: str
    link: str
    note: str
    pricing: Pricing | None
    extra_body: dict[str, Any]


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="weak",
        tier="Слабая",
        model="cohere/north-mini-code:free",
        provider="OpenRouter",
        base_url=OPENROUTER_BASE_URL,
        api_key_env="OPENROUTER_API_KEY",
        link="https://openrouter.ai/cohere/north-mini-code:free",
        note="MoE 30B, активно 3B на токен, бесплатная",
        # Стоимость приходит готовой в usage.cost, считать по прайсу не нужно.
        pricing=None,
        extra_body={"usage": {"include": True}},
    ),
    ModelSpec(
        key="medium",
        tier="Средняя",
        model="deepseek-v4-flash",
        provider="DeepSeek",
        base_url=DEEPSEEK_BASE_URL,
        api_key_env="DEEPSEEK_API_KEY",
        link="https://api-docs.deepseek.com/quick_start/pricing",
        note="быстрая модель DeepSeek, мышление выключено",
        pricing=Pricing(input_miss=0.22, input_hit=0.007, output=0.66),
        extra_body={"thinking": {"type": "disabled"}},
    ),
    ModelSpec(
        key="strong",
        tier="Сильная",
        model="deepseek-v4-pro",
        provider="DeepSeek",
        base_url=DEEPSEEK_BASE_URL,
        api_key_env="DEEPSEEK_API_KEY",
        link="https://api-docs.deepseek.com/quick_start/pricing",
        note="старшая модель DeepSeek, мышление выключено для равных условий",
        pricing=Pricing(input_miss=0.66, input_hit=0.022, output=1.98),
        extra_body={"thinking": {"type": "disabled"}},
    ),
)

DEFAULT_PROMPT = """\
У тебя есть 12 монет, одна фальшивая и отличается по весу, но неизвестно — тяжелее она или легче.
Имея чашечные весы и ровно 3 взвешивания, определи, какая монета фальшивая и тяжелее она или легче.
"""

_clients: dict[str, AsyncOpenAI] = {}


def client_for(spec: ModelSpec) -> AsyncOpenAI:
    if spec.base_url not in _clients:
        _clients[spec.base_url] = AsyncOpenAI(
            api_key=os.environ[spec.api_key_env],
            base_url=spec.base_url,
        )
    return _clients[spec.base_url]


def is_peak_hour(moment: datetime | None = None) -> bool:
    now = moment or datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    return any(start <= now.hour < end for start, end in PEAK_WINDOWS_UTC)


def price_multiplier(moment: datetime | None = None) -> float:
    return 2.0 if is_peak_hour(moment) else 1.0


@dataclass(frozen=True)
class Chunk:
    """Кусок потока: рассуждение модели или сам ответ."""

    kind: str
    text: str


@dataclass(frozen=True)
class Metrics:
    ttft_ms: int | None = None
    answer_ttft_ms: int | None = None
    total_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    output_tps: float | None = None
    cost_usd: float | None = None
    peak: bool = False
    finish_reason: str | None = None


def _messages(prompt: str) -> list[ChatCompletionMessageParam]:
    return [
        ChatCompletionSystemMessageParam(role="system", content=SYSTEM_PROMPT),
        ChatCompletionUserMessageParam(role="user", content=prompt),
    ]


def _cost_usd(spec: ModelSpec, usage: Any, multiplier: float) -> float | None:
    if usage is None:
        return None

    # OpenRouter считает стоимость сам и кладёт её в usage.
    if spec.pricing is None:
        cost = getattr(usage, "cost", None)
        return float(cost) if cost is not None else None

    prompt_tokens = usage.prompt_tokens or 0
    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    miss = getattr(usage, "prompt_cache_miss_tokens", None)
    if hit is None or miss is None:
        hit = _cached_tokens(usage)
        miss = max(prompt_tokens - hit, 0)

    per_token = multiplier / 1_000_000
    return (
        hit * spec.pricing.input_hit
        + miss * spec.pricing.input_miss
        + (usage.completion_tokens or 0) * spec.pricing.output
    ) * per_token


def _cached_tokens(usage: Any) -> int:
    details = getattr(usage, "prompt_tokens_details", None)
    return getattr(details, "cached_tokens", None) or 0


def _reasoning_tokens(usage: Any) -> int:
    details = getattr(usage, "completion_tokens_details", None)
    return getattr(details, "reasoning_tokens", None) or 0


async def stream_model(spec: ModelSpec, prompt: str) -> AsyncIterator[Chunk | Metrics]:
    """Стримит ответ одной модели и завершается метриками замера."""
    multiplier = price_multiplier()
    started = time.perf_counter()
    ttft: float | None = None
    answer_ttft: float | None = None
    finish_reason: str | None = None
    usage: Any = None

    stream = await client_for(spec).chat.completions.create(
        model=spec.model,
        messages=_messages(prompt),
        stream=True,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream_options={"include_usage": True},
        extra_body=spec.extra_body,
    )

    async for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        # Рассуждение приходит отдельным полем, которого нет в типах SDK.
        reasoning = getattr(choice.delta, "reasoning", None)
        answer = choice.delta.content

        for kind, text in (("reasoning", reasoning), ("answer", answer)):
            if not text:
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - started
            if kind == "answer" and answer_ttft is None:
                answer_ttft = now - started
            yield Chunk(kind=kind, text=text)

        if choice.finish_reason:
            finish_reason = choice.finish_reason

    total = time.perf_counter() - started
    completion_tokens = (usage.completion_tokens or 0) if usage else 0
    generation = total - (ttft or 0)

    yield Metrics(
        ttft_ms=_ms(ttft),
        answer_ttft_ms=_ms(answer_ttft),
        total_ms=_ms(total) or 0,
        prompt_tokens=(usage.prompt_tokens or 0) if usage else 0,
        completion_tokens=completion_tokens,
        reasoning_tokens=_reasoning_tokens(usage),
        cached_tokens=_cached_tokens(usage),
        output_tps=round(completion_tokens / generation, 1) if generation > 0 and completion_tokens else None,
        cost_usd=_cost_usd(spec, usage, multiplier),
        peak=multiplier > 1,
        finish_reason=finish_reason,
    )


def _ms(seconds: float | None) -> int | None:
    return None if seconds is None else round(seconds * 1000)
