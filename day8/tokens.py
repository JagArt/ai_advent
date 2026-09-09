from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import tiktoken
from openai.types.chat import ChatCompletionMessageParam

# Своего токенизатора DeepSeek в виде библиотеки не отдаёт, поэтому локальный счёт —
# это оценка на cl100k_base: сопоставимый по размеру BPE-словарь, но не тот же самый.
# Точные числа приходят в usage вместе с ответом, и расхождение мы показываем.
ENCODING = "cl100k_base"

_encoder = tiktoken.get_encoding(ENCODING)

# Сообщения уходят в модель не голым текстом: каждое обрамляется служебными
# токенами роли, а ответ начинается с затравки. Без этих поправок оценка
# занижена ровно на длину разметки — на коротких репликах это заметно.
MESSAGE_OVERHEAD = 4
REPLY_OVERHEAD = 3

# Жёсткий потолок deepseek-v4-flash: вход и выход вместе. Превысить его нельзя
# ничем — ни бюджетом контекста, ни лимитом ответа.
MODEL_CONTEXT_LIMIT = 1_048_576


@dataclass(frozen=True)
class Pricing:
    """Цены за 1M токенов в непиковые часы; в пиковые — вдвое дороже."""

    input_miss: float
    input_hit: float
    output: float


# deepseek-v4-flash: https://api-docs.deepseek.com/quick_start/pricing
PRICING = Pricing(input_miss=0.22, input_hit=0.007, output=0.66)

# Пиковые окна DeepSeek по UTC, пн–пт; в остальные часы цены вдвое ниже.
PEAK_WINDOWS_UTC = ((1, 4), (6, 10))


@dataclass(frozen=True)
class Usage:
    """Факт из ответа API — источник истины и для токенов, и для денег."""

    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def count_text(text: str) -> int:
    return len(_encoder.encode(text, disallowed_special=()))


def count_messages(messages: list[ChatCompletionMessageParam]) -> int:
    """Оценка запроса целиком: содержимое сообщений плюс разметка чата."""
    total = REPLY_OVERHEAD
    for message in messages:
        content = message.get("content") or ""
        total += MESSAGE_OVERHEAD + count_text(content if isinstance(content, str) else str(content))
    return total


def is_peak_hour(moment: datetime | None = None) -> bool:
    now = moment or datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    return any(start <= now.hour < end for start, end in PEAK_WINDOWS_UTC)


def price_multiplier(moment: datetime | None = None) -> float:
    return 2.0 if is_peak_hour(moment) else 1.0


def cost_usd(
    cached_tokens: int,
    miss_tokens: int,
    completion_tokens: int,
    moment: datetime | None = None,
) -> float:
    per_token = price_multiplier(moment) / 1_000_000
    return (
        cached_tokens * PRICING.input_hit
        + miss_tokens * PRICING.input_miss
        + completion_tokens * PRICING.output
    ) * per_token


def usage_from(raw: Any) -> Usage:
    """Разбор usage из ответа: кэш входа считается отдельно, он в 30 раз дешевле."""
    if raw is None:
        return Usage()

    prompt_tokens = getattr(raw, "prompt_tokens", 0) or 0
    hit = getattr(raw, "prompt_cache_hit_tokens", None)
    miss = getattr(raw, "prompt_cache_miss_tokens", None)
    if hit is None or miss is None:
        # Полей DeepSeek нет — берём общий для OpenAI-совместимых API разрез.
        details = getattr(raw, "prompt_tokens_details", None)
        hit = getattr(details, "cached_tokens", None) or 0
        miss = max(prompt_tokens - hit, 0)

    completion_tokens = getattr(raw, "completion_tokens", 0) or 0
    return Usage(
        prompt_tokens=prompt_tokens,
        cached_tokens=hit,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd(hit, miss, completion_tokens),
    )


def drift_percent(estimated: int, actual: int) -> float | None:
    """На сколько процентов локальная оценка разошлась с фактом из usage."""
    if not actual:
        return None
    return round((estimated - actual) / actual * 100, 1)
