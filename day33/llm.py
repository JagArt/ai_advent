"""Две модели за одним стримом с инструментами: qwen3-4b в LM Studio и DeepSeek в облаке.

Провайдеры и стрим — из [day31/llm.py](../day31/llm.py), без классификатора:
здесь модель выбирает оператор переключателем на странице, а не локальная
модель одним словом. У обоих провайдеров OpenAI-совместимый API, клиент один;
различаются адрес, ключ и то, как выключается рассуждение: у Qwen3 — маркер
`/no_think` в системном сообщении, у DeepSeek — `thinking: disabled` в теле.

Клиент облака создаётся при первом обращении: без `DEEPSEEK_API_KEY` локальный
режим обязан работать как ни в чём не бывало.
"""

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

PROVIDERS = ("local", "cloud")
TITLES = {"local": "локальная · LM Studio", "cloud": "облачная · DeepSeek"}

# Локальная модель — на соседней машине в сети, как в day32; localhost задаётся переменной.
LOCAL_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://192.168.77.47:1234/v1")
LOCAL_MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen3-4b")
CLOUD_BASE_URL = "https://api.deepseek.com"
CLOUD_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
MODELS = {"local": LOCAL_MODEL, "cloud": CLOUD_MODEL}

PROBE_TIMEOUT = 3
NO_THINK = "/no_think"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

_local = AsyncOpenAI(api_key="lm-studio", base_url=LOCAL_BASE_URL)
_cloud: AsyncOpenAI | None = None


def cloud_configured() -> bool:
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


def _cloud_client() -> AsyncOpenAI:
    global _cloud
    if _cloud is None:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("В окружении нет DEEPSEEK_API_KEY: облачная модель недоступна.")
        _cloud = AsyncOpenAI(api_key=key, base_url=CLOUD_BASE_URL)
    return _cloud


def _client(provider: str) -> AsyncOpenAI:
    if provider not in PROVIDERS:
        raise ValueError(f"Модели {provider!r} нет. Есть: {', '.join(PROVIDERS)}.")
    return _local if provider == "local" else _cloud_client()


# --- доступность -------------------------------------------------------------


@dataclass(frozen=True)
class Availability:
    """Что сейчас на связи: LM Studio с нужной моделью и ключ облака."""

    local_online: bool
    local_models: tuple[str, ...]
    local_error: str | None
    cloud_configured: bool

    @property
    def local_model_loaded(self) -> bool:
        return LOCAL_MODEL in self.local_models

    def as_dict(self) -> dict[str, Any]:
        return {
            "local": {
                "online": self.local_online,
                "base_url": LOCAL_BASE_URL,
                "model": LOCAL_MODEL,
                "models": list(self.local_models),
                "model_loaded": self.local_model_loaded,
                "error": self.local_error,
            },
            "cloud": {"configured": self.cloud_configured, "model": CLOUD_MODEL},
        }


async def probe() -> Availability:
    """LM Studio опрашивается с коротким таймаутом: страница не должна ждать молчащий сервер."""
    try:
        page = await _local.with_options(timeout=PROBE_TIMEOUT).models.list()
    except Exception as exc:
        return Availability(False, (), f"LM Studio не отвечает ({type(exc).__name__}). Запустите `lms server start`.", cloud_configured())
    return Availability(True, tuple(model.id for model in page.data), None, cloud_configured())


def check(provider: str, availability: Availability) -> None:
    """Понятный отказ до вызова модели, а не APIConnectionError из глубины стрима."""
    if provider == "local" and not availability.local_online:
        raise RuntimeError(availability.local_error or "LM Studio не отвечает.")
    if provider == "local" and not availability.local_model_loaded:
        raise RuntimeError(f"В LM Studio не загружена модель {LOCAL_MODEL}. Выполните `lms load {LOCAL_MODEL}`.")
    if provider == "cloud" and not availability.cloud_configured:
        raise RuntimeError("В окружении нет DEEPSEEK_API_KEY: облачная модель недоступна.")


# --- стрим с инструментами ---------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Delta:
    """Кусок ответа. Вызовы инструментов приходят вместе с `finish_reason`."""

    reasoning: str = ""
    content: str = ""
    finish_reason: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class _ThinkSplitter:
    """Разводит `<think>`-блок и ответ, если рассуждения пришли внутри `content`."""

    def __init__(self) -> None:
        self.inside = False

    def feed(self, text: str) -> tuple[str, str]:
        reasoning: list[str] = []
        content: list[str] = []
        while text:
            tag = THINK_CLOSE if self.inside else THINK_OPEN
            head, found, text = text.partition(tag)
            (reasoning if self.inside else content).append(head)
            if found:
                self.inside = not self.inside
        return "".join(reasoning), "".join(content)


def _for_provider(messages: list[ChatCompletionMessageParam], provider: str) -> list[ChatCompletionMessageParam]:
    """Локальной модели — `/no_think` в системном сообщении: оно действует на все раунды цикла,
    в том числе после ответов инструментов, где дописывать маркер некуда."""
    if provider != "local":
        return messages
    copied = [dict(message) for message in messages]
    for message in copied:
        if message.get("role") == "system" and NO_THINK not in str(message.get("content", "")):
            message["content"] = f"{message['content']}\n{NO_THINK}"
            break
    return copied  # type: ignore[return-value]


async def stream_chat(
    messages: list[ChatCompletionMessageParam],
    *,
    provider: str,
    tools: list[dict[str, Any]] | None,
    temperature: float,
    max_tokens: int,
) -> AsyncIterator[Delta]:
    """Один проход к модели: текст по кускам, последним — итог с `finish_reason` и вызовами."""
    client = _client(provider)
    extra: dict[str, Any] = {} if provider == "local" else {"extra_body": {"thinking": {"type": "disabled"}}}

    stream = await client.chat.completions.create(
        model=MODELS[provider],
        messages=_for_provider(messages, provider),
        stream=True,
        stream_options={"include_usage": True},
        temperature=temperature,
        max_tokens=max_tokens,
        **({"tools": tools, "tool_choice": "auto"} if tools else {}),
        **extra,
    )

    pending: dict[int, dict[str, str]] = {}
    splitter = _ThinkSplitter()
    finish: str | None = None
    usage = None

    async with stream:
        async for chunk in stream:
            if chunk.usage:
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]

            reasoning, content = splitter.feed(choice.delta.content or "")
            reasoning = (getattr(choice.delta, "reasoning_content", None) or "") + reasoning
            if reasoning or content:
                yield Delta(reasoning=reasoning, content=content)

            for call in choice.delta.tool_calls or []:
                slot = pending.setdefault(call.index or 0, {"id": "", "name": "", "arguments": ""})
                if call.id:
                    slot["id"] = call.id
                if call.function and call.function.name:
                    slot["name"] = call.function.name
                if call.function and call.function.arguments:
                    slot["arguments"] += call.function.arguments

            if choice.finish_reason:
                finish = choice.finish_reason

    calls = tuple(
        ToolCall(slot["id"] or f"call_{index}", slot["name"], slot["arguments"] or "{}")
        for index, slot in sorted(pending.items())
        if slot["name"]
    )
    yield Delta(
        finish_reason=finish or ("tool_calls" if calls else "stop"),
        tool_calls=calls,
        prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
    )
