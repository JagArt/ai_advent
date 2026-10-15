"""Две модели за одним циклом с инструментами и маршрутизатор между ними.

Провайдеры те же, что в [day28](../day28/llm.py): `qwen3-4b` в LM Studio и
DeepSeek в облаке, у обоих OpenAI-совместимый API, клиент один. Новое — зачем
их две одновременно. Конкретный вопрос — «на каком порту», «где лежит», «как
запустить» — закрывается выдержками из документации, и 4B-модель на этой машине
отвечает на него не хуже облака (day28), только бесплатно и офлайн. Вопрос про
проектирование, сравнение или план требует держать в голове несколько дней
проекта сразу и звать инструменты по ходу — здесь слабая модель теряется, и
такие вопросы уходят в DeepSeek.

Кто решает, какой вопрос какой, — сама локальная модель, одним коротким вызовом
без рассуждений: ответ одно слово, «факт» или «план». Это дёшево — десятые доли
секунды — и видно человеку: маршрут печатается строкой перед ответом, а флаги
`--local` и `--cloud` его перекрывают.

Фолбэки простые и честные: без LM Studio всё идёт в облако, без ключа — всё
локально, и в обоих случаях причина печатается вместо маршрута.
"""

import os
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

PROVIDERS = ("local", "cloud")
TITLES = {"local": "локальная · LM Studio", "cloud": "облачная · DeepSeek"}

LOCAL_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LOCAL_MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen3-4b")
CLOUD_BASE_URL = "https://api.deepseek.com"
CLOUD_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
MODELS = {"local": LOCAL_MODEL, "cloud": CLOUD_MODEL}

PROBE_TIMEOUT = 3
NO_THINK = "/no_think"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Классификатор: одно слово. «факт» — локальная модель, «план» — облако.
CLASSIFY_SYSTEM = (
    "Ты сортируешь вопросы разработчика о его проекте по сложности. Ответь ровно одним словом.\n"
    "факт — вопрос про конкретное и проверяемое: где лежит файл, какой порт или команда, "
    "что делает модуль, как запустить, что изменилось, какая ветка, сколько чего.\n"
    "план — вопрос требует спроектировать, спланировать, сравнить варианты, объяснить "
    "«почему так устроено» или «как лучше сделать», разбить работу на шаги, оценить риски.\n"
    "Никаких пояснений: только слово факт или слово план."
)
CLASSIFY_MAX_TOKENS = 24
CLASS_WORD = re.compile(r"\b(факт|план)\b", re.IGNORECASE)

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


# --- доступность и маршрут ---------------------------------------------------


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

    @property
    def any(self) -> bool:
        return self.local_online or self.cloud_configured

    def as_dict(self) -> dict[str, Any]:
        return {
            "local": {
                "online": self.local_online,
                "base_url": LOCAL_BASE_URL,
                "model": LOCAL_MODEL,
                "model_loaded": self.local_model_loaded,
                "error": self.local_error,
            },
            "cloud": {"configured": self.cloud_configured, "model": CLOUD_MODEL},
        }


async def probe() -> Availability:
    """LM Studio опрашивается с коротким таймаутом: ассистент не должен ждать молчащий сервер."""
    try:
        page = await _local.with_options(timeout=PROBE_TIMEOUT).models.list()
    except Exception as exc:
        return Availability(False, (), f"LM Studio не отвечает ({type(exc).__name__}). Запустите `lms server start`.", cloud_configured())
    return Availability(True, tuple(model.id for model in page.data), None, cloud_configured())


@dataclass(frozen=True)
class Route:
    """Куда уходит вопрос и почему — строка маршрута печатается перед ответом."""

    provider: str
    reason: str
    forced: bool = False
    verdict: str | None = None
    ms: int | None = None

    @property
    def model(self) -> str:
        return MODELS[self.provider]

    @property
    def title(self) -> str:
        return TITLES[self.provider]


def _strip_think(text: str) -> str:
    return re.sub(rf"{THINK_OPEN}.*?{THINK_CLOSE}", "", text, flags=re.DOTALL).strip()


async def classify(question: str) -> tuple[str | None, str, int]:
    """Локальная модель отвечает одним словом. Возвращает (слово или None, сырой ответ, мс)."""
    started = time.perf_counter()
    response = await _local.chat.completions.create(
        model=LOCAL_MODEL,
        messages=[
            {"role": "system", "content": f"{CLASSIFY_SYSTEM} {NO_THINK}"},
            {"role": "user", "content": f"Вопрос: {question} {NO_THINK}"},
        ],
        temperature=0.0,
        max_tokens=CLASSIFY_MAX_TOKENS,
    )
    raw = _strip_think(response.choices[0].message.content or "")
    match = CLASS_WORD.search(raw)
    ms = round((time.perf_counter() - started) * 1000)
    return (match.group(1).lower() if match else None), raw, ms


async def route(question: str, availability: Availability, forced: str | None = None) -> Route:
    """Маршрут вопроса: флаг, фолбэк по доступности или решение классификатора."""
    if not availability.any:
        raise RuntimeError(
            "Нет ни одной модели: LM Studio не отвечает и DEEPSEEK_API_KEY не задан. "
            "Запустите `lms server start` или добавьте ключ в .env."
        )

    if forced:
        if forced == "local" and not availability.local_online:
            raise RuntimeError(f"Запрошена локальная модель, но {availability.local_error}")
        if forced == "cloud" and not availability.cloud_configured:
            raise RuntimeError("Запрошена облачная модель, но в окружении нет DEEPSEEK_API_KEY.")
        return Route(forced, f"флаг --{forced}", forced=True)

    if not availability.local_online:
        return Route("cloud", "LM Studio не отвечает, всё уходит в облако")
    if not availability.cloud_configured:
        return Route("local", "ключа DeepSeek нет, всё остаётся на машине")

    try:
        verdict, raw, ms = await classify(question)
    except Exception as exc:
        return Route("cloud", f"классификатор не ответил ({type(exc).__name__}), вопрос ушёл в облако")

    if verdict == "план":
        return Route("cloud", "вопрос про план или устройство", verdict=verdict, ms=ms)
    if verdict == "факт":
        return Route("local", "вопрос про конкретный факт", verdict=verdict, ms=ms)
    return Route("local", f"классификатор ответил неясно ({raw[:30]!r}), по умолчанию локально", verdict=raw, ms=ms)


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
