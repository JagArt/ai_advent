"""Две модели за одним промптом: qwen3-4b в LM Studio и DeepSeek в облаке.

У обоих провайдеров API, совместимый с OpenAI, поэтому клиент один — `AsyncOpenAI`,
— а различаются адрес, ключ и то, как выключается рассуждение. У Qwen3 это маркер
`/no_think` в конце сообщения, у DeepSeek — `thinking: disabled` в теле запроса.
Рассуждение выключено по умолчанию у обоих: в [day27](../day27/README.md) оно
стоило 4B-модели от 5 до 60 секунд на запрос и ответ не улучшало. Для локальной
модели его можно включить галочкой — чтобы увидеть цену ещё раз, уже на RAG.

Промпт один на обе модели, и это главное условие дня. Если бы локальная модель
получала «упрощённый» промпт, разница в ответах говорила бы о промптах, а не о
моделях. Промпт при этом короче, чем в day24: там модель обязана была отдать json
с утверждениями и цитатами, и у 4B-модели такая форма рвётся чаще, чем
выдерживается, а оборванный json сравнивать не с чем. Здесь — свободный текст с
номерами выдержек в скобках и строкой отказа.

Порядок частей в промпте — не вкус, а результат пробы. В компоновке day22 —
правила, выдержки, вопрос последней строкой — `qwen3-4b` на вопросе про порт
отвечал «[1] В выдержках этого нет.» пять раз подряд, по одному на выдержку, хотя
ответ стоял в третьей; DeepSeek на том же промпте отвечал верно. Стоило
поставить вопрос первым, а правила — после выдержек и с отказом последним
пунктом, как 4B стала отвечать по делу и повторять вопрос в конце уже не путалась.
Облачной модели порядок безразличен, поэтому общий промпт собран так, как нужно
слабой.

Клиент облака создаётся при первом обращении: без `DEEPSEEK_API_KEY` локальный
режим обязан работать как ни в чём не бывало — в этом и состоит «RAG полностью
локально».
"""

import os
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

PROVIDERS = ("local", "cloud")
PROVIDER_TITLES = {"local": "локальная · LM Studio", "cloud": "облачная · DeepSeek"}

LOCAL_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LOCAL_MODEL = os.environ.get("LMSTUDIO_MODEL", "qwen3-4b")

CLOUD_BASE_URL = "https://api.deepseek.com"
CLOUD_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

MODELS = {"local": LOCAL_MODEL, "cloud": CLOUD_MODEL}

TEMPERATURE = 0.2
MAX_TOKENS = 600

REFUSAL = "В выдержках этого нет."

SYSTEM = (
    "Ты отвечаешь на вопросы о проекте AI Advent — репозитории с дневником итераций, "
    "где есть документация, код на Python и PDF.\n"
    "Отвечай по-русски, коротко и по делу: два-четыре предложения, без вступлений "
    "и без пересказа вопроса.\n"
    "Если чего-то не знаешь — скажи об этом прямо, не придумывай."
)

RULES = (
    "Правила ответа:\n"
    "1. Отвечай только по выдержкам, не добавляй ничего от себя.\n"
    "2. После каждого утверждения ставь номер выдержки в квадратных скобках, например [2].\n"
    "3. Числа, имена файлов и названия приводи ровно так, как в выдержках.\n"
    f"4. Только если ответа в выдержках нет — напиши одной строкой «{REFUSAL}»."
)

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


async def local_models() -> list[str]:
    page = await _local.models.list()
    return [model.id for model in page.data]


def messages_for(question: str, block: str, *, think: bool, provider: str) -> list[ChatCompletionMessageParam]:
    """Один и тот же промпт для обеих моделей; `/no_think` — единственное отличие."""
    user = (
        f"Вопрос: {question}\n\n"
        f"Выдержки из базы проекта, каждая под своим номером:\n\n{block}\n\n"
        f"{RULES}\n\n"
        f"Ответь на вопрос: {question}"
    )
    if provider == "local" and not think:
        user = f"{user} /no_think"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


@dataclass(frozen=True)
class Delta:
    reasoning: str = ""
    content: str = ""


@dataclass(frozen=True)
class Done:
    """Итог стрима: текст целиком и цена — время до первого токена, всего, токены."""

    provider: str
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
        return {**asdict(self), "refused": self.refused, "title": PROVIDER_TITLES[self.provider]}


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
    provider: str,
    think: bool = False,
) -> AsyncIterator[Delta | Done]:
    """Ответ выбранной модели по кускам, последним кадром — `Done` с замерами."""
    if provider not in PROVIDERS:
        raise ValueError(f"Модели {provider!r} нет. Есть: {', '.join(PROVIDERS)}.")

    client = _local if provider == "local" else _cloud_client()
    extra = {} if provider == "local" else {"extra_body": {"thinking": {"type": "disabled"}}}
    messages = messages_for(question, block, think=think, provider=provider)

    started = time.perf_counter()
    first_token: float | None = None
    first_answer: float | None = None
    finish_reason: str | None = None
    usage = None
    parts: list[str] = []
    splitter = _ThinkSplitter()

    stream = await client.chat.completions.create(
        model=MODELS[provider],
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=True,
        stream_options={"include_usage": True},
        **extra,
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
        provider=provider,
        model=MODELS[provider],
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


async def answer(question: str, block: str, *, provider: str, think: bool = False) -> Done:
    """Ответ целиком: для прогонов стрим не нужен, а замеры — те же."""
    async for item in stream_answer(question, block, provider=provider, think=think):
        if isinstance(item, Done):
            return item
    raise RuntimeError("Стрим закончился без итога.")
