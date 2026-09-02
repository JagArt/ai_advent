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
DEFAULT_EXPERTS = ("Аналитик", "Инженер", "Критик")
DEFAULT_PROMPT = """\
Интернет-магазин заметил, что за последний месяц количество заказов выросло на 20%, но общая прибыль снизилась на 10%.

Показатель	Прошлый месяц	Текущий месяц
Количество заказов	10 000	12 000
Средний чек	50 €	42 €
Средняя себестоимость заказа	30 €	29 €
Расходы на рекламу	80 000 €	120 000 €

Почему прибыль могла снизиться, несмотря на рост заказов?
"""

client = AsyncOpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


@dataclass(frozen=True)
class AnswerDelta:
    content: str = ""
    finish_reason: str | None = None
    stage: str | None = None


def _messages(system: str, user: str) -> list[ChatCompletionMessageParam]:
    return [
        ChatCompletionSystemMessageParam(role="system", content=system),
        ChatCompletionUserMessageParam(role="user", content=user),
    ]


def _build_system_prompt(*, step_by_step: bool, experts: list[str]) -> str:
    blocks = [FREE_SYSTEM_PROMPT]
    if step_by_step:
        blocks.append("Решай пошагово.")
    if experts:
        listed = "\n".join(f"- {name}" for name in experts)
        headings = "\n".join(f"## {name}" for name in experts)
        blocks.append(
            "В этом запросе собрана группа экспертов. Каждый эксперт решает задачу "
            "самостоятельно, со своей профессиональной точки зрения.\n\n"
            f"Эксперты:\n{listed}\n\n"
            "Ответь от лица каждого эксперта по отдельности. "
            "Для каждого эксперта начни секцию с заголовка ровно в таком виде:\n\n"
            f"{headings}\n\n"
            "Не пропускай ни одного эксперта."
        )
    return "\n\n".join(blocks)


async def _stream_chat(messages: list[ChatCompletionMessageParam]) -> AsyncIterator[AnswerDelta]:
    stream = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
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


async def stream_answer(
    prompt: str,
    *,
    step_by_step: bool = False,
    meta_prompt: bool = False,
    experts: list[str] | None = None,
) -> AsyncIterator[AnswerDelta]:
    names = [name.strip() for name in (experts or []) if name.strip()]
    system = _build_system_prompt(step_by_step=step_by_step, experts=names)

    if not meta_prompt:
        async for delta in _stream_chat(_messages(system, prompt)):
            yield delta
        return

    yield AnswerDelta(stage="prompt")
    generated_chunks: list[str] = []
    compose_prompt = (
        "Составь подробный промпт, который поможет решить следующую задачу. "
        "Промпт должен содержать саму задачу и инструкции, как её решать. "
        "Верни только готовый промпт, без пояснений и без markdown-обёртки.\n\n"
        f"Задача:\n{prompt}"
    )
    async for delta in _stream_chat(_messages(FREE_SYSTEM_PROMPT, compose_prompt)):
        if delta.content:
            generated_chunks.append(delta.content)
        yield delta

    generated = "".join(generated_chunks).strip()
    if not generated:
        raise RuntimeError("Модель не вернула промпт для второго шага")

    yield AnswerDelta(stage="answer")
    async for delta in _stream_chat(_messages(system, generated)):
        yield delta
