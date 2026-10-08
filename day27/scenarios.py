"""Три запроса к локальной модели из терминала, вывод сразу в markdown.

Каждый запрос уходит дважды — с рассуждением и с `/no_think`, — чтобы было видно,
сколько стоит размышление Qwen3 там, где оно не нужно, и где без него нельзя.
"""

import asyncio
import sys

from llm import BASE_URL, MODEL, Delta, Done, list_models, stream_answer
from prompts import PROMPTS


async def run(prompt: str, think: bool) -> tuple[str, Done]:
    answer: list[str] = []
    done: Done | None = None
    async for item in stream_answer(prompt, think=think):
        if isinstance(item, Delta):
            answer.append(item.content)
        else:
            done = item
    if done is None:
        raise RuntimeError("Стрим закончился без итога.")
    return "".join(answer).strip(), done


def cell(value: object, suffix: str = "") -> str:
    return "—" if value is None else f"{value}{suffix}"


async def main() -> None:
    try:
        models = await list_models()
    except Exception as exc:
        sys.exit(f"LM Studio на {BASE_URL} не отвечает ({exc}). Запустите `lms server start`.")
    if MODEL not in models:
        sys.exit(f"Модели {MODEL} нет на сервере. Доступны: {', '.join(models)}.")

    print(f"Сервер: `{BASE_URL}`, модель: `{MODEL}`\n")

    rows: list[str] = []
    answers: list[str] = []
    for preset in PROMPTS:
        for think in (True, False):
            answer, done = await run(preset.text, think)
            mode = "думает" if think else "/no_think"
            rows.append(
                f"| {preset.level} | {mode} | {cell(done.ttft_ms, ' мс')} | {cell(done.answer_ms, ' мс')} "
                f"| {done.total_ms / 1000:.1f} с | {cell(done.completion_tokens)} "
                f"| {cell(done.reasoning_tokens)} | {cell(done.tokens_per_sec)} | {cell(done.finish_reason)} |"
            )
            answers.append(f"### {preset.level}, {mode}\n\n> {preset.text}\n\nОжидается: {preset.expected}\n\n{answer}\n")

    print("| Запрос | Режим | Первый токен | Начало ответа | Всего | Токенов ответа | Из них рассуждения | Ток/с | finish_reason |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    print("\n".join(rows))
    print()
    print("\n".join(answers))


if __name__ == "__main__":
    asyncio.run(main())
