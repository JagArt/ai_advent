"""Замеры для README: короткий диалог, длинный, потеря памяти и переполнение контекста.

    python day8/scenarios.py            # все сценарии
    python day8/scenarios.py short long # только выбранные

Диалоги идут через того же агента, что и веб-чат, но в отдельной базе: замер не
попадает в историю приложения. Переполнение проверяется в обход агента — его
защита до API не пускает, а нам нужен именно отказ модели.
"""

import argparse
import asyncio
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

from openai import APIStatusError
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam

import tokens
from agent import Agent, AgentPlan, AgentTurn
from llm import MODEL, complete
from storage import Storage

SHORT_DIALOG = (
    "Объясни в трёх предложениях, что такое идемпотентность в HTTP.",
    "Какие методы идемпотентны, а какие нет?",
    "Как сделать POST идемпотентным?",
)

# Один длинный разговор на одну тему: вход растёт от хода к ходу, пока не упрётся
# в бюджет контекста — дальше он выходит на плато, и это видно в таблице.
LONG_DIALOG = (
    "Объясни в трёх предложениях, что такое идемпотентность в HTTP.",
    "Какие методы HTTP идемпотентны?",
    "Почему POST не идемпотентен?",
    "Что такое ключ идемпотентности?",
    "Где хранить такие ключи на сервере?",
    "Сколько времени их держать?",
    "Как отвечать на повторный запрос с тем же ключом?",
    "Чем идемпотентность отличается от безопасности метода?",
    "Идемпотентен ли DELETE, если ресурса уже нет?",
    "Какой код ответа вернуть на повторный DELETE?",
    "Как идемпотентность связана с ретраями клиента?",
    "Что делать, если ответ потерялся по таймауту?",
    "Как проверить идемпотентность в тестах?",
    "Какие ошибки чаще всего ломают идемпотентность?",
    "Причём тут гонки при параллельных запросах?",
    "Помогает ли блокировка в базе?",
    "Что даёт уникальный индекс по ключу?",
    "Как это выглядит в платёжном API?",
    "Что писать в документации про идемпотентность?",
    "Подведи итог в трёх пунктах.",
)

# Имя даётся в первом ходе, спрашивается в последнем: на большом бюджете агент
# помнит его, на маленьком первый ход уже выпал из окна.
MEMORY_FACT = "Запомни: меня зовут Игорь, мой любимый язык — Python. Ответь одним словом: понял."
MEMORY_FILLER = (
    "Расскажи в двух предложениях, что такое HTTP.",
    "А что такое TCP?",
    "Чем UDP отличается от TCP?",
    "Что такое DNS?",
    "Зачем нужен TLS?",
    "Что такое HTTP/2?",
)
MEMORY_QUESTION = "Как меня зовут и какой язык я люблю?"

# Латиница, потому что оценка и факт на ней расходятся слабее: размер запроса
# нужно подобрать так, чтобы модель наверняка его отвергла.
FILLER = (
    "Idempotency means that repeating the same request produces the same result, "
    "so a client may safely retry a failed call without changing server state. "
)

# Запас над лимитом: 2% с лихвой перекрывают погрешность калибровки.
OVERFLOW_MARGIN = 1.02

# Сколько токенов оставить сообщениям во втором случае: вместе с max_tokens = 384000
# они дают перебор, хотя сами по себе в лимит влезают.
SPLIT_MESSAGE_TOKENS = 700_000
MAX_OUTPUT_TOKENS = 384_000


def table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    print(f"| {' | '.join(headers)} |")
    print(f"| {' | '.join('---' for _ in headers)} |")
    for row in rows:
        print(f"| {' | '.join(str(cell) for cell in row)} |")
    print()


def money(value: float) -> str:
    return f"${value:.6f}"


def user(text: str) -> ChatCompletionUserMessageParam:
    return ChatCompletionUserMessageParam(role="user", content=text)


async def play(agent: Agent, prompts: tuple[str, ...], budget: int) -> list[dict[str, Any]]:
    """Проигрывает диалог ход за ходом и собирает числа каждого хода."""
    rows: list[dict[str, Any]] = []

    for prompt in prompts:
        plan: AgentPlan | None = None
        turn: AgentTurn | None = None
        parts: list[str] = []

        async for event in agent.ask(prompt, budget):
            if isinstance(event, AgentPlan):
                plan = event
            elif isinstance(event, AgentTurn):
                turn = event
            elif event.content:
                parts.append(event.content)

        if plan is None or turn is None:
            raise RuntimeError("Ход не дошёл до конца")

        rows.append(
            {
                "prompt": prompt,
                "answer": "".join(parts),
                "estimated": plan.request_tokens,
                "context_messages": plan.context_messages,
                "dropped": plan.dropped_messages,
                "prompt_tokens": turn.usage.prompt_tokens,
                "cached_tokens": turn.usage.cached_tokens,
                "completion_tokens": turn.usage.completion_tokens,
                "cost": turn.usage.cost_usd,
            },
        )
    return rows


def report(title: str, rows: list[dict[str, Any]], service_cost: float) -> None:
    print(f"### {title}\n")

    spent = 0.0
    table_rows: list[tuple[Any, ...]] = []
    for number, row in enumerate(rows, start=1):
        spent += row["cost"]
        drift = tokens.drift_percent(row["estimated"], row["prompt_tokens"])
        table_rows.append(
            (
                number,
                row["context_messages"],
                row["prompt_tokens"],
                row["cached_tokens"],
                row["completion_tokens"],
                row["prompt_tokens"] + row["completion_tokens"],
                f"{row['estimated']} ({drift:+.1f}%)" if drift is not None else row["estimated"],
                money(row["cost"]),
                money(spent),
            ),
        )

    table(
        (
            "Ход",
            "Контекст, сообщ.",
            "Вход",
            "из них кэш",
            "Ответ",
            "Всего",
            "Оценка",
            "Стоимость хода",
            "Накоплено",
        ),
        table_rows,
    )

    dropped = rows[-1]["dropped"] if rows else 0
    print(f"Ходов: {len(rows)}, вне контекста на последнем ходу: {dropped} сообщ.")
    print(f"Диалог: {money(spent)}, заголовок: {money(service_cost)}, всего: {money(spent + service_cost)}\n")


async def dialog_scenario(title: str, prompts: tuple[str, ...], budget: int) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory() as directory:
        storage = Storage(Path(directory) / "scenarios.db")
        await storage.init()
        session_id = await storage.create_session()
        agent = Agent(session_id, storage, context_budget=budget)

        rows = await play(agent, prompts, budget)
        turns = await agent.turns()

    service_cost = sum(turn["cost_usd"] for turn in turns if turn["kind"] != "turn")
    report(f"{title} (бюджет контекста {budget} токенов)", rows, service_cost)
    return rows


async def short() -> None:
    await dialog_scenario("Короткий диалог", SHORT_DIALOG, 4000)


async def long() -> None:
    await dialog_scenario("Длинный диалог", LONG_DIALOG, 4000)


async def memory() -> None:
    """Одинаковый диалог на двух бюджетах: разница только в том, что агент помнит."""
    prompts = (MEMORY_FACT, *MEMORY_FILLER, MEMORY_QUESTION)

    for budget in (4000, 500):
        rows = await dialog_scenario("Проверка памяти", prompts, budget)
        answer = " ".join(rows[-1]["answer"].split())
        print(f"Бюджет {budget}: «{answer}»\n")


async def calibrate(blocks: int = 200) -> float:
    """Сколько токенов DeepSeek видит в одном блоке текста: нужен точный размер."""
    answer = await complete([user(FILLER * blocks)], temperature=0, max_tokens=1)
    usage = tokens.usage_from(answer.usage)
    per_block = usage.prompt_tokens / blocks
    local = tokens.count_text(FILLER)
    print("### Калибровка размера\n")
    print(f"Блок текста: факт DeepSeek {per_block:.2f} токена, оценка cl100k_base {local}")
    print(f"Лимит модели: {tokens.MODEL_CONTEXT_LIMIT} токенов на вход и выход вместе\n")
    return per_block


async def probe(name: str, blocks: int, per_block: float, max_tokens: int) -> None:
    text = FILLER * blocks
    messages: list[ChatCompletionMessageParam] = [user(text)]
    projected = round(blocks * per_block)

    print(f"**{name}**\n")
    print(f"- сообщений: {len(text) / 1_000_000:.2f} МБ текста, {projected} токенов по факту DeepSeek")
    print(f"- оценка cl100k_base: {tokens.count_messages(messages)} токенов")
    print(f"- max_tokens: {max_tokens}, сумма {projected + max_tokens} против лимита {tokens.MODEL_CONTEXT_LIMIT}")

    try:
        answer = await complete(messages, temperature=0, max_tokens=max_tokens)
    except APIStatusError as error:
        body = error.body if isinstance(error.body, dict) else {}
        message = body.get("message") if isinstance(body, dict) else None
        print(f"- отказ: HTTP {error.status_code}, code {body.get('code')}, type {body.get('type')}")
        print(f"- текст: {message or error.message}\n")
        return

    usage = tokens.usage_from(answer.usage)
    print(f"- запрос приняли: вход {usage.prompt_tokens}, ответ {usage.completion_tokens}, {money(usage.cost_usd)}\n")


async def overflow() -> None:
    print("## Переполнение контекста\n")
    per_block = await calibrate()

    # Первый случай: сообщения сами по себе не влезают в контекст.
    blocks = math.ceil(tokens.MODEL_CONTEXT_LIMIT * OVERFLOW_MARGIN / per_block)
    await probe("Только сообщения, лимит ответа 32 токена", blocks, per_block, 32)

    # Второй случай: сообщения влезают, но вместе с запрошенным ответом — нет.
    split = math.ceil(SPLIT_MESSAGE_TOKENS / per_block)
    await probe(
        f"Сообщения на {SPLIT_MESSAGE_TOKENS} токенов плюс max_tokens {MAX_OUTPUT_TOKENS}",
        split,
        per_block,
        MAX_OUTPUT_TOKENS,
    )


SCENARIOS = {
    "short": short,
    "long": long,
    "memory": memory,
    "overflow": overflow,
}


async def main(names: list[str]) -> None:
    print(f"Модель: {MODEL}, тариф {'пиковый' if tokens.is_peak_hour() else 'непиковый'}\n")
    for name in names:
        await SCENARIOS[name]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="*", choices=sorted(SCENARIOS), default=sorted(SCENARIOS))
    arguments = parser.parse_args()
    # Переполнение отправляет несколько мегабайт текста, поэтому порядок сценариев
    # сохраняем как просили, но по умолчанию идут все и по алфавиту.
    asyncio.run(main(list(arguments.scenarios)))
    sys.exit(0)
