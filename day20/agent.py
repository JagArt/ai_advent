"""Ход агента над объединённым каталогом: тот же цикл, что в day19, другой каталог.

Инструментов теперь тринадцать и они с трёх разных серверов, а имена у них с
префиксом: `git__search`, `docs__search`, `vault__save_file`. Выбор сервера —
работа модели, маршрут — работа реестра, и в промпте сказано главное про оба:
одноимённые инструменты надо звать полным именем, а данные между серверами
передаются ссылками `{"$from": N}`, а не пересказом.

Потолок раундов выше, чем в day19: длинный флоу — это семь вызовов, то есть
восемь проходов к модели минимум, и упираться в потолок на сверке хешей обидно.
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from openai.types.chat import ChatCompletionMessageParam

import flow
import registry as reg
from llm import ToolCall, stream_chat

SYSTEM_PROMPT = (
    "Ты — агент над реестром MCP-серверов проекта AI Advent. Серверов несколько, "
    "и у каждого своё дело:\n{servers}\n"
    "Имя инструмента всегда с префиксом сервера: docs__search, git__search, "
    "vault__save_file. Одноимённые инструменты у разных серверов есть — "
    "{collisions}: выбирай сервер по смыслу задачи и зови полным именем.\n"
    "Инструменты возвращают не данные, а артефакт: номер, превью и хеш. Данные "
    "между серверами передаются ссылкой: в аргумент, который её ждёт, пиши "
    "{{\"$from\": номер артефакта}}. Никогда не переписывай найденное своими "
    "словами в аргументы — реестр подставит данные сам, точь-в-точь.\n"
    "Один вызов за раунд. Цифры, пути, хеши и номера артефактов бери из ответов "
    "инструментов, не придумывай. Если инструмент отказал, прочитай причину: в ней "
    "сказано, какой артефакт нужен и кто его создаёт.\n"
    "Отвечай по-русски, коротко. Сейчас {now}."
)
TEMPERATURE = 0.3
MAX_TOKENS = 1200
MAX_ROUNDS = 14


@dataclass(frozen=True)
class Connected:
    """Рукопожатие со всеми серверами реестра: кто на связи и что умеет."""

    info: dict[str, Any]
    run_id: int


@dataclass(frozen=True)
class Chunk:
    text: str


@dataclass(frozen=True)
class Call:
    """Один вызов инструмента: куда ушёл, что подставили, что вернулось."""

    round: int
    frame: dict[str, Any]


@dataclass(frozen=True)
class Done:
    rounds: int
    calls: int
    finish_reason: str | None
    run_id: int
    verdict: dict[str, Any] | None = None


Event = Connected | Chunk | Call | Done


@dataclass
class Round:
    """Что модель наговорила за один проход."""

    text: str = ""
    finish_reason: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)


def system_prompt(registry: reg.Registry) -> str:
    servers = "\n".join(
        f"- {item['name']}: {item['about']} ({len(item['tools'])} инструментов)"
        for item in registry.online
    )
    collisions = (
        ", ".join(
            f"{name} есть у " + " и ".join(qualified.split(reg.SEPARATOR)[0] for qualified in names)
            for name, names in registry.info()["collisions"].items()
        )
        or "сейчас таких нет"
    )
    return SYSTEM_PROMPT.format(
        servers=servers,
        collisions=collisions,
        now=datetime.now().astimezone().isoformat(timespec="seconds"),
    )


def _assistant_message(round_: Round) -> ChatCompletionMessageParam:
    return {
        "role": "assistant",
        "content": round_.text,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in round_.tool_calls
        ],
    }


async def answer(
    prompt: str, *, mode: str = "chat", query: str | None = None
) -> AsyncIterator[Event]:
    """Ход агента целиком: соединения со всеми серверами живут до конца ответа.

    Прогон открывается всегда, даже для вопроса в чате: маршруты одного вида,
    и в журнале должно быть видно всё, что уходило на серверы.
    """
    run_id = reg.open_run(query or prompt, mode)
    calls = 0
    status = "done"

    async with reg.connect() as registry:
        yield Connected(registry.info(), run_id)

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt(registry)},
            {"role": "user", "content": prompt},
        ]
        tools = registry.openai_tools()

        for number in range(1, MAX_ROUNDS + 1):
            round_ = Round()
            async for delta in _pass(messages, tools=tools):
                if isinstance(delta, Chunk):
                    yield delta
                else:
                    round_ = delta

            if not round_.tool_calls:
                verdict = flow.verdict(run_id) if mode == "agent" else None
                reg.finish_run(run_id, status, verdict=verdict)
                yield Done(number, calls, round_.finish_reason, run_id, verdict)
                return

            messages.append(_assistant_message(round_))

            for call in round_.tool_calls:
                record = await _dispatch(registry, call, run_id=run_id, source=mode)
                calls += 1
                if record.status != "ok":
                    status = "failed"
                yield Call(number, record.frame())
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": record.text()}
                )

        # Потолок раундов: лучше честный итог, чем бесконечное хождение по серверам.
        verdict = flow.verdict(run_id) if mode == "agent" else None
        reg.finish_run(run_id, "max_rounds", verdict=verdict)
        yield Done(MAX_ROUNDS, calls, "max_rounds", run_id, verdict)


async def _dispatch(
    registry: reg.Registry, call: ToolCall, *, run_id: int, source: str
) -> reg.Dispatch:
    try:
        arguments = json.loads(call.arguments or "{}")
    except json.JSONDecodeError as exc:
        return registry.refuse(
            call.name,
            f"Аргументы не разобрались как JSON: {exc}",
            "bad_arguments",
            run_id=run_id,
            source=source,
        )

    return await registry.dispatch(call.name, arguments, run_id=run_id, source=source)


async def _pass(
    messages: list[ChatCompletionMessageParam],
    *,
    tools: list[dict[str, Any]] | None,
) -> AsyncIterator[Chunk | Round]:
    """Отдаёт куски текста по мере их появления, а последним — итог прохода."""
    round_ = Round()

    async for delta in stream_chat(
        messages,
        tools=tools,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    ):
        if delta.content:
            round_.text += delta.content
            yield Chunk(delta.content)
        if delta.finish_reason:
            round_.finish_reason = delta.finish_reason
            round_.tool_calls = delta.tool_calls

    yield round_
