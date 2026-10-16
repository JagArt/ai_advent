"""Ход ассистента поддержки: чистый агент — всё, что он знает, он берёт инструментами.

Код не кладёт в промпт ни текста тикета, ни карточки пользователя, ни статей.
Системное сообщение несёт роль и указатели: какой тикет открыт, чей он, и что
содержимое надо взять через `crm_get_ticket`, `crm_get_user`, `kb_search`.
Дальше модель решает сама, в каком порядке и что звать, — и в ленте хода
видно каждое обращение к данным. Это сознательно противоположно гибриду
[day31](../day31/agent.py), где ветку и пять разделов документации код
подкладывал заранее: здесь интересно, справится ли модель без подстраховки,
и что делать, если нет.

Что делать, если нет, — толчок. Если первый раунд закончился без единого
вызова, значит, модель ответила, не читая ни тикета, ни базы знаний. Такой
ответ отбрасывается, в диалог добавляется сообщение «ты не открыл тикет и не
искал в базе знаний — вызови инструменты», и модели даётся ещё раунд. Толчок
один на ход, он виден в ленте и считается в итоге.

Разделы базы знаний нумеруются по порядку появления в результатах `kb_search`
и `kb_read` — продолжением, не заново на каждый поиск, — чтобы номер в ответе
указывал на один источник.
"""

import json
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai.types.chat import ChatCompletionMessageParam

import llm

SERVER_PATH = Path(__file__).resolve().parent / "server.py"

TEMPERATURE = 0.2
MAX_TOKENS = 900
HISTORY_TURNS = 6

# Бюджеты по провайдерам: у локальной модели окно короче, и ответы инструментов режутся жёстче.
MAX_ROUNDS = {"local": 5, "cloud": 6}
TOOL_RESULT_CHARS = {"local": 3500, "cloud": 9000}
EXCERPT_CHARS = {"local": 900, "cloud": 2500}

SYSTEM_PROMPT = (
    "Ты — ассистент оператора поддержки сервиса «Планёрка», таск-трекера для команд. Оператор "
    "разбирает обращение клиента и задаёт тебе вопросы. Отвечаешь ты оператору, а не клиенту, "
    "если оператор не просит составить ответ клиенту.\n"
    "Открыт тикет {ticket_id} пользователя {user_id}. Их содержимого ты не знаешь: текст обращения "
    "и переписку даёт crm_get_ticket, карточку пользователя — тариф, статус аккаунта, почту, 2FA, "
    "счётчик входов, команду и SSO, платформу и версию приложения — даёт crm_get_user. Как устроен "
    "продукт и что делать в типовой ситуации, ты тоже не знаешь: это даёт только kb_search по базе "
    "знаний, а похожие обращения и их решения — crm_search_tickets. Не отвечай по догадке. "
    "Отвечай по-русски. Сейчас {now}."
)

# Правила — в пользовательском сообщении, после вопроса: компоновка day28/day31 под слабую модель.
RULES = (
    "Правила ответа:\n"
    "1. Сначала инструменты: открой тикет (crm_get_ticket) и карточку пользователя (crm_get_user), "
    "найди подходящий раздел в базе знаний (kb_search). Без них не отвечай.\n"
    "2. Причину ищи в карточке пользователя — статус аккаунта, подтверждение почты, счётчик неудачных "
    "входов, 2FA, команда и её SSO, версия приложения, оплата — и только потом подбирай к ней раздел "
    "базы знаний. Причина должна подтверждаться полем карточки: блокировка — счётчиком входов, "
    "неподтверждённая почта — статусом аккаунта, устаревшее приложение — версией. Раздел, который "
    "карточка не подтверждает, причиной не называй. В kb_search передавай слова причины, например "
    "«блокировка неудачных попыток» или «домен не разрешён SSO», а не «не работает авторизация».\n"
    "3. Отвечай связным текстом, два-пять предложений, без вступлений. После утверждения из базы "
    "знаний ставь номер раздела в квадратных скобках, например [2]. Номер — дополнение к "
    "предложению, а не ответ.\n"
    "4. Числа, сроки, названия пунктов меню приводи ровно так, как в источниках.\n"
    "5. Чего нет ни в тикете, ни в карточке, ни в базе знаний — так и скажи."
)

NUDGE = (
    "Ты ответил, не вызвав ни одного инструмента, — значит, не читал ни тикет, ни карточку "
    "пользователя, ни базу знаний. Этот ответ отброшен. Вызови crm_get_ticket, crm_get_user и "
    "kb_search, затем ответь заново по их данным."
)


# --- соединение с MCP-сервером -----------------------------------------------


@dataclass(frozen=True)
class Connection:
    """Сессия с сервером поддержки и то, что он о себе рассказал при рукопожатии."""

    session: ClientSession
    server: str
    protocol: str
    tools: list[types.Tool]
    elapsed_ms: int

    @property
    def command(self) -> str:
        return f"{Path(sys.executable).name} day33/{SERVER_PATH.name}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "server": self.server,
            "protocol": self.protocol,
            "command": self.command,
            "elapsed_ms": self.elapsed_ms,
            "tools": [
                {"name": tool.name, "description": (tool.description or "").strip().splitlines()[0]}
                for tool in self.tools
            ],
        }

    def openai_tools(self, provider: str) -> list[dict[str, Any]]:
        """`inputSchema` MCP кладётся в `parameters` как есть: это один и тот же JSON Schema.

        Локальной модели — описания в одну строку и без `title` у полей: с полным
        каталогом qwen3-4b в day31 «залипала» и отвечала одним номером выдержки.
        """
        tools = []
        for tool in self.tools:
            description = (tool.description or "").strip()
            schema = json.loads(json.dumps(tool.input_schema))
            if provider == "local":
                description = description.splitlines()[0]
                for prop in schema.get("properties", {}).values():
                    prop.pop("title", None)
                    if "description" in prop:
                        prop["description"] = prop["description"].split(". ")[0].rstrip(".")[:90]
            tools.append(
                {
                    "type": "function",
                    "function": {"name": tool.name, "description": description, "parameters": schema},
                }
            )
        return tools


@asynccontextmanager
async def connect() -> AsyncIterator[Connection]:
    """Подпроцесс сервера живёт ровно столько, сколько открыт этот блок."""
    started = time.perf_counter()
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER_PATH)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            listed = await session.list_tools()
            yield Connection(
                session=session,
                server=f"{init.server_info.name} {init.server_info.version}".strip(),
                protocol=init.protocol_version,
                tools=listed.tools,
                elapsed_ms=round((time.perf_counter() - started) * 1000),
            )


@dataclass(frozen=True)
class ToolResult:
    """Один вызов инструмента: что просили, что вернулось, сколько заняло."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    payload: Any
    text: str
    ms: int

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "ok": self.ok, "ms": self.ms, "error": None if self.ok else self.text}


def _result_text(result: types.CallToolResult) -> str:
    text = "\n".join(block.text for block in result.content if isinstance(block, types.TextContent))
    if not text and result.structured_content is not None:
        text = json.dumps(result.structured_content, ensure_ascii=False)
    return text or "(инструмент вернул пустой ответ)"


def _payload(result: types.CallToolResult) -> Any:
    payload = result.structured_content
    if payload is None:
        try:
            return json.loads(_result_text(result))
        except json.JSONDecodeError:
            return {"text": _result_text(result)}
    if isinstance(payload, dict) and set(payload) == {"result"}:
        return payload["result"]
    return payload


async def call(session: ClientSession, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
    """Единственная дверь к серверу. Отказ инструмента — не исключение, а результат с `ok=False`."""
    arguments = arguments or {}
    started = time.perf_counter()
    try:
        result = await session.call_tool(name, arguments)
    except Exception as exc:  # noqa: BLE001
        text = f"Сервер не ответил: {type(exc).__name__}: {exc}"
        return ToolResult(name, arguments, False, None, text, round((time.perf_counter() - started) * 1000))

    ms = round((time.perf_counter() - started) * 1000)
    if result.is_error:
        return ToolResult(name, arguments, False, None, _result_text(result).removeprefix(f"Error executing tool {name}: "), ms)
    return ToolResult(name, arguments, True, _payload(result), _result_text(result), ms)


# --- источники ---------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    number: int
    slug: str
    title: str
    heading: str

    def as_dict(self) -> dict[str, Any]:
        return {"number": self.number, "slug": self.slug, "title": self.title, "heading": self.heading}


WHOLE = "статья целиком"


def _renumber(result: ToolResult, sources: list[Source], provider: str) -> ToolResult:
    """Разделы из `kb_search` и `kb_read` получают номера продолжением общего списка.

    Тела разделов для локальной модели режутся по бюджету: ей важнее увидеть
    пять разделов коротко, чем два целиком.
    """
    if not (result.ok and isinstance(result.payload, dict)):
        return result
    known = {(source.slug, source.heading): source.number for source in sources}

    def number_for(slug: str, title: str, heading: str) -> int:
        key = (slug, heading)
        if key not in known:
            known[key] = (sources[-1].number + 1) if sources else 1
            sources.append(Source(known[key], slug, title, heading))
        return known[key]

    payload = dict(result.payload)
    if result.name == "kb_search":
        hits = []
        for hit in payload["hits"]:
            body = hit["body"]
            cut = EXCERPT_CHARS[provider]
            if len(body) > cut:
                body = body[:cut] + " … [раздел обрезан — kb_read отдаст его целиком]"
            hits.append({**hit, "number": number_for(hit["slug"], hit["title"], hit["heading"]), "body": body})
        payload["hits"] = hits
        payload["note"] = "Номера разделов сквозные для всего ответа — ссылайся на них."
    elif result.name == "kb_read":
        payload["number"] = number_for(payload["slug"], payload["title"], payload["heading"] or WHOLE)
    else:
        return result
    return ToolResult(result.name, result.arguments, True, payload, json.dumps(payload, ensure_ascii=False), result.ms)


# --- промпт ------------------------------------------------------------------


def messages_for(
    question: str,
    ticket_id: str,
    user_id: str,
    history: list[dict[str, str]],
) -> list[ChatCompletionMessageParam]:
    """Системное сообщение с указателями, история прошлых ходов, вопрос с правилами."""
    system = SYSTEM_PROMPT.format(
        ticket_id=ticket_id,
        user_id=user_id,
        now=datetime.now().astimezone().isoformat(timespec="minutes"),
    )
    messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]
    for turn in history[-HISTORY_TURNS * 2 :]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})  # type: ignore[arg-type]
    messages.append(
        {"role": "user", "content": f"Вопрос оператора: {question}\n\n{RULES}\n\nОтветь на вопрос оператора: {question}"}
    )
    return messages


# --- события хода ------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    text: str


@dataclass(frozen=True)
class Reasoning:
    text: str


@dataclass(frozen=True)
class ToolCalled:
    round: int
    result: ToolResult


@dataclass(frozen=True)
class Nudged:
    """Первый раунд прошёл без вызовов: черновик отброшен, модели дан толчок."""

    round: int
    draft: str


@dataclass(frozen=True)
class Cleared:
    """Раунд кончился вызовами: текст до них — комментарий к инструментам, не ответ."""

    round: int


@dataclass(frozen=True)
class Done:
    provider: str
    model: str
    text: str
    sources: list[Source]
    rounds: int
    calls: int
    tools_used: list[str]
    nudged: bool
    finish_reason: str | None
    ttft_ms: int | None
    total_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "title": llm.TITLES[self.provider],
            "text": self.text,
            "sources": [source.as_dict() for source in self.sources],
            "rounds": self.rounds,
            "calls": self.calls,
            "tools_used": self.tools_used,
            "nudged": self.nudged,
            "finish_reason": self.finish_reason,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


Event = Chunk | Reasoning | ToolCalled | Nudged | Cleared | Done


def _assistant_message(text: str, calls: tuple[llm.ToolCall, ...]) -> ChatCompletionMessageParam:
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if calls:
        message["tool_calls"] = [
            {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
            for call in calls
        ]
    return message  # type: ignore[return-value]


def _tool_message(call: llm.ToolCall, result: ToolResult, provider: str) -> ChatCompletionMessageParam:
    text = result.text
    limit = TOOL_RESULT_CHARS[provider]
    if len(text) > limit:
        text = text[:limit] + "\n[ответ инструмента обрезан]"
    return {"role": "tool", "tool_call_id": call.id, "content": text}


async def answer(
    connection: Connection,
    *,
    question: str,
    ticket_id: str,
    user_id: str,
    provider: str,
    history: list[dict[str, str]] | None = None,
) -> AsyncIterator[Event]:
    """Ход целиком: промпт с указателями, раунды с инструментами, толчок, итог с источниками."""
    started = time.perf_counter()
    messages = messages_for(question, ticket_id, user_id, history or [])
    tools = connection.openai_tools(provider)
    sources: list[Source] = []
    first_token: float | None = None
    calls_made = 0
    tools_used: list[str] = []
    nudged = False
    prompt_tokens: int | None = None
    completion_tokens = 0
    text_parts: list[str] = []
    finish: str | None = None

    for number in range(1, MAX_ROUNDS[provider] + 1):
        round_text: list[str] = []
        tool_calls: tuple[llm.ToolCall, ...] = ()

        async for delta in llm.stream_chat(
            messages, provider=provider, tools=tools, temperature=TEMPERATURE, max_tokens=MAX_TOKENS
        ):
            if delta.reasoning:
                yield Reasoning(delta.reasoning)
            if delta.content:
                first_token = first_token or time.perf_counter()
                round_text.append(delta.content)
                yield Chunk(delta.content)
            if delta.finish_reason:
                finish = delta.finish_reason
                tool_calls = delta.tool_calls
                prompt_tokens = delta.prompt_tokens if delta.prompt_tokens is not None else prompt_tokens
                completion_tokens += delta.completion_tokens or 0

        if not tool_calls and calls_made == 0 and not nudged:
            # Ответ без единого обращения к данным — не ответ. Черновик отбрасывается,
            # модель получает толчок и ещё раунд; время до первого токена считается заново.
            nudged = True
            draft = "".join(round_text).strip()
            yield Nudged(number, draft)
            messages.append(_assistant_message(draft, ()))
            messages.append({"role": "user", "content": NUDGE})
            first_token = None
            continue

        if tool_calls:
            # Комментарий вроде «открою тикет» не ответ: страница его стирает, в итог он не идёт.
            if "".join(round_text).strip():
                yield Cleared(number)
            first_token = None
        else:
            text_parts.extend(round_text)
            break

        messages.append(_assistant_message("".join(round_text), tool_calls))
        for tool_call in tool_calls:
            try:
                arguments = json.loads(tool_call.arguments or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("аргументы должны быть объектом")
            except (json.JSONDecodeError, ValueError) as exc:
                result = ToolResult(tool_call.name, {}, False, None, f"Аргументы не разобрались как JSON: {exc}", 0)
            else:
                result = _renumber(await call(connection.session, tool_call.name, arguments), sources, provider)
            calls_made += 1
            tools_used.append(result.name)
            yield ToolCalled(number, result)
            messages.append(_tool_message(tool_call, result, provider))
    else:
        finish = "max_rounds"
        number = MAX_ROUNDS[provider]

    yield Done(
        provider=provider,
        model=llm.MODELS[provider],
        text="".join(text_parts).strip(),
        sources=sources,
        rounds=number,
        calls=calls_made,
        tools_used=tools_used,
        nudged=nudged,
        finish_reason=finish,
        ttft_ms=None if first_token is None else round((first_token - started) * 1000),
        total_ms=round((time.perf_counter() - started) * 1000),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens or None,
    )
