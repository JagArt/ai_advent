"""Ход `/help`: маршрут, предзагрузка контекста через MCP, цикл модели с инструментами.

Контекст у ответа двух родов, и приходят они по-разному. Что код кладёт в промпт
сам, до первого токена модели: ветка и состояние дерева (`git_branch`,
`git_status`) и пять разделов документации по словам вопроса (`docs_search`).
Это гарантированный минимум: даже если модель не позовёт ни одного инструмента,
она отвечает, зная, на какой ветке проект и что про него написано. Что модель
берёт сама по ходу ответа: `git_diff`, `git_files`, ещё один `docs_search` с
другими словами, `docs_read` раздела целиком. Оба рода идут через одну сессию
MCP — ту, что открыл ассистент при старте.

Так выбрано из-за локальной модели. Чистый агент — «всё через инструменты,
модель решает сама» — на 4B-модели регулярно отвечает без единого вызова, то
есть без контекста вовсе; чистая предзагрузка без инструментов не даст облачной
модели копнуть глубже вопроса. Гибрид даёт слабой модели готовые выдержки, а
сильной — ещё и инструменты.

Разделы документации, найденные по ходу, нумеруются продолжением — [6], [7] —
чтобы ссылка в ответе указывала на один источник, а не на «первый из второго
поиска».
"""

import json
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
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
DOCS_LIMIT = 5

# Бюджеты по провайдерам: у локальной модели окно короче, и выдержки режутся жёстче.
MAX_ROUNDS = {"local": 4, "cloud": 6}
EXCERPT_CHARS = {"local": 1200, "cloud": 2500}
EXCERPTS_TOTAL = {"local": 4500, "cloud": 12000}
TOOL_RESULT_CHARS = {"local": 3500, "cloud": 9000}

SYSTEM_PROMPT = (
    "Ты — ассистент разработчика проекта AI Advent. Это репозиторий-дневник: каждая итерация "
    "лежит в своей папке dayN с README.md и TASK.md, общий README — в корне, код каждого дня — "
    "в его папке.\n"
    "Состояние проекта сейчас: {state}.\n"
    "Инструменты: docs_search и docs_read — документация, git_diff, git_files, git_status, "
    "git_branch — репозиторий. Про изменения в коде, файлы и структуру папок спрашивай их, "
    "а не угадывай. Отвечай по-русски. Сейчас {now}."
)

# Правила идут в пользовательском сообщении после выдержек, а не в системном, — это
# компоновка day28, собранная под слабую модель. С правилами в системном сообщении
# qwen3-4b на части вопросов отвечала одной строкой «[3]»: номер выдержки без текста.
RULES = (
    "Правила ответа:\n"
    "1. Отвечай связным текстом, два-четыре предложения, без вступлений и без пересказа вопроса.\n"
    "2. Отвечай по выдержкам и по данным инструментов. После утверждения ставь номер выдержки "
    "в квадратных скобках, например [2]. Номер — дополнение к предложению, а не ответ.\n"
    "3. Числа, пути, порты и команды приводи ровно так, как в источниках.\n"
    "4. Если ответ есть в выдержках — отвечай сразу, инструменты не нужны. Если выдержек не "
    "хватает — вызови docs_search с другими словами или docs_read, чтобы прочитать раздел целиком.\n"
    "5. Чего нет ни в выдержках, ни в ответах инструментов — так и скажи."
)


# --- соединение с MCP-сервером -----------------------------------------------


@dataclass(frozen=True)
class Connection:
    """Сессия с сервером проекта и то, что он о себе рассказал при рукопожатии."""

    session: ClientSession
    server: str
    protocol: str
    capabilities: list[str]
    tools: list[types.Tool]
    elapsed_ms: int

    @property
    def command(self) -> str:
        return f"{Path(sys.executable).name} day31/{SERVER_PATH.name}"

    def openai_tools(self, provider: str) -> list[dict[str, Any]]:
        """`inputSchema` MCP кладётся в `parameters` как есть: это один и тот же JSON Schema.

        Локальной модели — те же инструменты, но с описаниями в одну строку и без
        `title` у полей: полный каталог занимает около 1400 токенов, и с ним qwen3-4b
        отвечала на вопрос про порт одной строкой «[1]». С коротким каталогом и
        подсказкой `ANSWER_HINT` та же модель отвечает предложением и зовёт
        инструменты там, где они нужны.
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
                capabilities=sorted(init.capabilities.model_dump(exclude_none=True)),
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


# --- предзагрузка ------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    number: int
    path: str
    heading: str
    body: str

    def as_dict(self) -> dict[str, Any]:
        return {"number": self.number, "path": self.path, "heading": self.heading}


@dataclass
class Context:
    """Что код положил в промпт до модели: ветка, дерево, разделы документации."""

    branch: ToolResult
    status: ToolResult
    docs: ToolResult
    sources: list[Source] = field(default_factory=list)

    @property
    def state(self) -> str:
        """Одна строка о проекте для системного промпта."""
        parts: list[str] = []
        if self.branch.ok:
            info = self.branch.payload
            line = f"ветка {info['branch']}"
            if info.get("upstream"):
                sync = []
                if info.get("ahead"):
                    sync.append(f"впереди на {info['ahead']}")
                if info.get("behind"):
                    sync.append(f"позади на {info['behind']}")
                line += f" (upstream {info['upstream']}" + (", " + ", ".join(sync) if sync else ", синхронизирована") + ")"
            if info.get("head"):
                # Хеш и дата, без темы коммита. С темой «Add day30 with a LAN service…» в
                # промпте qwen3-4b приписывала LAN-сервис новым файлам day31 — и с оговоркой
                # «это уже закоммичено» тоже. Тема остаётся в payload git_branch: кому
                # нужна, тот позовёт инструмент, а в строке «→ проект» её видит человек.
                head = info["head"]
                line += f"; последний коммит {head['short']} от {head['date'][:10]}"
            parts.append(line)
        else:
            parts.append(f"ветка неизвестна ({self.branch.text})")

        if self.status.ok:
            status = self.status.payload
            if status["clean"]:
                parts.append("рабочее дерево чистое")
            else:
                pieces = []
                for key, title in (("staged", "в индексе"), ("modified", "изменено"), ("untracked", "новых")):
                    if status[key]:
                        shown = ", ".join(status[key][:6]) + (", …" if len(status[key]) > 6 else "")
                        pieces.append(f"{title} {len(status[key])}: {shown}")
                parts.append("в рабочем дереве " + "; ".join(pieces))
        return "; ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch.payload if self.branch.ok else {"error": self.branch.text},
            "status": self.status.payload if self.status.ok else {"error": self.status.text},
            "docs": (
                {"count": self.docs.payload["count"], "strategy": self.docs.payload["strategy"], "index": self.docs.payload["index"]}
                if self.docs.ok
                else {"error": self.docs.text}
            ),
            "sources": [source.as_dict() for source in self.sources],
            "ms": self.branch.ms + self.status.ms + self.docs.ms,
        }


def _sources(hits: list[dict[str, Any]], start: int) -> list[Source]:
    return [Source(start + offset, hit["path"], hit["heading"], hit["body"]) for offset, hit in enumerate(hits)]


async def prefetch(session: ClientSession, question: str) -> Context:
    branch = await call(session, "git_branch")
    status = await call(session, "git_status")
    docs = await call(session, "docs_search", {"query": question, "limit": DOCS_LIMIT})
    context = Context(branch, status, docs)
    if docs.ok:
        context.sources = _sources(docs.payload["hits"], 1)
    return context


# --- промпт ------------------------------------------------------------------


def _excerpts(sources: list[Source], provider: str) -> str:
    """Выдержки под номерами, с путём и разделом в шапке. Режутся по бюджету провайдера."""
    budget = EXCERPTS_TOTAL[provider]
    parts: list[str] = []
    for source in sources:
        body = source.body[: min(EXCERPT_CHARS[provider], budget)]
        if not body:
            break
        budget -= len(body)
        tail = "\n[выдержка обрезана — docs_read отдаст раздел целиком]" if len(body) < len(source.body) else ""
        parts.append(f"[{source.number}] {source.path} · {source.heading}\n{body.strip()}{tail}")
    return "\n\n".join(parts)


def messages_for(question: str, context: Context, provider: str) -> list[ChatCompletionMessageParam]:
    """Вопрос первым, выдержки после, правила — в системном сообщении: порядок из day28, под слабую модель."""
    system = SYSTEM_PROMPT.format(
        state=context.state,
        now=datetime.now().astimezone().isoformat(timespec="minutes"),
    )
    if context.sources:
        excerpts = f"Выдержки из документации, каждая под своим номером:\n\n{_excerpts(context.sources, provider)}"
    else:
        excerpts = f"Выдержек из документации нет: {context.docs.text} Попробуй docs_search с другими словами."
    user = f"Вопрос: {question}\n\n{excerpts}\n\n{RULES}\n\nОтветь на вопрос: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --- события хода ------------------------------------------------------------


@dataclass(frozen=True)
class Prefetched:
    context: Context


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
class Done:
    route: llm.Route
    text: str
    sources: list[Source]
    rounds: int
    calls: int
    finish_reason: str | None
    ttft_ms: int | None
    total_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None


Event = Prefetched | Chunk | Reasoning | ToolCalled | Done


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


def _renumber(result: ToolResult, sources: list[Source]) -> ToolResult:
    """Разделы из повторного поиска получают номера продолжением общего списка."""
    if not (result.ok and result.name == "docs_search" and isinstance(result.payload, dict)):
        return result
    known = {(source.path, source.heading): source.number for source in sources}
    hits = []
    for hit in result.payload["hits"]:
        key = (hit["path"], hit["heading"])
        if key not in known:
            # Раздел новый — следующий номер; уже виденный — его прежний номер.
            known[key] = (sources[-1].number + 1) if sources else 1
            sources.append(Source(known[key], hit["path"], hit["heading"], hit["body"]))
        hits.append({**hit, "number": known[key]})
    payload = dict(result.payload)
    payload["hits"] = hits
    payload["note"] = "Номера разделов продолжают список выдержек из вопроса — ссылайся на них."
    return ToolResult(result.name, result.arguments, True, payload, json.dumps(payload, ensure_ascii=False), result.ms)


async def answer(connection: Connection, question: str, route: llm.Route) -> AsyncIterator[Event]:
    """Ход целиком: предзагрузка, промпт, раунды с инструментами, итог с источниками."""
    started = time.perf_counter()
    provider = route.provider

    context = await prefetch(connection.session, question)
    yield Prefetched(context)

    messages = messages_for(question, context, provider)
    tools = connection.openai_tools(provider)
    sources = list(context.sources)
    first_token: float | None = None
    calls_made = 0
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

        text_parts.extend(round_text)
        if not tool_calls:
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
                result = _renumber(await call(connection.session, tool_call.name, arguments), sources)
            calls_made += 1
            yield ToolCalled(number, result)
            messages.append(_tool_message(tool_call, result, provider))
    else:
        finish = "max_rounds"
        number = MAX_ROUNDS[provider]

    total_ms = round((time.perf_counter() - started) * 1000)
    yield Done(
        route=route,
        text="".join(text_parts).strip(),
        sources=sources,
        rounds=number,
        calls=calls_made,
        finish_reason=finish,
        ttft_ms=None if first_token is None else round((first_token - started) * 1000),
        total_ms=total_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens or None,
    )
