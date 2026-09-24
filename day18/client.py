"""MCP-клиент к постоянному серверу: подключается по HTTP, а не поднимает подпроцесс.

В day17 сервер жил ровно столько, сколько открыт `async with`. Здесь он
работает сам по себе, и клиент только приходит к нему: закрытие соединения
расписание не останавливает. Если сервер не запущен, `connect()` падает
сразу — это честнее, чем тихо поднять свой экземпляр с другим расписанием.
"""

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import httpx2
import mcp.types as types
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

# Адрес не импортируется из server.py: создание MCPServer настраивает логирование процесса.
URL = "http://127.0.0.1:8765/mcp"


@dataclass(frozen=True)
class Connection:
    session: ClientSession
    server: str
    protocol: str
    capabilities: list[str]
    tools: list[types.Tool]
    elapsed_ms: int

    def info(self) -> dict[str, Any]:
        """То, что уходит на страницу: рукопожатие и инструменты без объектов SDK."""
        return {
            "server": self.server,
            "protocol": self.protocol,
            "capabilities": self.capabilities,
            "url": URL,
            "elapsed_ms": self.elapsed_ms,
            "tools": [
                {
                    "name": tool.name,
                    "title": tool.title,
                    "description": (tool.description or "").strip(),
                    "input_schema": tool.input_schema,
                }
                for tool in self.tools
            ],
        }

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Вызов для кода, а не для модели: структурированный результат или исключение."""
        result = await self.session.call_tool(name, arguments or {})
        if result.is_error:
            raise RuntimeError(result_to_text(result))

        payload = result.structured_content
        if payload is None:
            return json.loads(result_to_text(result))
        # Список SDK заворачивает в {"result": [...]}: у structuredContent корень — объект.
        if set(payload) == {"result"}:
            return payload["result"]
        return payload


@asynccontextmanager
async def connect() -> AsyncIterator[Connection]:
    """Сессия живёт, пока открыт блок; сервер и его расписание — независимо от неё."""
    started = perf_counter()

    async with streamable_http_client(URL) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            listed = await session.list_tools()

            yield Connection(
                session=session,
                server=f"{init.server_info.name} {init.server_info.version}".strip(),
                protocol=init.protocol_version,
                capabilities=sorted(init.capabilities.model_dump(exclude_none=True)),
                tools=listed.tools,
                elapsed_ms=round((perf_counter() - started) * 1000),
            )


def as_openai_tools(tools: list[types.Tool]) -> list[dict[str, Any]]:
    """`inputSchema` инструмента MCP кладётся в `parameters` как есть: это один и тот же JSON Schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": (tool.description or "").strip(),
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


def result_to_text(result: types.CallToolResult) -> str:
    """Ответ инструмента для сообщения `role="tool"`: модель читает текст, а не объекты."""
    text = "\n".join(
        block.text for block in result.content if isinstance(block, types.TextContent)
    )

    if not text and result.structured_content is not None:
        text = json.dumps(result.structured_content, ensure_ascii=False)

    return text or "(инструмент вернул пустой ответ)"


def _pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


async def main() -> None:
    # sys.exit внутри except* снова завернулся бы в группу исключений и напечатал трейсбек.
    unreachable = None
    try:
        async with connect() as connection:
            info = connection.info()

            print(f"Соединение установлено за {info['elapsed_ms']} мс")
            print(f"  адрес:       {info['url']}")
            print(f"  сервер:      {info['server']}")
            print(f"  протокол:    {info['protocol']}")
            print(f"  возможности: {', '.join(info['capabilities'])}")
            print(f"\nИнструментов: {len(info['tools'])}")

            for tool in info["tools"]:
                params = tool["input_schema"].get("properties", {})
                required = set(tool["input_schema"].get("required", []))
                args = ", ".join(
                    name if name in required else f"{name}?" for name in params
                )
                print(f"\n  {tool['name']}({args})")
                print(f"    {tool['description']}")

            print("\nВызов list_jobs()")
            for job in await connection.call("list_jobs"):
                state = "активна" if job["active"] else "завершена"
                print(
                    f"  #{job['id']} {job['kind']:<8} {job['schedule']:<16} {state},"
                    f" запусков {job['runs']}, следующий {job['next_run_at'] or '—'}"
                    + (f", ошибка: {job['last_error']}" if job["last_error"] else "")
                )

            print("\nВызов get_summary(hours=1)")
            print(_pretty(await connection.call("get_summary", {"hours": 1})))
    except* httpx2.ConnectError as group:
        unreachable = group.exceptions[0]

    if unreachable is not None:
        sys.exit(f"Сервер {URL} недоступен: {unreachable}. Запустите python day18/server.py")


if __name__ == "__main__":
    asyncio.run(main())
