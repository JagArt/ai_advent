"""MCP-клиент: поднимает сервер подпроцессом, здоровается, вызывает инструмент.

Пара `stdio_client` + `ClientSession` взята вместо высокоуровневого `Client`
намеренно: `initialize()` возвращает ответ сервера на рукопожатие — имя, версию
протокола и capabilities, — а это и есть доказательство, что соединение живое.
"""

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PATH = Path(__file__).resolve().parent / "server.py"

server_params = StdioServerParameters(command=sys.executable, args=[str(SERVER_PATH)])


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
            "command": f"{Path(sys.executable).name} {SERVER_PATH.name}",
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


@asynccontextmanager
async def connect() -> AsyncIterator[Connection]:
    """Подпроцесс живёт ровно столько, сколько открыт этот блок."""
    started = perf_counter()

    async with stdio_client(server_params) as (read, write):
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


async def main() -> None:
    async with connect() as connection:
        info = connection.info()

        print(f"Соединение установлено за {info['elapsed_ms']} мс")
        print(f"  команда:     {info['command']}")
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

        print("\nВызов git_log(limit=3)")
        result = await connection.session.call_tool("git_log", {"limit": 3})
        print(result_to_text(result))


if __name__ == "__main__":
    asyncio.run(main())
