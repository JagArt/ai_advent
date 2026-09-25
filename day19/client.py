"""MCP-клиент: поднимает сервер подпроцессом, здоровается и вызывает инструменты.

Против day16 добавилось одно — подписка на прогресс. `run_pipeline` внутри одного
вызова делает три шага, и без уведомлений он молчал бы до самого конца. Кадр шага
приезжает в `message` уведомления строкой JSON: своего поля для данных у
progress-уведомления в протоколе нет.

CLI прогоняет цепочку целиком: печатает шаги по мере их выполнения и трассу в конце.
"""

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
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

# Кадр шага: то, что сервер положил в message уведомления.
StepHandler = Callable[[dict[str, Any]], Awaitable[None]]


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

    async def call(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        on_step: StepHandler | None = None,
    ) -> Any:
        """Вызов для кода, а не для модели: структурированный результат или исключение."""
        result = await self.session.call_tool(
            name,
            arguments or {},
            progress_callback=_progress(on_step) if on_step else None,
        )
        if result.is_error:
            raise RuntimeError(result_to_text(result))

        payload = result.structured_content
        if payload is None:
            return json.loads(result_to_text(result))
        # Список SDK заворачивает в {"result": [...]}: у structuredContent корень — объект.
        if set(payload) == {"result"}:
            return payload["result"]
        return payload


def _progress(on_step: StepHandler):
    """Уведомление прогресса в кадр шага. Чужой текст в message молча пропускается."""

    async def handler(progress: float, total: float | None, message: str | None) -> None:
        if not message:
            return
        try:
            frame = json.loads(message)
        except json.JSONDecodeError:
            return
        await on_step({"progress": progress, "total": total, **frame})

    return handler


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


QUERY = "как считаются токены и стоимость хода"


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
            args = ", ".join(name if name in required else f"{name}?" for name in params)
            print(f"\n  {tool['name']}({args})")
            print(f"    {tool['description'].splitlines()[0]}")

        print(f"\nВызов run_pipeline(query={QUERY!r})")

        async def on_step(frame: dict[str, Any]) -> None:
            if frame["status"] == "started":
                print(f"  {frame['position']}. {frame['tool']:<12} пошёл")
                return

            if frame["status"] == "failed":
                print(f"     {frame['tool']} не прошёл: {frame['error']}")
                return

            handle = frame["handle"]
            print(
                f"     готово за {frame['elapsed_ms']} мс:"
                f" артефакт #{handle['artifact_id']} ({handle['kind']}),"
                f" вход {handle.get('parent_id') or '—'}"
            )

        result = await connection.call(
            "run_pipeline", {"query": QUERY, "limit": 4}, on_step=on_step
        )

        print(f"\nПрогон #{result['run_id']}: {result['status']}, {result['elapsed_ms']} мс")
        for step in result["steps"]:
            artifacts = f"{step.get('artifact_in') or '—'} → {step.get('artifact_out') or '—'}"
            print(f"  {step['position']}. {step['tool']:<12} {artifacts:<10} {step['status']}")

        if result["file"]:
            print(f"\nФайл: {result['file']['path']}, {result['file']['bytes']} байт")
            print(f"  sha256: {result['file']['sha256']}")

        if result["error"]:
            print(f"\nЦепочка встала на {result['failed_at']}: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())
