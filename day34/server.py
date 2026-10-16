"""MCP-сервер песочницы: список, поиск, чтение, запись и diff файлов workspace.

Один сервер, stdio, поднимается подпроцессом веб-приложения. Писать в stdout
нельзя ничем, кроме протокола, поэтому логи — на WARNING.

Пять инструментов — вся память ассистента о файлах. Код в промпт содержимого
не кладёт: модель сама решает, что открыть и что переписать. Сервер
самостоятелен: `python day34/server.py` поднимает его для любого MCP-клиента.
"""

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

import workspace

RelPath = Annotated[
    str,
    Field(
        description="Относительный путь внутри workspace, например README.md или src/notes/api.py.",
        min_length=1,
    ),
]
OptionalPath = Annotated[
    str | None,
    Field(description="Ограничить поиск файлом или папкой внутри workspace. Без него — весь проект."),
]
Query = Annotated[
    str,
    Field(description="Что искать: подстрока или регулярное выражение, например create_note.", min_length=1),
]
UseRegex = Annotated[bool, Field(description="Если true — query это регулярное выражение.")]
SearchLimit = Annotated[int, Field(description="Сколько совпадений вернуть, от 1 до 40.", ge=1, le=40)]
ReadLimit = Annotated[int, Field(description="Потолок символов пронумерованного текста.", ge=200, le=20000)]
Body = Annotated[
    str,
    Field(
        description="Полный новый текст файла целиком, не фрагмент. Существующий файл перезаписывается.",
        min_length=1,
        max_length=workspace.MAX_BODY_CHARS,
    ),
]
DiffPath = Annotated[
    str | None,
    Field(description="Сверить один путь с эталоном. Без него — все отличия workspace от fixture."),
]
DiffLimit = Annotated[int, Field(description="Потолок символов патча.", ge=200, le=20000)]

mcp = MCPServer("notes files", version="34.0", log_level="WARNING")


def _run(action):
    try:
        return action()
    except workspace.WorkspaceError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def fs_list() -> dict[str, Any]:
    """Дерево текстовых файлов рабочей копии: пути и размеры. Содержимого нет."""
    return _run(workspace.tree)


@mcp.tool()
def fs_search(query: Query, path: OptionalPath = None, regex: UseRegex = False, limit: SearchLimit = 20) -> dict[str, Any]:
    """Ищет строку или regex по файлам workspace. Каждое совпадение — путь, номер строки и контекст."""
    return _run(lambda: workspace.search(query, path, regex, limit))


@mcp.tool()
def fs_read(path: RelPath, max_chars: ReadLimit = 8000) -> dict[str, Any]:
    """Читает текстовый файл с номерами строк. Большой файл обрезается по max_chars."""
    return _run(lambda: workspace.read(path, max_chars))


@mcp.tool()
def fs_write(path: RelPath, content: Body) -> dict[str, Any]:
    """Создаёт файл или полностью перезаписывает его. В ответе — размер и unified diff относительно эталона."""
    return _run(lambda: workspace.write(path, content))


@mcp.tool()
def fs_diff(path: DiffPath = None, max_chars: DiffLimit = 8000) -> dict[str, Any]:
    """Unified diff рабочей копии относительно эталона fixture. Без path — все изменённые файлы."""
    return _run(lambda: workspace.diff(path, max_chars))


if __name__ == "__main__":
    workspace.ensure()
    mcp.run()
