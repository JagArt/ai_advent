"""MCP-сервер поддержки Планёрки: CRM с пользователями и тикетами и база знаний.

Один сервер, stdio, поднимается подпроцессом веб-приложения и живёт, пока
живёт оно. Писать в stdout нельзя ничем, кроме протокола, поэтому логи — на
WARNING.

Шесть инструментов — вся память ассистента. Код в промпт данных не кладёт:
тикет, карточку пользователя и статьи модель берёт только отсюда, и в ленте
хода видно каждое обращение. Сервер самостоятелен: `python day33/server.py`
поднимает его для любого MCP-клиента.
"""

from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

import crm
import kb

TicketId = Annotated[str, Field(description="Идентификатор тикета вида «T-1007».", min_length=3)]
UserRef = Annotated[str, Field(description="Идентификатор пользователя вида «u-03» или его почта.", min_length=3)]
UserFilter = Annotated[
    str | None,
    Field(description="Пользователь — идентификатор вида «u-03» или почта. Без него — тикеты всех пользователей."),
]
StatusFilter = Annotated[
    str | None,
    Field(description="Статус тикета: open, pending или closed. Без него — любые.", pattern="^(open|pending|closed)$"),
]
ListLimit = Annotated[int, Field(description="Сколько тикетов вернуть, от 1 до 20.", ge=1, le=crm.MAX_LIMIT)]
TicketQuery = Annotated[
    str,
    Field(
        description=(
            "Слова, по которым искать похожие обращения: тема, симптом, название функции, например "
            "«не могу войти пароль» или «429 лимит API»."
        ),
        min_length=2,
    ),
]
Exclude = Annotated[str | None, Field(description="Тикет, который исключить из выдачи, — обычно текущий.")]
SearchLimit = Annotated[int, Field(description="Сколько результатов вернуть, от 1 до 10.", ge=1, le=10)]
KbQuery = Annotated[
    str,
    Field(
        description=(
            "Запрос к базе знаний продукта на естественном языке, например «неверный пароль блокировка» "
            "или «домен не разрешён SSO». Ищутся разделы статей FAQ и документации."
        ),
        min_length=2,
    ),
]
Slug = Annotated[str, Field(description="Слаг статьи из выдачи kb_search, например «blokirovka-vhoda».")]
Heading = Annotated[
    str | None,
    Field(description="Заголовок раздела из выдачи kb_search, как он там написан. Без него — статья целиком."),
]

mcp = MCPServer("Планёрка support", version="33.0", log_level="WARNING")


# --- CRM ---------------------------------------------------------------------


@mcp.tool()
def crm_get_ticket(ticket_id: TicketId) -> dict[str, Any]:
    """Тикет из CRM целиком: тема, текст обращения, переписка, статус, приоритет, теги, решение у закрытых.

    С этого начинается разбор любого обращения: без текста тикета неизвестно,
    на что жалуется пользователь.
    """
    try:
        return crm.get_ticket(ticket_id)
    except crm.CrmError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def crm_get_user(user: UserRef) -> dict[str, Any]:
    """Карточка пользователя из CRM: тариф, статус аккаунта, подтверждена ли почта, 2FA, счётчик неудачных входов и блокировка, команда и её SSO, платформа и версия приложения, оплата и интеграции.

    Здесь лежит контекст, которого нет в тексте тикета, — часто именно он
    объясняет причину обращения.
    """
    try:
        return crm.get_user(user)
    except crm.CrmError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def crm_list_tickets(user: UserFilter = None, status: StatusFilter = None, limit: ListLimit = 10) -> dict[str, Any]:
    """Список тикетов, новые первыми, с фильтрами по пользователю и статусу — история обращений клиента или очередь открытых."""
    try:
        return crm.list_tickets(user, status, limit)
    except crm.CrmError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def crm_search_tickets(query: TicketQuery, exclude: Exclude = None, limit: SearchLimit = 5) -> dict[str, Any]:
    """Похожие обращения других пользователей по словам: тема, текст, теги и решение. У закрытых есть поле resolution — чем кончилось."""
    try:
        return crm.search_tickets(query, exclude, limit)
    except crm.CrmError as exc:
        raise ToolError(str(exc)) from exc


# --- база знаний -------------------------------------------------------------


@mcp.tool()
def kb_search(query: KbQuery, limit: SearchLimit = 5) -> dict[str, Any]:
    """Ищет разделы базы знаний Планёрки — FAQ и документацию продукта — по словам запроса.

    Возвращает разделы с текстом целиком, пронумерованные. О том, как устроен
    продукт и что делать в типовой ситуации, отвечай только по ним.
    """
    try:
        return kb.search(query, limit)
    except kb.KbError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool()
def kb_read(slug: Slug, heading: Heading = None) -> dict[str, Any]:
    """Статья базы знаний целиком или один её раздел — по слагу и заголовку из выдачи kb_search."""
    try:
        return kb.read(slug, heading)
    except kb.KbError as exc:
        raise ToolError(str(exc)) from exc


if __name__ == "__main__":
    mcp.run()
