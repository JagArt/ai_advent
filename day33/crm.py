"""CRM поддержки Планёрки: пользователи и тикеты из `crm.json`.

Настоящей CRM за этим днём нет — есть JSON, который играет её роль: карточки
пользователей с тем, что обычно знает о клиенте система (тариф, статус, 2FA,
команда, устройство), и тикеты с перепиской. Файл перечитывается при каждом
обращении по mtime, поэтому его можно править, не перезапуская сервер.

Поиск похожих обращений — FTS5 в памяти по теме, тексту, тегам и решению:
оператору важно знать, что такой же вопрос уже решали, и чем кончилось.
"""

import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

CRM_PATH = Path(__file__).resolve().parent / "crm.json"
WORD = re.compile(r"[^\W_]{2,}", re.UNICODE)
MAX_QUERY_TOKENS = 12
MAX_LIMIT = 20

STATUSES = ("open", "pending", "closed")
STATUS_TITLES = {"open": "открыт", "pending": "ждёт ответа клиента", "closed": "закрыт"}
PRIORITY_TITLES = {"low": "низкий", "normal": "обычный", "high": "высокий"}
USER_STATUS_TITLES = {"active": "активен", "blocked": "заблокирован", "unconfirmed": "почта не подтверждена"}


class CrmError(ValueError):
    """Отказ с причиной: такого пользователя нет, тикет не найден, фильтр непонятен."""


_cache: tuple[float, dict[str, Any]] | None = None


def load() -> dict[str, Any]:
    """Данные CRM; перечитываются, когда файл меняется."""
    global _cache
    try:
        mtime = CRM_PATH.stat().st_mtime
    except FileNotFoundError as exc:
        raise CrmError(f"Файла CRM {CRM_PATH.name} нет рядом с сервером.") from exc
    if _cache is None or _cache[0] != mtime:
        data = json.loads(CRM_PATH.read_text(encoding="utf-8"))
        _cache = (mtime, data)
    return _cache[1]


def stats() -> dict[str, Any]:
    data = load()
    by_status = {status: 0 for status in STATUSES}
    for ticket in data["tickets"]:
        by_status[ticket["status"]] = by_status.get(ticket["status"], 0) + 1
    return {"users": len(data["users"]), "tickets": len(data["tickets"]), "by_status": by_status, "file": CRM_PATH.name}


# --- пользователи ------------------------------------------------------------


def _user_line(user: dict[str, Any]) -> str:
    """Карточка одной строкой по-русски — то, что CRM показывает оператору в шапке.

    Ничего не диагностирует: перечисляет поля словами, чтобы модель читала
    «почта не подтверждена, зарегистрирован 2026-10-13», а не `"status":
    "unconfirmed"` среди двадцати ключей JSON.
    """
    parts = [f"{user['name']}, тариф {user['plan'].capitalize()}"]
    parts.append(f"статус аккаунта: {USER_STATUS_TITLES.get(user['status'], user['status'])}")
    if user["status"] == "unconfirmed":
        parts.append(f"зарегистрирован {user['registered_at']}, почта не подтверждена")
    parts.append("2FA включена" if user["two_factor"] else "2FA выключена")
    failed = f"неудачных входов {user['failed_logins']}"
    if user.get("locked_until"):
        failed += f", заблокирован до {user['locked_until']}"
    parts.append(failed)
    if user.get("app_version"):
        parts.append(f"платформа {user['platform']}, приложение версии {user['app_version']}")
    else:
        parts.append(f"платформа {user['platform']}")
    team = user.get("team")
    if team:
        sso = f"SSO по домену {team['sso_domain']}" + (" (только SSO)" if team.get("sso_only") else "") if team.get("sso_domain") else "без SSO"
        parts.append(f"команда «{team['name']}», роль {team['role']}, {sso}; почта пользователя {user['email']}")
    billing = user.get("billing")
    if billing:
        parts.append(
            f"был на тарифе {billing['previous_plan'].capitalize()}, понижен {billing['downgraded_at']} "
            f"после {len(billing['last_payment_attempts'])} отклонённых платежей, проектов в архиве {billing['archived_projects']}"
        )
    return "; ".join(parts) + "."


def _user_summary(user: dict[str, Any]) -> dict[str, Any]:
    """Карточка пользователя с расшифровками: модели проще читать «заблокирован», чем `blocked`."""
    summary = {"summary": _user_line(user), **user}
    summary["status_title"] = USER_STATUS_TITLES.get(user["status"], user["status"])
    summary["plan_title"] = user["plan"].capitalize()
    return summary


def get_user(user: str) -> dict[str, Any]:
    """Пользователь по идентификатору вида `u-03` или по почте."""
    key = user.strip().lower()
    if not key:
        raise CrmError("Пустой идентификатор пользователя.")
    for item in load()["users"]:
        if item["id"].lower() == key or item["email"].lower() == key:
            return _user_summary(item)
    raise CrmError(f"Пользователя {user!r} в CRM нет. Ожидается идентификатор вида «u-03» или почта.")


# --- тикеты ------------------------------------------------------------------


def _ticket_summary(ticket: dict[str, Any], *, full: bool) -> dict[str, Any]:
    summary = {
        "id": ticket["id"],
        "user_id": ticket["user_id"],
        "subject": ticket["subject"],
        "status": ticket["status"],
        "status_title": STATUS_TITLES.get(ticket["status"], ticket["status"]),
        "priority": ticket["priority"],
        "priority_title": PRIORITY_TITLES.get(ticket["priority"], ticket["priority"]),
        "channel": ticket["channel"],
        "created_at": ticket["created_at"],
        "updated_at": ticket["updated_at"],
        "tags": ticket["tags"],
    }
    if full:
        summary["body"] = ticket["body"]
        summary["messages"] = ticket["messages"]
        summary["resolution"] = ticket["resolution"]
    else:
        summary["resolution"] = ticket["resolution"]
    return summary


def get_ticket(ticket_id: str) -> dict[str, Any]:
    """Тикет целиком: тема, текст, переписка, статус, решение у закрытых."""
    key = ticket_id.strip().upper()
    for item in load()["tickets"]:
        if item["id"].upper() == key:
            return _ticket_summary(item, full=True)
    raise CrmError(f"Тикета {ticket_id!r} в CRM нет. Ожидается идентификатор вида «T-1007».")


def list_tickets(user: str | None = None, status: str | None = None, limit: int = 20) -> dict[str, Any]:
    """Список тикетов, новые первыми; фильтры по пользователю и статусу."""
    if status is not None and status not in STATUSES:
        raise CrmError(f"Статуса {status!r} нет. Есть: {', '.join(STATUSES)}.")
    user_id = get_user(user)["id"] if user else None
    limit = max(1, min(int(limit), MAX_LIMIT))

    tickets = sorted(load()["tickets"], key=lambda item: item["created_at"], reverse=True)
    found = [
        item
        for item in tickets
        if (user_id is None or item["user_id"] == user_id) and (status is None or item["status"] == status)
    ]
    return {
        "user": user_id,
        "status": status,
        "total": len(found),
        "tickets": [_ticket_summary(item, full=False) for item in found[:limit]],
        "truncated": len(found) > limit,
    }


def _match(query: str) -> str:
    tokens = WORD.findall(query.lower())[:MAX_QUERY_TOKENS]
    if not tokens:
        raise CrmError(f"В запросе {query!r} нет слов для поиска.")
    return " OR ".join(f'"{token}"*' for token in tokens)


def search_tickets(query: str, exclude: str | None = None, limit: int = 5) -> dict[str, Any]:
    """Похожие обращения по словам: тема, текст, теги, решение. Закрытые с решением — самое ценное."""
    data = load()
    limit = max(1, min(int(limit), MAX_LIMIT))
    skip = exclude.strip().upper() if exclude else None

    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute(
            "CREATE VIRTUAL TABLE t USING fts5(id UNINDEXED, subject, body, tags, resolution,"
            " tokenize = 'unicode61 remove_diacritics 2')"
        )
        conn.executemany(
            "INSERT INTO t VALUES (?, ?, ?, ?, ?)",
            [
                (
                    item["id"],
                    item["subject"],
                    " ".join(message["text"] for message in item["messages"]),
                    " ".join(item["tags"]),
                    item["resolution"] or "",
                )
                for item in data["tickets"]
                if item["id"].upper() != skip
            ],
        )
        rows = conn.execute(
            "SELECT id, bm25(t, 0.0, 3.0, 1.0, 2.0, 2.0) AS rank FROM t WHERE t MATCH ? ORDER BY rank LIMIT ?",
            (_match(query), limit),
        ).fetchall()

    if not rows:
        raise CrmError(f"Похожих обращений по запросу {query!r} не нашлось.")

    by_id = {item["id"]: item for item in data["tickets"]}
    hits = []
    for ticket_id, rank in rows:
        ticket = by_id[ticket_id]
        hit = _ticket_summary(ticket, full=False)
        hit["score"] = round(-rank, 3)
        hit["excerpt"] = ticket["body"][:240] + ("…" if len(ticket["body"]) > 240 else "")
        hits.append(hit)
    return {"query": query, "exclude": skip, "count": len(hits), "tickets": hits}


if __name__ == "__main__":
    import sys

    print(json.dumps(stats(), ensure_ascii=False))
    if len(sys.argv) > 1:
        print(json.dumps(search_tickets(" ".join(sys.argv[1:])), ensure_ascii=False, indent=2))
