import asyncio
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

DB_PATH = Path(__file__).parent / "history.db"

# Запасной заголовок диалога — начало первого вопроса: длинную реплику незачем
# тащить из базы целиком, в строку списка всё равно попадёт пара слов.
TITLE_LIMIT = 80

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS messages_session_idx ON messages(session_id, id);

-- Счёт за каждый запрос к модели: и за ход диалога, и за служебный запрос
-- заголовка. Токены и стоимость лежат рядом с перепиской, поэтому панель
-- восстанавливается вместе с лентой, а не живёт до перезагрузки страницы.
CREATE TABLE IF NOT EXISTS turns (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind              TEXT NOT NULL DEFAULT 'turn',
    prompt_tokens     INTEGER NOT NULL,
    cached_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    estimated_tokens  INTEGER NOT NULL,
    context_messages  INTEGER NOT NULL,
    cost_usd          REAL NOT NULL,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS turns_session_idx ON turns(session_id, id);
"""

TURN_FIELDS = (
    "kind",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "estimated_tokens",
    "context_messages",
    "cost_usd",
)


class Storage:
    """История диалогов в SQLite: агент живёт в процессе, память — в файле."""

    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # Соединение на операцию: вызовы уходят в отдельный поток, поэтому одно
        # соединение на всех делить не приходится — и блокировку заводить не нужно.
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    async def init(self) -> None:
        await asyncio.to_thread(self._init)

    def _init(self) -> None:
        connection = sqlite3.connect(self.path)
        try:
            # WAL живёт в самой базе: пишущий ход не блокирует чтение ленты.
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)
        finally:
            connection.close()

    async def create_session(self) -> str:
        session_id = uuid4().hex
        await asyncio.to_thread(self._create_session, session_id)
        return session_id

    def _create_session(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute("INSERT INTO sessions (id) VALUES (?)", (session_id,))

    async def session_exists(self, session_id: str) -> bool:
        return await asyncio.to_thread(self._session_exists, session_id)

    def _session_exists(self, session_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return row is not None

    async def set_title(self, session_id: str, title: str) -> None:
        await asyncio.to_thread(self._set_title, session_id, title)

    def _set_title(self, session_id: str, title: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (title, session_id),
            )

    async def list_sessions(self) -> list[dict[str, Any]]:
        """Все диалоги для боковой панели: заголовок — суть первого запроса."""
        return await asyncio.to_thread(self._list_sessions)

    def _list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.id,
                       count(m.id) AS size,
                       -- Заголовок пишет агент после первого хода; пока его нет
                       -- (старый диалог, неудачный запрос) строка живёт началом вопроса.
                       coalesce(s.title,
                                (SELECT substr(content, 1, ?) FROM messages
                                  WHERE session_id = s.id AND role = 'user'
                                  ORDER BY id LIMIT 1)) AS title,
                       -- У пустой сессии сообщений нет: без coalesce она осталась бы
                       -- без времени и уехала в конец списка сразу после создания.
                       coalesce(max(m.created_at), s.created_at) AS updated_at
                FROM sessions s LEFT JOIN messages m ON m.session_id = s.id
                GROUP BY s.id
                -- created_at с точностью до секунды не разводит ходы внутри одной
                -- секунды, порядок создания сессий добирается из rowid.
                ORDER BY updated_at DESC, s.rowid DESC
                """,
                (TITLE_LIMIT,),
            ).fetchall()
        return [
            {"id": session_id, "size": size, "title": title, "updated_at": updated_at}
            for session_id, size, title, updated_at in rows
        ]

    async def delete_session(self, session_id: str) -> None:
        await asyncio.to_thread(self._delete_session, session_id)

    def _delete_session(self, session_id: str) -> None:
        # Сообщения уносит ON DELETE CASCADE — внешние ключи включены в _connect.
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    async def load_all(self, session_id: str) -> list[dict[str, str]]:
        """Вся переписка целиком — из неё страница восстанавливает ленту."""
        return await asyncio.to_thread(self._load_all, session_id)

    def _load_all(self, session_id: str) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]

    async def count(self, session_id: str) -> int:
        return await asyncio.to_thread(self._count, session_id)

    def _count(self, session_id: str) -> int:
        with self._connect() as connection:
            (total,) = connection.execute(
                "SELECT count(*) FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return total

    async def save_turn(
        self,
        session_id: str,
        prompt: str,
        answer: str,
        metrics: dict[str, Any],
    ) -> None:
        """Ход пишется целиком: вопрос без ответа в истории не остаётся."""
        await asyncio.to_thread(self._save_turn, session_id, prompt, answer, metrics)

    def _save_turn(
        self,
        session_id: str,
        prompt: str,
        answer: str,
        metrics: dict[str, Any],
    ) -> None:
        # Переписка и счёт за неё — одна транзакция: не бывает хода без метрик.
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                [(session_id, "user", prompt), (session_id, "assistant", answer)],
            )
            self._insert_turn(connection, session_id, metrics)

    async def save_service_turn(self, session_id: str, metrics: dict[str, Any]) -> None:
        """Запрос без реплик в ленте — например заголовок диалога: токены свои."""
        await asyncio.to_thread(self._save_service_turn, session_id, metrics)

    def _save_service_turn(self, session_id: str, metrics: dict[str, Any]) -> None:
        with self._connect() as connection:
            self._insert_turn(connection, session_id, metrics)

    @staticmethod
    def _insert_turn(
        connection: sqlite3.Connection,
        session_id: str,
        metrics: dict[str, Any],
    ) -> None:
        columns = ", ".join(("session_id", *TURN_FIELDS))
        placeholders = ", ".join("?" * (len(TURN_FIELDS) + 1))
        connection.execute(
            f"INSERT INTO turns ({columns}) VALUES ({placeholders})",
            (session_id, *(metrics[field] for field in TURN_FIELDS)),
        )

    async def load_turns(self, session_id: str) -> list[dict[str, Any]]:
        """Все запросы сессии по порядку — из них страница рисует таблицу токенов."""
        return await asyncio.to_thread(self._load_turns, session_id)

    def _load_turns(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {', '.join(TURN_FIELDS)} FROM turns WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(zip(TURN_FIELDS, row)) for row in rows]
