import asyncio
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from memory import LONGTERM, WORKING
from seed import SEED

DB_PATH = Path(__file__).parent / "history.db"

# Запасной заголовок диалога — начало первого вопроса: длинную реплику незачем
# тащить из базы целиком, в строку списка всё равно попадёт пара слов.
TITLE_LIMIT = 80

# Три уровня памяти — три разных места хранения, и это главное, что видно в схеме.
# Краткосрочная лежит в messages: реплики диалога, из которых агент берёт окно.
# Рабочая — в working, по строке на пункт ТЗ, с session_id: задача кончилась
# вместе с диалогом. Долговременная — в longterm, и session_id у неё нет вовсе:
# она старше любого диалога и переживает каждый из них.
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Краткосрочная память: реплики текущего диалога. Окно последних N из них уходит
-- в модель дословно, остальные остаются здесь как лента для чтения.
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS messages_session_idx ON messages(session_id, id);

-- Долговременная память: профиль, решения, знания. Без session_id — она общая
-- для всех диалогов и существует до того, как начался первый.
CREATE TABLE IF NOT EXISTS longterm (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    section    TEXT NOT NULL,
    text       TEXT NOT NULL,
    -- Откуда пункт взялся: seed — было до начала общения, agent — записал сам
    -- при включённом автосохранении, user — отправил пользователь кнопкой.
    origin     TEXT NOT NULL,
    -- В каком диалоге это узнали. У seed источника нет, и внешнего ключа тоже:
    -- удалённый диалог не должен уносить знание, которое из него вынесли.
    source_id  TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Рабочая память: ТЗ текущей задачи. Принадлежит сессии и уходит вместе с ней.
CREATE TABLE IF NOT EXISTS working (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    section    TEXT NOT NULL,
    text       TEXT NOT NULL,
    origin     TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS working_session_idx ON working(session_id, id);
"""

ITEM_FIELDS = ("id", "section", "text", "origin")


class Storage:
    """Три уровня памяти в SQLite: агент живёт в процессе, память — в файле."""

    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = path
        # Долговременная память общая, а агенты живут в процессе по одному на
        # диалог и держат её блок у себя. Счётчик изменений — способ им об этом
        # узнать: записали в одном диалоге, в остальных блок устарел.
        self.longterm_version = 0

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
            self._plant(connection)
        finally:
            connection.close()

    @staticmethod
    def _plant(connection: sqlite3.Connection) -> None:
        """Долговременная память заполняется до первого диалога, а не в нём.

        Профиль, решения и знания агент не выясняет заново при каждом запуске:
        они уже есть, когда пользователь открывает страницу. Заливка идёт при
        создании базы и при сбросе — дальше памятью распоряжается пользователь.
        """
        planted = connection.execute("SELECT count(*) FROM longterm").fetchone()[0]
        if planted:
            return

        # executescript выше закоммитился сам, как всякий DDL, а вот строки — нет:
        # без явного commit долговременная память исчезала бы при каждом запуске.
        connection.executemany(
            "INSERT INTO longterm (section, text, origin) VALUES (?, ?, 'seed')",
            SEED,
        )
        connection.commit()

    async def reset(self) -> None:
        """Вернуть базу к состоянию после первого запуска: seed и ни одного диалога."""
        await asyncio.to_thread(self._reset)

    def _reset(self) -> None:
        with self._connect() as connection:
            # CASCADE унесёт реплики и ТЗ вместе с сессиями. Долговременную
            # чистим сами: у неё нет внешнего ключа, и в этом как раз смысл.
            connection.execute("DELETE FROM sessions")
            connection.execute("DELETE FROM longterm")
            # Счётчики AUTOINCREMENT тоже к нулю: иначе seed получит id 10+
            # после пары прогонов, и «исходное» состояние этим выдаст себя.
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_sequence'"
            ).fetchone():
                connection.execute("DELETE FROM sqlite_sequence")
            self._plant(connection)
        self.longterm_version += 1

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
                       (SELECT count(*) FROM messages WHERE session_id = s.id) AS size,
                       -- Размер ТЗ прямо в списке: по нему видно, в какой задаче
                       -- рабочая память уже собрана, а какая только началась.
                       (SELECT count(*) FROM working WHERE session_id = s.id) AS working,
                       -- Заголовок пишет агент после первого хода; пока его нет
                       -- (старый диалог, неудачный запрос) строка живёт началом вопроса.
                       coalesce(s.title,
                                (SELECT substr(content, 1, ?) FROM messages
                                  WHERE session_id = s.id AND role = 'user'
                                  ORDER BY id LIMIT 1)) AS title,
                       -- У пустой сессии сообщений нет: без coalesce она осталась бы
                       -- без времени и уехала в конец списка сразу после создания.
                       coalesce((SELECT max(created_at) FROM messages WHERE session_id = s.id),
                                s.created_at) AS updated_at
                FROM sessions s
                -- created_at с точностью до секунды не разводит ходы внутри одной
                -- секунды, порядок создания сессий добирается из rowid.
                ORDER BY updated_at DESC, s.rowid DESC
                """,
                (TITLE_LIMIT,),
            ).fetchall()
        return [
            {"id": session_id, "size": size, "working": working, "title": title, "updated_at": updated_at}
            for session_id, size, working, title, updated_at in rows
        ]

    async def delete_session(self, session_id: str) -> None:
        await asyncio.to_thread(self._delete_session, session_id)

    def _delete_session(self, session_id: str) -> None:
        # Реплики и рабочую память уносит ON DELETE CASCADE — внешние ключи
        # включены в _connect. Долговременная остаётся: она не принадлежит диалогу.
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    async def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Краткосрочная память целиком: лента диалога в порядке разговора."""
        return await asyncio.to_thread(self._load_messages, session_id)

    def _load_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, role, content FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [
            {"id": message_id, "role": role, "content": content}
            for message_id, role, content in rows
        ]

    async def save_turn(self, session_id: str, prompt: str, answer: str) -> tuple[int, int]:
        """Ход пишется целиком: вопрос без ответа в истории не остаётся."""
        return await asyncio.to_thread(self._save_turn, session_id, prompt, answer)

    def _save_turn(self, session_id: str, prompt: str, answer: str) -> tuple[int, int]:
        # Обе реплики — одна транзакция: половины хода в ленте не бывает.
        with self._connect() as connection:
            ids = []
            for role, content in (("user", prompt), ("assistant", answer)):
                cursor = connection.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                    (session_id, role, content),
                )
                ids.append(cursor.lastrowid or 0)
        return ids[0], ids[1]

    async def load_longterm(self) -> list[dict[str, Any]]:
        """Долговременная память: одна на все диалоги, читается без session_id."""
        return await asyncio.to_thread(self._load_longterm)

    def _load_longterm(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {', '.join(ITEM_FIELDS)} FROM longterm ORDER BY id",
            ).fetchall()
        return [dict(zip(ITEM_FIELDS, row)) for row in rows]

    async def add_longterm(
        self,
        section: str,
        text: str,
        origin: str,
        source_id: str | None = None,
    ) -> int:
        item_id = await asyncio.to_thread(self._add_longterm, section, text, origin, source_id)
        self.longterm_version += 1
        return item_id

    def _add_longterm(self, section: str, text: str, origin: str, source_id: str | None) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO longterm (section, text, origin, source_id) VALUES (?, ?, ?, ?)",
                (section, text, origin, source_id),
            )
        return cursor.lastrowid or 0

    async def delete_longterm(self, item_id: int) -> bool:
        removed = await asyncio.to_thread(self._delete_longterm, item_id)
        if removed:
            self.longterm_version += 1
        return removed

    def _delete_longterm(self, item_id: int) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM longterm WHERE id = ?", (item_id,))
        return cursor.rowcount > 0

    async def load_working(self, session_id: str) -> list[dict[str, Any]]:
        """Рабочая память: ТЗ одной задачи, читается только вместе с её сессией."""
        return await asyncio.to_thread(self._load_working, session_id)

    def _load_working(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {', '.join(ITEM_FIELDS)} FROM working WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(zip(ITEM_FIELDS, row)) for row in rows]

    async def add_working(self, session_id: str, section: str, text: str, origin: str) -> int:
        return await asyncio.to_thread(self._add_working, session_id, section, text, origin)

    def _add_working(self, session_id: str, section: str, text: str, origin: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO working (session_id, section, text, origin) VALUES (?, ?, ?, ?)",
                (session_id, section, text, origin),
            )
        return cursor.lastrowid or 0

    async def delete_working(self, session_id: str, item_id: int) -> bool:
        return await asyncio.to_thread(self._delete_working, session_id, item_id)

    def _delete_working(self, session_id: str, item_id: int) -> bool:
        # session_id в условии не формальность: пункт чужого ТЗ из этого диалога
        # удалить нельзя, даже если угадать его id.
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM working WHERE id = ? AND session_id = ?",
                (item_id, session_id),
            )
        return cursor.rowcount > 0

    async def add(
        self,
        tier: str,
        session_id: str,
        section: str,
        text: str,
        origin: str,
    ) -> int:
        """Запись по выбранному уровню: сам выбор делает пользователь или агент."""
        if tier == LONGTERM:
            return await self.add_longterm(section, text, origin, session_id)
        if tier == WORKING:
            return await self.add_working(session_id, section, text, origin)
        raise ValueError(f"Неизвестный уровень памяти: {tier}")
