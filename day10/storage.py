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

# Имя ветки, с которой начинается любой диалог: пока ответвлений нет, она и есть
# весь диалог, поэтому в интерфейсе полоса веток на ней не показывается.
ROOT_BRANCH = "основная"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT,
    -- Активная ветка: диалог открывается там же, где его оставили.
    branch_id  INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Ветка — участок диалога, продолжающий родителя с определённого места. Дерево
-- держится на двух полях: parent_id — от кого ответвились, forked_after — id
-- последней унаследованной реплики, то есть сам checkpoint.
CREATE TABLE IF NOT EXISTS branches (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    parent_id    INTEGER REFERENCES branches(id) ON DELETE CASCADE,
    forked_after INTEGER NOT NULL DEFAULT 0,
    name         TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS branches_session_idx ON branches(session_id, id);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    -- Реплика принадлежит ветке, в которой сказана: соседняя ветка её не видит.
    branch_id  INTEGER NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS messages_session_idx ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS messages_branch_idx ON messages(branch_id, id);

-- Счёт за каждый запрос к модели: и за ход диалога, и за служебные — заголовок
-- диалога, обновление картотеки. Токены и стоимость лежат рядом с перепиской,
-- поэтому панель восстанавливается вместе с лентой.
CREATE TABLE IF NOT EXISTS turns (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    branch_id         INTEGER NOT NULL,
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

-- Картотека фактов, по строке на факт. Ключ уникален внутри ветки: ветки расходятся
-- не только репликами, но и памятью, поэтому у каждой картотека своя.
CREATE TABLE IF NOT EXISTS facts (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    branch_id  INTEGER NOT NULL REFERENCES branches(id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    -- Порядок появления: по нему видно, что агент узнал раньше остального.
    position   INTEGER NOT NULL,
    PRIMARY KEY (branch_id, key)
);
"""

BRANCH_FIELDS = ("id", "parent_id", "forked_after", "name")

TURN_FIELDS = (
    "branch_id",
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
        # Диалог начинается одной веткой: без неё репликам некуда лечь, поэтому
        # корневая ветка создаётся вместе с сессией и сразу становится активной.
        with self._connect() as connection:
            connection.execute("INSERT INTO sessions (id) VALUES (?)", (session_id,))
            cursor = connection.execute(
                "INSERT INTO branches (session_id, name) VALUES (?, ?)",
                (session_id, ROOT_BRANCH),
            )
            connection.execute(
                "UPDATE sessions SET branch_id = ? WHERE id = ?",
                (cursor.lastrowid, session_id),
            )

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
                       -- Веток обычно одна: их число в панели показывает, где
                       -- диалог разошёлся, ещё до того, как его откроют.
                       (SELECT count(*) FROM branches WHERE session_id = s.id) AS branches,
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
            {"id": session_id, "size": size, "branches": branches, "title": title, "updated_at": updated_at}
            for session_id, size, branches, title, updated_at in rows
        ]

    async def delete_session(self, session_id: str) -> None:
        await asyncio.to_thread(self._delete_session, session_id)

    def _delete_session(self, session_id: str) -> None:
        # Реплики, ветки и картотеку уносит ON DELETE CASCADE — внешние ключи
        # включены в _connect.
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    async def active_branch(self, session_id: str) -> int:
        """Ветка, в которой продолжается диалог: агент поднимает историю из неё."""
        return await asyncio.to_thread(self._active_branch, session_id)

    def _active_branch(self, session_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT branch_id FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None or row[0] is None:
            raise LookupError(f"У сессии {session_id} нет активной ветки")
        return row[0]

    async def set_branch(self, session_id: str, branch_id: int) -> None:
        await asyncio.to_thread(self._set_branch, session_id, branch_id)

    def _set_branch(self, session_id: str, branch_id: int) -> None:
        with self._connect() as connection:
            if not self._branch_belongs(connection, session_id, branch_id):
                raise LookupError(f"Ветки {branch_id} нет в сессии {session_id}")
            connection.execute(
                "UPDATE sessions SET branch_id = ? WHERE id = ?",
                (branch_id, session_id),
            )

    async def create_branch(
        self,
        session_id: str,
        message_id: int,
        name: str,
    ) -> int:
        """Ветка от checkpoint: реплика, на которой ветвимся, задаёт и родителя."""
        return await asyncio.to_thread(self._create_branch, session_id, message_id, name)

    def _create_branch(self, session_id: str, message_id: int, name: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT branch_id FROM messages WHERE id = ? AND session_id = ?",
                (message_id, session_id),
            ).fetchone()
            if row is None:
                raise LookupError(f"Реплики {message_id} нет в сессии {session_id}")

            # Родитель — ветка самой реплики, а не активная: ветвиться можно и от
            # места, унаследованного текущей веткой от предка.
            cursor = connection.execute(
                "INSERT INTO branches (session_id, parent_id, forked_after, name) VALUES (?, ?, ?, ?)",
                (session_id, row[0], message_id, name),
            )
            branch_id = cursor.lastrowid or 0

            # Картотека родителя переезжает в ветку копией: дальше она меняется
            # независимо, и две ветки помнят разное об одном и том же разговоре.
            connection.execute(
                """
                INSERT INTO facts (session_id, branch_id, key, value, position)
                SELECT session_id, ?, key, value, position FROM facts WHERE branch_id = ?
                """,
                (branch_id, row[0]),
            )
        return branch_id

    async def list_branches(self, session_id: str) -> list[dict[str, Any]]:
        """Ветки диалога с числом видимых реплик: своих плюс унаследованных."""
        return await asyncio.to_thread(self._list_branches, session_id)

    def _list_branches(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            branches = self._branches(connection, session_id)
            # Пары id-ветка по всей сессии разом: веток единицы, и считать по ним
            # длину каждой в Python дешевле, чем ходить в базу за каждой.
            pairs = connection.execute(
                "SELECT id, branch_id FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchall()

        sizes = {}
        for branch_id in branches:
            chain = self._chain(branches, branch_id)
            sizes[branch_id] = sum(
                1
                for message_id, owner in pairs
                for segment, ceiling in chain
                if owner == segment and (ceiling is None or message_id <= ceiling)
            )

        return [{**branch, "size": sizes[branch["id"]]} for branch in branches.values()]

    async def load_messages(self, session_id: str, branch_id: int) -> list[dict[str, Any]]:
        """История ветки: свои реплики и то, что досталось от предков до checkpoint."""
        return await asyncio.to_thread(self._load_messages, session_id, branch_id)

    def _load_messages(self, session_id: str, branch_id: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            chain = self._chain(self._branches(connection, session_id), branch_id)
            if not chain:
                return []

            conditions = []
            values: list[Any] = []
            for segment, ceiling in chain:
                if ceiling is None:
                    conditions.append("branch_id = ?")
                    values.append(segment)
                else:
                    conditions.append("(branch_id = ? AND id <= ?)")
                    values.extend((segment, ceiling))

            # Порядок по id и есть порядок разговора: реплики ветки появились
            # позже checkpoint, а значит и позже всего унаследованного.
            rows = connection.execute(
                f"""
                SELECT id, role, content FROM messages
                 WHERE session_id = ? AND ({' OR '.join(conditions)}) ORDER BY id
                """,
                (session_id, *values),
            ).fetchall()

        # id реплики нужен и агенту, и странице: по нему ставится checkpoint.
        return [
            {"id": message_id, "role": role, "content": content}
            for message_id, role, content in rows
        ]

    @staticmethod
    def _branches(connection: sqlite3.Connection, session_id: str) -> dict[int, dict[str, Any]]:
        rows = connection.execute(
            f"SELECT {', '.join(BRANCH_FIELDS)} FROM branches WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return {row[0]: dict(zip(BRANCH_FIELDS, row)) for row in rows}

    @staticmethod
    def _chain(branches: dict[int, dict[str, Any]], branch_id: int) -> list[tuple[int, int | None]]:
        """Цепочка от корня до ветки: у предков — граница, у самой ветки её нет."""
        chain: list[tuple[int, int | None]] = []
        current: int | None = branch_id
        ceiling: int | None = None

        while current is not None and current in branches:
            chain.append((current, ceiling))
            branch = branches[current]
            # Граница для родителя — checkpoint потомка: дальше него родительские
            # реплики принадлежат уже другой линии разговора.
            current, ceiling = branch["parent_id"], branch["forked_after"]

        chain.reverse()
        return chain

    @staticmethod
    def _branch_belongs(connection: sqlite3.Connection, session_id: str, branch_id: int) -> bool:
        row = connection.execute(
            "SELECT 1 FROM branches WHERE id = ? AND session_id = ?",
            (branch_id, session_id),
        ).fetchone()
        return row is not None

    async def save_turn(
        self,
        session_id: str,
        branch_id: int,
        prompt: str,
        answer: str,
        metrics: dict[str, Any],
    ) -> tuple[int, int]:
        """Ход пишется целиком: вопрос без ответа в истории не остаётся."""
        return await asyncio.to_thread(
            self._save_turn,
            session_id,
            branch_id,
            prompt,
            answer,
            metrics,
        )

    def _save_turn(
        self,
        session_id: str,
        branch_id: int,
        prompt: str,
        answer: str,
        metrics: dict[str, Any],
    ) -> tuple[int, int]:
        # Переписка и счёт за неё — одна транзакция: не бывает хода без метрик.
        with self._connect() as connection:
            ids = []
            for role, content in (("user", prompt), ("assistant", answer)):
                cursor = connection.execute(
                    """
                    INSERT INTO messages (session_id, branch_id, role, content)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, branch_id, role, content),
                )
                # id реплик возвращаются агенту: по ним страница ставит checkpoint,
                # а агент не перечитывает базу после каждого хода.
                ids.append(cursor.lastrowid or 0)
            self._insert_turn(connection, session_id, metrics)
        return ids[0], ids[1]

    async def save_service_turn(self, session_id: str, metrics: dict[str, Any]) -> None:
        """Запрос без реплик в ленте — заголовок или картотека: токены свои."""
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

    async def load_facts(self, branch_id: int) -> list[tuple[str, str]]:
        """Картотека ветки в порядке появления фактов."""
        return await asyncio.to_thread(self._load_facts, branch_id)

    def _load_facts(self, branch_id: int) -> list[tuple[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT key, value FROM facts WHERE branch_id = ? ORDER BY position",
                (branch_id,),
            ).fetchall()
        return [(key, value) for key, value in rows]

    async def save_facts(
        self,
        session_id: str,
        branch_id: int,
        items: tuple[tuple[str, str], ...],
        metrics: dict[str, Any],
    ) -> None:
        """Картотека и счёт за её обновление: разбор реплики тоже платный."""
        await asyncio.to_thread(self._save_facts, session_id, branch_id, items, metrics)

    def _save_facts(
        self,
        session_id: str,
        branch_id: int,
        items: tuple[tuple[str, str], ...],
        metrics: dict[str, Any],
    ) -> None:
        # Картотека перезаписывается целиком: вычеркнутый факт должен исчезнуть,
        # а порядок строк — совпасть с тем, что агент держит в памяти.
        with self._connect() as connection:
            connection.execute("DELETE FROM facts WHERE branch_id = ?", (branch_id,))
            connection.executemany(
                "INSERT INTO facts (session_id, branch_id, key, value, position) VALUES (?, ?, ?, ?, ?)",
                [
                    (session_id, branch_id, key, value, position)
                    for position, (key, value) in enumerate(items)
                ],
            )
            self._insert_turn(connection, session_id, metrics)
