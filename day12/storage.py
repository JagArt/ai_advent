import asyncio
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from memory import LONGTERM, WORKING
from profile import PROFILE, SCALE_KEYS
from seed import DEFAULT_PROFILE, PROFILES, SEED

DB_PATH = Path(__file__).parent / "history.db"

# Запасной заголовок диалога — начало первого вопроса: длинную реплику незачем
# тащить из базы целиком, в строку списка всё равно попадёт пара слов.
TITLE_LIMIT = 80

# Три уровня памяти — три разных места хранения, и это главное, что видно в схеме.
# Краткосрочная лежит в messages: реплики диалога, из которых агент берёт окно.
# Рабочая — в working, по строке на пункт ТЗ, с session_id: задача кончилась
# вместе с диалогом. Долговременная — в longterm, и session_id у неё нет вовсе:
# она старше любого диалога и переживает каждый из них.
#
# Профиль лежит в своих трёх таблицах и в схеме отличается от памяти сразу двумя
# вещами: он принадлежит человеку, а не диалогу, и половина его — не текст, а
# значение шкалы из фиксированного списка.
SCHEMA = """
-- Профиль: кто собеседник. Таблица идёт первой не для порядка — на неё ссылается
-- сессия, а профиль существует раньше любого диалога.
CREATE TABLE IF NOT EXISTS profiles (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    -- Порядок в селекторе: список профилей — не история, сортировать его по id
    -- значило бы менять порядок при переименовании.
    position   INTEGER NOT NULL DEFAULT 0
);

-- Шкалы предпочтений: по строке на шкалу, значение — слово из её списка. Пара
-- (профиль, шкала) первичный ключ, поэтому правка шкалы — это upsert, а не
-- накопление истории: у профиля одна длина ответа, а не пять.
CREATE TABLE IF NOT EXISTS profile_scales (
    profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    scale      TEXT NOT NULL,
    value      TEXT NOT NULL,
    PRIMARY KEY (profile_id, scale)
);

-- Свободные пункты профиля: то, что списком значений не описать. Устроены как
-- пункты памяти — раздел, формулировка, origin, — но принадлежат человеку.
CREATE TABLE IF NOT EXISTS profile_items (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    section    TEXT NOT NULL,
    text       TEXT NOT NULL,
    -- seed — было до начала общения, user — отправил пользователь кнопкой,
    -- agent — записал сам при включённом автосохранении.
    origin     TEXT NOT NULL,
    -- В каком диалоге это узнали. Внешнего ключа нет намеренно: удалённый диалог
    -- не должен уносить предпочтение, которое из него вынесли.
    source_id  TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS profile_items_idx ON profile_items(profile_id, id);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT,
    -- От чьего лица идёт этот диалог. Профиль не принадлежит диалогу, диалог
    -- только ссылается на него: один и тот же профиль ведёт сколько угодно задач.
    profile_id TEXT REFERENCES profiles(id),
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
        # То же и с профилем, только причина сильнее: один профиль ведёт несколько
        # диалогов, и правка предпочтения в одном из них должна дойти до всех.
        self.profile_version = 0

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
        """Профили и долговременная память заполняются до первого диалога, а не в нём.

        Кто собеседник, что команда решила и что она знает, агент не выясняет
        заново при каждом запуске: это уже есть, когда пользователь открывает
        страницу. Заливка идёт при создании базы и при сбросе — дальше и профилем,
        и памятью распоряжается пользователь.

        Два набора проверяются по отдельности: пустая память при заполненных
        профилях — это база, которую уже почистили руками, и восстанавливать в ней
        нужно только опустевшую половину.
        """
        # executescript выше закоммитился сам, как всякий DDL, а вот строки — нет:
        # без явного commit seed исчезал бы при каждом запуске.
        if not connection.execute("SELECT count(*) FROM profiles").fetchone()[0]:
            for position, preset in enumerate(PROFILES):
                connection.execute(
                    "INSERT INTO profiles (id, name, position) VALUES (?, ?, ?)",
                    (preset.id, preset.name, position),
                )
                connection.executemany(
                    "INSERT INTO profile_scales (profile_id, scale, value) VALUES (?, ?, ?)",
                    [(preset.id, scale, value) for scale, value in preset.scales.items()],
                )
                connection.executemany(
                    "INSERT INTO profile_items (profile_id, section, text, origin) VALUES (?, ?, ?, 'seed')",
                    [(preset.id, section, text) for section, text in preset.items],
                )

        if not connection.execute("SELECT count(*) FROM longterm").fetchone()[0]:
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
            # CASCADE унесёт реплики и ТЗ вместе с сессиями, шкалы и пункты — вместе
            # с профилями. Долговременную память чистим сами: у неё нет внешнего
            # ключа, и в этом как раз смысл.
            connection.execute("DELETE FROM sessions")
            connection.execute("DELETE FROM longterm")
            # Профили удаляются после сессий, иначе внешний ключ не даст: диалог,
            # который на них ссылается, к этому моменту должен быть уже удалён.
            connection.execute("DELETE FROM profiles")
            # Счётчики AUTOINCREMENT тоже к нулю: иначе seed получит id 10+
            # после пары прогонов, и «исходное» состояние этим выдаст себя.
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_sequence'"
            ).fetchone():
                connection.execute("DELETE FROM sqlite_sequence")
            self._plant(connection)
        self.longterm_version += 1
        self.profile_version += 1

    async def create_session(self, profile_id: str = DEFAULT_PROFILE) -> str:
        """Новый диалог всегда от чьего-то лица: профиль выбирается до первой реплики."""
        session_id = uuid4().hex
        await asyncio.to_thread(self._create_session, session_id, profile_id)
        return session_id

    def _create_session(self, session_id: str, profile_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions (id, profile_id) VALUES (?, ?)",
                (session_id, profile_id),
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
                       -- Размер ТЗ прямо в списке: по нему видно, в какой задаче
                       -- рабочая память уже собрана, а какая только началась.
                       (SELECT count(*) FROM working WHERE session_id = s.id) AS working,
                       -- От чьего лица шёл разговор: в панели один и тот же вопрос
                       -- от двух профилей иначе не отличить.
                       (SELECT name FROM profiles WHERE id = s.profile_id) AS profile,
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
            {
                "id": session_id,
                "size": size,
                "working": working,
                "profile": profile,
                "title": title,
                "updated_at": updated_at,
            }
            for session_id, size, working, profile, title, updated_at in rows
        ]

    async def delete_session(self, session_id: str) -> None:
        await asyncio.to_thread(self._delete_session, session_id)

    def _delete_session(self, session_id: str) -> None:
        # Реплики и рабочую память уносит ON DELETE CASCADE — внешние ключи
        # включены в _connect. Долговременная остаётся: она не принадлежит диалогу.
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    async def list_profiles(self) -> list[dict[str, Any]]:
        """Профили для селектора: только имена, без шкал и пунктов."""
        return await asyncio.to_thread(self._list_profiles)

    def _list_profiles(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, name FROM profiles ORDER BY position, id",
            ).fetchall()
        return [{"id": profile_id, "name": name} for profile_id, name in rows]

    async def load_profile(self, profile_id: str) -> dict[str, Any] | None:
        """Профиль целиком: имя, значения шкал и свободные пункты по разделам."""
        return await asyncio.to_thread(self._load_profile, profile_id)

    def _load_profile(self, profile_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, name FROM profiles WHERE id = ?",
                (profile_id,),
            ).fetchone()
            if row is None:
                return None

            scales = connection.execute(
                "SELECT scale, value FROM profile_scales WHERE profile_id = ?",
                (profile_id,),
            ).fetchall()
            items = connection.execute(
                f"SELECT {', '.join(ITEM_FIELDS)} FROM profile_items WHERE profile_id = ? ORDER BY id",
                (profile_id,),
            ).fetchall()

        return {
            "id": row[0],
            "name": row[1],
            # Шкала, которой в базе нет, не подставляется здесь: значение по
            # умолчанию знает модель профиля, а не таблица.
            "scales": {scale: value for scale, value in scales if scale in SCALE_KEYS},
            "items": [dict(zip(ITEM_FIELDS, item)) for item in items],
        }

    async def session_profile(self, session_id: str) -> str:
        """Профиль диалога. Пустой — значит база старше этой колонки: берём первый."""
        return await asyncio.to_thread(self._session_profile, session_id)

    def _session_profile(self, session_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT profile_id FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return (row[0] if row else None) or DEFAULT_PROFILE

    async def set_session_profile(self, session_id: str, profile_id: str) -> None:
        await asyncio.to_thread(self._set_session_profile, session_id, profile_id)

    def _set_session_profile(self, session_id: str, profile_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET profile_id = ? WHERE id = ?",
                (profile_id, session_id),
            )

    async def set_scale(self, profile_id: str, scale: str, value: str) -> None:
        """Правка шкалы: у профиля одно значение на шкалу, поэтому это upsert."""
        await asyncio.to_thread(self._set_scale, profile_id, scale, value)
        self.profile_version += 1

    def _set_scale(self, profile_id: str, scale: str, value: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO profile_scales (profile_id, scale, value) VALUES (?, ?, ?)
                ON CONFLICT (profile_id, scale) DO UPDATE SET value = excluded.value
                """,
                (profile_id, scale, value),
            )

    async def add_profile_item(
        self,
        profile_id: str,
        section: str,
        text: str,
        origin: str,
        source_id: str | None = None,
    ) -> int:
        item_id = await asyncio.to_thread(
            self._add_profile_item,
            profile_id,
            section,
            text,
            origin,
            source_id,
        )
        self.profile_version += 1
        return item_id

    def _add_profile_item(
        self,
        profile_id: str,
        section: str,
        text: str,
        origin: str,
        source_id: str | None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO profile_items (profile_id, section, text, origin, source_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (profile_id, section, text, origin, source_id),
            )
        return cursor.lastrowid or 0

    async def delete_profile_item(self, profile_id: str, item_id: int) -> bool:
        removed = await asyncio.to_thread(self._delete_profile_item, profile_id, item_id)
        if removed:
            self.profile_version += 1
        return removed

    def _delete_profile_item(self, profile_id: str, item_id: int) -> bool:
        # profile_id в условии не формальность: чужое предпочтение из этого диалога
        # удалить нельзя, даже если угадать его id.
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM profile_items WHERE id = ? AND profile_id = ?",
                (item_id, profile_id),
            )
        return cursor.rowcount > 0

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
        target: str,
        session_id: str,
        section: str,
        text: str,
        origin: str,
        profile_id: str = "",
    ) -> int:
        """Запись по выбранному адресу: сам выбор делает пользователь или агент.

        Адресов три, и session_id значит для каждого своё: рабочая память ему
        принадлежит, долговременная только помнит, где это узнали, а профилю он
        нужен ровно за этим же — сам пункт лежит у человека.
        """
        if target == LONGTERM:
            return await self.add_longterm(section, text, origin, session_id)
        if target == WORKING:
            return await self.add_working(session_id, section, text, origin)
        if target == PROFILE:
            return await self.add_profile_item(profile_id, section, text, origin, session_id)
        raise ValueError(f"Неизвестный адрес записи: {target}")
