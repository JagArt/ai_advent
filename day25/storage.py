"""Диалог, цель и источники в SQLite: чат переживает перезапуск процесса.

Схема продолжает [day7](../day7/storage.py) и [day11](../day11/storage.py), но
хранит на два вида записей больше, и оба появились из требований задания.

**Цель хранится журналом, а не полем.** Напрашивалось положить её колонкой в
`sessions` и переписывать на месте. Отвергнуто: задание просит проверить, что
ассистент не теряет цель, а проверить это по текущему значению нельзя — видно
только то, что стоит сейчас, а не то, сколько раз оно менялось и почему.
Поэтому каждая постановка цели — строка в `goals` с ходом и обоснованием, а
действующая цель — последняя из них.

**Источники хранятся по утверждениям.** В day24 ответ жил в памяти процесса:
вопрос — ответ — конец. В чате ответы складываются в историю, и вопрос «а был ли
у этого хода источник» задаётся спустя десять реплик. Хранить для этого текст
ответа бессмысленно — в нём номера выдержек, которые без контекста того хода
ничего не значат. Поэтому строка `turn_sources` держит сразу путь к файлу,
цитату и вердикт сверки: ход можно открыть и проверить, не пересобирая контекст.

У `facts` нет удаления. Пункт, снятый моделью, остаётся в таблице с `alive = 0`:
отменённое ограничение — это тоже история задачи, и в замере оно нужно наравне с
действующими.
"""

import asyncio
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import state
from state import Fact, Goal, TaskState

DB_PATH = Path(__file__).resolve().parent / "chat.db"

# Запасной заголовок диалога — начало первой реплики.
TITLE_LIMIT = 80

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT,
    -- Режим памяти задан на диалог, а не на ход: сменить его посередине значило бы
    -- получить историю, половина которой собрана по другим правилам.
    mode       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Реплики диалога. `turn` нумерует ходы, а не сообщения: вопрос и ответ одного
-- хода делят номер, и по нему строка связывается с turns, goals и facts.
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn       INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS messages_session_idx ON messages(session_id, id);

-- Журнал цели: по строке на каждую постановку. Действующая цель — последняя.
CREATE TABLE IF NOT EXISTS goals (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn       INTEGER NOT NULL,
    text       TEXT NOT NULL,
    -- Чем обоснована смена. У первой постановки пусто: до неё цели не было.
    why        TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS goals_session_idx ON goals(session_id, id);

-- Пункты памяти задачи. `fact_id` — номер внутри диалога: он печатается в промпт
-- трекера и по нему приходит `drop`, поэтому сквозной автоинкремент тут не годится.
CREATE TABLE IF NOT EXISTS facts (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    fact_id    INTEGER NOT NULL,
    section    TEXT NOT NULL,
    text       TEXT NOT NULL,
    turn       INTEGER NOT NULL,
    -- Снятый пункт не удаляется: отменённое ограничение — тоже история задачи.
    alive      INTEGER NOT NULL DEFAULT 1,
    dropped_at INTEGER,
    PRIMARY KEY (session_id, fact_id)
);

-- Ход целиком: что спросили, по чему искали, чем ответили и чего это стоило.
CREATE TABLE IF NOT EXISTS turns (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn              INTEGER NOT NULL,
    mode              TEXT NOT NULL,
    question          TEXT NOT NULL,
    standalone        TEXT NOT NULL,
    answer            TEXT NOT NULL,
    refused           INTEGER NOT NULL DEFAULT 0,
    gate_reason       TEXT NOT NULL DEFAULT '',
    clarify           TEXT NOT NULL DEFAULT '',
    kept_goal         TEXT NOT NULL DEFAULT '',
    claims            INTEGER NOT NULL DEFAULT 0,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    stage_tokens      INTEGER NOT NULL DEFAULT 0,
    memory_tokens     INTEGER NOT NULL DEFAULT 0,
    seconds           REAL    NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS turns_session_idx ON turns(session_id, turn);

-- Источники хода по утверждениям: путь, цитата и вердикт сверки лежат рядом,
-- чтобы ход можно было открыть, не пересобирая его контекст.
CREATE TABLE IF NOT EXISTS turn_sources (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn       INTEGER NOT NULL,
    position   INTEGER NOT NULL,
    claim      TEXT NOT NULL,
    number     INTEGER,
    chunk_id   INTEGER,
    path       TEXT NOT NULL DEFAULT '',
    section    TEXT NOT NULL DEFAULT '',
    quote      TEXT NOT NULL DEFAULT '',
    verdict    TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS turn_sources_idx ON turn_sources(session_id, turn);
"""


class Storage:
    """Чат в файле. Агент живёт в процессе, разговор и память задачи — здесь."""

    def __init__(self, path: Path = DB_PATH) -> None:
        self.path = path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # Соединение на операцию: вызовы уходят в отдельный поток, делить одно
        # соединение между ними нельзя, а заводить блокировку — незачем.
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    # --- жизнь базы ----------------------------------------------------------

    async def init(self) -> None:
        await asyncio.to_thread(self._init)

    def _init(self) -> None:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)
        finally:
            connection.close()

    async def forget(self) -> None:
        """Стереть все диалоги. Схема остаётся."""
        await asyncio.to_thread(self._forget)

    def _forget(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions")
            for table in ("messages", "goals", "facts", "turns", "turn_sources"):
                connection.execute(f"DELETE FROM {table}")

    # --- диалоги -------------------------------------------------------------

    async def open(self, mode: str, title: str = "") -> str:
        return await asyncio.to_thread(self._open, mode, title)

    def _open(self, mode: str, title: str) -> str:
        session_id = uuid4().hex
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions (id, title, mode) VALUES (?, ?, ?)",
                (session_id, title, mode),
            )
        return session_id

    async def rename(self, session_id: str, title: str) -> None:
        await asyncio.to_thread(self._rename, session_id, title)

    def _rename(self, session_id: str, title: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET title = ? WHERE id = ? AND (title IS NULL OR title = '')",
                (title[:TITLE_LIMIT], session_id),
            )

    async def sessions(self) -> list[dict[str, object]]:
        return await asyncio.to_thread(self._sessions)

    def _sessions(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.id, s.title, s.mode, s.created_at,
                       (SELECT count(*) FROM turns t WHERE t.session_id = s.id) AS turns,
                       (SELECT g.text FROM goals g WHERE g.session_id = s.id
                        ORDER BY g.id DESC LIMIT 1) AS goal
                FROM sessions s
                ORDER BY s.created_at DESC, s.id DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    async def session(self, session_id: str) -> dict[str, object] | None:
        return await asyncio.to_thread(self._session, session_id)

    def _session(self, session_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, title, mode, created_at FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    async def drop(self, session_id: str) -> None:
        await asyncio.to_thread(self._drop, session_id)

    def _drop(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    # --- реплики -------------------------------------------------------------

    async def history(self, session_id: str) -> list[dict[str, object]]:
        return await asyncio.to_thread(self._history, session_id)

    def _history(self, session_id: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT turn, role, content FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    async def turns_done(self, session_id: str) -> int:
        return await asyncio.to_thread(self._turns_done, session_id)

    def _turns_done(self, session_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT coalesce(max(turn), 0) FROM turns WHERE session_id = ?", (session_id,)
            ).fetchone()
        return int(row[0])

    # --- память задачи -------------------------------------------------------

    async def load_state(self, session_id: str) -> TaskState:
        return await asyncio.to_thread(self._load_state, session_id)

    def _load_state(self, session_id: str) -> TaskState:
        """Снимок состояния из базы: действующая цель и живые пункты.

        Номера пунктов берутся из `fact_id`, а не пересчитываются по порядку:
        трекер ссылается на них в `drop`, и смена номеров после перезапуска
        означала бы, что модель отменяет не тот пункт, который видела.
        """
        with self._connect() as connection:
            goal_row = connection.execute(
                "SELECT text, turn, why FROM goals WHERE session_id = ? ORDER BY id DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            fact_rows = connection.execute(
                """
                SELECT fact_id, section, text, turn FROM facts
                WHERE session_id = ? AND alive = 1
                ORDER BY fact_id
                """,
                (session_id,),
            ).fetchall()

        goal = (
            Goal(text=goal_row["text"], turn=goal_row["turn"], why=goal_row["why"])
            if goal_row
            else None
        )
        facts = tuple(
            Fact(
                id=row["fact_id"],
                section=row["section"],
                text=row["text"],
                turn=row["turn"],
            )
            for row in fact_rows
            if row["section"] in state.SECTIONS
        )
        return TaskState(goal=goal, facts=facts)

    async def save_state(
        self,
        session_id: str,
        turn: int,
        goal: Goal | None,
        added: tuple[Fact, ...],
        dropped: tuple[Fact, ...],
    ) -> None:
        await asyncio.to_thread(self._save_state, session_id, turn, goal, added, dropped)

    def _save_state(
        self,
        session_id: str,
        turn: int,
        goal: Goal | None,
        added: tuple[Fact, ...],
        dropped: tuple[Fact, ...],
    ) -> None:
        with self._connect() as connection:
            if goal is not None:
                connection.execute(
                    "INSERT INTO goals (session_id, turn, text, why) VALUES (?, ?, ?, ?)",
                    (session_id, turn, goal.text, goal.why),
                )
            if added:
                connection.executemany(
                    """
                    INSERT INTO facts (session_id, fact_id, section, text, turn)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (session_id, fact.id, fact.section, fact.text, fact.turn)
                        for fact in added
                    ],
                )
            if dropped:
                connection.executemany(
                    "UPDATE facts SET alive = 0, dropped_at = ? WHERE session_id = ? AND fact_id = ?",
                    [(turn, session_id, fact.id) for fact in dropped],
                )

    async def goal_log(self, session_id: str) -> list[dict[str, object]]:
        return await asyncio.to_thread(self._goal_log, session_id)

    def _goal_log(self, session_id: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT turn, text, why FROM goals WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    # --- ходы ----------------------------------------------------------------

    async def save_turn(
        self,
        session_id: str,
        turn: int,
        row: dict[str, object],
        sources: list[dict[str, object]],
        question: str,
        answer: str,
    ) -> None:
        await asyncio.to_thread(
            self._save_turn, session_id, turn, row, sources, question, answer
        )

    def _save_turn(
        self,
        session_id: str,
        turn: int,
        row: dict[str, object],
        sources: list[dict[str, object]],
        question: str,
        answer: str,
    ) -> None:
        columns = (
            "mode",
            "question",
            "standalone",
            "answer",
            "refused",
            "gate_reason",
            "clarify",
            "kept_goal",
            "claims",
            "prompt_tokens",
            "completion_tokens",
            "stage_tokens",
            "memory_tokens",
            "seconds",
        )
        with self._connect() as connection:
            connection.executemany(
                "INSERT INTO messages (session_id, turn, role, content) VALUES (?, ?, ?, ?)",
                [
                    (session_id, turn, "user", question),
                    (session_id, turn, "assistant", answer),
                ],
            )
            connection.execute(
                f"""
                INSERT INTO turns (session_id, turn, {", ".join(columns)})
                VALUES (?, ?, {", ".join("?" for _ in columns)})
                """,
                (session_id, turn, *(row.get(name) for name in columns)),
            )
            if sources:
                connection.executemany(
                    """
                    INSERT INTO turn_sources
                        (session_id, turn, position, claim, number, chunk_id,
                         path, section, quote, verdict)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            session_id,
                            turn,
                            position,
                            item["claim"],
                            item["number"],
                            item["chunk_id"],
                            item["path"],
                            item["section"],
                            item["quote"],
                            item["verdict"],
                        )
                        for position, item in enumerate(sources, start=1)
                    ],
                )

    async def transcript(self, session_id: str) -> list[dict[str, object]]:
        """Диалог для чтения: ходы вместе с их источниками."""
        return await asyncio.to_thread(self._transcript, session_id)

    def _transcript(self, session_id: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            turns = connection.execute(
                "SELECT * FROM turns WHERE session_id = ? ORDER BY turn", (session_id,)
            ).fetchall()
            sources = connection.execute(
                "SELECT * FROM turn_sources WHERE session_id = ? ORDER BY turn, position",
                (session_id,),
            ).fetchall()

        by_turn: dict[int, list[dict[str, object]]] = {}
        for row in sources:
            by_turn.setdefault(row["turn"], []).append(dict(row))

        return [{**dict(row), "sources": by_turn.get(row["turn"], [])} for row in turns]

    async def stats(self, session_id: str) -> dict[str, object]:
        """Сводка по диалогу: ходы, источники, отказы и цена."""
        return await asyncio.to_thread(self._stats, session_id)

    def _stats(self, session_id: str) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT count(*) AS turns,
                       coalesce(sum(refused), 0) AS refused,
                       coalesce(sum(prompt_tokens + completion_tokens + stage_tokens), 0) AS tokens,
                       coalesce(sum(seconds), 0) AS seconds
                FROM turns WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            cited = connection.execute(
                "SELECT count(DISTINCT turn) FROM turn_sources WHERE session_id = ? AND path <> ''",
                (session_id,),
            ).fetchone()[0]
        return {**dict(row), "turns_with_sources": int(cited)}


def dumps(payload: object) -> str:
    """Короткая обёртка для сохранения снимков в json — ею пользуются прогоны."""
    return json.dumps(payload, ensure_ascii=False, indent=2)
