"""Индекс в SQLite: метаданные строками, векторы блобами, поиск перебором.

Отдельного векторного хранилища здесь нет намеренно. Чанков во всём корпусе
порядка двух тысяч на стратегию, вектор — 256 чисел `float32`, то есть вся матрица
занимает около двух мегабайт. Перемножить её на вектор запроса numpy успевает за
единицы миллисекунд, и приближённый поиск — ANN, IVF, HNSW — тут решал бы задачу,
которой нет: точный ответ и так дешевле, чем любое приближение к нему.

Что действительно важно — метаданные. Чанк без `path`, `section` и `[start, end)`
нельзя ни показать человеку, ни сверить с эталоном, поэтому они лежат рядом с
вектором в одной строке, а не собираются потом по сторонним файлам. Текст
документа целиком хранится тоже здесь: по нему восстанавливается контекст чанка и
ищутся эталонные отрывки, когда файл на диске успел измениться.

Колонки называются `start_char` и `end_char`, потому что `end` в SQLite —
ключевое слово, и `ORDER BY end` разбирается как начало выражения `CASE`.
"""

import json
import sqlite3
import statistics
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import chunking
import corpus
import embed
from chunking import Chunk
from corpus import Document

DB_PATH = Path(__file__).resolve().parent / "index.db"
SNIPPET_CHARS = 240

SCHEMA = """
CREATE TABLE IF NOT EXISTS indexes (
    strategy       TEXT PRIMARY KEY,
    params         TEXT    NOT NULL,
    model          TEXT    NOT NULL,
    dimensions     INTEGER NOT NULL,
    documents      INTEGER NOT NULL,
    chunks         INTEGER NOT NULL,
    chars          INTEGER NOT NULL,
    tokens         INTEGER NOT NULL,
    median_tokens  REAL    NOT NULL,
    p95_tokens     INTEGER NOT NULL,
    max_tokens     INTEGER NOT NULL,
    chunk_seconds  REAL    NOT NULL,
    embed_seconds  REAL    NOT NULL,
    vector_bytes   INTEGER NOT NULL,
    built_at       TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    path     TEXT PRIMARY KEY,
    source   TEXT    NOT NULL,
    title    TEXT    NOT NULL,
    sha256   TEXT    NOT NULL,
    chars    INTEGER NOT NULL,
    text     TEXT    NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy   TEXT    NOT NULL,
    source     TEXT    NOT NULL,
    path       TEXT    NOT NULL,
    title      TEXT    NOT NULL,
    section    TEXT    NOT NULL,
    ordinal    INTEGER NOT NULL,
    start_char INTEGER NOT NULL,
    end_char   INTEGER NOT NULL,
    chars      INTEGER NOT NULL,
    tokens     INTEGER NOT NULL,
    sha256     TEXT    NOT NULL,
    text       TEXT    NOT NULL,
    vector     BLOB    NOT NULL,
    UNIQUE (strategy, path, ordinal)
);
CREATE INDEX IF NOT EXISTS chunks_by_strategy ON chunks (strategy, chunk_id);
CREATE TABLE IF NOT EXISTS probes (
    probe_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    query      TEXT    NOT NULL,
    question   TEXT    NOT NULL,
    path       TEXT    NOT NULL,
    section    TEXT    NOT NULL,
    start_char INTEGER NOT NULL,
    end_char   INTEGER NOT NULL,
    passage    TEXT    NOT NULL,
    doc_sha256 TEXT    NOT NULL,
    model      TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    UNIQUE (question)
);
"""


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(DB_PATH, timeout=30)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            conn.executescript(SCHEMA)
            yield conn


@dataclass(frozen=True)
class Hit:
    """Найденный чанк: близость и все метаданные, по которым его видно и сверяемо."""

    chunk_id: int
    strategy: str
    source: str
    path: str
    title: str
    section: str
    ordinal: int
    start: int
    end: int
    chars: int
    tokens: int
    score: float
    snippet: str

    def as_dict(self) -> dict[str, object]:
        data = self.__dict__.copy()
        data["score"] = round(self.score, 4)
        return data


def _percentile(values: list[int], share: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(int(share * len(ordered)), len(ordered) - 1)]


def _snippet(text: str) -> str:
    body = " ".join(text.split())
    return body if len(body) <= SNIPPET_CHARS else body[:SNIPPET_CHARS] + "…"


def _write_documents(conn: sqlite3.Connection, documents: list[Document]) -> None:
    """Документы общие для всех стратегий: пишутся один раз и обновляются по месту."""
    conn.executemany(
        "INSERT INTO documents (path, source, title, sha256, chars, text)"
        " VALUES (?, ?, ?, ?, ?, ?)"
        " ON CONFLICT (path) DO UPDATE SET source = excluded.source, title = excluded.title,"
        " sha256 = excluded.sha256, chars = excluded.chars, text = excluded.text",
        [
            (document.path, document.source, document.title, document.sha256,
             document.chars, document.text)
            for document in documents
        ],
    )


def _write_chunks(conn: sqlite3.Connection, chunks: list[Chunk], vectors: np.ndarray) -> None:
    conn.execute("DELETE FROM chunks WHERE strategy = ?", (chunks[0].strategy,))
    conn.executemany(
        "INSERT INTO chunks (strategy, source, path, title, section, ordinal, start_char,"
        " end_char, chars, tokens, sha256, text, vector)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (chunk.strategy, chunk.source, chunk.path, chunk.title, chunk.section,
             chunk.ordinal, chunk.start, chunk.end, chunk.chars, chunk.tokens,
             chunk.sha256, chunk.text, embed.to_blob(vector))
            for chunk, vector in zip(chunks, vectors, strict=True)
        ],
    )


def build(strategy: str, documents: list[Document] | None = None) -> dict[str, object]:
    """Собрать индекс одной стратегии заново: чанки, векторы, строки, статистика."""
    documents = documents if documents is not None else corpus.load()
    tokenizer = embed.ModelTokenizer()

    started = time.perf_counter()
    chunks = chunking.build(documents, strategy, tokenizer)
    chunk_seconds = time.perf_counter() - started

    if not chunks:
        raise RuntimeError(f"Стратегия {strategy!r} не дала ни одного чанка: индексировать нечего.")

    encoded = embed.encode([chunk.text for chunk in chunks])
    tokens = [chunk.tokens for chunk in chunks]

    row = {
        "strategy": strategy,
        "params": json.dumps(chunking.defaults(strategy), ensure_ascii=False),
        "model": embed.MODEL_NAME,
        "dimensions": embed.dimensions(),
        "documents": len({chunk.path for chunk in chunks}),
        "chunks": len(chunks),
        "chars": sum(chunk.chars for chunk in chunks),
        "tokens": sum(tokens),
        "median_tokens": round(statistics.median(tokens), 1),
        "p95_tokens": _percentile(tokens, 0.95),
        "max_tokens": max(tokens),
        "chunk_seconds": round(chunk_seconds, 3),
        "embed_seconds": round(encoded.seconds, 3),
        "vector_bytes": encoded.nbytes,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    with db() as conn:
        _write_documents(conn, documents)
        _write_chunks(conn, chunks, encoded.vectors)
        conn.execute(
            f"INSERT INTO indexes ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})"
            " ON CONFLICT (strategy) DO UPDATE SET "
            + ", ".join(f"{name} = excluded.{name}" for name in row if name != "strategy"),
            tuple(row.values()),
        )

    _MATRICES.pop(strategy, None)
    return row


def build_all(
    documents: list[Document] | None = None, strategies: tuple[str, ...] = chunking.STRATEGIES
) -> list[dict[str, object]]:
    """Корпус читается один раз на все стратегии: сравнивать надо один и тот же текст."""
    documents = documents if documents is not None else corpus.load()
    return [build(strategy, documents) for strategy in strategies]


# --- чтение ------------------------------------------------------------------


def built() -> dict[str, dict[str, object]]:
    with db() as conn:
        rows = conn.execute("SELECT * FROM indexes ORDER BY strategy").fetchall()
    return {row["strategy"]: dict(row) for row in rows}


def corpus_summary() -> dict[str, object]:
    with db() as conn:
        rows = conn.execute(
            "SELECT source, COUNT(*) AS files, SUM(chars) AS chars FROM documents GROUP BY source"
        ).fetchall()

    by_source = {row["source"]: {"files": row["files"], "chars": row["chars"]} for row in rows}
    chars = sum(slot["chars"] for slot in by_source.values())
    return {
        "files": sum(slot["files"] for slot in by_source.values()),
        "chars": chars,
        "pages": round(chars / 1800, 1),
        "by_source": by_source,
    }


def document(path: str) -> dict[str, object] | None:
    with db() as conn:
        row = conn.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()
    return dict(row) if row else None


def chunk(chunk_id: int) -> dict[str, object] | None:
    with db() as conn:
        row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        if row is None:
            return None

        neighbours = conn.execute(
            "SELECT chunk_id, ordinal, section FROM chunks"
            " WHERE strategy = ? AND path = ? AND ordinal BETWEEN ? AND ? ORDER BY ordinal",
            (row["strategy"], row["path"], row["ordinal"] - 1, row["ordinal"] + 1),
        ).fetchall()

    data = dict(row)
    data.pop("vector")
    data["neighbours"] = [dict(item) for item in neighbours]
    return data


def chunks_of(strategy: str, path: str) -> list[dict[str, object]]:
    """Все чанки одного документа в одной стратегии — так видно, где прошли границы."""
    with db() as conn:
        rows = conn.execute(
            "SELECT chunk_id, section, ordinal, start_char, end_char, chars, tokens"
            " FROM chunks WHERE strategy = ? AND path = ? ORDER BY ordinal",
            (strategy, path),
        ).fetchall()
    return [dict(row) for row in rows]


def paths(strategy: str) -> list[dict[str, object]]:
    with db() as conn:
        rows = conn.execute(
            "SELECT c.path, c.source, c.title, COUNT(*) AS chunks, SUM(c.tokens) AS tokens"
            " FROM chunks c WHERE c.strategy = ? GROUP BY c.path ORDER BY c.path",
            (strategy,),
        ).fetchall()
    return [dict(row) for row in rows]


# --- поиск -------------------------------------------------------------------

# Матрица одной стратегии: два мегабайта, которые незачем читать на каждый запрос.
_MATRICES: dict[str, tuple[np.ndarray, list[sqlite3.Row]]] = {}


def matrix(strategy: str) -> tuple[np.ndarray, list[sqlite3.Row]]:
    if strategy in _MATRICES:
        return _MATRICES[strategy]

    with db() as conn:
        rows = conn.execute(
            "SELECT chunk_id, strategy, source, path, title, section, ordinal,"
            " start_char, end_char, chars, tokens, vector, text"
            " FROM chunks WHERE strategy = ? ORDER BY chunk_id",
            (strategy,),
        ).fetchall()

    if not rows:
        raise RuntimeError(
            f"Индекса стратегии {strategy!r} нет. Соберите его: python day21/scenarios.py build"
        )

    vectors = embed.from_blobs([row["vector"] for row in rows])
    _MATRICES[strategy] = (vectors, rows)
    return _MATRICES[strategy]


def search_vector(strategy: str, vector: np.ndarray, limit: int) -> list[Hit]:
    """Top-k по скалярному произведению: векторы нормированы, это и есть косинус."""
    vectors, rows = matrix(strategy)
    scores = vectors @ vector

    limit = min(limit, len(rows))
    top = np.argpartition(-scores, limit - 1)[:limit] if limit < len(rows) else np.arange(len(rows))
    order = top[np.argsort(-scores[top])]

    return [
        Hit(
            chunk_id=rows[position]["chunk_id"],
            strategy=rows[position]["strategy"],
            source=rows[position]["source"],
            path=rows[position]["path"],
            title=rows[position]["title"],
            section=rows[position]["section"],
            ordinal=rows[position]["ordinal"],
            start=rows[position]["start_char"],
            end=rows[position]["end_char"],
            chars=rows[position]["chars"],
            tokens=rows[position]["tokens"],
            score=float(scores[position]),
            snippet=_snippet(rows[position]["text"]),
        )
        for position in order
    ]


def search(strategy: str, query: str, limit: int = 5) -> list[Hit]:
    return search_vector(strategy, embed.encode_one(query), limit)


def search_all(query: str, limit: int = 5) -> dict[str, list[Hit]]:
    """Один запрос по всем собранным стратегиям: вектор запроса считается один раз.

    Порядок стратегий — как в `chunking.STRATEGIES`, а не как их вернул SQL: на
    странице и в CLI колонки должны стоять в одном и том же порядке везде.
    """
    vector = embed.encode_one(query)
    ready = built()
    return {
        strategy: search_vector(strategy, vector, limit)
        for strategy in chunking.STRATEGIES
        if strategy in ready
    }
