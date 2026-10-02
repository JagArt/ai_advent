"""Индекс в SQLite: метаданные строками, векторы блобами, рядом — FTS5.

От [day21](../day21/index.py) здесь два отличия, и оба следуют из того, что день
про ответ, а не про разбиение.

Первое: стратегия одна, и столбца `strategy` больше нет. Второе важнее — рядом с
векторами появился полнотекстовый индекс. Причина названа числами в day21: на
вопросе, заданном человеком, вектор находил нужный чанк в 23 случаях из ста, а на
отрывке того же текста — в 84. Разрыв не про разбиение, а про статические
эмбеддинги, которые сопоставляют формулировки, а не смысл: имена вроде
`sse_frame` или `AgentRegistry` в таблице векторов размазаны, и вопрос про них
вектор не ловит. Лексика ловит их буквально, поэтому индекса здесь два.

FTS5 заведена таблицей внешнего содержимого: свою копию текста она не держит, а
читает его из `chunks` по `content_rowid`. Пять тысяч чанков дублировать незачем,
а главное — две копии текста рано или поздно разъезжаются.

Отдельного векторного хранилища по-прежнему нет. Вся матрица — около двух
мегабайт, numpy перемножает её на вектор запроса за доли миллисекунды, и ANN
решал бы здесь задачу, которой нет.

Колонки называются `start_char` и `end_char`, потому что `end` в SQLite —
ключевое слово, и `ORDER BY end` разбирается как начало выражения `CASE`.
"""

import json
import re
import sqlite3
import statistics
import time
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import chunking
import corpus
import embed
from chunking import Chunk
from corpus import Document

DB_PATH = Path(__file__).resolve().parent / "index.db"

WORD = re.compile(r"\w+", re.UNICODE)

# В day22 потолок был 24 слова, и его хватало: в поиск приходил вопрос человека.
# Здесь приходит ещё и гипотетический ответ из [rewrite.py](rewrite.py) — абзац в
# два-три предложения, и обрезка по двадцать четвёртому слову отрезала бы ему
# половину смысла. Повторы при этом схлопываются, поэтому потолок считается по
# разным словам, а не по всем подряд.
MAX_QUERY_TOKENS = 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS index_info (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    strategy       TEXT    NOT NULL,
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
    fts_seconds    REAL    NOT NULL,
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
    UNIQUE (path, ordinal)
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5 (
    section,
    title,
    text,
    content = 'chunks',
    content_rowid = 'chunk_id',
    tokenize = 'unicode61 remove_diacritics 2'
);
"""

# Раздел весит больше заголовка файла, а заголовок — больше тела: имя раздела
# человек в вопросе называет почти дословно, а тело совпадает с ним случайно.
BM25_WEIGHTS = (4.0, 2.0, 1.0)


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(DB_PATH, timeout=30)) as conn:
        conn.row_factory = sqlite3.Row
        with conn:
            conn.executescript(SCHEMA)
            yield conn


def _percentile(values: list[int], share: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(int(share * len(ordered)), len(ordered) - 1)]


# --- сборка ------------------------------------------------------------------


def _write_documents(conn: sqlite3.Connection, documents: list[Document]) -> None:
    """Документы по месту, а выбывшие — вон.

    Обновления по конфликту мало: файл, который выпал из корпуса, останется
    строкой навсегда и будет отвечать на запросы о путях и источниках текстом,
    которого в индексе уже нет.
    """
    paths = [document.path for document in documents]
    conn.execute(
        f"DELETE FROM documents WHERE path NOT IN ({', '.join('?' * len(paths))})", paths
    )
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
    """Чанки переписываются целиком: индекс собирается заново, а не дописывается."""
    conn.execute("DELETE FROM chunks")
    conn.executemany(
        "INSERT INTO chunks (source, path, title, section, ordinal, start_char,"
        " end_char, chars, tokens, sha256, text, vector)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (chunk.source, chunk.path, chunk.title, chunk.section, chunk.ordinal,
             chunk.start, chunk.end, chunk.chars, chunk.tokens, chunk.sha256,
             chunk.text, embed.to_blob(vector))
            for chunk, vector in zip(chunks, vectors, strict=True)
        ],
    )


def _rebuild_fts(conn: sqlite3.Connection) -> float:
    """Перестроить полнотекстовый индекс по содержимому `chunks`.

    Таблица внешнего содержимого за правками `chunks` сама не следит — обновлять
    её надо явно. Триггеры тут были бы лишними: чанки пишутся один раз на сборку,
    и `rebuild` по готовой таблице дешевле, чем тысячи срабатываний по строке.
    """
    started = time.perf_counter()
    conn.execute("INSERT INTO chunks_fts (chunks_fts) VALUES ('rebuild')")
    return time.perf_counter() - started


def build(documents: list[Document] | None = None) -> dict[str, object]:
    """Собрать индекс заново: чанки, векторы, строки, полнотекстовый индекс."""
    documents = documents if documents is not None else corpus.load()
    tokenizer = embed.ModelTokenizer()

    started = time.perf_counter()
    chunks = chunking.build(documents, tokenizer)
    chunk_seconds = time.perf_counter() - started

    if not chunks:
        raise RuntimeError("Корпус не дал ни одного чанка: индексировать нечего.")

    encoded = embed.encode([chunk.text for chunk in chunks])
    tokens = [chunk.tokens for chunk in chunks]

    with db() as conn:
        _write_documents(conn, documents)
        _write_chunks(conn, chunks, encoded.vectors)
        fts_seconds = _rebuild_fts(conn)

        row = {
            "id": 1,
            "strategy": chunking.STRATEGY,
            "params": json.dumps(chunking.defaults(), ensure_ascii=False),
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
            "fts_seconds": round(fts_seconds, 3),
            "vector_bytes": encoded.nbytes,
            "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        conn.execute(
            f"INSERT INTO index_info ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})"
            " ON CONFLICT (id) DO UPDATE SET "
            + ", ".join(f"{name} = excluded.{name}" for name in row if name != "id"),
            tuple(row.values()),
        )

    forget()
    return {name: value for name, value in row.items() if name != "id"}


# --- чтение ------------------------------------------------------------------


def built() -> dict[str, object] | None:
    with db() as conn:
        row = conn.execute("SELECT * FROM index_info WHERE id = 1").fetchone()
    return {name: row[name] for name in row.keys() if name != "id"} if row else None


def require() -> dict[str, object]:
    info = built()
    if info is None:
        raise RuntimeError("Индекса нет. Соберите его: python day24/scenarios.py build")
    return info


def document(path: str) -> dict[str, object] | None:
    with db() as conn:
        row = conn.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()
    return dict(row) if row else None


def sources() -> dict[str, str]:
    """Источник каждого документа одним запросом: нужен всем сводкам по источникам."""
    with db() as conn:
        rows = conn.execute("SELECT path, source FROM documents").fetchall()
    return {row["path"]: row["source"] for row in rows}


def chunk(chunk_id: int) -> dict[str, object] | None:
    """Чанк целиком: текст, все метаданные и соседи — чтобы видеть, где прошла граница."""
    with db() as conn:
        row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        if row is None:
            return None

        neighbours = conn.execute(
            "SELECT chunk_id, ordinal, section FROM chunks"
            " WHERE path = ? AND ordinal BETWEEN ? AND ? ORDER BY ordinal",
            (row["path"], row["ordinal"] - 1, row["ordinal"] + 1),
        ).fetchall()

    data = dict(row)
    data.pop("vector")
    data["neighbours"] = [dict(item) for item in neighbours]
    return data


def chunks_of(path: str) -> list[dict[str, object]]:
    with db() as conn:
        rows = conn.execute(
            "SELECT chunk_id, section, ordinal, start_char, end_char, chars, tokens"
            " FROM chunks WHERE path = ? ORDER BY ordinal",
            (path,),
        ).fetchall()
    return [dict(row) for row in rows]


def paths() -> list[dict[str, object]]:
    with db() as conn:
        rows = conn.execute(
            "SELECT path, source, title, COUNT(*) AS chunks, SUM(tokens) AS tokens"
            " FROM chunks GROUP BY path ORDER BY path",
        ).fetchall()
    return [dict(row) for row in rows]


# --- два индекса --------------------------------------------------------------

# Матрица: два мегабайта, которые незачем читать из базы на каждый запрос.
_MATRIX: tuple[np.ndarray, list[sqlite3.Row]] | None = None


def forget() -> None:
    global _MATRIX
    _MATRIX = None


def matrix() -> tuple[np.ndarray, list[sqlite3.Row]]:
    """Все векторы одним куском памяти и строки чанков в том же порядке."""
    global _MATRIX
    if _MATRIX is not None:
        return _MATRIX

    with db() as conn:
        rows = conn.execute(
            "SELECT chunk_id, source, path, title, section, ordinal,"
            " start_char, end_char, chars, tokens, vector, text"
            " FROM chunks ORDER BY chunk_id",
        ).fetchall()

    if not rows:
        raise RuntimeError("Индекса нет. Соберите его: python day24/scenarios.py build")

    _MATRIX = (embed.from_blobs([row["vector"] for row in rows]), rows)
    return _MATRIX


def match_expression(query: str) -> str | None:
    """Вопрос человека в выражение FTS5.

    Термы берутся префиксными, потому что `unicode61` русской морфологии не знает:
    «индексация» в вопросе и «индексации» в тексте для него разные слова, а
    «индексаци*» покрывает обе. Токены ещё и закавычены — иначе «AND» из вопроса
    стало бы оператором, а дефис отрицанием.

    Повтор слова в запросе ничего не добавляет: `"порт"* OR "порт"*` — это тот же
    отбор и тот же bm25. Для вопроса это мелочь, а для гипотетического ответа
    HyDE — нет: в абзаце слова повторяются, и без схлопывания потолок уходил бы
    на них вместо новых термов.
    """
    tokens = dict.fromkeys(WORD.findall(query.lower()))
    terms = list(tokens)[:MAX_QUERY_TOKENS]
    return " OR ".join(f'"{token}"*' for token in terms) if terms else None


def lexical_ranked(query: str, limit: int) -> list[tuple[int, float]]:
    """Идентификаторы чанков по bm25, лучший первым.

    Термы объединяются через OR, а не AND: вопрос на естественном языке почти
    никогда не входит в чанк всеми словами, и строгий запрос чаще всего пуст.
    Отсев при этом не теряется — его делает bm25, которому редкое слово весит
    больше частого.
    """
    expression = match_expression(query)
    if expression is None:
        return []

    weights = ", ".join(str(weight) for weight in BM25_WEIGHTS)
    with db() as conn:
        rows = conn.execute(
            f"SELECT rowid, bm25(chunks_fts, {weights}) AS rank FROM chunks_fts"
            " WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (expression, limit),
        ).fetchall()

    # bm25 в SQLite тем меньше, чем лучше совпадение; наружу отдаём «больше — лучше».
    return [(row["rowid"], -float(row["rank"])) for row in rows]


def vectors_by_id(chunk_ids: list[int]) -> dict[int, np.ndarray]:
    """Векторы названных чанков из уже прочитанной матрицы.

    Нужны реранкеру: считать эмбеддинг чанка заново он мог бы, но тогда сравнивал
    бы не с тем, что лежит в индексе, а с тем, что получилось сейчас.
    """
    vectors, rows = matrix()
    wanted = set(chunk_ids)
    return {
        row["chunk_id"]: vectors[position]
        for position, row in enumerate(rows)
        if row["chunk_id"] in wanted
    }


def rows_by_id(chunk_ids: list[int]) -> dict[int, sqlite3.Row]:
    """Строки чанков из уже прочитанной матрицы: второй раз в базу ходить незачем."""
    _, rows = matrix()
    wanted = set(chunk_ids)
    return {row["chunk_id"]: row for row in rows if row["chunk_id"] in wanted}
