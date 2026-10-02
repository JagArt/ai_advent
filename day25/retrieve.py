"""Три способа найти чанк по вопросу: вектор, лексика и их объединение.

Нужда в третьем способе — прямой вывод из [day21](../day21/README.md). Там одним
и тем же индексом искали трижды: коротким поисковым запросом, вопросом на
естественном языке и самим эталонным отрывком. Отрывок находил свой чанк в 84
случаях из ста, вопрос про тот же отрывок — в 23. Разрыв в 0.6 recall лежит
целиком на стороне модели: `model2vec` — статические эмбеддинги, они усредняют
таблицу векторов и сопоставляют формулировки, а не смысл.

Из этого следует, что RAG поверх одного вектора носил бы в контекст мусор в трёх
случаях из четырёх, и сравнение «с RAG / без RAG» мерило бы не RAG. Поэтому
рядом стоит лексика: она берёт ровно то, на чём вектор слаб, — имена `sse_frame`
и `AgentRegistry`, числа, названия файлов. У неё своя слабость, зеркальная:
слова, которых в тексте нет дословно, она не найдёт никогда.

Объединяются списки по RRF — Reciprocal Rank Fusion, `score = Σ 1/(60 + место)`.
Складывать bm25 с косинусом напрямую нельзя: у них нет общей шкалы, bm25 не
ограничен сверху и зависит от длины запроса, а косинус живёт в [-1, 1]. RRF
складывает места, а не числа, и поэтому ему шкалы не нужны вовсе. Константа 60 —
из исходной статьи; её смысл в том, чтобы разница между первым и вторым местом не
подавляла всё остальное.

Ретривер здесь один интерфейс и три реализации — ради того, чтобы прогнать их по
одному набору проб и выбрать не на вкус, а по числам. Числа выбрали не то, что
ожидалось: см. `DEFAULT` ниже и [probes.py](probes.py).
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

import embed
import index

# Сколько кандидатов берётся у каждого ретривера до объединения. Чанк, который
# вектор поставил сороковым, а лексика третьим, должен доехать до RRF — иначе
# объединять было бы нечего, кроме и так совпавших верхушек.
POOL = 40

# Константа RRF из исходной статьи Cormack и соавторов.
RRF_K = 60

SNIPPET_CHARS = 240

RETRIEVERS = ("vector", "lexical", "hybrid")
RETRIEVER_TITLES = {
    "vector": "вектор",
    "lexical": "лексика · FTS5",
    "hybrid": "вектор + лексика · RRF",
}


@dataclass(frozen=True)
class Hit:
    """Найденный чанк: все метаданные и то, кто именно его нашёл.

    `ranks` держит место чанка у каждого ретривера до объединения. Без него
    гибридная выдача — просто список, по которому не видно, принесла его лексика,
    вектор или оба; а весь смысл объединения в том, что они приносят разное.
    """

    chunk_id: int
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
    text: str
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def snippet(self) -> str:
        body = " ".join(self.text.split())
        return body if len(body) <= SNIPPET_CHARS else body[:SNIPPET_CHARS] + "…"

    def as_dict(self) -> dict[str, object]:
        """Для страницы и отчётов: метаданные и отрывок, но не текст целиком."""
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "path": self.path,
            "title": self.title,
            "section": self.section,
            "ordinal": self.ordinal,
            "start": self.start,
            "end": self.end,
            "chars": self.chars,
            "tokens": self.tokens,
            "score": round(self.score, 4),
            "ranks": self.ranks,
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class Found:
    """Выдача одного ретривера на один вопрос вместе с ценой поиска."""

    query: str
    retriever: str
    hits: list[Hit]
    seconds: float

    def as_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "retriever": self.retriever,
            "title": RETRIEVER_TITLES[self.retriever],
            "seconds": round(self.seconds, 4),
            "hits": [hit.as_dict() for hit in self.hits],
        }


def _hit(row, score: float, ranks: dict[str, int]) -> Hit:
    return Hit(
        chunk_id=row["chunk_id"],
        source=row["source"],
        path=row["path"],
        title=row["title"],
        section=row["section"],
        ordinal=row["ordinal"],
        start=row["start_char"],
        end=row["end_char"],
        chars=row["chars"],
        tokens=row["tokens"],
        score=score,
        text=row["text"],
        ranks=ranks,
    )


# --- ретриверы ---------------------------------------------------------------


def _vector_ranked(query: str, limit: int, vector: np.ndarray | None = None) -> list[tuple[int, float]]:
    """Top-k по скалярному произведению: векторы нормированы, это и есть косинус."""
    vectors, rows = index.matrix()
    vector = embed.encode_one(query) if vector is None else vector
    scores = vectors @ vector

    limit = min(limit, len(rows))
    top = np.argpartition(-scores, limit - 1)[:limit] if limit < len(rows) else np.arange(len(rows))
    order = top[np.argsort(-scores[top])]

    return [(rows[position]["chunk_id"], float(scores[position])) for position in order]


def _collect(ranked: list[tuple[int, float]], ranks_name: str) -> list[Hit]:
    rows = index.rows_by_id([chunk_id for chunk_id, _ in ranked])
    return [
        _hit(rows[chunk_id], score, {ranks_name: place})
        for place, (chunk_id, score) in enumerate(ranked, start=1)
        if chunk_id in rows
    ]


def vector(query: str, limit: int) -> list[Hit]:
    return _collect(_vector_ranked(query, limit), "vector")


def lexical(query: str, limit: int) -> list[Hit]:
    return _collect(index.lexical_ranked(query, limit), "lexical")


def hybrid(query: str, limit: int) -> list[Hit]:
    """Объединение двух списков по местам, а не по счёту.

    Пул берётся шире выдачи: чанк, найденный одним ретривером далеко, но другим
    близко, после сложения мест поднимается наверх — ради этого всё и затевалось.
    """
    lists = {
        "vector": _vector_ranked(query, POOL),
        "lexical": index.lexical_ranked(query, POOL),
    }

    fused: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}

    for name, ranked in lists.items():
        for place, (chunk_id, _) in enumerate(ranked, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1 / (RRF_K + place)
            ranks.setdefault(chunk_id, {})[name] = place

    order = sorted(fused, key=lambda chunk_id: -fused[chunk_id])[:limit]
    rows = index.rows_by_id(order)
    return [_hit(rows[chunk_id], fused[chunk_id], ranks[chunk_id]) for chunk_id in order if chunk_id in rows]


_RETRIEVERS: dict[str, Callable[[str, int], list[Hit]]] = {
    "vector": vector,
    "lexical": lexical,
    "hybrid": hybrid,
}

# Под RAG стоит лексика, и это не то, чем день начинался. На 119 пробах day21,
# заданных вопросом, recall@5 вышел 0.23 у вектора, 0.66 у лексики и 0.44 у
# гибрида: равный голос в RRF дал слабому ретриверу право утащить сильного вниз.
# Приглушение голоса вектора делу не помогает — при весе 0.1 recall@5 поднимается
# до 0.68, то есть на две пробы, а recall@1 падает с 0.40 до 0.32. Потолок
# объединения и так низкий: вектор находит всего 5 отрывков из 119, которых не
# нашла лексика, против её 55. Разбор — в README.
DEFAULT = "lexical"


def search(query: str, limit: int = 5, retriever: str = DEFAULT) -> Found:
    """Один вопрос одним ретривером, с замером времени поиска."""
    if retriever not in _RETRIEVERS:
        raise ValueError(f"Ретривера {retriever!r} нет. Есть: {', '.join(RETRIEVERS)}.")

    query = query.strip()
    if not query:
        raise ValueError("Пустой запрос искать нечем.")

    started = time.perf_counter()
    hits = _RETRIEVERS[retriever](query, limit)
    return Found(query=query, retriever=retriever, hits=hits, seconds=time.perf_counter() - started)


def search_all(query: str, limit: int = 5) -> dict[str, Found]:
    """Один вопрос всеми тремя ретриверами — так разницу видно рядом."""
    return {name: search(query, limit, name) for name in RETRIEVERS}
