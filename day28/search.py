"""Локальный поиск по индексу day25: тот же код, та же база, ни одного вызова модели.

День про генерацию, а не про поиск, поэтому поиск здесь не переписан, а взят у
[day25](../day25/retrieve.py) как есть: лексика FTS5 по `day25/index.db`, top-5.
На 119 пробах day22 она дала recall@5 0.66 против 0.44 у гибрида, и ничего
лучшего без вызова модели в репозитории нет. Переписывание и LLM-реранкер day23
подняли бы выдачу до 0.81, но ценой вызова модели **до** ответа — и тогда у
локального и облачного режимов был бы разный контекст, а сравнение моделей
превратилось бы в сравнение поисков.

Модули day25 подключаются через `sys.path` на время импорта: у day28 нет своих
`retrieve`, `index`, `embed`, `corpus` и `chunking`, так что перекрыть их нечем,
а путь после импорта убирается — иначе `llm` day25 перехватил бы `llm` этого дня.

Базу этот модуль не создаёт. `index.db()` в day25 выполняет `CREATE TABLE IF NOT
EXISTS` при каждом открытии, и обращение к несуществующему файлу оставило бы
после себя пустой индекс, который дальше выглядел бы как собранный. Поэтому
первым делом проверяется, что файл есть.
"""

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

DAY25 = Path(__file__).resolve().parent.parent / "day25"
INDEX_PATH = DAY25 / "index.db"
BUILD_HINT = "python day25/scenarios.py"

TOP_K = 5
RETRIEVER = "lexical"


def _load_day25() -> tuple[ModuleType, ModuleType]:
    sys.path.insert(0, str(DAY25))
    try:
        index = importlib.import_module("index")
        retrieve = importlib.import_module("retrieve")
    finally:
        sys.path.remove(str(DAY25))
    return index, retrieve


_index, _retrieve = _load_day25()


class IndexMissing(RuntimeError):
    pass


def require_index() -> dict[str, object]:
    """Сведения об индексе day25 или ошибка с командой сборки — без побочных эффектов."""
    if not INDEX_PATH.exists():
        raise IndexMissing(f"Индекса {INDEX_PATH} нет. Соберите его: {BUILD_HINT}")
    info = _index.built()
    if info is None:
        raise IndexMissing(f"Файл {INDEX_PATH} есть, но индекс в нём не собран: {BUILD_HINT}")
    return info


def index_status() -> dict[str, object]:
    try:
        info = require_index()
    except IndexMissing as exc:
        return {"ready": False, "path": str(INDEX_PATH), "error": str(exc)}
    return {
        "ready": True,
        "path": str(INDEX_PATH),
        "documents": info["documents"],
        "chunks": info["chunks"],
        "built_at": info["built_at"],
        "retriever": RETRIEVER,
        "top_k": TOP_K,
    }


@dataclass(frozen=True)
class Source:
    """Выдержка под своим номером: по нему ответ на неё ссылается."""

    number: int
    chunk_id: int
    source: str
    path: str
    section: str
    score: float
    tokens: int
    text: str

    @property
    def snippet(self) -> str:
        body = " ".join(self.text.split())
        return body if len(body) <= 240 else body[:240] + "…"

    def as_dict(self) -> dict[str, object]:
        return {
            "number": self.number,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "path": self.path,
            "section": self.section,
            "score": round(self.score, 3),
            "tokens": self.tokens,
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class Context:
    """Блок, который пристёгивается к вопросу, и всё, что о нём надо знать."""

    query: str
    sources: list[Source]
    seconds: float

    @property
    def block(self) -> str:
        """Выдержки под номерами, с путём и разделом в шапке: по ним ответ проверяем."""
        return "\n\n".join(
            f"[{source.number}] {source.path} · {source.section}\n{source.text.strip()}"
            for source in self.sources
        )

    @property
    def tokens(self) -> int:
        return sum(source.tokens for source in self.sources)

    @property
    def paths(self) -> list[str]:
        seen: list[str] = []
        for source in self.sources:
            if source.path not in seen:
                seen.append(source.path)
        return seen

    def as_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "retriever": RETRIEVER,
            "seconds": round(self.seconds, 4),
            "tokens": self.tokens,
            "paths": self.paths,
            "sources": [source.as_dict() for source in self.sources],
        }


def prepare(question: str, limit: int = TOP_K) -> Context:
    """Найти чанки и сложить из них контекст. Без модели, без ключа, без сети."""
    require_index()
    found = _retrieve.search(question, limit, RETRIEVER)
    sources = [
        Source(
            number=number,
            chunk_id=hit.chunk_id,
            source=hit.source,
            path=hit.path,
            section=hit.section,
            score=hit.score,
            tokens=hit.tokens,
            text=hit.text,
        )
        for number, hit in enumerate(found.hits, start=1)
    ]
    return Context(query=found.query, sources=sources, seconds=found.seconds)
