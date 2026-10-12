"""Локальный поиск по индексу day25 — как в day28 — плюс укладка выдержек в бюджет токенов.

Поиск не меняется: лексика FTS5 по `day25/index.db`, top-5, кодом day25 через
`sys.path`. День про то, как модель читает контекст, а не про то, как он ищется,
и у обоих профилей выдача одна и та же — иначе разница в ответах говорила бы о
поиске, а не об оптимизации.

Новое здесь — `Context.fit(budget)`. В day28 почти всё время локального ответа
уходило не на генерацию, а на чтение промпта: пять выдержек по 260 токенов в
среднем давали около 2000 токенов, и 4B переваривала их за пять секунд до первого
токена. Бюджет режет это чтение в два шага. Сначала потолок делится между
выдержками так, чтобы короткие остались целыми, а резались только длинные
(`fit`). Потом каждая длинная ужимается до своей доли не с хвоста, а по смыслу:
остаются предложения, в которых встречаются слова вопроса (`Source.squeeze`).
Все пять выдержек и их номера сохраняются, поэтому ссылки [n] в ответе
указывают на те же выдержки, что и без бюджета.

Токены считаются тем же токенайзером, которым day25 резал чанки: поле `tokens`
у чанка уже в этих единицах, и предложения меряются им же. Для Qwen это не
точная мера — её токенов на тот же текст выходит примерно на треть больше, —
но пропорции она держит, а точная цифра видна в `prompt_tokens` ответа.

Базу этот модуль не создаёт: `index.db()` в day25 делает `CREATE TABLE IF NOT
EXISTS`, и обращение к несуществующему файлу оставило бы пустой индекс, похожий
на собранный. Поэтому первым делом проверяется, что файл есть.
"""

import importlib
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType

DAY25 = Path(__file__).resolve().parent.parent / "day25"
INDEX_PATH = DAY25 / "index.db"
BUILD_HINT = "python day25/scenarios.py"

TOP_K = 5
RETRIEVER = "lexical"
MIN_TAIL = 40  # меньше этого на выдержку не выделяем: обрывок без смысла только мешает
GAP = "…"

SENTENCE = re.compile(r"(?<=[.!?;])\s+|\n+")
WORD = re.compile(r"[\w-]+")
STOPWORDS = {
    "как", "что", "где", "когда", "почему", "зачем", "какой", "какая", "какие", "каком", "каких",
    "это", "этот", "эта", "для", "или", "если", "при", "про", "над", "под", "без", "все", "всё",
    "его", "её", "их", "они", "она", "оно", "там", "тут", "так", "уже", "ещё", "еще", "чем", "чём",
    "the", "and", "for", "not", "with",
}


def _stems(text: str) -> set[str]:
    """Слова вопроса без служебных, обрезанные до четырёх символов — грубая замена морфологии."""
    return {word[:4] for word in (w.lower() for w in WORD.findall(text)) if len(word) >= 3 and word not in STOPWORDS}


def _overlap(sentence: str, stems: set[str]) -> int:
    words = {word[:4] for word in (w.lower() for w in WORD.findall(sentence))}
    return len(words & stems)


def _load_day25() -> tuple[ModuleType, ModuleType, ModuleType]:
    sys.path.insert(0, str(DAY25))
    try:
        index = importlib.import_module("index")
        retrieve = importlib.import_module("retrieve")
        embed = importlib.import_module("embed")
    finally:
        sys.path.remove(str(DAY25))
    return index, retrieve, embed


_index, _retrieve, _embed = _load_day25()
_tokenizer = _embed.ModelTokenizer()


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
    trimmed: bool = False

    @property
    def snippet(self) -> str:
        body = " ".join(self.text.split())
        return body if len(body) <= 240 else body[:240] + "…"

    def squeeze(self, cap: int, query: str) -> "Source":
        """Ужать выдержку до `cap` токенов, оставив предложения, где есть слова вопроса.

        Хвост резать нельзя: в контрольном наборе ответ на вопрос #1 стоит в
        последних ста символах выдержки на 1175, и простая обрезка его теряла.
        Вместо этого выдержка делится на предложения (у кода — на строки), каждое
        получает балл — сколько разных слов вопроса в нём встретилось, — и в
        выдержке остаются самые полезные, в исходном порядке, пока хватает
        бюджета. Первое предложение остаётся всегда: без него непонятно, о чём
        вообще выдержка. Пропуски отмечаются строкой «…».
        """
        sentences = [s for s in SENTENCE.split(self.text.strip()) if s.strip()]
        sizes = [_tokenizer.count(s) for s in sentences]
        if sum(sizes) <= cap:
            return self

        stems = _stems(query)
        scored = [(0 if i == 0 else -_overlap(s, stems), i) for i, s in enumerate(sentences)]
        scored.sort()  # первое предложение и затем по убыванию совпадений, при равенстве — раньше в тексте

        kept: set[int] = set()
        used = 0
        for _, i in scored:
            if used + sizes[i] > cap:
                continue
            kept.add(i)
            used += sizes[i]

        parts: list[str] = []
        for i, sentence in enumerate(sentences):
            if i in kept:
                parts.append(sentence.strip())
            elif not parts or parts[-1] != GAP:
                parts.append(GAP)
        return replace(self, text="\n".join(parts), tokens=used, trimmed=True)

    def as_dict(self) -> dict[str, object]:
        return {
            "number": self.number,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "path": self.path,
            "section": self.section,
            "score": round(self.score, 3),
            "tokens": self.tokens,
            "trimmed": self.trimmed,
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class Context:
    """Блок, который пристёгивается к вопросу, и всё, что о нём надо знать."""

    query: str
    sources: list[Source]
    seconds: float
    budget: int | None = None
    dropped: int = 0

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

    def fit(self, budget: int) -> "Context":
        """Уложить выдержки в бюджет, срезая длинные и не трогая короткие.

        Общий потолок делится между выдержками поровну; те, что короче своей доли,
        остаются целиком, а их остаток перераспределяется между длинными — пока
        потолок не перестанет меняться. Так пять выдержек по 400, 398, 400, 105 и
        108 токенов при бюджете 1200 дают 329, 329, 329, 105 и 108: все пять на
        месте, порезаны только три длинные. Выдержка, которой досталось меньше
        `MIN_TAIL`, отбрасывается — обрывок без смысла только мешает.
        """
        if self.tokens <= budget:
            return replace(self, budget=budget, dropped=0)

        cap = budget
        remaining = list(self.sources)
        spare = budget
        while remaining:
            cap = spare // len(remaining)
            whole = [source for source in remaining if source.tokens <= cap]
            if not whole:
                break
            spare -= sum(source.tokens for source in whole)
            remaining = [source for source in remaining if source.tokens > cap]

        kept: list[Source] = []
        for source in self.sources:
            if source.tokens <= cap:
                kept.append(source)
            elif cap >= MIN_TAIL:
                kept.append(source.squeeze(cap, self.query))
        return replace(self, sources=kept, budget=budget, dropped=len(self.sources) - len(kept))

    def as_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "retriever": RETRIEVER,
            "seconds": round(self.seconds, 4),
            "tokens": self.tokens,
            "budget": self.budget,
            "dropped": self.dropped,
            "trimmed": sum(source.trimmed for source in self.sources),
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
