"""Этап после поиска: кандидатов переупорядочивают и отсекают нерелевантных.

В [day22](../day22/README.md) в контекст уходили первые пять чанков по bm25, и
проверять их не проверял никто. Насколько это плохо, видно на одном запросе — про
порт хранилища day20:

    22.21  day20/servers/vault.py · модуль · часть 1
    19.71  day20/servers/vault.py · def verify · часть 2
    19.51  day20/servers/vault.py · def _db + def _digest
    19.47  day20/servers/vault.py · def _inside
    19.39  day20/servers/vault.py · def journal_list

Две болячки сразу. Первая: счёт практически ровный — 19.4 против 22.2, — то есть
bm25 сам по себе не говорит, где ответ, а где просто тот же файл. Вторая: все
пять выдержек из одного файла, и четыре из них про подписи и журнал, а не про
порт. Контекст занят, а ответа в нём одна строка из пяти.

Отсюда два разных лекарства, и их нельзя путать. **Реранкинг** переупорядочивает
кандидатов — для этого нужна оценка релевантности. **Фильтрация** выбрасывает
тех, кто ниже порога, и ограничивает число выдержек из одного файла. Первое
улучшает место ответа, второе — чистоту и цену контекста.

Реранкеров четыре, и все отдают оценку в шкале 0..1, иначе порог значил бы у
каждого своё:

* `none` — оценки нет вовсе, и это не заглушка, а точка отсчёта: отсекать нечем,
  фильтр вырождается в обрезку по top-K, то есть ровно в поведение day22.
* `heuristic` — bm25 делится на лучший в пуле. Абсолютной шкалы у bm25 нет: он не
  ограничен сверху и растёт с числом термов в запросе, поэтому сравнивать можно
  только внутри одной выдачи. Порог получается относительный — «насколько хуже
  лучшего кандидата», — и ровно такой счёт, как в примере выше, он и ловит.
* `vector` — косинус между запросом и чанком на локальных эмбеддингах. Проверяемое
  утверждение: day22 измерил вектор как слабый *ретривер* — recall@5 0.23 на 2 229
  чанках, — но здесь ему надо упорядочить двадцать, а это другая задача. Стоит
  ноль: векторы уже лежат в индексе.
* `llm` — один вызов на запрос: модель видит пронумерованных кандидатов и ставит
  каждому 0–3. Дорого и медленно, зато умеет то, чего не умеют остальные трое, —
  отличить «про тот же модуль» от «отвечает на вопрос».

Чем сравнивают — тоже разное, и это не оговорка. `vector` сравнивает чанк с тем,
**чем искали**: если запрос переписан в гипотетический ответ, косинус считается
до него, и это воспроизводит ту самую пробу day21, которая дала recall 0.84.
А `llm` судит по тому, **что спросил человек**: релевантность определяется
вопросом, и показывать судье приманку для поиска значило бы спрашивать его не о
том.

Оценки `llm` кэшируются в [relevance.json](relevance.json) по подписи пула. Без
кэша матрица «3 переписывания × 4 реранкера» на 119 пробах стоила бы 357 вызовов
на каждый прогон, а развертка по порогу — ещё столько же на каждое значение
порога. С кэшем развертка бесплатна: порог меняет только `arrange`, а оценки
остаются те же.
"""

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import embed
import index
import llm
from retrieve import Hit

CACHE_PATH = Path(__file__).resolve().parent / "relevance.json"

RERANKERS = ("none", "heuristic", "vector", "llm")
RERANKER_TITLES = {
    "none": "без реранкинга",
    "heuristic": "эвристика · bm25 к лучшему",
    "vector": "вектор · косинус",
    "llm": "модель · оценка 0–3",
}

# Порог у каждого реранкера свой, и одним числом тут не обойтись: у `heuristic`
# оценка относительная и жмётся к единице, у `vector` это косинус статических
# эмбеддингов, у `llm` шкала дискретная в четыре деления. Значения не придуманы,
# а сняты с развертки на 119 пробах — `python day23/scenarios.py sweep`:
#
#   `heuristic` — recall@5 держится на 0.60 до порога 0.67 и начинает падать на
#     0.70. Берём 0.65: контекст с 4 653 символов сжимается до 3 436, то есть на
#     четверть, и ни одна проба за это не заплачена. Садиться ровно на обрыв
#     незачем — bm25 зависит от числа термов в запросе, и у другого вопроса излом
#     сдвинется.
#   `vector` — выше 0.40 recall@5 обваливается с 0.43 до 0.34, а на 0.60 от
#     выдачи остаётся 0.36 выдержки из пяти. Рабочей точки у него нет: 0.35 —
#     это «почти не фильтруй», и стоит он здесь не как инструмент, а как
#     измеренный отрицательный результат.
#   `llm` — шкала дискретная, так что порог означает ровно «оценка 2 и выше».
#     Всё от 0.35 до 0.67 даёт одно и то же: recall@5 0.80 при 3.33 выдержках
#     против 4.99. Строже — только «оценка 3», и это 2.06 выдержки при recall@5
#     0.78: минус две пробы за минус 38% контекста, обмен спорный.
THRESHOLDS: dict[str, float | None] = {
    "none": None,
    "heuristic": 0.65,
    "vector": 0.35,
    "llm": 2 / 3,
}

# Сколько выдержек из одного файла пускать в контекст. Один — слишком строго:
# ответ часто размазан по двум соседним чанкам одного README.
PER_PATH = 2

# Оценка `llm` ставится по началу чанка: медиана разбиения — 261 токен, то есть
# около девятисот символов, и в этот потолок почти весь корпус влезает целиком.
# Стоит он против чанка-гиганта, который раздул бы единственный вызов.
JUDGE_CHARS = 1200

JUDGE_TEMPERATURE = 0.0
JUDGE_TOKENS = 400
CONCURRENCY = 5

JUDGE = (
    "Ты отбираешь выдержки из базы проекта AI Advent — те, которыми можно "
    "ответить на вопрос.\n"
    "Дан вопрос и пронумерованные выдержки. Оцени каждую:\n"
    "3 — содержит прямой ответ на вопрос;\n"
    "2 — отвечает частично или содержит нужную для ответа деталь;\n"
    "1 — про ту же область, но ответа не содержит;\n"
    "0 — к вопросу не относится.\n"
    "Совпадение слов релевантности не делает: выдержка из того же файла, но про "
    "другое, — это 1 или 0.\n"
    "Ответь по одной строке на каждую выдержку, строго в виде «номер: оценка», "
    "без пояснений и без пропусков."
)

VERDICT = re.compile(r"^\s*\[?(\d+)\]?\s*[:.)\-]\s*([0-3])", re.MULTILINE)

BELOW = "ниже порога"
CROWDED = "потолок на файл"
OVERFLOW = "не вошёл в top-K"


@dataclass(frozen=True)
class Scored:
    """Кандидат с оценкой релевантности и судьбой: дошёл или за что отсеян."""

    hit: Hit
    place: int
    relevance: float | None
    kept: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            **self.hit.as_dict(),
            "place": self.place,
            "relevance": None if self.relevance is None else round(self.relevance, 3),
            "kept": self.kept,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class Ranked:
    """Итог второго этапа: кто дошёл, кто отсеян и чего стоила оценка."""

    reranker: str
    threshold: float | None
    per_path: int | None
    pool: int
    kept: list[Scored] = field(default_factory=list)
    dropped: list[Scored] = field(default_factory=list)
    seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    note: str = ""

    @property
    def hits(self) -> list[Hit]:
        return [item.hit for item in self.kept]

    @property
    def moved(self) -> int:
        """Сколько выдержек реранкинг переставил относительно порядка поиска."""
        return sum(
            1 for place, item in enumerate(self.kept, start=1) if item.place != place
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "reranker": self.reranker,
            "title": RERANKER_TITLES[self.reranker],
            "threshold": self.threshold,
            "per_path": self.per_path,
            "pool": self.pool,
            "kept": [item.as_dict() for item in self.kept],
            "dropped": [item.as_dict() for item in self.dropped],
            "moved": self.moved,
            "seconds": round(self.seconds, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "note": self.note,
        }


# --- оценки ------------------------------------------------------------------


def _relative(hits: list[Hit]) -> list[float]:
    """bm25 к лучшему в пуле: абсолютной шкалы у него нет, относительная есть."""
    best = max((hit.score for hit in hits), default=0.0)
    if best <= 0:
        return [0.0 for _ in hits]
    return [max(0.0, min(1.0, hit.score / best)) for hit in hits]


def _cosine(query: str, hits: list[Hit]) -> list[float]:
    """Косинус запроса и чанка. Векторы нормированы, так что это скалярное произведение.

    Векторы чанков берутся из индекса, а не считаются заново: иначе реранкер
    сравнивал бы запрос не с тем, по чему идёт поиск.
    """
    vector = embed.encode_one(query)
    known = index.vectors_by_id([hit.chunk_id for hit in hits])
    return [
        max(0.0, min(1.0, float(known[hit.chunk_id] @ vector))) if hit.chunk_id in known else 0.0
        for hit in hits
    ]


# --- кэш оценок модели -------------------------------------------------------

_CACHE: dict[str, list[int]] | None = None
_ENTRIES: dict[str, dict[str, object]] | None = None


def signature(question: str, hits: list[Hit]) -> str:
    """Подпись пула: вопрос и содержимое кандидатов.

    По тексту, а не по `chunk_id`: идентификаторы живут от сборки до сборки, а
    оценка относится к тексту. Пересобрали индекс — чанки те же, кэш в силе.
    """
    parts = [question.strip(), *(hashlib.sha256(hit.text.encode()).hexdigest()[:12] for hit in hits)]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _load() -> dict[str, list[int]]:
    global _CACHE, _ENTRIES
    if _CACHE is not None and _ENTRIES is not None:
        return _CACHE

    _CACHE, _ENTRIES = {}, {}
    if CACHE_PATH.is_file():
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        for item in payload.get("entries", []):
            _CACHE[item["key"]] = item["scores"]
            _ENTRIES[item["key"]] = item

    return _CACHE


def _save() -> None:
    _load()
    entries = sorted(_ENTRIES.values(), key=lambda item: item["key"])
    CACHE_PATH.write_text(
        json.dumps(
            {"model": llm.MODEL, "scale": "0–3", "entries": entries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _remember(key: str, question: str, hits: list[Hit], scores: list[int]) -> None:
    _load()[key] = scores
    _ENTRIES[key] = {
        "key": key,
        "question": question,
        "chunks": [f"{hit.path}#{hit.ordinal}" for hit in hits],
        "scores": scores,
    }
    _save()


def _candidates(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{number}] {hit.path} · {hit.section}\n{hit.text.strip()[:JUDGE_CHARS]}"
        for number, hit in enumerate(hits, start=1)
    )


def _verdicts(text: str, total: int) -> list[int | None]:
    """Оценки по номерам, а если номера сбиты — по порядку строк.

    Поблажка понадобилась не из снисходительности. На двадцати кандидатах модель
    путает нумерацию двумя разными способами, и различать их обязательно.

    Первый — подписала строку чужим номером, но строк ровно двадцать и идут они
    по порядку: `3: 0`, `2: 0`, `3: 2`, `4: 2`. Здесь читать позиционно
    безопаснее, чем терять пробу: порядок соблюдён, сбита только подпись.

    Второй — честно пропустила кандидата: девятнадцать строк, пронумерованных с
    двойки. Домысливать тут нельзя, и место пропущенного остаётся пустым, чтобы
    его можно было переспросить отдельно.
    """
    found = VERDICT.findall(text)
    by_number = {
        int(number): int(score)
        for number, score in found
        if 1 <= int(number) <= total
    }

    if len(by_number) == total:
        return [by_number[number] for number in range(1, total + 1)]

    if len(found) == total:
        return [int(score) for _, score in found]

    return [by_number.get(number) for number in range(1, total + 1)]


async def _ask(
    question: str, hits: list[Hit], picked: list[int]
) -> tuple[dict[int, int], llm.Reply | None, str]:
    """Спросить про названных кандидатов. Нумерация в промпте своя, наружу — исходная."""
    chosen = [hits[position] for position in picked]

    try:
        reply = await llm.complete(
            [
                {"role": "system", "content": JUDGE},
                {"role": "user", "content": f"Вопрос: {question}\n\n{_candidates(chosen)}"},
            ],
            temperature=JUDGE_TEMPERATURE,
            max_tokens=JUDGE_TOKENS,
        )
    except Exception as exc:
        return {}, None, f"реранкер не ответил: {type(exc).__name__}"

    return (
        {
            picked[position]: score
            for position, score in enumerate(_verdicts(reply.text, len(chosen)))
            if score is not None
        },
        reply,
        "",
    )


async def judge(question: str, hits: list[Hit]) -> tuple[list[int] | None, llm.Reply | None, str]:
    """Оценки модели на весь пул, с кэшем по подписи пула.

    Вызов один, пока модель не пропускает кандидатов. Пропущенных переспрашиваем
    отдельным коротким вызовом — и это не повтор того же запроса: температура
    нулевая, и тот же промпт дал бы тот же пропуск. Спрашивать надо про других.
    """
    key = signature(question, hits)
    cached = _load().get(key)
    if cached is not None:
        return cached, None, ""

    scores: dict[int, int] = {}
    asked = [0, 0]

    for _ in range(2):
        missing = [position for position in range(len(hits)) if position not in scores]
        if not missing:
            break

        got, reply, note = await _ask(question, hits, missing)
        if reply is not None:
            asked[0] += reply.prompt_tokens
            asked[1] += reply.completion_tokens
        if note:
            return None, None, note
        if not got:
            break
        scores.update(got)

    spent = llm.Reply(text="", prompt_tokens=asked[0], completion_tokens=asked[1], seconds=0.0)
    short = len(hits) - len(scores)
    if short:
        return None, spent, f"реранкер не оценил {short} выдержек из {len(hits)}"

    ordered = [scores[position] for position in range(len(hits))]
    _remember(key, question, hits, ordered)
    return ordered, spent, ""


# --- расстановка -------------------------------------------------------------


def arrange(
    hits: list[Hit],
    relevance: list[float | None],
    *,
    reranker: str,
    top_k: int,
    threshold: float | None,
    per_path: int | None,
) -> tuple[list[Scored], list[Scored]]:
    """Порядок, порог, потолок на файл и обрезка — в этом порядке и только так.

    Порядок шагов не случаен. Сначала оценка, потому что по ней сортируют. Потом
    порог: он решает, релевантен ли кандидат сам по себе, и соседи тут ни при чём.
    Потом потолок на файл — он про разнообразие, и считать его надо по тем, кто
    порог уже прошёл, иначе место занял бы отсеянный. И только потом top-K.
    """
    scored = [
        Scored(hit=hit, place=place, relevance=value)
        for place, (hit, value) in enumerate(zip(hits, relevance, strict=True), start=1)
    ]

    # Оценки нет — переставлять не по чему, и порядок поиска остаётся как есть.
    if any(item.relevance is None for item in scored):
        order = scored
    else:
        order = sorted(scored, key=lambda item: (-item.relevance, item.place))

    kept: list[Scored] = []
    dropped: list[Scored] = []
    taken: dict[str, int] = {}

    for item in order:
        if threshold is not None and item.relevance is not None and item.relevance < threshold:
            dropped.append(replace(item, reason=BELOW))
            continue

        if per_path is not None and taken.get(item.hit.path, 0) >= per_path:
            dropped.append(replace(item, reason=CROWDED))
            continue

        if len(kept) >= top_k:
            dropped.append(replace(item, reason=OVERFLOW))
            continue

        taken[item.hit.path] = taken.get(item.hit.path, 0) + 1
        kept.append(replace(item, kept=True))

    return kept, dropped


# --- интерфейс ---------------------------------------------------------------


def scores_cached(
    reranker: str, hits: list[Hit], *, query: str, question: str
) -> list[float | None]:
    """Оценки без сети. У `llm` берутся из кэша, и его отсутствие — ошибка, не ноль."""
    if reranker not in RERANKERS:
        raise ValueError(f"Реранкера {reranker!r} нет. Есть: {', '.join(RERANKERS)}.")

    if not hits:
        return []

    match reranker:
        case "none":
            return [None] * len(hits)
        case "heuristic":
            return list(_relative(hits))
        case "vector":
            return list(_cosine(query, hits))
        case _:
            found = _load().get(signature(question, hits))
            if found is None:
                raise RuntimeError(
                    "Оценок реранкера в кэше нет."
                    " Прогрейте: python day23/scenarios.py rerank"
                )
            return [score / 3 for score in found]


def _ranked(
    reranker: str,
    hits: list[Hit],
    relevance: list[float | None],
    *,
    top_k: int,
    threshold: float | None,
    per_path: int | None,
    seconds: float,
    reply: llm.Reply | None = None,
    note: str = "",
) -> Ranked:
    """Общий хвост обоих входов: оценки уже есть, дальше дело только за фильтром."""
    # Сорванная оценка не должна тихо отсечь всё: без оценок фильтровать нечем.
    if any(value is None for value in relevance):
        threshold = None

    kept, dropped = arrange(
        hits,
        relevance,
        reranker=reranker,
        top_k=top_k,
        threshold=threshold,
        per_path=per_path,
    )

    return Ranked(
        reranker=reranker,
        threshold=threshold,
        per_path=per_path,
        pool=len(hits),
        kept=kept,
        dropped=dropped,
        seconds=seconds,
        prompt_tokens=reply.prompt_tokens if reply else 0,
        completion_tokens=reply.completion_tokens if reply else 0,
        note=note,
    )


def apply_cached(
    reranker: str,
    hits: list[Hit],
    *,
    query: str,
    question: str,
    top_k: int,
    threshold: float | None = None,
    per_path: int | None = PER_PATH,
) -> Ranked:
    """Весь второй этап без сети. Нужен замерам: матрица и развертка — локальный счёт."""
    started = time.perf_counter()
    relevance = scores_cached(reranker, hits, query=query, question=question)
    return _ranked(
        reranker,
        hits,
        relevance,
        top_k=top_k,
        threshold=threshold,
        per_path=per_path,
        seconds=time.perf_counter() - started,
    )


async def apply(
    reranker: str,
    hits: list[Hit],
    *,
    query: str,
    question: str,
    top_k: int,
    threshold: float | None = None,
    per_path: int | None = PER_PATH,
) -> Ranked:
    """Весь второй этап на одну выдачу: оценить, переставить, отсечь, обрезать."""
    if reranker not in RERANKERS:
        raise ValueError(f"Реранкера {reranker!r} нет. Есть: {', '.join(RERANKERS)}.")

    if reranker != "llm" or not hits:
        return apply_cached(
            reranker, hits, query=query, question=question,
            top_k=top_k, threshold=threshold, per_path=per_path,
        )

    started = time.perf_counter()
    raw, reply, note = await judge(question, hits)
    relevance: list[float | None] = (
        [score / 3 for score in raw] if raw else [None] * len(hits)
    )

    return _ranked(
        reranker,
        hits,
        relevance,
        top_k=top_k,
        threshold=threshold,
        per_path=per_path,
        seconds=time.perf_counter() - started,
        reply=reply,
        note=note,
    )


async def warm(pools: list[tuple[str, list[Hit]]], attempts: int = 2) -> dict[str, object]:
    """Прогреть кэш оценок на готовых пулах: один вызов на пул, что есть — не трогаем.

    Попытки две, потому что сорваться вызов может и на стороне провайдера, а
    пустая клетка в кэше потом обрушит весь замер. Что не далось и со второй —
    едет наружу списком: замеру нужно знать, по скольким пробам фильтра не было.
    """
    gate = asyncio.Semaphore(CONCURRENCY)
    asked = 0
    tokens = [0, 0]
    failed: dict[str, str] = {}

    async def one(question: str, hits: list[Hit]) -> None:
        nonlocal asked
        async with gate:
            _, reply, note = await judge(question, hits)
        asked += 1
        if reply:
            tokens[0] += reply.prompt_tokens
            tokens[1] += reply.completion_tokens
        if note:
            failed[question] = note
        else:
            failed.pop(question, None)

    for _ in range(attempts):
        wanted = [
            (question, hits)
            for question, hits in pools
            if hits and _load().get(signature(question, hits)) is None
        ]
        if not wanted:
            break
        await asyncio.gather(*(one(question, hits) for question, hits in wanted))

    return {
        "pools": len(pools),
        "asked": asked,
        "prompt_tokens": tokens[0],
        "completion_tokens": tokens[1],
        "failed": [f"{question[:70]}…: {note}" for question, note in failed.items()],
    }
