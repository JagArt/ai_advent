"""Два набора, четыре режима и три проверки задания — плюс судья над цитатами.

Контрольный набор перенесён из [day22](../day22/questions.json) без правок, и
это по-прежнему сознательно: линейка должна быть той же, иначе «стало лучше»
нельзя предъявить. Рядом встал второй набор, [weak.json](weak.json), которого в
прошлых днях не было, — восемь вопросов, где верный ответ один: отказ.

Он понадобился потому, что мерить отказ по контрольному набору нечем. Вопросов
«вне базы» там два, и на двух пробах разница между «отказался всегда» и
«отказался случайно» неразличима. Второй набор при этом не заменяет первый, а
дополняет его с другой стороны: контрольный ловит **ложные** отказы, то есть
цену режима, а слабый — **верные**, то есть его пользу.

## Три проверки задания и почему третья требует отдельного судьи

Задание просит проверить три вещи: есть ли в ответе источники, есть ли цитаты и
совпадает ли смысл ответа с цитатами. Первые две механические и считаются по
структуре ответа. Третья — нет, и подменить её сверкой нельзя.

Разницу видно на примере. Модель приводит дословную цитату «порт 8770 занят
хранилищем day20» и пишет рядом утверждение «хранилище day20 слушает 8765».
Цитата настоящая, [verify.py](verify.py) ставит `дословно`, источник на месте,
а утверждение противоречит собственному подкреплению. Поймать это может только
тот, кто прочитает обе строки.

Поэтому судей здесь два, и у них разные глаза:

* **судья качества** — как в day22 и day23: видит вопрос, ожидание и ответ, не
  видит ни режима, ни контекста. Отвечает, насколько ответ соответствует
  ожидаемому содержанию;
* **судья-сверщик** — новый: видит **только** утверждение и его цитату. Ни
  вопроса, ни корпуса, ни остального ответа. Его спрашивают об одном: следует
  ли утверждение из этой строки. Контекст ему не показывают намеренно — зная
  правильный ответ на вопрос, он подтверждал бы верные утверждения независимо
  от того, подкреплены они цитатой или нет, а мерить надо именно подкрепление.

## Отказ считается по слоям, а не целиком

Колонка «отказался» ничего не объясняет: отказ слоя в коде и отказ самой модели
лечатся разным. Первый — порогом, второй — промптом. Поэтому у каждого отказа
записано, кто его поставил: `контекст пуст`, `ниже порога` или `правило в
промпте`.
"""

import asyncio
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import answer
import cite
import corpus
import gate
import index
import llm
import pipeline
import rerank
import rewrite
import verify
from pipeline import Plan

BASE_DIR = Path(__file__).resolve().parent
QUESTIONS_PATH = BASE_DIR / "questions.json"
WEAK_PATH = BASE_DIR / "weak.json"

CONCURRENCY = 5
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 160
ENTAIL_MAX_TOKENS = 120

KINDS = ("в базе", "общий", "вне базы")
WEAK_KINDS = ("вне базы", "неоднозначный")

# Обороты, которыми ответ признаёт незнание. Список заведомо неполон — он и не
# может быть полным, — поэтому итоговую оценку отказа ставит судья.
REFUSALS = (
    "в базе ответа нет",
    "нет данных",
    "нет информации",
    "не указан",
    "не указана",
    "не названа",
    "не названо",
    "не нашёл",
    "не нашел",
    "не знаю",
    "отсутству",
    "не содержится",
    "не приводится",
    "не могу",
)

JUDGE = (
    "Ты проверяешь ответ на вопрос о проекте AI Advent. Тебе даны вопрос, "
    "ожидаемое содержание ответа и сам ответ. Оцени, насколько ответ "
    "соответствует ожидаемому содержанию:\n"
    "2 — отвечает по существу, ничего важного не упущено и ничего не выдумано;\n"
    "1 — отвечает частично: верно, но неполно, либо верное смешано с лишним;\n"
    "0 — не отвечает, уходит от вопроса или противоречит ожидаемому.\n"
    "Если ожидаемое содержание — отказ, то отказ оценивай на 2, а уверенный "
    "выдуманный ответ на 0.\n"
    "Оценивай только соответствие ожиданию, а не стиль и не длину.\n"
    "Ответь ровно двумя строками:\n"
    "оценка: 0, 1 или 2\n"
    "почему: одно короткое предложение"
)

ENTAIL = (
    "Тебе даны цитата из документа и утверждение, которое на неё ссылается.\n"
    "Реши одно: следует ли утверждение из цитаты.\n"
    "да — цитата подтверждает утверждение целиком;\n"
    "частично — цитата подтверждает часть утверждения, остальное в ней не сказано;\n"
    "нет — цитата утверждения не подтверждает или противоречит ему.\n"
    "Своих знаний о предмете не привлекай: кроме цитаты, у тебя ничего нет, и "
    "верное само по себе утверждение без подтверждения в цитате — это «нет».\n"
    "Ответь ровно двумя строками:\n"
    "следует: да, частично или нет\n"
    "почему: одно короткое предложение"
)

SCORE = re.compile(r"^\s*оценка\s*:\s*([012])", re.IGNORECASE | re.MULTILINE)
WHY = re.compile(r"^\s*почему\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
FOLLOWS = re.compile(r"^\s*следует\s*:\s*(да|частично|нет)", re.IGNORECASE | re.MULTILINE)

# Балл сверщика в той же шкале 0..1, что и доли рядом: «частично» — половина.
ENTAIL_SCORES = {"да": 1.0, "частично": 0.5, "нет": 0.0}

DIGIT_GROUP = re.compile(r"(\d)[\s\u00a0\u202f](?=\d{3}\b)")


@dataclass(frozen=True)
class Question:
    """Контрольный вопрос вместе с тем, по чему его будут проверять."""

    id: int
    kind: str
    question: str
    expect: str
    facts: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Entailed:
    """Одно утверждение глазами сверщика: следует ли оно из своей цитаты."""

    text: str
    quote: str
    verdict: str
    why: str

    @property
    def score(self) -> float | None:
        return ENTAIL_SCORES.get(self.verdict)

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "quote": self.quote,
            "verdict": self.verdict,
            "why": self.why,
            "score": self.score,
        }


@dataclass(frozen=True)
class Graded:
    """Один ответ со всеми оценками: механические доли и два судьи."""

    question_id: int
    kind: str
    question: str
    mode: str
    text: str
    facts_found: list[str]
    facts_missed: list[str]
    sources_found: list[str]
    sources_missed: list[str]
    context_paths: list[str]
    cited_paths: list[str]
    kept: int
    pool: int
    dropped: int
    best_relevance: float | None
    refused: bool
    refusal: str
    clarify: str
    claims: int
    quoted: int
    sourced: int
    exact: int
    grounded: int
    fabricated: int
    broken: str
    entailed: list[Entailed]
    judge: int | None
    judge_why: str
    prompt_tokens: int
    stage_tokens: int
    completion_tokens: int
    seconds: float
    stage_seconds: float
    structured: dict[str, object] | None
    verify: dict[str, object] | None
    context: dict[str, object] | None

    # --- механика по источникам и фактам (как в day22 и day23) ---

    @property
    def facts_share(self) -> float | None:
        total = len(self.facts_found) + len(self.facts_missed)
        return len(self.facts_found) / total if total else None

    @property
    def expected_sources(self) -> set[str]:
        return set(self.sources_found) | set(self.sources_missed)

    @property
    def sources_hit(self) -> bool | None:
        """Попал ли в выдачу хоть один файл, по которому можно ответить."""
        return bool(self.sources_found) if self.expected_sources else None

    @property
    def precision(self) -> float | None:
        """Доля выдержек из файлов, по которым на вопрос можно ответить."""
        if not self.expected_sources or not self.context_paths:
            return None
        return sum(
            1 for path in self.context_paths if path in self.expected_sources
        ) / len(self.context_paths)

    @property
    def cited_hit(self) -> bool | None:
        """Сослался ли ответ хотя бы на один из ожидаемых источников."""
        if not self.expected_sources:
            return None
        return bool(self.expected_sources & set(self.cited_paths))

    # --- три проверки задания ---

    @property
    def has_sources(self) -> bool | None:
        """Есть ли источник у каждого утверждения. У отказа вопрос не стоит."""
        if self.refused or not self.claims:
            return None
        return self.sourced == self.claims

    @property
    def has_quotes(self) -> bool | None:
        """Есть ли цитата у каждого утверждения."""
        if self.refused or not self.claims:
            return None
        return self.quoted == self.claims

    @property
    def exact_share(self) -> float | None:
        """Доля цитат, найденных в своей выдержке дословно."""
        return self.exact / self.claims if self.claims else None

    @property
    def grounded_share(self) -> float | None:
        return self.grounded / self.claims if self.claims else None

    @property
    def entail_share(self) -> float | None:
        """Доля утверждений, следующих из своей цитаты. «Частично» — половина."""
        scores = [score for item in self.entailed if (score := item.score) is not None]
        return sum(scores) / len(scores) if scores else None

    # --- отказ ---

    @property
    def refused_rightly(self) -> bool | None:
        """Отказ там, где отказ и нужен. На вопросах из базы величина не определена."""
        if self.kind not in ("вне базы", "неоднозначный"):
            return None
        return self.refused

    @property
    def refused_wrongly(self) -> bool | None:
        """Отказ там, где ответ в базе был. Цена режима «не знаю»."""
        if self.kind in ("вне базы", "неоднозначный"):
            return None
        return self.refused

    @property
    def asked_back(self) -> bool | None:
        """Сопроводил ли отказ просьбой уточнить."""
        return bool(self.clarify.strip()) if self.refused else None

    @property
    def total_prompt_tokens(self) -> int:
        return self.prompt_tokens + self.stage_tokens

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["entailed"] = [item.as_dict() for item in self.entailed]
        data["facts_share"] = self.facts_share
        data["sources_hit"] = self.sources_hit
        data["precision"] = self.precision
        data["cited_hit"] = self.cited_hit
        data["has_sources"] = self.has_sources
        data["has_quotes"] = self.has_quotes
        data["exact_share"] = self.exact_share
        data["grounded_share"] = self.grounded_share
        data["entail_share"] = self.entail_share
        data["refused_rightly"] = self.refused_rightly
        data["refused_wrongly"] = self.refused_wrongly
        data["asked_back"] = self.asked_back
        data["total_prompt_tokens"] = self.total_prompt_tokens
        return data


# --- наборы ------------------------------------------------------------------


def load() -> list[Question]:
    payload = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    return [Question(**item) for item in payload["questions"]]


def load_weak() -> list[Question]:
    """Набор слабого контекста. Фактов и источников у него нет по определению."""
    payload = json.loads(WEAK_PATH.read_text(encoding="utf-8"))
    return [Question(**item, facts=[], sources=[]) for item in payload["questions"]]


def normal(text: str) -> str:
    """К виду, в котором факт из набора сравним с тем, что написала модель."""
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    while DIGIT_GROUP.search(text):
        text = DIGIT_GROUP.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def check() -> list[str]:
    """Проверить сам набор: факты обязаны находиться в названных источниках."""
    documents = {document.path: normal(document.text) for document in corpus.load()}
    notes: list[str] = []

    for item in load():
        if item.kind == "вне базы":
            if item.facts or item.sources:
                notes.append(f"#{item.id}: вопрос «вне базы» не может иметь фактов и источников")
            continue

        for path in item.sources:
            if path not in documents:
                notes.append(f"#{item.id}: источника {path} нет в корпусе")

        texts = [documents[path] for path in item.sources if path in documents]
        for fact in item.facts:
            if not any(normal(fact) in text for text in texts):
                notes.append(f"#{item.id}: факта «{fact}» нет ни в одном из названных источников")

    for item in load_weak():
        if item.kind not in WEAK_KINDS:
            notes.append(f"слабый #{item.id}: неизвестный тип «{item.kind}»")

    return notes


# --- судьи -------------------------------------------------------------------


def looks_refused(text: str) -> bool:
    body = normal(text)
    return any(marker in body for marker in REFUSALS)


async def judge(item: Question, text: str, gated: asyncio.Semaphore) -> tuple[int | None, str]:
    """Балл судьи качества. Режим и контекст ему не сообщаются."""
    task = (
        f"Вопрос: {item.question}\n\n"
        f"Ожидаемое содержание: {item.expect}\n\n"
        f"Ответ: {text or '(пусто)'}"
    )

    async with gated:
        try:
            reply = await llm.complete(
                [{"role": "system", "content": JUDGE}, {"role": "user", "content": task}],
                temperature=JUDGE_TEMPERATURE,
                max_tokens=JUDGE_MAX_TOKENS,
            )
        except Exception as exc:
            return None, f"судья не ответил: {type(exc).__name__}"

    score = SCORE.search(reply.text)
    why = WHY.search(reply.text)
    return (int(score.group(1)) if score else None), (why.group(1).strip() if why else "")


async def entail(claim: cite.Claim, gated: asyncio.Semaphore) -> Entailed:
    """Следует ли утверждение из своей цитаты. Сверщик не видит ни вопроса, ни базы."""
    quote = claim.quote.strip()
    if not quote:
        return Entailed(text=claim.text, quote="", verdict="нет", why="цитаты нет вовсе")

    task = f"Цитата: {quote}\n\nУтверждение: {claim.text.strip()}"

    async with gated:
        try:
            reply = await llm.complete(
                [{"role": "system", "content": ENTAIL}, {"role": "user", "content": task}],
                temperature=JUDGE_TEMPERATURE,
                max_tokens=ENTAIL_MAX_TOKENS,
            )
        except Exception as exc:
            return Entailed(
                text=claim.text, quote=quote, verdict="",
                why=f"сверщик не ответил: {type(exc).__name__}",
            )

    verdict = FOLLOWS.search(reply.text)
    why = WHY.search(reply.text)
    return Entailed(
        text=claim.text,
        quote=quote,
        verdict=verdict.group(1).lower() if verdict else "",
        why=why.group(1).strip() if why else "",
    )


async def entail_all(given: answer.Answer, gated: asyncio.Semaphore) -> list[Entailed]:
    """Сверщик по всем утверждениям ответа. У свободного текста утверждений нет."""
    if not given.claims:
        return []
    return list(await asyncio.gather(*(entail(claim, gated) for claim in given.claims)))


# --- оценка ------------------------------------------------------------------


def grade(
    item: Question,
    given: answer.Answer,
    score: int | None,
    why: str,
    entailed: list[Entailed],
) -> Graded:
    """Механические доли по одному ответу. Без модели и без случайности."""
    body = normal(given.text)
    found = [fact for fact in item.facts if normal(fact) in body]

    context = given.context
    retrieved = context.paths if context else []
    in_context = [path for path in item.sources if path in retrieved]
    report = given.report or verify.Report()

    return Graded(
        question_id=item.id,
        kind=item.kind,
        question=item.question,
        mode=given.mode,
        text=given.text,
        facts_found=found,
        facts_missed=[fact for fact in item.facts if fact not in found],
        sources_found=in_context,
        sources_missed=[path for path in item.sources if path not in in_context],
        context_paths=[source.path for source in context.sources] if context else [],
        cited_paths=given.cited_paths,
        kept=len(context.sources) if context else 0,
        pool=context.ranked.pool if context else 0,
        dropped=len(context.dropped) if context else 0,
        best_relevance=context.best_relevance if context else None,
        refused=given.refused or looks_refused(given.text),
        refusal=given.refusal,
        clarify=given.clarify,
        claims=report.claims,
        quoted=report.quoted,
        sourced=report.sourced,
        exact=report.exact,
        grounded=report.grounded,
        fabricated=report.fabricated,
        broken=given.structured.broken if given.structured else "",
        entailed=entailed,
        judge=score,
        judge_why=why,
        prompt_tokens=given.reply.prompt_tokens,
        stage_tokens=context.stage_tokens if context else 0,
        completion_tokens=given.reply.completion_tokens,
        seconds=given.reply.seconds,
        stage_seconds=round(context.stage_seconds, 3) if context else 0.0,
        structured=given.structured.as_dict() if given.structured else None,
        verify=given.report.as_dict() if given.report else None,
        context=context.as_dict() if context else None,
    )


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _share(graded: list[Graded], name: str) -> float | None:
    values = [
        float(value) for item in graded if (value := getattr(item, name)) is not None
    ]
    return _mean(values)


def summarize(graded: list[Graded]) -> dict[str, object]:
    """Сводка по одному режиму. Пустые колонки остаются пустыми, а не нулевыми."""
    judged = [item.judge for item in graded if item.judge is not None]
    with_context = [item for item in graded if item.context]
    structured = [item for item in graded if item.structured is not None]
    answered = [item for item in structured if not item.refused and item.claims]

    reasons: dict[str, int] = {}
    for item in graded:
        if item.refusal:
            reasons[item.refusal] = reasons.get(item.refusal, 0) + 1

    return {
        "questions": len(graded),
        "facts": _share(graded, "facts_share"),
        "sources": _share(graded, "sources_hit"),
        "precision": _share(graded, "precision"),
        "cited": _share(graded, "cited_hit"),
        "judge": _mean([float(score) for score in judged]),
        "judge_full": sum(1 for score in judged if score == 2),
        # Три проверки задания.
        "has_sources": _share(graded, "has_sources"),
        "has_quotes": _share(graded, "has_quotes"),
        "exact": _share(graded, "exact_share"),
        "grounded": _share(graded, "grounded_share"),
        "entail": _share(graded, "entail_share"),
        "claims": _mean([float(item.claims) for item in answered]),
        "fabricated": sum(item.fabricated for item in graded),
        "broken": sum(1 for item in structured if item.broken),
        # Отказ.
        "refused": sum(1 for item in graded if item.refused),
        "refused_rightly": _share(graded, "refused_rightly"),
        "refused_wrongly": _share(graded, "refused_wrongly"),
        "asked_back": _share(graded, "asked_back"),
        "reasons": reasons,
        # Цена.
        "kept": _mean([float(item.kept) for item in with_context]),
        "empty": sum(1 for item in with_context if item.kept == 0),
        "prompt_tokens": round(sum(item.prompt_tokens for item in graded) / len(graded)),
        "stage_tokens": round(sum(item.stage_tokens for item in graded) / len(graded)),
        "total_prompt_tokens": round(
            sum(item.total_prompt_tokens for item in graded) / len(graded)
        ),
        "completion_tokens": round(sum(item.completion_tokens for item in graded) / len(graded)),
        "seconds": round(sum(item.seconds for item in graded) / len(graded), 2),
        "stage_seconds": round(sum(item.stage_seconds for item in graded) / len(graded), 2),
    }


def by_kind(graded: list[Graded], kinds: tuple[str, ...] = KINDS) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[Graded]] = {}
    for item in graded:
        grouped.setdefault(item.kind, []).append(item)

    return {kind: summarize(grouped[kind]) for kind in kinds if kind in grouped}


# --- прогон -------------------------------------------------------------------


async def _one(
    item: Question,
    plans: dict[str, Plan | None],
    gated: asyncio.Semaphore,
    threshold: float | None,
) -> dict[str, Graded]:
    """Один вопрос во всех режимах. Одинаковые планы считаются один раз."""
    prepared = await answer.contexts(item.question, plans=plans)

    async def one(mode: str) -> answer.Answer:
        async with gated:
            return await answer.ask(
                item.question, mode, context=prepared[mode], threshold=threshold
            )

    answers = dict(
        zip(plans, await asyncio.gather(*(one(mode) for mode in plans)), strict=True)
    )

    scores, entailments = await asyncio.gather(
        asyncio.gather(*(judge(item, answers[mode].text, gated) for mode in plans)),
        asyncio.gather(*(entail_all(answers[mode], gated) for mode in plans)),
    )

    return {
        mode: grade(item, answers[mode], *score, entailed)
        for mode, score, entailed in zip(plans, scores, entailments, strict=True)
    }


async def _run(
    questions: list[Question],
    *,
    kinds: tuple[str, ...],
    rewriter: str | None = None,
    reranker: str | None = None,
    threshold: float | None = None,
    gate_threshold: float | None = None,
) -> dict[str, object]:
    """Общий прогон: оба набора отличаются только списком вопросов и типами."""
    info = index.require()

    plans = {
        mode: answer.plan_of(mode, rewriter=rewriter, reranker=reranker, threshold=threshold)
        for mode in answer.MODES
    }

    # Кэши прогреваются до прогона, а не по ходу: иначе вопросы на четыре
    # режима открыли бы разом десятки запросов к модели, и первый же лимит
    # провайдера сорвал бы часть набора.
    await pipeline.heat(
        [item.question for item in questions],
        [plan for plan in plans.values() if plan is not None],
    )

    gated = asyncio.Semaphore(CONCURRENCY)
    rows = await asyncio.gather(
        *(_one(item, plans, gated, gate_threshold) for item in questions)
    )

    graded = {mode: [row[mode] for row in rows] for mode in plans}

    return {
        "index": info,
        "model": llm.MODEL,
        "pool": pipeline.POOL,
        "top_k": pipeline.TOP_K,
        "per_path": rerank.PER_PATH,
        "gate_threshold": gate.THRESHOLD if gate_threshold is None else gate_threshold,
        "notes": check(),
        "questions": [asdict(item) for item in questions],
        "modes": list(answer.MODES),
        "mode_titles": answer.MODE_TITLES,
        "structured_modes": list(answer.STRUCTURED),
        "gated_modes": list(answer.GATED),
        "plans": {mode: plan.as_dict() if plan else None for mode, plan in plans.items()},
        "rewriter_titles": rewrite.MODE_TITLES,
        "reranker_titles": rerank.RERANKER_TITLES,
        "summary": {mode: summarize(items) for mode, items in graded.items()},
        "by_kind": {mode: by_kind(items, kinds) for mode, items in graded.items()},
        "graded": {
            mode: [item.as_dict() for item in items] for mode, items in graded.items()
        },
    }


async def run(**overrides: object) -> dict[str, object]:
    """Контрольный набор во всех четырёх режимах со всеми оценками."""
    return await _run(load(), kinds=KINDS, **overrides)  # type: ignore[arg-type]


async def run_weak(**overrides: object) -> dict[str, object]:
    """Набор слабого контекста: то же самое, но верный ответ везде — отказ."""
    return await _run(load_weak(), kinds=WEAK_KINDS, **overrides)  # type: ignore[arg-type]


# --- развертка по порогу отказа ----------------------------------------------

# Шкала реранкера дискретная — 0, 1/3, 2/3, 1, — поэтому порогов, которые
# что-то меняют, ровно четыре. Промежуточные значения в сетке стоят затем,
# чтобы это было видно в таблице, а не принималось на слово.
GATE_GRID = (0.0, 0.34, 0.5, 0.67, 0.84, 1.0)


async def sweep() -> dict[str, object]:
    """Что порог отказа даёт и что отнимает — на обоих наборах сразу.

    Модель здесь не вызывается вовсе, и это не экономия, а свойство замера:
    слой в коде решает по одному числу — оценке лучшей дошедшей выдержки, — и
    число это уже лежит в кэше реранкера. Поэтому развертка по шести порогам
    стоит ровно столько же, сколько по одному.
    """
    info = index.require()
    plan = answer.PLANS["guarded"]
    assert plan is not None

    strong = load()
    weak = load_weak()
    everything = strong + weak

    await pipeline.heat([item.question for item in everything], [plan])

    best: dict[int, float | None] = {}
    rows: list[dict[str, object]] = []

    for position, item in enumerate(everything):
        context = await asyncio.to_thread(pipeline.prepare_cached, item.question, plan)
        best[position] = context.best_relevance
        rows.append(
            {
                "id": item.id,
                "set": "контрольный" if item in strong else "слабый",
                "kind": item.kind,
                "question": item.question,
                "kept": len(context.sources),
                "best": context.best_relevance,
            }
        )

    # «Отказать верно» — на слабом наборе, «отказать зря» — на вопросах
    # контрольного набора, у которых ответ в базе есть.
    right = [position for position, item in enumerate(everything) if item in weak]
    wrong = [
        position
        for position, item in enumerate(everything)
        if item in strong and item.kind != "вне базы"
    ]
    outside = [
        position
        for position, item in enumerate(everything)
        if item in strong and item.kind == "вне базы"
    ]

    def refuses(position: int, threshold: float) -> bool:
        value = best[position]
        return value is None or value < threshold

    results = []
    for threshold in GATE_GRID:
        caught = [position for position in right if refuses(position, threshold)]
        lost = [position for position in wrong if refuses(position, threshold)]
        results.append(
            {
                "threshold": threshold,
                "right": len(caught) / len(right) if right else None,
                "right_count": len(caught),
                "right_total": len(right),
                "wrong": len(lost) / len(wrong) if wrong else None,
                "wrong_count": len(lost),
                "wrong_total": len(wrong),
                "outside": sum(1 for position in outside if refuses(position, threshold)),
                "outside_total": len(outside),
                "by_kind": {
                    kind: sum(
                        1
                        for position in right
                        if everything[position].kind == kind and refuses(position, threshold)
                    )
                    for kind in WEAK_KINDS
                },
                "lost_ids": [everything[position].id for position in lost],
            }
        )

    return {
        "index": info,
        "grid": list(GATE_GRID),
        "chosen": gate.THRESHOLD,
        "plan": plan.as_dict(),
        "kinds": list(WEAK_KINDS),
        "weak_total": {
            kind: sum(1 for item in weak if item.kind == kind) for kind in WEAK_KINDS
        },
        "rows": rows,
        "results": results,
    }
