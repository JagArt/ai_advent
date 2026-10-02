"""Прогон из терминала: цитаты, отказы, оба набора и развертка — сразу в markdown.

    python day24/scenarios.py                 # всё: корпус, индекс, сверка, цитаты, оба набора
    python day24/scenarios.py build           # индексация, без модели и без ключа
    python day24/scenarios.py check           # сверить наборы вопросов с корпусом
    python day24/scenarios.py rewrite         # переписать запросы обоих наборов, в кэш
    python day24/scenarios.py rerank "вопрос" # пул, оценки, что отсеяно
    python day24/scenarios.py ask "вопрос"    # один вопрос во всех четырёх режимах
    python day24/scenarios.py quotes          # все цитаты набора и их сверка с чанками
    python day24/scenarios.py quotes --runs 12  # то же, но доли по двенадцати прогонам
    python day24/scenarios.py questions       # контрольный набор: три проверки и судьи
    python day24/scenarios.py weak            # восемь вопросов слабого контекста
    python day24/scenarios.py gate            # развертка по порогу отказа

Таблицы отсюда уезжают в README без правок. Ключ нужен шагам `rewrite`,
`rerank`, `ask`, `quotes`, `questions` и `weak`; `gate` требует его только в
первый раз, пока кэш оценок пуст, — дальше считается на машине.
"""

import argparse
import asyncio
import sys

import answer
import corpus
import evaluate
import gate
import index
import pipeline
import rerank
import retrieve
import rewrite
import verify
from corpus import SOURCE_TITLES


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _thousands(value: float) -> str:
    return f"{round(value):,}".replace(",", " ")


def _share(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _threshold(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _cut(text: str, limit: int = 90) -> str:
    body = " ".join(text.split())
    return body if len(body) <= limit else body[: limit - 1] + "…"


def _plural(count: int, one: str, few: str, many: str) -> str:
    tail, tens = count % 10, count % 100
    if tail == 1 and tens != 11:
        return f"{count} {one}"
    if 2 <= tail <= 4 and not 12 <= tens <= 14:
        return f"{count} {few}"
    return f"{count} {many}"


def show_corpus() -> None:
    documents, notes = corpus.scan()
    stats = corpus.summary(documents)

    print("## Корпус\n")
    print(
        _table(
            ["Источник", "Файлов", "Символов", "Страниц"],
            [
                [
                    SOURCE_TITLES[source],
                    str(slot["files"]),
                    _thousands(slot["chars"]),
                    f"{slot['chars'] / 1800:.0f}",
                ]
                for source, slot in stats["by_source"].items()
            ]
            + [
                [
                    "**всего**",
                    f"**{stats['files']}**",
                    f"**{_thousands(stats['chars'])}**",
                    f"**{stats['pages']:.0f}**",
                ]
            ],
        )
    )
    print(
        "\nИз корпуса исключены три папки: своя, day22 и day23 —"
        " их README разбирают все десять контрольных вопросов вместе с ответами."
    )

    for note in notes:
        print(f"\nМимо корпуса: {note}")


def build() -> None:
    documents = corpus.load()
    print(f"Индексация: {len(documents)} документов, модель {index.embed.MODEL_NAME}\n")

    row = index.build(documents)
    print(
        f"  {row['chunks']} чанков, чанкинг {row['chunk_seconds']:.1f} с,"
        f" эмбеддинг {row['embed_seconds']:.1f} с,"
        f" FTS5 {row['fts_seconds']:.2f} с,"
        f" векторы {row['vector_bytes'] / 1e6:.2f} МБ"
    )

    print("\n## Цена индекса\n")
    print(
        _table(
            ["Чанков", "Медиана, ток.", "p95", "Макс.", "Чанкинг", "Эмбеддинг", "FTS5", "Векторы"],
            [
                [
                    _thousands(row["chunks"]),
                    f"{row['median_tokens']:.0f}",
                    str(row["p95_tokens"]),
                    str(row["max_tokens"]),
                    f"{row['chunk_seconds']:.1f} с",
                    f"{row['embed_seconds']:.1f} с",
                    f"{row['fts_seconds']:.2f} с",
                    f"{row['vector_bytes'] / 1e6:.2f} МБ",
                ]
            ],
        )
    )


def check() -> None:
    notes = evaluate.check()
    items = evaluate.load()
    weak = evaluate.load_weak()

    counted: dict[str, int] = {}
    for item in items:
        counted[item.kind] = counted.get(item.kind, 0) + 1

    parts = ", ".join(f"{kind}: {count}" for kind, count in counted.items())
    print(f"Контрольный набор: {len(items)} вопросов ({parts})")

    counted = {}
    for item in weak:
        counted[item.kind] = counted.get(item.kind, 0) + 1
    parts = ", ".join(f"{kind}: {count}" for kind, count in counted.items())
    print(f"Слабый контекст: {len(weak)} вопросов ({parts})\n")

    if not notes:
        print("Все обязательные факты нашлись в названных источниках.")
        return

    for note in notes:
        print(f"* {note}")
    raise RuntimeError("Набор разошёлся с корпусом: сверьте questions.json.")


# --- этапы -------------------------------------------------------------------


def warm() -> None:
    """Переписать запросы обоих наборов и сложить в кэш."""
    everything = [item.question for item in evaluate.load() + evaluate.load_weak()]

    report = asyncio.run(rewrite.warm((answer.REWRITER,), everything))
    print(
        f"Переписано заново: {report['asked']} запросов"
        f" на {report['questions']} вопросов,"
        f" {_thousands(report['prompt_tokens'] + report['completion_tokens'])} токенов,"
        f" {report['seconds']:.0f} с.\n"
    )

    for note in report["failed"]:
        print(f"* {note}")


def inspect(question: str, retriever: str) -> None:
    """Один вопрос: что принёс поиск, что оставил фильтр и что скажет порог."""
    plan = pipeline.Plan.of(
        rewriter=answer.REWRITER, reranker=answer.RERANKER, retriever=retriever
    )
    context = asyncio.run(pipeline.prepare(question, plan))
    verdict = gate.inspect(context)

    print(f"## Контекст: «{question}»\n")
    print(
        f"Пул {context.ranked.pool} → {len(context.sources)} выдержек"
        f" ({context.tokens} токенов, {len(context.paths)} файлов),"
        f" лучшая оценка {_share(context.best_relevance)},"
        f" порог отказа {_threshold(verdict.threshold)} —"
        f" {'пропускает' if verdict.passed else f'отказ: {verdict.reason}'}\n"
    )

    rows = [
        [
            str(source.number),
            _share(source.relevance),
            str(source.place),
            f"`{source.path}`",
            source.section,
            "дошло",
        ]
        for source in context.sources
    ] + [
        [
            "—",
            _share(item.relevance),
            str(item.place),
            f"`{item.hit.path}`",
            item.hit.section,
            item.reason,
        ]
        for item in sorted(context.dropped, key=lambda item: item.place)
    ]
    print(_table(["#", "Оценка", "Место", "Файл", "Раздел", "Судьба"], rows))

    if not verdict.passed:
        print(f"\nУточнение: {verdict.clarify}")


# --- ответы -------------------------------------------------------------------


def _claims_table(given: answer.Answer) -> str:
    """Утверждения ответа вместе с их подкреплением."""
    report = given.report
    if report is None or not report.checked:
        return ""

    return _table(
        ["Утверждение", "Источник", "Цитата", "Сверка"],
        [
            [
                _cut(item.claim.text, 70),
                f"`{item.path}`" if item.path else f"[{item.claim.source}] — нет",
                _cut(item.claim.quote, 70) or "—",
                item.verdict,
            ]
            for item in report.checked
        ],
    )


def ask(question: str, **overrides: object) -> None:
    plans = {mode: answer.plan_of(mode, **overrides) for mode in answer.MODES}
    answers = asyncio.run(answer.every(question, plans=plans))

    print(f"## Вопрос: «{question}»\n")

    for mode in answer.MODES:
        given = answers[mode]
        context = given.context
        print(f"### {answer.MODE_TITLES[mode]}\n")

        if context is not None:
            print(
                f"Пул {context.ranked.pool} → {len(context.sources)} выдержек,"
                f" {context.tokens} токенов, {len(context.paths)} файлов,"
                f" лучшая оценка {_share(context.best_relevance)}\n"
            )
            for source in context.sources:
                relevance = "" if source.relevance is None else f" · {source.relevance:.2f}"
                print(f"  [{source.number}] {source.path} · {source.section}{relevance}")
            print()

        if given.refusal:
            print(f"**Отказ — {given.refusal}.** {given.clarify}\n")
        else:
            print(f"{given.text}\n")

        table = _claims_table(given)
        if table:
            print(table + "\n")

        print(
            f"_{given.reply.prompt_tokens} токенов запроса"
            + (f" + {context.stage_tokens} на этапы" if context and context.stage_tokens else "")
            + f", {given.reply.completion_tokens} ответа, {given.reply.seconds:.1f} с"
            + (f", подкреплено: {', '.join(given.cited_paths)}" if given.cited_paths else "")
            + "_\n"
        )


def quotes(runs: int = 1) -> None:
    """Все цитаты контрольного набора и их сверка. Без судей — только механика.

    Выдумка здесь редкая — порядка одной на несколько сотен цитат, — и на
    десяти вопросах её можно не встретить ни разу. Поэтому `--runs N` гоняет
    набор заново N раз: таблица цитата к цитате печатается по первому проходу,
    а доли считаются по всем, и ненайденные выписываются целиком.
    """
    questions = evaluate.load()
    plan = answer.PLANS["cited"]

    asyncio.run(pipeline.heat([item.question for item in questions], [plan]))
    passes = [
        asyncio.run(_gather([answer.ask(item.question, "cited") for item in questions]))
        for _ in range(runs)
    ]

    print("## Цитаты и их сверка\n")
    print(
        f"{len(questions)} вопросов в режиме «{answer.MODE_TITLES['cited']}»"
        + (f", {_plural(runs, 'прогон', 'прогона', 'прогонов')} подряд" if runs > 1 else "")
        + ". Цитата ищется в тексте той выдержки, на которую сослалось утверждение;"
        " пробелы, регистр, ё и типографика приводятся к одному виду, числа — нет.\n"
    )

    counted: dict[str, int] = {}
    total = 0
    rows: list[list[str]] = []
    loose: list[list[str]] = []

    for number, answers in enumerate(passes):
        first = number == 0
        for item, given in zip(questions, answers, strict=True):
            report = given.report
            if report is None or not report.checked:
                if first:
                    verdict = "отказ" if given.refused else "утверждений нет"
                    rows.append([str(item.id), "—", "—", verdict, "—"])
                continue

            for position, checked in enumerate(report.checked):
                total += 1
                counted[checked.verdict] = counted.get(checked.verdict, 0) + 1
                if first:
                    rows.append(
                        [
                            str(item.id) if position == 0 else "",
                            _cut(checked.claim.text, 60),
                            f"`{checked.path}`" if checked.path else "—",
                            checked.verdict,
                            f"{checked.similarity:.2f}",
                        ]
                    )
                if not checked.grounded:
                    loose.append(
                        [
                            str(number + 1),
                            str(item.id),
                            checked.verdict,
                            _cut(checked.claim.quote, 70),
                            f"{checked.similarity:.2f}",
                        ]
                    )

    print(_table(["#", "Утверждение", "Где нашлась", "Сверка", "Похожесть"], rows))

    print("\n### Вердикты\n")
    print(
        _table(
            ["Вердикт", "Цитат", "Доля"],
            [
                [name, str(counted.get(name, 0)), _share(counted.get(name, 0) / total if total else None)]
                for name in verify.VERDICTS
                if counted.get(name)
            ]
            + [["**всего**", f"**{total}**", "**1.00**"]],
        )
    )

    if loose:
        print("\n### Чего не нашлось в чанке\n")
        print(_table(["Прогон", "#", "Вердикт", "Цитата", "Похожесть"], loose))


async def _gather(tasks: list) -> list:
    gated = asyncio.Semaphore(evaluate.CONCURRENCY)

    async def one(task):
        async with gated:
            return await task

    return list(await asyncio.gather(*(one(task) for task in tasks)))


# --- наборы -------------------------------------------------------------------


def _modes_table(report: dict) -> None:
    modes = report["modes"]
    titles = report["mode_titles"]

    print("### Что делает каждый режим\n")
    print(
        _table(
            ["Режим", "Контекст", "Форма ответа", "Порог отказа"],
            [
                [
                    f"**{titles[mode]}**",
                    "нет" if report["plans"][mode] is None else "есть",
                    "json с цитатами" if mode in report["structured_modes"] else "свободный текст",
                    _threshold(report["gate_threshold"]) if mode in report["gated_modes"] else "—",
                ]
                for mode in modes
            ],
        )
    )


def questions(**overrides: object) -> None:
    report = asyncio.run(evaluate.run(**overrides))

    for note in report["notes"]:
        print(f"* {note}")
    if report["notes"]:
        print()

    modes = report["modes"]
    titles = report["mode_titles"]
    summary = report["summary"]

    print("## Контрольный набор\n")
    print(
        f"{len(report['questions'])} вопросов, модель {report['model']},"
        f" пул {report['pool']}, в контекст не больше {report['top_k']},"
        f" порог отказа {_threshold(report['gate_threshold'])}.\n"
    )

    _modes_table(report)

    print("\n### Три проверки задания\n")
    print(
        _table(
            ["Режим", "Источники есть", "Цитаты есть", "Цитата дословна",
             "Цитата нашлась", "Смысл следует из цитаты", "Утверждений", "Выдумано цитат"],
            [
                [
                    f"**{titles[mode]}**",
                    _share(summary[mode]["has_sources"]),
                    _share(summary[mode]["has_quotes"]),
                    _share(summary[mode]["exact"]),
                    _share(summary[mode]["grounded"]),
                    _share(summary[mode]["entail"]),
                    _share(summary[mode]["claims"]),
                    str(summary[mode]["fabricated"]),
                ]
                for mode in modes
            ],
        )
    )

    print("\n### Качество\n")
    print(
        _table(
            ["Режим", "Факты", "Источник найден", "Точность контекста", "Ответ сослался",
             "Судья, 0–2", "На 2 балла", "Выдержек", "Токенов запроса"],
            [
                [
                    f"**{titles[mode]}**",
                    _share(summary[mode]["facts"]),
                    _share(summary[mode]["sources"]),
                    _share(summary[mode]["precision"]),
                    _share(summary[mode]["cited"]),
                    _share(summary[mode]["judge"]),
                    f"{summary[mode]['judge_full']} из {summary[mode]['questions']}",
                    _share(summary[mode]["kept"]),
                    _thousands(summary[mode]["total_prompt_tokens"]),
                ]
                for mode in modes
            ],
        )
    )

    print("\n### Отказы\n")
    print(
        _table(
            ["Режим", "Отказов всего", "Верных «вне базы»", "Ложных", "С уточнением", "Чем вызван"],
            [
                [
                    f"**{titles[mode]}**",
                    str(summary[mode]["refused"]),
                    _share(summary[mode]["refused_rightly"]),
                    _share(summary[mode]["refused_wrongly"]),
                    _share(summary[mode]["asked_back"]),
                    ", ".join(
                        f"{reason}: {count}"
                        for reason, count in summary[mode]["reasons"].items()
                    )
                    or "—",
                ]
                for mode in modes
            ],
        )
    )

    print("\n### По типам вопросов — судья\n")
    kinds = list(report["by_kind"][modes[0]])
    print(
        _table(
            ["Тип", "Вопросов"] + [titles[mode] for mode in modes],
            [
                [kind, str(report["by_kind"][modes[0]][kind]["questions"])]
                + [_share(report["by_kind"][mode][kind]["judge"]) for mode in modes]
                for kind in kinds
            ],
        )
    )

    print("\n### Цена\n")
    print(
        _table(
            ["Режим", "Токенов на этапы", "Токенов запроса", "Токенов ответа",
             "Секунд на этапы", "Секунд ответа"],
            [
                [
                    f"**{titles[mode]}**",
                    _thousands(summary[mode]["stage_tokens"]),
                    _thousands(summary[mode]["prompt_tokens"]),
                    _thousands(summary[mode]["completion_tokens"]),
                    f"{summary[mode]['stage_seconds']:.2f}",
                    f"{summary[mode]['seconds']:.2f}",
                ]
                for mode in modes
            ],
        )
    )

    print("\n### Повопросно — судья\n")
    print(
        _table(
            ["#", "Тип", "Вопрос"] + [titles[mode] for mode in modes],
            [
                [
                    str(report["graded"][modes[0]][position]["question_id"]),
                    report["graded"][modes[0]][position]["kind"],
                    report["graded"][modes[0]][position]["question"],
                ]
                + [
                    str(
                        report["graded"][mode][position]["judge"]
                        if report["graded"][mode][position]["judge"] is not None
                        else "—"
                    )
                    for mode in modes
                ]
                for position in range(len(report["questions"]))
            ],
        )
    )


def weak(**overrides: object) -> None:
    report = asyncio.run(evaluate.run_weak(**overrides))

    modes = report["modes"]
    titles = report["mode_titles"]
    summary = report["summary"]

    print("## Слабый контекст\n")
    print(
        f"{len(report['questions'])} вопросов, на которые верный ответ — отказ:"
        f" половина вне базы, половина неоднозначных."
        f" Порог отказа {_threshold(report['gate_threshold'])}.\n"
    )

    print("### Отказы\n")
    print(
        _table(
            ["Режим", "Отказов", "Доля", "С уточнением", "Судья, 0–2", "На 2 балла", "Чем вызван"],
            [
                [
                    f"**{titles[mode]}**",
                    f"{summary[mode]['refused']} из {summary[mode]['questions']}",
                    _share(summary[mode]["refused_rightly"]),
                    _share(summary[mode]["asked_back"]),
                    _share(summary[mode]["judge"]),
                    f"{summary[mode]['judge_full']} из {summary[mode]['questions']}",
                    ", ".join(
                        f"{reason}: {count}"
                        for reason, count in summary[mode]["reasons"].items()
                    )
                    or "—",
                ]
                for mode in modes
            ],
        )
    )

    print("\n### По типам — доля отказов\n")
    kinds = list(report["by_kind"][modes[0]])
    print(
        _table(
            ["Тип", "Вопросов"] + [titles[mode] for mode in modes],
            [
                [kind, str(report["by_kind"][modes[0]][kind]["questions"])]
                + [
                    _share(report["by_kind"][mode][kind]["refused_rightly"])
                    for mode in modes
                ]
                for kind in kinds
            ],
        )
    )

    print("\n### Повопросно\n")
    print(
        _table(
            ["#", "Тип", "Вопрос"] + [titles[mode] for mode in modes],
            [
                [
                    str(report["graded"][modes[0]][position]["question_id"]),
                    report["graded"][modes[0]][position]["kind"],
                    _cut(report["graded"][modes[0]][position]["question"], 60),
                ]
                + [
                    "отказ" if report["graded"][mode][position]["refused"] else "ответил"
                    for mode in modes
                ]
                for position in range(len(report["questions"]))
            ],
        )
    )

    print("\n### Как звучит уточнение\n")
    for mode in report["gated_modes"]:
        for row in report["graded"][mode]:
            if row["clarify"]:
                print(f"**#{row['question_id']}** — {row['question']}\n")
                print(f"    {row['clarify']}\n")


def sweep() -> None:
    report = asyncio.run(evaluate.sweep())

    print("## Развертка по порогу отказа\n")
    print(
        "Слой в коде решает по одному числу — оценке лучшей дошедшей выдержки."
        " Оценки лежат в кэше реранкера, поэтому вся развертка считается на машине."
        " «Отказал верно» — доля на слабом наборе, «отказал зря» — доля на вопросах"
        " контрольного набора, ответ на которые в базе есть.\n"
    )

    chosen = report["chosen"]
    print(
        _table(
            ["Порог", "Отказал верно", "Отказал зря", "Вне базы", "Неоднозначный", "Что потерял"],
            [
                [
                    _threshold(row["threshold"])
                    + ("  ←" if abs(row["threshold"] - chosen) < 1e-6 else ""),
                    f"{_share(row['right'])} ({row['right_count']} из {row['right_total']})",
                    f"{_share(row['wrong'])} ({row['wrong_count']} из {row['wrong_total']})",
                    f"{row['by_kind']['вне базы']} из {report['weak_total']['вне базы']}",
                    f"{row['by_kind']['неоднозначный']} из {report['weak_total']['неоднозначный']}",
                    ", ".join(f"#{number}" for number in row["lost_ids"]) or "—",
                ]
                for row in report["results"]
            ],
        )
    )

    print("\n### Оценка лучшей выдержки повопросно\n")
    print(
        _table(
            ["Набор", "#", "Тип", "Вопрос", "Выдержек", "Лучшая оценка"],
            [
                [
                    row["set"],
                    str(row["id"]),
                    row["kind"],
                    _cut(row["question"], 60),
                    str(row["kept"]),
                    _share(row["best"]),
                ]
                for row in report["rows"]
            ],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогон RAG day24")
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=(
            "all", "corpus", "build", "check", "rewrite", "rerank",
            "ask", "quotes", "questions", "weak", "gate",
        ),
    )
    parser.add_argument("question", nargs="?", default="Зачем проекту ключ OpenRouter?")
    parser.add_argument("--retriever", default=None, choices=retrieve.RETRIEVERS)
    parser.add_argument("--rewriter", default=None, choices=rewrite.MODES)
    parser.add_argument("--reranker", default=None, choices=rerank.RERANKERS)
    parser.add_argument("--pool", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None, dest="top_k")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--gate", type=float, default=None, dest="gate_threshold")
    parser.add_argument("--runs", type=int, default=1, dest="passes")
    args = parser.parse_args()

    overrides = {
        "retriever": args.retriever,
        "rewriter": args.rewriter,
        "reranker": args.reranker,
        "pool": args.pool,
        "top_k": args.top_k,
        "threshold": args.threshold,
    }
    runs = {
        "rewriter": args.rewriter,
        "reranker": args.reranker,
        "threshold": args.threshold,
        "gate_threshold": args.gate_threshold,
    }

    match args.step:
        case "corpus":
            show_corpus()
        case "build":
            show_corpus()
            print()
            build()
        case "check":
            check()
        case "rewrite":
            warm()
        case "rerank":
            inspect(args.question, args.retriever or retrieve.DEFAULT)
        case "ask":
            ask(args.question, **overrides)
        case "quotes":
            quotes(args.passes)
        case "questions":
            questions(**runs)
        case "weak":
            weak(**runs)
        case "gate":
            sweep()
        case _:
            show_corpus()
            print()
            build()
            print()
            check()
            print()
            sweep()
            print()
            quotes()
            print()
            questions(**runs)
            print()
            weak(**runs)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyboardInterrupt) as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
