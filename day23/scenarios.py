"""Прогон из терминала: конвейер, матрица режимов, развертка и набор — в markdown.

    python day23/scenarios.py                 # всё: корпус, индекс, сверка, матрица, набор
    python day23/scenarios.py build           # индексация, без модели и без ключа
    python day23/scenarios.py check           # сверить набор вопросов с корпусом
    python day23/scenarios.py rewrite         # переписать запросы набора и проб, в кэш
    python day23/scenarios.py rerank "вопрос" # пул, оценки всех реранкеров, что отсеяно
    python day23/scenarios.py matrix          # 3 переписывания x 4 реранкера на 119 пробах
    python day23/scenarios.py sweep           # развертка по порогу отсечения
    python day23/scenarios.py ask "вопрос"    # один вопрос во всех пяти режимах
    python day23/scenarios.py questions       # контрольный набор и сравнение режимов

Таблицы отсюда уезжают в README без правок. Ключ нужен шагам `rewrite`, `rerank`,
`ask` и `questions`; `matrix` и `sweep` требуют его только в первый раз, пока
кэши пусты, — дальше считаются на машине.
"""

import argparse
import asyncio
import sys

import answer
import corpus
import evaluate
import index
import pipeline
import probes
import rerank
import retrieve
import rewrite
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
        "\nИз корпуса исключены две папки: своя и day22 —"
        " его README разбирает все десять контрольных вопросов вместе с ответами."
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

    counted: dict[str, int] = {}
    for item in items:
        counted[item.kind] = counted.get(item.kind, 0) + 1

    parts = ", ".join(f"{kind}: {count}" for kind, count in counted.items())
    print(f"Контрольный набор: {len(items)} вопросов ({parts})\n")

    if not notes:
        print("Все обязательные факты нашлись в названных источниках.")
        return

    for note in notes:
        print(f"* {note}")
    raise RuntimeError("Набор разошёлся с корпусом: сверьте questions.json.")


# --- этапы -------------------------------------------------------------------


def warm() -> None:
    """Переписать запросы набора и проб, сложить в кэш и показать примеры."""
    loaded, _ = probes.load()
    questions = [item.question for item in evaluate.load()]
    everything = questions + [probe.question for probe in loaded]

    report = asyncio.run(rewrite.warm(rewrite.MODES, everything))
    print(
        f"Переписано заново: {report['asked']} запросов"
        f" ({', '.join(report['modes'])}) на {report['questions']} вопросов,"
        f" {_thousands(report['prompt_tokens'] + report['completion_tokens'])} токенов,"
        f" {report['seconds']:.0f} с.\n"
    )

    for note in report["failed"]:
        print(f"* {note}")

    print("## Цена переписывания\n")
    cost = rewrite.cost(rewrite.MODES, everything)
    print(
        _table(
            ["Режим", "Запросов", "Токенов запроса", "Токенов ответа", "Секунд", "Символов в запросе"],
            [
                [
                    f"`{mode}` — {rewrite.MODE_TITLES[mode]}",
                    str(cost[mode]["queries"]),
                    str(cost[mode]["prompt_tokens"]),
                    str(cost[mode]["completion_tokens"]),
                    f"{cost[mode]['seconds']:.2f}",
                    str(cost[mode]["chars"]),
                ]
                for mode in rewrite.MODES
            ],
        )
    )

    print("\n### Один вопрос тремя способами\n")
    sample = questions[0] if questions else everything[0]
    for mode in rewrite.MODES:
        found = rewrite.cached(mode, sample)
        print(f"**{rewrite.MODE_TITLES[mode]}**\n\n    {found.query if found else '—'}\n")


def inspect(question: str, retriever: str) -> None:
    """Один вопрос: пул кандидатов и что с ним сделал каждый реранкер."""
    print(f"## Второй этап: «{question}»\n")

    for reranker in rerank.RERANKERS:
        plan = pipeline.Plan.of(reranker=reranker, retriever=retriever)
        context = asyncio.run(pipeline.prepare(question, plan))
        ranked = context.ranked

        print(
            f"### {rerank.RERANKER_TITLES[reranker]}"
            f" — порог {_threshold(plan.threshold)},"
            f" потолок на файл {plan.per_path or '—'}\n"
        )
        print(
            f"Пул {ranked.pool} → {len(context.sources)} выдержек"
            f" ({context.tokens} токенов, {len(context.paths)} файлов),"
            f" переставлено {ranked.moved},"
            f" {ranked.seconds * 1000:.0f} мс"
            + (f", {ranked.prompt_tokens} токенов запроса" if ranked.prompt_tokens else "")
            + "\n"
        )
        if ranked.note:
            print(f"{ranked.note}\n")

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
        print()


def matrix() -> None:
    report = asyncio.run(probes.compare())

    for note in report["notes"]:
        print(f"* {note}")
    if report["notes"]:
        print()

    print("## Матрица: переписывание × реранкер\n")
    print(
        f"Набор: {len(report['probes'])} эталонных отрывков day21, спрашиваем вопросом."
        f" Пул {report['pool']} кандидатов, в контекст не больше {report['top_k']},"
        f" потолок {report['per_path']} выдержки на файл."
        f" Попадание — чанк из того же файла накрыл не меньше"
        f" {report['overlap_share']:.0%} эталонного отрывка.\n"
    )

    def cell(rewriter: str, reranker: str, key: str, digits: int = 2) -> str:
        value = report["results"][rewriter][reranker]["metrics"].get(key)
        if value is None:
            return "—"
        return f"{value:.{digits}f}" if digits else _thousands(value)

    for key, title in (
        ("recall@5", "recall@5 — нужный чанк доехал до промпта"),
        ("recall@1", "recall@1 — он же первым"),
        ("mrr@5", "MRR@5"),
        ("pool_recall", "Потолок режима: нужный чанк попал в пул"),
    ):
        print(f"### {title}\n")
        print(
            _table(
                ["Реранкер"] + [f"`{name}`" for name in report["rewriters"]],
                [
                    [f"`{reranker}`"]
                    + [cell(rewriter, reranker, key) for rewriter in report["rewriters"]]
                    for reranker in report["rerankers"]
                ],
            )
        )
        print()

    print("### Цена контекста и цена фильтрации\n")
    print(
        _table(
            ["Переписывание", "Реранкер", "recall@5", "MRR@5", "Выдержек", "Файлов",
             "Символов", "Пусто", "Потеряно"],
            [
                [
                    f"`{rewriter}`",
                    f"`{reranker}`",
                    cell(rewriter, reranker, "recall@5"),
                    cell(rewriter, reranker, "mrr@5"),
                    cell(rewriter, reranker, "kept"),
                    cell(rewriter, reranker, "paths"),
                    cell(rewriter, reranker, "context_chars", 0),
                    cell(rewriter, reranker, "empty"),
                    cell(rewriter, reranker, "lost"),
                ]
                for rewriter in report["rewriters"]
                for reranker in report["rerankers"]
            ],
        )
    )

    print("\n### Цена переписывания\n")
    print(
        _table(
            ["Режим", "Токенов на запрос", "Секунд", "Символов в запросе"],
            [
                [
                    f"`{mode}`",
                    str(
                        report["cost"][mode]["prompt_tokens"]
                        + report["cost"][mode]["completion_tokens"]
                    ),
                    f"{report['cost'][mode]['seconds']:.2f}",
                    str(report["cost"][mode]["chars"]),
                ]
                for mode in report["rewriters"]
            ],
        )
    )


def sweep(rewriter: str) -> None:
    report = asyncio.run(probes.sweep(rewriter=rewriter))

    print("## Развертка по порогу отсечения\n")
    print(
        f"{report['probes']} проб, переписывание `{report['rewriter']}`,"
        f" пул {report['pool']}, в контекст не больше {report['top_k']},"
        f" потолок {report['per_path']} выдержки на файл."
        " Оценки взяты из кэшей, поэтому меняется только отсечка.\n"
    )

    for reranker in report["rerankers"]:
        chosen = report["chosen"][reranker]
        print(f"### {report['reranker_titles'][reranker]}\n")
        print(
            _table(
                ["Порог", "recall@5", "recall@1", "MRR@5", "Выдержек", "Файлов",
                 "Символов", "Пусто", "Потеряно"],
                [
                    [
                        _threshold(row["threshold"])
                        + ("  ←" if chosen is not None and abs(row["threshold"] - chosen) < 1e-6 else ""),
                        f"{row['metrics']['recall@5']:.2f}",
                        f"{row['metrics']['recall@1']:.2f}",
                        f"{row['metrics']['mrr@5']:.2f}",
                        f"{row['metrics']['kept']:.2f}",
                        f"{row['metrics']['paths']:.2f}",
                        _thousands(row["metrics"]["context_chars"]),
                        f"{row['metrics']['empty']:.2f}",
                        _share(row["metrics"]["lost"]),
                    ]
                    for row in report["results"][reranker]
                ],
            )
        )
        print()


# --- ответы -------------------------------------------------------------------


def ask(question: str, **overrides: object) -> None:
    plans = {mode: answer.plan_of(mode, **overrides) for mode in answer.MODES}
    answers = asyncio.run(answer.every(question, plans=plans))

    print(f"## Вопрос: «{question}»\n")

    for mode in answer.MODES:
        given = answers[mode]
        context = given.context
        print(f"### {answer.MODE_TITLES[mode]}\n")

        if context is not None:
            plan = context.plan
            print(
                f"Поиск — {retrieve.RETRIEVER_TITLES[plan.retriever]}"
                f", переписывание — {rewrite.MODE_TITLES[plan.rewriter]}"
                f", фильтр — {rerank.RERANKER_TITLES[plan.reranker]}"
                f" (порог {_threshold(plan.threshold)})\n"
            )
            if context.rewritten.changed:
                print(f"    запрос: {context.rewritten.query}\n")
            print(
                f"Пул {context.ranked.pool} → {len(context.sources)} выдержек,"
                f" {context.tokens} токенов, {len(context.paths)} файлов,"
                f" {context.stage_seconds * 1000:.0f} мс на этапы\n"
            )
            for source in context.sources:
                relevance = "" if source.relevance is None else f" · {source.relevance:.2f}"
                print(f"  [{source.number}] {source.path} · {source.section}{relevance}")
            print()

        print(given.text)
        print(
            f"\n_{given.reply.prompt_tokens} токенов запроса"
            + (f" + {context.stage_tokens} на этапы" if context and context.stage_tokens else "")
            + f", {given.reply.completion_tokens} ответа, {given.reply.seconds:.1f} с"
            + (f", подкреплено: {', '.join(given.cited_paths)}" if given.cited_paths else "")
            + "_\n"
        )


def questions(**overrides: object) -> None:
    report = asyncio.run(evaluate.run(**overrides))

    for note in report["notes"]:
        print(f"* {note}")
    if report["notes"]:
        print()

    modes = report["modes"]
    titles = report["mode_titles"]

    print("## Контрольный набор\n")
    print(
        f"{len(report['questions'])} вопросов, модель {report['model']},"
        f" пул {report['pool']}, в контекст не больше {report['top_k']}.\n"
    )

    print("### Что делает каждый режим\n")
    print(
        _table(
            ["Режим", "Контекст", "Переписывание", "Фильтр", "Порог"],
            [
                [
                    f"**{titles[mode]}**",
                    "нет" if report["plans"][mode] is None else "есть",
                    "—" if report["plans"][mode] is None
                    else report["rewriter_titles"][report["plans"][mode]["rewriter"]],
                    "—" if report["plans"][mode] is None
                    else report["reranker_titles"][report["plans"][mode]["reranker"]],
                    "—" if report["plans"][mode] is None
                    else _threshold(report["plans"][mode]["threshold"]),
                ]
                for mode in modes
            ],
        )
    )

    print("\n### Качество\n")
    print(
        _table(
            ["Режим", "Факты", "Источник найден", "Точность контекста", "Ответ сослался",
             "Судья, 0–2", "На 2 балла", "Отказы «вне базы»", "Выдержек", "Токенов запроса"],
            [
                [
                    f"**{titles[mode]}**",
                    _share(report["summary"][mode]["facts"]),
                    _share(report["summary"][mode]["sources"]),
                    _share(report["summary"][mode]["precision"]),
                    _share(report["summary"][mode]["cited"]),
                    _share(report["summary"][mode]["judge"]),
                    f"{report['summary'][mode]['judge_full']} из {report['summary'][mode]['questions']}",
                    f"{report['summary'][mode]['refused_outside']} из {report['summary'][mode]['outside']}",
                    _share(report["summary"][mode]["kept"]),
                    _thousands(report["summary"][mode]["total_prompt_tokens"]),
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

    print("\n### По типам вопросов — факты\n")
    print(
        _table(
            ["Тип", "Вопросов"] + [titles[mode] for mode in modes],
            [
                [kind, str(report["by_kind"][modes[0]][kind]["questions"])]
                + [_share(report["by_kind"][mode][kind]["facts"]) for mode in modes]
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
                    _thousands(report["summary"][mode]["stage_tokens"]),
                    _thousands(report["summary"][mode]["prompt_tokens"]),
                    _thousands(report["summary"][mode]["completion_tokens"]),
                    f"{report['summary'][mode]['stage_seconds']:.2f}",
                    f"{report['summary'][mode]['seconds']:.2f}",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогон RAG day23")
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=(
            "all", "corpus", "build", "check", "rewrite", "rerank",
            "matrix", "sweep", "ask", "questions",
        ),
    )
    parser.add_argument("question", nargs="?", default="Зачем проекту ключ OpenRouter?")
    parser.add_argument("--retriever", default=None, choices=retrieve.RETRIEVERS)
    parser.add_argument("--rewriter", default=None, choices=rewrite.MODES)
    parser.add_argument("--reranker", default=None, choices=rerank.RERANKERS)
    parser.add_argument("--pool", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None, dest="top_k")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    overrides = {
        "retriever": args.retriever,
        "rewriter": args.rewriter,
        "reranker": args.reranker,
        "pool": args.pool,
        "top_k": args.top_k,
        "threshold": args.threshold,
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
        case "matrix":
            matrix()
        case "sweep":
            sweep(args.rewriter or "none")
        case "ask":
            ask(args.question, **overrides)
        case "questions":
            questions(
                rewriter=args.rewriter, reranker=args.reranker, threshold=args.threshold
            )
        case _:
            show_corpus()
            print()
            build()
            print()
            check()
            print()
            matrix()
            print()
            questions(
                rewriter=args.rewriter, reranker=args.reranker, threshold=args.threshold
            )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyboardInterrupt) as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
