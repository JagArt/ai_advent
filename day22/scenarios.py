"""Прогон из терминала: поиск, два режима и контрольный набор — сразу в markdown.

    python day22/scenarios.py               # всё: корпус, индекс, ретриверы, набор
    python day22/scenarios.py build         # индексация, без модели и без ключа
    python day22/scenarios.py check         # сверить набор вопросов с корпусом
    python day22/scenarios.py retrieval     # три ретривера на пробах day21
    python day22/scenarios.py ask "вопрос"  # один вопрос в двух режимах
    python day22/scenarios.py questions     # контрольный набор и сравнение

Таблицы отсюда уезжают в README без правок. Ключ нужен шагам `ask` и
`questions` — всё остальное считается на машине.
"""

import argparse
import asyncio
import sys

import answer
import corpus
import evaluate
import index
import probes
import retrieve
from corpus import SOURCE_TITLES


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _thousands(value: float) -> str:
    return f"{round(value):,}".replace(",", " ")


def _share(value: float | None) -> str:
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
    print("\nПапка day22 в корпус не входит: её докстринги разбирают контрольные вопросы числами.")

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


def retrieval() -> None:
    report = probes.compare()

    for note in report["notes"]:
        print(f"* {note}")
    if report["notes"]:
        print()

    print("## Три ретривера\n")
    print(
        f"Набор: {len(report['probes'])} эталонных отрывков day21, top-{report['top_k']}."
        f" Попадание — чанк из того же файла накрыл не меньше"
        f" {report['overlap_share']:.0%} эталонного отрывка.\n"
    )

    print("### recall@5 по способу спросить\n")
    print(
        _table(
            ["Ретривер"] + [probes.KIND_TITLES[kind] for kind in report["kinds"]],
            [
                [f"`{name}`"]
                + [
                    f"{report['results'][kind][name]['metrics']['recall@5']:.2f}"
                    for kind in report["kinds"]
                ]
                for name in report["retrievers"]
            ],
        )
    )

    for kind in report["kinds"]:
        print(f"\n### Проба — {probes.KIND_TITLES[kind]}\n")
        print(
            _table(
                ["Ретривер", "recall@1", "recall@3", "recall@5", "MRR@5", "Символов до ответа"],
                [
                    [
                        f"`{name}`",
                        *(
                            f"{report['results'][kind][name]['metrics'][key]:.2f}"
                            for key in ("recall@1", "recall@3", "recall@5", "mrr@5")
                        ),
                        _thousands(report["results"][kind][name]["metrics"]["chars_to_hit"]),
                    ]
                    for name in report["retrievers"]
                ],
            )
        )


def ask(question: str, retriever: str) -> None:
    answers = asyncio.run(answer.both(question, retriever=retriever))
    context = answers["rag"].context

    print(f"## Вопрос: «{question}»\n")
    print(
        f"Поиск — {retrieve.RETRIEVER_TITLES[retriever]}:"
        f" {len(context.sources)} выдержек, {context.tokens} токенов,"
        f" {context.seconds * 1000:.1f} мс\n"
    )

    for number, source in enumerate(context.sources, start=1):
        print(f"  [{number}] {source.path} · {source.section}")

    for mode in answer.MODES:
        given = answers[mode]
        print(f"\n### {answer.MODE_TITLES[mode]}\n")
        print(given.text)
        print(
            f"\n_{given.reply.prompt_tokens} токенов запроса,"
            f" {given.reply.completion_tokens} ответа, {given.reply.seconds:.1f} с"
            + (f", подкреплено: {', '.join(given.cited_paths)}" if given.cited_paths else "")
            + "_"
        )


def questions(retriever: str) -> None:
    report = asyncio.run(evaluate.run(retriever))

    for note in report["notes"]:
        print(f"* {note}")
    if report["notes"]:
        print()

    modes = report["modes"]
    titles = report["mode_titles"]

    print("## Контрольный набор\n")
    print(
        f"{len(report['questions'])} вопросов, поиск — {report['retriever_title']},"
        f" top-{report['top_k']}, модель {report['model']}.\n"
    )

    print(
        _table(
            ["Режим", "Факты", "Источник найден", "Ответ сослался", "Судья, 0–2",
             "На 2 балла", "Отказы «вне базы»", "Токенов запроса"],
            [
                [
                    f"**{titles[mode]}**",
                    _share(report["summary"][mode]["facts"]),
                    _share(report["summary"][mode]["sources"]),
                    _share(report["summary"][mode]["cited"]),
                    _share(report["summary"][mode]["judge"]),
                    f"{report['summary'][mode]['judge_full']} из {report['summary'][mode]['questions']}",
                    f"{report['summary'][mode]['refused_outside']} из {report['summary'][mode]['outside']}",
                    _thousands(report["summary"][mode]["prompt_tokens"]),
                ]
                for mode in modes
            ],
        )
    )

    print("\n### По типам вопросов\n")
    kinds = list(report["by_kind"][modes[0]])
    print(
        _table(
            ["Тип", "Вопросов"]
            + [f"{titles[mode]}, судья" for mode in modes]
            + [f"{titles[mode]}, факты" for mode in modes],
            [
                [kind, str(report["by_kind"][modes[0]][kind]["questions"])]
                + [_share(report["by_kind"][mode][kind]["judge"]) for mode in modes]
                + [_share(report["by_kind"][mode][kind]["facts"]) for mode in modes]
                for kind in kinds
            ],
        )
    )

    print("\n### Повопросно\n")
    print(
        _table(
            ["#", "Тип", "Вопрос", "Судья без RAG", "Судья с RAG", "Факты без", "Факты с", "Источник"],
            [
                [
                    str(plain["question_id"]),
                    plain["kind"],
                    plain["question"],
                    str(plain["judge"] if plain["judge"] is not None else "—"),
                    str(rag["judge"] if rag["judge"] is not None else "—"),
                    _share(plain["facts_share"]),
                    _share(rag["facts_share"]),
                    "—" if rag["sources_hit"] is None else ("да" if rag["sources_hit"] else "мимо"),
                ]
                for plain, rag in zip(
                    report["graded"][modes[0]], report["graded"][modes[1]], strict=True
                )
            ],
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогон RAG day22")
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=("all", "corpus", "build", "check", "retrieval", "ask", "questions"),
    )
    parser.add_argument("question", nargs="?", default="Зачем проекту ключ OpenRouter?")
    parser.add_argument("--retriever", default=retrieve.DEFAULT, choices=retrieve.RETRIEVERS)
    args = parser.parse_args()

    match args.step:
        case "corpus":
            show_corpus()
        case "build":
            show_corpus()
            print()
            build()
        case "check":
            check()
        case "retrieval":
            retrieval()
        case "ask":
            ask(args.question, args.retriever)
        case "questions":
            questions(args.retriever)
        case _:
            show_corpus()
            print()
            build()
            print()
            check()
            print()
            retrieval()
            print()
            questions(args.retriever)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyboardInterrupt) as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
