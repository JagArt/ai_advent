"""Прогон пайплайна из терминала: сборка индексов и сравнение стратегий в markdown.

    python day21/scenarios.py              # корпус, индексы, сравнение
    python day21/scenarios.py corpus       # только сводка корпуса
    python day21/scenarios.py build        # только индексация, без модели и без ключа
    python day21/scenarios.py probes       # заново придумать набор проб
    python day21/scenarios.py compare      # только сравнение по готовому набору
    python day21/scenarios.py search "как считаются токены"

Вывод сразу markdown: таблицы отсюда уезжают в README без правок. Ключ нужен
только шагу `probes` — индексация, поиск и сравнение работают без модели.
"""

import argparse
import asyncio
import sys

import chunking
import corpus
import evaluate
import index
from corpus import SOURCE_TITLES


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _thousands(value: float) -> str:
    return f"{round(value):,}".replace(",", " ")


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

    for note in notes:
        print(f"\nМимо корпуса: {note}")

    if not any(document.source == "pdf" for document in documents):
        print("\nPDF в корпусе нет: положите файлы в `day21/corpus/`, они подхватятся сами.")


def build() -> None:
    documents = corpus.load()
    print(f"Индексация: {len(documents)} документов, модель {index.embed.MODEL_NAME}\n")

    rows = []
    for strategy in chunking.STRATEGIES:
        row = index.build(strategy, documents)
        print(
            f"  {strategy:11} {row['chunks']:5} чанков,"
            f" чанкинг {row['chunk_seconds']:5.1f} с,"
            f" эмбеддинг {row['embed_seconds']:5.1f} с,"
            f" векторы {row['vector_bytes'] / 1e6:.2f} МБ"
        )
        rows.append(row)

    print("\n## Цена индекса\n")
    print(
        _table(
            ["Стратегия", "Что режет", "Чанков", "Медиана, ток.", "p95", "Макс.", "Чанкинг", "Эмбеддинг", "Векторы"],
            [
                [
                    f"`{row['strategy']}`",
                    chunking.STRATEGY_TITLES[row["strategy"]],
                    str(row["chunks"]),
                    f"{row['median_tokens']:.0f}",
                    str(row["p95_tokens"]),
                    str(row["max_tokens"]),
                    f"{row['chunk_seconds']:.1f} с",
                    f"{row['embed_seconds']:.1f} с",
                    f"{row['vector_bytes'] / 1e6:.2f} МБ",
                ]
                for row in rows
            ],
        )
    )


def probes() -> None:
    print("Придумываю пробы по случайным отрывкам корпуса…\n")
    items = asyncio.run(evaluate.generate())
    evaluate.save(items)

    sources = {document.path: document.source for document in corpus.load()}
    counted: dict[str, int] = {}
    for item in items:
        source = sources.get(item.path, "?")
        counted[source] = counted.get(source, 0) + 1

    parts = ", ".join(f"{SOURCE_TITLES.get(key, key)}: {value}" for key, value in sorted(counted.items()))
    print(f"Проб: {len(items)} ({parts}), сохранены в probes.json\n")

    for item in items[:6]:
        print(f"  {item.path} «{item.section[:40]}»")
        print(f"    запрос: {item.query}")
        print(f"    вопрос: {item.question}")


def _metrics_table(report: dict, kind: str) -> str:
    results = report["results"][kind]
    return _table(
        ["Стратегия", "Чанков", "Медиана, ток.", "recall@1", "recall@3", "recall@5", "MRR@5", "Символов до ответа"],
        [
            [
                f"`{strategy}`",
                str(results[strategy]["index"]["chunks"]),
                f"{results[strategy]['index']['median_tokens']:.0f}",
                f"{results[strategy]['metrics']['recall@1']:.2f}",
                f"{results[strategy]['metrics']['recall@3']:.2f}",
                f"{results[strategy]['metrics']['recall@5']:.2f}",
                f"{results[strategy]['metrics']['mrr@5']:.2f}",
                _thousands(results[strategy]["metrics"]["chars_to_hit"]),
            ]
            for strategy in report["strategies"]
        ],
    )


def compare() -> None:
    report = evaluate.compare()

    for note in report["notes"]:
        print(f"* {note}")
    if report["notes"]:
        print()

    print("## Сравнение стратегий\n")
    print(
        f"Набор: {len(report['probes'])} эталонных отрывков, top-{report['top_k']}."
        f" Попадание — чанк из того же файла накрыл не меньше"
        f" {report['overlap_share']:.0%} эталонного отрывка.\n"
    )

    print("### recall@5 по способу спросить\n")
    print(
        _table(
            ["Стратегия"] + [evaluate.KIND_TITLES[kind] for kind in report["kinds"]],
            [
                [f"`{strategy}`"]
                + [
                    f"{report['results'][kind][strategy]['metrics']['recall@5']:.2f}"
                    for kind in report["kinds"]
                ]
                for strategy in report["strategies"]
            ],
        )
    )

    for kind in report["kinds"]:
        print(f"\n### Проба — {evaluate.KIND_TITLES[kind]}\n")
        print(_metrics_table(report, kind))

    sources = sorted(
        {
            source
            for strategy in report["strategies"]
            for source in report["results"]["query"][strategy]["by_source"]
        }
    )
    print("\n### recall@5 по источникам, поисковый запрос\n")
    print(
        _table(
            ["Стратегия"] + [SOURCE_TITLES.get(source, source) for source in sources],
            [
                [f"`{strategy}`"]
                + [
                    (
                        f"{slot[source]['recall@5']:.2f} ({slot[source]['probes']})"
                        if source in (slot := report["results"]["query"][strategy]["by_source"])
                        else "—"
                    )
                    for source in sources
                ]
                for strategy in report["strategies"]
            ],
        )
    )

    print("\n### Где стратегии разошлись, поисковый запрос\n")
    results = report["results"]["query"]
    shown = 0
    for position, probe in enumerate(report["probes"]):
        ranks = {
            strategy: results[strategy]["scored"][position]["hit_rank"]
            for strategy in report["strategies"]
        }
        if len({rank is not None for rank in ranks.values()}) < 2:
            continue

        found = ", ".join(
            f"{strategy}: {'#' + str(rank) if rank else 'мимо'}" for strategy, rank in ranks.items()
        )
        print(f"* «{probe['query']}» — `{probe['path']}` — {found}")
        shown += 1
        if shown >= 10:
            break


def search(query: str, limit: int) -> None:
    print(f"## Поиск: «{query}»\n")
    for strategy, hits in index.search_all(query, limit).items():
        print(f"**`{strategy}`**\n")
        print(
            _table(
                ["Близость", "Файл", "Раздел", "Токенов"],
                [
                    [f"{hit.score:.3f}", f"`{hit.path}`", hit.section, str(hit.tokens)]
                    for hit in hits
                ],
            )
        )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Пайплайн индексации day21")
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=("all", "corpus", "build", "probes", "compare", "search"),
    )
    parser.add_argument("query", nargs="?", default="как считаются токены запроса")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    match args.step:
        case "corpus":
            show_corpus()
        case "build":
            show_corpus()
            print()
            build()
        case "probes":
            probes()
        case "compare":
            compare()
        case "search":
            search(args.query, args.limit)
        case _:
            show_corpus()
            print()
            build()
            print()
            if not evaluate.PROBES_PATH.is_file():
                probes()
                print()
            compare()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, KeyboardInterrupt) as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
