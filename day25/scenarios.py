"""Прогон из терминала: два сценария в трёх режимах — сразу в markdown.

    python day25/scenarios.py                # всё: корпус, индекс, оба сценария, таблицы
    python day25/scenarios.py corpus         # состав корпуса, без модели и без ключа
    python day25/scenarios.py build          # индексация, без модели и без ключа
    python day25/scenarios.py dialogs        # разметка обоих сценариев, без модели
    python day25/scenarios.py run            # оба сценария в трёх режимах
    python day25/scenarios.py run memory     # один сценарий в трёх режимах
    python day25/scenarios.py dialog memory  # один сценарий ход за ходом, режим tracked
    python day25/scenarios.py sweep          # развертка по порогу отказа на сценарии
    python day25/scenarios.py goal           # удержание цели: журнал, сверка, отказы кода
    python day25/scenarios.py chat           # интерактивный чат в терминале

Таблицы отсюда уезжают в README без правок. Ключ нужен всему, кроме `corpus`,
`build` и `dialogs`.
"""

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

import chat
import corpus
import dialogs
import evaluate
import index
import state
import storage
from corpus import SOURCE_TITLES
from dialogs import Dialogue
from evaluate import Run


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _thousands(value: float) -> str:
    return f"{round(value):,}".replace(",", " ")


def _share(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"

def _score(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _cut(text: str, limit: int = 80) -> str:
    body = " ".join(text.split())
    return body if len(body) <= limit else body[: limit - 1] + "…"


def _mark(value: bool | None) -> str:
    if value is None:
        return "—"
    return "да" if value else "**нет**"


def _bold(value: str, best: bool) -> str:
    return f"**{value}**" if best else value


# --- корпус и индекс ---------------------------------------------------------


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
        "\nИз корпуса исключены четыре папки: своя, day22, day23 и day24 —"
        " их README разбирают контрольные вопросы вместе с ответами."
    )

    for note in notes:
        print(f"\nМимо корпуса: {note}")


def build() -> None:
    documents = corpus.load()
    print(f"Индексация: {len(documents)} документов, модель {index.embed.MODEL_NAME}\n")

    row = index.build(documents)
    print("## Цена индекса\n")
    print(
        _table(
            ["Чанков", "Медиана, ток.", "p95", "Макс.", "Чанкинг", "Эмбеддинг", "FTS5", "Векторы"],
            [
                [
                    str(row["chunks"]),
                    str(row["median_tokens"]),
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


def ensure_index() -> None:
    if index.built() is None:
        print("Индекса нет, собираю.\n")
        build()
        print()


# --- сценарии ----------------------------------------------------------------


def show_dialogs() -> None:
    print("## Сценарии\n")
    print(
        _table(
            ["Сценарий", "Реплик", "Цель", "Ссылок", "Проб", "Рамок"],
            [
                [
                    f"`{dialogue.key}`",
                    str(dialogue.length),
                    dialogue.goal,
                    str(len(dialogue.of(dialogs.REFERENTIAL))),
                    str(len(dialogue.of(dialogs.PROBE))),
                    str(sum(1 for say in dialogue.says if say.framed)),
                ]
                for dialogue in dialogs.every()
            ],
        )
    )

    for dialogue in dialogs.every():
        print(f"\n### `{dialogue.key}` — {dialogue.title}\n")
        print(
            _table(
                ["#", "Род", "Реплика", "Ждём файл", "Рамка"],
                [
                    [
                        str(number),
                        dialogs.KIND_TITLES[say.kind],
                        _cut(say.text, 70),
                        ", ".join(f"`{path}`" for path in say.expect_paths) or "—",
                        ", ".join(
                            [f"есть «{word}»" for word in say.must_say]
                            + [f"нет «{word}»" for word in say.must_not_say]
                        )
                        or "—",
                    ]
                    for number, say in enumerate(dialogue.says, start=1)
                ],
            )
        )


# --- таблицы замера ----------------------------------------------------------


def modes_table(runs: list[Run]) -> None:
    folded = evaluate.by_mode(runs)
    best_goal = max((row["goal"] or 0) for row in folded.values())
    best_resolved = max((row["resolved"] or 0) for row in folded.values())

    print("## Три режима памяти\n")
    print(
        _table(
            [
                "Режим",
                "Ответов с источниками",
                "Цитата дословна",
                "Ссылка разрешена",
                "Рамка соблюдена",
                "Судья цели, 0–2",
                "На 2 балла",
                "Пробы, 0–2",
                "Отказов",
                "Лучшая выдержка",
            ],
            [
                [
                    row["title"],
                    f"{_share(row['answered_sources'])} ({row['answered']} отв.)",
                    _share(row["exact"]),
                    _bold(
                        f"{_share(row['resolved'])} ({row['resolved_count']} из {row['referential']})",
                        row["resolved"] == best_resolved,
                    ),
                    _share(row["framed"]),
                    _bold(_score(row["goal"]), row["goal"] == best_goal),
                    f"{row['goal_two']} из {row['goal_scored']}",
                    _score(row["probe_goal"]),
                    f"{row['refused']} из {row['turns']}",
                    _share(row["best"]),
                ]
                for row in folded.values()
            ],
        )
    )


def kinds_table(runs: list[Run]) -> None:
    folded = evaluate.by_kind(runs)

    print("\n## Судья цели по роду реплики\n")
    print(
        _table(
            ["Род реплики", "Ходов", *(chat.MODE_TITLES[mode] for mode in chat.MODES)],
            [
                [
                    dialogs.KIND_TITLES[kind],
                    str(next(iter(row.values()))["turns"]),
                    *(_score(row.get(mode, {}).get("goal")) for mode in chat.MODES),
                ]
                for kind, row in folded.items()
            ],
        )
    )


def goal_table(runs: list[Run]) -> None:
    print("\n## Цель: что от неё осталось к последнему ходу\n")
    print(
        _table(
            ["Сценарий", "Режим", "Цель в памяти", "Сверка", "Код не дал сменить"],
            [
                [
                    f"`{run.dialogue}`",
                    chat.MODE_TITLES[run.mode],
                    _cut(run.state.goal.text, 60) if run.state.goal else "—",
                    run.goal_verdict,
                    str(sum(1 for turn in run.turns if turn.kept_goal)),
                ]
                for run in runs
                if chat.memory_of(run.mode).tracked
            ],
        )
    )

    for run in runs:
        if chat.memory_of(run.mode).tracked and run.goal_why:
            print(f"\n`{run.dialogue}`: {run.goal_why}")


def cost_table(runs: list[Run]) -> None:
    folded = evaluate.by_mode(runs)

    print("\n## Цена памяти\n")
    print(
        _table(
            [
                "",
                *(row["title"] for row in folded.values()),
            ],
            [
                [name, *(fmt(row[key]) for row in folded.values())]
                for name, key, fmt in (
                    ("Токенов запроса", "prompt_tokens", _thousands),
                    ("Токенов ответа", "completion_tokens", _thousands),
                    ("Токенов на этапы", "stage_tokens", _thousands),
                    ("Из них память, оценка", "memory_tokens", _thousands),
                    ("Токенов на ход", "total_tokens", _thousands),
                    ("Секунд на ход", "seconds", lambda value: f"{value:.1f}"),
                    ("Переписано реплик", "rewritten", lambda value: f"{value:.0f}"),
                )
            ],
        )
    )


def state_block(run: Run) -> None:
    print(f"\n### Память задачи после {run.length}-го хода — `{run.dialogue}`\n")
    if run.state.goal:
        print(f"**Цель:** {run.state.goal.text}\n")
    rows = [
        [str(fact.id), fact.section, fact.text, str(fact.turn)] for fact in run.state.facts
    ]
    if rows:
        print(_table(["#", "Раздел", "Пункт", "Ход"], rows))
    else:
        print("Пунктов не записано.")


def turns_table(run: Run) -> None:
    print(f"\n### Ходы — `{run.dialogue}`, {chat.MODE_TITLES[run.mode]}\n")
    print(
        _table(
            ["#", "Род", "Реплика", "Искали по", "Источники", "Рамка", "Цель"],
            [
                [
                    str(turn.number),
                    dialogs.KIND_TITLES[turn.kind],
                    _cut(turn.question, 44),
                    _cut(turn.standalone, 52) if turn.changed else "как есть",
                    ", ".join(f"`{path}`" for path in turn.paths) or "**нет**",
                    _mark(turn.framed_ok),
                    "—" if turn.goal is None else str(turn.goal),
                ]
                for turn in run.turns
            ],
        )
    )


def sweep_table(rows: list[dict[str, object]], key: str) -> None:
    print(f"\n## Развертка по порогу отказа — `{key}`, {chat.MODE_TITLES[chat.MODES[-1]]}\n")
    print(
        _table(
            ["Порог", "Отказов", "Из них кодом", "Из них моделью", "Источники есть", "Цитата дословна", "Выдумано"],
            [
                [
                    f"{row['threshold']:.2f}",
                    f"{row['refused']} из {row['turns']}",
                    str(row["by_code"]),
                    str(row["by_model"]),
                    _share(row["sources"]),
                    _share(row["exact"]),
                    str(row["fabricated"]),
                ]
                for row in rows
            ],
        )
    )


def misses(runs: list[Run]) -> None:
    """Повопросный разбор просадок: без него доли ничего не объясняют."""
    rows: list[list[str]] = []
    for run in runs:
        for turn in run.turns:
            if turn.resolved_ok is False or turn.framed_ok is False or turn.goal == 0:
                reason = []
                if turn.resolved_ok is False:
                    reason.append("ссылка не разрешена")
                if turn.missing:
                    reason.append("нет " + ", ".join(f"«{word}»" for word in turn.missing))
                if turn.forbidden:
                    reason.append("есть " + ", ".join(f"«{word}»" for word in turn.forbidden))
                if turn.goal == 0:
                    reason.append("мимо цели")
                rows.append(
                    [
                        f"`{run.dialogue}`",
                        chat.MODE_TITLES[run.mode],
                        f"#{turn.number}",
                        _cut(turn.question, 40),
                        "; ".join(reason),
                        _cut(turn.why, 60),
                    ]
                )

    print("\n## Где просадки\n")
    if not rows:
        print("Ни одной: все ссылки разрешены, рамки соблюдены, мимо цели ни одного хода.")
        return
    print(_table(["Сценарий", "Режим", "Ход", "Реплика", "Что не так", "Судья"], rows))


# --- шаги --------------------------------------------------------------------


async def run_all(keys: tuple[str, ...], threshold: float | None) -> list[Run]:
    ensure_index()
    with tempfile.TemporaryDirectory() as temp:
        store = storage.Storage(Path(temp) / "scenarios.db")
        runs = await evaluate.every(keys=keys, store=store, threshold=threshold)

    order = {(key, mode): position for position, (key, mode) in enumerate(
        (key, mode) for key in keys for mode in chat.MODES
    )}
    return sorted(runs, key=lambda run: order[(run.dialogue, run.mode)])


async def step_run(keys: tuple[str, ...], threshold: float | None) -> list[Run]:
    runs = await run_all(keys, threshold)

    modes_table(runs)
    kinds_table(runs)
    goal_table(runs)
    cost_table(runs)
    misses(runs)

    print("\n## По сценариям\n")
    print(
        _table(
            ["Сценарий", "Режим", "Реплик", "Источники", "Цитата дословна", "Судья цели", "Отказов"],
            [
                [
                    f"`{row['dialogue']}`",
                    str(row["mode_title"]),
                    str(row["turns"]),
                    _share(row["sources"]),
                    _share(row["exact"]),
                    _score(row["goal"]),
                    str(row["refused"]),
                ]
                for row in (run.summary() for run in runs)
            ],
        )
    )
    return runs


async def step_dialog(key: str, threshold: float | None) -> None:
    ensure_index()
    dialogue = dialogs.load(key)

    with tempfile.TemporaryDirectory() as temp:
        store = storage.Storage(Path(temp) / "dialog.db")
        runs = await evaluate.every(
            keys=(key,), modes=chat.MODES, store=store, threshold=threshold
        )

    for mode in chat.MODES:
        run = next(run for run in runs if run.mode == mode)
        turns_table(run)

    tracked = next(run for run in runs if run.mode == "tracked")
    state_block(tracked)

    print(f"\n### Ответы — `{dialogue.key}`, {chat.MODE_TITLES['tracked']}\n")
    for turn in tracked.turns:
        print(f"**#{turn.number} [{dialogs.KIND_TITLES[turn.kind]}]** {turn.question}\n")
        print(f"{turn.text}\n")
        print(f"*источники: {', '.join(f'`{path}`' for path in turn.paths) or 'нет'}*\n")


async def step_sweep(key: str) -> None:
    ensure_index()
    with tempfile.TemporaryDirectory() as temp:
        store = storage.Storage(Path(temp) / "sweep.db")
        rows = await evaluate.sweep(key=key, store=store)
    sweep_table(rows, key)


async def step_goal(threshold: float | None) -> None:
    runs = await run_all(dialogs.KEYS, threshold)
    goal_table(runs)

    print("\n## Журнал цели\n")
    rows: list[list[str]] = []
    for run in runs:
        if not chat.memory_of(run.mode).tracked:
            continue
        for turn in run.turns:
            if turn.kept_goal:
                rows.append(
                    [f"`{run.dialogue}`", f"#{turn.number}", turn.kind, turn.kept_goal]
                )
    if rows:
        print(_table(["Сценарий", "Ход", "Род", "Почему цель осталась"], rows))
    else:
        print("Код ни разу не пришлось вмешиваться: модель цель не переписывала.")

    print("\n## Что попало в память\n")
    for run in runs:
        if chat.memory_of(run.mode).tracked:
            state_block(run)


async def step_chat(mode: str, threshold: float | None) -> None:
    """Интерактивный чат в терминале: тот же `Dialog`, что и на странице."""
    ensure_index()
    store = storage.Storage()
    await store.init()
    dialog = await chat.Dialog.open(store, mode, "терминал")

    print(f"Чат, режим «{chat.MODE_TITLES[mode]}». Пустая строка — выход.\n")
    while True:
        try:
            question = input("вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break

        turn = await dialog.ask(question, threshold)
        print(f"\n{turn.text}\n")
        if turn.paths:
            print("источники:")
            for claim, checked in zip(
                turn.claims, (turn.report.checked if turn.report else []), strict=False
            ):
                print(f"  · {checked.path or '?'} — {checked.verdict}: {_cut(claim.quote, 70)}")
        if turn.clarify:
            print(f"уточнение: {turn.clarify}")
        if turn.state.goal:
            print(f"\nцель: {turn.state.goal.text}")
        for fact in turn.state.facts:
            print(f"  [{fact.id}] {fact.section}: {fact.text}")
        print(f"\n({turn.total_tokens} токенов, {turn.seconds:.1f} с)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогон чата с RAG и памятью задачи, day25")
    parser.add_argument(
        "step",
        nargs="?",
        default="all",
        choices=("all", "corpus", "build", "dialogs", "run", "dialog", "sweep", "goal", "chat"),
    )
    parser.add_argument("key", nargs="?", default="", help="сценарий: memory или mcp")
    parser.add_argument("--gate", type=float, default=None, help="порог отказа")
    parser.add_argument(
        "--mode", default="tracked", choices=chat.MODES, help="режим памяти для `chat`"
    )
    args = parser.parse_args()

    keys = (args.key,) if args.key else dialogs.KEYS
    for key in keys:
        dialogs.load(key)

    match args.step:
        case "corpus":
            show_corpus()
        case "build":
            build()
        case "dialogs":
            show_dialogs()
        case "run":
            asyncio.run(step_run(keys, args.gate))
        case "dialog":
            asyncio.run(step_dialog(args.key or dialogs.KEYS[0], args.gate))
        case "sweep":
            asyncio.run(step_sweep(args.key or dialogs.KEYS[0]))
        case "goal":
            asyncio.run(step_goal(args.gate))
        case "chat":
            asyncio.run(step_chat(args.mode, args.gate))
        case _:
            show_corpus()
            print()
            build()
            print()
            show_dialogs()
            print()
            asyncio.run(step_run(keys, args.gate))
            asyncio.run(step_sweep(keys[0]))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        sys.exit(f"Не вышло: {error}")
