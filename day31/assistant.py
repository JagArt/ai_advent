"""Ассистент разработчика в терминале: `/help` отвечает на вопросы о проекте.

Без аргументов — диалог: ассистент поднимает MCP-сервер проекта, здоровается,
печатает, что знает о репозитории и какие модели на связи, и ждёт команд.
С аргументами — один вопрос, один ответ, выход: так удобно звать из скриптов
и так сняты расшифровки в README.

    python day31/assistant.py
    python day31/assistant.py "/help На каком порту слушает хранилище day20?"
    python day31/assistant.py --cloud "Как добавить новый инструмент в MCP-сервер day31?"

Команды диалога: `/help` без вопроса — что умеет; `/help <вопрос>` — ответ;
флаги `--local` и `--cloud` в конце вопроса перекрывают автоматический маршрут;
`/quit` — выход. Текст без `/` не уходит в модель: ассистент напоминает про
`/help` — команда здесь не украшение, а явное согласие потратить вызов модели.

Соединение с сервером одно на весь диалог, модели опрашиваются при старте и
перед каждым вопросом заново — LM Studio можно поднять, не перезапуская ассистента.
"""

import asyncio
import sys
from dataclasses import dataclass

import agent
import docs
import llm

PROMPT = "> "
FLAGS = {"--local": "local", "--cloud": "cloud"}
QUIT = {"/quit", "/exit", "/q"}

EXAMPLES = (
    "На каком порту слушает постоянный MCP-сервер хранилища day20?",
    "Какая сейчас ветка и что изменилось в рабочем дереве?",
    "Как устроены имена инструментов в реестре day20?",
    "Как бы ты спроектировал day32 — планировщик поверх MCP day18?",
)


@dataclass(frozen=True)
class Command:
    """Разобранная строка: что делать и с каким вопросом."""

    kind: str  # help | ask | quit | empty | other
    question: str = ""
    forced: str | None = None


def parse(line: str) -> Command:
    text = line.strip()
    if not text:
        return Command("empty")
    if text.lower() in QUIT:
        return Command("quit")

    words = text.split()
    forced = None
    rest: list[str] = []
    for word in words:
        if word.lower() in FLAGS:
            forced = FLAGS[word.lower()]
        else:
            rest.append(word)

    if rest and rest[0].lower() == "/help":
        rest = rest[1:]
        return Command("ask" if rest else "help", " ".join(rest), forced)
    if rest and rest[0].startswith("/"):
        return Command("other", " ".join(rest), forced)
    # Одиночный запуск принимает вопрос и без /help: аргументы командной строки — уже команда.
    return Command("plain", " ".join(rest), forced)


# --- вывод -------------------------------------------------------------------


def say(text: str = "") -> None:
    print(text, flush=True)


def seconds(ms: int | None) -> str:
    return "—" if ms is None else f"{ms / 1000:.1f} с"


def tool_line(result: agent.ToolResult) -> str:
    args = ", ".join(f"{key}={value!r}" for key, value in result.arguments.items())
    mark = "" if result.ok else " · отказ: " + result.text.splitlines()[0][:120]
    return f"  ⚙ {result.name}({args}) · {result.ms} мс{mark}"


def route_line(route: llm.Route) -> str:
    timing = f" ({route.ms / 1000:.1f} с)" if route.ms is not None else ""
    return f"→ модель: {route.model} · {route.title} — {route.reason}{timing}"


def context_lines(context: agent.Context) -> list[str]:
    state = context.state
    if context.branch.ok and (head := context.branch.payload.get("head")):
        # Человеку тема коммита полезна; модели она не уходит — см. `Context.state`.
        state = state.replace(f"коммит {head['short']}", f"коммит {head['short']} «{head['subject'][:60]}»", 1)
    lines = [f"→ проект: {state}"]
    if context.docs.ok:
        payload = context.docs.payload
        index = payload["index"]
        lines.append(
            f"→ документация: {payload['count']} разделов ({payload['strategy']}) · "
            f"в индексе файлов: {index['files']}, разделов: {index['sections']}"
        )
    else:
        lines.append(f"→ документация: {context.docs.text}")
    return lines


def availability_lines(availability: llm.Availability) -> list[str]:
    if availability.local_online:
        loaded = "модель загружена" if availability.local_model_loaded else f"модель {llm.LOCAL_MODEL} не загружена — `lms load {llm.LOCAL_MODEL}`"
        local = f"{llm.LOCAL_MODEL} · LM Studio {llm.LOCAL_BASE_URL} — на связи, {loaded}"
    else:
        local = f"{llm.LOCAL_MODEL} · LM Studio — {availability.local_error}"
    cloud = f"{llm.CLOUD_MODEL} · DeepSeek — " + ("ключ есть" if availability.cloud_configured else "нет DEEPSEEK_API_KEY")
    return [f"  локальная: {local}", f"  облачная:  {cloud}"]


def banner(connection: agent.Connection, availability: llm.Availability, branch: agent.ToolResult) -> None:
    stats = docs.reindex()
    names = ", ".join(tool.name for tool in connection.tools)
    say("AI Advent · ассистент разработчика")
    say(f"  MCP: {connection.server} · протокол {connection.protocol} · `{connection.command}` · {connection.elapsed_ms} мс")
    say(f"  инструменты ({len(connection.tools)}): {names}")
    say(f"  документация: файлов {stats['files']}, разделов {stats['sections']} · {docs.DB_PATH.name}")
    if branch.ok:
        head = branch.payload.get("head") or {}
        say(f"  ветка: {branch.payload['branch']} · {head.get('short', '')} «{head.get('subject', '')[:70]}»")
    else:
        say(f"  ветка: {branch.text}")
    for line in availability_lines(availability):
        say(line)
    say("Команды: /help — что умею · /help <вопрос> [--local|--cloud] · /quit")
    say()


def capabilities(connection: agent.Connection, availability: llm.Availability) -> None:
    """`/help` без вопроса: что ассистент знает и как решает, кому отвечать."""
    stats = docs.reindex()
    say("Отвечаю на вопросы о проекте AI Advent: структура папок, что делает день, как запустить, где порт или команда.")
    say()
    say("Откуда беру ответ:")
    say(f"  документация — README корня и README/TASK всех дней: файлов {stats['files']}, разделов {stats['sections']}; "
        "к вопросу подкладываются пять самых близких разделов")
    say(f"  проект через MCP `{connection.server}` — ветка и состояние дерева всегда, остальное модель зовёт сама:")
    for tool in connection.tools:
        first = (tool.description or "").strip().splitlines()[0]
        say(f"    {tool.name} — {first}")
    say()
    say("Какая модель отвечает — решает локальная модель одним словом:")
    say("  факт (где лежит, какой порт, как запустить, что изменилось) → локальная")
    say("  план (спроектировать, сравнить, почему так, как лучше) → облачная")
    say("  флаги --local и --cloud в конце вопроса перекрывают выбор")
    for line in availability_lines(availability):
        say(line)
    say()
    say("Примеры:")
    for example in EXAMPLES:
        say(f"  /help {example}")


async def ask(connection: agent.Connection, question: str, forced: str | None) -> None:
    availability = await llm.probe()
    try:
        route = await llm.route(question, availability, forced)
    except RuntimeError as exc:
        say(f"✗ {exc}")
        return

    say(route_line(route))
    answered = False
    try:
        async for event in agent.answer(connection, question, route):
            if isinstance(event, agent.Prefetched):
                for line in context_lines(event.context):
                    say(line)
                say()
            elif isinstance(event, agent.Chunk):
                if not answered:
                    answered = True
                print(event.text, end="", flush=True)
            elif isinstance(event, agent.ToolCalled):
                if answered:
                    print(flush=True)
                    answered = False
                say(tool_line(event.result))
            elif isinstance(event, agent.Done):
                if answered:
                    print(flush=True)
                if not event.text:
                    say("(модель не дала текста)")
                say()
                if event.sources:
                    say("Источники:")
                    for source in event.sources:
                        say(f"  [{source.number}] {source.path} — «{source.heading}»")
                tokens = (
                    f" · токены {event.prompt_tokens} → {event.completion_tokens}"
                    if event.prompt_tokens is not None
                    else ""
                )
                finish = "" if event.finish_reason in ("stop", None) else f" · {event.finish_reason}"
                say(
                    f"— {route.model} · раундов {event.rounds} · вызовов {event.calls} · "
                    f"первый токен {seconds(event.ttft_ms)} · всего {seconds(event.total_ms)}{tokens}{finish}"
                )
    except Exception as exc:  # noqa: BLE001
        if answered:
            print(flush=True)
        say(f"✗ {type(exc).__name__}: {exc}")


async def handle(connection: agent.Connection, command: Command, *, interactive: bool) -> bool:
    """Исполняет команду. Возвращает False, когда диалог пора закрывать."""
    if command.kind == "quit":
        return False
    if command.kind == "empty":
        return True
    if command.kind == "help":
        capabilities(connection, await llm.probe())
        return True
    if command.kind == "ask" or (command.kind == "plain" and not interactive):
        await ask(connection, command.question, command.forced)
        return True
    if command.kind == "plain":
        say("Вопросы — через /help: «/help " + command.question[:60] + "». Просто /help — что я умею.")
        return True
    say(f"Команды {command.question.split()[0]} нет. Есть /help, /help <вопрос>, /quit.")
    return True


async def repl(connection: agent.Connection) -> None:
    try:
        import readline  # noqa: F401  — история и редактирование строки в input()
    except ImportError:
        pass

    availability = await llm.probe()
    banner(connection, availability, await agent.call(connection.session, "git_branch"))

    while True:
        try:
            line = await asyncio.to_thread(input, PROMPT)
        except (EOFError, KeyboardInterrupt):
            say()
            break
        if not await handle(connection, parse(line), interactive=True):
            break
        say()


async def main(argv: list[str]) -> None:
    async with agent.connect() as connection:
        if argv:
            await handle(connection, parse(" ".join(argv)), interactive=False)
        else:
            await repl(connection)


if __name__ == "__main__":
    try:
        asyncio.run(main(sys.argv[1:]))
    except KeyboardInterrupt:
        pass
