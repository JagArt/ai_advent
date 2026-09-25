"""Реестр из терминала: рукопожатие со всеми серверами, каталог и длинный флоу.

В day19 клиент поднимал один сервер и знал про него всё. Здесь он знает только
реестр: какие серверы в нём объявлены, что у них за транспорт и какие инструменты
собрались в общий каталог. Дальше он просит реестр пройти контракт и печатает шаги
по мере их выполнения — семь вызовов на три сервера.

Ключ нужен: тезисы внутри `docs__summarize` делает модель.
"""

import asyncio
import sys
from typing import Any

import flow
import registry as reg

QUERY = "артефакты и пайплайн day19"


def _servers(info: dict[str, Any]) -> None:
    print("Реестр MCP-серверов")
    for server in info["servers"]:
        if not server["online"]:
            print(f"\n  {server['name']:<6} нет связи — {server['error']}")
            continue

        print(
            f"\n  {server['name']:<6} {server['transport']:<6} {server['endpoint']}"
            f"  ({server['elapsed_ms']} мс)"
        )
        print(f"    {server['server']}, протокол {server['protocol']}")
        print(f"    {server['lifetime']}")
        print(f"    инструменты: {', '.join(server['tools'])}")

    print(f"\nВ каталоге инструментов: {info['tools']}")
    for name, qualified in info["collisions"].items():
        print(f"  одноимённые: {name} — {', '.join(qualified)}")


def _catalog(info: dict[str, Any]) -> None:
    print("\nКаталог")
    for entry in info["catalog"]:
        needs = ", ".join(
            f"{need['arg']}=$from {need['kind']}" + ("?" if need["optional"] else "")
            for need in entry["needs"]
        )
        print(f"  {entry['qualified']:<24} → {entry['produces']:<8} {needs}")


def _step_line(frame: dict[str, Any]) -> None:
    if frame["stage"] == "started":
        print(f"  {frame['position']}. {frame['tool']:<22} пошёл — {frame['why']}")
        return

    if frame["status"] != "ok":
        print(f"     {frame['status']}: {frame['error']}")
        return

    refs = (
        ", ".join(f"{ref['arg']} ← #{ref['artifact_id']}.{ref['field']}" for ref in frame["refs"])
        or "аргументы запроса"
    )
    handle = frame["handle"]
    print(
        f"     готово за {frame['elapsed_ms']} мс: {refs}"
        f" → артефакт #{handle['artifact_id']} ({handle['kind']})"
    )


def _verdict(verdict: dict[str, Any]) -> None:
    print("\nПроверка порядка")
    print(f"  контракт:  {' → '.join(verdict['contract'])}")
    print(f"  как вышло: {' → '.join(verdict['actual'])}")
    print(f"  серверы:   {' → '.join(verdict['servers'])}")
    print(
        f"  порядок {'совпал' if verdict['order_ok'] else 'отклонился'},"
        f" совпало {verdict['matched']} из {len(verdict['contract'])},"
        f" вызовов {verdict['calls']}, отказов реестра {verdict['refused']}"
    )
    if verdict["missing"]:
        print(f"  не позвали: {', '.join(verdict['missing'])}")
    if verdict["extra"]:
        print(f"  лишние: {', '.join(verdict['extra'])}")


async def main() -> None:
    query = " ".join(sys.argv[1:]) or QUERY

    async with reg.connect() as registry:
        info = registry.info()
        _servers(info)
        _catalog(info)

        if len(registry.online) < len(info["servers"]):
            print("\nФлоу не запускается: в реестре не все серверы на связи.")
            return

        print(f"\nДлинный флоу по запросу {query!r}")
        task = flow.Task(query=query, limit=4, bullets=5)
        run_id = reg.open_run(query, "flow")
        result: dict[str, Any] = {}

        async for frame in flow.run(registry, task, run_id=run_id):
            if frame["event"] == "step":
                _step_line(frame)
            else:
                result = frame

    print(f"\nПрогон #{result['run_id']}: {result['status']}, {result['elapsed_ms']} мс")
    _verdict(result["verdict"])

    if result["file"]:
        print(f"\nФайл: {result['file']['path']}, {result['file']['bytes']} байт")
        print(f"  sha256: {result['file']['sha256']}")
    if result["check"]:
        check = result["check"]
        print(f"  сверка на диске: {'сошлось' if check['ok'] else 'не сошлось'}")
        print(f"  в журнале хранилища: {'есть' if check['in_journal'] else 'нет'}")
    if result["error"]:
        print(f"\nФлоу встал на {result['failed_at']}: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())
