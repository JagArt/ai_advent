"""Проверка лимитов шлюза: без модели и, отдельно, на живом сервисе.

    python day30/scenarios.py limits   # обрезка, частота, очередь, запрет открытого дефолтного ключа
    python day30/scenarios.py burst    # параллельные запросы к уже запущенному сервису

`limits` ничего не слушает и не ходит в LM Studio. `burst` ждёт `python day30/server.py`
и модель на порту LM Studio: он как раз про то, что очередь, 429 и потолок контекста
держатся на настоящем HTTP, а не только в функции.
"""

import argparse
import asyncio
import sys
import time

import httpx

import config
from limits import QueueFull, RateLimiter, SlotQueue, count_messages, trim

BLOB = "Правило номер семь: порт хранилища 8770. "


def expect(failures: list[str], name: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"- ok: {name}")
        return
    failures.append(name)
    suffix = f" — {detail}" if detail else ""
    print(f"- FAIL: {name}{suffix}")


def check_trim(failures: list[str]) -> None:
    short = [
        {"role": "system", "content": "Отвечай коротко."},
        {"role": "user", "content": "Какой порт?"},
    ]
    fitted = trim(short, max_context=200, max_tokens=40)
    expect(failures, "короткая история не режется", fitted.dropped == 0 and not fitted.truncated)
    expect(failures, "система и реплика на месте", fitted.messages == short)
    expect(failures, "оценка не выше бюджета", fitted.tokens <= 160 and fitted.tokens == count_messages(short))

    history: list[dict[str, str]] = [{"role": "system", "content": "Держи это правило: порт 8770."}]
    for index in range(12):
        history.append({"role": "user", "content": f"СТАРЫЙ-{index}. " + BLOB * 30})
        history.append({"role": "assistant", "content": f"ОТВЕТ-{index}. " + BLOB * 30})
    history.append({"role": "user", "content": "Какой порт у хранилища? ХВОСТ"})

    budget = 400
    max_tokens = 80
    cut = trim(history, max_context=budget + max_tokens, max_tokens=max_tokens)
    expect(failures, "длинная история теряет старые ходы", cut.dropped > 0, f"dropped={cut.dropped}")
    expect(failures, "после обрезки оценка внутри бюджета", cut.tokens <= budget, f"tokens={cut.tokens}")
    expect(failures, "системное сообщение осталось", any(item["content"].startswith("Держи это правило") for item in cut.messages))
    expect(failures, "хвост последней реплики остался", any("ХВОСТ" in item["content"] for item in cut.messages))
    expect(failures, "самый старый ход снят", all("СТАРЫЙ-0" not in item["content"] for item in cut.messages))

    huge = "НАЧАЛО " + ("слово " * 4000) + "КОНЕЦ"
    alone = trim(
        [{"role": "user", "content": huge}],
        max_context=300,
        max_tokens=50,
    )
    expect(failures, "одна длинная реплика укорачивается", alone.truncated and alone.dropped == 0)
    expect(failures, "у длинной реплики остаётся конец", alone.messages[-1]["content"].endswith("КОНЕЦ") or "КОНЕЦ" in alone.messages[-1]["content"][-40:])
    expect(failures, "начало длинной реплики срезано", "НАЧАЛО" not in alone.messages[-1]["content"])
    expect(failures, "укороченная реплика внутри бюджета", alone.tokens <= 250, f"tokens={alone.tokens}")

    # Влезает в окно целиком, но не во входе вместе с запасом на ответ — режется из-за запаса.
    reserved = trim(history, max_context=count_messages(history), max_tokens=200)
    expect(
        failures,
        "запас на ответ отнимает место у истории",
        reserved.dropped > 0 or reserved.truncated,
        f"dropped={reserved.dropped} truncated={reserved.truncated}",
    )


def check_rate(failures: list[str]) -> None:
    now = [0.0]
    limiter = RateLimiter(limit=3, window_s=60, clock=lambda: now[0])

    first = [limiter.check("day30-local") for _ in range(3)]
    expect(failures, "первые запросы окна проходят", all(item.allowed for item in first))
    expect(failures, "остаток убывает", [item.remaining for item in first] == [2, 1, 0])

    blocked = limiter.check("day30-local")
    expect(failures, "следующий запрос получает отказ", not blocked.allowed and blocked.remaining == 0)
    expect(failures, "у отказа есть Retry-After", blocked.retry_after > 0, f"retry={blocked.retry_after}")

    again = limiter.check("day30-local")
    expect(
        failures,
        "повтор отказа не отодвигает окно",
        not again.allowed and abs(again.retry_after - blocked.retry_after) < 1e-9,
    )
    expect(failures, "чужой ключ идёт своим окном", limiter.check("other").allowed and limiter.check("other").remaining == 1)

    now[0] = 60.0
    expect(failures, "после окна запрос снова проходит", limiter.check("day30-local").allowed)
    expect(failures, "peek не тратит попытку", limiter.peek("fresh") == 3 and limiter.peek("fresh") == 3)


async def check_queue(failures: list[str]) -> None:
    queue = SlotQueue(slots=1, depth=2)
    release = asyncio.Event()

    async def hold() -> None:
        await queue.enter()
        await release.wait()
        await queue.leave()

    holder = asyncio.create_task(hold())
    for _ in range(50):
        if queue.waiting == 0 and holder.done() is False and release.is_set() is False:
            # Слот занят, когда enter держателя уже вернулся: waiting снова 0, задача жива.
            if queue.busy():
                break
        await asyncio.sleep(0.01)
    else:
        expect(failures, "держатель занял слот", False)
        holder.cancel()
        return

    expect(failures, "держатель занял слот", queue.busy())

    first = asyncio.create_task(queue.enter())
    second = asyncio.create_task(queue.enter())
    for _ in range(50):
        if queue.waiting == 2:
            break
        await asyncio.sleep(0.01)
    expect(failures, "двое ждут в пределах глубины", queue.waiting == 2, f"waiting={queue.waiting}")

    try:
        await queue.enter()
    except QueueFull:
        expect(failures, "третий за пределом глубины получает отказ", True)
    else:
        expect(failures, "третий за пределом глубины получает отказ", False)
        await queue.leave()

    release.set()
    await holder
    waited = await first
    expect(failures, "ждавший запрос дождался слота", waited > 0, f"wait={waited}")
    await queue.leave()
    await second
    await queue.leave()


def check_bind(failures: list[str]) -> None:
    expect(failures, "0.0.0.0 с ключом по умолчанию запрещён", config.refuses_public_default("0.0.0.0", config.DEFAULT_API_KEY))
    expect(failures, ":: с ключом по умолчанию запрещён", config.refuses_public_default("::", config.DEFAULT_API_KEY))
    expect(failures, "loopback с ключом по умолчанию можно", not config.refuses_public_default("127.0.0.1", config.DEFAULT_API_KEY))
    expect(failures, "0.0.0.0 со своим ключом можно", not config.refuses_public_default("0.0.0.0", "secret"))


async def limits() -> None:
    failures: list[str] = []
    check_trim(failures)
    check_rate(failures)
    await check_queue(failures)
    check_bind(failures)
    if failures:
        sys.exit(f"Не сошлось проверок: {len(failures)}.")
    print(f"\nЛимиты сходятся, модель не вызывалась.")


def _header_int(response: httpx.Response, name: str) -> int:
    raw = response.headers.get(name)
    if raw is None:
        raise RuntimeError(f"В ответе нет заголовка {name}.")
    return int(raw)


async def _post(client: httpx.AsyncClient, messages: list[dict[str, str]], *, max_tokens: int) -> httpx.Response:
    return await client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {config.SERVICE_API_KEY}"},
        json={"model": config.MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": False},
    )


def _prompt_tokens(response: httpx.Response) -> int | None:
    if response.status_code != 200:
        return None
    usage = response.json().get("usage") or {}
    value = usage.get("prompt_tokens")
    return None if value is None else int(value)


async def burst() -> None:
    ping = [
        {"role": "system", "content": "Ответь одним словом: да."},
        {"role": "user", "content": "Скажи да."},
    ]
    timeout = httpx.Timeout(180, connect=5)
    async with httpx.AsyncClient(base_url=config.SERVICE_BASE_URL, timeout=timeout) as client:
        try:
            health = await client.get("/health")
            health.raise_for_status()
        except httpx.HTTPError as exc:
            sys.exit(f"Сервис на {config.SERVICE_BASE_URL} не отвечает ({exc}). Запустите `python day30/server.py`.")

        info = health.json()
        upstream = info.get("upstream") or {}
        if not upstream.get("online") or not upstream.get("model_available"):
            detail = upstream.get("error") or "модели нет в списке"
            sys.exit(f"LM Studio не готов: {detail}. Запустите `lms server start` и загрузите {config.MODEL}.")

        remaining_response = await client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {config.SERVICE_API_KEY}"},
        )
        remaining = _header_int(remaining_response, "x-ratelimit-remaining")
        limit = int(info["rate_limit"])
        depth = int(info["queue_limit"])
        slots = int(info["slots"])
        max_context = int(info["max_context"])
        print(
            f"Сервис: `{config.SERVICE_BASE_URL}`, модель `{info['model']}`, "
            f"слотов {slots}, очередь {depth}, частота {remaining}/{limit}, контекст {max_context}.\n"
        )

        print("## Страница без ключа\n")
        page = await client.get("/")
        is_html = page.status_code == 200 and "text/html" in page.headers.get("content-type", "")
        if not is_html:
            sys.exit(f"`GET /` вернул {page.status_code} {page.headers.get('content-type')}: страница не отдаётся.")
        page_chat = await client.post("/api/chat", json={"messages": [{"role": "user", "content": "Скажи да."}], "think": False})
        page_ok = page_chat.status_code == 200 and "event: done" in page_chat.text
        page_meta = "event: meta" in page_chat.text
        if not page_ok:
            sys.exit(f"`POST /api/chat` без ключа: {page_chat.status_code}, тело: {page_chat.text[:200]}")
        print(f"`GET /` — HTML, `POST /api/chat` без ключа — 200, в потоке meta: {'да' if page_meta else 'нет'}, done: да.\n")

        pile = slots + depth + 1
        if remaining < pile + 2:
            sys.exit(
                f"В окне частоты осталось {remaining}, для прогона нужно хотя бы {pile + 2}. "
                f"Подождите {info['rate_window_s']} с или перезапустите сервис."
            )

        print(f"## Очередь: {pile} запросов сразу, слот один, глубина {depth}\n")
        started = time.perf_counter()
        piled = await asyncio.gather(*[_post(client, ping, max_tokens=8) for _ in range(pile)])
        span_s = time.perf_counter() - started
        rows = []
        statuses: dict[int, int] = {}
        waits: list[int] = []
        for index, response in enumerate(piled, start=1):
            statuses[response.status_code] = statuses.get(response.status_code, 0) + 1
            wait = response.headers.get("x-queue-wait-ms", "—")
            if response.status_code == 200:
                waits.append(int(wait))
            rows.append(f"| {index} | {response.status_code} | {wait} | {response.elapsed.total_seconds() * 1000:.0f} |")
        print("| # | статус | очередь, мс | всего, мс |")
        print("| --- | --- | --- | --- |")
        print("\n".join(rows))
        print()
        ok = statuses.get(200, 0)
        rejected = statuses.get(503, 0)
        print(f"За стену прошло {span_s:.1f} с. Ответов 200: {ok}, отказов 503: {rejected}.")
        if ok != slots + depth or rejected != 1:
            sys.exit(f"Ожидались {slots + depth} ответа 200 и один 503, пришло {statuses}.")
        if max(waits) <= min(waits):
            sys.exit(f"Ожидания в очереди не разошлись: {waits}.")
        print(f"Ожидание в очереди от {min(waits)} до {max(waits)} мс — слот один, запросы прошли по очереди.\n")

        print("## Потолок контекста\n")
        history: list[dict[str, str]] = [{"role": "system", "content": "Отвечай числом."}]
        for index in range(20):
            history.append({"role": "user", "content": f"Ход {index}. " + BLOB * 40})
            history.append({"role": "assistant", "content": f"Ответ {index}. " + BLOB * 40})
        history.append({"role": "user", "content": "Какой порт? Ответь числом."})
        before = count_messages(history)
        fat = await _post(client, history, max_tokens=16)
        if fat.status_code != 200:
            sys.exit(f"Длинная история вернулась {fat.status_code}: {fat.text}")
        dropped = _header_int(fat, "x-context-dropped")
        context_tokens = _header_int(fat, "x-context-tokens")
        prompt_tokens = _prompt_tokens(fat)
        # Запас — max_tokens этого запроса, но не выше потолка ответа сервиса.
        reserved = min(16, int(info["max_tokens"]))
        budget = max_context - reserved
        print(f"Оценка истории до сервиса: {before} токенов.")
        print(
            f"Срезано сообщений: {dropped}. Оценка после обрезки: {context_tokens} "
            f"при бюджете {budget}. `prompt_tokens` модели: {prompt_tokens}."
        )
        if dropped <= 0:
            sys.exit("Сервис не срезал ни одного сообщения.")
        if context_tokens > budget:
            sys.exit(f"Оценка {context_tokens} выше бюджета промпта {budget}.")
        if prompt_tokens is None or prompt_tokens > max_context:
            sys.exit(f"`prompt_tokens` {prompt_tokens} выше потолка {max_context}.")
        print()

        print("## Частота\n")
        seen_429 = False
        passed = 0
        while not seen_429 and passed < limit + 2:
            response = await _post(client, ping, max_tokens=1)
            if response.status_code == 429:
                retry = response.headers.get("retry-after", "—")
                print(f"429 после ещё {passed} успешных коротких запросов. `Retry-After`: {retry} с." if passed != 1 else f"429 после ещё одного успешного короткого запроса. `Retry-After`: {retry} с.")
                seen_429 = True
                break
            if response.status_code != 200:
                sys.exit(f"Короткий запрос вернулся {response.status_code}: {response.text}")
            passed += 1
        if not seen_429:
            sys.exit("Окно частоты так и не закрылось.")

    print("\nЖивой сервис: очередь сериализует, контекст режется, лишняя частота получает 429.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка лимитов приватного сервиса локальной LLM")
    parser.add_argument("command", choices=("limits", "burst"))
    args = parser.parse_args()
    asyncio.run(limits() if args.command == "limits" else burst())


if __name__ == "__main__":
    main()
