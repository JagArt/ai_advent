"""Допуск запроса к локальной модели: частота, потолок контекста, глубина очереди.

Считается без сети и без LM Studio, чтобы прогон `scenarios.py limits` ловил
регрессию лимитов отдельно от того, отвечает ли модель. Оценка токенов — тот же
`cl100k_base`, что в day8: своего токенизатора Qwen библиотека не отдаёт, точное
число приходит позже в `usage` модели.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

import tiktoken

ENCODING = "cl100k_base"
MESSAGE_OVERHEAD = 4
REPLY_OVERHEAD = 3

_encoder = tiktoken.get_encoding(ENCODING)


def count_text(text: str) -> int:
    return len(_encoder.encode(text, disallowed_special=()))


def count_messages(messages: list[dict[str, str]]) -> int:
    """Оценка запроса целиком: текст плюс разметка роли на каждое сообщение."""
    total = REPLY_OVERHEAD
    for message in messages:
        total += MESSAGE_OVERHEAD + count_text(message.get("content") or "")
    return total


@dataclass(frozen=True)
class Trimmed:
    """История, которая уже влезает в бюджет промпта.

    `dropped` — сколько сообщений снято целиком. `truncated` — у оставшегося
    хвоста отрезали начало: последняя реплика важнее старой, поэтому режется
    голова, а не конец.
    """

    messages: list[dict[str, str]]
    dropped: int
    truncated: bool
    tokens: int


def trim(messages: list[dict[str, str]], *, max_context: int, max_tokens: int) -> Trimmed:
    """Укладывает историю в `max_context - max_tokens`.

    Запас `max_tokens` — место под ответ: иначе промпт занимает весь потолок,
    и генерация упирается в окно. Системные сообщения держатся до последнего.
    Из диалога снимаются самые старые ходы. Если одна реплика всё равно длиннее
    бюджета, остаётся её хвост.
    """
    budget = max_context - max_tokens
    if budget < 32:
        raise ValueError(
            f"Потолок контекста {max_context} не оставляет места под ответ в {max_tokens} токенов."
        )

    system = [_copy(message) for message in messages if message["role"] == "system"]
    rest = [_copy(message) for message in messages if message["role"] != "system"]
    dropped = 0

    # Последний ход не выбрасываем: если он сам длиннее бюджета, ниже останется его хвост.
    while len(rest) > 1 and count_messages(system + rest) > budget:
        rest.pop(0)
        dropped += 1

    while len(system) > 1 and count_messages(system + rest) > budget:
        system.pop(0)
        dropped += 1

    kept = system + rest
    truncated = False
    if count_messages(kept) > budget and kept:
        before = kept[-1]["content"]
        kept[-1] = _truncate_from_start(kept[-1], kept[:-1], budget)
        truncated = kept[-1]["content"] != before

    return Trimmed(messages=kept, dropped=dropped, truncated=truncated, tokens=count_messages(kept))


def _copy(message: dict[str, str]) -> dict[str, str]:
    return {"role": message["role"], "content": message.get("content") or ""}


def _truncate_from_start(message: dict[str, str], others: list[dict[str, str]], budget: int) -> dict[str, str]:
    """Самый длинный хвост `message`, с которым весь список укладывается в бюджет."""
    tokens = _encoder.encode(message["content"], disallowed_special=())
    best = ""
    lo = 0
    hi = len(tokens)
    while lo <= hi:
        mid = (lo + hi) // 2
        tail = _encoder.decode(tokens[-mid:]) if mid else ""
        candidate = tail if mid == len(tokens) else f"…{tail}"
        trial = {"role": message["role"], "content": candidate}
        if count_messages([*others, trial]) <= budget:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return {"role": message["role"], "content": best}


@dataclass(frozen=True)
class RateDecision:
    allowed: bool
    remaining: int
    retry_after: float


class RateLimiter:
    """Скользящее окно на ключ. Отклонённый запрос в окно не записывается.

    Иначе клиент, который повторяет запрос сразу после 429, отодвигал бы
    восстановление окна каждым повтором. Часы подставляются снаружи, чтобы
    прогон не ждал настоящие шестьдесят секунд.
    """

    def __init__(self, limit: int, window_s: float, clock: Callable[[], float] | None = None) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock or time.monotonic
        self._hits: dict[str, list[float]] = {}

    def _fresh(self, key: str, now: float) -> list[float]:
        hits = [moment for moment in self._hits.get(key, ()) if now - moment < self.window_s]
        self._hits[key] = hits
        return hits

    def peek(self, key: str) -> int:
        hits = self._fresh(key, self._clock())
        return max(self.limit - len(hits), 0)

    def check(self, key: str) -> RateDecision:
        now = self._clock()
        hits = self._fresh(key, now)
        if len(hits) >= self.limit:
            retry = 0.0 if not hits else self.window_s - (now - hits[0])
            return RateDecision(allowed=False, remaining=0, retry_after=retry)
        hits.append(now)
        self._hits[key] = hits
        return RateDecision(allowed=True, remaining=self.limit - len(hits), retry_after=0.0)


class QueueFull(Exception):
    """Ждущих уже столько, сколько разрешает глубина. Новый запрос не встаёт в хвост."""


class SlotQueue:
    """Один слот на модель: лишние запросы ждут, а не делят GPU.

    `depth` — сколько можно держать в ожидании, не считая того, кто уже
    генерирует. Сверх этого — `QueueFull`, сразу, без ожидания.
    """

    def __init__(self, slots: int, depth: int) -> None:
        if slots < 1:
            raise ValueError("Слотов должно быть хотя бы один.")
        if depth < 0:
            raise ValueError("Глубина очереди не бывает отрицательной.")
        self.slots = slots
        self.depth = depth
        self.waiting = 0
        self._admitted = 0
        self._semaphore = asyncio.Semaphore(slots)
        self._lock = asyncio.Lock()

    async def enter(self) -> float:
        """Занимает слот и возвращает, сколько секунд запрос простоял в очереди.

        В допуск входят и тот, кто уже генерирует, и те, кто ждёт: иначе задача,
        которая ещё не успела уменьшить счётчик после захвата слота, съедает
        место ожидания, и отказ приходит на запрос раньше.
        """
        async with self._lock:
            if self._admitted >= self.slots + self.depth:
                raise QueueFull()
            self._admitted += 1
            self.waiting += 1
        started = time.perf_counter()
        try:
            await self._semaphore.acquire()
        except BaseException:
            async with self._lock:
                self._admitted -= 1
                self.waiting -= 1
            raise
        async with self._lock:
            self.waiting -= 1
        return time.perf_counter() - started

    async def leave(self) -> None:
        async with self._lock:
            self._admitted -= 1
        self._semaphore.release()

    def busy(self) -> bool:
        return self._semaphore.locked()
