"""Запуск сценария отдельным процессом: страница читает его вывод построчно.

Отчёт печатается десятками `print` в stdout, ход работы — в stderr. Перехватить
их внутри сервера можно было бы только подменой `sys.stdout`, а она глобальная на
все запросы, поэтому сценарий запускается тем же способом, что и из терминала:
`python -u day10/scenarios.py compare`. Заодно прогон не делит состояние с чатом,
а остановка сводится к kill.
"""

import asyncio
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

SCRIPT = Path(__file__).parent / "scenarios.py"

# Прогон идёт минуты и стоит денег: второй одновременный испортил бы замер и упёрся
# бы в рейт-лимиты, поэтому очереди нет — есть отказ.
BUSY_MESSAGE = "Прогон сценария уже идёт"


@dataclass(frozen=True)
class Line:
    """Строка отчёта из stdout — она же строка markdown для README."""

    text: str


@dataclass(frozen=True)
class Progress:
    """Ход работы из stderr: до конца прогона в stdout ничего нет."""

    text: str


@dataclass(frozen=True)
class Finished:
    """Конец прогона: код возврата и сколько всё это заняло."""

    exit_code: int
    seconds: float


Output = Line | Progress | Finished


class ScenarioBusy(RuntimeError):
    """Прогон уже идёт: второй запуск не встаёт в очередь, а получает отказ."""


class ScenarioRunner:
    """Один прогон за раз: процесс, два насоса на его потоки и kill при отмене."""

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._started = 0.0

    @property
    def busy(self) -> bool:
        # Завершившийся процесс занятым не считается: если поток отчёта до страницы
        # так и не дошёл, следующий запуск не должен упираться в его призрак.
        return self._process is not None and self._process.returncode is None

    async def start(self, name: str) -> None:
        """Процесс поднимается до ответа: отказ нужен со статусом, а не в потоке."""
        if self.busy:
            raise ScenarioBusy(BUSY_MESSAGE)

        self._started = time.monotonic()
        self._process = await asyncio.create_subprocess_exec(
            sys.executable,
            # -u: без буфера строки приходят по мере печати, а не пачкой в конце.
            "-u",
            str(SCRIPT),
            name,
            # Команда та же, что в README, — из корня проекта. Ключи API процесс
            # наследует из окружения: load_dotenv в сервере их уже положил.
            cwd=SCRIPT.parent.parent,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def stream(self) -> AsyncIterator[Output]:
        """Строки обоих потоков в порядке появления, в конце — код возврата."""
        process = self._process
        if process is None:
            raise RuntimeError("Сценарий не запущен")

        queue: asyncio.Queue[Output | None] = asyncio.Queue()
        pumps = [
            asyncio.create_task(self._pump(process.stdout, Line, queue)),
            asyncio.create_task(self._pump(process.stderr, Progress, queue)),
        ]

        try:
            # Потоков два, и каждый в конце кладёт в очередь None: пока не закрылись
            # оба, отчёт не дочитан.
            open_pumps = len(pumps)
            while open_pumps:
                item = await queue.get()
                if item is None:
                    open_pumps -= 1
                    continue
                yield item

            yield Finished(
                exit_code=await process.wait(),
                seconds=time.monotonic() - self._started,
            )
        finally:
            # Страница закрылась или пользователь остановил прогон: процесс сидит
            # в запросе к модели и без kill доиграет сценарий за наши деньги.
            for pump in pumps:
                pump.cancel()
            if process.returncode is None:
                process.kill()
            self._process = None

    @staticmethod
    async def _pump(
        reader: asyncio.StreamReader,
        wrap: type[Line] | type[Progress],
        queue: asyncio.Queue[Output | None],
    ) -> None:
        async for raw in reader:
            # errors="replace": оборванная на середине символа строка не должна
            # ронять прогон, который считался две минуты.
            await queue.put(wrap(text=raw.decode(errors="replace").rstrip("\n")))
        await queue.put(None)
