"""Markdown отчёта: строки, а не печать.

Отчёт собирают двое: сценарий печатает свои таблицы в stdout, сравнение отдаёт их
страницей в ответе на запрос. Печатать здесь нельзя ни тому, ни другому, поэтому
хелперы возвращают строки, а куда они пойдут — решает вызывающий.
"""

from typing import Any

DASH = "—"


def oneline(text: str) -> str:
    return " ".join(text.split())


def cell(value: Any) -> str:
    # Таблица идёт прямиком в README, а в ответах модели и судьи попадаются
    # вертикальные черты: неэкранированная сломала бы разметку.
    return str(value).replace("|", "\\|")


def table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> list[str]:
    """Таблица вместе с пустой строкой после неё: следующий блок начинается с чистого."""
    lines = [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join('---' for _ in headers)} |",
    ]
    lines.extend(f"| {' | '.join(cell(value) for value in row)} |" for row in rows)
    lines.append("")
    return lines


def money(value: float) -> str:
    return f"${value:.6f}"


def percent(before: int | float, after: int | float) -> str:
    if not before:
        return DASH
    return f"{(after - before) / before * 100:+.1f}%"
