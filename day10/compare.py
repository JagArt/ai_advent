"""Сравнение прогонов: сведение уже посчитанных чисел, без единого запроса к модели.

Прогон сценария сам себя не сравнивает — он играет диалог, задаёт контрольные
вопросы, получает оценки судьи и складывает всё это в `RunResult`. Сравнение
приходит потом и работает только с этими результатами: страница присылает то, что
хранит в сессии, а здесь из них собираются те же markdown-таблицы, что печатает
сценарий, только с колонкой на каждый прогон.

Поэтому кнопка «Сравнить» бесплатна и срабатывает мгновенно, а прогоны можно
сравнивать в любом наборе — хоть чистый против фактов, хоть все четыре сразу.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

import tokens
from report import DASH, money, oneline, percent, table

# Порядок колонок в сравнении: чистый прогон — точка отсчёта, поэтому он первый,
# а не в том порядке, в каком его случилось прогнать.
ORDER = ("clean", "window", "facts", "branches")

# Максимум за контрольный вопрос: столько ставит судья, если названы все факты.
MAX_SCORE = 2


class RunTurn(BaseModel):
    """Один ход прогона: что ушло в модель и что пришло в счёте."""

    number: int
    # dialog или check: контрольные вопросы нумеруются отдельно от диалога.
    kind: str = "dialog"
    branch: str = ""
    context_messages: int
    facts_tokens: int = 0
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    cost_usd: float


class RunCheck(BaseModel):
    """Контрольный вопрос с оценкой судьи: оценка ставится внутри прогона."""

    question: str
    reference: str
    answer: str
    # -1 — судья ответил не по форме: оценки нет, но запрос оплачен.
    score: int
    comment: str = ""
    branch: str = ""


class RunResult(BaseModel):
    """Результат одного прогона — всё, что нужно, чтобы сравнить его с другим."""

    scenario: str
    label: str
    strategy: str
    model: str
    peak: bool
    # Отпечаток диалога и настроек: чистый прогон переносится между сессиями, и
    # сравнивать его с прогоном по другому диалогу нельзя.
    fingerprint: str
    window_messages: int
    budget: int
    turns: list[RunTurn]
    checks: list[RunCheck]
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    service_tokens: int = 0
    dialog_cost: float
    service_cost: float
    judge_cost: float
    total_cost: float

    @property
    def score(self) -> int:
        return sum(check.score for check in self.checks if check.score > 0)

    @property
    def max_score(self) -> int:
        return MAX_SCORE * len(self.checks)


class CompareRequest(BaseModel):
    """Что сравнивать: прогоны приходят со страницы, там они и хранятся."""

    runs: list[RunResult] = Field(min_length=2)


def ordered(runs: list[RunResult]) -> list[RunResult]:
    return sorted(runs, key=lambda run: (ORDER.index(run.scenario) if run.scenario in ORDER else len(ORDER)))


def miss_price(input_tokens: int) -> float:
    """Во сколько встал бы вход, если бы кэша префикса не существовало.

    Кэш и стратегия тянут счёт в разные стороны: картотека добавляет токенов, но
    стоит в начале запроса и потому попадает в кэш. Эта цена показывает эффект
    одной стратегии, без второго.
    """
    return tokens.cost_usd(0, input_tokens, 0)


def report(runs: list[RunResult]) -> list[str]:
    """Сводный отчёт по прогонам: тот же markdown, что печатает сценарий."""
    runs = ordered(runs)
    lines = ["## Сравнение прогонов", ""]
    lines.extend(intro(runs))
    lines.extend(report_turns(runs))
    lines.extend(report_totals(runs))
    lines.extend(report_quality(runs))
    lines.extend(verdict(runs))
    return lines


def intro(runs: list[RunResult]) -> list[str]:
    names = ", ".join(f"{run.label} (ходов: {len(run.turns)})" for run in runs)
    lines = [
        f"Сравниваются прогоны из сессии: {names}. Числа взяты из самих прогонов — "
        "сравнение ничего не пересчитывает и не спрашивает модель заново.",
        "",
    ]

    fingerprints = {run.fingerprint for run in runs}
    if len(fingerprints) > 1:
        # Чистый прогон живёт дольше сессии, а диалог и настройки замера могли за
        # это время поменяться: тогда колонки считают разное и сравнивать их нельзя.
        lines.extend(
            [
                "Внимание: прогоны сделаны по разным версиям диалога или настроек "
                "(отпечатки не совпали) — числа в колонках сравнимы не напрямую.",
                "",
            ],
        )

    models = {run.model for run in runs}
    if len(models) > 1:
        lines.extend([f"Внимание: прогоны шли на разных моделях: {', '.join(sorted(models))}.", ""])

    return lines


def report_turns(runs: list[RunResult]) -> list[str]:
    """Главная таблица: один и тот же ход во всех прогонах, вход против входа."""
    lines = ["### Вход по ходам", ""]

    # Ходы сводятся по номеру внутри своего вида, а не по месту в списке: прогоны
    # разной длины (Branching играет общее начало и две ветки) иначе разъехались
    # бы, и контрольный вопрос одного встал бы против диалога другого.
    columns = [{(turn.kind, turn.number): turn for turn in run.turns} for run in runs]
    keys = sorted(
        {key for column in columns for key in column},
        key=lambda key: (key[0] != "dialog", key[1]),
    )

    rows: list[tuple[Any, ...]] = []
    for key in keys:
        label = f"вопрос {key[1]}" if key[0] == "check" else str(key[1])
        rows.append((label, *(turn_cell(column.get(key)) for column in columns)))

    lines.extend(table(("Ход", *(run.label for run in runs)), rows))
    lines.append(
        "В ячейке — входные токены хода и контекст, из которого он собран. "
        "У Branching ходов меньше: после общего начала разговор идёт в двух ветках, "
        "и в ячейке подписано, в какой именно.",
    )
    lines.append("")
    return lines


def turn_cell(turn: RunTurn | None) -> str:
    if turn is None:
        return DASH
    note = f"{turn.prompt_tokens} / {turn.context_messages} сообщ."
    if turn.facts_tokens:
        note += f" + картотека {turn.facts_tokens} ток."
    if turn.branch:
        note += f" [{turn.branch}]"
    return note


@dataclass(frozen=True)
class Metric:
    """Строка таблицы итогов: как достать число из прогона и как его показать."""

    title: str
    value: Callable[[RunResult], int | float]
    format: Callable[[Any], str] = str
    # Проценты уместны там, где числа сравнимы: длина прогона задана сценарием, и
    # процент от неё ничего не объясняет.
    relative: bool = True


METRICS = (
    Metric("Ходов", lambda run: len(run.turns), relative=False),
    Metric("Входные токены", lambda run: run.input_tokens),
    Metric("из них из кэша", lambda run: run.cached_tokens),
    Metric("Выходные токены", lambda run: run.output_tokens),
    Metric("Токены служебных запросов", lambda run: run.service_tokens),
    Metric("Вход по цене без кэша", lambda run: miss_price(run.input_tokens), money),
    Metric("Стоимость диалога", lambda run: run.dialog_cost, money),
    Metric("Стоимость служебных запросов", lambda run: run.service_cost, money),
    Metric("Работа судьи", lambda run: run.judge_cost, money),
    Metric("Всего", lambda run: run.total_cost, money),
)


def report_totals(runs: list[RunResult]) -> list[str]:
    """Итоги: колонка на прогон, процент — от первого прогона в таблице."""
    lines = ["### Итоги прогонов", ""]

    base = runs[0]
    rows: list[tuple[Any, ...]] = []
    for metric in METRICS:
        baseline = metric.value(base)
        rows.append(
            (
                metric.title,
                *(
                    metric.format(metric.value(run))
                    if run is base or not metric.relative
                    else f"{metric.format(metric.value(run))} ({percent(baseline, metric.value(run))})"
                    for run in runs
                ),
            ),
        )

    lines.extend(table(("Показатель", *(run.label for run in runs)), rows))
    lines.append(f"Процент считается от прогона «{base.label}» — он в таблице точка отсчёта.")
    lines.append("")
    return lines


def report_quality(runs: list[RunResult]) -> list[str]:
    """Оценки судьи, поставленные внутри прогонов: здесь они только сводятся."""
    lines = ["### Качество на контрольных вопросах", ""]

    # Вопросы совпадают у прогонов с одним диалогом, а у Branching свои — и заданы
    # в каждой ветке. Ключ поэтому из вопроса и ветки, а порядок — как встретился.
    keys: list[tuple[str, str]] = []
    scores: dict[tuple[str, str], dict[str, RunCheck]] = {}
    for run in runs:
        for check in run.checks:
            key = (oneline(check.question), check.branch)
            if key not in scores:
                keys.append(key)
                scores[key] = {}
            scores[key][run.scenario] = check

    rows: list[tuple[Any, ...]] = []
    for question, branch in keys:
        label = f"{question} [{branch}]" if branch else question
        rows.append(
            (
                label,
                *(score_cell(scores[(question, branch)].get(run.scenario)) for run in runs),
            ),
        )

    rows.append(
        (
            "Сумма",
            *(f"{run.score} из {run.max_score}" for run in runs),
        ),
    )

    lines.extend(table(("Вопрос", *(run.label for run in runs)), rows))
    lines.append(
        "Оценку ставит модель-судья внутри прогона: 2 — все факты названы, "
        "1 — часть потеряна, 0 — фактов нет. Вопрос, которого в прогоне не было, "
        f"помечен «{DASH}».",
    )
    lines.append("")
    lines.extend(report_answers(runs))
    return lines


def score_cell(check: RunCheck | None) -> str:
    if check is None:
        return DASH
    if check.score < 0:
        return f"{DASH} (ответ судьи не разобран)"
    if not check.comment:
        return str(check.score)
    return f"{check.score} — {oneline(check.comment)[:80]}"


def report_answers(runs: list[RunResult]) -> list[str]:
    """Сами ответы: без них оценка судьи — цифра, которую нельзя проверить."""
    lines = ["#### Ответы агента", ""]

    rows: list[tuple[Any, ...]] = []
    for run in runs:
        for check in run.checks:
            question = oneline(check.question)
            rows.append(
                (
                    f"{question} [{check.branch}]" if check.branch else question,
                    run.label,
                    check.score if check.score >= 0 else DASH,
                    oneline(check.answer)[:200],
                ),
            )

    lines.extend(table(("Вопрос", "Прогон", "Оценка", "Ответ"), rows))
    return lines


def verdict(runs: list[RunResult]) -> list[str]:
    """Короткий вывод из чисел: кто дешевле, кто помнит и во сколько это встало."""
    cheapest = min(runs, key=lambda run: run.total_cost)
    # Вопросов в прогонах может быть разное число, поэтому доля, а не сумма.
    best = max(runs, key=lambda run: run.score / run.max_score if run.max_score else 0)

    lines = [
        f"Дешевле всех обошёлся «{cheapest.label}» — {money(cheapest.total_cost)}. "
        f"Больше всех фактов удержал «{best.label}» — {best.score} из {best.max_score}.",
        "",
    ]

    if best.total_cost > cheapest.total_cost:
        lines.extend(
            [
                f"Память стоит денег: «{best.label}» дороже «{cheapest.label}» на "
                f"{percent(cheapest.total_cost, best.total_cost)}.",
                "",
            ],
        )

    return lines
