"""Сверка цитаты с чанком: единственная проверка дня, которой не нужна модель.

Цитата — это утверждение о тексте, а не о мире: «вот такая строка стоит в
выдержке [2]». Проверяется оно поиском подстроки, и в этом вся его ценность.
Судья может ошибиться, механическая доля фактов — застрять на формулировке, а
здесь либо строка в чанке есть, либо её там нет, и прогон даёт один и тот же
ответ при любой температуре.

## Что считать «дословно»

Буквальное сравнение байтов провалилось бы на пустом месте. Модель переносит
`8770` из выдержки верно, но пишет `127.0.0.1:8770` слитно там, где в тексте
стоял неразрывный пробел; меняет «ё» на «е»; заменяет кавычки-ёлочки на прямые,
а длинное тире на дефис. Это не выдумка, а транскрипция, и штрафовать за неё
значило бы мерить типографику вместо подкреплённости.

Поэтому обе стороны приводятся к одному виду: пробелы схлопываются, регистр
снимается, ё становится е, кавычки и тире — одного сорта. Приведение
запоминает, из какого места исходного текста взялся каждый символ, — иначе
найденную цитату нельзя было бы подсветить в оригинале, а подсветка и есть
то, ради чего сверка нужна человеку, а не только таблице.

## Пять вердиктов вместо «да/нет»

Двоичного ответа мало, потому что «не нашлось» бывает четырёх разных сортов, и
лечатся они по-разному:

* `дословно` — нашлась как есть;
* `с правкой` — нашлась, но модель поправила слово или пропустила кусок: длинный
  общий отрезок на месте, похожесть окна выше порога;
* `не из той выдержки` — цитата настоящая, а номер назван чужой. Текст в базе
  есть, ошибка в адресе, и валить это в одну кучу с выдумкой нельзя;
* `нет такой выдержки` — номер показывает в пустоту: модель сослалась на [7],
  когда выдержек пять. Чистая выдумка, но выдумка ссылки, а не текста;
* `не найдена` — такой строки нет нигде в контексте. Вот это и есть цитата,
  которую модель сочинила.

Порог похожести выбран один раз и не подбирался по результату: `с правкой`
требует и длинного общего отрезка (0.6 длины цитаты), и похожести окна 0.8.
Два условия вместо одного нужны потому, что длинная цитата с одной общей
половиной даёт отрезок 0.5 при похожести 0.7, и ни то, ни другое само по себе
не отличает правку от пересказа.

## Числа снисхождения не получают

У `с правкой` есть изъян, который пришлось закрыть отдельно. «Слушает на порту
8771» против «слушает на порту 8770» — это похожесть 0.97, то есть по меркам
окна почти та же строка. А по сути это ровно та галлюцинация, ради которой день
и затевался: порт назван неверно, и ответ, опирающийся на такую цитату, врёт.
Поэтому числа сверяются отдельно и строго: все цифровые группы цитаты обязаны
стоять в найденном окне. Не стоят — вердикт `не найдена`, и никакая похожесть
букв этого не перебьёт.
"""

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from cite import Claim

VERBATIM = "дословно"
NEAR = "с правкой"
MISPLACED = "не из той выдержки"
NO_SOURCE = "нет такой выдержки"
MISSING = "не найдена"
NO_QUOTE = "цитаты нет"

VERDICTS = (VERBATIM, NEAR, MISPLACED, NO_SOURCE, MISSING, NO_QUOTE)

# Цитата короче этого ничего не подкрепляет: «8770» найдётся в любом чанке про
# порт, и сверка превратилась бы в проверку того, что модель умеет копировать
# четыре символа.
MIN_CHARS = 20

# Длина общего отрезка к длине цитаты и похожесть окна — оба условия сразу.
NEAR_BLOCK = 0.6
NEAR_RATIO = 0.8

_DIGITS = re.compile(r"\d+")
_FOLD = {
    "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u200b": "",
    "ё": "е", "Ё": "е",
    "«": '"', "»": '"', "“": '"', "”": '"', "„": '"',
    "‘": "'", "’": "'",
    "–": "-", "—": "-", "−": "-", "‑": "-",
    "…": "...",
}


@dataclass(frozen=True)
class Checked:
    """Одна цитата после сверки: вердикт и место, если место нашлось."""

    claim: Claim
    verdict: str
    source: int | None = None
    chunk_id: int | None = None
    path: str = ""
    section: str = ""
    start: int = -1
    end: int = -1
    similarity: float = 0.0

    @property
    def exact(self) -> bool:
        return self.verdict == VERBATIM

    @property
    def grounded(self) -> bool:
        """Цитата настоящая и названа верно: только на такую можно опереться."""
        return self.verdict in (VERBATIM, NEAR)

    @property
    def fabricated(self) -> bool:
        """Текста нет нигде либо ссылка показывает в пустоту."""
        return self.verdict in (MISSING, NO_SOURCE, NO_QUOTE)

    @property
    def located(self) -> bool:
        return self.start >= 0

    def as_dict(self) -> dict[str, object]:
        return {
            **self.claim.as_dict(),
            "verdict": self.verdict,
            "exact": self.exact,
            "grounded": self.grounded,
            "fabricated": self.fabricated,
            "found_source": self.source,
            "chunk_id": self.chunk_id,
            "path": self.path,
            "section": self.section,
            "start": self.start,
            "end": self.end,
            "similarity": round(self.similarity, 3),
        }


@dataclass(frozen=True)
class Report:
    """Сверка всех цитат одного ответа. Доли считаются от числа утверждений."""

    checked: list[Checked] = field(default_factory=list)

    @property
    def claims(self) -> int:
        return len(self.checked)

    @property
    def quoted(self) -> int:
        return sum(1 for item in self.checked if item.claim.quote.strip())

    @property
    def sourced(self) -> int:
        return sum(1 for item in self.checked if item.verdict != NO_SOURCE)

    @property
    def exact(self) -> int:
        return sum(1 for item in self.checked if item.exact)

    @property
    def grounded(self) -> int:
        return sum(1 for item in self.checked if item.grounded)

    @property
    def fabricated(self) -> int:
        return sum(1 for item in self.checked if item.fabricated)

    def share(self, count: int) -> float | None:
        return count / self.claims if self.claims else None

    def as_dict(self) -> dict[str, object]:
        return {
            "claims": self.claims,
            "quoted": self.quoted,
            "sourced": self.sourced,
            "exact": self.exact,
            "grounded": self.grounded,
            "fabricated": self.fabricated,
            "exact_share": self.share(self.exact),
            "grounded_share": self.share(self.grounded),
            "items": [item.as_dict() for item in self.checked],
        }


# --- приведение --------------------------------------------------------------


def fold(text: str) -> tuple[str, list[int]]:
    """Текст в сравнимом виде вместе с картой смещений в исходник.

    Карта нужна ровно затем, чтобы вернуть найденную цитату на место: индекс
    `i` приведённой строки пришёл из символа `offsets[i]` исходной.
    """
    folded: list[str] = []
    offsets: list[int] = []
    space = True  # ведущие пробелы съедаются так же, как лишние внутренние

    for position, symbol in enumerate(text):
        if symbol.isspace() or symbol == "\u00a0":
            if not space:
                folded.append(" ")
                offsets.append(position)
                space = True
            continue

        space = False
        for piece in _FOLD.get(symbol, symbol).lower():
            folded.append(piece)
            offsets.append(position)

    while folded and folded[-1] == " ":
        folded.pop()
        offsets.pop()

    return "".join(folded), offsets


def _span(offsets: list[int], start: int, length: int, source: str) -> tuple[int, int]:
    """Смещения в исходном тексте по куску приведённого."""
    if length <= 0 or start >= len(offsets):
        return -1, -1

    begin = offsets[start]
    last = offsets[min(start + length, len(offsets)) - 1]
    return begin, min(last + 1, len(source))


def _near(quote: str, text: str, offsets: list[int], source: str) -> tuple[int, int, float]:
    """Лучшее окно под цитату и его похожесть. Возвращает (-1, -1, доля) при промахе."""
    if not quote or not text:
        return -1, -1, 0.0

    matcher = SequenceMatcher(None, quote, text, autojunk=False)
    block = matcher.find_longest_match(0, len(quote), 0, len(text))
    if block.size < len(quote) * NEAR_BLOCK:
        return -1, -1, block.size / len(quote)

    # Окно берётся по якорю: кусок текста той же длины, что цитата, выровненный
    # по найденному общему отрезку.
    begin = max(0, min(block.b - block.a, len(text) - len(quote)))
    window = text[begin : begin + len(quote)]
    ratio = SequenceMatcher(None, quote, window, autojunk=False).ratio()

    if ratio < NEAR_RATIO:
        return -1, -1, ratio

    # Буквы можно поправить, числа — нет. Порт 8771 вместо 8770 даёт похожесть
    # 0.97 и прошёл бы как опечатка, хотя это и есть выдуманный факт.
    if not set(_DIGITS.findall(quote)) <= set(_DIGITS.findall(window)):
        return -1, -1, ratio

    start, end = _span(offsets, begin, len(window), source)
    return start, end, ratio


# --- сверка ------------------------------------------------------------------


@dataclass(frozen=True)
class Passage:
    """Выдержка контекста глазами сверки: номер, адрес и текст целиком."""

    number: int
    chunk_id: int
    path: str
    section: str
    text: str


def _look(quote: str, passage: Passage) -> tuple[int, int, float]:
    """Где цитата стоит в выдержке: точно, приблизительно или нигде."""
    folded, offsets = fold(passage.text)
    needle, _ = fold(quote)

    found = folded.find(needle)
    if found >= 0:
        start, end = _span(offsets, found, len(needle), passage.text)
        return start, end, 1.0

    return _near(needle, folded, offsets, passage.text)


def check(claim: Claim, passages: dict[int, Passage]) -> Checked:
    """Одна цитата против контекста: сначала своя выдержка, потом все остальные."""
    quote = claim.quote.strip()
    if len(fold(quote)[0]) < MIN_CHARS:
        return Checked(claim=claim, verdict=NO_QUOTE)

    own = passages.get(claim.source) if claim.source is not None else None
    if own is None:
        # Номер в пустоту. Цитату всё равно ищем: если текст настоящий, это
        # сбитая ссылка, и путать её с выдумкой нельзя.
        for passage in passages.values():
            start, end, ratio = _look(quote, passage)
            if start >= 0:
                return Checked(
                    claim=claim, verdict=MISPLACED, source=passage.number,
                    chunk_id=passage.chunk_id, path=passage.path, section=passage.section,
                    start=start, end=end, similarity=ratio,
                )
        return Checked(claim=claim, verdict=NO_SOURCE)

    start, end, ratio = _look(quote, own)
    if start >= 0:
        return Checked(
            claim=claim, verdict=VERBATIM if ratio >= 1.0 else NEAR, source=own.number,
            chunk_id=own.chunk_id, path=own.path, section=own.section,
            start=start, end=end, similarity=ratio,
        )

    best = ratio
    for passage in passages.values():
        if passage.number == own.number:
            continue
        other_start, other_end, other_ratio = _look(quote, passage)
        if other_start >= 0:
            return Checked(
                claim=claim, verdict=MISPLACED, source=passage.number,
                chunk_id=passage.chunk_id, path=passage.path, section=passage.section,
                start=other_start, end=other_end, similarity=other_ratio,
            )
        best = max(best, other_ratio)

    return Checked(claim=claim, verdict=MISSING, similarity=best)


def review(claims: list[Claim], passages: dict[int, Passage]) -> Report:
    return Report(checked=[check(claim, passages) for claim in claims])
