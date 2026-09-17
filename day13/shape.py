"""Форма ответа в числах: чем ответ одному профилю отличается от ответа другому.

Профиль управляет не содержанием, а видом ответа, поэтому проверять его лучше
измеримым, а не пересказом. Здесь считается то, на что прямо указывают шкалы:
длина — словами, формат — строками списка и таблицами, обращение — местоимениями,
запрет эмодзи — самими эмодзи. Оценку «в целом похоже на профиль» даёт судья-модель
в прогоне, а этот модуль отвечает за ту часть, где мнение не нужно.

Меры простые нарочно: они должны читаться в таблице README и не требовать доверия
к тому, кто их считал.
"""

import re
from dataclasses import dataclass

# Строка списка: дефис, звёздочка, точка или номер в начале строки. Пункты модель
# оформляет по-разному, но начало строки выдаёт список в любом виде.
BULLET = re.compile(r"^\s*(?:[-*•‣–]|\d+[.)])\s+", re.MULTILINE)

# Строка таблицы markdown: две вертикальные черты и что-то между ними.
TABLE = re.compile(r"^\s*\|.*\|", re.MULTILINE)

# Только огороженный блок: отступом в четыре пробела модели оформляют продолжение
# пункта списка гораздо чаще, чем код, и такая мерка ловила бы не то.
CODE = re.compile(r"```")

# Обращение видно по местоимениям: их формы не спутать, а вот глаголы («смотри» и
# «смотрите») различать пришлось бы морфологией, и ради двух слов это лишнее.
INFORMAL = re.compile(r"\b(ты|тебя|тебе|тобой|твой|твоя|твои|твоего|твоей|твоих|твоим)\b")
FORMAL = re.compile(r"\b(вы|вас|вам|вами|ваш|ваша|ваши|вашего|вашей|ваших|вашим)\b")

EMOJI = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U0001f000-\U0001f2ff"
    "\u2600-\u27bf"
    "\u2b00-\u2bff"
    "\ufe0f"
    "]"
)


@dataclass(frozen=True)
class Shape:
    """Мерка одного ответа: только то, чем управляют шкалы профиля."""

    words: int
    bullets: int
    # Строк таблицы, а не таблиц: считать таблицы целиком незачем — по числу строк
    # видно и то, что таблица есть, и то, насколько она подробная.
    tables: int
    code: bool
    # «ты», «вы» или «—», если местоимений в ответе не нашлось вовсе.
    address: str
    emoji: int

    @property
    def line(self) -> str:
        parts = [f"{self.words} сл."]
        parts.append(f"{self.bullets} п. списком" if self.bullets else "без списка")
        if self.tables:
            parts.append("таблица")
        if self.code:
            parts.append("код")
        parts.append(f"на «{self.address}»" if self.address != "—" else "без обращения")
        if self.emoji:
            parts.append(f"эмодзи {self.emoji}")
        return " · ".join(parts)

    def as_dict(self) -> dict[str, object]:
        return {
            "words": self.words,
            "bullets": self.bullets,
            "tables": self.tables,
            "code": self.code,
            "address": self.address,
            "emoji": self.emoji,
            "line": self.line,
        }


def measure(text: str) -> Shape:
    lowered = text.lower()
    informal = len(INFORMAL.findall(lowered))
    formal = len(FORMAL.findall(lowered))

    return Shape(
        words=len(text.split()),
        bullets=len(BULLET.findall(text)),
        tables=len(TABLE.findall(text)),
        code=bool(CODE.search(text)),
        # Побеждает то обращение, которого больше: одиночное «вы» посреди ответа на
        # «ты» бывает частью цитаты или устойчивого оборота.
        address="ты" if informal > formal else "вы" if formal > informal else "—",
        emoji=len(EMOJI.findall(text)),
    )
