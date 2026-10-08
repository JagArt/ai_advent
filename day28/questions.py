"""Контрольный набор day24 без правок: десять вопросов, у каждого ожидание и факты.

Набор переносится, а не пишется заново, по той же причине, что и в day23–day25:
иначе числа дня не с чем сравнить. Он нужен и странице — готовыми вопросами, —
и прогону, где по полю `facts` считается качество ответа.
"""

import json
from dataclasses import dataclass
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "day24" / "questions.json"

KINDS = ("в базе", "общий", "вне базы")


@dataclass(frozen=True)
class Question:
    id: int
    kind: str
    question: str
    expect: str
    facts: list[str]
    sources: list[str]

    @property
    def out_of_base(self) -> bool:
        return self.kind == "вне базы"

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "question": self.question,
            "expect": self.expect,
            "facts": self.facts,
            "sources": self.sources,
        }


def load() -> list[Question]:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    return [
        Question(
            id=item["id"],
            kind=item["kind"],
            question=item["question"],
            expect=item["expect"],
            facts=list(item.get("facts", [])),
            sources=list(item.get("sources", [])),
        )
        for item in data["questions"]
    ]
