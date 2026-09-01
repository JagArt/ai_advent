import json
from pathlib import Path

from pydantic import BaseModel, Field

CONSTRAINTS_PATH = Path(__file__).parent / "constraints.json"


class Section(BaseModel):
    title: str
    rules: list[str] = Field(min_length=1)


class Constraints(BaseModel):
    system_prompt: str
    sections: list[Section] = Field(min_length=1)
    params: dict = Field(default_factory=dict)

    def build_system_prompt(self) -> str:
        blocks = [self.system_prompt]

        for section in self.sections:
            rules = "\n".join(f"- {rule}" for rule in section.rules)
            blocks.append(f"{section.title}:\n{rules}")

        return "\n\n".join(blocks)


def load_constraints() -> Constraints:
    return Constraints.model_validate(json.loads(CONSTRAINTS_PATH.read_text(encoding="utf-8")))


constraints = load_constraints()
