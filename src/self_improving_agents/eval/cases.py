from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class EvalCase:
    id: str
    ticket: str
    expected_category: str
    expected_priority: str
    expected_phrase: str


def load_cases(path: str | Path) -> list[EvalCase]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [EvalCase(**item) for item in data["cases"]]
