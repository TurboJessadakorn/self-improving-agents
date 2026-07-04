from __future__ import annotations

from dataclasses import dataclass

from ..agents.pipeline import PipelineResult
from .cases import EvalCase


@dataclass
class EvalResult:
    case: EvalCase
    actual: PipelineResult
    category_ok: bool
    priority_ok: bool
    phrase_ok: bool

    @property
    def score(self) -> float:
        checks = (self.category_ok, self.priority_ok, self.phrase_ok)
        return sum(checks) / len(checks)

    def to_example_dict(self) -> dict:
        return {
            "ticket": self.case.ticket,
            "expected": {"category": self.case.expected_category, "priority": self.case.expected_priority},
            "actual": {"category": self.actual.category, "priority": self.actual.priority},
            "expected_phrase": self.case.expected_phrase,
            "actual_reply": self.actual.reply,
        }


def score_case(case: EvalCase, actual: PipelineResult) -> EvalResult:
    return EvalResult(
        case=case,
        actual=actual,
        category_ok=actual.category.lower() == case.expected_category.lower(),
        priority_ok=actual.priority.lower() == case.expected_priority.lower(),
        phrase_ok=case.expected_phrase.lower() in actual.reply.lower(),
    )
