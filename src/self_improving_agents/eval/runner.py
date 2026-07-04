from __future__ import annotations

from dataclasses import dataclass

from ..agents.pipeline import SupportTriagePipeline
from ..llm.base import LLMClient
from .cases import EvalCase
from .scorer import EvalResult, score_case


@dataclass
class EvalReport:
    results: list[EvalResult]

    @property
    def mean_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)


def run_eval(pipeline: SupportTriagePipeline, llm: LLMClient, cases: list[EvalCase]) -> EvalReport:
    results = [score_case(case, pipeline.run(llm, case.ticket)) for case in cases]
    return EvalReport(results=results)
