from __future__ import annotations

from dataclasses import dataclass

from ..agents.pipeline import SupportTriagePipeline
from ..eval.cases import EvalCase
from ..eval.runner import EvalReport, run_eval
from ..llm.base import LLMClient
from .diagnose import diagnose
from .propose import propose_new_prompt
from .validate import is_improvement


@dataclass
class IterationLog:
    iteration: int
    target_agent: str
    diagnosis_summary: str
    baseline_score: float
    candidate_score: float | None
    accepted: bool
    error: str | None = None


@dataclass
class CoachResult:
    pipeline: SupportTriagePipeline
    final_report: EvalReport
    history: list[IterationLog]


def run_coach(
    pipeline: SupportTriagePipeline,
    llm: LLMClient,
    cases: list[EvalCase],
    max_iterations: int = 5,
) -> CoachResult:
    """Diagnose -> propose -> validate -> keep only if it helps.

    Mutates the given pipeline's agent prompts in place when a candidate is
    accepted. Stops when there's nothing left to diagnose, a proposal is
    rejected (a deterministic backend would just repeat the same rejection),
    or max_iterations is reached.
    """
    history: list[IterationLog] = []
    current_report = run_eval(pipeline, llm, cases)

    for i in range(1, max_iterations + 1):
        diagnosis = diagnose(current_report)
        if diagnosis is None:
            break

        agent = pipeline.classifier if diagnosis.target_agent == "classifier" else pipeline.responder
        original_prompt = agent.prompt

        try:
            candidate_prompt = propose_new_prompt(llm, diagnosis.target_agent, original_prompt, diagnosis)
        except Exception as exc:
            history.append(
                IterationLog(
                    iteration=i,
                    target_agent=diagnosis.target_agent,
                    diagnosis_summary=diagnosis.summary,
                    baseline_score=current_report.mean_score,
                    candidate_score=None,
                    accepted=False,
                    error=str(exc),
                )
            )
            break

        agent.prompt = candidate_prompt
        candidate_report = run_eval(pipeline, llm, cases)
        accepted = is_improvement(current_report, candidate_report)

        history.append(
            IterationLog(
                iteration=i,
                target_agent=diagnosis.target_agent,
                diagnosis_summary=diagnosis.summary,
                baseline_score=current_report.mean_score,
                candidate_score=candidate_report.mean_score,
                accepted=accepted,
            )
        )

        if accepted:
            current_report = candidate_report
        else:
            agent.prompt = original_prompt
            break

    return CoachResult(pipeline=pipeline, final_report=current_report, history=history)
