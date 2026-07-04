from __future__ import annotations

from dataclasses import dataclass

from ..eval.runner import EvalReport

MAX_EXAMPLES = 5


@dataclass
class Diagnosis:
    target_agent: str
    summary: str
    failing_examples: list[dict]


def diagnose(report: EvalReport) -> Diagnosis | None:
    """Attribute failures to whichever agent is responsible, and summarize them.

    Category/priority mistakes are the classifier's fault; a missing required
    phrase in the reply is the responder's fault. Returns None once every
    check passes — there's nothing left for the Coach to fix.
    """
    classifier_failures = [r for r in report.results if not (r.category_ok and r.priority_ok)]
    responder_failures = [r for r in report.results if not r.phrase_ok]

    if not classifier_failures and not responder_failures:
        return None

    if len(classifier_failures) >= len(responder_failures):
        target, failures = "classifier", classifier_failures
    else:
        target, failures = "responder", responder_failures

    total = len(report.results)
    summary = (
        f"{len(failures)}/{total} cases have incorrect {target} output. "
        "Review the failing examples below and adjust the agent's rules to cover the missing patterns."
    )
    return Diagnosis(
        target_agent=target,
        summary=summary,
        failing_examples=[r.to_example_dict() for r in failures[:MAX_EXAMPLES]],
    )
