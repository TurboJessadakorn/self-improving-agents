from __future__ import annotations

from ..eval.runner import EvalReport


def is_improvement(baseline: EvalReport, candidate: EvalReport) -> bool:
    """A candidate prompt is only kept if it strictly improves the mean score."""
    return candidate.mean_score > baseline.mean_score
