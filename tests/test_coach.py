from pathlib import Path

from self_improving_agents.agents.pipeline import SupportTriagePipeline
from self_improving_agents.coach.diagnose import diagnose
from self_improving_agents.coach.loop import run_coach
from self_improving_agents.eval.cases import load_cases
from self_improving_agents.eval.runner import run_eval
from self_improving_agents.llm.mock_client import MockClient

CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_cases" / "support_triage.yaml"


def test_diagnose_returns_none_when_everything_passes():
    cases = load_cases(CASES_PATH)
    llm = MockClient()
    pipeline = SupportTriagePipeline()
    result = run_coach(pipeline, llm, cases)  # fixes the account-classification gap
    assert diagnose(result.final_report) is None


def test_diagnose_attributes_baseline_failures_to_classifier():
    cases = load_cases(CASES_PATH)
    report = run_eval(SupportTriagePipeline(), MockClient(), cases)
    diagnosis = diagnose(report)
    assert diagnosis is not None
    assert diagnosis.target_agent == "classifier"
    assert len(diagnosis.failing_examples) == 3


def test_coach_loop_improves_and_keeps_the_fix():
    cases = load_cases(CASES_PATH)
    llm = MockClient()
    pipeline = SupportTriagePipeline()

    result = run_coach(pipeline, llm, cases, max_iterations=5)

    assert len(result.history) == 1
    assert result.history[0].accepted is True
    assert result.history[0].baseline_score == 0.8
    assert result.history[0].candidate_score == 1.0
    assert result.final_report.mean_score == 1.0
    # the fix should be applied to the live pipeline object, not just a copy
    assert "account" in result.pipeline.classifier.prompt


def test_coach_loop_stops_when_nothing_left_to_fix():
    cases = load_cases(CASES_PATH)
    llm = MockClient()
    pipeline = SupportTriagePipeline()

    first = run_coach(pipeline, llm, cases, max_iterations=5)
    assert len(first.history) == 1

    second = run_coach(pipeline, llm, cases, max_iterations=5)
    assert second.history == []
    assert second.final_report.mean_score == 1.0
