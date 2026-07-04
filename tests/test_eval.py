from pathlib import Path

from self_improving_agents.agents.pipeline import SupportTriagePipeline
from self_improving_agents.eval.cases import load_cases
from self_improving_agents.eval.runner import run_eval
from self_improving_agents.llm.mock_client import MockClient

CASES_PATH = Path(__file__).resolve().parents[1] / "data" / "eval_cases" / "support_triage.yaml"


def test_load_cases_reads_all_fixtures():
    cases = load_cases(CASES_PATH)
    assert len(cases) == 10
    assert {c.id for c in cases} >= {"billing-refund", "account-login"}


def test_baseline_prompt_misclassifies_account_tickets():
    cases = load_cases(CASES_PATH)
    report = run_eval(SupportTriagePipeline(), MockClient(), cases)

    assert report.mean_score == 0.8

    failing_ids = {r.case.id for r in report.results if r.score < 1.0}
    assert failing_ids == {"account-login", "account-reset", "account-update"}
    for result in report.results:
        if result.case.id in failing_ids:
            assert result.actual.category == "general"
