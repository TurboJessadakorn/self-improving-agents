from __future__ import annotations

import argparse
from pathlib import Path

from .agents.pipeline import SupportTriagePipeline
from .coach.loop import run_coach
from .eval.cases import load_cases
from .eval.runner import EvalReport, run_eval
from .llm.base import LLMClient

DEFAULT_CASES_PATH = Path(__file__).resolve().parents[2] / "data" / "eval_cases" / "support_triage.yaml"


def _make_llm(backend: str, model: str | None) -> LLMClient:
    """Factory kept here (the only caller) so anthropic stays an optional import."""
    if backend == "mock":
        from .llm.mock_client import MockClient

        return MockClient()
    if backend == "anthropic":
        from .llm.anthropic_client import AnthropicClient

        kwargs = {"model": model} if model else {}
        return AnthropicClient(**kwargs)
    raise ValueError(f"Unknown LLM backend {backend!r}. Choose 'mock' or 'anthropic'.")


def _print_report(report: EvalReport) -> None:
    for result in report.results:
        status = "PASS" if result.score == 1.0 else "FAIL"
        print(
            f"[{status}] {result.case.id}: score={result.score:.2f} "
            f"category={result.actual.category} priority={result.actual.priority}"
        )
    print(f"\nmean score: {report.mean_score:.2%} ({len(report.results)} cases)")


def cmd_eval(args: argparse.Namespace) -> None:
    llm = _make_llm(args.llm, args.model)
    cases = load_cases(args.cases)
    pipeline = SupportTriagePipeline()
    report = run_eval(pipeline, llm, cases)
    _print_report(report)


def cmd_coach(args: argparse.Namespace) -> None:
    llm = _make_llm(args.llm, args.model)
    cases = load_cases(args.cases)
    pipeline = SupportTriagePipeline()

    result = run_coach(pipeline, llm, cases, max_iterations=args.max_iterations)

    for log in result.history:
        outcome = "accepted" if log.accepted else "rejected"
        candidate = f"{log.candidate_score:.2%}" if log.candidate_score is not None else "n/a"
        print(
            f"iteration {log.iteration}: target={log.target_agent} "
            f"baseline={log.baseline_score:.2%} candidate={candidate} -> {outcome}"
        )
        print(f"  diagnosis: {log.diagnosis_summary}")
        if log.error:
            print(f"  error: {log.error}")

    print()
    _print_report(result.final_report)

    print("\nfinal classifier prompt:\n" + result.pipeline.classifier.prompt)
    print("\nfinal responder prompt:\n" + result.pipeline.responder.prompt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sia", description="Self-improving agents: eval-driven prompt coaching.")
    parser.add_argument("--llm", choices=["mock", "anthropic"], default="mock", help="LLM backend to use.")
    parser.add_argument("--model", default=None, help="Model name, if the backend supports one.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH), help="Path to an eval cases YAML file.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval", help="Run the eval suite against the current agent prompts.")
    eval_parser.set_defaults(func=cmd_eval)

    coach_parser = subparsers.add_parser("coach", help="Run the Coach's diagnose/propose/validate loop.")
    coach_parser.add_argument("--max-iterations", type=int, default=5)
    coach_parser.set_defaults(func=cmd_coach)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
