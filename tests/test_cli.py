from self_improving_agents.cli import build_parser, main


def test_eval_command_runs_and_reports_mean_score(capsys):
    main(["eval"])
    out = capsys.readouterr().out
    assert "mean score: 80.00%" in out


def test_coach_command_runs_and_accepts_an_improvement(capsys):
    main(["coach"])
    out = capsys.readouterr().out
    assert "-> accepted" in out
    assert "mean score: 100.00%" in out


def test_parser_rejects_unknown_backend():
    parser = build_parser()
    try:
        parser.parse_args(["--llm", "not-a-backend", "eval"])
        assert False, "expected SystemExit"
    except SystemExit:
        pass
