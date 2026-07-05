# self-improving-agents
An eval-driven Coach agent that improves other agents' prompts in a multi-agent system — diagnose, propose, validate, keep only what helps.

## Demo: support ticket triage

A two-agent pipeline (`classifier` -> `responder`) handles support tickets. Each
agent's behavior is entirely defined by its own editable system prompt. A
10-case eval fixture scores the pipeline's output, and the Coach uses that
score to decide whether a proposed prompt change actually helps:

1. **Diagnose** — run the eval, attribute failures to whichever agent caused
   them (wrong category/priority -> classifier, missing reply phrase ->
   responder).
2. **Propose** — ask the LLM for a revised prompt given the diagnosis and the
   concrete failing examples.
3. **Validate** — re-run the eval with the candidate prompt swapped in.
4. **Keep if better** — accept the change only if it strictly improves the
   mean score; otherwise revert.

The bundled fixture is seeded with a gap on purpose: the classifier's default
prompt only has rules for `billing` and `technical`, so `account` tickets
fall through to `general`. Baseline score is 80%; the Coach finds the gap,
proposes a new rule, and closes it to 100%.

## Try it

```bash
pip install -e .
sia eval    # baseline: 80%
sia coach   # diagnoses the account-classification gap, proposes a fix, keeps it -> 100%
```

Both commands default to an offline `mock` LLM backend (see
[`llm/mock_client.py`](src/self_improving_agents/llm/mock_client.py)) so the
whole loop runs without an API key. Pass `--llm anthropic` to use a real
Claude model instead:

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=...
sia coach --llm anthropic --model claude-sonnet-5
```

## Layout

```
src/self_improving_agents/
  llm/       # provider-agnostic LLMClient interface (Mock + Anthropic backends)
  agents/    # classifier + responder agents, chained into a pipeline
  eval/      # eval cases, scoring, and the report runner
  coach/     # diagnose -> propose -> validate -> keep-if-better loop
  cli.py     # `sia eval` / `sia coach`
data/eval_cases/support_triage.yaml   # the 10-case fixture
tests/
```

Run the test suite with:

```bash
pip install -e ".[dev]"
pytest
```
