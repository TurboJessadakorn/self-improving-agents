from __future__ import annotations

from ..llm.base import LLMClient
from .diagnose import Diagnosis

COACH_SYSTEM_PROMPT = """You are a Coach that improves prompts for other agents in a multi-agent system.
You are given the agent's current system prompt, a diagnosis of why it is
failing evaluation cases, and a list of concrete failing examples.
Return ONLY the full revised system prompt text for the agent -- no commentary,
no markdown fences. Preserve rules that already work; add or adjust rules to
fix the failures while keeping the agent's existing format and style."""


def build_propose_prompt(agent_name: str, current_prompt: str, diagnosis: Diagnosis) -> str:
    lines = [
        f"## Target agent: {agent_name}",
        "",
        "## Current prompt",
        current_prompt,
        "",
        "## Diagnosis",
        diagnosis.summary,
        "",
        "## Failing cases",
    ]
    if agent_name == "classifier":
        for ex in diagnosis.failing_examples:
            lines.append(f'- ticket: "{ex["ticket"]}"')
            lines.append(f'  expected: category={ex["expected"]["category"]}, priority={ex["expected"]["priority"]}')
            lines.append(f'  actual: category={ex["actual"]["category"]}, priority={ex["actual"]["priority"]}')
    elif agent_name == "responder":
        for ex in diagnosis.failing_examples:
            lines.append(f'- ticket: "{ex["ticket"]}"')
            lines.append(f'  category: {ex["actual"]["category"]}')
            lines.append(f'  expected_phrase: "{ex["expected_phrase"]}"')
            lines.append(f'  actual_reply: "{ex["actual_reply"]}"')
    else:
        raise ValueError(f"Unknown target agent {agent_name!r}")
    return "\n".join(lines)


def propose_new_prompt(llm: LLMClient, agent_name: str, current_prompt: str, diagnosis: Diagnosis) -> str:
    user = build_propose_prompt(agent_name, current_prompt, diagnosis)
    candidate = llm.complete(COACH_SYSTEM_PROMPT, user).strip()
    if not candidate:
        raise RuntimeError("Coach LLM returned an empty prompt")
    return candidate
