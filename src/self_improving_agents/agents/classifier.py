from __future__ import annotations

import re
from dataclasses import dataclass

from ..llm.base import LLMClient
from .base import Agent

DEFAULT_CLASSIFIER_PROMPT = """You are a support ticket classifier.
Categories: billing, technical, account, general.
Priorities: low, medium, high.

Rules (checked in order, first match wins):
- If the ticket mentions "refund", "charge", or "invoice" -> category=billing, priority=medium.
- If the ticket mentions "crash", "error", or "bug" -> category=technical, priority=high.
- Default -> category=general, priority=low.

Respond with exactly one line in the form: category=<category>, priority=<priority>
"""

_RESPONSE_RE = re.compile(r"category=(\w+),\s*priority=(\w+)", re.IGNORECASE)


@dataclass
class Classification:
    category: str
    priority: str


class ClassifierAgent(Agent):
    def __init__(self, prompt: str = DEFAULT_CLASSIFIER_PROMPT) -> None:
        super().__init__(name="classifier", prompt=prompt)

    def classify(self, llm: LLMClient, ticket: str) -> Classification:
        raw = llm.complete(self.prompt, ticket)
        match = _RESPONSE_RE.search(raw)
        if not match:
            raise ValueError(f"Could not parse classifier output: {raw!r}")
        return Classification(category=match.group(1).lower(), priority=match.group(2).lower())
