"""Provider-agnostic LLM client interface."""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Minimal interface every LLM backend must satisfy.

    Keeping this to a single method lets the Coach and demo agents stay
    provider-agnostic: swap the backend without touching call sites.
    """

    def complete(self, system: str, user: str) -> str:
        """Return the model's text response for a system + user prompt pair."""
        ...
