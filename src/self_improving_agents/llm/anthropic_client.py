"""Anthropic-backed LLMClient implementation.

The `anthropic` package is imported lazily so the rest of the codebase
(including the full test suite, via MockClient) works without it installed.
"""

from __future__ import annotations


class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 2048) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the package
            raise ImportError(
                "The 'anthropic' package is required for AnthropicClient. "
                "Install it with: pip install self-improving-agents[anthropic]"
            ) from exc

        self._client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
