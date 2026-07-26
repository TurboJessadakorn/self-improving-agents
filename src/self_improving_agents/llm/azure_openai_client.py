"""Azure OpenAI-backed LLMClient implementation.

The `openai` package is imported lazily so the rest of the codebase
(including the full test suite, via MockClient) works without it installed.

Unlike AnthropicClient, there's no sensible default for `deployment` --
Azure OpenAI is addressed by a deployment name you configure in your own
Azure resource, not a universal model name, so it's required. `endpoint`,
`api_key`, and `api_version` are optional here and fall back to the
`AzureOpenAI` SDK's own environment-variable defaults (`AZURE_OPENAI_ENDPOINT`,
`AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION`) when omitted, so this file
doesn't have to hardcode an API version that could go stale.
"""

from __future__ import annotations


class AzureOpenAIClient:
    def __init__(
        self,
        deployment: str,
        api_version: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover - exercised only without the package
            raise ImportError(
                "The 'openai' package is required for AzureOpenAIClient. "
                "Install it with: pip install self-improving-agents[azure-openai]"
            ) from exc

        client_kwargs = {}
        if api_version:
            client_kwargs["api_version"] = api_version
        if endpoint:
            client_kwargs["azure_endpoint"] = endpoint
        if api_key:
            client_kwargs["api_key"] = api_key

        self._client = AzureOpenAI(**client_kwargs)
        self.deployment = deployment
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.deployment,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""
