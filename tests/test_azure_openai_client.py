import types

import pytest

pytest.importorskip("openai")

from self_improving_agents.llm.azure_openai_client import AzureOpenAIClient


class _FakeCompletions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        message = types.SimpleNamespace(content="hi from azure")
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


class _FakeAzureOpenAI:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.chat = types.SimpleNamespace(completions=_FakeCompletions())


def test_azure_openai_client_sends_system_and_user_messages(monkeypatch):
    monkeypatch.setattr("openai.AzureOpenAI", _FakeAzureOpenAI)

    client = AzureOpenAIClient(
        deployment="my-deployment",
        api_version="2024-10-21",
        endpoint="https://example.openai.azure.com/",
        api_key="secret",
    )
    result = client.complete("system prompt", "user prompt")

    assert result == "hi from azure"
    assert client._client.init_kwargs == {
        "api_version": "2024-10-21",
        "azure_endpoint": "https://example.openai.azure.com/",
        "api_key": "secret",
    }
    sent = client._client.chat.completions.last_kwargs
    assert sent["model"] == "my-deployment"
    assert sent["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]


def test_azure_openai_client_omits_unset_optional_kwargs(monkeypatch):
    monkeypatch.setattr("openai.AzureOpenAI", _FakeAzureOpenAI)

    client = AzureOpenAIClient(deployment="my-deployment")

    assert client._client.init_kwargs == {}
