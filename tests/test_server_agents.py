import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("llamphouse")

from self_improving_agents.llm.mock_client import MockClient
from self_improving_agents.server.agents import RESPONDER_AGENT_ID, ClassifierServerAgent, ResponderServerAgent


class FakeContext:
    """Stands in for llamphouse's Context: just the bits our agents touch."""

    def __init__(self, ticket: str, run_metadata: dict | None = None):
        self.messages = [SimpleNamespace(text=ticket)]
        self.run = SimpleNamespace(metadata=run_metadata or {})
        self.handover_calls: list[tuple] = []
        self.inserted_messages: list[str] = []

    async def handover_to_agent(self, agent_id, message, *, metadata=None):
        self.handover_calls.append((agent_id, message, metadata))

    async def insert_message(self, content):
        self.inserted_messages.append(content)


def test_classifier_server_agent_hands_off_with_classification_metadata():
    ticket = "I was charged twice for my subscription, please refund me."
    context = FakeContext(ticket)
    agent = ClassifierServerAgent(MockClient(), id="classifier")

    asyncio.run(agent.run(context))

    assert len(context.handover_calls) == 1
    agent_id, message, metadata = context.handover_calls[0]
    assert agent_id == RESPONDER_AGENT_ID
    assert message == ticket
    assert metadata == {"category": "billing", "priority": "medium"}


def test_responder_server_agent_drafts_reply_from_run_metadata():
    context = FakeContext(
        "I was charged twice for my subscription, please refund me.",
        run_metadata={"category": "billing", "priority": "medium"},
    )
    agent = ResponderServerAgent(MockClient(), id="responder")

    asyncio.run(agent.run(context))

    assert len(context.inserted_messages) == 1
    assert "processed within 3-5 business days" in context.inserted_messages[0]


def test_responder_server_agent_falls_back_to_default_without_metadata():
    context = FakeContext("What are your business hours?")
    agent = ResponderServerAgent(MockClient(), id="responder")

    asyncio.run(agent.run(context))

    assert "our team will follow up shortly" in context.inserted_messages[0]
