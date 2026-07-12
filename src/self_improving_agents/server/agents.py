"""LLAMPHouse Agent wrappers around the framework-agnostic domain agents.

ClassifierAgent/ResponderAgent (agents/classifier.py, agents/responder.py)
know nothing about LLAMPHouse -- they only depend on the LLMClient interface.
These wrappers are the thin adapter layer that lets the same domain agents
be served over LLAMPHouse's OpenAI-compatible API instead of only being
driven in-process by the eval harness and Coach.

`handover_to_agent()` reuses the caller's thread and does not persist its
`message` argument as a new message (the target agent already sees the same
conversation) -- so the classifier's category/priority can't be handed off
as message text. Structured handoff data goes through the run's `metadata`
instead, read back via `context.run.metadata` on the responder side.
"""

from __future__ import annotations

from llamphouse.core import Agent
from llamphouse.core.context import Context

from ..agents.classifier import ClassifierAgent
from ..agents.responder import ResponderAgent
from ..llm.base import LLMClient

RESPONDER_AGENT_ID = "responder"


class ClassifierServerAgent(Agent):
    """Classifies the incoming ticket, then hands off to the responder agent."""

    def __init__(self, llm: LLMClient, classifier: ClassifierAgent | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._llm = llm
        self._classifier = classifier or ClassifierAgent()

    async def run(self, context: Context) -> None:
        ticket = context.messages[-1].text
        classification = self._classifier.classify(self._llm, ticket)
        await context.handover_to_agent(
            RESPONDER_AGENT_ID,
            ticket,
            metadata={"category": classification.category, "priority": classification.priority},
        )


class ResponderServerAgent(Agent):
    """Drafts the reply once handed a classified ticket."""

    def __init__(self, llm: LLMClient, responder: ResponderAgent | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._llm = llm
        self._responder = responder or ResponderAgent()

    async def run(self, context: Context) -> None:
        ticket = context.messages[-1].text
        run_metadata = context.run.metadata or {}
        category = run_metadata.get("category", "general")
        priority = run_metadata.get("priority", "low")
        reply = self._responder.respond(self._llm, ticket, category, priority)
        await context.insert_message(reply)
