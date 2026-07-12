from __future__ import annotations

from ..llm.base import LLMClient
from .agents import RESPONDER_AGENT_ID, ClassifierServerAgent, ResponderServerAgent


def build_app(llm: LLMClient):
    """Wire the classifier/responder pair into a LLAMPHouse server instance."""
    from llamphouse.core import LLAMPHouse
    from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore

    classifier = ClassifierServerAgent(
        llm,
        id="classifier",
        name="Ticket Classifier",
        description="Classifies support tickets by category and priority, then hands off to the responder.",
    )
    responder = ResponderServerAgent(
        llm,
        id=RESPONDER_AGENT_ID,
        name="Ticket Responder",
        description="Drafts a reply once a ticket has been classified.",
    )
    return LLAMPHouse(agents=[classifier, responder], data_store=InMemoryDataStore())
