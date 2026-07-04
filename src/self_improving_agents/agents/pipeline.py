from __future__ import annotations

from dataclasses import dataclass

from ..llm.base import LLMClient
from .classifier import ClassifierAgent
from .responder import ResponderAgent


@dataclass
class PipelineResult:
    category: str
    priority: str
    reply: str


class SupportTriagePipeline:
    """Two-agent pipeline: classify the ticket, then draft a reply."""

    def __init__(
        self,
        classifier: ClassifierAgent | None = None,
        responder: ResponderAgent | None = None,
    ) -> None:
        self.classifier = classifier or ClassifierAgent()
        self.responder = responder or ResponderAgent()

    def run(self, llm: LLMClient, ticket: str) -> PipelineResult:
        classification = self.classifier.classify(llm, ticket)
        reply = self.responder.respond(llm, ticket, classification.category, classification.priority)
        return PipelineResult(category=classification.category, priority=classification.priority, reply=reply)
