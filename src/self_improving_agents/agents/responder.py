from __future__ import annotations

from ..llm.base import LLMClient
from .base import Agent

DEFAULT_RESPONDER_PROMPT = """You are a support responder. Write a short, empathetic reply to the customer.

Rules (checked in order, first match wins):
- If category=billing -> include the phrase "processed within 3-5 business days".
- Default -> include the phrase "our team will follow up shortly".

Output only the reply text.
"""


class ResponderAgent(Agent):
    def __init__(self, prompt: str = DEFAULT_RESPONDER_PROMPT) -> None:
        super().__init__(name="responder", prompt=prompt)

    def respond(self, llm: LLMClient, ticket: str, category: str, priority: str) -> str:
        user = f"Ticket: {ticket}\nCategory: {category}\nPriority: {priority}"
        return llm.complete(self.prompt, user).strip()
