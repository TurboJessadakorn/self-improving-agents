from self_improving_agents.agents.classifier import ClassifierAgent
from self_improving_agents.agents.pipeline import SupportTriagePipeline
from self_improving_agents.agents.responder import ResponderAgent
from self_improving_agents.llm.mock_client import MockClient


def test_classifier_matches_first_rule():
    llm = MockClient()
    classification = ClassifierAgent().classify(llm, "I was charged twice, please refund me.")
    assert classification.category == "billing"
    assert classification.priority == "medium"


def test_classifier_falls_back_to_default():
    llm = MockClient()
    classification = ClassifierAgent().classify(llm, "What are your business hours?")
    assert classification.category == "general"
    assert classification.priority == "low"


def test_responder_uses_category_specific_phrase():
    llm = MockClient()
    reply = ResponderAgent().respond(llm, "please refund me", category="billing", priority="medium")
    assert "processed within 3-5 business days" in reply


def test_responder_falls_back_to_default_phrase():
    llm = MockClient()
    reply = ResponderAgent().respond(llm, "app crashes", category="technical", priority="high")
    assert "our team will follow up shortly" in reply


def test_pipeline_chains_classifier_and_responder():
    llm = MockClient()
    result = SupportTriagePipeline().run(llm, "The app crashes every time I open settings.")
    assert result.category == "technical"
    assert result.priority == "high"
    assert "our team will follow up shortly" in result.reply
