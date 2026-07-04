"""A deterministic, offline stand-in for a real LLM.

MockClient doesn't call any API. Instead it recognizes which *role* it is
being asked to play (classifier / responder / coach) from a marker in the
system prompt, then interprets a small rule DSL embedded in that prompt:

    - If the ticket mentions "a", "b", or "c" -> category=X, priority=Y.
    - Default -> category=X, priority=Y.

This lets the demo agents' behavior genuinely depend on their prompt text
(so editing the prompt changes outcomes, just like a real LLM would), and
lets the Coach's propose step be exercised end-to-end without network access
or an API key. AnthropicClient implements the same `complete` interface and
would use its own judgement instead of this fixed DSL.
"""

from __future__ import annotations

import re
from collections import Counter

_CLASSIFIER_RULE_RE = re.compile(
    r'^-\s*If the ticket mentions (.+?)\s*->\s*category=(\w+),\s*priority=(\w+)\.$'
)
_CLASSIFIER_DEFAULT_RE = re.compile(r'^-\s*Default\s*->\s*category=(\w+),\s*priority=(\w+)\.$')
_RESPONDER_RULE_RE = re.compile(r'^-\s*If category=(\w+)\s*->\s*include the phrase "([^"]+)"\.$')
_RESPONDER_DEFAULT_RE = re.compile(r'^-\s*Default\s*->\s*include the phrase "([^"]+)"\.$')
_QUOTED_RE = re.compile(r'"([^"]+)"')
_CATEGORY_FIELD_RE = re.compile(r'Category:\s*(\w+)', re.IGNORECASE)
_TARGET_AGENT_RE = re.compile(r'##\s*Target agent:\s*(\w+)')
_SECTION_RE_TEMPLATE = r'##\s*{name}\s*\n(.*?)(?=\n##\s|\Z)'
_TOKEN_RE = re.compile(r"[a-zA-Z']+")
_FAILING_CLASSIFIER_CASE_RE = re.compile(
    r'-\s*ticket:\s*"([^"]*)"\s*\n\s*expected:\s*category=(\w+),\s*priority=(\w+)\s*\n'
    r'\s*actual:\s*category=(\w+),\s*priority=(\w+)'
)
_FAILING_RESPONDER_CASE_RE = re.compile(
    r'-\s*ticket:\s*"([^"]*)"\s*\n\s*category:\s*(\w+)\s*\n'
    r'\s*expected_phrase:\s*"([^"]*)"\s*\n\s*actual_reply:\s*"([^"]*)"'
)

_STOPWORDS = {
    "i", "im", "my", "me", "the", "a", "an", "to", "it", "is", "are", "was",
    "please", "help", "and", "or", "for", "of", "on", "this", "that", "can",
    "cant", "do", "does", "how", "what", "when", "very", "into", "with",
    "you", "your", "about", "would", "could", "should", "there", "their",
}
_MIN_KEYWORD_LEN = 5


class MockClient:
    """Offline LLM stand-in. See module docstring."""

    def complete(self, system: str, user: str) -> str:
        if "You are a support ticket classifier" in system:
            return self._classify(system, user)
        if "You are a support responder" in system:
            return self._respond(system, user)
        if "You are a Coach" in system:
            return self._propose(user)
        raise NotImplementedError(
            "MockClient does not recognize this role. Add a matching branch "
            "in MockClient.complete, or use AnthropicClient instead."
        )

    # -- classifier -----------------------------------------------------

    def _classify(self, system: str, ticket: str) -> str:
        ticket_lower = ticket.lower()
        for line in system.splitlines():
            line = line.strip()
            rule = _CLASSIFIER_RULE_RE.match(line)
            if rule:
                keywords = _QUOTED_RE.findall(rule.group(1))
                if any(kw.lower() in ticket_lower for kw in keywords):
                    return f"category={rule.group(2)}, priority={rule.group(3)}"
                continue
            default = _CLASSIFIER_DEFAULT_RE.match(line)
            if default:
                return f"category={default.group(1)}, priority={default.group(2)}"
        return "category=general, priority=low"

    # -- responder --------------------------------------------------------

    def _respond(self, system: str, user: str) -> str:
        match = _CATEGORY_FIELD_RE.search(user)
        category = match.group(1).lower() if match else "general"
        phrase = None
        for line in system.splitlines():
            line = line.strip()
            rule = _RESPONDER_RULE_RE.match(line)
            if rule:
                if rule.group(1).lower() == category:
                    phrase = rule.group(2)
                    break
                continue
            default = _RESPONDER_DEFAULT_RE.match(line)
            if default:
                phrase = default.group(1)
                break
        if phrase is None:
            phrase = "our team will follow up shortly"
        return f"Thanks for reaching out. {phrase}. We appreciate your patience."

    # -- coach propose ------------------------------------------------------

    def _propose(self, user: str) -> str:
        target_match = _TARGET_AGENT_RE.search(user)
        if not target_match:
            raise ValueError("Coach prompt is missing '## Target agent: <name>'")
        target = target_match.group(1)
        current_prompt = _extract_section(user, "Current prompt")
        if target == "classifier":
            return self._propose_classifier(current_prompt, user)
        if target == "responder":
            return self._propose_responder(current_prompt, user)
        raise ValueError(f"MockClient cannot propose fixes for unknown agent {target!r}")

    def _propose_classifier(self, current_prompt: str, user: str) -> str:
        groups: dict[tuple[str, str], list[str]] = {}
        for ticket, exp_cat, exp_pri, _act_cat, _act_pri in _FAILING_CLASSIFIER_CASE_RE.findall(user):
            groups.setdefault((exp_cat, exp_pri), []).append(ticket)

        new_rules = []
        for (category, priority), tickets in groups.items():
            keywords = _extract_keywords(tickets)
            if not keywords:
                continue
            new_rules.append(
                f'- If the ticket mentions {_format_keyword_list(keywords)} '
                f'-> category={category}, priority={priority}.'
            )
        return _insert_before_default(current_prompt, new_rules, _CLASSIFIER_DEFAULT_RE)

    def _propose_responder(self, current_prompt: str, user: str) -> str:
        phrase_by_category: dict[str, str] = {}
        for _ticket, category, expected_phrase, _actual_reply in _FAILING_RESPONDER_CASE_RE.findall(user):
            phrase_by_category.setdefault(category, expected_phrase)

        new_rules = [
            f'- If category={category} -> include the phrase "{phrase}".'
            for category, phrase in phrase_by_category.items()
        ]
        return _insert_before_default(current_prompt, new_rules, _RESPONDER_DEFAULT_RE)


def _extract_section(text: str, name: str) -> str:
    pattern = _SECTION_RE_TEMPLATE.format(name=re.escape(name))
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip("\n") if match else ""


def _tokenize(text: str) -> list[str]:
    return [tok.lower().strip("'") for tok in _TOKEN_RE.findall(text)]


def _extract_keywords(tickets: list[str], top_n: int = 3) -> list[str]:
    counts: Counter[str] = Counter()
    for ticket in tickets:
        seen = set()
        for token in _tokenize(ticket):
            if len(token) < _MIN_KEYWORD_LEN or token in _STOPWORDS:
                continue
            if token not in seen:
                counts[token] += 1
                seen.add(token)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [word for word, _ in ranked[:top_n]]


def _format_keyword_list(keywords: list[str]) -> str:
    quoted = [f'"{kw}"' for kw in keywords]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + f', or {quoted[-1]}'


def _insert_before_default(prompt: str, new_rules: list[str], default_re: re.Pattern) -> str:
    if not new_rules:
        return prompt
    lines = prompt.splitlines()
    out = []
    inserted = False
    for line in lines:
        if not inserted and default_re.match(line.strip()):
            out.extend(new_rules)
            inserted = True
        out.append(line)
    if not inserted:
        out.extend(new_rules)
    return "\n".join(out)
