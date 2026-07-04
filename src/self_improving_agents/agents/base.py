class Agent:
    """A named agent whose behavior is entirely defined by its editable prompt."""

    def __init__(self, name: str, prompt: str) -> None:
        self.name = name
        self.prompt = prompt
