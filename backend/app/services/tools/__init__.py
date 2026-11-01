"""What a tool is, for the agent loop."""
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ToolOutput:
    content: str  # what the model will read
    sources: list[dict] = field(default_factory=list)  # e.g. KB chunks, for citations


@dataclass(frozen=True)
class Tool:
    name: str
    description: str  # the model reads this to decide WHEN to use the tool: write it well
    parameters: dict  # JSON Schema of the arguments
    run: Callable[[dict], ToolOutput]

    def schema(self) -> dict:
        """The tool as the OpenAI format describes it to the model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
