from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Provider-agnostic response.

    `text` is the assistant's natural-language output (may be empty).
    `tool_calls` is the structured list of tool invocations the model wants.
    `raw` keeps the provider-native object for debugging / trace.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


class LLMProvider(Protocol):
    def call(self, messages: list[dict], tools_spec: list[dict]) -> LLMResponse: ...
