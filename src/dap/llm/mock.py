from __future__ import annotations

from typing import Iterable

from dap.llm.base import LLMProvider, LLMResponse, ToolCall


class MockLLM(LLMProvider):
    """Deterministic provider that replays a scripted sequence of responses.

    Each script entry is one LLMResponse. After the script is exhausted, raises.
    Useful for demos and tests where we want full control over agent trajectory.
    """

    def __init__(self, script: Iterable[LLMResponse]) -> None:
        self._script = list(script)
        self._cursor = 0
        self.calls: list[tuple[list[dict], list[dict]]] = []

    def call(self, messages: list[dict], tools_spec: list[dict]) -> LLMResponse:
        self.calls.append((messages, tools_spec))
        if self._cursor >= len(self._script):
            raise RuntimeError("MockLLM script exhausted")
        resp = self._script[self._cursor]
        self._cursor += 1
        return resp


def text(s: str) -> LLMResponse:
    return LLMResponse(text=s)


def call_tool(tool_id: str, name: str, **arguments) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(id=tool_id, name=name, arguments=arguments)])
