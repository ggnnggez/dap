from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HookPoint(str, Enum):
    PRE_LLM = "pre_llm"
    POST_LLM = "post_llm"
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"


class Action(str, Enum):
    ALLOW = "allow"
    MODIFY = "modify"
    BLOCK = "block"
    ABORT = "abort"


@dataclass(frozen=True)
class Decision:
    action: Action
    payload: Any = None
    reason: str | None = None
    source: str | None = None  # constraint name; filled by registry

    @classmethod
    def allow(cls) -> "Decision":
        return cls(Action.ALLOW)

    @classmethod
    def modify(cls, payload: Any, reason: str | None = None) -> "Decision":
        return cls(Action.MODIFY, payload=payload, reason=reason)

    @classmethod
    def block(cls, reason: str) -> "Decision":
        return cls(Action.BLOCK, reason=reason)

    @classmethod
    def abort(cls, reason: str) -> "Decision":
        return cls(Action.ABORT, reason=reason)


@dataclass
class StepContext:
    """Mutable per-step context passed to constraints.

    Field population by hook:
      pre_llm:   messages, tools_spec
      post_llm:  messages, tools_spec, llm_response, pending_tool_calls
      pre_tool:  messages, tool_call
      post_tool: messages, tool_call, tool_result
    """

    step_id: int
    hook: HookPoint
    messages: list[dict] = field(default_factory=list)
    tools_spec: list[dict] = field(default_factory=list)
    llm_response: Any = None
    pending_tool_calls: list[Any] = field(default_factory=list)
    tool_call: Any = None
    tool_result: Any = None
    scratch: dict[str, Any] = field(default_factory=dict)
