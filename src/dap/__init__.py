from dap.runtime.hooks import HookPoint, Decision, StepContext
from dap.runtime.loop import AgentLoop
from dap.constraints.base import Constraint
from dap.constraints.registry import ConstraintRegistry
from dap.trace.events import TraceEvent
from dap.trace.writer import JsonlTracer
from dap.llm.base import LLMProvider, LLMResponse, ToolCall
from dap.llm.mock import MockLLM


def __getattr__(name: str):
    # Lazy import: keep litellm out of base import path so dap[core] stays light.
    if name == "LiteLLMProvider":
        from dap.llm.litellm_provider import LiteLLMProvider
        return LiteLLMProvider
    raise AttributeError(f"module 'dap' has no attribute {name!r}")

__all__ = [
    "HookPoint",
    "Decision",
    "StepContext",
    "AgentLoop",
    "Constraint",
    "ConstraintRegistry",
    "TraceEvent",
    "JsonlTracer",
    "LLMProvider",
    "LLMResponse",
    "ToolCall",
    "MockLLM",
    "LiteLLMProvider",
]
