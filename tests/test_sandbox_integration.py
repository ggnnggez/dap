"""End-to-end: Sandbox constraint rewires tool execution through the loop.

We don't exercise real subprocesses here (covered in test_executor.py) —
these tests pin down the wiring: pre_tool MODIFY with an executor key must
override the loop's default, per-call, honoring applies_to filters.
"""

from __future__ import annotations

from typing import Any

from dap import AgentLoop, ConstraintRegistry
from dap.constraints.builtin import Sandbox
from dap.llm.mock import MockLLM, call_tool, text
from dap.runtime.executor import DirectExecutor
from dap.runtime.tools import Tool


class SpyExecutor:
    """Records every execute() call but delegates to DirectExecutor."""

    def __init__(self, label: str = "spy") -> None:
        self.label = label
        self.calls: list[tuple[str, dict]] = []
        self._inner = DirectExecutor()

    def execute(self, tool: Tool, arguments: dict) -> Any:
        self.calls.append((tool.name, dict(arguments)))
        return self._inner.execute(tool, arguments)


def _tool(name: str, func) -> Tool:
    return Tool(name=name, description="", parameters={"type": "object"}, func=func)


def _add(a: int, b: int) -> int:
    return a + b


def _concat(x: str, y: str) -> str:
    return x + y


def test_no_sandbox_uses_default_executor():
    default_spy = SpyExecutor("default")
    llm = MockLLM(script=[call_tool("c1", "add", a=1, b=2), text("ok")])
    loop = AgentLoop(
        llm=llm,
        tools=[_tool("add", _add)],
        default_executor=default_spy,
    )
    loop.run("go")
    assert default_spy.calls == [("add", {"a": 1, "b": 2})]


def test_sandbox_constraint_overrides_default_for_matching_tool():
    default_spy = SpyExecutor("default")
    sandbox_spy = SpyExecutor("sandbox")

    registry = ConstraintRegistry()
    registry.mount(Sandbox(executor=sandbox_spy))

    llm = MockLLM(script=[call_tool("c1", "add", a=1, b=2), text("ok")])
    loop = AgentLoop(
        llm=llm,
        tools=[_tool("add", _add)],
        registry=registry,
        default_executor=default_spy,
    )
    loop.run("go")
    assert sandbox_spy.calls == [("add", {"a": 1, "b": 2})]
    assert default_spy.calls == []  # nothing touched the default


def test_sandbox_applies_to_filters_per_tool():
    default_spy = SpyExecutor("default")
    sandbox_spy = SpyExecutor("sandbox")

    registry = ConstraintRegistry()
    registry.mount(
        Sandbox(
            executor=sandbox_spy,
            applies_to=lambda tc: tc.name == "concat",  # only string tool
        )
    )

    llm = MockLLM(
        script=[
            call_tool("c1", "add", a=1, b=2),         # should go to default
            call_tool("c2", "concat", x="hi", y="!"), # should go to sandbox
            text("done"),
        ]
    )
    loop = AgentLoop(
        llm=llm,
        tools=[_tool("add", _add), _tool("concat", _concat)],
        registry=registry,
        default_executor=default_spy,
    )
    loop.run("go")
    assert default_spy.calls == [("add", {"a": 1, "b": 2})]
    assert sandbox_spy.calls == [("concat", {"x": "hi", "y": "!"})]


def test_override_does_not_leak_across_tool_calls():
    """StepContext is rebuilt per pre_tool fire; an override in one call
    must not bleed into the next when the predicate no longer matches."""
    default_spy = SpyExecutor("default")
    sandbox_spy = SpyExecutor("sandbox")

    calls_seen: list[str] = []

    registry = ConstraintRegistry()
    registry.mount(
        Sandbox(
            executor=sandbox_spy,
            applies_to=lambda tc: (calls_seen.append(tc.name) or tc.name == "concat"),
        )
    )
    llm = MockLLM(
        script=[
            call_tool("c1", "concat", x="a", y="b"),  # sandbox
            call_tool("c2", "add", a=1, b=2),         # default
            text("done"),
        ]
    )
    loop = AgentLoop(
        llm=llm,
        tools=[_tool("add", _add), _tool("concat", _concat)],
        registry=registry,
        default_executor=default_spy,
    )
    loop.run("go")
    assert sandbox_spy.calls == [("concat", {"x": "a", "y": "b"})]
    assert default_spy.calls == [("add", {"a": 1, "b": 2})]
    assert calls_seen == ["concat", "add"]  # predicate invoked each call
