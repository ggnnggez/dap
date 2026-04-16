"""Smoke demo: a scripted agent that intentionally loops. LoopAdvisory
(advise-category) inserts a blocked-notification tool_result on the 3rd
repeated call; the scripted model then 'notices' and answers. InjectReminder
shows up in the trace on subsequent pre_llm turns.

This demo picks LoopAdvisory rather than enforce.LoopGuard on purpose: the
point is to show the prompt-mediated path working end-to-end. For the
runtime-enforced counterpart, see loop_demo.py.

Run:
    python -m examples.hello_agent
or:
    PYTHONPATH=src python examples/hello_agent.py
"""

from __future__ import annotations

import json
from pathlib import Path

from dap import (
    AgentLoop,
    ConstraintRegistry,
    JsonlTracer,
    MockLLM,
)
from dap.constraints.builtin.advise import InjectReminder, LoopAdvisory
from dap.constraints.builtin.enforce import MaxToolCalls
from dap.llm.mock import call_tool, text
from dap.runtime.tools import Tool


def add(a: int, b: int) -> int:
    return a + b


def main() -> None:
    tools = [
        Tool(
            name="add",
            description="Add two integers.",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
            func=add,
        )
    ]

    # Scripted trajectory: model insists on calling add(1,2) three times,
    # then finally answers. LoopAdvisory should block the 3rd call via a
    # [blocked by loop_advisory: ...] tool_result.
    llm = MockLLM(
        script=[
            call_tool("c1", "add", a=1, b=2),
            call_tool("c2", "add", a=1, b=2),
            call_tool("c3", "add", a=1, b=2),  # blocked (prompt-mediated)
            text("the answer is 3"),
        ]
    )

    registry = ConstraintRegistry()
    registry.mount(MaxToolCalls(limit=10))
    registry.mount(LoopAdvisory(window=5, threshold=3))
    registry.mount(InjectReminder(after_step=2, reminder="Stop repeating; produce the final answer."))

    trace_path = Path(__file__).parent / "trace.jsonl"
    trace_path.unlink(missing_ok=True)
    with JsonlTracer(trace_path) as tracer:
        loop = AgentLoop(llm=llm, tools=tools, registry=registry, tracer=tracer, max_steps=10)
        result = loop.run("compute 1 + 2")

    print("== run result ==")
    print(f"  finished      : {result.finished}")
    print(f"  steps         : {result.steps}")
    print(f"  final_text    : {result.final_text!r}")
    print(f"  abort_reason  : {result.abort_reason}")
    print()
    print(f"== trace written to {trace_path} ==")
    print("== last 6 trace events ==")
    lines = trace_path.read_text().splitlines()[-6:]
    for line in lines:
        ev = json.loads(line)
        print(f"  step {ev['step_id']:>2}  {ev['kind']:<22}  {_short(ev['payload'])}")


def _short(payload: dict, n: int = 90) -> str:
    s = json.dumps(payload, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 1] + "…"


if __name__ == "__main__":
    main()
