"""Sandbox demo: a tool that sleeps too long is killed by SubprocessExecutor,
and the agent gets an error result it can react to.

Shows:
  - A constraint (Sandbox) swapping the executor per-call.
  - The loop catching the TimeoutError and turning it into a string result.
  - post_tool still firing so other constraints observe the sandbox outcome.

Run:
    PYTHONPATH=src .venv/bin/python examples/sandbox_demo.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dap import AgentLoop, ConstraintRegistry, JsonlTracer, MockLLM, SubprocessExecutor
from dap.constraints.builtin.enforce import Sandbox
from dap.llm.mock import call_tool, text
from dap.runtime.tools import Tool


def nap(seconds: float) -> str:
    time.sleep(seconds)
    return f"slept {seconds}s"


TOOLS = [
    Tool(
        name="nap",
        description="Sleep for N seconds then return.",
        parameters={
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
            "required": ["seconds"],
        },
        func=nap,
    ),
]


def main() -> None:
    # Scripted: model asks to sleep 5s, the sandbox has a 1s budget → timeout →
    # model sees the error and responds.
    llm = MockLLM(
        script=[
            call_tool("c1", "nap", seconds=5.0),
            text("the nap timed out, aborting"),
        ]
    )

    registry = ConstraintRegistry()
    registry.mount(Sandbox(executor=SubprocessExecutor(timeout_s=1.0)))

    trace_path = Path(__file__).parent / "trace_sandbox.jsonl"
    trace_path.unlink(missing_ok=True)
    with JsonlTracer(trace_path) as tracer:
        loop = AgentLoop(llm=llm, tools=TOOLS, registry=registry, tracer=tracer, max_steps=5)
        t0 = time.monotonic()
        result = loop.run("take a nap for 5 seconds")
        elapsed = time.monotonic() - t0

    print("== run result ==")
    print(f"  finished     : {result.finished}")
    print(f"  steps        : {result.steps}")
    print(f"  final_text   : {result.final_text!r}")
    print(f"  elapsed      : {elapsed:.2f}s  (confirms timeout, not real 5s)")
    print()
    print("== tool activity ==")
    for line in trace_path.read_text().splitlines():
        ev = json.loads(line)
        if ev["kind"] in ("tool_call", "tool_result", "constraint_decision"):
            if ev["kind"] == "constraint_decision" and ev["payload"]["action"] == "allow":
                continue
            p = json.dumps(ev["payload"], ensure_ascii=False)
            print(f"  step {ev['step_id']}  {ev['kind']:<22}  {p[:140]}")


if __name__ == "__main__":
    main()
