"""Real-LLM demo using LiteLLM. Defaults to Claude via Anthropic.

    ANTHROPIC_API_KEY=sk-ant-... .venv/bin/python examples/litellm_agent.py

Override the model at the command line:

    DAP_MODEL=openai/gpt-4o OPENAI_API_KEY=... .venv/bin/python examples/litellm_agent.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dap import AgentLoop, ConstraintRegistry, JsonlTracer, LiteLLMProvider
from dap.constraints.builtin import InjectReminder, LoopDetector, MaxToolCalls
from dap.runtime.tools import Tool


def add(a: float, b: float) -> float:
    return a + b


def multiply(a: float, b: float) -> float:
    return a * b


TOOLS = [
    Tool(
        name="add",
        description="Add two numbers and return the sum.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        func=add,
    ),
    Tool(
        name="multiply",
        description="Multiply two numbers and return the product.",
        parameters={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
        func=multiply,
    ),
]


def _has_credentials_for(model: str) -> bool:
    if model.startswith("anthropic/"):
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if model.startswith("openai/"):
        return bool(os.environ.get("OPENAI_API_KEY"))
    return True  # let LiteLLM raise if it can't auth


def main() -> int:
    model = os.environ.get("DAP_MODEL", "anthropic/claude-sonnet-4-6")
    if not _has_credentials_for(model):
        print(f"[skip] no API credentials for {model}; set the relevant *_API_KEY env var.")
        return 0

    provider = LiteLLMProvider(model=model, temperature=0.0)

    registry = ConstraintRegistry()
    registry.mount(MaxToolCalls(limit=8))
    registry.mount(LoopDetector(window=5, threshold=3))
    registry.mount(InjectReminder(after_step=4, reminder="Wrap up: produce the final answer now."))

    trace_path = Path(__file__).parent / "trace_litellm.jsonl"
    trace_path.unlink(missing_ok=True)
    with JsonlTracer(trace_path) as tracer:
        loop = AgentLoop(provider, TOOLS, registry=registry, tracer=tracer, max_steps=8)
        result = loop.run("What is (3 + 4) * 5? Use the tools.")

    print("== run result ==")
    print(f"  finished     : {result.finished}")
    print(f"  steps        : {result.steps}")
    print(f"  final_text   : {result.final_text}")
    print(f"  abort_reason : {result.abort_reason}")
    print(f"  trace        : {trace_path}")

    print("\n== tool activity ==")
    for line in trace_path.read_text().splitlines():
        ev = json.loads(line)
        if ev["kind"] in ("tool_call", "tool_result", "constraint_decision"):
            if ev["kind"] == "constraint_decision" and ev["payload"]["action"] == "allow":
                continue
            print(f"  step {ev['step_id']}  {ev['kind']:<22}  {json.dumps(ev['payload'], ensure_ascii=False)[:120]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
