"""Real-LLM constraint-trip demo: a tool that pretends to be rate-limited,
a prompt that tells the model to retry the same query. LoopGuard
(enforce-category) aborts the run on the third identical call, regardless
of what the model does next.

This demo deliberately uses enforce.LoopGuard rather than advise.LoopAdvisory:
the whole point is that a real LLM told to "retry the exact same query" will
cheerfully ignore a prompt-mediated block notification. LoopGuard's abort
terminates the loop before the model gets another turn — no cooperation
required. Compare the trace against an advise-based variant to see the
difference in enforcement semantics.

This is the "real LLM hits a real dap constraint" evidence run — the
counterpart to litellm_agent.py (happy path) and hello_agent.py (scripted).

    .venv/bin/python examples/loop_demo.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from dap import AgentLoop, ConstraintRegistry, JsonlTracer, LiteLLMProvider
from dap.constraints.builtin.enforce import LoopGuard, MaxToolCalls
from dap.runtime.tools import Tool

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)


def flaky_api(query: str) -> str:
    """Pretends to be rate-limited regardless of input — bait for a loop."""
    return "ERROR 429: rate limited. Please retry the same request shortly."


TOOLS = [
    Tool(
        name="flaky_api",
        description="Query the data API. May fail transiently with rate-limit errors.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        func=flaky_api,
    ),
]


def _has_credentials_for(model: str) -> bool:
    if model.startswith("anthropic/"):
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if model.startswith("openai/"):
        return bool(os.environ.get("OPENAI_API_KEY"))
    if model.startswith("moonshot/"):
        return bool(os.environ.get("MOONSHOT_API_KEY"))
    return True


def main() -> int:
    model = os.environ.get("DAP_MODEL", "anthropic/claude-sonnet-4-6")
    if not _has_credentials_for(model):
        print(f"[skip] no API credentials for {model}.")
        return 0

    provider = LiteLLMProvider(model=model)

    registry = ConstraintRegistry()
    # MaxToolCalls left high so LoopGuard is what bites, not the budget.
    registry.mount(MaxToolCalls(limit=15))
    registry.mount(LoopGuard(window=5, threshold=3))

    trace_path = Path(__file__).parent / "trace_loop.jsonl"
    trace_path.unlink(missing_ok=True)
    with JsonlTracer(trace_path) as tracer:
        loop = AgentLoop(provider, TOOLS, registry=registry, tracer=tracer, max_steps=12)
        result = loop.run(
            "Query the data API for the value of 'foo' using the flaky_api tool. "
            "If you get a rate-limit error, retry with the EXACT SAME query — do "
            "not vary the query string. Keep retrying until you get a real value."
        )

    print("== run result ==")
    print(f"  finished     : {result.finished}")
    print(f"  steps        : {result.steps}")
    print(f"  abort_reason : {result.abort_reason}")
    print(f"  final_text   : {result.final_text!r}")
    print(f"  trace        : {trace_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
