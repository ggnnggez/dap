"""Enforce-category constraints: direct runtime-level enforcement.

A constraint belongs here only if its effect does NOT depend on the LLM
reading a message and adapting. Mechanisms that qualify:

  - Decision.abort  → AgentLoop terminates the run.
  - Decision.modify with payload routed to runtime state (e.g. executor swap,
    tool-call argument coercion) that changes execution without informing
    the model.

If the constraint's tangible effect is "a message shows up in the next LLM
input", it is NOT enforce. Put it in advise/ instead.
"""

from dap.constraints.builtin.enforce.loop_guard import LoopGuard
from dap.constraints.builtin.enforce.max_tool_calls import MaxToolCalls
from dap.constraints.builtin.enforce.sandbox import Sandbox

__all__ = ["LoopGuard", "MaxToolCalls", "Sandbox"]
