from __future__ import annotations

from typing import Any

from dap.runtime.hooks import Decision, HookPoint, StepContext


class MaxToolCalls:
    """Enforce. Abort the run once total tool calls exceed `limit`.

    Enforcement is runtime-level: Decision.abort terminates AgentLoop.run
    regardless of what the model does next. The model cannot "retry through"
    this constraint.
    """

    name = "max_tool_calls"

    def __init__(self, limit: int = 10) -> None:
        self.limit = limit
        self._count = 0

    def hooks(self) -> set[HookPoint]:
        return {HookPoint.PRE_TOOL}

    def on_event(self, ctx: StepContext) -> Decision:
        self._count += 1
        if self._count > self.limit:
            return Decision.abort(f"tool call budget exhausted ({self.limit})")
        return Decision.allow()

    def state(self) -> dict[str, Any]:
        return {"count": self._count, "limit": self.limit}

    def update_params(self, params: dict[str, Any]) -> None:
        if "limit" in params:
            self.limit = int(params["limit"])
