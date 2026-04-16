from __future__ import annotations

from typing import Any, Callable

from dap.runtime.executor import ToolExecutor
from dap.runtime.hooks import Decision, HookPoint, StepContext


class Sandbox:
    """A-type. Routes matching tool calls through a long-lived ToolExecutor.

    Owning the executor on the constraint instance (not per-call) matters:
    container/VM-backed executors have expensive startup that must not be
    paid on every tool call. Mount one Sandbox per isolation boundary.

    `applies_to` is a predicate on the ToolCall. Defaults to all tools.
    Mount multiple Sandbox instances with different predicates to route
    different tools through different isolation levels.
    """

    name = "sandbox"

    def __init__(
        self,
        executor: ToolExecutor,
        applies_to: Callable[[Any], bool] | None = None,
        name: str | None = None,
    ) -> None:
        if name is not None:
            self.name = name
        self._executor = executor
        self._applies_to = applies_to or (lambda _tc: True)

    def hooks(self) -> set[HookPoint]:
        return {HookPoint.PRE_TOOL}

    def on_event(self, ctx: StepContext) -> Decision:
        if not self._applies_to(ctx.tool_call):
            return Decision.allow()
        return Decision.modify(
            {"executor": self._executor},
            reason=f"sandboxed via {type(self._executor).__name__}",
        )

    def state(self) -> dict[str, Any]:
        return {"executor": type(self._executor).__name__}
