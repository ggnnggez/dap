from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from dap.runtime.hooks import Decision, HookPoint, StepContext


@runtime_checkable
class Constraint(Protocol):
    """A pluggable behavior-shaping module.

    Implementations declare which hook points they care about via `hooks()`,
    receive a StepContext at each fire, and return a Decision. They may carry
    serializable state in `state()` so it can be captured in trace events.
    """

    name: str

    def hooks(self) -> set[HookPoint]: ...

    def on_event(self, ctx: StepContext) -> Decision: ...

    def state(self) -> dict[str, Any]:
        return {}

    def update_params(self, params: dict[str, Any]) -> None:
        """Hot-update parameters. Default: assign each key as attribute."""
        for k, v in params.items():
            setattr(self, k, v)
