from __future__ import annotations

from typing import Any

from dap.runtime.hooks import Decision, HookPoint, StepContext


class InjectReminder:
    """B-type. After `after_step` steps, prepend a reminder system message
    on every subsequent pre_llm. Demonstrates that B-type (prompt-shaping)
    constraints fit the same hook+Decision interface as A-type.
    """

    name = "inject_reminder"

    def __init__(self, after_step: int = 3, reminder: str = "Stay focused on the task.") -> None:
        self.after_step = after_step
        self.reminder = reminder

    def hooks(self) -> set[HookPoint]:
        return {HookPoint.PRE_LLM}

    def on_event(self, ctx: StepContext) -> Decision:
        if ctx.step_id < self.after_step:
            return Decision.allow()
        new_messages = [{"role": "system", "content": self.reminder}, *ctx.messages]
        return Decision.modify({"messages": new_messages}, reason="reminder injected")

    def state(self) -> dict[str, Any]:
        return {"after_step": self.after_step, "reminder": self.reminder}
