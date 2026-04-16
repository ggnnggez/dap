from __future__ import annotations

import json
from collections import deque
from typing import Any

from dap.runtime.hooks import Decision, HookPoint, StepContext


class LoopDetector:
    """A-type. Block a tool call if the same (name, arguments) has fired
    `threshold` times within the most recent `window` tool calls.

    Block (not abort): the model gets a blocked-result message and may adapt.
    """

    name = "loop_detector"

    def __init__(self, window: int = 5, threshold: int = 3) -> None:
        self.window = window
        self.threshold = threshold
        self._recent: deque[str] = deque(maxlen=window)

    def hooks(self) -> set[HookPoint]:
        return {HookPoint.PRE_TOOL}

    def on_event(self, ctx: StepContext) -> Decision:
        tc = ctx.tool_call
        key = _key(tc.name, tc.arguments)
        count = sum(1 for k in self._recent if k == key) + 1
        self._recent.append(key)
        if count >= self.threshold:
            return Decision.block(
                f"repeated call {tc.name}({tc.arguments}) hit {count}/{self.threshold} in last {self.window}"
            )
        return Decision.allow()

    def state(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "threshold": self.threshold,
            "recent": list(self._recent),
        }

    def update_params(self, params: dict[str, Any]) -> None:
        if "window" in params:
            self.window = int(params["window"])
            new_q: deque[str] = deque(self._recent, maxlen=self.window)
            self._recent = new_q
        if "threshold" in params:
            self.threshold = int(params["threshold"])


def _key(name: str, args: dict) -> str:
    return name + "|" + json.dumps(args, sort_keys=True, default=repr)
