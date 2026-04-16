from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    STEP_START = "step_start"
    STEP_END = "step_end"
    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CONSTRAINT_DECISION = "constraint_decision"
    CONSTRAINT_MOUNT = "constraint_mount"
    CONSTRAINT_UNMOUNT = "constraint_unmount"


@dataclass
class TraceEvent:
    kind: EventKind
    step_id: int
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d
