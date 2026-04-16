from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from dap.constraints.base import Constraint
from dap.runtime.hooks import Action, Decision, HookPoint, StepContext


@dataclass
class FireResult:
    """Aggregated outcome of firing a hook across all subscribed constraints.

    Resolution rule: ABORT > BLOCK > MODIFY (chained) > ALLOW.
    Decisions are returned in fire order so the loop can record provenance.
    """

    final: Decision
    decisions: list[Decision] = field(default_factory=list)


class ConstraintRegistry:
    """Holds mounted constraints. Supports hot mount/unmount/update_params.

    Mutations apply at the next hook fire — there is no in-flight
    invalidation, which keeps trace causality clean.
    """

    def __init__(self) -> None:
        self._constraints: dict[str, Constraint] = {}

    def mount(self, constraint: Constraint) -> None:
        if constraint.name in self._constraints:
            raise ValueError(f"constraint already mounted: {constraint.name}")
        self._constraints[constraint.name] = constraint

    def unmount(self, name: str) -> None:
        self._constraints.pop(name, None)

    def update_params(self, name: str, params: dict[str, Any]) -> None:
        self._constraints[name].update_params(params)

    def get(self, name: str) -> Constraint:
        return self._constraints[name]

    def names(self) -> list[str]:
        return list(self._constraints)

    def fire(
        self,
        hook: HookPoint,
        ctx: StepContext,
        on_decision: Callable[[Decision, str], None] | None = None,
    ) -> FireResult:
        decisions: list[Decision] = []
        final = Decision.allow()
        for name, c in list(self._constraints.items()):
            if hook not in c.hooks():
                continue
            d = c.on_event(ctx)
            d = Decision(d.action, d.payload, d.reason, source=name)
            decisions.append(d)
            if on_decision is not None:
                on_decision(d, name)
            if d.action == Action.ABORT:
                return FireResult(final=d, decisions=decisions)
            if d.action == Action.BLOCK:
                # Block short-circuits remaining constraints for this hook.
                return FireResult(final=d, decisions=decisions)
            if d.action == Action.MODIFY:
                # Apply payload immediately so downstream constraints see it.
                _apply_modify(hook, ctx, d.payload)
                final = d
        return FireResult(final=final, decisions=decisions)


def _apply_modify(hook: HookPoint, ctx: StepContext, payload: Any) -> None:
    """Map a MODIFY payload onto the StepContext for the given hook.

    Payload shape per hook:
      pre_llm:   {"messages": [...]} or {"messages": [...], "tools_spec": [...]}
      post_llm:  {"pending_tool_calls": [...]}
      pre_tool:  {"tool_call": {...}}
      post_tool: {"tool_result": {...}}
    """
    if not isinstance(payload, dict):
        raise TypeError(f"MODIFY payload must be dict, got {type(payload)}")
    if hook == HookPoint.PRE_LLM:
        if "messages" in payload:
            ctx.messages = payload["messages"]
        if "tools_spec" in payload:
            ctx.tools_spec = payload["tools_spec"]
    elif hook == HookPoint.POST_LLM:
        if "pending_tool_calls" in payload:
            ctx.pending_tool_calls = payload["pending_tool_calls"]
    elif hook == HookPoint.PRE_TOOL:
        if "tool_call" in payload:
            ctx.tool_call = payload["tool_call"]
        if "executor" in payload:
            ctx.executor_override = payload["executor"]
    elif hook == HookPoint.POST_TOOL:
        if "tool_result" in payload:
            ctx.tool_result = payload["tool_result"]
