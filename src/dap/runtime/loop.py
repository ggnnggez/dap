from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dap.constraints.registry import ConstraintRegistry
from dap.llm.base import LLMProvider, LLMResponse, ToolCall
from dap.runtime.hooks import Action, Decision, HookPoint, StepContext
from dap.runtime.tools import Tool
from dap.trace.events import EventKind, TraceEvent
from dap.trace.writer import NullTracer, Tracer


@dataclass
class RunResult:
    finished: bool
    final_text: str | None
    abort_reason: str | None
    steps: int
    messages: list[dict] = field(default_factory=list)


class AgentLoop:
    """Step-driven agent loop with four hook points.

    Per step:
        pre_llm  -> llm.call -> post_llm -> {pre_tool -> exec -> post_tool}*
    Constraints can ALLOW / MODIFY / BLOCK / ABORT at any hook.
    BLOCK at pre_tool skips that single tool. BLOCK at any other hook is
    treated as ABORT for v0 (semantics for finer-grained block at other
    hooks can be added later when a real use case demands it).
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: list[Tool],
        registry: ConstraintRegistry | None = None,
        tracer: Tracer | None = None,
        max_steps: int = 25,
    ) -> None:
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.tools_spec = [t.to_spec() for t in tools]
        self.registry = registry or ConstraintRegistry()
        self.tracer = tracer or NullTracer()
        self.max_steps = max_steps
        self._step_id = 0

    def run(self, task: str) -> RunResult:
        messages: list[dict] = [{"role": "user", "content": task}]
        self._emit(EventKind.RUN_START, {"task": task, "constraints": self.registry.names()})

        last_text = ""
        for step in range(self.max_steps):
            self._step_id = step
            self._emit(EventKind.STEP_START, {})

            # ----- pre_llm -----
            ctx = StepContext(
                step_id=step,
                hook=HookPoint.PRE_LLM,
                messages=messages,
                tools_spec=self.tools_spec,
            )
            res = self._fire(HookPoint.PRE_LLM, ctx)
            if res.final.action == Action.ABORT:
                return self._abort(res.final, step, messages)
            messages = ctx.messages  # may have been mutated by MODIFY
            tools_spec = ctx.tools_spec

            # ----- llm.call -----
            self._emit(EventKind.LLM_CALL, {"messages": messages, "tools_spec": tools_spec})
            response = self.llm.call(messages, tools_spec)
            self._emit(
                EventKind.LLM_RESPONSE,
                {"text": response.text, "tool_calls": [_tc_dict(t) for t in response.tool_calls]},
            )
            messages.append(_assistant_message(response))
            last_text = response.text or last_text

            # ----- post_llm -----
            ctx = StepContext(
                step_id=step,
                hook=HookPoint.POST_LLM,
                messages=messages,
                tools_spec=tools_spec,
                llm_response=response,
                pending_tool_calls=list(response.tool_calls),
            )
            res = self._fire(HookPoint.POST_LLM, ctx)
            if res.final.action == Action.ABORT:
                return self._abort(res.final, step, messages)
            tool_calls = ctx.pending_tool_calls

            # natural termination: no tool calls requested
            if not tool_calls:
                self._emit(EventKind.STEP_END, {"reason": "no_tool_calls"})
                self._emit(EventKind.RUN_END, {"finished": True})
                return RunResult(
                    finished=True,
                    final_text=last_text,
                    abort_reason=None,
                    steps=step + 1,
                    messages=messages,
                )

            # ----- per-tool: pre_tool -> exec -> post_tool -----
            for tc in tool_calls:
                ctx = StepContext(
                    step_id=step,
                    hook=HookPoint.PRE_TOOL,
                    messages=messages,
                    tool_call=tc,
                )
                res = self._fire(HookPoint.PRE_TOOL, ctx)
                if res.final.action == Action.ABORT:
                    return self._abort(res.final, step, messages)
                if res.final.action == Action.BLOCK:
                    # skip this tool; tell the model why
                    blocked_msg = f"[blocked by {res.final.source}: {res.final.reason}]"
                    messages.append(_tool_result_message(tc, blocked_msg))
                    continue
                tc = ctx.tool_call

                self._emit(
                    EventKind.TOOL_CALL,
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments},
                )
                try:
                    result = self.tools[tc.name].func(**tc.arguments)
                except Exception as e:
                    result = f"[tool error: {type(e).__name__}: {e}]"
                self._emit(
                    EventKind.TOOL_RESULT,
                    {"id": tc.id, "name": tc.name, "result": _stringify(result)},
                )

                ctx = StepContext(
                    step_id=step,
                    hook=HookPoint.POST_TOOL,
                    messages=messages,
                    tool_call=tc,
                    tool_result=result,
                )
                res = self._fire(HookPoint.POST_TOOL, ctx)
                if res.final.action == Action.ABORT:
                    return self._abort(res.final, step, messages)
                result = ctx.tool_result
                messages.append(_tool_result_message(tc, _stringify(result)))

            self._emit(EventKind.STEP_END, {})

        # ran out of steps
        reason = f"max_steps ({self.max_steps}) reached"
        self._emit(EventKind.RUN_END, {"finished": False, "reason": reason})
        return RunResult(
            finished=False,
            final_text=last_text,
            abort_reason=reason,
            steps=self.max_steps,
            messages=messages,
        )

    # ----- internals -----

    def _fire(self, hook: HookPoint, ctx: StepContext):
        def record(d: Decision, name: str) -> None:
            self._emit(
                EventKind.CONSTRAINT_DECISION,
                {
                    "hook": hook.value,
                    "constraint": name,
                    "action": d.action.value,
                    "reason": d.reason,
                },
            )
        return self.registry.fire(hook, ctx, on_decision=record)

    def _emit(self, kind: EventKind, payload: dict[str, Any]) -> None:
        self.tracer.emit(TraceEvent(kind=kind, step_id=self._step_id, payload=payload))

    def _abort(self, decision: Decision, step: int, messages: list[dict]) -> RunResult:
        reason = f"{decision.source}: {decision.reason}"
        self._emit(EventKind.RUN_END, {"finished": False, "reason": reason})
        return RunResult(
            finished=False,
            final_text=None,
            abort_reason=reason,
            steps=step + 1,
            messages=messages,
        )


# ----- message helpers -----

def _assistant_message(resp: LLMResponse) -> dict:
    msg: dict = {"role": "assistant", "content": resp.text}
    if resp.tool_calls:
        msg["tool_calls"] = [_tc_dict(t) for t in resp.tool_calls]
    return msg


def _tool_result_message(tc: ToolCall, result: str) -> dict:
    return {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": result}


def _tc_dict(tc: ToolCall) -> dict:
    return {"id": tc.id, "name": tc.name, "arguments": tc.arguments}


def _stringify(x: Any) -> str:
    if isinstance(x, str):
        return x
    return repr(x)
