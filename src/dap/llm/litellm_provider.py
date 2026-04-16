"""LiteLLM-backed provider.

LiteLLM normalizes ~all major provider APIs onto the OpenAI chat-completions
shape, so this module only needs to translate between dap's internal message /
tool format and OpenAI format.

Internal vs OpenAI:
  - Internal assistant-with-tool-calls:
        {"role": "assistant", "content": str, "tool_calls": [
            {"id", "name", "arguments": dict}
        ]}
  - OpenAI assistant-with-tool-calls:
        {"role": "assistant", "content": str | None, "tool_calls": [
            {"id", "type": "function",
             "function": {"name", "arguments": <JSON-encoded string>}}
        ]}
  - tool / user / system messages are identical in both shapes.

  - Internal tool spec:    {"name", "description", "parameters"}
  - OpenAI tool spec:      {"type": "function", "function": {...same three...}}
"""

from __future__ import annotations

import json
from typing import Any

from dap.llm.base import LLMProvider, LLMResponse, ToolCall


class LiteLLMProvider(LLMProvider):
    """Provider backed by `litellm.completion`. Routes to whatever provider
    LiteLLM resolves from the model string (anthropic/..., openai/..., etc.).
    Extra kwargs (temperature, max_tokens, ...) are forwarded on every call.
    """

    def __init__(self, model: str, **default_params: Any) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "litellm is not installed. Install with: pip install 'dap[litellm]'"
            ) from e
        self.model = model
        self.default_params = default_params

    def call(self, messages: list[dict], tools_spec: list[dict]) -> LLMResponse:
        import litellm

        kwargs: dict[str, Any] = dict(self.default_params)
        kwargs["model"] = self.model
        kwargs["messages"] = [_to_openai_message(m) for m in messages]
        if tools_spec:
            kwargs["tools"] = [_to_openai_tool(t) for t in tools_spec]

        resp = litellm.completion(**kwargs)
        return _from_litellm_response(resp)


def _to_openai_message(m: dict) -> dict:
    role = m["role"]
    if role == "assistant" and m.get("tool_calls"):
        return {
            "role": "assistant",
            "content": m.get("content") or None,
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for tc in m["tool_calls"]
            ],
        }
    if role == "tool":
        return {
            "role": "tool",
            "tool_call_id": m["tool_call_id"],
            "content": m["content"],
        }
    return {"role": role, "content": m["content"]}


def _to_openai_tool(spec: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": spec.get("description", ""),
            "parameters": spec.get("parameters", {"type": "object", "properties": {}}),
        },
    }


def _from_litellm_response(resp: Any) -> LLMResponse:
    msg = resp.choices[0].message
    text = getattr(msg, "content", None) or ""
    tool_calls: list[ToolCall] = []
    raw_tcs = getattr(msg, "tool_calls", None) or []
    for tc in raw_tcs:
        args_raw = tc.function.arguments
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
        else:
            args = args_raw or {}
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
    return LLMResponse(text=text, tool_calls=tool_calls, raw=resp)
