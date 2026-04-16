"""Translation-layer tests for LiteLLMProvider.

We don't make real API calls — `litellm.completion` is monkeypatched. The
goal is to pin down the format contract at the provider boundary so that
rewiring providers later doesn't silently break the agent loop.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from dap.llm.base import ToolCall
from dap.llm.litellm_provider import (
    LiteLLMProvider,
    _from_litellm_response,
    _to_openai_message,
    _to_openai_tool,
)


# ---------- outbound translation ----------

def test_user_message_passthrough():
    assert _to_openai_message({"role": "user", "content": "hi"}) == {
        "role": "user",
        "content": "hi",
    }


def test_assistant_with_tool_calls_reshaped_and_args_json_encoded():
    internal = {
        "role": "assistant",
        "content": "thinking…",
        "tool_calls": [
            {"id": "c1", "name": "add", "arguments": {"a": 1, "b": 2}},
        ],
    }
    out = _to_openai_message(internal)
    assert out["role"] == "assistant"
    assert out["content"] == "thinking…"
    assert len(out["tool_calls"]) == 1
    tc = out["tool_calls"][0]
    assert tc["id"] == "c1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "add"
    # arguments must be a JSON string per OpenAI spec
    assert isinstance(tc["function"]["arguments"], str)
    assert json.loads(tc["function"]["arguments"]) == {"a": 1, "b": 2}


def test_assistant_with_empty_content_becomes_none():
    out = _to_openai_message({
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "x", "name": "n", "arguments": {}}],
    })
    assert out["content"] is None


def test_tool_result_message_passthrough():
    internal = {"role": "tool", "tool_call_id": "c1", "name": "add", "content": "3"}
    out = _to_openai_message(internal)
    assert out == {"role": "tool", "tool_call_id": "c1", "content": "3"}


def test_to_openai_tool_wraps_in_function_envelope():
    spec = {"name": "add", "description": "Add.", "parameters": {"type": "object"}}
    out = _to_openai_tool(spec)
    assert out == {
        "type": "function",
        "function": {"name": "add", "description": "Add.", "parameters": {"type": "object"}},
    }


# ---------- inbound translation ----------

def _fake_response(content: str | None, tool_calls: list[dict] | None = None):
    """Build a litellm-like response object with attribute access."""
    msg = SimpleNamespace(
        content=content,
        tool_calls=[
            SimpleNamespace(
                id=tc["id"],
                function=SimpleNamespace(name=tc["name"], arguments=tc["arguments"]),
            )
            for tc in (tool_calls or [])
        ] or None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_from_response_text_only():
    r = _fake_response(content="hello")
    out = _from_litellm_response(r)
    assert out.text == "hello"
    assert out.tool_calls == []


def test_from_response_parses_json_string_arguments():
    r = _fake_response(
        content=None,
        tool_calls=[{"id": "c1", "name": "add", "arguments": '{"a": 1, "b": 2}'}],
    )
    out = _from_litellm_response(r)
    assert out.text == ""
    assert out.tool_calls == [ToolCall(id="c1", name="add", arguments={"a": 1, "b": 2})]


def test_from_response_handles_malformed_json_gracefully():
    r = _fake_response(
        content=None,
        tool_calls=[{"id": "c1", "name": "add", "arguments": "{not json"}],
    )
    out = _from_litellm_response(r)
    assert out.tool_calls[0].arguments == {"_raw": "{not json"}


# ---------- end-to-end with monkeypatched litellm ----------

def test_provider_call_pipes_translated_request_and_returns_normalized_response(monkeypatch):
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response(
            content=None,
            tool_calls=[{"id": "c1", "name": "add", "arguments": '{"a": 5, "b": 7}'}],
        )

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)

    provider = LiteLLMProvider(model="anthropic/claude-sonnet-4-6", temperature=0.0)
    messages = [
        {"role": "user", "content": "5+7?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c0", "name": "add", "arguments": {"a": 5, "b": 7}}],
        },
        {"role": "tool", "tool_call_id": "c0", "name": "add", "content": "12"},
    ]
    tools_spec = [{"name": "add", "description": "", "parameters": {"type": "object"}}]

    resp = provider.call(messages, tools_spec)

    # request side: model, default params, translated messages and tools
    assert captured["model"] == "anthropic/claude-sonnet-4-6"
    assert captured["temperature"] == 0.0
    assert captured["tools"][0]["type"] == "function"
    assert captured["messages"][1]["tool_calls"][0]["function"]["name"] == "add"
    assert captured["messages"][2]["role"] == "tool"

    # response side
    assert resp.tool_calls == [ToolCall(id="c1", name="add", arguments={"a": 5, "b": 7})]


def test_import_error_message_when_litellm_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "litellm":
            raise ImportError("no module named litellm")
        return real_import(name, *a, **kw)

    # Force the constructor's lazy import to fail.
    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="dap\\[litellm\\]"):
        LiteLLMProvider(model="anthropic/claude-sonnet-4-6")
