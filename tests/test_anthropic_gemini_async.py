"""GA-W2 测试 —— Anthropic + Gemini 适配器 真异步路径

venv 没装 anthropic / google-genai SDK，全部用 mock。
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clear_agent.core.exceptions import ClearAgentException
from clear_agent.core.llm_adapters import AnthropicAdapter, GeminiAdapter
from clear_agent.core.llm_response import LLMResponse, LLMToolResponse, ToolCall


# ==================== Section A: Anthropic 真异步 ====================


def _make_anthropic_adapter():
    return AnthropicAdapter(
        api_key="sk-x",
        base_url="https://api.anthropic.com",
        timeout=30,
        model="claude-3-5-sonnet-20241022",
    )


def test_anthropic_has_async_methods():
    a = _make_anthropic_adapter()
    assert hasattr(a, "ainvoke_async")
    assert hasattr(a, "ainvoke_with_tools_async")
    assert hasattr(a, "create_async_client")


def test_anthropic_create_async_client_without_sdk_raises(monkeypatch):
    """anthropic 包不可用时应抛友好错误（用 monkeypatch 隔离，不依赖真实卸载）"""
    import sys

    # 让任何 ``from anthropic import ...`` / ``import anthropic`` 重新触发 ImportError
    monkeypatch.setitem(sys.modules, "anthropic", None)

    a = _make_anthropic_adapter()
    with pytest.raises(ClearAgentException) as exc_info:
        a.create_async_client()
    assert "anthropic" in str(exc_info.value).lower()


def _make_anthropic_text_response(text: str, usage: dict = None):
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    u = usage or {"input": 10, "output": 5}
    resp.usage.input_tokens = u["input"]
    resp.usage.output_tokens = u["output"]
    return resp


def test_anthropic_ainvoke_async_text_response():
    a = _make_anthropic_adapter()
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        return_value=_make_anthropic_text_response("hello async")
    )
    a._async_client = fake_client

    out = asyncio.run(a.ainvoke_async([{"role": "user", "content": "hi"}]))
    assert isinstance(out, LLMResponse)
    assert out.content == "hello async"
    assert out.usage["total_tokens"] == 15
    assert out.model == "claude-3-5-sonnet-20241022"


def test_anthropic_ainvoke_async_extracts_system_message():
    a = _make_anthropic_adapter()
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        return_value=_make_anthropic_text_response("ok")
    )
    a._async_client = fake_client

    asyncio.run(
        a.ainvoke_async(
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ]
        )
    )
    # 检查 system 是否被提到顶层 system 参数
    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert call_kwargs.get("system") == "be brief"
    # converted_messages 不含 system
    assert all(m["role"] != "system" for m in call_kwargs["messages"])


def test_anthropic_ainvoke_async_propagates_failure():
    a = _make_anthropic_adapter()
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))
    a._async_client = fake_client

    with pytest.raises(ClearAgentException) as exc_info:
        asyncio.run(a.ainvoke_async([{"role": "user", "content": "hi"}]))
    assert "异步" in str(exc_info.value) or "Anthropic" in str(exc_info.value)


def test_anthropic_ainvoke_with_tools_async_extracts_tool_calls():
    a = _make_anthropic_adapter()

    fake_text = MagicMock()
    fake_text.type = "text"
    fake_text.text = "I will use the tool"
    fake_tu = MagicMock()
    fake_tu.type = "tool_use"
    fake_tu.id = "tool_001"
    fake_tu.name = "calculator"
    fake_tu.input = {"expression": "1+1"}
    fake_resp = MagicMock()
    fake_resp.content = [fake_text, fake_tu]
    fake_resp.usage.input_tokens = 20
    fake_resp.usage.output_tokens = 10

    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=fake_resp)
    a._async_client = fake_client

    out = asyncio.run(
        a.ainvoke_with_tools_async(
            [{"role": "user", "content": "calc"}],
            tools=[{"name": "calculator", "description": "calc"}],
        )
    )
    assert isinstance(out, LLMToolResponse)
    assert "I will use the tool" in (out.content or "")
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "calculator"
    assert out.tool_calls[0].id == "tool_001"
    args = json.loads(out.tool_calls[0].arguments)
    assert args == {"expression": "1+1"}


def test_anthropic_ainvoke_with_tools_async_strips_openai_tool_choice():
    """OpenAI 风格 tool_choice='auto' 应该不被传给 Anthropic SDK"""
    a = _make_anthropic_adapter()
    fake_resp = MagicMock()
    fake_resp.content = []
    fake_resp.usage.input_tokens = 0
    fake_resp.usage.output_tokens = 0
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=fake_resp)
    a._async_client = fake_client

    asyncio.run(
        a.ainvoke_with_tools_async(
            [{"role": "user", "content": "hi"}],
            tools=[],
            tool_choice="auto",  # 应被剥离
        )
    )
    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert "tool_choice" not in call_kwargs


def test_anthropic_ainvoke_with_tools_async_converts_openai_tool_schema():
    """OpenAI function schema 应转成 Anthropic tool schema。"""
    a = _make_anthropic_adapter()
    fake_resp = MagicMock()
    fake_resp.content = []
    fake_resp.usage.input_tokens = 0
    fake_resp.usage.output_tokens = 0
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=fake_resp)
    a._async_client = fake_client

    openai_tool = {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Run a calculation",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    }

    asyncio.run(
        a.ainvoke_with_tools_async(
            [{"role": "user", "content": "calc"}],
            tools=[openai_tool],
        )
    )

    call_kwargs = fake_client.messages.create.call_args.kwargs
    assert call_kwargs["tools"] == [
        {
            "name": "calculator",
            "description": "Run a calculation",
            "input_schema": openai_tool["function"]["parameters"],
        }
    ]


def test_anthropic_ainvoke_with_tools_async_converts_openai_tool_messages():
    """OpenAI assistant/tool messages 应转成 Anthropic tool_use/tool_result blocks。"""
    a = _make_anthropic_adapter()
    fake_resp = MagicMock()
    fake_resp.content = []
    fake_resp.usage.input_tokens = 0
    fake_resp.usage.output_tokens = 0
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=fake_resp)
    a._async_client = fake_client

    asyncio.run(
        a.ainvoke_with_tools_async(
            [
                {"role": "user", "content": "calc"},
                {
                    "role": "assistant",
                    "content": "using a tool",
                    "tool_calls": [
                        {
                            "id": "toolu_1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression": "1+1"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "toolu_1", "content": "2"},
            ],
            tools=[],
        )
    )

    messages = fake_client.messages.create.call_args.kwargs["messages"]
    assert messages[1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "using a tool"},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "calculator",
                "input": {"expression": "1+1"},
            },
        ],
    }
    assert messages[2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "2"}
        ],
    }


def test_anthropic_ainvoke_with_tools_async_propagates_failure():
    a = _make_anthropic_adapter()
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(side_effect=RuntimeError("api down"))
    a._async_client = fake_client

    with pytest.raises(ClearAgentException):
        asyncio.run(
            a.ainvoke_with_tools_async(
                [{"role": "user", "content": "hi"}], tools=[]
            )
        )


# ==================== Section B: Gemini 真异步 ====================


def _make_gemini_adapter():
    return GeminiAdapter(
        api_key="x",
        base_url="https://generativelanguage.googleapis.com",
        timeout=30,
        model="gemini-1.5-pro",
    )


def test_gemini_has_async_methods():
    g = _make_gemini_adapter()
    assert hasattr(g, "ainvoke_async")
    assert hasattr(g, "ainvoke_with_tools_async")


def test_gemini_ainvoke_async_delegates_to_invoke():
    """Gemini 走 to_thread(invoke)；invoke 被 mock"""
    g = _make_gemini_adapter()
    fake_resp = LLMResponse(content="gem", model="gemini-1.5-pro", usage={})
    g.invoke = MagicMock(return_value=fake_resp)

    out = asyncio.run(g.ainvoke_async([{"role": "user", "content": "hi"}]))
    assert out.content == "gem"
    g.invoke.assert_called_once()


def test_gemini_ainvoke_with_tools_async_strips_tool_choice():
    """Gemini ainvoke_with_tools_async 应剥离 tool_choice 再调 invoke_with_tools"""
    g = _make_gemini_adapter()
    fake_resp = LLMToolResponse(content="ok", tool_calls=[], model="gemini-1.5-pro", usage={})
    g.invoke_with_tools = MagicMock(return_value=fake_resp)

    asyncio.run(
        g.ainvoke_with_tools_async(
            [{"role": "user", "content": "hi"}],
            tools=[],
            tool_choice="auto",  # 应被剥离
        )
    )
    call_kwargs = g.invoke_with_tools.call_args.kwargs
    assert "tool_choice" not in call_kwargs


def test_gemini_ainvoke_with_tools_async_returns_response():
    g = _make_gemini_adapter()
    fake_resp = LLMToolResponse(
        content=None,
        tool_calls=[ToolCall(id="c1", name="t", arguments="{}")],
        model="gemini-1.5-pro",
        usage={},
    )
    g.invoke_with_tools = MagicMock(return_value=fake_resp)

    out = asyncio.run(
        g.ainvoke_with_tools_async([{"role": "user", "content": "x"}], tools=[])
    )
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "t"


# ==================== Section C: 集成 —— ClearAgentLLM 自动选真异步 ====================


def test_clear_agent_llm_picks_anthropic_adapter_async_path():
    """ClearAgentLLM(base_url=anthropic) 走真异步 path"""
    from clear_agent import ClearAgentLLM

    llm = ClearAgentLLM(
        model="claude-3-5-sonnet-20241022",
        api_key="x",
        base_url="https://api.anthropic.com",
    )
    assert isinstance(llm._adapter, AnthropicAdapter)
    # async 方法必须存在
    assert hasattr(llm._adapter, "ainvoke_async")
    assert hasattr(llm._adapter, "ainvoke_with_tools_async")

    fake_resp = LLMResponse(content="anth", model="claude", usage={})
    llm._adapter.ainvoke_async = AsyncMock(return_value=fake_resp)
    out = asyncio.run(llm.ainvoke([{"role": "user", "content": "hi"}]))
    assert out.content == "anth"


def test_clear_agent_llm_picks_gemini_adapter_async_path():
    from clear_agent import ClearAgentLLM

    llm = ClearAgentLLM(
        model="gemini-1.5-pro",
        api_key="x",
        base_url="https://generativelanguage.googleapis.com/v1",
    )
    assert isinstance(llm._adapter, GeminiAdapter)
    assert hasattr(llm._adapter, "ainvoke_async")

    fake_resp = LLMResponse(content="gem", model="gemini-1.5-pro", usage={})
    llm._adapter.ainvoke_async = AsyncMock(return_value=fake_resp)
    out = asyncio.run(llm.ainvoke([{"role": "user", "content": "hi"}]))
    assert out.content == "gem"
