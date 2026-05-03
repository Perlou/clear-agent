"""thinking model reasoning_content 在多轮对话中的捕获 + 回写策略测试

回归点：
- DeepSeek V4 thinking 模式要求 reasoning_content 在下一轮回写
- DeepSeek-R1（``deepseek-reasoner``）禁止回写
- 其他 provider 无所谓（写了被忽略）

历史 bug：
- ``_is_thinking_model`` 名单写死，``deepseek-v4-flash`` 不在内 → 不捕获
- 多轮 agent loop 手工拼 assistant message，从来不带 reasoning_content
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from clear_agent.core.llm import ClearAgentLLM
from clear_agent.core.llm_adapters import OpenAIAdapter
from clear_agent.core.llm_response import LLMToolResponse, ToolCall


# ============ Fixture：构造 OpenAIAdapter ============


def _make_adapter(model: str) -> OpenAIAdapter:
    return OpenAIAdapter(
        api_key="sk-test",
        base_url="https://api.test.com/v1",
        timeout=30,
        model=model,
    )


# ============ A. 通用捕获 ============


class TestCaptureReasoningContent:
    """通用捕获：不再依赖 thinking_keywords 名单"""

    def test_captures_for_v4_flash_even_though_keyword_not_listed(self):
        """deepseek-v4-flash 不在历史 thinking_keywords 名单里，但仍应被捕获"""
        adapter = _make_adapter("deepseek-v4-flash")
        choice = SimpleNamespace(
            message=SimpleNamespace(reasoning_content="思考内容 abc"),
        )
        rc = adapter._capture_reasoning_content(choice)
        assert rc == "思考内容 abc"

    def test_captures_when_field_on_choice(self):
        """部分 provider 把 reasoning_content 放在 choice 而非 message 上"""
        adapter = _make_adapter("any-model")
        choice = SimpleNamespace(
            message=SimpleNamespace(),
            reasoning_content="fallback rc",
        )
        rc = adapter._capture_reasoning_content(choice)
        assert rc == "fallback rc"

    def test_returns_none_when_field_absent(self):
        adapter = _make_adapter("gpt-4o-mini")
        choice = SimpleNamespace(message=SimpleNamespace())
        rc = adapter._capture_reasoning_content(choice)
        assert rc is None

    def test_returns_none_for_empty_string(self):
        adapter = _make_adapter("any")
        choice = SimpleNamespace(message=SimpleNamespace(reasoning_content=""))
        rc = adapter._capture_reasoning_content(choice)
        assert rc is None


# ============ B. 回写策略 ============


class TestEchoStrategy:
    """每个 adapter 决定是否在多轮中回写 reasoning_content"""

    def test_default_echoes_for_deepseek_v4_flash(self):
        adapter = _make_adapter("deepseek-v4-flash")
        assert adapter._should_echo_reasoning() is True

    def test_default_echoes_for_deepseek_v4_pro(self):
        adapter = _make_adapter("deepseek-v4-pro")
        assert adapter._should_echo_reasoning() is True

    def test_default_echoes_for_gpt(self):
        adapter = _make_adapter("gpt-4o")
        assert adapter._should_echo_reasoning() is True

    def test_does_not_echo_for_deepseek_reasoner(self):
        """R1 禁止回写"""
        adapter = _make_adapter("deepseek-reasoner")
        assert adapter._should_echo_reasoning() is False

    def test_does_not_echo_for_deepseek_r1(self):
        """deepseek-r1 别名也算"""
        adapter = _make_adapter("deepseek-r1-distill-qwen-7b")
        assert adapter._should_echo_reasoning() is False


# ============ C. serialize_assistant_message ============


class TestSerializeAssistantMessage:
    def test_basic_message_without_tool_calls_or_reasoning(self):
        adapter = _make_adapter("gpt-4o")
        resp = LLMToolResponse(
            content="hello",
            tool_calls=[],
            model="gpt-4o",
        )
        msg = adapter.serialize_assistant_message(resp)
        assert msg == {"role": "assistant", "content": "hello"}

    def test_includes_tool_calls(self):
        adapter = _make_adapter("gpt-4o")
        resp = LLMToolResponse(
            content=None,
            tool_calls=[ToolCall(id="call_1", name="add", arguments='{"a":1,"b":2}')],
            model="gpt-4o",
        )
        msg = adapter.serialize_assistant_message(resp)
        assert msg["tool_calls"] == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a":1,"b":2}'},
            }
        ]

    def test_echoes_reasoning_for_v4_flash(self):
        """deepseek-v4-flash thinking 模式：必须把 reasoning_content 回写"""
        adapter = _make_adapter("deepseek-v4-flash")
        resp = LLMToolResponse(
            content="text",
            tool_calls=[],
            model="deepseek-v4-flash",
            reasoning_content="思考过程",
        )
        msg = adapter.serialize_assistant_message(resp)
        assert msg.get("reasoning_content") == "思考过程"

    def test_does_not_echo_reasoning_for_deepseek_reasoner(self):
        """R1：禁止回写"""
        adapter = _make_adapter("deepseek-reasoner")
        resp = LLMToolResponse(
            content="answer",
            tool_calls=[],
            model="deepseek-reasoner",
            reasoning_content="不能传回去",
        )
        msg = adapter.serialize_assistant_message(resp)
        assert "reasoning_content" not in msg

    def test_no_reasoning_field_when_response_has_none(self):
        adapter = _make_adapter("deepseek-v4-flash")
        resp = LLMToolResponse(
            content="text",
            tool_calls=[],
            model="deepseek-v4-flash",
            reasoning_content=None,
        )
        msg = adapter.serialize_assistant_message(resp)
        assert "reasoning_content" not in msg


# ============ D. ClearAgentLLM 代理 ============


class TestClearAgentLLMSerializerProxy:
    """ClearAgentLLM.serialize_assistant_message 应代理给 adapter"""

    def test_delegates_to_adapter(self):
        with patch.dict(
            "os.environ",
            {
                "LLM_MODEL_ID": "deepseek-v4-flash",
                "LLM_API_KEY": "sk-test",
                "LLM_BASE_URL": "https://api.test.com/v1",
            },
        ):
            llm = ClearAgentLLM()
        resp = LLMToolResponse(
            content="x",
            tool_calls=[],
            model="deepseek-v4-flash",
            reasoning_content="rc",
        )
        msg = llm.serialize_assistant_message(resp)
        assert msg["role"] == "assistant"
        assert msg["content"] == "x"
        assert msg["reasoning_content"] == "rc"


# ============ E. 端到端：多轮 ReAct 第二轮要回写 ============


class TestReActGraphEchoesReasoning:
    """模拟 ReAct loop：第二轮 messages 必须包含上一轮的 reasoning_content"""

    def test_react_node_serializes_with_reasoning(self):
        """_make_llm_node 的输出 assistant message 应包含 reasoning_content"""
        from clear_agent.agents._react_graph import _make_llm_node

        # 构造一个假 LLM
        mock_llm = MagicMock()
        mock_llm.invoke_with_tools.return_value = LLMToolResponse(
            content="思考完毕",
            tool_calls=[ToolCall(id="c1", name="Thought", arguments='{"reasoning":"rrr"}')],
            model="deepseek-v4-flash",
            reasoning_content="multi-step thinking",
            usage={"total_tokens": 50},
        )
        # 让 serialize_assistant_message 走真实 OpenAIAdapter 逻辑
        real_adapter = _make_adapter("deepseek-v4-flash")
        mock_llm.serialize_assistant_message = real_adapter.serialize_assistant_message

        node_fn = _make_llm_node(mock_llm, tool_schemas=[])
        update = node_fn({"messages": [{"role": "user", "content": "hi"}]})

        new_msgs = update["messages"]
        assert len(new_msgs) == 1
        assistant = new_msgs[0]
        assert assistant["role"] == "assistant"
        assert assistant["content"] == "思考完毕"
        assert assistant["reasoning_content"] == "multi-step thinking"
        assert assistant["tool_calls"][0]["function"]["name"] == "Thought"
