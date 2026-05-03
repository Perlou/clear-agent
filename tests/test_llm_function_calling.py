"""测试 ClearAgentLLM 的 Function Calling 功能（OpenAI 兼容路径）

注：v2.0+ ``invoke_with_tools`` 返回 ``LLMToolResponse``（统一封装），
不再返回原生的 ``ChatCompletion``。本测试文件相应更新。
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from clear_agent.core.exceptions import ClearAgentException
from clear_agent.core.llm import ClearAgentLLM
from clear_agent.core.llm_response import LLMToolResponse


def _make_completion(
    content: str | None = None,
    tool_calls: list | None = None,
    model: str = "test-model",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
):
    """构造一个仿真的 OpenAI ChatCompletion 响应对象"""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


class TestLLMFunctionCalling:
    """测试 LLM 的 Function Calling 接口"""

    @pytest.fixture
    def mock_openai_client(self):
        """patch ``openai.OpenAI`` —— 适配 v2 lazy import 的 adapter 架构

        v2 中 ``OpenAI`` 类在 ``OpenAIAdapter.create_client()`` 内部 lazy import，
        模块级 ``clear_agent.core.llm.OpenAI`` 已不存在，所以直接 patch 上游。
        """
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def llm(self, mock_openai_client):
        """创建 ClearAgentLLM 实例"""
        with patch.dict(
            "os.environ",
            {
                "LLM_API_KEY": "test-key",
                "LLM_BASE_URL": "https://api.test.com/v1",
                "LLM_MODEL_ID": "test-model",
            },
        ):
            return ClearAgentLLM()

    def test_invoke_with_tools_basic(self, llm, mock_openai_client):
        """基本调用：mock 客户端、断言传参 + 返回 LLMToolResponse"""
        messages = [{"role": "user", "content": "计算 2+3"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "执行数学计算",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ]

        # 仿真返回：纯文本（无 tool_calls）
        mock_openai_client.chat.completions.create.return_value = _make_completion(
            content="5"
        )

        response = llm.invoke_with_tools(messages, tools, tool_choice="auto")

        # 调用参数验证
        mock_openai_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["messages"] == messages
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == "auto"

        # 返回值验证：LLMToolResponse
        assert isinstance(response, LLMToolResponse)
        assert response.content == "5"
        assert response.tool_calls == []
        assert response.usage["total_tokens"] == 15

    def test_invoke_with_tools_custom_params(self, llm, mock_openai_client):
        """自定义参数（temperature/max_tokens/tool_choice）正确透传"""
        messages = [{"role": "user", "content": "测试"}]
        tools = []
        mock_openai_client.chat.completions.create.return_value = _make_completion(
            content="ok"
        )

        llm.invoke_with_tools(
            messages,
            tools,
            tool_choice="required",
            temperature=0.5,
            max_tokens=1000,
        )

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 1000
        assert call_kwargs["tool_choice"] == "required"

    def test_invoke_with_tools_error_handling(self, llm, mock_openai_client):
        """底层抛异常应被包装为 ClearAgentException 并保留原因"""
        messages = [{"role": "user", "content": "测试"}]
        tools = []
        mock_openai_client.chat.completions.create.side_effect = Exception("API 错误")

        with pytest.raises(ClearAgentException) as exc_info:
            llm.invoke_with_tools(messages, tools)

        # adapter 用的中文错误消息：'OpenAI Function Calling调用失败: <原因>'
        msg = str(exc_info.value)
        assert "Function Calling" in msg and "调用失败" in msg
        assert "API 错误" in msg

    def test_invoke_with_tools_returns_tool_calls(self, llm, mock_openai_client):
        """模型返回 tool_calls 时应正确解析为 ``ToolCall`` 列表"""
        tool_call = SimpleNamespace(
            id="call_001",
            function=SimpleNamespace(
                name="calculate", arguments='{"expression": "2+3"}'
            ),
        )
        mock_openai_client.chat.completions.create.return_value = _make_completion(
            content=None, tool_calls=[tool_call]
        )

        response = llm.invoke_with_tools(
            [{"role": "user", "content": "算 2+3"}], tools=[]
        )

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_001"
        assert response.tool_calls[0].name == "calculate"
        assert response.tool_calls[0].arguments == '{"expression": "2+3"}'


class TestLLMFunctionCallingIntegration:
    """集成测试 - 需要真实 LLM"""

    @pytest.mark.skip(reason="需要真实 LLM 环境")
    def test_real_function_calling(self):
        llm = ClearAgentLLM()
        response = llm.invoke_with_tools(
            [{"role": "user", "content": "帮我计算 15 * 8"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "执行数学计算",
                        "parameters": {
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
                            "required": ["expression"],
                        },
                    },
                }
            ],
        )
        assert isinstance(response, LLMToolResponse)
        assert response.tool_calls
        assert response.tool_calls[0].name == "calculate"
