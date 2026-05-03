"""structured output 在 thinking 模型下的兼容性测试

历史 bug：
- ``deepseek-v4-flash`` thinking 模式下，强制 ``tool_choice = {"type":"function",...}`` 会 400
  'deepseek-reasoner does not support this tool_choice'
- ``deepseek-reasoner`` (R1) 完全不支持 function calling 和 response_format
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from clear_agent.core.llm_response import LLMToolResponse, ToolCall, LLMResponse
from clear_agent.core.structured import (
    StructuredLLM,
    _auto_method,
    _is_function_calling_unsupported,
    _is_strict_tool_choice_unsupported,
)


class _FakePlan(BaseModel):
    city: str = Field(description="城市")
    days: int = Field(description="天数")


# ============ A. 模型能力探测 ============


class TestStrictToolChoiceUnsupported:
    def test_deepseek_v4_flash_is_strict_unsupported(self):
        assert _is_strict_tool_choice_unsupported(
            "deepseek-v4-flash", "https://api.deepseek.com/v1"
        )

    def test_deepseek_v4_pro_is_strict_unsupported(self):
        assert _is_strict_tool_choice_unsupported(
            "deepseek-v4-pro", "https://api.deepseek.com/v1"
        )

    def test_deepseek_reasoner_is_strict_unsupported(self):
        assert _is_strict_tool_choice_unsupported(
            "deepseek-reasoner", "https://api.deepseek.com/v1"
        )

    def test_qwq_is_strict_unsupported(self):
        assert _is_strict_tool_choice_unsupported("qwq-32b-preview", "any")

    def test_deepseek_chat_is_supported(self):
        """非 thinking 模式应支持强制 tool_choice"""
        assert not _is_strict_tool_choice_unsupported(
            "deepseek-chat", "https://api.deepseek.com/v1"
        )

    def test_gpt4o_is_supported(self):
        assert not _is_strict_tool_choice_unsupported(
            "gpt-4o", "https://api.openai.com/v1"
        )


class TestFunctionCallingUnsupported:
    """完全不支持 function calling 的纯推理模型"""

    def test_deepseek_reasoner_unsupported(self):
        assert _is_function_calling_unsupported("deepseek-reasoner", "any")

    def test_deepseek_r1_distill_unsupported(self):
        assert _is_function_calling_unsupported(
            "deepseek-r1-distill-qwen-7b", "any"
        )

    def test_v4_flash_supports_function_calling(self):
        """V4-Flash thinking 仍支持 tool_choice='auto'，只是不支持强制 dict"""
        assert not _is_function_calling_unsupported(
            "deepseek-v4-flash", "https://api.deepseek.com/v1"
        )


# ============ B. _auto_method 选择 ============


class TestAutoMethod:
    def test_gpt4o_chooses_json_schema(self):
        assert _auto_method("gpt-4o", "https://api.openai.com/v1") == "json_schema"

    def test_deepseek_chat_chooses_function_calling(self):
        assert (
            _auto_method("deepseek-chat", "https://api.deepseek.com/v1")
            == "function_calling"
        )

    def test_deepseek_v4_flash_chooses_function_calling(self):
        """V4-Flash 用 function_calling，内部会自动降级 tool_choice"""
        assert (
            _auto_method("deepseek-v4-flash", "https://api.deepseek.com/v1")
            == "function_calling"
        )

    def test_deepseek_reasoner_chooses_prompt_json(self):
        """R1 完全不支持 tools → prompt_json 兜底"""
        assert (
            _auto_method("deepseek-reasoner", "https://api.deepseek.com/v1")
            == "prompt_json"
        )


# ============ C. function_calling 对 thinking 模型自动降级 ============


def _make_mock_llm(model: str, base_url: str = "https://api.deepseek.com/v1"):
    """构造一个 mock LLM，记录传给 invoke_with_tools 的参数"""
    llm = MagicMock()
    llm.model = model
    llm.base_url = base_url
    llm.invoke_with_tools.return_value = LLMToolResponse(
        content="",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="_FakePlan",
                arguments='{"city":"北京","days":3}',
            )
        ],
        model=model,
    )
    return llm


class TestFunctionCallingDegrade:
    def test_strict_tool_choice_for_normal_model(self):
        """普通模型应仍传强制 tool_choice dict"""
        llm = _make_mock_llm("deepseek-chat")
        sllm = StructuredLLM(llm, _FakePlan, method="function_calling")
        sllm.invoke([{"role": "user", "content": "北京 3 天"}])

        call_kwargs = llm.invoke_with_tools.call_args
        tc = call_kwargs.kwargs["tool_choice"]
        assert isinstance(tc, dict)
        assert tc["function"]["name"] == "_FakePlan"

    def test_thinking_model_degrades_to_auto(self):
        """V4-Flash thinking 模型应降级 tool_choice='auto' 并注入 prompt"""
        llm = _make_mock_llm("deepseek-v4-flash")
        sllm = StructuredLLM(llm, _FakePlan, method="function_calling")
        sllm.invoke([{"role": "user", "content": "北京 3 天"}])

        call_kwargs = llm.invoke_with_tools.call_args
        assert call_kwargs.kwargs["tool_choice"] == "auto"

        # 应该已经注入了"必须调用此工具"的 system prompt
        sent_messages = call_kwargs.kwargs["messages"]
        system_msg = next((m for m in sent_messages if m["role"] == "system"), None)
        assert system_msg is not None
        assert "_FakePlan" in system_msg["content"]
        assert "必须" in system_msg["content"] or "must" in system_msg["content"].lower()


# ============ D. prompt_json 不传 response_format ============


class TestPromptJsonMethod:
    def test_does_not_pass_response_format(self):
        """prompt_json 兜底路径：不能传 response_format（R1 / 类似不支持）"""
        llm = MagicMock()
        llm.model = "deepseek-reasoner"
        llm.base_url = "https://api.deepseek.com/v1"
        llm.invoke.return_value = LLMResponse(
            content='```json\n{"city":"上海","days":2}\n```',
            model="deepseek-reasoner",
        )

        sllm = StructuredLLM(llm, _FakePlan, method="prompt_json")
        plan = sllm.invoke([{"role": "user", "content": "上海 2 天"}])

        # 验证 invoke 被调用 + 没有 response_format
        call_kwargs = llm.invoke.call_args
        assert "response_format" not in call_kwargs.kwargs

        # 验证从 ```json fence 里抽出的 JSON 能正确 parse
        assert plan.city == "上海"
        assert plan.days == 2

    def test_extracts_json_from_messy_text(self):
        """模型可能混进解释性文字，兜底 extract 应工作"""
        llm = MagicMock()
        llm.model = "deepseek-reasoner"
        llm.base_url = "https://api.deepseek.com/v1"
        llm.invoke.return_value = LLMResponse(
            content='好的，这是结果：\n{"city":"杭州","days":5}\n希望对你有帮助。',
            model="deepseek-reasoner",
        )

        sllm = StructuredLLM(llm, _FakePlan, method="prompt_json")
        plan = sllm.invoke([{"role": "user", "content": "杭州 5 天"}])
        assert plan.city == "杭州"
        assert plan.days == 5


# ============ E. with_structured_output 端到端 auto 选择 ============


class TestEndToEndAutoSelection:
    """从 ClearAgentLLM.with_structured_output 进入，验证 auto 选对了 method"""

    def test_v4_flash_auto_picks_function_calling_with_degrade(self):
        with patch.dict(
            "os.environ",
            {
                "LLM_MODEL_ID": "deepseek-v4-flash",
                "LLM_API_KEY": "sk-test",
                "LLM_BASE_URL": "https://api.deepseek.com/v1",
            },
        ):
            from clear_agent.core.llm import ClearAgentLLM

            llm = ClearAgentLLM()
            sllm = llm.with_structured_output(_FakePlan)
            assert sllm.method == "function_calling"

    def test_reasoner_auto_picks_prompt_json(self):
        with patch.dict(
            "os.environ",
            {
                "LLM_MODEL_ID": "deepseek-reasoner",
                "LLM_API_KEY": "sk-test",
                "LLM_BASE_URL": "https://api.deepseek.com/v1",
            },
        ):
            from clear_agent.core.llm import ClearAgentLLM

            llm = ClearAgentLLM()
            sllm = llm.with_structured_output(_FakePlan)
            assert sllm.method == "prompt_json"
