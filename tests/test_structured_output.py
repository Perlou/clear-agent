"""结构化输出（with_structured_output）测试

不调用真实 LLM——通过 mock LLM 验证：
- function_calling / json_mode / json_schema 三种 method 的请求构造正确
- 简单 / 嵌套 / Optional / Enum schema 都能解析
- 失败重试机制工作
- include_raw 模式返回 dict
- 同步与异步等价
- _auto_method 选择规则
"""

from __future__ import annotations

import asyncio
import json
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, Field

from clear_agent import StructuredOutputError
from clear_agent.core.llm_response import LLMResponse, LLMToolResponse, ToolCall
from clear_agent.core.structured import (
    StructuredLLM,
    _auto_method,
    _schema_to_function,
    _strip_json_fence,
)


# ==================== Mock LLM ====================


class _MockLLM:
    """模拟 ClearAgentLLM 的最小子集

    根据预设响应序列依次返回。每次调用 invoke / invoke_with_tools / a* 时
    弹出队头响应。允许混合三种 method 并行用同一序列。
    """

    def __init__(
        self,
        invoke_responses: Optional[List[LLMResponse]] = None,
        tool_responses: Optional[List[LLMToolResponse]] = None,
        model: str = "mock-model",
        base_url: str = "https://mock/v1",
    ):
        self._invoke = list(invoke_responses or [])
        self._tools = list(tool_responses or [])
        self.invoke_calls: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.model = model
        self.base_url = base_url

    # --- 同步 ---

    def invoke(self, messages, **kwargs):
        self.invoke_calls.append({"messages": list(messages), "kwargs": dict(kwargs)})
        if not self._invoke:
            raise AssertionError("LLM.invoke 调用次数超过预设")
        return self._invoke.pop(0)

    def invoke_with_tools(self, messages, tools, tool_choice="auto", **kwargs):
        self.tool_calls.append(
            {
                "messages": list(messages),
                "tools": list(tools),
                "tool_choice": tool_choice,
                "kwargs": dict(kwargs),
            }
        )
        if not self._tools:
            raise AssertionError("LLM.invoke_with_tools 调用次数超过预设")
        return self._tools.pop(0)

    # --- 异步 ---

    async def ainvoke(self, messages, **kwargs):
        return self.invoke(messages, **kwargs)

    async def ainvoke_with_tools(self, messages, tools, tool_choice="auto", **kwargs):
        return self.invoke_with_tools(messages, tools, tool_choice=tool_choice, **kwargs)


def _resp(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="mock-model", usage={"total_tokens": 10})


def _tool_resp(payload: Dict[str, Any], call_id: str = "tc-1") -> LLMToolResponse:
    return LLMToolResponse(
        content=None,
        tool_calls=[
            ToolCall(id=call_id, name="X", arguments=json.dumps(payload))
        ],
        model="mock-model",
        usage={"total_tokens": 12},
    )


# ==================== Section A: 工具函数单元测试 ====================


def test_auto_method_openai_gpt4o_picks_json_schema():
    assert _auto_method("gpt-4o", "https://api.openai.com/v1") == "json_schema"


def test_auto_method_openai_gpt4_1_picks_json_schema():
    assert _auto_method("gpt-4.1-2025-01-01", "https://api.openai.com/v1") == "json_schema"


def test_auto_method_deepseek_picks_function_calling():
    assert _auto_method("deepseek-chat", "https://api.deepseek.com/v1") == "function_calling"


def test_auto_method_anthropic_picks_function_calling():
    assert _auto_method("claude-3-sonnet", "https://api.anthropic.com/v1") == "function_calling"


def test_auto_method_qwen_picks_function_calling():
    assert (
        _auto_method("qwen-max", "https://dashscope.aliyuncs.com/v1") == "function_calling"
    )


def test_strip_json_fence_handles_with_lang():
    assert _strip_json_fence('```json\n{"a":1}\n```') == '{"a":1}'


def test_strip_json_fence_handles_no_lang():
    assert _strip_json_fence('```\n{"a":1}\n```') == '{"a":1}'


def test_strip_json_fence_passthrough():
    assert _strip_json_fence('{"a":1}') == '{"a":1}'


def test_schema_to_function_shape():
    class P(BaseModel):
        """A person."""

        name: str
        age: int

    fn = _schema_to_function(P)
    assert fn["type"] == "function"
    assert fn["function"]["name"] == "P"
    assert "A person." in fn["function"]["description"]
    assert "properties" in fn["function"]["parameters"]


# ==================== Section B: function_calling 路径 ====================


class Person(BaseModel):
    name: str
    age: int
    occupation: Optional[str] = None


def test_function_calling_simple_schema():
    """function_calling: schema 严格匹配 → 返回 Person 实例"""
    llm = _MockLLM(
        tool_responses=[_tool_resp({"name": "Alice", "age": 30, "occupation": "teacher"})]
    )
    s = StructuredLLM(llm, Person, method="function_calling", max_retries=0)

    out = s.invoke([{"role": "user", "content": "Alice 30 teacher"}])

    assert isinstance(out, Person)
    assert out.name == "Alice"
    assert out.age == 30
    assert out.occupation == "teacher"
    # 强制了 tool_choice 到 schema 名
    assert llm.tool_calls[0]["tool_choice"] == {
        "type": "function",
        "function": {"name": "Person"},
    }


def test_function_calling_optional_field_default():
    """缺 optional 字段 → 走 default"""
    llm = _MockLLM(tool_responses=[_tool_resp({"name": "Bob", "age": 25})])
    s = StructuredLLM(llm, Person, method="function_calling", max_retries=0)

    out = s.invoke([{"role": "user", "content": "x"}])
    assert out.occupation is None


# ==================== Section C: 嵌套 schema ====================


class Item(BaseModel):
    sku: str
    qty: int


class Order(BaseModel):
    order_id: str
    items: List[Item]


def test_function_calling_nested_schema():
    payload = {
        "order_id": "O-1",
        "items": [{"sku": "A", "qty": 2}, {"sku": "B", "qty": 5}],
    }
    llm = _MockLLM(tool_responses=[_tool_resp(payload)])
    s = StructuredLLM(llm, Order, method="function_calling", max_retries=0)

    out = s.invoke([{"role": "user", "content": "place order"}])
    assert out.order_id == "O-1"
    assert len(out.items) == 2
    assert out.items[0].sku == "A"
    assert out.items[1].qty == 5


# ==================== Section D: Enum 字段 ====================


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Task(BaseModel):
    title: str
    priority: Priority


def test_function_calling_enum_field():
    llm = _MockLLM(tool_responses=[_tool_resp({"title": "t", "priority": "high"})])
    s = StructuredLLM(llm, Task, method="function_calling", max_retries=0)

    out = s.invoke([{"role": "user", "content": "make task"}])
    assert out.priority == Priority.HIGH


def test_function_calling_invalid_enum_raises_after_retries():
    """Enum 限制：传 'urgent' 不在范围 → 重试耗尽 → 抛 StructuredOutputError"""
    llm = _MockLLM(
        tool_responses=[
            _tool_resp({"title": "t", "priority": "urgent"}),
            _tool_resp({"title": "t", "priority": "also-bad"}),
        ]
    )
    s = StructuredLLM(llm, Task, method="function_calling", max_retries=1)

    with pytest.raises(StructuredOutputError) as exc_info:
        s.invoke([{"role": "user", "content": "x"}])
    # last_error 应是 ValidationError
    assert exc_info.value.last_error is not None


# ==================== Section E: 失败重试 ====================


def test_retry_recovers_on_second_attempt():
    """第 1 次返回不合法 → 第 2 次返回合法 → 成功"""
    llm = _MockLLM(
        tool_responses=[
            _tool_resp({"name": "X", "age": "thirty"}),  # age 字符串 → 校验失败
            _tool_resp({"name": "X", "age": 30}),
        ]
    )
    s = StructuredLLM(llm, Person, method="function_calling", max_retries=2)

    out = s.invoke([{"role": "user", "content": "?"}])
    assert isinstance(out, Person)
    assert out.age == 30
    # 重试时把错误信息追加到 messages 里
    second_call_msgs = llm.tool_calls[1]["messages"]
    assert any(
        "failed validation" in (m.get("content") or "").lower()
        for m in second_call_msgs
    )


def test_retry_exhausts_then_raises():
    """3 次都失败 → 抛 StructuredOutputError"""
    llm = _MockLLM(
        tool_responses=[
            _tool_resp({"name": "X", "age": "bad"}),
            _tool_resp({"name": "X", "age": "still bad"}),
            _tool_resp({"name": "X", "age": None}),
        ]
    )
    s = StructuredLLM(llm, Person, method="function_calling", max_retries=2)

    with pytest.raises(StructuredOutputError):
        s.invoke([{"role": "user", "content": "?"}])


# ==================== Section F: include_raw 模式 ====================


def test_include_raw_success():
    raw_resp = _tool_resp({"name": "A", "age": 1})
    llm = _MockLLM(tool_responses=[raw_resp])
    s = StructuredLLM(
        llm, Person, method="function_calling", include_raw=True, max_retries=0
    )

    out = s.invoke([{"role": "user", "content": "?"}])
    assert isinstance(out, dict)
    assert isinstance(out["parsed"], Person)
    assert out["raw"] is raw_resp
    assert out["parsing_error"] is None


def test_include_raw_failure_returns_dict_no_raise():
    """include_raw=True 下解析失败不抛，而是 parsed=None + parsing_error 填充"""
    llm = _MockLLM(tool_responses=[_tool_resp({"name": "A", "age": "bad"})])
    s = StructuredLLM(
        llm, Person, method="function_calling", include_raw=True, max_retries=0
    )

    out = s.invoke([{"role": "user", "content": "?"}])
    assert out["parsed"] is None
    assert out["parsing_error"] is not None
    assert out["raw"] is not None


# ==================== Section G: json_mode 路径 ====================


def test_json_mode_injects_schema_into_system_prompt():
    """json_mode: system 消息里带 JSON Schema 提示，response_format=json_object"""
    llm = _MockLLM(invoke_responses=[_resp('{"name": "A", "age": 1}')])
    s = StructuredLLM(llm, Person, method="json_mode", max_retries=0)

    out = s.invoke([{"role": "user", "content": "extract"}])
    assert isinstance(out, Person)
    # response_format 透传
    call = llm.invoke_calls[0]
    assert call["kwargs"]["response_format"] == {"type": "json_object"}
    # system prompt 注入了 schema
    sys_msg = call["messages"][0]
    assert sys_msg["role"] == "system"
    assert "JSON Schema" in sys_msg["content"]


def test_json_mode_merges_with_existing_system_prompt():
    """已有 system 消息 → 在其末尾追加 schema 提示，不新增"""
    llm = _MockLLM(invoke_responses=[_resp('{"name": "A", "age": 1}')])
    s = StructuredLLM(llm, Person, method="json_mode", max_retries=0)

    s.invoke(
        [
            {"role": "system", "content": "ORIGINAL_SYS"},
            {"role": "user", "content": "go"},
        ]
    )
    msgs = llm.invoke_calls[0]["messages"]
    # 仍然只有 1 个 system 消息
    assert sum(1 for m in msgs if m["role"] == "system") == 1
    # 原内容保留
    assert "ORIGINAL_SYS" in msgs[0]["content"]
    # 追加了 schema 信息
    assert "JSON Schema" in msgs[0]["content"]


def test_json_mode_strips_code_fence():
    """LLM 嵌 ```json...``` fence 也能解析"""
    llm = _MockLLM(invoke_responses=[_resp('```json\n{"name":"X","age":2}\n```')])
    s = StructuredLLM(llm, Person, method="json_mode", max_retries=0)
    out = s.invoke([{"role": "user", "content": "?"}])
    assert out.age == 2


# ==================== Section H: json_schema 路径 ====================


def test_json_schema_passes_strict_response_format():
    llm = _MockLLM(invoke_responses=[_resp('{"name":"X","age":3}')])
    s = StructuredLLM(llm, Person, method="json_schema", max_retries=0)

    out = s.invoke([{"role": "user", "content": "?"}])
    assert out.age == 3
    rf = llm.invoke_calls[0]["kwargs"]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "Person"


# ==================== Section I: 异步对偶 ====================


def test_ainvoke_function_calling_equivalent():
    llm = _MockLLM(tool_responses=[_tool_resp({"name": "A", "age": 4})])
    s = StructuredLLM(llm, Person, method="function_calling", max_retries=0)

    out = asyncio.run(s.ainvoke([{"role": "user", "content": "?"}]))
    assert isinstance(out, Person)
    assert out.age == 4


def test_ainvoke_json_mode_equivalent():
    llm = _MockLLM(invoke_responses=[_resp('{"name":"B","age":5}')])
    s = StructuredLLM(llm, Person, method="json_mode", max_retries=0)

    out = asyncio.run(s.ainvoke([{"role": "user", "content": "?"}]))
    assert out.age == 5


def test_ainvoke_retry_recovery():
    llm = _MockLLM(
        tool_responses=[
            _tool_resp({"name": "X", "age": "bad"}),
            _tool_resp({"name": "X", "age": 6}),
        ]
    )
    s = StructuredLLM(llm, Person, method="function_calling", max_retries=2)

    out = asyncio.run(s.ainvoke([{"role": "user", "content": "?"}]))
    assert out.age == 6


# ==================== Section J: ClearAgentLLM.with_structured_output 集成 ====================


def test_clear_agent_llm_with_structured_output_returns_StructuredLLM():
    from clear_agent import ClearAgentLLM

    llm = ClearAgentLLM(
        model="deepseek-chat",
        api_key="x",
        base_url="https://api.deepseek.com/v1",
    )
    s = llm.with_structured_output(Person)
    assert isinstance(s, StructuredLLM)
    assert s.schema is Person
    # auto 选 function_calling
    assert s.method == "function_calling"


def test_clear_agent_llm_with_structured_output_explicit_method():
    from clear_agent import ClearAgentLLM

    llm = ClearAgentLLM(
        model="any", api_key="x", base_url="https://example/v1"
    )
    s = llm.with_structured_output(Person, method="json_mode")
    assert s.method == "json_mode"


def test_clear_agent_llm_with_structured_output_invalid_method_raises():
    from clear_agent import ClearAgentLLM, ClearAgentException

    llm = ClearAgentLLM(model="any", api_key="x", base_url="https://example/v1")
    with pytest.raises(ClearAgentException):
        llm.with_structured_output(Person, method="bogus_method")


# ==================== Section K: 顶层导出 ====================


def test_top_level_structured_imports():
    import clear_agent

    assert hasattr(clear_agent, "StructuredLLM")
    assert hasattr(clear_agent, "StructuredOutputError")
