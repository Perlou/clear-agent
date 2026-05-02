"""Simple Agent StateGraph builder 测试

验证：
- build_simple_graph 在「无工具」「有工具」两种形态下流程正确
- max_iterations 终止与 router 单元行为
- checkpointer 集成
- SimpleAgent.as_graph() 与 build_simple_graph 直接调用产物等价
- 顶层导入与旧 SimpleAgent API 向后兼容
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from clear_agent.agents import build_simple_graph
from clear_agent.agents._simple_graph import (
    SimpleGraphState,
    _router_after_llm,
    _router_after_tools,
)
from clear_agent.core.checkpoint import InMemoryCheckpointer
from clear_agent.core.graph import END, RunConfig
from clear_agent.core.llm_response import LLMResponse, LLMToolResponse, ToolCall


# ==================== Mock LLM ====================


class _MockLLM:
    """模拟 ClearAgentLLM；按预设脚本返回 LLMResponse / LLMToolResponse 序列

    SimpleGraph 在「有工具」时调 invoke_with_tools，在「无工具」时调 invoke。
    """

    def __init__(
        self,
        invoke_responses: Optional[List[LLMResponse]] = None,
        tool_responses: Optional[List[LLMToolResponse]] = None,
    ):
        self._invoke = list(invoke_responses or [])
        self._tools = list(tool_responses or [])
        self.invoke_calls: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.model = "mock-model"

    def invoke(self, messages, **kwargs):
        self.invoke_calls.append({"messages": list(messages)})
        if not self._invoke:
            raise AssertionError("LLM.invoke 调用次数超过预设")
        return self._invoke.pop(0)

    def invoke_with_tools(self, messages, tools, tool_choice="auto", **kwargs):
        self.tool_calls.append({"messages": list(messages), "tools": list(tools)})
        if not self._tools:
            raise AssertionError("LLM.invoke_with_tools 调用次数超过预设")
        return self._tools.pop(0)


def _resp(content: str, total_tokens: int = 5) -> LLMResponse:
    return LLMResponse(
        content=content, model="mock-model", usage={"total_tokens": total_tokens}
    )


def _tool_resp(
    content: Optional[str] = None,
    tool_calls: Optional[List[ToolCall]] = None,
    total_tokens: int = 7,
) -> LLMToolResponse:
    return LLMToolResponse(
        content=content,
        tool_calls=tool_calls or [],
        model="mock-model",
        usage={"total_tokens": total_tokens},
    )


def _make_call(name: str, args: dict, call_id: str = "tc-1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(args))


# ==================== Test 1: 无工具的极简两节点 ====================


def test_simple_graph_no_tools_direct_reply():
    """无 tool_registry → START → llm → END，invoke 一次返回内容"""
    llm = _MockLLM(invoke_responses=[_resp("Hello!", total_tokens=8)])
    compiled = build_simple_graph(llm)

    result = compiled.invoke(
        {"messages": [{"role": "user", "content": "hi"}], "max_iterations": 3}
    )

    assert result["final_answer"] == "Hello!"
    assert result["total_tokens"] == 8
    assert result["iterations"] == 1
    # 只调用了 invoke（无 tool path）
    assert len(llm.invoke_calls) == 1
    assert llm.tool_calls == []


# ==================== Test 2: 有工具但 LLM 直接给文本 ====================


def test_simple_graph_with_tools_but_llm_replies_directly():
    """有工具注册但 LLM 没调用 → final_answer 设为 content，立即结束"""
    from clear_agent import CalculatorTool, ToolRegistry

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    llm = _MockLLM(tool_responses=[_tool_resp(content="No tool needed")])
    compiled = build_simple_graph(llm, tool_registry=registry)

    result = compiled.invoke(
        {"messages": [{"role": "user", "content": "?"}], "max_iterations": 3}
    )

    assert result["final_answer"] == "No tool needed"
    assert result["iterations"] == 1
    # 注意：tool_schemas 非空，所以走 invoke_with_tools 分支
    assert len(llm.tool_calls) == 1
    assert llm.invoke_calls == []


# ==================== Test 3: 工具调用 + 二轮回复 ====================


def test_simple_graph_tool_call_then_final_answer():
    """LLM 第 1 轮调工具 → 工具执行 → LLM 第 2 轮给文本 → 终止"""
    from clear_agent import CalculatorTool, ToolRegistry

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    calc_name = "python_calculator"

    llm = _MockLLM(
        tool_responses=[
            _tool_resp(tool_calls=[_make_call(calc_name, {"expression": "2+3"})]),
            _tool_resp(content="answer is 5"),
        ]
    )
    compiled = build_simple_graph(llm, tool_registry=registry)

    result = compiled.invoke(
        {"messages": [{"role": "user", "content": "calc"}], "max_iterations": 3}
    )

    assert result["final_answer"] == "answer is 5"
    # 两次 LLM 调用 + 一轮工具
    assert len(llm.tool_calls) == 2
    tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
    assert any("5" in m["content"] for m in tool_msgs)
    assert result["iterations"] == 2


# ==================== Test 4: max_iterations 终止 ====================


def test_simple_graph_max_iterations_terminates():
    """LLM 一直调工具不停 → 达到 max_iterations 后路由到 end"""
    from clear_agent import CalculatorTool, ToolRegistry

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    calc_name = "python_calculator"

    # 准备 5 个全是工具调用的响应
    responses = [
        _tool_resp(
            tool_calls=[_make_call(calc_name, {"expression": "1+1"}, call_id=f"tc-{i}")]
        )
        for i in range(5)
    ]
    llm = _MockLLM(tool_responses=responses)
    compiled = build_simple_graph(llm, tool_registry=registry)

    result = compiled.invoke(
        {"messages": [{"role": "user", "content": "loop"}], "max_iterations": 2}
    )

    # iterations 不会无限增长
    assert result["iterations"] <= 2
    assert result.get("final_answer") is None


# ==================== Test 5: 工具参数 JSON 解析失败友好降级 ====================


def test_simple_graph_bad_tool_arguments_continues():
    """LLM 给出非法 JSON arguments → tool 节点写入解析错误消息，但不崩"""
    from clear_agent import CalculatorTool, ToolRegistry

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    calc_name = "python_calculator"

    bad_call = ToolCall(id="tc-bad", name=calc_name, arguments="{invalid json")
    llm = _MockLLM(
        tool_responses=[
            _tool_resp(tool_calls=[bad_call]),
            _tool_resp(content="recovered"),
        ]
    )
    compiled = build_simple_graph(llm, tool_registry=registry)

    result = compiled.invoke(
        {"messages": [{"role": "user", "content": "x"}], "max_iterations": 3}
    )

    assert result["final_answer"] == "recovered"
    tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
    assert any("参数解析失败" in m["content"] for m in tool_msgs)


# ==================== Test 6: 未注册工具友好降级 ====================


def test_simple_graph_unknown_tool_name_continues():
    """LLM 调一个未注册的工具名 → tool 节点写入未注册消息，但不崩"""
    from clear_agent import CalculatorTool, ToolRegistry

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    llm = _MockLLM(
        tool_responses=[
            _tool_resp(tool_calls=[_make_call("nonexistent_tool", {"x": 1})]),
            _tool_resp(content="ok"),
        ]
    )
    compiled = build_simple_graph(llm, tool_registry=registry)

    result = compiled.invoke(
        {"messages": [{"role": "user", "content": "x"}], "max_iterations": 3}
    )

    assert result["final_answer"] == "ok"
    tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
    assert any("未注册" in m["content"] for m in tool_msgs)


# ==================== Test 7: checkpointer 集成 ====================


def test_simple_graph_checkpointer_writes_per_node():
    llm = _MockLLM(invoke_responses=[_resp("hello")])
    ck = InMemoryCheckpointer()
    compiled = build_simple_graph(llm, checkpointer=ck)

    compiled.invoke(
        {"messages": [{"role": "user", "content": "go"}], "max_iterations": 3},
        config=RunConfig(thread_id="thread-simple"),
    )

    ckpts = ck.list("thread-simple")
    assert len(ckpts) >= 1
    assert ckpts[0].next_nodes == [END]


# ==================== Test 8: SimpleAgent.as_graph() 等价 ====================


def test_simple_agent_as_graph_equivalent():
    """SimpleAgent.as_graph() 返回的 CompiledGraph 与 build_simple_graph 直接调用一致"""
    from clear_agent.agents import SimpleAgent

    llm = _MockLLM(invoke_responses=[_resp("equiv")])
    agent = SimpleAgent(name="t", llm=llm)
    compiled = agent.as_graph()

    result = compiled.invoke(
        {"messages": [{"role": "user", "content": "?"}], "max_iterations": 3}
    )

    assert result["final_answer"] == "equiv"


def test_simple_agent_as_graph_with_tools_uses_tool_path():
    """SimpleAgent 带 tool_registry 时，as_graph 的图也带 tools 节点"""
    from clear_agent import CalculatorTool, ToolRegistry
    from clear_agent.agents import SimpleAgent

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    llm = _MockLLM(tool_responses=[_tool_resp(content="direct")])
    agent = SimpleAgent(name="t", llm=llm, tool_registry=registry)
    compiled = agent.as_graph()

    result = compiled.invoke(
        {"messages": [{"role": "user", "content": "?"}], "max_iterations": 3}
    )

    assert result["final_answer"] == "direct"
    # 走的是 invoke_with_tools 分支
    assert len(llm.tool_calls) == 1


# ==================== Test 9: router 单元测试 ====================


def test_router_after_llm_end_on_final_answer():
    state: SimpleGraphState = {"final_answer": "x", "iterations": 1}
    assert _router_after_llm(state) == "end"


def test_router_after_llm_end_on_max_iterations():
    state: SimpleGraphState = {"iterations": 3, "max_iterations": 3}
    assert _router_after_llm(state) == "end"


def test_router_after_llm_tools_when_pending():
    fake_call = MagicMock()
    state: SimpleGraphState = {
        "iterations": 1,
        "max_iterations": 3,
        "tool_calls_pending": [fake_call],
    }
    assert _router_after_llm(state) == "tools"


def test_router_after_llm_end_when_no_pending_no_answer():
    state: SimpleGraphState = {"iterations": 1, "max_iterations": 3}
    assert _router_after_llm(state) == "end"


def test_router_after_tools_back_to_llm():
    state: SimpleGraphState = {"iterations": 1, "max_iterations": 3}
    assert _router_after_tools(state) == "llm"


def test_router_after_tools_end_on_max_iterations():
    state: SimpleGraphState = {"iterations": 3, "max_iterations": 3}
    assert _router_after_tools(state) == "end"


# ==================== Test 10: 旧 SimpleAgent API 向后兼容 ====================


def test_legacy_simple_agent_construction_intact():
    from clear_agent.agents import SimpleAgent

    llm = _MockLLM()
    agent = SimpleAgent(name="legacy", llm=llm)

    assert hasattr(agent, "run")
    assert hasattr(agent, "arun")
    assert hasattr(agent, "as_graph")


def test_top_level_simple_graph_imports_intact():
    from clear_agent.agents import (
        build_simple_graph,
        SimpleGraphState,
    )

    assert callable(build_simple_graph)
    # SimpleGraphState 是 TypedDict（在运行时本质是 dict 子类）
    assert SimpleGraphState is not None
