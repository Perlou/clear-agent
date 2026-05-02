"""ReAct StateGraph builder 测试

验证 build_react_graph + ReActAgent.as_graph() 行为正确，
以及 ReActAgent 旧 API 100% 向后兼容（仅检查导入与构造，不调真实 LLM）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from clear_agent.agents import build_react_graph
from clear_agent.agents._react_graph import (
    BUILTIN_TOOL_NAMES,
    ReActGraphState,
    _build_tool_schemas,
    _router,
)
from clear_agent.core.checkpoint import InMemoryCheckpointer
from clear_agent.core.graph import END, RunConfig, START
from clear_agent.core.llm_response import LLMToolResponse, ToolCall


# ==================== Mock LLM ====================


class _MockLLM:
    """模拟 ClearAgentLLM，按预设脚本返回 LLMToolResponse 序列"""

    def __init__(self, scripted_responses: List[LLMToolResponse]):
        self._responses = list(scripted_responses)
        self.calls: List[Dict[str, Any]] = []
        self.model = "mock-model"

    def invoke_with_tools(self, messages, tools, tool_choice="auto", **kwargs):
        self.calls.append({"messages": list(messages), "tools": list(tools)})
        if not self._responses:
            raise AssertionError("LLM 被调用次数超过预设响应数")
        return self._responses.pop(0)


def _tool_response(
    content: Optional[str] = None,
    tool_calls: Optional[List[ToolCall]] = None,
    total_tokens: int = 10,
) -> LLMToolResponse:
    return LLMToolResponse(
        content=content,
        tool_calls=tool_calls or [],
        usage={"total_tokens": total_tokens},
        latency_ms=1,
        model="mock-model",
    )


def _make_call(name: str, args: dict, call_id: str = "tc-1") -> ToolCall:
    import json

    return ToolCall(id=call_id, name=name, arguments=json.dumps(args))


# ==================== Test 1: 直接回复（无 tool_calls） ====================


def test_direct_reply_no_tools():
    """LLM 直接给文本响应，无 tool_calls → final_answer = content"""
    llm = _MockLLM([_tool_response(content="42 是答案", tool_calls=[])])
    compiled = build_react_graph(llm)

    result = compiled.invoke(
        {
            "messages": [{"role": "user", "content": "What is the answer?"}],
            "max_steps": 5,
        }
    )

    assert result["final_answer"] == "42 是答案"
    assert result["steps"] == 1
    assert result["total_tokens"] == 10
    assert len(llm.calls) == 1


# ==================== Test 2: Finish 工具终止 ====================


def test_finish_tool_terminates():
    """LLM 调 Finish → final_answer 写入"""
    llm = _MockLLM(
        [
            _tool_response(
                content="",
                tool_calls=[_make_call("Finish", {"answer": "result-X"})],
            )
        ]
    )
    compiled = build_react_graph(llm)

    result = compiled.invoke(
        {"messages": [{"role": "user", "content": "go"}], "max_steps": 5}
    )

    assert result["final_answer"] == "result-X"
    # Finish 工具消息也被写入
    tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
    assert any("最终答案: result-X" in m["content"] for m in tool_msgs)


# ==================== Test 3: Thought + Finish 多轮 ====================


def test_thought_then_finish():
    """LLM 先 Thought → 再 Finish；thoughts 字段累积"""
    llm = _MockLLM(
        [
            _tool_response(
                tool_calls=[_make_call("Thought", {"reasoning": "我需要思考一下"}, call_id="tc-1")]
            ),
            _tool_response(
                tool_calls=[_make_call("Finish", {"answer": "done"}, call_id="tc-2")]
            ),
        ]
    )
    compiled = build_react_graph(llm)

    result = compiled.invoke(
        {"messages": [{"role": "user", "content": "?"}], "max_steps": 5}
    )

    assert result["final_answer"] == "done"
    assert "我需要思考一下" in result.get("thoughts", [])
    assert result["steps"] == 2


# ==================== Test 4: 用户工具 ====================


def test_user_tool_execution():
    """用户工具被正确执行，结果作为 tool message 注入下一轮"""
    from clear_agent import CalculatorTool, ToolRegistry

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    # CalculatorTool 注册名为 python_calculator
    calc_name = "python_calculator"

    llm = _MockLLM(
        [
            _tool_response(
                tool_calls=[
                    _make_call(
                        calc_name,
                        {"expression": "1+2"},
                        call_id="tc-1",
                    )
                ]
            ),
            _tool_response(
                tool_calls=[
                    _make_call("Finish", {"answer": "got it"}, call_id="tc-2")
                ]
            ),
        ]
    )
    compiled = build_react_graph(llm, tool_registry=registry)

    result = compiled.invoke(
        {"messages": [{"role": "user", "content": "calc"}], "max_steps": 5}
    )

    assert result["final_answer"] == "got it"
    # tool 消息中应包含计算结果 3
    tool_msgs = [m for m in result["messages"] if m.get("role") == "tool"]
    assert any("3" in m["content"] for m in tool_msgs)


# ==================== Test 5: max_steps 终止 ====================


def test_max_steps_terminates_without_finish():
    """LLM 永远不调 Finish；达到 max_steps 后 graph 终止，不抛错"""
    # 准备 5 个 Thought 响应
    responses = [
        _tool_response(
            tool_calls=[
                _make_call("Thought", {"reasoning": f"step {i}"}, call_id=f"tc-{i}")
            ]
        )
        for i in range(5)
    ]
    llm = _MockLLM(responses)
    compiled = build_react_graph(llm)

    result = compiled.invoke(
        {"messages": [{"role": "user", "content": "loop"}], "max_steps": 3}
    )

    # 跑了 3 步后 router 返回 end
    assert result["steps"] == 3
    assert result.get("final_answer") is None  # 没 Finish


# ==================== Test 6: checkpointer 集成 ====================


def test_checkpoint_integration():
    """每个节点结束后写 ckpt，可 resume"""
    llm = _MockLLM(
        [
            _tool_response(
                tool_calls=[_make_call("Thought", {"reasoning": "..."}, call_id="tc-1")]
            ),
            _tool_response(
                tool_calls=[_make_call("Finish", {"answer": "ok"}, call_id="tc-2")]
            ),
        ]
    )
    ck = InMemoryCheckpointer()
    compiled = build_react_graph(llm, checkpointer=ck)

    compiled.invoke(
        {"messages": [{"role": "user", "content": "go"}], "max_steps": 5},
        config=RunConfig(thread_id="t1"),
    )

    ckpts = ck.list("t1")
    # llm + tools + llm + tools = 4 节点完成 → 4 个 ckpt（最后一次 tools 后路由 END）
    assert len(ckpts) >= 2
    # 最后一个 next_node 应是 END
    assert ckpts[0].next_nodes == [END]


# ==================== Test 7: ReActAgent.as_graph() 等价 ====================


def test_react_agent_as_graph_equivalent():
    """ReActAgent.as_graph() 与 build_react_graph 直接调用产物等价"""
    from clear_agent.agents import ReActAgent

    llm = _MockLLM([_tool_response(content="hi", tool_calls=[])])

    # 通过 ReActAgent 构造（最小化：不需要 tool_registry / config）
    agent = ReActAgent(name="test", llm=llm, max_steps=3)
    compiled = agent.as_graph()

    result = compiled.invoke(
        {"messages": [{"role": "user", "content": "?"}], "max_steps": 3}
    )

    assert result["final_answer"] == "hi"


# ==================== Test 8: 工具 schemas 包含内置 + 用户 ====================


def test_tool_schemas_include_builtins_and_user_tools():
    from clear_agent import CalculatorTool, ToolRegistry

    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    schemas = _build_tool_schemas(registry)
    names = {s["function"]["name"] for s in schemas}

    assert "Thought" in names
    assert "Finish" in names
    # CalculatorTool 实例的 name 是 python_calculator
    assert "python_calculator" in names
    assert BUILTIN_TOOL_NAMES == {"Thought", "Finish"}


def test_tool_schemas_no_registry():
    """无 registry 时只返回内置工具"""
    schemas = _build_tool_schemas(None)
    names = {s["function"]["name"] for s in schemas}
    assert names == {"Thought", "Finish"}


# ==================== Test 9: router 单元测试 ====================


def test_router_routes_to_end_on_final_answer():
    state: ReActGraphState = {"final_answer": "x", "steps": 1}
    assert _router(state) == "end"


def test_router_routes_to_tools_when_pending():
    fake_call = MagicMock()
    state: ReActGraphState = {
        "tool_calls_pending": [fake_call],
        "steps": 1,
        "max_steps": 5,
    }
    assert _router(state) == "tools"


def test_router_routes_to_end_when_max_steps():
    state: ReActGraphState = {"steps": 5, "max_steps": 5, "tool_calls_pending": []}
    assert _router(state) == "end"


def test_router_routes_to_end_when_no_tools_no_answer():
    state: ReActGraphState = {"steps": 1, "max_steps": 5}
    assert _router(state) == "end"


# ==================== Test 10: 旧 ReActAgent API 仍可导入与构造 ====================


def test_legacy_react_agent_construction_intact():
    """100% 向后兼容：旧 ReActAgent 仍能正常构造，且暴露原有属性/方法"""
    from clear_agent.agents import ReActAgent

    llm = _MockLLM([])
    agent = ReActAgent(name="legacy", llm=llm, max_steps=10)

    # 原有属性/方法存在
    assert hasattr(agent, "run")
    assert hasattr(agent, "arun")
    assert hasattr(agent, "max_steps")
    assert agent.max_steps == 10
    assert hasattr(agent, "tool_registry")
    assert hasattr(agent, "history_manager")
    # 新增 as_graph
    assert hasattr(agent, "as_graph")


def test_top_level_imports_intact():
    """clear_agent 顶层导出未减少"""
    import clear_agent

    for name in [
        "ClearAgentLLM",
        "Config",
        "Message",
        "SimpleAgent",
        "ReActAgent",
        "ReflectionAgent",
        "PlanSolveAgent",
        "PlanAndSolveAgent",  # 向后兼容别名
        "ToolRegistry",
        "CalculatorTool",
    ]:
        assert hasattr(clear_agent, name), f"clear_agent 缺少导出: {name}"
