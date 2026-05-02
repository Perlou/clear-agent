"""Multi-agent 范式包测试

覆盖三种范式：
- Handoff 原语（数据类 / 工具 schema / tool_calls 解析）
- Supervisor pattern（中心化路由）
- Swarm pattern（去中心化 handoff）
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from clear_agent.core.checkpoint import InMemoryCheckpointer
from clear_agent.core.graph import END, RunConfig
from clear_agent.multiagent import (
    HANDOFF_END,
    HANDOFF_TOOL_PREFIX,
    Handoff,
    SupervisorState,
    SwarmState,
    build_supervisor_graph,
    build_swarm_graph,
    make_handoff_tool,
    make_handoff_tool_name,
    make_handoff_tools,
    parse_handoff_from_tool_calls,
)


# ==================== Section A: Handoff 数据类 ====================


def test_handoff_basic():
    h = Handoff(target="writer")
    assert h.target == "writer"
    assert h.message == ""
    assert h.state_patch == {}


def test_handoff_with_message_and_patch():
    h = Handoff(target="researcher", message="find papers", state_patch={"topic": "AI"})
    d = h.to_dict()
    assert d["target"] == "researcher"
    assert d["message"] == "find papers"
    assert d["state_patch"] == {"topic": "AI"}


# ==================== Section B: Handoff Tool Schema ====================


def test_make_handoff_tool_name_basic():
    assert make_handoff_tool_name("writer") == f"{HANDOFF_TOOL_PREFIX}writer"


def test_make_handoff_tool_name_sanitizes_special_chars():
    """非法字符替换为 _"""
    out = make_handoff_tool_name("writer-2.0")
    assert "-" not in out
    assert "." not in out


def test_make_handoff_tool_shape():
    tool = make_handoff_tool("researcher")
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "transfer_to_researcher"
    assert "Transfer control" in tool["function"]["description"]
    params = tool["function"]["parameters"]
    assert params["type"] == "object"
    assert "message" in params["properties"]


def test_make_handoff_tool_custom_description():
    tool = make_handoff_tool("x", description="My custom desc")
    assert tool["function"]["description"] == "My custom desc"


def test_make_handoff_tools_batch():
    tools = make_handoff_tools(["a", "b", "c"])
    assert len(tools) == 3
    names = {t["function"]["name"] for t in tools}
    assert names == {"transfer_to_a", "transfer_to_b", "transfer_to_c"}


def test_make_handoff_tools_with_descriptions():
    tools = make_handoff_tools(["a", "b"], descriptions={"a": "alpha role"})
    by_name = {t["function"]["name"]: t for t in tools}
    assert by_name["transfer_to_a"]["function"]["description"] == "alpha role"
    # b 走默认
    assert "Transfer control" in by_name["transfer_to_b"]["function"]["description"]


# ==================== Section C: parse_handoff_from_tool_calls ====================


class _FakeToolCall:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


def test_parse_handoff_from_object_tool_calls():
    tc = _FakeToolCall("transfer_to_writer", json.dumps({"message": "do it"}))
    ho = parse_handoff_from_tool_calls([tc])
    assert ho is not None
    assert ho.target == "writer"
    assert ho.message == "do it"


def test_parse_handoff_from_dict_tool_calls():
    tc = {"name": "transfer_to_writer", "arguments": json.dumps({"message": "x"})}
    ho = parse_handoff_from_tool_calls([tc])
    assert ho is not None
    assert ho.target == "writer"


def test_parse_handoff_no_handoff_call_returns_none():
    tc = _FakeToolCall("calculator", "{}")
    assert parse_handoff_from_tool_calls([tc]) is None


def test_parse_handoff_empty_list():
    assert parse_handoff_from_tool_calls([]) is None


def test_parse_handoff_picks_first_handoff_among_others():
    tcs = [
        _FakeToolCall("calculator", "{}"),
        _FakeToolCall("transfer_to_b", json.dumps({"message": "go b"})),
        _FakeToolCall("transfer_to_a", json.dumps({"message": "go a"})),  # 不会取这个
    ]
    ho = parse_handoff_from_tool_calls(tcs)
    assert ho.target == "b"
    assert ho.message == "go b"


def test_parse_handoff_invalid_json_args_returns_empty_message():
    tc = _FakeToolCall("transfer_to_x", "{not valid json")
    ho = parse_handoff_from_tool_calls([tc])
    assert ho is not None
    assert ho.target == "x"
    assert ho.message == ""


def test_parse_handoff_dict_args():
    tc = _FakeToolCall("transfer_to_x", {"message": "from dict"})
    ho = parse_handoff_from_tool_calls([tc])
    assert ho.message == "from dict"


# ==================== Section D: Supervisor build 边界 ====================


def test_build_supervisor_empty_workers_raises():
    with pytest.raises(ValueError):
        build_supervisor_graph(lambda s: {}, workers={})


def test_build_supervisor_reserved_name_raises():
    with pytest.raises(ValueError):
        build_supervisor_graph(
            lambda s: {}, workers={"supervisor": lambda s: {}}
        )


def test_build_supervisor_start_name_raises():
    with pytest.raises(ValueError):
        build_supervisor_graph(lambda s: {}, workers={END: lambda s: {}})


# ==================== Section E: Supervisor 端到端 ====================


def test_supervisor_routes_through_workers_and_terminates():
    """supervisor 路由 researcher → writer → END"""
    routes = ["researcher", "writer", HANDOFF_END]

    def supervisor(state):
        n = state.get("handoff_count", 0)
        return {"active_agent": routes[n] if n < len(routes) else HANDOFF_END}

    def researcher(state):
        return {"messages": [{"role": "assistant", "content": "data"}]}

    def writer(state):
        return {"messages": [{"role": "assistant", "content": "report"}]}

    g = build_supervisor_graph(supervisor, {"researcher": researcher, "writer": writer})
    result = g.invoke({"messages": []})

    assert result["handoff_count"] == 2
    msgs = [m for m in result["messages"] if m.get("role") == "assistant"]
    contents = [m["content"] for m in msgs]
    assert "data" in contents
    assert "report" in contents


def test_supervisor_max_handoffs_enforced():
    """supervisor 永远不返回 END，max_handoffs 强制终止"""

    def supervisor(state):
        return {"active_agent": "loop"}

    def loop_worker(state):
        return {"messages": [{"role": "assistant", "content": "loop"}]}

    g = build_supervisor_graph(supervisor, {"loop": loop_worker}, max_handoffs=3)
    result = g.invoke({"messages": [], "max_handoffs": 3})
    assert result["handoff_count"] == 3


def test_supervisor_unknown_target_routes_to_end():
    """supervisor 返回不存在的 worker 名 → 路由到 END"""

    def supervisor(state):
        return {"active_agent": "ghost"}

    def real(state):
        return {}

    g = build_supervisor_graph(supervisor, {"real": real})
    result = g.invoke({"messages": []})
    # 第一次 supervisor 直接路由 END
    assert result.get("handoff_count", 0) == 0


def test_supervisor_active_agent_none_routes_to_end():
    def supervisor(state):
        return {}  # 不设 active_agent

    def w(state):
        return {}

    g = build_supervisor_graph(supervisor, {"w": w})
    result = g.invoke({"messages": []})
    assert result.get("handoff_count", 0) == 0


def test_supervisor_worker_clears_active_agent():
    """worker 跑完后 active_agent 应被清空，让 supervisor 重新决策"""
    calls = {"supervisor": 0}

    def supervisor(state):
        calls["supervisor"] += 1
        # 第 1 次路 w；第 2 次 END
        return {"active_agent": "w" if calls["supervisor"] == 1 else HANDOFF_END}

    def w(state):
        return {"result": "done"}

    g = build_supervisor_graph(supervisor, {"w": w})
    result = g.invoke({"messages": []})
    assert result["result"] == "done"
    # supervisor 被调用 2 次
    assert calls["supervisor"] == 2


def test_supervisor_with_checkpointer():
    """checkpointer 集成正常"""

    def supervisor(state):
        n = state.get("handoff_count", 0)
        return {"active_agent": "w" if n == 0 else HANDOFF_END}

    def w(state):
        return {"messages": [{"role": "assistant", "content": "x"}]}

    ck = InMemoryCheckpointer()
    g = build_supervisor_graph(supervisor, {"w": w}, checkpointer=ck)
    g.invoke({"messages": []}, config=RunConfig(thread_id="t1"))
    ckpts = ck.list("t1")
    assert len(ckpts) >= 2
    assert ckpts[0].next_nodes == [END]


# ==================== Section F: Swarm build 边界 ====================


def test_build_swarm_empty_agents_raises():
    with pytest.raises(ValueError):
        build_swarm_graph({}, default_active="x")


def test_build_swarm_unknown_default_active_raises():
    with pytest.raises(ValueError):
        build_swarm_graph({"a": lambda s: {}}, default_active="ghost")


def test_build_swarm_reserved_name_raises():
    with pytest.raises(ValueError):
        build_swarm_graph(
            {"_swarm_dispatch": lambda s: {}}, default_active="_swarm_dispatch"
        )


# ==================== Section G: Swarm 端到端 ====================


def test_swarm_handoff_chain():
    """planner → executor → END"""

    def planner(state):
        return {
            "messages": [{"role": "assistant", "content": "plan"}],
            "active_agent": "executor",
        }

    def executor(state):
        return {
            "messages": [{"role": "assistant", "content": "execute"}],
            "active_agent": HANDOFF_END,
        }

    g = build_swarm_graph(
        {"planner": planner, "executor": executor}, default_active="planner"
    )
    result = g.invoke({"messages": []})
    msgs = [m["content"] for m in result["messages"] if m.get("role") == "assistant"]
    assert "plan" in msgs
    assert "execute" in msgs
    assert result["handoff_count"] == 2  # planner→executor + executor→END


def test_swarm_self_continuation_no_handoff_count():
    """agent 返回自己名字 → 不算 handoff，但会再次进入"""
    counts = {"a": 0}

    def a(state):
        counts["a"] += 1
        if counts["a"] >= 3:
            return {"active_agent": HANDOFF_END}
        return {
            "messages": [{"role": "assistant", "content": f"step{counts['a']}"}],
            "active_agent": "a",  # 继续自己
        }

    g = build_swarm_graph({"a": a}, default_active="a", max_handoffs=10)
    result = g.invoke({"messages": []})
    assert counts["a"] == 3
    # 自我继续不计 handoff_count；最后 a→END 算 1
    assert result["handoff_count"] == 1


def test_swarm_max_handoffs_terminates():
    """两 agent 互相 ping-pong，max_handoffs=3 强制终止"""

    def a(state):
        return {"active_agent": "b"}

    def b(state):
        return {"active_agent": "a"}

    g = build_swarm_graph({"a": a, "b": b}, default_active="a", max_handoffs=3)
    result = g.invoke({"messages": []})
    assert result["handoff_count"] == 3


def test_swarm_unknown_handoff_target_routes_end():
    def a(state):
        return {"active_agent": "ghost"}

    g = build_swarm_graph({"a": a}, default_active="a")
    result = g.invoke({"messages": []})
    # 走完 a 一次后路由发现 ghost 不存在 → END
    # handoff_count: a→ghost 算切换不同名 → +1
    assert result["handoff_count"] == 1


def test_swarm_default_active_used_when_none():
    """state 中没指定 active_agent → 用 default_active"""
    captured = {}

    def a(state):
        captured["active_at_a"] = state.get("active_agent")
        return {"active_agent": HANDOFF_END}

    g = build_swarm_graph({"a": a}, default_active="a")
    g.invoke({"messages": []})
    assert captured["active_at_a"] == "a"


def test_swarm_explicit_active_overrides_default():
    """state 中显式传 active_agent → 入口直接走该 agent"""

    def a(state):
        return {"active_agent": HANDOFF_END}

    def b(state):
        return {
            "messages": [{"role": "assistant", "content": "b ran"}],
            "active_agent": HANDOFF_END,
        }

    g = build_swarm_graph({"a": a, "b": b}, default_active="a")
    result = g.invoke({"messages": [], "active_agent": "b"})
    contents = [m["content"] for m in result["messages"] if m.get("role") == "assistant"]
    assert "b ran" in contents


def test_swarm_with_checkpointer():
    def a(state):
        return {"active_agent": HANDOFF_END}

    ck = InMemoryCheckpointer()
    g = build_swarm_graph({"a": a}, default_active="a", checkpointer=ck)
    g.invoke({"messages": []}, config=RunConfig(thread_id="s1"))
    ckpts = ck.list("s1")
    assert len(ckpts) >= 2
    assert ckpts[0].next_nodes == [END]


# ==================== Section H: 顶层导入 ====================


def test_top_level_multiagent_imports():
    from clear_agent.multiagent import (
        HANDOFF_END,
        Handoff,
        SupervisorState,
        SwarmState,
        build_supervisor_graph,
        build_swarm_graph,
        make_handoff_tool,
        parse_handoff_from_tool_calls,
    )

    assert callable(build_supervisor_graph)
    assert callable(build_swarm_graph)
    assert callable(make_handoff_tool)
    assert callable(parse_handoff_from_tool_calls)
    assert HANDOFF_END is not None
    assert Handoff is not None


# ==================== Section I: 集成 —— supervisor + ReActAgent ====================


def test_supervisor_with_callable_wrapping_react_agent():
    """演示：把 ReActAgent.run 包成 worker 接口"""
    from unittest.mock import patch

    # 包一层 stub agent worker
    def make_react_worker(agent_name: str):
        def _worker(state):
            return {
                "messages": [
                    {"role": "assistant", "content": f"{agent_name} ran"}
                ],
            }

        return _worker

    routes = ["a", "b", HANDOFF_END]

    def supervisor(state):
        n = state.get("handoff_count", 0)
        return {"active_agent": routes[n] if n < len(routes) else HANDOFF_END}

    g = build_supervisor_graph(
        supervisor,
        {"a": make_react_worker("a"), "b": make_react_worker("b")},
    )
    result = g.invoke({"messages": []})
    contents = [m["content"] for m in result["messages"] if m.get("role") == "assistant"]
    assert "a ran" in contents
    assert "b ran" in contents
