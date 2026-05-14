"""StateGraph 基础测试

覆盖 project_docs/01-graph-architecture.md §7 测试清单的 8 项。
"""

from __future__ import annotations

import asyncio
from typing import Annotated, List, TypedDict

import pytest

from clear_agent.core.checkpoint import InMemoryCheckpointer
from clear_agent.core.graph import (
    END,
    START,
    CompiledGraph,
    GraphCompileError,
    GraphError,
    GraphRecursionError,
    RunConfig,
    StateGraph,
    add_messages,
    append_list,
    merge_dict,
    replace,
)


# ==================== State schemas ====================


class SimpleState(TypedDict, total=False):
    counter: int
    log: Annotated[List[str], append_list]


class MsgState(TypedDict, total=False):
    messages: Annotated[List[dict], add_messages]
    metadata: Annotated[dict, merge_dict]


# ==================== Test 1: 线性图执行顺序 ====================


def test_linear_graph_executes_in_order():
    """START → A → B → END 顺序执行，state 累计正确"""
    g: StateGraph[SimpleState] = StateGraph(SimpleState)

    def node_a(s):
        return {"counter": (s.get("counter") or 0) + 1, "log": "A"}

    def node_b(s):
        return {"counter": s["counter"] + 10, "log": "B"}

    g.add_node("a", node_a)
    g.add_node("b", node_b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)

    compiled = g.compile()
    out = compiled.invoke({"counter": 0, "log": []})

    assert out["counter"] == 11
    assert out["log"] == ["A", "B"]


# ==================== Test 2: 条件分支 ====================


def test_conditional_edges_route_correctly():
    """router 返回不同值走不同分支"""
    g: StateGraph[SimpleState] = StateGraph(SimpleState)

    def start_node(s):
        return {"log": "start"}

    def even(s):
        return {"log": "even"}

    def odd(s):
        return {"log": "odd"}

    def router(s):
        return "even" if s.get("counter", 0) % 2 == 0 else "odd"

    g.add_node("start", start_node)
    g.add_node("even", even)
    g.add_node("odd", odd)
    g.add_edge(START, "start")
    g.add_conditional_edges(
        "start", router, {"even": "even", "odd": "odd"}
    )
    g.add_edge("even", END)
    g.add_edge("odd", END)
    compiled = g.compile()

    out_even = compiled.invoke({"counter": 4, "log": []})
    assert "even" in out_even["log"]
    assert "odd" not in out_even["log"]

    out_odd = compiled.invoke({"counter": 7, "log": []})
    assert "odd" in out_odd["log"]
    assert "even" not in out_odd["log"]


# ==================== Test 3: 循环终止 ====================


def test_loop_terminates_via_max_steps():
    """死循环 + max_steps=10 → 抛 GraphRecursionError"""
    g: StateGraph[SimpleState] = StateGraph(SimpleState)

    def loop_node(s):
        return {"counter": (s.get("counter") or 0) + 1}

    g.add_node("loop", loop_node)
    g.add_edge(START, "loop")
    g.add_conditional_edges("loop", lambda s: "loop")  # 永远回自己
    compiled = g.compile()

    with pytest.raises(GraphRecursionError):
        compiled.invoke({"counter": 0, "log": []}, config=RunConfig(max_steps=10))


def test_recursion_limit_triggers_error():
    """同节点重入超 recursion_limit"""
    g: StateGraph[SimpleState] = StateGraph(SimpleState)

    def loop_node(s):
        return {"counter": (s.get("counter") or 0) + 1}

    g.add_node("loop", loop_node)
    g.add_edge(START, "loop")
    g.add_conditional_edges("loop", lambda s: "loop")
    compiled = g.compile()

    with pytest.raises(GraphRecursionError):
        compiled.invoke(
            {"counter": 0, "log": []},
            config=RunConfig(max_steps=100, recursion_limit=5),
        )


# ==================== Test 4: Reducer 合并 ====================


def test_add_messages_appends_and_dedupes():
    """add_messages 按 id 去重并追加"""
    g: StateGraph[MsgState] = StateGraph(MsgState)

    def write_msg1(s):
        return {"messages": [{"id": "m1", "content": "hello"}]}

    def write_msg2(s):
        return {"messages": [{"id": "m2", "content": "world"}]}

    def write_msg1_again(s):
        # 同 id 应该覆盖而非重复
        return {"messages": [{"id": "m1", "content": "updated"}]}

    g.add_node("a", write_msg1)
    g.add_node("b", write_msg2)
    g.add_node("c", write_msg1_again)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", END)
    compiled = g.compile()

    out = compiled.invoke({"messages": [], "metadata": {}})
    assert len(out["messages"]) == 2
    # m1 应被 c 覆盖
    m1 = next(m for m in out["messages"] if m["id"] == "m1")
    assert m1["content"] == "updated"


def test_merge_dict_reducer():
    """merge_dict 浅合并字典"""
    g: StateGraph[MsgState] = StateGraph(MsgState)

    def n1(s):
        return {"metadata": {"a": 1, "b": 2}}

    def n2(s):
        return {"metadata": {"b": 20, "c": 3}}

    g.add_node("n1", n1)
    g.add_node("n2", n2)
    g.add_edge(START, "n1")
    g.add_edge("n1", "n2")
    g.add_edge("n2", END)
    compiled = g.compile()

    out = compiled.invoke({"messages": [], "metadata": {}})
    assert out["metadata"] == {"a": 1, "b": 20, "c": 3}


def test_custom_reducer_via_set_reducer():
    """显式 set_reducer 覆盖 Annotated 元数据"""

    class S(TypedDict, total=False):
        items: list

    def keep_max_3(old, new):
        out = (old or []) + (new if isinstance(new, list) else [new])
        return out[-3:]

    g = StateGraph(S)
    g.set_reducer("items", keep_max_3)
    g.add_node("push1", lambda s: {"items": [1]})
    g.add_node("push2", lambda s: {"items": [2]})
    g.add_node("push3", lambda s: {"items": [3]})
    g.add_node("push4", lambda s: {"items": [4]})
    g.add_edge(START, "push1")
    g.add_edge("push1", "push2")
    g.add_edge("push2", "push3")
    g.add_edge("push3", "push4")
    g.add_edge("push4", END)
    out = g.compile().invoke({"items": []})
    assert out["items"] == [2, 3, 4]


# ==================== Test 5: 同步 / 异步等价 ====================


def test_sync_async_equivalence():
    """invoke 与 ainvoke 同输入返回相同 state"""
    g: StateGraph[SimpleState] = StateGraph(SimpleState)

    def n1(s):
        return {"counter": s.get("counter", 0) + 5, "log": "n1"}

    def n2(s):
        return {"counter": s["counter"] * 2, "log": "n2"}

    g.add_node("n1", n1)
    g.add_node("n2", n2)
    g.add_edge(START, "n1")
    g.add_edge("n1", "n2")
    g.add_edge("n2", END)
    compiled = g.compile()

    sync_out = compiled.invoke({"counter": 1, "log": []})
    async_out = asyncio.run(compiled.ainvoke({"counter": 1, "log": []}))

    assert sync_out == async_out


def test_async_node_in_async_invoke():
    """async 节点可以混在同步图里，由 ainvoke 跑"""
    g: StateGraph[SimpleState] = StateGraph(SimpleState)

    async def async_node(s):
        await asyncio.sleep(0)
        return {"counter": 42, "log": "async_done"}

    g.add_node("a", async_node)
    g.add_edge(START, "a")
    g.add_edge("a", END)
    compiled = g.compile()

    out = asyncio.run(compiled.ainvoke({"counter": 0, "log": []}))
    assert out["counter"] == 42


def test_async_node_with_sync_invoke_raises():
    """同步 invoke 调用 async 节点应抛 GraphError"""
    from clear_agent.core.graph import GraphError

    g: StateGraph[SimpleState] = StateGraph(SimpleState)

    async def async_node(s):
        return {"counter": 1}

    g.add_node("a", async_node)
    g.add_edge(START, "a")
    g.add_edge("a", END)
    compiled = g.compile()

    with pytest.raises(GraphError):
        compiled.invoke({"counter": 0, "log": []})


# ==================== Test 6: 流式事件顺序 ====================


def test_stream_event_sequence():
    """stream 产出事件顺序符合 NODE_START → NODE_FINISH → EDGE → CHECKPOINT"""
    g: StateGraph[SimpleState] = StateGraph(SimpleState)

    g.add_node("a", lambda s: {"counter": 1, "log": "a"})
    g.add_node("b", lambda s: {"counter": s["counter"] + 1, "log": "b"})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    compiled = g.compile(checkpointer=InMemoryCheckpointer())

    events = list(compiled.stream({"counter": 0, "log": []}))
    types = [e.type for e in events]

    # 第一个 edge 来自 START 路由
    assert types[0] == "edge"
    # 然后 a: node_start, node_finish, edge, checkpoint
    assert "node_start" in types
    assert "node_finish" in types
    assert "checkpoint" in types
    assert types[-1] == "end"
    # node_start 必然在 node_finish 之前
    a_start = next(i for i, e in enumerate(events) if e.type == "node_start" and e.node == "a")
    a_finish = next(i for i, e in enumerate(events) if e.type == "node_finish" and e.node == "a")
    assert a_start < a_finish


# ==================== Test 7: Mermaid 输出 ====================


def test_draw_mermaid_basic():
    """draw_mermaid 生成有效 mermaid 语法"""
    g: StateGraph[SimpleState] = StateGraph(SimpleState)
    g.add_node("a", lambda s: {})
    g.add_node("b", lambda s: {})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    compiled = g.compile()

    out = compiled.draw_mermaid()
    assert out.startswith("flowchart TD")
    assert "__start__" in out
    assert "__end__" in out
    assert "a --> b" in out


def test_draw_mermaid_with_conditional():
    """条件边用虚线 -.label.->"""
    g: StateGraph[SimpleState] = StateGraph(SimpleState)
    g.add_node("a", lambda s: {})
    g.add_node("b", lambda s: {})
    g.add_node("c", lambda s: {})
    g.add_edge(START, "a")
    g.add_conditional_edges("a", lambda s: "x", {"x": "b", "y": "c"})
    g.add_edge("b", END)
    g.add_edge("c", END)
    compiled = g.compile()

    out = compiled.draw_mermaid()
    assert "-.x.->" in out or "-.y.->" in out


def test_compile_rejects_multiple_static_edges_from_same_source():
    """当前执行器只支持单后继；多条静态边必须显式拒绝。"""
    g: StateGraph[SimpleState] = StateGraph(SimpleState)
    g.add_node("a", lambda s: {})
    g.add_node("b", lambda s: {})
    g.add_node("c", lambda s: {})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("a", "c")
    g.add_edge("b", END)
    g.add_edge("c", END)

    with pytest.raises(GraphCompileError, match="multiple outgoing"):
        g.compile()


def test_compile_rejects_conditional_mapping_fanout():
    """mapping value 为 list 会被旧实现静默取第一个，必须拒绝。"""
    g: StateGraph[SimpleState] = StateGraph(SimpleState)
    g.add_node("a", lambda s: {})
    g.add_node("b", lambda s: {})
    g.add_node("c", lambda s: {})
    g.add_edge(START, "a")
    g.add_conditional_edges("a", lambda s: "both", {"both": ["b", "c"]})
    g.add_edge("b", END)
    g.add_edge("c", END)

    with pytest.raises(GraphCompileError, match="fan-out"):
        g.compile()


def test_runtime_rejects_router_list_fanout():
    """router 直接返回 list 时也不能静默丢分支。"""
    g: StateGraph[SimpleState] = StateGraph(SimpleState)
    g.add_node("a", lambda s: {"log": "a"})
    g.add_node("b", lambda s: {"log": "b"})
    g.add_node("c", lambda s: {"log": "c"})
    g.add_edge(START, "a")
    g.add_conditional_edges("a", lambda s: ["b", "c"])
    g.add_edge("b", END)
    g.add_edge("c", END)

    compiled = g.compile()
    with pytest.raises(GraphError, match="fan-out"):
        compiled.invoke({"log": []})


# ==================== Test 8: 错误传播 ====================


def test_error_in_node_raises_by_default():
    """节点抛异常时 RunConfig.on_error='raise'（默认）抛出"""
    g: StateGraph[SimpleState] = StateGraph(SimpleState)

    def boom(s):
        raise ValueError("boom!")

    g.add_node("boom", boom)
    g.add_edge(START, "boom")
    g.add_edge("boom", END)
    compiled = g.compile()

    with pytest.raises(ValueError, match="boom"):
        compiled.invoke({"counter": 0, "log": []})


def test_error_record_and_continue():
    """on_error='record_and_continue' 把错误写进 state 并跳到 END"""
    g: StateGraph[SimpleState] = StateGraph(SimpleState)

    def boom(s):
        raise ValueError("boom!")

    g.add_node("boom", boom)
    g.add_edge(START, "boom")
    g.add_edge("boom", END)
    compiled = g.compile()

    out = compiled.invoke(
        {"counter": 0, "log": []},
        config=RunConfig(on_error="record_and_continue"),
    )
    assert out["__error__"] == "boom!"
    assert out["__error_node__"] == "boom"


# ==================== 编译期校验 ====================


def test_compile_error_unknown_node_in_edge():
    g: StateGraph[SimpleState] = StateGraph(SimpleState)
    g.add_node("a", lambda s: {})
    g.add_edge(START, "a")
    g.add_edge("a", "nonexistent")  # 未定义
    with pytest.raises(GraphCompileError):
        g.compile()


def test_compile_error_no_start_edge():
    g: StateGraph[SimpleState] = StateGraph(SimpleState)
    g.add_node("a", lambda s: {})
    g.add_edge("a", END)
    with pytest.raises(GraphCompileError):
        g.compile()


def test_reserved_name_rejected():
    g: StateGraph[SimpleState] = StateGraph(SimpleState)
    with pytest.raises(GraphCompileError):
        g.add_node(START, lambda s: {})


# ==================== Checkpointer 集成（W1 范围内的最小验证）====================


def test_checkpointer_writes_per_node():
    g: StateGraph[SimpleState] = StateGraph(SimpleState)
    g.add_node("a", lambda s: {"counter": 1, "log": "a"})
    g.add_node("b", lambda s: {"counter": s["counter"] + 1, "log": "b"})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)

    ck = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=ck)
    out = compiled.invoke(
        {"counter": 0, "log": []}, config=RunConfig(thread_id="t1")
    )

    ckpts = ck.list("t1")
    # 经过 a 和 b 两个节点 → 至少 2 个 checkpoint
    assert len(ckpts) == 2
    # 最新的 next_node 应是 END
    assert ckpts[0].next_nodes == [END]
    assert ckpts[0].metadata["node"] == "b"


def test_get_state_returns_latest():
    g: StateGraph[SimpleState] = StateGraph(SimpleState)
    g.add_node("a", lambda s: {"counter": 99, "log": "a"})
    g.add_edge(START, "a")
    g.add_edge("a", END)

    ck = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=ck)
    compiled.invoke({"counter": 0, "log": []}, config=RunConfig(thread_id="t1"))

    state = compiled.get_state("t1")
    assert state is not None
    assert state["counter"] == 99
