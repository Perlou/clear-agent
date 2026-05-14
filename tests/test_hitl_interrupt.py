"""HITL（Human-in-the-Loop）测试

覆盖 project_docs/03-hitl-guide.md §8 测试清单。
用合成 graph（无 LLM）验证 interrupt / resume 全部语义。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, TypedDict

import pytest

from clear_agent.core.config import Config
from clear_agent.core.checkpoint import (
    InMemoryCheckpointer,
    JsonFileCheckpointer,
    SqliteCheckpointer,
)
from clear_agent.core.graph import (
    END,
    START,
    RunConfig,
    StateGraph,
    append_list,
)
from clear_agent.hitl import (
    GraphPaused,
    InterruptExpiredError,
    approval,
    edit_state,
    interrupt,
    validate_tool_args,
)


# ==================== 公共 fixtures ====================


class S(TypedDict, total=False):
    log: Annotated[List[str], append_list]
    payload: Dict[str, Any]
    decision: Any


# ==================== Test 1: 基础 interrupt ====================


def test_interrupt_raises_graph_paused_and_writes_ckpt():
    """节点内 interrupt(payload) → GraphPaused 抛出 + ckpt 写入（source=interrupt）"""

    def n1(state):
        decision = interrupt({"type": "approval", "prompt": "go?"})
        return {"log": f"got:{decision}", "decision": decision}

    g = StateGraph(S)
    g.add_node("n1", n1)
    g.add_edge(START, "n1")
    g.add_edge("n1", END)
    ck = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=ck)

    with pytest.raises(GraphPaused) as ei:
        compiled.invoke({"log": []}, config=RunConfig(thread_id="t1"))

    paused = ei.value
    assert paused.thread_id == "t1"
    assert paused.payload["type"] == "approval"
    assert paused.checkpoint_id

    # checkpoint 已写入
    ckpts = ck.list("t1")
    assert len(ckpts) == 1
    assert ckpts[0].metadata["source"] == "interrupt"
    assert ckpts[0].metadata["payload"]["prompt"] == "go?"
    assert ckpts[0].next_nodes == ["n1"]  # 重入相同节点


# ==================== Test 2: resume(value=...) 注入 ====================


def test_resume_with_value_continues_execution():
    """resume(value=X) 后节点重入，interrupt() 直接返回 X"""

    def n1(state):
        decision = interrupt({"type": "approval", "prompt": "?"})
        return {"log": f"got:{decision}", "decision": decision}

    g = StateGraph(S)
    g.add_node("n1", n1)
    g.add_edge(START, "n1")
    g.add_edge("n1", END)
    ck = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=ck)

    with pytest.raises(GraphPaused):
        compiled.invoke({"log": []}, config=RunConfig(thread_id="t1"))

    # 注入 "approve"
    final = compiled.resume(thread_id="t1", value="approve")

    assert final["decision"] == "approve"
    assert "got:approve" in final["log"]


def test_resume_expired_interrupt_raises():
    """interrupt checkpoint 超过 TTL 后不应继续 resume。"""

    def n1(state):
        decision = interrupt({"type": "approval", "prompt": "?"})
        return {"log": f"got:{decision}", "decision": decision}

    g = StateGraph(S)
    g.add_node("n1", n1)
    g.add_edge(START, "n1")
    g.add_edge("n1", END)
    ck = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=ck)

    with pytest.raises(GraphPaused):
        compiled.invoke({"log": []}, config=RunConfig(thread_id="t-expired"))

    ckpt = ck.list("t-expired")[0]
    ttl = Config().hitl_interrupt_ttl_seconds
    ckpt.created_at = datetime.now() - timedelta(seconds=ttl + 1)

    with pytest.raises(InterruptExpiredError):
        compiled.resume(thread_id="t-expired", value="approve")


# ==================== Test 3: 进程重启后 resume ====================


def test_resume_after_process_restart(tmp_path):
    """JsonFile checkpointer：第一个 compiled 抛 paused，新 compiled 实例能 resume"""

    def n1(state):
        decision = interrupt({"type": "approval", "prompt": "?"})
        return {"log": f"got:{decision}"}

    base = str(tmp_path / "ckpts")

    # 进程 #1：抛 GraphPaused
    g1 = StateGraph(S)
    g1.add_node("n1", n1)
    g1.add_edge(START, "n1")
    g1.add_edge("n1", END)
    ck1 = JsonFileCheckpointer(base_dir=base)
    compiled1 = g1.compile(checkpointer=ck1)
    with pytest.raises(GraphPaused):
        compiled1.invoke({"log": []}, config=RunConfig(thread_id="t-restart"))

    # 进程 #2（新实例）：能 resume
    g2 = StateGraph(S)
    g2.add_node("n1", n1)
    g2.add_edge(START, "n1")
    g2.add_edge("n1", END)
    ck2 = JsonFileCheckpointer(base_dir=base)
    compiled2 = g2.compile(checkpointer=ck2)

    final = compiled2.resume(thread_id="t-restart", value="ok")
    assert "got:ok" in final["log"]


# ==================== Test 4: 流式中断事件 ====================


def test_stream_emits_interrupt_event_then_raises():
    """astream/stream 在中断时 yield type='interrupt' 事件，紧接抛 GraphPaused"""

    def n1(state):
        decision = interrupt({"type": "approval", "prompt": "?"})
        return {"log": f"got:{decision}"}

    g = StateGraph(S)
    g.add_node("n1", n1)
    g.add_edge(START, "n1")
    g.add_edge("n1", END)
    ck = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=ck)

    events: List[Any] = []

    with pytest.raises(GraphPaused):
        for ev in compiled.stream({"log": []}, config=RunConfig(thread_id="t1")):
            events.append(ev)

    types = [e.type for e in events]
    assert "interrupt" in types
    interrupt_ev = next(e for e in events if e.type == "interrupt")
    assert interrupt_ev.data["payload"]["prompt"] == "?"
    assert interrupt_ev.data["checkpoint_id"]


# ==================== Test 5: approval 三选项 ====================


def test_approval_pattern_three_options():
    """approval helper 支持 approve/reject/edit 三种选项；接受 dict 或字符串回执"""

    def n1(state):
        choice = approval("Send?", options=["approve", "reject", "edit"])
        return {"log": f"choice:{choice}", "decision": choice}

    g = StateGraph(S)
    g.add_node("n1", n1)
    g.add_edge(START, "n1")
    g.add_edge("n1", END)
    ck = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=ck)

    # approve（字符串 value）
    with pytest.raises(GraphPaused):
        compiled.invoke({"log": []}, config=RunConfig(thread_id="t-a"))
    final_a = compiled.resume(thread_id="t-a", value="approve")
    assert final_a["decision"] == "approve"

    # reject（dict 形式 {"choice": "reject"}）
    ck.clear()
    with pytest.raises(GraphPaused):
        compiled.invoke({"log": []}, config=RunConfig(thread_id="t-b"))
    final_b = compiled.resume(thread_id="t-b", value={"choice": "reject"})
    assert final_b["decision"] == "reject"


# ==================== Test 6: edit_state 修改字段 ====================


def test_edit_state_pattern_modifies_fields():
    """edit_state 让用户改字段，未提及的字段保留原值"""

    def n1(state):
        edited = edit_state(
            ["plan", "tone"],
            current_values={"plan": "原计划", "tone": "casual"},
        )
        return {
            "log": f"plan:{edited['plan']}",
            "payload": edited,
        }

    g = StateGraph(S)
    g.add_node("n1", n1)
    g.add_edge(START, "n1")
    g.add_edge("n1", END)
    ck = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=ck)

    with pytest.raises(GraphPaused):
        compiled.invoke({"log": []}, config=RunConfig(thread_id="t1"))

    # 只改 plan，不改 tone
    final = compiled.resume(thread_id="t1", value={"plan": "改后的计划"})
    assert final["payload"] == {"plan": "改后的计划", "tone": "casual"}


# ==================== Test 7: validate_tool_args 改参 / 拒绝 ====================


def test_validate_tool_args_modify():
    """validate_tool_args 接受 args 改写"""

    def n1(state):
        validated = validate_tool_args(
            "send_email",
            proposed_args={"to": "alice@example.com", "body": "..."},
            sensitive_fields=["to"],
        )
        return {"payload": validated}

    g = StateGraph(S)
    g.add_node("n1", n1)
    g.add_edge(START, "n1")
    g.add_edge("n1", END)
    ck = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=ck)

    with pytest.raises(GraphPaused):
        compiled.invoke({"log": []}, config=RunConfig(thread_id="t1"))

    # 改 to 字段
    final = compiled.resume(
        thread_id="t1",
        value={"approved": True, "args": {"to": "bob@example.com", "body": "..."}},
    )
    assert final["payload"]["to"] == "bob@example.com"


def test_validate_tool_args_reject_raises():
    """approved=False 时节点内 validate_tool_args 抛 ValueError，graph 默认 raise"""

    def n1(state):
        validate_tool_args("danger", proposed_args={"x": 1})
        return {}  # not reached

    g = StateGraph(S)
    g.add_node("n1", n1)
    g.add_edge(START, "n1")
    g.add_edge("n1", END)
    ck = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=ck)

    with pytest.raises(GraphPaused):
        compiled.invoke({"log": []}, config=RunConfig(thread_id="t1"))

    with pytest.raises(ValueError, match="拒绝"):
        compiled.resume(thread_id="t1", value={"approved": False, "reason": "no"})


# ==================== Test 8: 嵌套 interrupt（同节点连续两次） ====================


def test_two_interrupts_in_same_node_run_sequentially():
    """同一个节点连续两次 interrupt，需要两次 resume；每次 value 仅消费一次"""

    def n1(state):
        a = interrupt({"type": "custom", "step": "first"})
        b = interrupt({"type": "custom", "step": "second"})
        return {"log": f"a={a},b={b}"}

    g = StateGraph(S)
    g.add_node("n1", n1)
    g.add_edge(START, "n1")
    g.add_edge("n1", END)
    ck = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=ck)

    # 第一次 interrupt
    with pytest.raises(GraphPaused) as ei1:
        compiled.invoke({"log": []}, config=RunConfig(thread_id="t1"))
    assert ei1.value.payload["step"] == "first"

    # 注入 "X"，但节点会重入并再次 interrupt（第二次）
    with pytest.raises(GraphPaused) as ei2:
        compiled.resume(thread_id="t1", value="X")
    assert ei2.value.payload["step"] == "second"

    # 注入 "Y"，节点完整执行
    final = compiled.resume(thread_id="t1", value="Y")
    assert "a=X,b=Y" in final["log"]


# ==================== Test 9: 普通流程（无 interrupt） 不受影响 ====================


def test_no_interrupt_no_pause():
    """没有调 interrupt 的图正常执行"""

    def n1(state):
        return {"log": "n1"}

    g = StateGraph(S)
    g.add_node("n1", n1)
    g.add_edge(START, "n1")
    g.add_edge("n1", END)
    ck = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=ck)

    final = compiled.invoke({"log": []})
    assert final["log"] == ["n1"]
    # 没有 interrupt 类型的 ckpt
    assert all(c.metadata.get("source") != "interrupt" for c in ck.list("t1"))


# ==================== Test 10: interrupt 在节点外调用抛错 ====================


def test_interrupt_outside_node_raises():
    """直接在 graph 外调 interrupt → RuntimeError"""
    with pytest.raises(RuntimeError, match="只能在 graph 节点函数内调用"):
        interrupt({"type": "x"})


# ==================== Test 11: SqliteCheckpointer 也支持 HITL ====================


def test_sqlite_checkpointer_with_interrupt(tmp_path):
    def n1(state):
        return {"log": f"got:{interrupt({'type': 'a'})}"}

    g = StateGraph(S)
    g.add_node("n1", n1)
    g.add_edge(START, "n1")
    g.add_edge("n1", END)
    ck = SqliteCheckpointer(db_path=str(tmp_path / "h.db"))
    compiled = g.compile(checkpointer=ck)

    with pytest.raises(GraphPaused):
        compiled.invoke({"log": []}, config=RunConfig(thread_id="t1"))

    final = compiled.resume(thread_id="t1", value="OK")
    assert "got:OK" in final["log"]


# ==================== Test 12: state_patch + resume(value) 同时 ====================


def test_resume_with_value_and_state_patch():
    """resume 同时支持 value（HITL 回执）和 state_patch（state 字段修改）"""

    def n1(state):
        choice = interrupt({"type": "approval"})
        return {"log": f"choice:{choice},counter:{state.get('counter', 0)}"}

    class S2(TypedDict, total=False):
        log: Annotated[List[str], append_list]
        counter: int

    g = StateGraph(S2)
    g.add_node("n1", n1)
    g.add_edge(START, "n1")
    g.add_edge("n1", END)
    ck = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=ck)

    with pytest.raises(GraphPaused):
        compiled.invoke({"log": [], "counter": 1}, config=RunConfig(thread_id="t1"))

    final = compiled.resume(
        thread_id="t1",
        value="ok",
        state_patch={"counter": 999},
    )
    assert "choice:ok,counter:999" in final["log"]
