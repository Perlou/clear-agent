"""HITL（Human-in-the-Loop）演示 —— 审批工作流

跑这个文件可看到：
1. 节点中调 ``interrupt()`` → graph 暂停 + 写 checkpoint
2. 调用方收到 ``GraphPaused`` 并展示 payload
3. ``compiled.resume(thread_id, value=...)`` 注入决策续跑
4. 同一 thread 多次中断的回放行为

不依赖外部 LLM；用纯 Python 节点演示流程。

运行：
    python examples/hitl_demo.py
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, TypedDict

from clear_agent.core.checkpoint import InMemoryCheckpointer
from clear_agent.core.graph import (
    END,
    START,
    RunConfig,
    StateGraph,
    add_messages,
)
from clear_agent.core.interrupt import GraphPaused, interrupt


class State(TypedDict, total=False):
    messages: Annotated[List[Dict[str, Any]], add_messages]
    draft: str
    sent: bool


def draft_node(state: State) -> Dict[str, Any]:
    return {
        "draft": "Hi! Want to grab coffee tomorrow at 10?",
        "messages": [{"role": "system", "content": "起草完成"}],
    }


def approval_node(state: State) -> Dict[str, Any]:
    """关键节点：调 interrupt 暂停等待人类审批"""
    decision = interrupt(
        {
            "type": "approval",
            "message": "Send this email?",
            "draft": state.get("draft", ""),
        }
    )
    if not decision.get("approved"):
        return {
            "messages": [{"role": "system", "content": f"❌ 用户拒绝：{decision.get('reason', '')}"}],
            "sent": False,
        }
    return {
        "messages": [{"role": "system", "content": "✅ 邮件已发送"}],
        "sent": True,
    }


def main() -> None:
    print("=" * 60)
    print("HITL Demo: 审批工作流")
    print("=" * 60)

    g: StateGraph[State] = StateGraph(State)
    g.add_node("draft", draft_node)
    g.add_node("approval", approval_node)
    g.add_edge(START, "draft")
    g.add_edge("draft", "approval")
    g.add_edge("approval", END)

    ck = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=ck)

    # ============== 第一次跑 ==============
    print("\n[Round 1] 跑到 approval 节点会触发暂停")
    try:
        compiled.invoke({"messages": []}, config=RunConfig(thread_id="email-1"))
    except GraphPaused as p:
        print(f"  📨 收到暂停信号 thread={p.thread_id}")
        print(f"  payload: {p.payload}")

    # 模拟用户拒绝
    print("\n[User Decision] 拒绝")
    try:
        result = compiled.resume(
            "email-1",
            value={"approved": False, "reason": "时间不合适"},
        )
        print(f"  最终结果: sent={result.get('sent')}")
        for m in result.get("messages", []):
            if m.get("role") == "system":
                print(f"    {m['content']}")
    except GraphPaused:
        print("  ⚠️ 又被暂停了（不应发生）")

    # ============== 第二个 thread 演示批准路径 ==============
    print("\n" + "=" * 60)
    print("[Round 2] 新 thread —— 用户批准")
    print("=" * 60)
    try:
        compiled.invoke({"messages": []}, config=RunConfig(thread_id="email-2"))
    except GraphPaused as p:
        print(f"  📨 暂停 thread={p.thread_id}")

    result = compiled.resume("email-2", value={"approved": True})
    print(f"  最终结果: sent={result.get('sent')}")
    for m in result.get("messages", []):
        if m.get("role") == "system":
            print(f"    {m['content']}")

    # ============== Checkpoint 历史 ==============
    print("\n" + "=" * 60)
    print("Checkpoint 历史（time travel）")
    print("=" * 60)
    for ck_record in ck.list("email-1"):
        print(f"  ckpt_id={ck_record.id[:8]}… next={ck_record.next_nodes}")

    print("\n✅ HITL demo 完成")


if __name__ == "__main__":
    main()
