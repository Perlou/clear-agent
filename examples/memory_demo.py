"""Memory + RAG demo（β-W4）

本 demo 不依赖外部 API：使用纯内存模拟 embedder + DocumentStore，演示：
1. WorkingMemory 短期会话记忆
2. SemanticMemory 长期向量+图谱记忆（mock store）
3. MemoryManager 多子系统协调
4. 在 graph 节点里集成 Memory 实现「带记忆的对话」

运行：
    python examples/memory_demo.py
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock

from clear_agent.memory import (
    MemoryConfig,
    MemoryItem,
    MemoryManager,
    SemanticMemory,
    WorkingMemory,
)


def _mock_embedder(dim: int = 384):
    """简易 embedder：把文本长度作为单维向量"""
    e = MagicMock()
    e.encode.side_effect = lambda text: [float(len(text))] * dim
    e.dimension = dim
    return e


def _mock_qdrant_store():
    s = MagicMock()
    s.add_vectors.return_value = True
    s.search_similar.return_value = []
    return s


# ==================================================================
# Part 1: WorkingMemory 短期会话记忆
# ==================================================================


def demo_working_memory() -> None:
    print("=" * 60)
    print("Part 1: WorkingMemory（短期会话记忆）")
    print("=" * 60)

    cfg = MemoryConfig(
        working_memory_capacity=5,
        working_memory_tokens=100,
        working_memory_ttl_minutes=60,
    )
    wm = WorkingMemory(cfg)

    # 模拟一段对话
    turns = [
        ("用户问怎么调试图执行卡死", 0.6),
        ("助手回答：用 trace_logger 加 step 级日志", 0.5),
        ("用户接着问 HITL 怎么用", 0.7),
        ("助手回答：interrupt() + resume(value=...)", 0.8),
        ("用户感谢", 0.2),
    ]
    for i, (text, imp) in enumerate(turns):
        wm.add(
            MemoryItem(
                id=f"turn-{i}",
                content=text,
                memory_type="working",
                user_id="alice",
                timestamp=datetime.now(),
                importance=imp,
            )
        )

    print(f"已加入 {len(turns)} 轮对话")
    print()

    # 检索相关
    print('检索 "HITL"：')
    for m in wm.retrieve("HITL", limit=3):
        print(f"  [{m.importance:.1f}] {m.content}")

    print()
    print("上下文摘要（按重要性 / 时间倒序）：")
    print(wm.get_context_summary(max_length=200))

    print()
    print("统计：")
    s = wm.get_stats()
    print(f"  count={s['count']} tokens={s['current_tokens']}/{s['max_tokens']}")
    print(f"  avg_importance={s['avg_importance']:.2f}")

    # 遗忘低重要性
    print()
    n = wm.forget(strategy="importance_based", threshold=0.3)
    print(f"按重要性遗忘 → 清理了 {n} 条")
    print(f"剩余 count={wm.get_stats()['count']}")


# ==================================================================
# Part 2: SemanticMemory 长期向量+图谱记忆（mock）
# ==================================================================


def demo_semantic_memory() -> None:
    print()
    print("=" * 60)
    print("Part 2: SemanticMemory（长期向量+图谱，mock embedder/store）")
    print("=" * 60)

    sm = SemanticMemory(
        config=MemoryConfig(),
        embedding_model=_mock_embedder(),
        vector_store=_mock_qdrant_store(),
        nlp=None,  # fallback：单词当实体
    )

    # 加入若干语义记忆
    facts = [
        "Alice works at OpenAI as a researcher",
        "Bob works at OpenAI as an engineer",
        "Alice and Bob collaborate on Codex project",
    ]
    for i, fact in enumerate(facts):
        sm.add(
            MemoryItem(
                id=f"fact-{i}",
                content=fact,
                memory_type="semantic",
                user_id="default",
                timestamp=datetime.now(),
                importance=0.7,
            )
        )

    print(f"已加入 {len(facts)} 条事实")
    print()

    # 内存图谱状态
    print("内存图谱：")
    stats = sm.get_stats()
    print(f"  实体数: {stats['entities_count']}")
    print(f"  关系数: {stats['relations_count']}")

    print()
    print("查询 'Alice'：")
    matches = sm.search_entities("Alice", limit=5)
    for e in matches:
        print(f"  实体: {e.name} (类型: {e.entity_type}, 频率: {e.frequency})")

    # 取一个实体的相关实体（CO_OCCURS）
    if matches:
        first_id = matches[0].entity_id
        related = sm.get_related_entities(first_id, max_hops=1)
        print(f"\n与 '{matches[0].name}' 共现的实体：")
        for r in related[:5]:
            print(
                f"  - {r['entity'].name} via {r['relation_type']} "
                f"(strength={r['strength']:.2f}, distance={r['distance']})"
            )

    # 导出图谱
    graph = sm.export_knowledge_graph()
    print()
    print(f"图谱导出: {graph['graph_stats']}")


# ==================================================================
# Part 3: MemoryManager 多子系统协调
# ==================================================================


def demo_memory_manager() -> None:
    print()
    print("=" * 60)
    print("Part 3: MemoryManager（多子系统协调）")
    print("=" * 60)

    mgr = MemoryManager()
    mgr.register("working", WorkingMemory(MemoryConfig()))
    mgr.register(
        "semantic",
        SemanticMemory(
            config=MemoryConfig(),
            embedding_model=_mock_embedder(),
            vector_store=_mock_qdrant_store(),
        ),
    )

    print(f"已注册子系统: {mgr.types()}")

    # 按 type 自动路由
    mgr.add(
        MemoryItem(
            id="w1", content="临时记一下：用户想要 RAG demo",
            memory_type="working", user_id="alice",
            timestamp=datetime.now(), importance=0.5,
        )
    )
    mgr.add(
        MemoryItem(
            id="s1", content="alpha beta gamma 是项目里三个核心模块",
            memory_type="semantic", user_id="alice",
            timestamp=datetime.now(), importance=0.9,
        )
    )

    print()
    print("跨子系统检索 'alpha'：")
    for m in mgr.retrieve("alpha", limit=10):
        print(f"  [{m.memory_type}] importance={m.importance:.2f}: {m.content[:60]}")

    print()
    print("聚合统计：")
    s = mgr.get_stats()
    print(f"  total={s['total_count']}")
    for mt, st in s["by_type"].items():
        print(f"    {mt}: count={st.get('count', 0)}")


# ==================================================================
# Part 4: graph 节点里集成 Memory
# ==================================================================


def demo_memory_aware_graph() -> None:
    print()
    print("=" * 60)
    print("Part 4: 在 StateGraph 节点里集成 Memory")
    print("=" * 60)

    from typing import Annotated, TypedDict
    from clear_agent.core.graph import StateGraph, START, END, add_messages

    mgr = MemoryManager()
    mgr.register("working", WorkingMemory(MemoryConfig()))

    class State(TypedDict, total=False):
        question: str
        answer: str
        history_count: int

    def retrieve_node(state: State) -> Dict[str, Any]:
        """从 working memory 召回相关历史"""
        hits = mgr.retrieve(state["question"], limit=3)
        return {"history_count": len(hits)}

    def respond_node(state: State) -> Dict[str, Any]:
        """生成回答（这里写死，真实场景调 LLM）+ 把这一轮存进 memory"""
        answer = f"针对「{state['question']}」的答复（参考了 {state['history_count']} 条历史）"
        mgr.add(
            MemoryItem(
                id=f"turn-{datetime.now().isoformat()}",
                content=f"Q: {state['question']}\nA: {answer}",
                memory_type="working",
                user_id="alice",
                timestamp=datetime.now(),
                importance=0.6,
            )
        )
        return {"answer": answer}

    g: StateGraph[State] = StateGraph(State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("respond", respond_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "respond")
    g.add_edge("respond", END)
    compiled = g.compile()

    # 跑三轮
    for q in [
        "如何调试 graph 卡死？",
        "上一轮提到的 trace_logger 怎么用？",
        "HITL 是什么？",
    ]:
        result = compiled.invoke({"question": q})
        print(f"Q: {q}")
        print(f"  历史命中: {result['history_count']}, A: {result['answer']}")

    print()
    print(f"Memory 最终状态: {mgr.get_stats()['total_count']} 条")


def main() -> None:
    demo_working_memory()
    demo_semantic_memory()
    demo_memory_manager()
    demo_memory_aware_graph()
    print()
    print("✅ memory_demo 跑通 — β-W4 Memory + RAG 集成验证通过")


if __name__ == "__main__":
    main()
