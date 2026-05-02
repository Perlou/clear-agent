"""Reflection Agent StateGraph builder 测试

验证：
- generate → reflect → revise 顺序执行
- final_answer 来自 revise 阶段
- history 累积三阶段的产物
- total_tokens 累计
- ReflectionAgent.as_graph() 等价
- checkpointer 集成
- 旧 ReflectionAgent API 向后兼容
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from clear_agent.agents import build_reflection_graph
from clear_agent.agents._reflection_graph import ReflectionGraphState
from clear_agent.core.checkpoint import InMemoryCheckpointer
from clear_agent.core.graph import END, RunConfig
from clear_agent.core.llm_response import LLMResponse


# ==================== Mock LLM ====================


class _MockLLM:
    """按预设脚本依次返回 LLMResponse"""

    def __init__(self, responses: List[LLMResponse]):
        self._responses = list(responses)
        self.calls: List[List[Dict[str, Any]]] = []
        self.model = "mock-model"

    def invoke(self, messages, **kwargs):
        self.calls.append(list(messages))
        if not self._responses:
            raise AssertionError("LLM.invoke 调用次数超过预设")
        return self._responses.pop(0)


def _resp(content: str, total_tokens: int = 5) -> LLMResponse:
    return LLMResponse(
        content=content, model="mock-model", usage={"total_tokens": total_tokens}
    )


# ==================== Test 1: 三阶段顺序执行 ====================


def test_reflection_graph_three_phases_in_order():
    """generate → reflect → revise 严格按序，且 final_answer 来自 revise"""
    llm = _MockLLM(
        [
            _resp("[draft answer]", total_tokens=10),
            _resp("[critique points 1,2,3]", total_tokens=20),
            _resp("[final revised answer]", total_tokens=30),
        ]
    )
    compiled = build_reflection_graph(llm)

    result = compiled.invoke({"question": "What is 2+2?"})

    assert result["draft"] == "[draft answer]"
    assert result["critique"] == "[critique points 1,2,3]"
    assert result["final_answer"] == "[final revised answer]"
    # 总 token = 10+20+30
    assert result["total_tokens"] == 60
    # 三次 LLM 调用
    assert len(llm.calls) == 3


# ==================== Test 2: history 字段累积三阶段产物 ====================


def test_reflection_graph_history_accumulates():
    """history 列表按顺序记录 generate/reflect/revise 三阶段"""
    llm = _MockLLM(
        [
            _resp("draft-x"),
            _resp("critique-x"),
            _resp("revised-x"),
        ]
    )
    compiled = build_reflection_graph(llm)

    result = compiled.invoke({"question": "Q?"})

    history = result.get("history") or []
    assert len(history) == 3
    assert history[0] == {"phase": "generate", "content": "draft-x"}
    assert history[1] == {"phase": "reflect", "content": "critique-x"}
    assert history[2] == {"phase": "revise", "content": "revised-x"}


# ==================== Test 3: messages 字段记录三阶段输出 ====================


def test_reflection_graph_messages_record_phase_outputs():
    """messages 中应有 [draft], [critique] 与 final_answer 三条 assistant 消息"""
    llm = _MockLLM(
        [
            _resp("D"),
            _resp("C"),
            _resp("R"),
        ]
    )
    compiled = build_reflection_graph(llm)

    result = compiled.invoke({"question": "Q"})

    msgs = [m for m in result.get("messages") or [] if m.get("role") == "assistant"]
    assert len(msgs) == 3
    assert msgs[0]["content"].startswith("[draft]")
    assert msgs[1]["content"].startswith("[critique]")
    assert msgs[2]["content"] == "R"  # revise 直接给最终答案


# ==================== Test 4: prompts 包含期待的字段插值 ====================


def test_reflection_graph_prompts_use_question_and_draft():
    """reflect prompt 应同时包含 question 和 draft；revise prompt 包含 critique"""
    llm = _MockLLM(
        [
            _resp("DRAFT-CONTENT"),
            _resp("CRITIQUE-CONTENT"),
            _resp("FINAL"),
        ]
    )
    compiled = build_reflection_graph(llm)

    compiled.invoke({"question": "Q-AAA"})

    # 第 1 次 (generate)
    gen_prompt = llm.calls[0][0]["content"]
    assert "Q-AAA" in gen_prompt
    # 第 2 次 (reflect)
    refl_prompt = llm.calls[1][0]["content"]
    assert "Q-AAA" in refl_prompt
    assert "DRAFT-CONTENT" in refl_prompt
    # 第 3 次 (revise)
    revise_prompt = llm.calls[2][0]["content"]
    assert "DRAFT-CONTENT" in revise_prompt
    assert "CRITIQUE-CONTENT" in revise_prompt


# ==================== Test 5: checkpointer 集成 ====================


def test_reflection_graph_checkpointer_writes_one_per_node():
    """三个节点 → 至少 3 个 ckpt；最后一个 next_node = END"""
    llm = _MockLLM([_resp("d"), _resp("c"), _resp("r")])
    ck = InMemoryCheckpointer()
    compiled = build_reflection_graph(llm, checkpointer=ck)

    compiled.invoke(
        {"question": "Q"}, config=RunConfig(thread_id="thread-refl")
    )

    ckpts = ck.list("thread-refl")
    assert len(ckpts) >= 3
    # ck.list 默认按时间倒序：最新的在 [0]
    assert ckpts[0].next_nodes == [END]


# ==================== Test 6: ReflectionAgent.as_graph() 等价 ====================


def test_reflection_agent_as_graph_equivalent():
    """ReflectionAgent.as_graph() 与 build_reflection_graph 直接调用产物等价"""
    from clear_agent.agents import ReflectionAgent

    llm = _MockLLM([_resp("d"), _resp("c"), _resp("r-final")])
    agent = ReflectionAgent(name="test", llm=llm)
    compiled = agent.as_graph()

    result = compiled.invoke({"question": "Q"})

    assert result["final_answer"] == "r-final"


# ==================== Test 7: 多 run 互不影响 ====================


def test_reflection_graph_independent_runs():
    """同一 compiled graph 跑两次 ↔ 两次互不污染"""
    llm = _MockLLM(
        [
            _resp("d1"),
            _resp("c1"),
            _resp("r1"),
            _resp("d2"),
            _resp("c2"),
            _resp("r2"),
        ]
    )
    compiled = build_reflection_graph(llm)

    r1 = compiled.invoke({"question": "Q1"})
    r2 = compiled.invoke({"question": "Q2"})

    assert r1["final_answer"] == "r1"
    assert r2["final_answer"] == "r2"
    # history 不会跨 run 拼接
    assert len(r1.get("history") or []) == 3
    assert len(r2.get("history") or []) == 3


# ==================== Test 8: 旧 ReflectionAgent API 向后兼容 ====================


def test_legacy_reflection_agent_construction_intact():
    from clear_agent.agents import ReflectionAgent

    llm = _MockLLM([])
    agent = ReflectionAgent(name="legacy", llm=llm, max_iterations=5)

    assert agent.max_iterations == 5
    assert hasattr(agent, "run")
    assert hasattr(agent, "arun")
    assert hasattr(agent, "as_graph")


def test_top_level_reflection_graph_imports_intact():
    from clear_agent.agents import build_reflection_graph, ReflectionGraphState

    assert callable(build_reflection_graph)
    assert ReflectionGraphState is not None
