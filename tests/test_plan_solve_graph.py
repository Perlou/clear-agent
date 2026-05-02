"""Plan-Solve Agent StateGraph builder 测试

验证：
- _parse_plan 在 JSON / 包裹 JSON / 兜底按行 三种格式下都能解析
- plan → execute (循环) → finalize 流程正确，step_results 顺序累积
- 空 plan 时 _route_after_plan 直接走 finalize
- _route_after_execute 在每步后正确路由
- PlanSolveAgent.as_graph() 等价
- checkpointer 集成
- 旧 PlanSolveAgent / PlanAndSolveAgent 向后兼容
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from clear_agent.agents import build_plan_solve_graph
from clear_agent.agents._plan_solve_graph import (
    PlanSolveGraphState,
    _parse_plan,
    _route_after_execute,
    _route_after_plan,
)
from clear_agent.core.checkpoint import InMemoryCheckpointer
from clear_agent.core.graph import END, RunConfig
from clear_agent.core.llm_response import LLMResponse


# ==================== Mock LLM ====================


class _MockLLM:
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


# ==================== Section A: _parse_plan 单元测试 ====================


def test_parse_plan_pure_json_array():
    out = _parse_plan('["a", "b", "c"]')
    assert out == ["a", "b", "c"]


def test_parse_plan_json_with_text_around():
    """LLM 常常在 JSON 前后加解释 → 仍能抽出"""
    out = _parse_plan('Sure, here is the plan:\n["step1", "step2"]\nDone.')
    assert out == ["step1", "step2"]


def test_parse_plan_json_with_trailing_comma_falls_back_to_lines():
    """带 trailing comma 的非法 JSON → 走兜底按行解析"""
    out = _parse_plan("1. first\n2. second\n3. third")
    assert out == ["first", "second", "third"]


def test_parse_plan_markdown_dash_list():
    """Markdown 列表（- / *）也能兜底"""
    out = _parse_plan("- alpha\n- beta\n- gamma")
    assert out == ["alpha", "beta", "gamma"]


def test_parse_plan_empty_input():
    assert _parse_plan("") == []


def test_parse_plan_skips_blank_lines():
    out = _parse_plan("first\n\n\nsecond")
    assert out == ["first", "second"]


def test_parse_plan_non_string_elements_coerced_to_str():
    """JSON 数组里混入非字符串 → 走兜底（split lines），不会崩"""
    out = _parse_plan('[1, 2, 3]')
    # 第一次直接 parse 会得到 [1,2,3]，但 all(isinstance(x, str)) 失败
    # 第二次走 re 抽取 → 同样 list[int] → str 化
    assert out == ["1", "2", "3"]


# ==================== Section B: 路由器单元测试 ====================


def test_route_after_plan_with_steps_goes_execute():
    state: PlanSolveGraphState = {"plan": ["a", "b"]}
    assert _route_after_plan(state) == "execute"


def test_route_after_plan_empty_goes_finalize():
    state: PlanSolveGraphState = {"plan": []}
    assert _route_after_plan(state) == "finalize"


def test_route_after_plan_missing_goes_finalize():
    state: PlanSolveGraphState = {}
    assert _route_after_plan(state) == "finalize"


def test_route_after_execute_more_steps_loops_back():
    state: PlanSolveGraphState = {"plan": ["a", "b", "c"], "current_step_idx": 1}
    assert _route_after_execute(state) == "execute"


def test_route_after_execute_done_goes_finalize():
    state: PlanSolveGraphState = {"plan": ["a", "b"], "current_step_idx": 2}
    assert _route_after_execute(state) == "finalize"


# ==================== Section C: 端到端流程 ====================


def test_plan_solve_graph_full_flow_3_steps():
    """plan → execute × 3 → finalize；step_results 顺序累积"""
    llm = _MockLLM(
        [
            _resp('["step A", "step B", "step C"]', total_tokens=10),
            _resp("result-A", total_tokens=11),
            _resp("result-B", total_tokens=12),
            _resp("result-C", total_tokens=13),
            _resp("FINAL ANSWER", total_tokens=14),
        ]
    )
    compiled = build_plan_solve_graph(llm)

    result = compiled.invoke({"question": "Solve X"})

    assert result["plan"] == ["step A", "step B", "step C"]
    assert result["step_results"] == ["result-A", "result-B", "result-C"]
    assert result["current_step_idx"] == 3
    assert result["final_answer"] == "FINAL ANSWER"
    # token 累计 = 10+11+12+13+14
    assert result["total_tokens"] == 60
    # 5 次 LLM 调用：1 plan + 3 execute + 1 finalize
    assert len(llm.calls) == 5


def test_plan_solve_graph_empty_plan_skips_execute():
    """LLM 给空数组 → 直接 finalize，不调用 execute"""
    llm = _MockLLM(
        [
            _resp("[]"),
            _resp("nothing to do"),
        ]
    )
    compiled = build_plan_solve_graph(llm)

    result = compiled.invoke({"question": "trivial"})

    assert result["plan"] == []
    assert result["final_answer"] == "nothing to do"
    # 无 execute 节点被调用 → 总共只有 plan + finalize 两次
    assert len(llm.calls) == 2


def test_plan_solve_graph_single_step_plan():
    """只有 1 步：plan → execute × 1 → finalize"""
    llm = _MockLLM(
        [
            _resp('["only step"]'),
            _resp("only result"),
            _resp("the answer"),
        ]
    )
    compiled = build_plan_solve_graph(llm)

    result = compiled.invoke({"question": "single"})

    assert result["plan"] == ["only step"]
    assert result["step_results"] == ["only result"]
    assert result["final_answer"] == "the answer"
    assert len(llm.calls) == 3


def test_plan_solve_graph_execute_prompt_has_context():
    """每一步 execute prompt 应包含问题、完整计划、上一步结果、当前步骤"""
    llm = _MockLLM(
        [
            _resp('["X", "Y"]'),
            _resp("R1"),
            _resp("R2"),
            _resp("FINAL"),
        ]
    )
    compiled = build_plan_solve_graph(llm)

    compiled.invoke({"question": "Q-MARKER"})

    # llm.calls[0] = plan, [1] = execute step 1, [2] = execute step 2, [3] = finalize
    exec1 = llm.calls[1][0]["content"]
    exec2 = llm.calls[2][0]["content"]
    finalize_prompt = llm.calls[3][0]["content"]

    # execute prompt 包含问题文本
    assert "Q-MARKER" in exec1
    assert "Q-MARKER" in exec2
    # 第二步应能看到第一步结果
    assert "R1" in exec2
    # finalize 应汇总所有步骤
    assert "R1" in finalize_prompt
    assert "R2" in finalize_prompt


# ==================== Section D: checkpointer 集成 ====================


def test_plan_solve_graph_checkpointer_writes_per_node():
    """plan + 2*execute + finalize = 4 个 ckpt（实际可能因路由更多）"""
    llm = _MockLLM(
        [
            _resp('["a", "b"]'),
            _resp("ra"),
            _resp("rb"),
            _resp("F"),
        ]
    )
    ck = InMemoryCheckpointer()
    compiled = build_plan_solve_graph(llm, checkpointer=ck)

    compiled.invoke(
        {"question": "Q"}, config=RunConfig(thread_id="thread-ps")
    )

    ckpts = ck.list("thread-ps")
    assert len(ckpts) >= 4
    assert ckpts[0].next_nodes == [END]


# ==================== Section E: as_graph() 等价 ====================


def test_plan_solve_agent_as_graph_equivalent():
    """PlanSolveAgent.as_graph() 与 build_plan_solve_graph 直接调用产物等价"""
    from clear_agent.agents import PlanSolveAgent

    llm = _MockLLM(
        [
            _resp('["s1"]'),
            _resp("r1"),
            _resp("done"),
        ]
    )
    agent = PlanSolveAgent(name="test", llm=llm)
    compiled = agent.as_graph()

    result = compiled.invoke({"question": "Q"})

    assert result["final_answer"] == "done"


# ==================== Section F: 旧 API 向后兼容 ====================


def test_legacy_plan_solve_agent_construction_intact():
    from clear_agent.agents import PlanSolveAgent

    llm = _MockLLM([])
    agent = PlanSolveAgent(name="legacy", llm=llm)

    assert hasattr(agent, "run")
    assert hasattr(agent, "arun")
    assert hasattr(agent, "as_graph")
    assert hasattr(agent, "planner")
    assert hasattr(agent, "executor")


def test_plan_and_solve_agent_alias_intact():
    """PlanAndSolveAgent 是 PlanSolveAgent 的向后兼容别名"""
    from clear_agent.agents import PlanAndSolveAgent, PlanSolveAgent

    assert PlanAndSolveAgent is PlanSolveAgent


def test_top_level_plan_solve_graph_imports_intact():
    from clear_agent.agents import build_plan_solve_graph, PlanSolveGraphState

    assert callable(build_plan_solve_graph)
    assert PlanSolveGraphState is not None
