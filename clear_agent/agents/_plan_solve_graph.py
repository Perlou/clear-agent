"""Plan-Solve Agent 的 StateGraph 构建器

经典「先规划再执行」流程：
- plan: LLM 把任务拆解为有序步骤列表
- execute: 循环执行每步（每次执行一步并 advance 索引）
- finalize: 汇总所有步骤结果，生成最终答案

用条件边实现内部循环，无需重新发明 while。
"""

from __future__ import annotations

import json
import re
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
    TypedDict,
)

from ..core.checkpoint import BaseCheckpointer
from ..core.config import Config
from ..core.graph import (
    END,
    START,
    CompiledGraph,
    StateGraph,
    append_list,
)

if TYPE_CHECKING:
    from ..core.llm import ClearAgentLLM


class PlanSolveGraphState(TypedDict, total=False):
    """Plan-Solve 状态

    Attributes:
        question: 用户原问题
        plan: 步骤列表（字符串 list）
        current_step_idx: 当前执行到第几步（0-based）
        step_results: 每步执行结果列表（与 plan 一一对应）
        final_answer: finalize 节点产出
        total_tokens: 累计 token
    """

    question: str
    plan: List[str]
    current_step_idx: int
    step_results: Annotated[List[str], append_list]
    final_answer: Optional[str]
    total_tokens: int


_PLAN_PROMPT = """你是一个任务规划专家。请把以下问题拆解为 2-5 个有序的可执行步骤。

要求：
- 每步应该是独立可执行的单元
- 用 JSON 数组返回，例如：["步骤1", "步骤2", "步骤3"]
- 不要解释，直接返回 JSON 数组

问题: {question}
"""

_EXECUTE_PROMPT = """你是一个步骤执行者。当前正在执行第 {idx}/{total} 步。

整体问题: {question}
完整计划: {plan}
之前步骤的结果: {prev_results}

当前步骤: {step}

请仅针对此步骤给出具体执行结果（简洁明了）：
"""

_FINALIZE_PROMPT = """你是一个总结专家。基于以下所有步骤的执行结果，给出对原问题的最终答案。

问题: {question}
计划:
{plan_with_results}

请综合给出最终答案：
"""


def _parse_plan(text: str) -> List[str]:
    """从 LLM 输出抽取 JSON 数组步骤列表

    宽容解析：先尝试整段 json.loads；失败则用正则抽取 JSON 数组。
    """
    text = (text or "").strip()
    # 直接 parse
    try:
        result = json.loads(text)
        if isinstance(result, list) and all(isinstance(x, str) for x in result):
            return result
    except json.JSONDecodeError:
        pass

    # 抽取第一个 [...]
    m = re.search(r"\[.*?\]", text, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, list):
                return [str(x) for x in result]
        except json.JSONDecodeError:
            pass

    # 兜底：按行拆，过滤空行和 markdown 序号
    lines = [
        re.sub(r"^[\d\.\-\*\s]+", "", ln).strip()
        for ln in text.splitlines()
        if ln.strip()
    ]
    return [ln for ln in lines if ln]


def _make_plan_node(
    llm: "ClearAgentLLM",
) -> Callable[[PlanSolveGraphState], Dict[str, Any]]:
    def plan_node(state: PlanSolveGraphState) -> Dict[str, Any]:
        prompt = _PLAN_PROMPT.format(question=state.get("question", ""))
        response = llm.invoke([{"role": "user", "content": prompt}])
        plan = _parse_plan(response.content or "")
        delta = (response.usage or {}).get("total_tokens", 0) or 0
        return {
            "plan": plan,
            "current_step_idx": 0,
            "total_tokens": (state.get("total_tokens") or 0) + delta,
        }

    return plan_node


def _make_execute_node(
    llm: "ClearAgentLLM",
) -> Callable[[PlanSolveGraphState], Dict[str, Any]]:
    def execute_node(state: PlanSolveGraphState) -> Dict[str, Any]:
        plan = state.get("plan") or []
        idx = state.get("current_step_idx") or 0
        if idx >= len(plan):
            return {}  # 安全兜底（路由不应让我们到这）

        step = plan[idx]
        prev = state.get("step_results") or []
        prompt = _EXECUTE_PROMPT.format(
            idx=idx + 1,
            total=len(plan),
            question=state.get("question", ""),
            plan=json.dumps(plan, ensure_ascii=False),
            prev_results=json.dumps(prev, ensure_ascii=False),
            step=step,
        )
        response = llm.invoke([{"role": "user", "content": prompt}])
        delta = (response.usage or {}).get("total_tokens", 0) or 0
        return {
            "step_results": [response.content or ""],
            "current_step_idx": idx + 1,
            "total_tokens": (state.get("total_tokens") or 0) + delta,
        }

    return execute_node


def _make_finalize_node(
    llm: "ClearAgentLLM",
) -> Callable[[PlanSolveGraphState], Dict[str, Any]]:
    def finalize_node(state: PlanSolveGraphState) -> Dict[str, Any]:
        plan = state.get("plan") or []
        results = state.get("step_results") or []
        zipped = "\n".join(
            f"{i + 1}. {p}\n   结果: {r}"
            for i, (p, r) in enumerate(zip(plan, results))
        )
        prompt = _FINALIZE_PROMPT.format(
            question=state.get("question", ""),
            plan_with_results=zipped,
        )
        response = llm.invoke([{"role": "user", "content": prompt}])
        delta = (response.usage or {}).get("total_tokens", 0) or 0
        return {
            "final_answer": response.content,
            "total_tokens": (state.get("total_tokens") or 0) + delta,
        }

    return finalize_node


def _route_after_plan(state: PlanSolveGraphState) -> str:
    """plan 后：有步骤就执行，否则直接 finalize"""
    return "execute" if (state.get("plan") or []) else "finalize"


def _route_after_execute(state: PlanSolveGraphState) -> str:
    """execute 后：还有步骤继续执行，否则 finalize"""
    plan = state.get("plan") or []
    idx = state.get("current_step_idx") or 0
    return "execute" if idx < len(plan) else "finalize"


def build_plan_solve_graph(
    llm: "ClearAgentLLM",
    config: Optional[Config] = None,
    checkpointer: Optional[BaseCheckpointer] = None,
) -> CompiledGraph[PlanSolveGraphState]:
    """构建 Plan-Solve StateGraph 并编译"""
    g: StateGraph[PlanSolveGraphState] = StateGraph(PlanSolveGraphState)
    g.add_node("plan", _make_plan_node(llm))
    g.add_node("execute", _make_execute_node(llm))
    g.add_node("finalize", _make_finalize_node(llm))
    g.add_edge(START, "plan")
    g.add_conditional_edges(
        "plan", _route_after_plan, {"execute": "execute", "finalize": "finalize"}
    )
    g.add_conditional_edges(
        "execute", _route_after_execute, {"execute": "execute", "finalize": "finalize"}
    )
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


__all__ = ["PlanSolveGraphState", "build_plan_solve_graph", "_parse_plan"]
