"""Reflection Agent 的 StateGraph 构建器

经典 generate → reflect → revise → END 三相流程：
- generate: LLM 生成初版回答
- reflect: LLM 审视初版，给出改进建议
- revise: LLM 根据反思修订
- 一次反思即终止（多轮反思留作 P1 扩展）
"""

from __future__ import annotations

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
    add_messages,
    append_list,
)

if TYPE_CHECKING:
    from ..core.llm import ClearAgentLLM


class ReflectionGraphState(TypedDict, total=False):
    """Reflection 状态

    Attributes:
        messages: 完整对话历史
        question: 用户原问题
        draft: generate 阶段产出
        critique: reflect 阶段产出
        final_answer: revise 阶段产出
        history: 各阶段产出的有序快照（可观测性）
    """

    messages: Annotated[List[Dict[str, Any]], add_messages]
    question: str
    draft: Optional[str]
    critique: Optional[str]
    final_answer: Optional[str]
    history: Annotated[List[Dict[str, str]], append_list]
    total_tokens: int


_GENERATE_PROMPT = """你是一个反思型专家的「生成阶段」。
请对以下问题给出初步答案。要求清晰、有结构。

问题: {question}
"""

_REFLECT_PROMPT = """你是一个反思型专家的「反思阶段」。
针对刚才的初步答案，请指出 3-5 个可改进的点，包括但不限于：
- 准确性
- 完整性
- 论证质量
- 表达清晰度

问题: {question}
初步答案:
{draft}
"""

_REVISE_PROMPT = """你是一个反思型专家的「修订阶段」。
基于反思要点，重新输出一份改进后的最终答案。直接给出最终答案本身，不要解释修改细节。

问题: {question}
初步答案:
{draft}
反思要点:
{critique}
"""


def _make_generate_node(
    llm: "ClearAgentLLM",
) -> Callable[[ReflectionGraphState], Dict[str, Any]]:
    def generate_node(state: ReflectionGraphState) -> Dict[str, Any]:
        question = state.get("question", "")
        prompt = _GENERATE_PROMPT.format(question=question)
        response = llm.invoke([{"role": "user", "content": prompt}])
        delta = (response.usage or {}).get("total_tokens", 0) or 0
        return {
            "draft": response.content,
            "messages": [
                {"role": "assistant", "content": f"[draft] {response.content}"}
            ],
            "history": [{"phase": "generate", "content": response.content}],
            "total_tokens": (state.get("total_tokens") or 0) + delta,
        }

    return generate_node


def _make_reflect_node(
    llm: "ClearAgentLLM",
) -> Callable[[ReflectionGraphState], Dict[str, Any]]:
    def reflect_node(state: ReflectionGraphState) -> Dict[str, Any]:
        prompt = _REFLECT_PROMPT.format(
            question=state.get("question", ""),
            draft=state.get("draft", ""),
        )
        response = llm.invoke([{"role": "user", "content": prompt}])
        delta = (response.usage or {}).get("total_tokens", 0) or 0
        return {
            "critique": response.content,
            "messages": [
                {"role": "assistant", "content": f"[critique] {response.content}"}
            ],
            "history": [{"phase": "reflect", "content": response.content}],
            "total_tokens": (state.get("total_tokens") or 0) + delta,
        }

    return reflect_node


def _make_revise_node(
    llm: "ClearAgentLLM",
) -> Callable[[ReflectionGraphState], Dict[str, Any]]:
    def revise_node(state: ReflectionGraphState) -> Dict[str, Any]:
        prompt = _REVISE_PROMPT.format(
            question=state.get("question", ""),
            draft=state.get("draft", ""),
            critique=state.get("critique", ""),
        )
        response = llm.invoke([{"role": "user", "content": prompt}])
        delta = (response.usage or {}).get("total_tokens", 0) or 0
        return {
            "final_answer": response.content,
            "messages": [
                {"role": "assistant", "content": response.content}
            ],
            "history": [{"phase": "revise", "content": response.content}],
            "total_tokens": (state.get("total_tokens") or 0) + delta,
        }

    return revise_node


def build_reflection_graph(
    llm: "ClearAgentLLM",
    config: Optional[Config] = None,
    checkpointer: Optional[BaseCheckpointer] = None,
) -> CompiledGraph[ReflectionGraphState]:
    """构建 Reflection StateGraph 并编译

    流程: START → generate → reflect → revise → END
    """
    g: StateGraph[ReflectionGraphState] = StateGraph(ReflectionGraphState)
    g.add_node("generate", _make_generate_node(llm))
    g.add_node("reflect", _make_reflect_node(llm))
    g.add_node("revise", _make_revise_node(llm))
    g.add_edge(START, "generate")
    g.add_edge("generate", "reflect")
    g.add_edge("reflect", "revise")
    g.add_edge("revise", END)
    return g.compile(checkpointer=checkpointer)


__all__ = ["ReflectionGraphState", "build_reflection_graph"]
