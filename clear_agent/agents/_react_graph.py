"""ReAct Agent 的 StateGraph 构建器

把 ReActAgent._run_impl 的循环逻辑拆解为 graph 节点，便于：
- 检查点 / resume
- HITL 中断（W3 接入）
- 时间旅行
- 与其他 graph 组合

设计：
- 节点：llm_node（推理）→ tool_executor_node（执行工具）→ 回到 llm_node
- 路由：has_finish_or_no_tools → END，否则 → tool_executor_node
- State 字段使用 Annotated reducer：messages 追加去重，total_tokens 累加

向后兼容：
- 旧 ReActAgent.run() / arun() 完全不动
- 此模块作为额外的「构建器」，老用户无感
"""

from __future__ import annotations

import json
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

from ..core.graph import (
    END,
    START,
    CompiledGraph,
    StateGraph,
    add_messages,
    append_list,
)
from ..core.checkpoint import BaseCheckpointer
from ..core.config import Config

if TYPE_CHECKING:
    from ..core.llm import ClearAgentLLM
    from ..tools.registry import ToolRegistry


# ==================== State Schema ====================


class ReActGraphState(TypedDict, total=False):
    """ReAct graph 执行状态

    字段语义：
        messages: OpenAI 兼容的 messages 列表（含 system/user/assistant/tool）
        tool_calls_pending: LLM 最近一次返回的待执行 tool_calls
        total_tokens: 累计 token 消耗
        steps: 累计 LLM 调用步数
        max_steps: 单次执行允许的最大 LLM 步数
        final_answer: 终止时的最终答案
        thoughts: Thought 工具记录的推理（可观测性）
    """

    messages: Annotated[List[Dict[str, Any]], add_messages]
    tool_calls_pending: List[Any]
    total_tokens: int
    steps: int
    max_steps: int
    final_answer: Optional[str]
    thoughts: Annotated[List[str], append_list]


# ==================== 内置工具 schema（独立，避免与 ReActAgent 实例耦合） ====================


_THOUGHT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "Thought",
        "description": "分析问题，制定策略，记录推理过程。在需要思考时调用此工具。",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "你的推理过程和分析"},
            },
            "required": ["reasoning"],
        },
    },
}

_FINISH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "Finish",
        "description": "当你有足够信息得出结论时，使用此工具返回最终答案。",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "最终答案"},
            },
            "required": ["answer"],
        },
    },
}

BUILTIN_TOOL_NAMES = {"Thought", "Finish"}


def _build_tool_schemas(registry: Optional["ToolRegistry"]) -> List[Dict[str, Any]]:
    """构建给 LLM 的 tool schemas（含内置 + 用户工具）"""
    schemas: List[Dict[str, Any]] = [_THOUGHT_SCHEMA, _FINISH_SCHEMA]
    if registry is not None:
        # ToolRegistry.list_tools() 返回 list[str]（名字），用 get_all_tools() 拿 Tool 对象
        get_all = getattr(registry, "get_all_tools", None)
        if callable(get_all):
            for tool in get_all():
                schemas.append(tool.to_openai_schema())
        else:
            # fallback: 兼容自定义 registry
            for name in registry.list_tools():
                tool = registry.get_tool(name) if hasattr(registry, "get_tool") else None
                if tool is not None:
                    schemas.append(tool.to_openai_schema())
    return schemas


# ==================== 节点工厂 ====================


def _make_llm_node(
    llm: "ClearAgentLLM", tool_schemas: List[Dict[str, Any]]
) -> Callable[[ReActGraphState], Dict[str, Any]]:
    """LLM 推理节点

    职责：
    - 调用 llm.invoke_with_tools（同步）
    - 解析 tool_calls
    - 累加 tokens / 增加 step
    - 把 assistant message 写入 messages
    """

    def llm_node(state: ReActGraphState) -> Dict[str, Any]:
        messages = list(state.get("messages") or [])
        response = llm.invoke_with_tools(
            messages=messages, tools=tool_schemas, tool_choice="auto"
        )

        tool_calls = response.tool_calls or []

        # 优先用 adapter 序列化（自动处理 reasoning_content 回写策略）；
        # 缺失该方法时退回 OpenAI 原生格式（向后兼容外部自定义 LLM / mock）
        if hasattr(llm, "serialize_assistant_message"):
            assistant_msg: Dict[str, Any] = llm.serialize_assistant_message(response)
        else:
            assistant_msg = {
                "role": "assistant",
                "content": response.content,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in tool_calls
                ]

        delta_tokens = 0
        if response.usage:
            delta_tokens = response.usage.get("total_tokens", 0) or 0

        update: Dict[str, Any] = {
            "messages": [assistant_msg],
            "tool_calls_pending": tool_calls,
            "total_tokens": (state.get("total_tokens") or 0) + delta_tokens,
            "steps": (state.get("steps") or 0) + 1,
        }

        # 没有 tool_calls：直接把 content 作为 final_answer
        if not tool_calls:
            update["final_answer"] = response.content or ""

        return update

    return llm_node


def _make_tool_executor_node(
    registry: Optional["ToolRegistry"],
) -> Callable[[ReActGraphState], Dict[str, Any]]:
    """工具执行节点

    职责：
    - 串行执行 tool_calls_pending（W2 范围；并行执行属 P1）
    - Thought：仅记录到 thoughts，不真调
    - Finish：写 final_answer，下一轮路由直接到 END
    - 用户工具：通过 registry 执行，返回 ToolResponse 文本
    """

    def tool_executor_node(state: ReActGraphState) -> Dict[str, Any]:
        tool_calls = state.get("tool_calls_pending") or []
        new_messages: List[Dict[str, Any]] = []
        thoughts: List[str] = []
        final_answer: Optional[str] = None

        for tc in tool_calls:
            tool_name = tc.name
            tool_call_id = tc.id
            try:
                args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
            except json.JSONDecodeError as e:
                new_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"错误：参数格式不正确 - {e}",
                    }
                )
                continue

            if tool_name == "Thought":
                reasoning = args.get("reasoning", "")
                thoughts.append(reasoning)
                new_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"推理: {reasoning}",
                    }
                )
                continue

            if tool_name == "Finish":
                answer = args.get("answer", "")
                final_answer = answer
                new_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"最终答案: {answer}",
                    }
                )
                continue

            # 用户工具：交给 registry
            if registry is None:
                new_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"错误：工具 {tool_name} 未注册（registry 为空）",
                    }
                )
                continue

            tool = registry.get_tool(tool_name) if hasattr(registry, "get_tool") else None
            if tool is None:
                new_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"错误：工具 {tool_name} 未注册",
                    }
                )
                continue

            try:
                response = tool.run_with_timing(args)
                content = response.text if hasattr(response, "text") else str(response)
            except Exception as e:
                content = f"工具执行失败: {e}"

            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                }
            )

        update: Dict[str, Any] = {
            "messages": new_messages,
            "tool_calls_pending": [],
        }
        if thoughts:
            update["thoughts"] = thoughts
        if final_answer is not None:
            update["final_answer"] = final_answer
        return update

    return tool_executor_node


# ==================== 路由 ====================


def _router_after_llm(state: ReActGraphState) -> str:
    """LLM 节点后的路由

    规则:
    - 有 final_answer（无 tool_calls 时由 llm_node 设置）→ END
    - 步数达到 max_steps → END
    - 有待执行的 tool_calls → tool_executor
    - 否则 → END（兜底）
    """
    if state.get("final_answer") is not None:
        return "end"
    max_steps = state.get("max_steps") or 5
    if (state.get("steps") or 0) >= max_steps:
        return "end"
    if state.get("tool_calls_pending"):
        return "tools"
    return "end"


def _router_after_tools(state: ReActGraphState) -> str:
    """工具节点后的路由

    规则:
    - 有 final_answer（Finish 工具触发）→ END
    - 步数达到 max_steps → END（兜底）
    - 否则 → llm 继续推理
    """
    if state.get("final_answer") is not None:
        return "end"
    max_steps = state.get("max_steps") or 5
    if (state.get("steps") or 0) >= max_steps:
        return "end"
    return "llm"


# 向后兼容别名（旧测试可能引用）
_router = _router_after_llm


# ==================== 公开 API ====================


def build_react_graph(
    llm: "ClearAgentLLM",
    tool_registry: Optional["ToolRegistry"] = None,
    config: Optional[Config] = None,
    checkpointer: Optional[BaseCheckpointer] = None,
) -> CompiledGraph[ReActGraphState]:
    """构建 ReAct StateGraph 并编译

    与 ReActAgent.run() 行为等价（同步版本），但获得：
    - per-node 自动 checkpoint
    - resume / time-travel
    - 流式事件（compiled.stream）
    - HITL 接入点（W3）

    用法:
        from clear_agent.agents import build_react_graph
        from clear_agent.core.checkpoint import InMemoryCheckpointer

        compiled = build_react_graph(llm, registry, checkpointer=InMemoryCheckpointer())
        result = compiled.invoke({
            "messages": [{"role": "user", "content": "hello"}],
            "max_steps": 5,
        })
        print(result["final_answer"])

    Args:
        llm: LLM 实例
        tool_registry: 工具注册表（可选）
        config: 配置（暂未读入；预留扩展）
        checkpointer: 持久化（可选）

    Returns:
        编译后的 CompiledGraph
    """
    tool_schemas = _build_tool_schemas(tool_registry)
    llm_node = _make_llm_node(llm, tool_schemas)
    tool_executor_node = _make_tool_executor_node(tool_registry)

    g: StateGraph[ReActGraphState] = StateGraph(ReActGraphState)
    g.add_node("llm", llm_node)
    g.add_node("tools", tool_executor_node)
    g.add_edge(START, "llm")
    g.add_conditional_edges(
        "llm", _router_after_llm, {"tools": "tools", "end": END}
    )
    g.add_conditional_edges(
        "tools", _router_after_tools, {"llm": "llm", "end": END}
    )

    return g.compile(checkpointer=checkpointer)


__all__ = [
    "ReActGraphState",
    "build_react_graph",
    "BUILTIN_TOOL_NAMES",
]
