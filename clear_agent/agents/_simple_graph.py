"""Simple Agent 的 StateGraph 构建器

结构与 ReAct 类似但更简单：
- 无内置 Thought/Finish 工具
- LLM 直接给文本响应或调用用户工具
- 路由：has_tool_calls → tools → llm；no_tool_calls → END

适用场景：轻量对话 + 可选 Function Calling。
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

from ..core.checkpoint import BaseCheckpointer
from ..core.config import Config
from ..core.graph import (
    END,
    START,
    CompiledGraph,
    StateGraph,
    add_messages,
)

if TYPE_CHECKING:
    from ..core.llm import ClearAgentLLM
    from ..tools.registry import ToolRegistry


class SimpleGraphState(TypedDict, total=False):
    messages: Annotated[List[Dict[str, Any]], add_messages]
    tool_calls_pending: List[Any]
    total_tokens: int
    iterations: int
    max_iterations: int
    final_answer: Optional[str]


def _build_user_tool_schemas(registry: Optional["ToolRegistry"]) -> List[Dict[str, Any]]:
    """仅用户工具，无内置工具"""
    if registry is None:
        return []
    schemas: List[Dict[str, Any]] = []
    get_all = getattr(registry, "get_all_tools", None)
    if callable(get_all):
        for tool in get_all():
            schemas.append(tool.to_openai_schema())
    else:
        for name in registry.list_tools():
            tool = registry.get_tool(name) if hasattr(registry, "get_tool") else None
            if tool is not None:
                schemas.append(tool.to_openai_schema())
    return schemas


def _make_llm_node(
    llm: "ClearAgentLLM", tool_schemas: List[Dict[str, Any]]
) -> Callable[[SimpleGraphState], Dict[str, Any]]:
    def llm_node(state: SimpleGraphState) -> Dict[str, Any]:
        messages = list(state.get("messages") or [])
        response: Any

        if tool_schemas:
            response = llm.invoke_with_tools(
                messages=messages, tools=tool_schemas, tool_choice="auto"
            )
            tool_calls = response.tool_calls or []
            content = response.content
        else:
            # 纯对话：用 invoke 拿 content
            response = llm.invoke(messages)
            tool_calls = []
            content = response.content

        if tool_schemas and hasattr(llm, "serialize_assistant_message"):
            assistant_msg = llm.serialize_assistant_message(response)
        else:
            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in tool_calls
                ]

        delta_tokens = (response.usage or {}).get("total_tokens", 0) or 0

        update: Dict[str, Any] = {
            "messages": [assistant_msg],
            "tool_calls_pending": tool_calls,
            "total_tokens": (state.get("total_tokens") or 0) + delta_tokens,
            "iterations": (state.get("iterations") or 0) + 1,
        }
        if not tool_calls:
            update["final_answer"] = content or ""
        return update

    return llm_node


def _make_tool_executor_node(
    registry: Optional["ToolRegistry"],
) -> Callable[[SimpleGraphState], Dict[str, Any]]:
    def tool_executor_node(state: SimpleGraphState) -> Dict[str, Any]:
        tool_calls = state.get("tool_calls_pending") or []
        new_messages: List[Dict[str, Any]] = []

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
                        "content": f"参数解析失败: {e}",
                    }
                )
                continue

            tool = registry.get_tool(tool_name) if registry and hasattr(registry, "get_tool") else None
            if tool is None:
                new_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"工具 {tool_name} 未注册",
                    }
                )
                continue

            try:
                response = tool.run_with_timing(args)
                content = response.text if hasattr(response, "text") else str(response)
            except Exception as e:
                content = f"工具执行失败: {e}"

            new_messages.append(
                {"role": "tool", "tool_call_id": tool_call_id, "content": content}
            )

        return {"messages": new_messages, "tool_calls_pending": []}

    return tool_executor_node


def _router_after_llm(state: SimpleGraphState) -> str:
    if state.get("final_answer") is not None:
        return "end"
    max_iter = state.get("max_iterations") or 3
    if (state.get("iterations") or 0) >= max_iter:
        return "end"
    if state.get("tool_calls_pending"):
        return "tools"
    return "end"


def _router_after_tools(state: SimpleGraphState) -> str:
    if state.get("final_answer") is not None:
        return "end"
    max_iter = state.get("max_iterations") or 3
    if (state.get("iterations") or 0) >= max_iter:
        return "end"
    return "llm"


def build_simple_graph(
    llm: "ClearAgentLLM",
    tool_registry: Optional["ToolRegistry"] = None,
    config: Optional[Config] = None,
    checkpointer: Optional[BaseCheckpointer] = None,
) -> CompiledGraph[SimpleGraphState]:
    """构建 Simple StateGraph 并编译"""
    tool_schemas = _build_user_tool_schemas(tool_registry)
    llm_node = _make_llm_node(llm, tool_schemas)
    tool_node = _make_tool_executor_node(tool_registry)

    g: StateGraph[SimpleGraphState] = StateGraph(SimpleGraphState)
    g.add_node("llm", llm_node)
    if tool_schemas:
        g.add_node("tools", tool_node)
        g.add_edge(START, "llm")
        g.add_conditional_edges("llm", _router_after_llm, {"tools": "tools", "end": END})
        g.add_conditional_edges("tools", _router_after_tools, {"llm": "llm", "end": END})
    else:
        # 无工具：极简两节点
        g.add_edge(START, "llm")
        g.add_edge("llm", END)

    return g.compile(checkpointer=checkpointer)


__all__ = ["SimpleGraphState", "build_simple_graph"]
