"""Supervisor pattern —— 中心化协调器

一个中心 supervisor 节点决定每一轮把控制权路由到哪个 worker，worker 完成后
回到 supervisor 由其决定下一步。

```
        ┌─────────┐
        │  START  │
        └────┬────┘
             ▼
       ┌─────────────┐         ┌──────────┐
   ┌──▶│ supervisor  │──route─▶│ worker_a │──┐
   │   └─────────────┘         └──────────┘  │
   │          ▲                ┌──────────┐  │
   │          └────────────────│ worker_b │◀─┤
   │                           └──────────┘  │
   │                                         │
   └─────────────────────────────────────────┘
                                  │
                                  ▼
                              ┌─────┐
                              │ END │  (when supervisor routes to END
                              └─────┘   or max_handoffs reached)
```

worker 是任意 ``Callable[[state], partial_state]`` 形式（即标准 graph node）。
用户可以包装 ``Agent.run`` / ``build_*_graph().invoke`` 适配该签名。

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
    Union,
)

from ..core.checkpoint import BaseCheckpointer
from ..core.graph import (
    END,
    START,
    CompiledGraph,
    StateGraph,
    add_messages,
)
from .handoff import HANDOFF_END

if TYPE_CHECKING:
    pass

# ==================== State ====================

class SupervisorState(TypedDict, total=False):
    """中心化 supervisor 模式的标准 state

    Attributes:
        messages: 共享对话历史（追加 + 去重）
        active_agent: supervisor 路由到的 worker 名；``HANDOFF_END`` 表示终止
        handoff_count: 累计移交次数（防死循环）
        max_handoffs: 上限，超限自动终止
        result: 任意结果字段（worker 写入；用户自定义 schema 可扩展）
    """

    messages: Annotated[List[Dict[str, Any]], add_messages]
    active_agent: Optional[str]
    handoff_count: int
    max_handoffs: int
    result: Optional[Any]

# ==================== 主入口 ====================

def build_supervisor_graph(
    supervisor: Callable[[Dict[str, Any]], Dict[str, Any]],
    workers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
    max_handoffs: int = 10,
    checkpointer: Optional[BaseCheckpointer] = None,
) -> CompiledGraph[SupervisorState]:
    """构建一个中心化 supervisor 图

    Args:
        supervisor: 协调器节点函数。**必须**在返回的 state 中设置
            ``active_agent`` 字段为下一个 worker 名（或 ``HANDOFF_END`` 终止）。
            常见做法：让 LLM 决策 + 用 ``parse_handoff_from_tool_calls`` 解析。
        workers: ``{name -> node_fn}`` 注册表。每个 worker 完成后自动回到 supervisor。
        max_handoffs: 最大移交次数，超限路由到 END。
        checkpointer: 可选；接入后每节点写快照。

    Returns:
        ``CompiledGraph[SupervisorState]``。

    Example:
        >>> def supervisor(state):
        ...     # 简单规则：第 1 轮路由 researcher，第 2 轮路由 writer，否则 END
        ...     n = state.get("handoff_count", 0)
        ...     return {"active_agent": ["researcher", "writer", HANDOFF_END][n] if n < 3 else HANDOFF_END}
        >>> def researcher(state):
        ...     return {"messages": [{"role": "assistant", "content": "found data"}]}
        >>> def writer(state):
        ...     return {"messages": [{"role": "assistant", "content": "wrote report"}]}
        >>> graph = build_supervisor_graph(supervisor, {"researcher": researcher, "writer": writer})
        >>> result = graph.invoke({"messages": [], "max_handoffs": 5})
    """
    if not workers:
        raise ValueError("workers 不能为空")

    reserved = {START, END, "supervisor"}
    for name in workers:
        if name in reserved:
            raise ValueError(f"worker 名 '{name}' 是保留字")

    g: StateGraph[SupervisorState] = StateGraph(SupervisorState)

    # supervisor 节点：包一层默认 max_handoffs
    def _supervisor_wrapper(state: SupervisorState) -> Dict[str, Any]:
        out = supervisor(dict(state))
        if not isinstance(out, dict):
            out = {}
        # 若用户没设 max_handoffs，注入默认
        if "max_handoffs" not in state and "max_handoffs" not in out:
            out["max_handoffs"] = max_handoffs
        return out

    g.add_node("supervisor", _supervisor_wrapper)

    # 注册每个 worker —— 完成后自增 handoff_count
    for name, fn in workers.items():
        captured_fn = fn
        captured_name = name

        def _make_worker_wrapper(
            worker_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
            worker_name: str,
        ) -> Callable[[SupervisorState], Dict[str, Any]]:
            def _wrapper(state: SupervisorState) -> Dict[str, Any]:
                out = worker_fn(dict(state))
                if not isinstance(out, dict):
                    out = {}
                out["handoff_count"] = (state.get("handoff_count") or 0) + 1
                # worker 完成后清空 active_agent，让 supervisor 重新决策
                out["active_agent"] = None
                return out

            _wrapper.__name__ = f"_supervisor_worker_{worker_name}"
            return _wrapper

        g.add_node(captured_name, _make_worker_wrapper(captured_fn, captured_name))

    # 路由：supervisor → 选定的 worker / END
    def _route_after_supervisor(state: SupervisorState) -> str:
        max_h = state.get("max_handoffs") or max_handoffs
        if (state.get("handoff_count") or 0) >= max_h:
            return "end"
        target = state.get("active_agent")
        if target is None or target == HANDOFF_END or target not in workers:
            return "end"
        return target

    g.add_edge(START, "supervisor")
    routing: Dict[str, Union[str, List[str]]] = {
        "end": END,
        **{name: name for name in workers},
    }
    g.add_conditional_edges("supervisor", _route_after_supervisor, routing)

    # 每个 worker 跑完回到 supervisor
    for name in workers:
        g.add_edge(name, "supervisor")

    return g.compile(checkpointer=checkpointer)

__all__ = ["SupervisorState", "build_supervisor_graph"]
