"""Swarm pattern —— 去中心化 agent 群

各 agent 之间直接通过 ``Handoff`` 互相移交，没有中心协调器。

```
   ┌──────────┐                ┌──────────┐
   │ agent_a  │──handoff─────▶│ agent_b  │
   └──────────┘                └────┬─────┘
        ▲                           │
        └───────handoff─────────────┘
```

每个 agent 节点函数返回 ``{"active_agent": "<next>"}`` 或保持自身（继续工作 N 轮）。
``HANDOFF_END`` 触发图终止。

与 supervisor 模式相比，swarm 适合 agents 之间有明确技能分工、希望减少中心瓶颈
的场景；缺点是路由逻辑分散在每个 agent 内部，需要 agent 自身懂何时移交。
"""

from __future__ import annotations

from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypedDict,
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


# ==================== State ====================


class SwarmState(TypedDict, total=False):
    """去中心化 swarm 模式的标准 state

    Attributes:
        messages: 共享对话历史
        active_agent: 当前掌控权的 agent 名；agent 函数本身会返回 ``active_agent`` 决定下一跳
        handoff_count: 累计移交次数
        max_handoffs: 上限
        result: 任意结果
    """

    messages: Annotated[List[Dict[str, Any]], add_messages]
    active_agent: Optional[str]
    handoff_count: int
    max_handoffs: int
    result: Optional[Any]


# ==================== 主入口 ====================


def build_swarm_graph(
    agents: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
    default_active: str,
    max_handoffs: int = 10,
    checkpointer: Optional[BaseCheckpointer] = None,
) -> CompiledGraph[SwarmState]:
    """构建一个去中心化 swarm 图

    Args:
        agents: ``{name -> node_fn}``。每个 agent 节点 **应当** 在返回中设置
            ``active_agent`` 字段：
              - 自身名字（继续工作）
              - 其他 agent 名（handoff）
              - ``HANDOFF_END`` （终止图）
              - 不设置或 None → 默认终止
        default_active: 入口活跃 agent 名（必须在 agents 里）
        max_handoffs: 上限，超限路由到 END
        checkpointer: 可选

    Returns:
        ``CompiledGraph[SwarmState]``。

    Example:
        >>> def planner(state):
        ...     return {
        ...         "messages": [{"role": "assistant", "content": "planned"}],
        ...         "active_agent": "executor",
        ...     }
        >>> def executor(state):
        ...     return {
        ...         "messages": [{"role": "assistant", "content": "executed"}],
        ...         "active_agent": HANDOFF_END,
        ...     }
        >>> graph = build_swarm_graph(
        ...     {"planner": planner, "executor": executor},
        ...     default_active="planner",
        ... )
    """
    if not agents:
        raise ValueError("agents 不能为空")
    if default_active not in agents:
        raise ValueError(
            f"default_active='{default_active}' 不在 agents 中: {list(agents.keys())}"
        )

    reserved = {START, END, "_swarm_dispatch"}
    for name in agents:
        if name in reserved:
            raise ValueError(f"agent 名 '{name}' 是保留字")

    g: StateGraph[SwarmState] = StateGraph(SwarmState)

    # 一个 dispatch 节点：负责设置 default_active 和 max_handoffs（首次进入时）
    def _dispatch(state: SwarmState) -> Dict[str, Any]:
        update: Dict[str, Any] = {}
        if state.get("active_agent") is None:
            update["active_agent"] = default_active
        if state.get("max_handoffs") is None and "max_handoffs" not in state:
            update["max_handoffs"] = max_handoffs
        return update

    g.add_node("_swarm_dispatch", _dispatch)

    # 注册每个 agent —— 自增 handoff_count（如果 active_agent 变了）
    for name, fn in agents.items():
        captured_fn = fn
        captured_name = name

        def _make_agent_wrapper(agent_fn, agent_name):
            def _wrapper(state: SwarmState) -> Dict[str, Any]:
                out = agent_fn(dict(state))
                if not isinstance(out, dict):
                    out = {}
                # 计算 handoff_count
                next_active = out.get("active_agent")
                old_count = state.get("handoff_count") or 0
                # 切换到不同 agent / 终止 → 计 1 次 handoff
                if next_active is not None and next_active != agent_name:
                    out["handoff_count"] = old_count + 1
                else:
                    out["handoff_count"] = old_count
                return out

            _wrapper.__name__ = f"_swarm_agent_{agent_name}"
            return _wrapper

        g.add_node(captured_name, _make_agent_wrapper(captured_fn, captured_name))

    # 路由器：根据 active_agent 决定下一跳
    def _route(state: SwarmState) -> str:
        max_h = state.get("max_handoffs") or max_handoffs
        if (state.get("handoff_count") or 0) >= max_h:
            return "end"
        target = state.get("active_agent")
        if target is None or target == HANDOFF_END:
            return "end"
        if target not in agents:
            return "end"
        return target

    g.add_edge(START, "_swarm_dispatch")
    routing = {"end": END, **{name: name for name in agents}}
    g.add_conditional_edges("_swarm_dispatch", _route, routing)
    # 每个 agent 完成后回到 dispatch 重路由
    for name in agents:
        g.add_conditional_edges(name, _route, routing)

    return g.compile(checkpointer=checkpointer)


__all__ = ["SwarmState", "build_swarm_graph"]
