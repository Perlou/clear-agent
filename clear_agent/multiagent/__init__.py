"""Multi-agent 范式包（基于 StateGraph 原生）

三种范式：

1. **Supervisor**（中心化）—— 一个 supervisor 节点决策路由，worker 跑完回到 supervisor
   - ``build_supervisor_graph(supervisor, workers)``
2. **Swarm**（去中心化）—— agents 之间直接通过 ``Handoff`` 互相移交
   - ``build_swarm_graph(agents, default_active)``
3. **Handoff** 原语 —— 给 LLM 提供 ``transfer_to_*`` 工具集，让 agent 自己决定移交
   - ``Handoff`` / ``make_handoff_tools`` / ``parse_handoff_from_tool_calls``

详见 plan §三 "Multi-agent 范式包" 
"""

from .handoff import (
    HANDOFF_END,
    HANDOFF_TOOL_PREFIX,
    Handoff,
    make_handoff_tool,
    make_handoff_tool_name,
    make_handoff_tools,
    parse_handoff_from_tool_calls,
)
from .supervisor import SupervisorState, build_supervisor_graph
from .swarm import SwarmState, build_swarm_graph

__all__ = [
    # handoff
    "Handoff",
    "HANDOFF_END",
    "HANDOFF_TOOL_PREFIX",
    "make_handoff_tool_name",
    "make_handoff_tool",
    "make_handoff_tools",
    "parse_handoff_from_tool_calls",
    # supervisor
    "SupervisorState",
    "build_supervisor_graph",
    # swarm
    "SwarmState",
    "build_swarm_graph",
]
