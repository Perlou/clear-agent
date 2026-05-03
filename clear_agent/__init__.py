"""
ClearAgent - 灵活、可扩展的多智能体框架

基于OpenAI原生API构建，提供简洁高效的智能体开发体验。
"""

# 配置第三方库的日志级别，减少噪音
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

from .version import __version__, __author__, __email__, __description__

# 核心组件
from .core.llm import ClearAgentLLM
from .core.config import Config
from .core.message import Message
from .core.exceptions import ClearAgentException
from .core.structured import StructuredLLM, StructuredOutputError

# Agent实现
from .agents.simple_agent import SimpleAgent
from .agents.react_agent import ReActAgent
from .agents.reflection_agent import ReflectionAgent
from .agents.plan_solve_agent import PlanSolveAgent

# 向后兼容别名
PlanAndSolveAgent = PlanSolveAgent

# 工具系统
from .tools.registry import ToolRegistry, global_registry
from .tools.builtin.calculator import CalculatorTool, calculate

# 2.0 graph builders（可选导入）
try:
    from .agents import build_react_graph
    from .core.graph import StateGraph, CompiledGraph, START, END, RunConfig
    from .core.checkpoint import (
        BaseCheckpointer,
        InMemoryCheckpointer,
        JsonFileCheckpointer,
        SqliteCheckpointer,
        make_checkpointer,
    )
    _GRAPH_AVAILABLE = True
except ImportError:
    _GRAPH_AVAILABLE = False

# Multi-agent 范式包（依赖 graph）
try:
    from .multiagent import (
        HANDOFF_END,
        Handoff,
        build_supervisor_graph,
        build_swarm_graph,
        make_handoff_tool,
        make_handoff_tools,
        parse_handoff_from_tool_calls,
    )
    _MULTIAGENT_AVAILABLE = True
except ImportError:
    _MULTIAGENT_AVAILABLE = False

__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    # 核心组件
    "ClearAgentLLM",
    "Config",
    "Message",
    "ClearAgentException",
    "StructuredLLM",
    "StructuredOutputError",
    # Agent范式
    "SimpleAgent",
    "ReActAgent",
    "ReflectionAgent",
    "PlanSolveAgent",
    "PlanAndSolveAgent",  # 向后兼容别名
    # 工具系统
    "ToolRegistry",
    "global_registry",
    "CalculatorTool",
    "calculate",
]

if _GRAPH_AVAILABLE:
    __all__ += [
        "build_react_graph",
        "StateGraph",
        "CompiledGraph",
        "START",
        "END",
        "RunConfig",
        "BaseCheckpointer",
        "InMemoryCheckpointer",
        "JsonFileCheckpointer",
        "SqliteCheckpointer",
        "make_checkpointer",
    ]

if _MULTIAGENT_AVAILABLE:
    __all__ += [
        "HANDOFF_END",
        "Handoff",
        "build_supervisor_graph",
        "build_swarm_graph",
        "make_handoff_tool",
        "make_handoff_tools",
        "parse_handoff_from_tool_calls",
    ]
