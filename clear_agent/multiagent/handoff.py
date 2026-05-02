"""Handoff 原语 —— agent 之间的控制权移交

Multi-agent 三种范式（supervisor / swarm / handoff）共用的基础 API：

- ``Handoff`` 数据类：要把控制权交给哪个 ``target`` agent，附带可选 ``message`` 和
  ``state_patch``
- ``make_handoff_tool(targets)``：把可移交目标列表构建成 OpenAI function-calling schema
- ``parse_handoff_from_tool_calls(tool_calls)``：从 LLM tool_calls 中抽取 ``Handoff``

设计与 LangGraph swarm 的 handoff 工具行为对齐，但不引入 langgraph 依赖。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Handoff 工具名前缀，OpenAI function name 必须 ≤ 64 字符 + 限定字符集
HANDOFF_TOOL_PREFIX = "transfer_to_"


# Multi-agent 终止信号：active_agent 设为该值 → 图终止
HANDOFF_END = "__handoff_end__"


@dataclass
class Handoff:
    """从一个 agent 移交控制权给另一个 agent

    Attributes:
        target: 接管的 agent 名称（必须在 ``build_*_graph`` 注册的 agents 里）
        message: 可选的移交说明（写入 messages 让接管 agent 看到上下文）
        state_patch: 可选的 state 字段覆盖（每个 agent 可能有自己的工作字段）
    """

    target: str
    message: str = ""
    state_patch: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "message": self.message,
            "state_patch": dict(self.state_patch),
        }


def make_handoff_tool_name(target: str) -> str:
    """规范化目标 agent 名为合法的工具名"""
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in target)
    return f"{HANDOFF_TOOL_PREFIX}{safe}"


def make_handoff_tool(target: str, description: Optional[str] = None) -> Dict[str, Any]:
    """构建单个目标的 OpenAI function-calling schema"""
    desc = description or f"Transfer control to agent '{target}'."
    return {
        "type": "function",
        "function": {
            "name": make_handoff_tool_name(target),
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Optional handoff message to the next agent.",
                    },
                },
                "required": [],
            },
        },
    }


def make_handoff_tools(
    targets: List[str], descriptions: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """批量构建 handoff 工具集"""
    descs = descriptions or {}
    return [make_handoff_tool(t, descs.get(t)) for t in targets]


def parse_handoff_from_tool_calls(tool_calls: List[Any]) -> Optional[Handoff]:
    """从 ``LLMToolResponse.tool_calls`` 中抽取第一个 ``transfer_to_*`` 调用

    Returns:
        ``Handoff`` 实例；找不到返回 None
    """
    if not tool_calls:
        return None
    for tc in tool_calls:
        name = getattr(tc, "name", None) or (
            tc.get("name") if isinstance(tc, dict) else None
        )
        if not name or not name.startswith(HANDOFF_TOOL_PREFIX):
            continue
        target = name[len(HANDOFF_TOOL_PREFIX):]
        # 解析 arguments
        args_raw = getattr(tc, "arguments", None) or (
            tc.get("arguments") if isinstance(tc, dict) else None
        )
        message = ""
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
                message = args.get("message", "") if isinstance(args, dict) else ""
            except json.JSONDecodeError:
                message = ""
        elif isinstance(args_raw, dict):
            message = args_raw.get("message", "")
        return Handoff(target=target, message=message)
    return None


__all__ = [
    "Handoff",
    "HANDOFF_TOOL_PREFIX",
    "HANDOFF_END",
    "make_handoff_tool_name",
    "make_handoff_tool",
    "make_handoff_tools",
    "parse_handoff_from_tool_calls",
]
