"""MCP（Model Context Protocol）适配层

把 MCP 协议的工具接口与 ClearAgent ``Tool`` 双向桥接。

**两个方向**：
- ``MCPToolAdapter``：把外部 MCP server 暴露的工具包装成 ClearAgent ``Tool``，
  让 ``ToolRegistry`` 可以直接注册使用
- (反向) ``MCPServer.export_tools(registry)``：把 ClearAgent ToolRegistry 转成
  MCP server 期望的 schema 列表（让 Cursor / Claude Desktop 调用 ClearAgent 工具）

详见 ``project_docs/07-anton-agents-port.md`` §5（MCP 协议引入）。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..core.exceptions import ClearAgentException
from ..tools.base import Tool, ToolParameter
from ..tools.response import ToolResponse


class MCPException(ClearAgentException):
    """MCP 协议相关错误"""


class MCPToolAdapter(Tool):
    """把单个 MCP tool 包装为 ClearAgent ``Tool``

    ``call_fn`` 期望签名：``(name: str, args: dict) -> str | Awaitable[str]``
    —— 由 ``MCPClient`` 提供。

    Args:
        name: MCP tool 名（保持原样，便于反查）
        description: MCP tool 描述
        input_schema: MCP tool 的 ``inputSchema``（JSON Schema 形式）
        call_fn: 调用回调（同步或异步）；同步 fn 直接返回 str；async fn 由本类包一层 asyncio.run
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Optional[Dict[str, Any]],
        call_fn: Callable[[str, Dict[str, Any]], Any],
    ):
        super().__init__(name=name, description=description)
        self.input_schema = input_schema or {"type": "object", "properties": {}}
        self._call_fn = call_fn

    def get_parameters(self) -> List[ToolParameter]:
        """从 MCP inputSchema 转 ClearAgent ToolParameter 列表"""
        props = (self.input_schema or {}).get("properties", {}) or {}
        required = set((self.input_schema or {}).get("required", []) or [])
        out: List[ToolParameter] = []
        for pname, pdef in props.items():
            ptype = (pdef.get("type") if isinstance(pdef, dict) else None) or "string"
            pdesc = (pdef.get("description") if isinstance(pdef, dict) else "") or ""
            out.append(
                ToolParameter(
                    name=pname,
                    type=str(ptype),
                    description=pdesc,
                    required=pname in required,
                )
            )
        return out

    def to_openai_schema(self) -> Dict[str, Any]:
        """直接复用 MCP inputSchema 作为 OpenAI function parameters"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        try:
            result = self._call_fn(self.name, parameters)
            # async result → 包一层 asyncio.run
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)
            text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
            return ToolResponse.success(text=text, data={"raw": result})
        except Exception as e:
            return ToolResponse.error(
                code="MCP_TOOL_FAILED",
                message=f"MCP tool '{self.name}' 调用失败: {e}",
            )


def mcp_tool_to_clear_agent(
    mcp_tool: Any,
    call_fn: Callable[[str, Dict[str, Any]], Any],
) -> MCPToolAdapter:
    """把 MCP SDK 的 ``Tool`` 对象（或 dict）转为 ``MCPToolAdapter``

    支持以下三种输入：
    - mcp.types.Tool 对象（含 ``.name`` / ``.description`` / ``.inputSchema``）
    - dict ``{"name", "description", "inputSchema"}``
    - dict ``{"name", "description", "input_schema"}``
    """

    def _attr(obj: Any, name: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, obj.get(_to_snake(name), default))
        return getattr(obj, name, getattr(obj, _to_snake(name), default))

    name = _attr(mcp_tool, "name") or ""
    description = _attr(mcp_tool, "description") or ""
    input_schema = _attr(mcp_tool, "inputSchema") or _attr(mcp_tool, "input_schema") or {}
    return MCPToolAdapter(
        name=str(name),
        description=str(description),
        input_schema=input_schema if isinstance(input_schema, dict) else {},
        call_fn=call_fn,
    )


def clear_agent_tool_to_mcp_schema(tool: Tool) -> Dict[str, Any]:
    """把 ClearAgent ``Tool`` 转为 MCP server 期望的 tool descriptor

    返回结构：``{"name", "description", "inputSchema": {JSON Schema}}``
    可直接交给 MCP server 注册（如 ``@server.list_tools()`` 回调）。
    """
    schema = tool.to_openai_schema()
    fn = schema.get("function", {})
    return {
        "name": fn.get("name") or tool.name,
        "description": fn.get("description") or tool.description,
        "inputSchema": fn.get("parameters") or {"type": "object", "properties": {}},
    }


def _to_snake(name: str) -> str:
    """``inputSchema`` -> ``input_schema``（驼峰转下划线）"""
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


__all__ = [
    "MCPException",
    "MCPToolAdapter",
    "mcp_tool_to_clear_agent",
    "clear_agent_tool_to_mcp_schema",
]
