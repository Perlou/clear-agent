"""MCPServer —— 把 ClearAgent ToolRegistry 暴露为 MCP server

让 Cursor / Claude Desktop / 任何 MCP 客户端通过 stdio 或 HTTP 调用
ClearAgent 注册的工具。

依赖（optional）：
```bash
pip install clear-agent[mcp]
```

最小用法：

```python
from clear_agent.mcp import MCPServer
from clear_agent.tools.registry import ToolRegistry
from clear_agent.tools.builtin.calculator import CalculatorTool

registry = ToolRegistry()
registry.register_tool(CalculatorTool())

server = MCPServer(registry, name="clear-agent-tools")
server.run(transport="stdio")  # 阻塞运行；通常是从 stdin 接收 mcp 协议消息
```

Cursor 的 `mcp.json` 配置：
```json
{
  "mcpServers": {
    "clear-agent": {
      "command": "python",
      "args": ["-m", "my_module"]
    }
  }
}
```
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .adapter import MCPException, clear_agent_tool_to_mcp_schema

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


logger = logging.getLogger(__name__)


def _check_mcp_installed() -> None:
    try:
        import mcp  # type: ignore  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "MCP 集成需要安装官方 mcp SDK：pip install clear-agent[mcp]"
        ) from e


class MCPServer:
    """把 ClearAgent ``ToolRegistry`` 暴露为 MCP server

    Args:
        registry: ClearAgent ``ToolRegistry``，所有注册工具会暴露给 MCP 客户端
        name: server 名（向客户端报告）
        version: server 版本

    Note:
        本类是同步入口的薄包装；底层 mcp SDK 是 async 的，``run`` 内部用 ``asyncio.run``。
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        name: str = "clear-agent",
        version: str = "2.0.0",
    ):
        self.registry = registry
        self.name = name
        self.version = version

    # ==================== 公开接口 ====================

    def list_tools(self) -> List[Dict[str, Any]]:
        """返回 ClearAgent 工具的 MCP schema 列表（同步，便于检查）"""
        return [
            clear_agent_tool_to_mcp_schema(t) for t in self._iter_registered_tools()
        ]

    def run(self, transport: str = "stdio") -> None:
        """启动 MCP server（阻塞）

        Args:
            transport: ``"stdio"``（默认，子进程模式）/ ``"streamable_http"``
        """
        _check_mcp_installed()
        if transport == "stdio":
            asyncio.run(self._run_stdio())
        elif transport == "streamable_http":
            asyncio.run(self._run_http())
        else:
            raise MCPException(f"不支持的 transport: {transport}")

    # ==================== 内部 ====================

    def _iter_registered_tools(self):
        """从 ToolRegistry 取出全部工具（兼容多种 registry 接口）"""
        get_all = getattr(self.registry, "get_all_tools", None)
        if callable(get_all):
            yield from get_all()
            return
        # fallback: list_tools() + get_tool()
        list_fn = getattr(self.registry, "list_tools", None)
        get_fn = getattr(self.registry, "get_tool", None)
        if callable(list_fn) and callable(get_fn):
            for n in list_fn():
                t = get_fn(n)
                if t is not None:
                    yield t

    async def _run_stdio(self) -> None:
        """启动 stdio MCP server"""
        from mcp.server import Server  # type: ignore
        from mcp.server.stdio import stdio_server  # type: ignore
        from mcp.types import TextContent, Tool as MCPTool  # type: ignore

        server: Any = Server(self.name)

        @server.list_tools()  # type: ignore[misc]
        async def _handle_list_tools() -> List[Any]:
            mcp_tools: List[Any] = []
            for t in self._iter_registered_tools():
                schema = clear_agent_tool_to_mcp_schema(t)
                mcp_tools.append(
                    MCPTool(
                        name=schema["name"],
                        description=schema["description"],
                        inputSchema=schema["inputSchema"],
                    )
                )
            return mcp_tools

        @server.call_tool()  # type: ignore[misc]
        async def _handle_call_tool(
            name: str, arguments: Optional[Dict[str, Any]] = None
        ) -> List[Any]:
            tool = self._find_tool_by_name(name)
            if tool is None:
                return [TextContent(type="text", text=f"Tool '{name}' not found")]
            try:
                resp = tool.run_with_timing(arguments or {})
                text = getattr(resp, "text", None) or str(resp)
                return [TextContent(type="text", text=text)]
            except Exception as e:
                return [TextContent(type="text", text=f"Tool error: {e}")]

        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    async def _run_http(self) -> None:
        """启动 HTTP MCP server（最小实现；详细路由用户自接 FastAPI）"""
        raise NotImplementedError(
            "streamable_http transport 的 MCPServer 实现需要用户接入 FastAPI / Starlette。"
            "参考 docs/mcp-guide.md 给出的 ASGI 集成示例。"
        )

    def _find_tool_by_name(self, name: str):
        for t in self._iter_registered_tools():
            if t.name == name:
                return t
        return None


__all__ = ["MCPServer"]
