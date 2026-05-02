"""MCPClient —— 连接外部 MCP server，把工具吃进 ClearAgent ``ToolRegistry``

最小用法：

```python
from clear_agent.mcp import MCPClient
from clear_agent.tools.registry import ToolRegistry

# stdio：启动子进程模式（比如 npx 启动一个 MCP server）
client = MCPClient.connect_stdio(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)

registry = ToolRegistry()
client.register_to(registry)   # 自动 list_tools 并注册

# 现在 ToolRegistry 里就有了 fs 工具，可直接用于 ReAct agent
```

依赖（optional）：``pip install clear-agent[mcp]``。

mcp SDK 是 async 的，本类把同步 API 桥接到内部 ``asyncio.run``。
对于长连接 / 流式 / 高并发场景，请用 ``MCPClient.aopen()`` async context manager。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .adapter import (
    MCPException,
    MCPToolAdapter,
    mcp_tool_to_clear_agent,
)

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


class MCPClient:
    """同步入口的 MCP 客户端

    构造方式：
    - ``MCPClient.connect_stdio(command, args)`` —— 启动子进程
    - ``MCPClient.connect_sse(url)`` —— 连接 SSE / streamable_http endpoint
    - ``MCPClient(transport_factory=...)`` —— 用户自定义 anyio 传输

    不直接调构造函数，请用工厂方法。

    Attributes:
        transport: ``"stdio"`` / ``"sse"`` / ``"custom"``
        is_connected: 是否已 initialize
        tools_cache: 上次 ``list_tools`` 的结果（懒加载）
    """

    def __init__(
        self,
        transport: str,
        transport_params: Optional[Dict[str, Any]] = None,
    ):
        self.transport = transport
        self.transport_params = transport_params or {}
        self.is_connected = False
        self.tools_cache: Optional[List[Dict[str, Any]]] = None

    # ==================== 工厂 ====================

    @classmethod
    def connect_stdio(
        cls, command: str, args: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None
    ) -> "MCPClient":
        """启动子进程模式连接（最常用）"""
        return cls(
            transport="stdio",
            transport_params={"command": command, "args": args or [], "env": env or {}},
        )

    @classmethod
    def connect_sse(cls, url: str, headers: Optional[Dict[str, str]] = None) -> "MCPClient":
        """SSE / Streamable HTTP 模式（远程 MCP server）"""
        return cls(
            transport="sse",
            transport_params={"url": url, "headers": headers or {}},
        )

    # ==================== 同步 API（内部 asyncio.run） ====================

    def list_tools(self) -> List[Dict[str, Any]]:
        """同步 list_tools；返回 ``[{"name", "description", "inputSchema"}, ...]``"""
        _check_mcp_installed()
        result = asyncio.run(self._async_list_tools())
        self.tools_cache = result
        return result

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """同步调一次远程工具；返回文本结果"""
        _check_mcp_installed()
        return asyncio.run(self._async_call_tool(name, arguments or {}))

    def get_clear_agent_tools(self) -> List[MCPToolAdapter]:
        """返回包装好的 ``MCPToolAdapter`` 列表（call_fn 走 self.call_tool）"""
        tools = self.list_tools()
        return [
            mcp_tool_to_clear_agent(t, call_fn=self.call_tool) for t in tools
        ]

    def register_to(self, registry: "ToolRegistry") -> int:
        """把所有 MCP tools 一次性注册到 ClearAgent ``ToolRegistry``

        Returns:
            注册的工具数
        """
        tools = self.get_clear_agent_tools()
        n = 0
        for t in tools:
            try:
                registry.register_tool(t)
                n += 1
            except Exception as e:
                logger.warning(f"⚠️ 注册 MCP tool '{t.name}' 失败: {e}")
        return n

    # ==================== 异步实现 ====================

    async def _async_list_tools(self) -> List[Dict[str, Any]]:
        async with self._connect() as session:
            resp = await session.list_tools()
            tools_attr = getattr(resp, "tools", None) or resp
            out: List[Dict[str, Any]] = []
            for t in tools_attr or []:
                out.append(
                    {
                        "name": getattr(t, "name", "") or (t.get("name", "") if isinstance(t, dict) else ""),
                        "description": getattr(t, "description", "")
                        or (t.get("description", "") if isinstance(t, dict) else ""),
                        "inputSchema": getattr(t, "inputSchema", None)
                        or (t.get("inputSchema") if isinstance(t, dict) else None)
                        or {},
                    }
                )
            return out

    async def _async_call_tool(
        self, name: str, arguments: Dict[str, Any]
    ) -> str:
        async with self._connect() as session:
            result = await session.call_tool(name, arguments)
            # mcp 返回 CallToolResult.content: List[TextContent | ImageContent | ...]
            content = getattr(result, "content", None) or result
            if isinstance(content, list):
                texts: List[str] = []
                for c in content:
                    text = getattr(c, "text", None)
                    if text is None and isinstance(c, dict):
                        text = c.get("text") or c.get("data") or str(c)
                    if text is not None:
                        texts.append(str(text))
                return "\n".join(texts)
            return str(content)

    def _connect(self):
        """返回 async context manager；进入后 yield ``ClientSession``"""
        if self.transport == "stdio":
            return _stdio_session(**self.transport_params)
        if self.transport == "sse":
            return _sse_session(**self.transport_params)
        raise MCPException(f"不支持的 transport: {self.transport}")


# ==================== 内部 async helpers ====================


class _stdio_session:
    """async context manager: 启动 stdio 子进程 + 初始化 ClientSession"""

    def __init__(self, command: str, args: List[str], env: Dict[str, str]):
        self.command = command
        self.args = args
        self.env = env
        self._stdio_cm = None
        self._session_cm = None

    async def __aenter__(self):
        from mcp import ClientSession  # type: ignore
        from mcp.client.stdio import StdioServerParameters, stdio_client  # type: ignore

        params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=self.env or None,
        )
        self._stdio_cm = stdio_client(params)
        read, write = await self._stdio_cm.__aenter__()
        self._session_cm = ClientSession(read, write)
        session = await self._session_cm.__aenter__()
        await session.initialize()
        return session

    async def __aexit__(self, exc_type, exc, tb):
        if self._session_cm is not None:
            await self._session_cm.__aexit__(exc_type, exc, tb)
        if self._stdio_cm is not None:
            await self._stdio_cm.__aexit__(exc_type, exc, tb)


class _sse_session:
    """async context manager: 连接 SSE / streamable_http endpoint"""

    def __init__(self, url: str, headers: Dict[str, str]):
        self.url = url
        self.headers = headers
        self._sse_cm = None
        self._session_cm = None

    async def __aenter__(self):
        from mcp import ClientSession  # type: ignore
        try:
            from mcp.client.sse import sse_client  # type: ignore
        except ImportError:
            from mcp.client.streamable_http import streamablehttp_client as sse_client  # type: ignore

        self._sse_cm = sse_client(self.url, headers=self.headers)
        streams = await self._sse_cm.__aenter__()
        # 兼容返回 (read, write) 或 (read, write, _)
        read, write = streams[0], streams[1]
        self._session_cm = ClientSession(read, write)
        session = await self._session_cm.__aenter__()
        await session.initialize()
        return session

    async def __aexit__(self, exc_type, exc, tb):
        if self._session_cm is not None:
            await self._session_cm.__aexit__(exc_type, exc, tb)
        if self._sse_cm is not None:
            await self._sse_cm.__aexit__(exc_type, exc, tb)


__all__ = ["MCPClient"]
