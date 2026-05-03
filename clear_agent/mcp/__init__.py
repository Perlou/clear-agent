"""MCP（Model Context Protocol）集成

让 ClearAgent 与 MCP 生态互通：

- ``MCPClient`` —— 连接外部 MCP server（如 ``@modelcontextprotocol/server-filesystem``、
  github MCP、custom Python MCP server），把其工具自动注册到 ``ToolRegistry``
- ``MCPServer`` —— 把 ClearAgent ``ToolRegistry`` 暴露为 MCP server，
  让 Cursor / Claude Desktop 直接调用
- ``MCPToolAdapter`` —— 单个 MCP tool ↔ ClearAgent ``Tool`` 双向适配

依赖（optional）：

```bash
pip install clear-agent[mcp]
```

详见 ``docs/mcp-guide.md`` 
"""

from .adapter import (
    MCPException,
    MCPToolAdapter,
    clear_agent_tool_to_mcp_schema,
    mcp_tool_to_clear_agent,
)
from .client import MCPClient
from .server import MCPServer

__all__ = [
    "MCPException",
    "MCPToolAdapter",
    "MCPClient",
    "MCPServer",
    "mcp_tool_to_clear_agent",
    "clear_agent_tool_to_mcp_schema",
]
