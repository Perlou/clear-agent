# MCP 协议集成

[Model Context Protocol](https://modelcontextprotocol.io/) 是 Anthropic 推动的工具/资源协议标准，被 Cursor、Claude Desktop、Cline 等支持。ClearAgent 提供双向集成。

## 安装

```bash
pip install "clear-agent[mcp]"
```

## 作为 MCP 客户端：吃外部 MCP 工具

```python
from clear_agent import ToolRegistry
from clear_agent.mcp import MCPClient

# stdio 模式：启动子进程作为 MCP server
client = MCPClient.connect_stdio(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)

# SSE 模式：连远程 MCP server
# client = MCPClient.connect_sse(url="https://my-mcp.example/sse", headers={"X-Auth": "..."})

registry = ToolRegistry()
n = client.register_to(registry)
print(f"已注册 {n} 个 MCP 工具")

# 现在 ToolRegistry 里就有了 fs 工具，可直接给 ReActAgent 用
agent = ReActAgent(name="x", llm=llm, tool_registry=registry)
```

每个 MCP 工具会被包成 `MCPToolAdapter`（实现 ClearAgent `Tool` 接口）。

### 散件用法

```python
client = MCPClient.connect_stdio(command="...", args=[])

tools = client.list_tools()
# → [{"name": "read_file", "description": "...", "inputSchema": {...}}, ...]

result = client.call_tool("read_file", {"path": "/tmp/x.txt"})
# → "file contents..."
```

## 作为 MCP 服务端：暴露给 Cursor / Claude Desktop

```python
# my_mcp_server.py
from clear_agent.mcp import MCPServer
from clear_agent import ToolRegistry, CalculatorTool
from clear_agent.tools.from_pydantic import pydantic_tool
from pydantic import BaseModel, Field

# 准备工具
class WeatherArgs(BaseModel):
    city: str = Field(description="city name")

@pydantic_tool(description="get weather")
def get_weather(args: WeatherArgs) -> str:
    return f"{args.city}: 22°C sunny"

registry = ToolRegistry()
registry.register_tool(CalculatorTool())
registry.register_tool(get_weather)

# 启动 MCP server
server = MCPServer(registry, name="my-clear-agent-tools")
server.run(transport="stdio")   # 阻塞监听 stdin
```

Cursor 配置（`~/.cursor/mcp.json`）：
```json
{
  "mcpServers": {
    "clear-agent-tools": {
      "command": "python",
      "args": ["my_mcp_server.py"]
    }
  }
}
```

Claude Desktop 配置（`claude_desktop_config.json` 同样格式）。

## Schema 双向转换

```python
from clear_agent.mcp import (
    mcp_tool_to_clear_agent,
    clear_agent_tool_to_mcp_schema,
)

# MCP tool → ClearAgent Tool
ca_tool = mcp_tool_to_clear_agent(
    {"name": "x", "description": "...", "inputSchema": {...}},
    call_fn=lambda name, args: "result",
)

# ClearAgent Tool → MCP tool descriptor
mcp_schema = clear_agent_tool_to_mcp_schema(my_tool)
# → {"name", "description", "inputSchema"}
```

## 同步 / 异步

`MCPClient` 是同步入口，内部 `asyncio.run` 包装 mcp SDK 的 async API。
对于长连接 / 流式 / 高并发，建议直接用 mcp SDK 的 async API + `MCPToolAdapter`。

## 注意事项

- **必装 mcp SDK**：未装时 `MCPClient.list_tools()` / `MCPServer.run()` 抛 `ImportError` 友好提示；但 `MCPToolAdapter` / schema 转换函数不依赖 SDK，可直接用
- **stdio 模式**：mcp 协议通过 stdin/stdout 通信，所以 server 进程不要往 stdout 打日志（会污染协议）；改用 `logging` 写到 stderr 或文件
- **流式响应**：mcp 协议支持流式 tool result，`MCPClient.call_tool` 会把多段 `TextContent` 拼接成一个字符串返回；需要流式渲染时自行用 mcp SDK
