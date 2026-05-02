"""MCP 集成测试

mcp SDK 是 optional dep —— venv 里没装，测试通过 monkeypatch 模拟 mcp 模块或
直接验证 lazy-import + ImportError 友好提示路径。
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clear_agent.mcp import (
    MCPClient,
    MCPException,
    MCPServer,
    MCPToolAdapter,
    clear_agent_tool_to_mcp_schema,
    mcp_tool_to_clear_agent,
)
from clear_agent.tools.base import Tool, ToolParameter
from clear_agent.tools.builtin.calculator import CalculatorTool
from clear_agent.tools.registry import ToolRegistry
from clear_agent.tools.response import ToolResponse


# ==================== Section A: MCPToolAdapter ====================


def test_adapter_basic_construction():
    ad = MCPToolAdapter(
        name="echo",
        description="Echo input",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        call_fn=lambda n, a: "ok",
    )
    assert ad.name == "echo"
    assert ad.description == "Echo input"


def test_adapter_get_parameters_required():
    ad = MCPToolAdapter(
        name="t",
        description="d",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "string", "description": "param a"}},
            "required": ["a"],
        },
        call_fn=lambda n, a: "",
    )
    params = ad.get_parameters()
    assert len(params) == 1
    assert params[0].name == "a"
    assert params[0].type == "string"
    assert params[0].description == "param a"
    assert params[0].required


def test_adapter_get_parameters_optional():
    ad = MCPToolAdapter(
        name="t",
        description="d",
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a"],
        },
        call_fn=lambda n, a: "",
    )
    params = ad.get_parameters()
    by_name = {p.name: p for p in params}
    assert by_name["a"].required
    assert not by_name["b"].required
    assert by_name["b"].type == "integer"


def test_adapter_get_parameters_empty_schema():
    ad = MCPToolAdapter(
        name="t", description="d", input_schema=None, call_fn=lambda n, a: ""
    )
    assert ad.get_parameters() == []


def test_adapter_to_openai_schema():
    schema_in = {"type": "object", "properties": {"x": {"type": "number"}}}
    ad = MCPToolAdapter(
        name="my_tool", description="my desc", input_schema=schema_in,
        call_fn=lambda n, a: "",
    )
    out = ad.to_openai_schema()
    assert out["type"] == "function"
    assert out["function"]["name"] == "my_tool"
    assert out["function"]["description"] == "my desc"
    assert out["function"]["parameters"] == schema_in


def test_adapter_run_sync_call_fn():
    captured = {}

    def fake_call(name, args):
        captured["name"] = name
        captured["args"] = dict(args)
        return "result_text"

    ad = MCPToolAdapter(name="x", description="x", input_schema={}, call_fn=fake_call)
    resp = ad.run({"a": 1})
    assert resp.status.value == "success"
    assert resp.text == "result_text"
    assert captured["name"] == "x"
    assert captured["args"] == {"a": 1}


def test_adapter_run_async_call_fn():
    """async fn 被 asyncio.run 包装"""

    async def afake(name, args):
        return f"async-{name}"

    ad = MCPToolAdapter(name="ax", description="d", input_schema={}, call_fn=afake)
    resp = ad.run({})
    assert resp.status.value == "success"
    assert resp.text == "async-ax"


def test_adapter_run_non_string_result_serialized_to_json():
    """call_fn 返回 dict / list → 自动 json.dumps"""
    ad = MCPToolAdapter(
        name="x", description="d", input_schema={},
        call_fn=lambda n, a: {"foo": "bar", "n": 42},
    )
    resp = ad.run({})
    assert resp.status.value == "success"
    assert "foo" in resp.text and "bar" in resp.text


def test_adapter_run_call_fn_raises_returns_error_response():
    def boom(name, args):
        raise RuntimeError("server gone")

    ad = MCPToolAdapter(name="x", description="d", input_schema={}, call_fn=boom)
    resp = ad.run({})
    assert resp.status.value == "error"
    assert "server gone" in (resp.text or "") or "MCP tool" in (resp.text or "")


def test_adapter_is_clear_agent_tool_subclass():
    """MCPToolAdapter 必须可作为 ClearAgent Tool 注册到 ToolRegistry"""
    ad = MCPToolAdapter(
        name="t", description="d", input_schema={}, call_fn=lambda n, a: "x"
    )
    assert isinstance(ad, Tool)
    reg = ToolRegistry()
    reg.register_tool(ad)
    assert "t" in reg.list_tools()


# ==================== Section B: mcp_tool_to_clear_agent ====================


def test_mcp_tool_to_clear_agent_dict_input():
    mcp_tool = {
        "name": "fs_read",
        "description": "Read file",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
    }
    ad = mcp_tool_to_clear_agent(mcp_tool, call_fn=lambda n, a: "content")
    assert ad.name == "fs_read"
    assert ad.description == "Read file"
    assert ad.input_schema["properties"]["path"]["type"] == "string"


def test_mcp_tool_to_clear_agent_dict_snake_case():
    """支持 input_schema 键名（蛇形命名变体）"""
    mcp_tool = {
        "name": "x", "description": "d",
        "input_schema": {"type": "object"},
    }
    ad = mcp_tool_to_clear_agent(mcp_tool, call_fn=lambda n, a: "")
    assert ad.input_schema == {"type": "object"}


def test_mcp_tool_to_clear_agent_object_input():
    """支持 mcp.types.Tool-like 对象（含属性而非 dict 键）"""
    mock = MagicMock()
    mock.name = "obj_tool"
    mock.description = "from object"
    mock.inputSchema = {"type": "object", "properties": {}}
    ad = mcp_tool_to_clear_agent(mock, call_fn=lambda n, a: "")
    assert ad.name == "obj_tool"
    assert ad.description == "from object"


def test_mcp_tool_to_clear_agent_missing_fields_defaults():
    ad = mcp_tool_to_clear_agent({"name": "x"}, call_fn=lambda n, a: "")
    assert ad.name == "x"
    assert ad.description == ""
    # MCPToolAdapter 默认空 schema 会 fallback 到 {"type": "object", "properties": {}}
    assert ad.input_schema == {"type": "object", "properties": {}}


# ==================== Section C: clear_agent_tool_to_mcp_schema ====================


def test_clear_agent_tool_to_mcp_schema_calculator():
    schema = clear_agent_tool_to_mcp_schema(CalculatorTool())
    assert "name" in schema
    assert "description" in schema
    assert "inputSchema" in schema
    assert schema["inputSchema"]["type"] == "object"


def test_clear_agent_tool_to_mcp_schema_preserves_function_name():
    schema = clear_agent_tool_to_mcp_schema(CalculatorTool())
    assert schema["name"] == "python_calculator"


def test_clear_agent_tool_to_mcp_schema_roundtrip():
    """ClearAgent → MCP schema → ClearAgent adapter，name 应保留"""
    calc = CalculatorTool()
    mcp_schema = clear_agent_tool_to_mcp_schema(calc)
    ad = mcp_tool_to_clear_agent(mcp_schema, call_fn=lambda n, a: "")
    assert ad.name == calc.name


# ==================== Section D: MCPClient（无 mcp SDK 时的行为） ====================


def test_mcp_client_factory_stdio_constructs_without_imports():
    """connect_stdio 不应触发 mcp import"""
    c = MCPClient.connect_stdio(command="echo", args=["hi"])
    assert c.transport == "stdio"
    assert c.transport_params["command"] == "echo"
    assert c.transport_params["args"] == ["hi"]
    assert c.is_connected is False


def test_mcp_client_factory_sse():
    c = MCPClient.connect_sse(url="http://x/sse", headers={"X": "1"})
    assert c.transport == "sse"
    assert c.transport_params["url"] == "http://x/sse"
    assert c.transport_params["headers"] == {"X": "1"}


def test_mcp_client_list_tools_raises_when_mcp_missing():
    """venv 里没装 mcp → list_tools 抛 ImportError 友好提示"""
    c = MCPClient.connect_stdio(command="x")
    with pytest.raises(ImportError) as exc_info:
        c.list_tools()
    assert "mcp SDK" in str(exc_info.value) or "clear-agent[mcp]" in str(exc_info.value)


def test_mcp_client_call_tool_raises_when_mcp_missing():
    c = MCPClient.connect_stdio(command="x")
    with pytest.raises(ImportError):
        c.call_tool("name", {})


def test_mcp_client_unknown_transport_raises():
    c = MCPClient(transport="bogus", transport_params={})
    # mcp 即便装了，未知 transport 也走 MCPException
    with patch("clear_agent.mcp.client._check_mcp_installed"):
        with pytest.raises(MCPException):
            c.list_tools()


# ==================== Section E: MCPClient with mocked mcp SDK ====================


def test_mcp_client_list_tools_with_fake_session(monkeypatch):
    """模拟整个 _connect 路径，验证 list_tools 的转换逻辑"""

    class FakeMCPTool:
        def __init__(self, name, description, inputSchema):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema

    fake_resp = MagicMock()
    fake_resp.tools = [
        FakeMCPTool("a", "tool a", {"type": "object"}),
        FakeMCPTool("b", "tool b", {"type": "object", "properties": {"x": {"type": "string"}}}),
    ]

    fake_session = MagicMock()
    fake_session.list_tools = AsyncMock(return_value=fake_resp)

    class FakeContextMgr:
        async def __aenter__(self):
            return fake_session
        async def __aexit__(self, *a):
            return False

    c = MCPClient.connect_stdio(command="x")
    # 略过 mcp 装包检查
    monkeypatch.setattr("clear_agent.mcp.client._check_mcp_installed", lambda: None)
    monkeypatch.setattr(c, "_connect", lambda: FakeContextMgr())

    tools = c.list_tools()
    assert len(tools) == 2
    assert tools[0]["name"] == "a"
    assert tools[1]["name"] == "b"
    assert tools[1]["inputSchema"]["properties"]["x"]["type"] == "string"


def test_mcp_client_call_tool_with_fake_session(monkeypatch):
    """模拟 call_tool 返回 CallToolResult.content[TextContent]"""

    class FakeText:
        def __init__(self, text):
            self.text = text

    class FakeResult:
        def __init__(self, contents):
            self.content = contents

    fake_session = MagicMock()
    fake_session.call_tool = AsyncMock(
        return_value=FakeResult([FakeText("hello"), FakeText("world")])
    )

    class FakeContextMgr:
        async def __aenter__(self):
            return fake_session
        async def __aexit__(self, *a):
            return False

    c = MCPClient.connect_stdio(command="x")
    monkeypatch.setattr("clear_agent.mcp.client._check_mcp_installed", lambda: None)
    monkeypatch.setattr(c, "_connect", lambda: FakeContextMgr())

    text = c.call_tool("any", {"a": 1})
    assert text == "hello\nworld"
    fake_session.call_tool.assert_called_once_with("any", {"a": 1})


def test_mcp_client_get_clear_agent_tools(monkeypatch):
    """list_tools 后转 MCPToolAdapter 列表"""

    class FakeMCPTool:
        def __init__(self, name):
            self.name = name
            self.description = f"desc {name}"
            self.inputSchema = {}

    fake_resp = MagicMock()
    fake_resp.tools = [FakeMCPTool("t1"), FakeMCPTool("t2")]
    fake_session = MagicMock()
    fake_session.list_tools = AsyncMock(return_value=fake_resp)

    class FakeContextMgr:
        async def __aenter__(self): return fake_session
        async def __aexit__(self, *a): return False

    c = MCPClient.connect_stdio(command="x")
    monkeypatch.setattr("clear_agent.mcp.client._check_mcp_installed", lambda: None)
    monkeypatch.setattr(c, "_connect", lambda: FakeContextMgr())

    tools = c.get_clear_agent_tools()
    assert len(tools) == 2
    assert all(isinstance(t, MCPToolAdapter) for t in tools)
    assert {t.name for t in tools} == {"t1", "t2"}


def test_mcp_client_register_to(monkeypatch):
    """一行注册到 ClearAgent ToolRegistry"""

    class FakeMCPTool:
        def __init__(self, name):
            self.name = name
            self.description = ""
            self.inputSchema = {}

    fake_resp = MagicMock()
    fake_resp.tools = [FakeMCPTool("alpha"), FakeMCPTool("beta")]
    fake_session = MagicMock()
    fake_session.list_tools = AsyncMock(return_value=fake_resp)

    class FakeContextMgr:
        async def __aenter__(self): return fake_session
        async def __aexit__(self, *a): return False

    c = MCPClient.connect_stdio(command="x")
    monkeypatch.setattr("clear_agent.mcp.client._check_mcp_installed", lambda: None)
    monkeypatch.setattr(c, "_connect", lambda: FakeContextMgr())

    reg = ToolRegistry()
    n = c.register_to(reg)
    assert n == 2
    assert "alpha" in reg.list_tools()
    assert "beta" in reg.list_tools()


# ==================== Section F: MCPServer ====================


def test_mcp_server_list_tools_works_without_mcp_sdk():
    """list_tools 不需要 mcp SDK（只是 schema 转换）"""
    reg = ToolRegistry()
    reg.register_tool(CalculatorTool())
    server = MCPServer(reg, name="test")
    tools = server.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "python_calculator"
    assert "inputSchema" in tools[0]


def test_mcp_server_list_tools_empty_registry():
    reg = ToolRegistry()
    server = MCPServer(reg)
    assert server.list_tools() == []


def test_mcp_server_run_stdio_raises_when_mcp_missing():
    reg = ToolRegistry()
    reg.register_tool(CalculatorTool())
    server = MCPServer(reg)
    with pytest.raises(ImportError):
        server.run(transport="stdio")


def test_mcp_server_run_unknown_transport():
    reg = ToolRegistry()
    server = MCPServer(reg)
    with patch("clear_agent.mcp.server._check_mcp_installed"):
        with pytest.raises(MCPException):
            server.run(transport="bogus")


def test_mcp_server_run_streamable_http_not_implemented():
    reg = ToolRegistry()
    server = MCPServer(reg)
    with patch("clear_agent.mcp.server._check_mcp_installed"):
        with pytest.raises(NotImplementedError):
            server.run(transport="streamable_http")


def test_mcp_server_default_name_and_version():
    server = MCPServer(ToolRegistry())
    assert server.name == "clear-agent"
    assert server.version == "2.0.0"


def test_mcp_server_custom_name():
    server = MCPServer(ToolRegistry(), name="my-tools", version="1.2.3")
    assert server.name == "my-tools"
    assert server.version == "1.2.3"


def test_mcp_server_iter_tools_via_get_all_tools():
    """ToolRegistry.get_all_tools 路径"""
    reg = ToolRegistry()
    reg.register_tool(CalculatorTool())
    server = MCPServer(reg)
    tools = list(server._iter_registered_tools())
    assert len(tools) == 1


def test_mcp_server_find_tool_by_name():
    reg = ToolRegistry()
    reg.register_tool(CalculatorTool())
    server = MCPServer(reg)
    t = server._find_tool_by_name("python_calculator")
    assert t is not None
    assert t.name == "python_calculator"
    assert server._find_tool_by_name("ghost") is None


# ==================== Section G: 顶层导出 ====================


def test_top_level_mcp_imports():
    from clear_agent.mcp import (
        MCPClient,
        MCPException,
        MCPServer,
        MCPToolAdapter,
        clear_agent_tool_to_mcp_schema,
        mcp_tool_to_clear_agent,
    )

    assert MCPClient is not None
    assert MCPServer is not None
    assert MCPToolAdapter is not None
    assert callable(mcp_tool_to_clear_agent)
    assert callable(clear_agent_tool_to_mcp_schema)
