"""RC-W3 性能基建测试 —— Callbacks + Parallel + 真异步"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from clear_agent.core.callbacks import (
    HOOK_NAMES,
    BaseCallbackHandler,
    CallbackManager,
    LoggingCallbackHandler,
    MetricsCallbackHandler,
)
from clear_agent.core.parallel import (
    arun_tools_parallel,
    gather_with_concurrency,
    run_tools_parallel,
)
from clear_agent.tools.builtin.calculator import CalculatorTool
from clear_agent.tools.registry import ToolRegistry


# ==================== Section A: BaseCallbackHandler & CallbackManager ====================


class _CountingHandler(BaseCallbackHandler):
    def __init__(self):
        self.events: List[tuple] = []

    def on_llm_start(self, prompts, model=None, **kw):
        self.events.append(("llm_start", model))

    def on_llm_end(self, response, **kw):
        self.events.append(("llm_end", None))

    def on_tool_start(self, tool_name, arguments, **kw):
        self.events.append(("tool_start", tool_name))

    def on_tool_end(self, tool_name, response, **kw):
        self.events.append(("tool_end", tool_name))

    def on_node_start(self, node_name, state, **kw):
        self.events.append(("node_start", node_name))

    def on_node_end(self, node_name, state, **kw):
        self.events.append(("node_end", node_name))


def test_hook_names_complete():
    assert "on_llm_start" in HOOK_NAMES
    assert "on_tool_start" in HOOK_NAMES
    assert "on_node_start" in HOOK_NAMES
    assert "on_retriever_start" in HOOK_NAMES
    assert len(HOOK_NAMES) >= 11


def test_base_handler_default_no_op():
    h = BaseCallbackHandler()
    # 默认 hooks 都是 no-op
    h.on_llm_start([], model="x")
    h.on_tool_start("t", {})
    h.on_node_start("n", {})


def test_callback_manager_basic_fire():
    h = _CountingHandler()
    mgr = CallbackManager()
    mgr.add(h)
    mgr.fire("on_tool_start", tool_name="calc", arguments={"x": 1})
    mgr.fire("on_tool_end", tool_name="calc", response="ok")
    assert h.events == [("tool_start", "calc"), ("tool_end", "calc")]


def test_callback_manager_unknown_hook_raises():
    mgr = CallbackManager()
    with pytest.raises(ValueError):
        mgr.fire("on_bogus_hook")


def test_callback_manager_multiple_handlers():
    h1 = _CountingHandler()
    h2 = _CountingHandler()
    mgr = CallbackManager([h1, h2])
    assert len(mgr) == 2
    mgr.fire("on_tool_start", tool_name="x", arguments={})
    assert h1.events == [("tool_start", "x")]
    assert h2.events == [("tool_start", "x")]


def test_callback_manager_add_dedup():
    h = _CountingHandler()
    mgr = CallbackManager()
    mgr.add(h)
    mgr.add(h)  # 重复 add 不增加
    assert len(mgr) == 1


def test_callback_manager_remove():
    h = _CountingHandler()
    mgr = CallbackManager([h])
    assert mgr.remove(h)
    assert len(mgr) == 0
    assert not mgr.remove(h)  # 第二次返回 False


def test_callback_manager_clear():
    mgr = CallbackManager([_CountingHandler(), _CountingHandler()])
    mgr.clear()
    assert len(mgr) == 0


def test_callback_manager_swallow_handler_errors_by_default():
    """默认 swallow_errors=True，handler 抛错不传播"""

    class BadHandler(BaseCallbackHandler):
        def on_tool_start(self, *a, **kw):
            raise RuntimeError("bad")

    h = _CountingHandler()
    mgr = CallbackManager([BadHandler(), h])
    mgr.fire("on_tool_start", tool_name="x", arguments={})
    # 后注册的 handler 仍能收到事件
    assert h.events == [("tool_start", "x")]


def test_callback_manager_swallow_errors_false_propagates():
    class BadHandler(BaseCallbackHandler):
        def on_tool_start(self, *a, **kw):
            raise RuntimeError("bad")

    mgr = CallbackManager([BadHandler()], swallow_errors=False)
    with pytest.raises(RuntimeError):
        mgr.fire("on_tool_start", tool_name="x", arguments={})


def test_callback_manager_afire_sync_handler():
    """afire 兼容同步 handler"""
    h = _CountingHandler()
    mgr = CallbackManager([h])

    async def run():
        await mgr.afire("on_tool_start", tool_name="x", arguments={})

    asyncio.run(run())
    assert h.events == [("tool_start", "x")]


def test_callback_manager_afire_async_handler():
    class AsyncH(BaseCallbackHandler):
        def __init__(self):
            self.called = False

        async def on_tool_start(self, tool_name, arguments, **kw):
            self.called = True

    h = AsyncH()
    mgr = CallbackManager([h])

    async def run():
        await mgr.afire("on_tool_start", tool_name="x", arguments={})

    asyncio.run(run())
    assert h.called


def test_callback_manager_afire_swallows_async_errors():
    class BadAsync(BaseCallbackHandler):
        async def on_tool_start(self, *a, **kw):
            raise RuntimeError("async bad")

    h = _CountingHandler()
    mgr = CallbackManager([BadAsync(), h])

    async def run():
        await mgr.afire("on_tool_start", tool_name="x", arguments={})

    asyncio.run(run())
    assert h.events == [("tool_start", "x")]


# ==================== Section B: 内置 handlers ====================


def test_logging_handler_does_not_crash():
    h = LoggingCallbackHandler()
    h.on_llm_start([{"role": "user", "content": "hi"}], model="gpt")
    h.on_llm_end(MagicMock(usage={"total_tokens": 100}))
    h.on_llm_error(RuntimeError("x"))
    h.on_tool_start("calc", {"x": 1})
    h.on_tool_end("calc", MagicMock())
    h.on_tool_error("calc", RuntimeError("x"))
    h.on_node_start("n1", {})
    h.on_node_end("n1", {})
    h.on_retriever_start("query")
    h.on_retriever_end([1, 2, 3])


def test_metrics_handler_counts_llm():
    m = MetricsCallbackHandler()
    p = ["msg"]
    m.on_llm_start(p, model="gpt")
    m.on_llm_end(MagicMock(usage={"total_tokens": 50}))
    m.on_llm_start(p, model="gpt")
    m.on_llm_end(MagicMock(usage={"total_tokens": 30}))
    assert m.metrics["llm"]["calls"] == 2
    assert m.metrics["llm"]["total_tokens"] == 80


def test_metrics_handler_counts_tool_errors():
    m = MetricsCallbackHandler()
    m.on_tool_start("calc", {})
    m.on_tool_error("calc", RuntimeError("x"))
    assert m.metrics["tool"]["errors"] == 1
    assert m.metrics["tool"]["by_name"]["calc"]["errors"] == 1


def test_metrics_handler_counts_nodes():
    m = MetricsCallbackHandler()
    m.on_node_start("a", {})
    m.on_node_end("a", {})
    m.on_node_start("b", {})
    m.on_node_end("b", {})
    m.on_node_start("a", {})
    m.on_node_end("a", {})
    assert m.metrics["node"]["calls"] == 3
    assert m.metrics["node"]["by_name"]["a"] == 2
    assert m.metrics["node"]["by_name"]["b"] == 1


def test_metrics_handler_counts_retriever():
    m = MetricsCallbackHandler()
    m.on_retriever_start("query")
    m.on_retriever_end([1, 2, 3, 4])
    m.on_retriever_start("q2")
    m.on_retriever_end([5])
    assert m.metrics["retriever"]["calls"] == 2
    assert m.metrics["retriever"]["total_hits"] == 5


def test_metrics_handler_reset():
    m = MetricsCallbackHandler()
    m.on_tool_start("x", {})
    m.reset()
    assert m.metrics["tool"]["calls"] == 0


# ==================== Section C: run_tools_parallel ====================


class _FakeTC:
    def __init__(self, id, name, arguments):
        self.id = id
        self.name = name
        self.arguments = arguments


def _make_calc_registry():
    reg = ToolRegistry()
    reg.register_tool(CalculatorTool())
    return reg


def test_parallel_empty_returns_empty():
    reg = _make_calc_registry()
    assert run_tools_parallel([], reg) == []


def test_parallel_single_tool_call():
    reg = _make_calc_registry()
    tcs = [_FakeTC("1", "python_calculator", json.dumps({"expression": "2+2"}))]
    out = run_tools_parallel(tcs, reg, max_workers=1)
    assert len(out) == 1
    assert "4" in out[0]["content"]
    assert out[0]["error"] is None
    assert out[0]["tool_call_id"] == "1"


def test_parallel_multiple_tool_calls_preserves_order():
    reg = _make_calc_registry()
    tcs = [
        _FakeTC(str(i), "python_calculator", json.dumps({"expression": f"{i}+{i}"}))
        for i in range(5)
    ]
    out = run_tools_parallel(tcs, reg, max_workers=4)
    assert len(out) == 5
    # 顺序保留
    assert [r["tool_call_id"] for r in out] == ["0", "1", "2", "3", "4"]


def test_parallel_unknown_tool_returns_error():
    reg = _make_calc_registry()
    tc = _FakeTC("x", "ghost_tool", "{}")
    out = run_tools_parallel([tc], reg)
    assert out[0]["error"] == "TOOL_NOT_FOUND"


def test_parallel_invalid_json_returns_error():
    reg = _make_calc_registry()
    tc = _FakeTC("x", "python_calculator", "{invalid")
    out = run_tools_parallel([tc], reg)
    assert out[0]["error"] == "JSON_DECODE_ERROR"


def test_parallel_dict_tool_calls():
    """支持 dict 形式的 tool_call"""
    reg = _make_calc_registry()
    tc_dict = {
        "id": "x",
        "name": "python_calculator",
        "arguments": json.dumps({"expression": "5*5"}),
    }
    out = run_tools_parallel([tc_dict], reg)
    assert "25" in out[0]["content"]


def test_parallel_isolates_one_failure_among_many():
    """一个 tool_call 失败不影响其他"""
    reg = _make_calc_registry()
    tcs = [
        _FakeTC("ok", "python_calculator", json.dumps({"expression": "1+1"})),
        _FakeTC("bad", "ghost_tool", "{}"),
        _FakeTC("ok2", "python_calculator", json.dumps({"expression": "2+2"})),
    ]
    out = run_tools_parallel(tcs, reg)
    by_id = {r["tool_call_id"]: r for r in out}
    assert by_id["ok"]["error"] is None
    assert by_id["bad"]["error"] == "TOOL_NOT_FOUND"
    assert by_id["ok2"]["error"] is None


def test_parallel_max_workers_one_falls_back_to_serial():
    """max_workers=1 走串行，结果一致"""
    reg = _make_calc_registry()
    tcs = [
        _FakeTC(str(i), "python_calculator", json.dumps({"expression": f"{i}+1"}))
        for i in range(3)
    ]
    out = run_tools_parallel(tcs, reg, max_workers=1)
    assert len(out) == 3
    assert all(r["error"] is None for r in out)


# ==================== Section D: arun_tools_parallel ====================


def test_arun_empty_returns_empty():
    reg = _make_calc_registry()
    assert asyncio.run(arun_tools_parallel([], reg)) == []


def test_arun_basic():
    reg = _make_calc_registry()
    tcs = [
        _FakeTC(str(i), "python_calculator", json.dumps({"expression": f"{i}*2"}))
        for i in range(3)
    ]
    out = asyncio.run(arun_tools_parallel(tcs, reg, max_concurrency=3))
    assert len(out) == 3
    assert all(r["error"] is None for r in out)


def test_arun_unknown_tool():
    reg = _make_calc_registry()
    tc = _FakeTC("x", "ghost", "{}")
    out = asyncio.run(arun_tools_parallel([tc], reg))
    assert out[0]["error"] == "TOOL_NOT_FOUND"


def test_arun_invalid_json():
    reg = _make_calc_registry()
    tc = _FakeTC("x", "python_calculator", "{bad")
    out = asyncio.run(arun_tools_parallel([tc], reg))
    assert out[0]["error"] == "JSON_DECODE_ERROR"


# ==================== Section E: gather_with_concurrency ====================


def test_gather_with_concurrency_runs_all():
    async def task(i):
        await asyncio.sleep(0.001)
        return i * 2

    coros = [task(i) for i in range(5)]
    out = asyncio.run(gather_with_concurrency(coros, max_concurrency=2))
    assert out == [0, 2, 4, 6, 8]


def test_gather_with_concurrency_limits_parallelism():
    """同一时刻最多 max_concurrency 个 coroutine 在跑"""
    in_flight = {"current": 0, "max": 0}

    async def task(i):
        in_flight["current"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["current"])
        await asyncio.sleep(0.01)
        in_flight["current"] -= 1
        return i

    coros = [task(i) for i in range(10)]
    asyncio.run(gather_with_concurrency(coros, max_concurrency=3))
    # 验证从未超过 3
    assert in_flight["max"] <= 3


# ==================== Section F: ClearAgentLLM 真异步路径 ====================


def test_clear_agent_llm_ainvoke_uses_adapter_async_when_available():
    """ClearAgentLLM.ainvoke 优先调 adapter.ainvoke_async"""
    from clear_agent import ClearAgentLLM
    from clear_agent.core.llm_response import LLMResponse

    llm = ClearAgentLLM(
        model="gpt-4o", api_key="x", base_url="https://api.openai.com/v1"
    )

    fake_resp = LLMResponse(content="async response", model="gpt-4o", usage={})
    fake_async_fn = AsyncMock(return_value=fake_resp)
    llm._adapter.ainvoke_async = fake_async_fn  # type: ignore[attr-defined]

    out = asyncio.run(llm.ainvoke([{"role": "user", "content": "hi"}]))
    assert out.content == "async response"
    fake_async_fn.assert_called_once()


def test_clear_agent_llm_ainvoke_with_tools_uses_adapter_async():
    from clear_agent import ClearAgentLLM
    from clear_agent.core.llm_response import LLMToolResponse

    llm = ClearAgentLLM(
        model="gpt-4o", api_key="x", base_url="https://api.openai.com/v1"
    )
    fake_resp = LLMToolResponse(content="tool resp", tool_calls=[], model="gpt-4o", usage={})
    fake_async_fn = AsyncMock(return_value=fake_resp)
    llm._adapter.ainvoke_with_tools_async = fake_async_fn  # type: ignore[attr-defined]

    out = asyncio.run(
        llm.ainvoke_with_tools(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "x"}}],
        )
    )
    assert out.content == "tool resp"
    fake_async_fn.assert_called_once()


def test_clear_agent_llm_ainvoke_falls_back_to_thread_pool():
    """adapter 没 ainvoke_async → 回退到线程池包装"""
    from clear_agent import ClearAgentLLM
    from clear_agent.core.llm_response import LLMResponse

    llm = ClearAgentLLM(
        model="custom", api_key="x", base_url="https://example/v1"
    )
    # 模拟 adapter 不带 ainvoke_async（不同 base_url 走的是 OpenAIAdapter，强制移除）
    if hasattr(llm._adapter, "ainvoke_async"):
        try:
            delattr(llm._adapter, "ainvoke_async")
        except AttributeError:
            # 是类方法不能删 → monkey-patch 设为 None 触发 callable 判定 fail
            llm._adapter.ainvoke_async = None  # type: ignore[attr-defined]

    fake_sync = MagicMock(
        return_value=LLMResponse(content="sync response", model="custom", usage={})
    )
    llm.invoke = fake_sync  # type: ignore[method-assign]

    out = asyncio.run(llm.ainvoke([{"role": "user", "content": "hi"}]))
    assert out.content == "sync response"
    fake_sync.assert_called_once()


def test_clear_agent_llm_ainvoke_with_tools_fallback_preserves_call_kwargs():
    from clear_agent import ClearAgentLLM
    from clear_agent.core.llm_response import LLMToolResponse

    llm = ClearAgentLLM(
        model="custom",
        api_key="x",
        base_url="https://example/v1",
        temperature=0.4,
        max_tokens=55,
    )
    llm._adapter.ainvoke_with_tools_async = None  # type: ignore[attr-defined]

    fake_resp = LLMToolResponse(content="ok", tool_calls=[], model="custom", usage={})
    fake_sync = MagicMock(return_value=fake_resp)
    llm.invoke_with_tools = fake_sync  # type: ignore[method-assign]

    out = asyncio.run(
        llm.ainvoke_with_tools(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "x"}}],
            tool_choice="required",
            temperature=0.1,
            max_tokens=77,
        )
    )

    assert out.content == "ok"
    call_kwargs = fake_sync.call_args.kwargs
    assert call_kwargs["tool_choice"] == "required"
    assert call_kwargs["temperature"] == 0.1
    assert call_kwargs["max_tokens"] == 77


def test_clear_agent_llm_stream_invoke_uses_default_temperature():
    from clear_agent import ClearAgentLLM

    llm = ClearAgentLLM(
        model="custom",
        api_key="x",
        base_url="https://example/v1",
        temperature=0.25,
    )
    llm._adapter.stream_invoke = MagicMock(return_value=iter(["a", "b"]))

    assert list(llm.stream_invoke([{"role": "user", "content": "hi"}])) == ["a", "b"]

    assert llm._adapter.stream_invoke.call_args.kwargs["temperature"] == 0.25


# ==================== Section G: 顶层导入 ====================


def test_top_level_callbacks_and_parallel_imports():
    from clear_agent.core.callbacks import (
        BaseCallbackHandler,
        CallbackManager,
        LoggingCallbackHandler,
        MetricsCallbackHandler,
    )
    from clear_agent.core.parallel import (
        arun_tools_parallel,
        gather_with_concurrency,
        run_tools_parallel,
    )

    assert callable(run_tools_parallel)
    assert callable(arun_tools_parallel)
    assert callable(gather_with_concurrency)
    assert BaseCallbackHandler is not None
    assert CallbackManager is not None
