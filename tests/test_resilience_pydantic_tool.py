"""GA-W1 测试 —— Resilience + Pydantic Tool 推导"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from clear_agent.core.resilience import (
    RetryPolicy,
    aretry,
    random_choice,
    retry,
    round_robin,
    with_fallbacks,
    with_fallbacks_async,
    with_retry,
    with_retry_async,
)
from clear_agent.tools.from_pydantic import pydantic_tool, tool_from_pydantic
from clear_agent.tools.response import ToolResponse


# ==================== Section A: RetryPolicy ====================


def test_retry_policy_compute_delay_no_jitter():
    p = RetryPolicy(backoff=1.0, jitter=0.0)
    assert p.compute_delay(1) == 1.0
    assert p.compute_delay(2) == 2.0
    assert p.compute_delay(3) == 4.0
    assert p.compute_delay(4) == 8.0


def test_retry_policy_max_backoff_caps():
    p = RetryPolicy(backoff=1.0, max_backoff=5.0, jitter=0.0)
    assert p.compute_delay(10) == 5.0


def test_retry_policy_jitter_within_range():
    """jitter=0.5 时 delay 应在 [0.5*base, 1.5*base] 之间"""
    p = RetryPolicy(backoff=2.0, jitter=0.5)
    for _ in range(20):
        d = p.compute_delay(1)
        assert 1.0 <= d <= 3.0  # base=2, jitter=0.5 → [1.0, 3.0]


def test_retry_policy_attempt_zero_no_delay():
    p = RetryPolicy(backoff=1.0)
    assert p.compute_delay(0) == 0.0


# ==================== Section B: with_retry / @retry ====================


def test_retry_recovers_on_third_attempt():
    n = {"x": 0}

    @retry(max_attempts=3, backoff=0.001)
    def fn():
        n["x"] += 1
        if n["x"] < 3:
            raise ConnectionError("transient")
        return "ok"

    assert fn() == "ok"
    assert n["x"] == 3


def test_retry_exhausts_raises_last():
    n = {"x": 0}

    @retry(max_attempts=2, backoff=0.001)
    def fn():
        n["x"] += 1
        raise ValueError("nope")

    with pytest.raises(ValueError):
        fn()
    assert n["x"] == 2


def test_retry_does_not_retry_on_unrelated_exception():
    """retry_on=ConnectionError 时 ValueError 应立刻抛"""
    n = {"x": 0}

    @retry(max_attempts=3, backoff=0.001, retry_on=ConnectionError)
    def fn():
        n["x"] += 1
        raise ValueError("not a network error")

    with pytest.raises(ValueError):
        fn()
    assert n["x"] == 1  # 没重试


def test_retry_invokes_on_retry_callback():
    n = {"calls": []}

    def on_retry(attempt, exc, delay):
        n["calls"].append((attempt, type(exc).__name__))

    @retry(max_attempts=3, backoff=0.001, on_retry=on_retry)
    def fn():
        raise ConnectionError("x")

    with pytest.raises(ConnectionError):
        fn()
    # 失败 3 次，触发 2 次 on_retry
    assert len(n["calls"]) == 2


def test_retry_callback_exception_does_not_propagate():
    """on_retry 抛异常不影响主流程"""
    n = {"x": 0}

    def bad_callback(attempt, exc, delay):
        raise RuntimeError("callback bad")

    @retry(max_attempts=2, backoff=0.001, on_retry=bad_callback)
    def fn():
        n["x"] += 1
        raise ConnectionError("real")

    with pytest.raises(ConnectionError):
        fn()


def test_with_retry_wrapper_form():
    n = {"x": 0}

    def f():
        n["x"] += 1
        if n["x"] < 2:
            raise ValueError("y")
        return 42

    safe_f = with_retry(f, max_attempts=2, backoff=0.001)
    assert safe_f() == 42


def test_retry_preserves_args_kwargs():
    @retry(max_attempts=2, backoff=0.001)
    def fn(a, b, c=10):
        return a + b + c

    assert fn(1, 2, c=5) == 8


def test_retry_first_attempt_success_no_sleep():
    """首次成功不应睡眠"""
    t0 = time.time()

    @retry(max_attempts=5, backoff=10.0)
    def fn():
        return "fast"

    assert fn() == "fast"
    assert time.time() - t0 < 0.5  # 不应睡 10s


# ==================== Section C: async retry ====================


def test_aretry_recovers():
    n = {"x": 0}

    @aretry(max_attempts=3, backoff=0.001)
    async def fn():
        n["x"] += 1
        if n["x"] < 2:
            raise ConnectionError("a")
        return "aok"

    assert asyncio.run(fn()) == "aok"


def test_aretry_exhausts():
    @aretry(max_attempts=2, backoff=0.001)
    async def fn():
        raise ValueError("y")

    with pytest.raises(ValueError):
        asyncio.run(fn())


def test_with_retry_async_wrapper():
    async def f():
        return "ok"

    safe = with_retry_async(f, max_attempts=2, backoff=0.001)
    assert asyncio.run(safe()) == "ok"


# ==================== Section D: with_fallbacks ====================


def test_fallbacks_primary_succeeds():
    def primary(x):
        return f"primary: {x}"

    def fb(x):
        return f"fb: {x}"

    safe = with_fallbacks(primary, [fb])
    assert safe(1) == "primary: 1"


def test_fallbacks_uses_first_fallback():
    def primary(x):
        raise RuntimeError("down")

    def fb(x):
        return f"fb: {x}"

    safe = with_fallbacks(primary, [fb])
    assert safe(99) == "fb: 99"


def test_fallbacks_chain_through_multiple():
    def primary(x):
        raise RuntimeError("p")

    def fb1(x):
        raise RuntimeError("f1")

    def fb2(x):
        return f"f2: {x}"

    safe = with_fallbacks(primary, [fb1, fb2])
    assert safe(7) == "f2: 7"


def test_fallbacks_all_fail_raises_last():
    def f_bad(x):
        raise RuntimeError("bad")

    safe = with_fallbacks(f_bad, [f_bad, f_bad])
    with pytest.raises(RuntimeError, match="bad"):
        safe(0)


def test_fallbacks_only_on_specified_exceptions():
    def primary(x):
        raise KeyError("specific")

    def fb(x):
        return "fb"

    safe = with_fallbacks(primary, [fb], fallback_on=ValueError)
    with pytest.raises(KeyError):
        safe(0)


def test_fallbacks_invokes_callback():
    calls = []

    def on_fallback(idx, exc):
        calls.append((idx, type(exc).__name__))

    def primary(x):
        raise RuntimeError("p")

    def fb(x):
        return "ok"

    safe = with_fallbacks(primary, [fb], on_fallback=on_fallback)
    safe(1)
    assert calls == [(0, "RuntimeError")]


def test_async_fallbacks():
    async def primary(x):
        raise RuntimeError("p")

    async def fb(x):
        return f"fb: {x}"

    safe = with_fallbacks_async(primary, [fb])
    assert asyncio.run(safe(5)) == "fb: 5"


# ==================== Section E: 负载均衡 ====================


def test_round_robin_cycles_through():
    calls = []

    def make(i):
        def f():
            calls.append(i)
            return i

        return f

    rr = round_robin([make(0), make(1), make(2)])
    [rr() for _ in range(7)]
    assert calls == [0, 1, 2, 0, 1, 2, 0]


def test_round_robin_empty_raises():
    with pytest.raises(ValueError):
        round_robin([])


def test_random_choice_picks_one():
    def f0():
        return 0

    def f1():
        return 1

    rc = random_choice([f0, f1], seed=42)
    out = [rc() for _ in range(20)]
    # 至少各出现一次（统计上）
    assert 0 in out and 1 in out


def test_random_choice_empty_raises():
    with pytest.raises(ValueError):
        random_choice([])


# ==================== Section F: pydantic_tool ====================


class _AddArgs(BaseModel):
    """Add two integers"""

    a: int = Field(description="first number")
    b: int = Field(description="second number")
    note: str = Field(default="", description="optional note")


def test_pydantic_tool_decorator_basic():
    @pydantic_tool(description="Add a + b")
    def add(args: _AddArgs) -> int:
        return args.a + args.b

    assert add.name == "add"
    assert add.description == "Add a + b"


def test_pydantic_tool_decorator_default_name_from_fn():
    @pydantic_tool()
    def my_tool(args: _AddArgs) -> int:
        """My tool"""
        return args.a + args.b

    assert my_tool.name == "my_tool"
    assert "My tool" in my_tool.description


def test_pydantic_tool_decorator_inferred_schema_from_annotation():
    @pydantic_tool(name="x")
    def fn(args: _AddArgs) -> int:
        return 0

    assert fn.args_schema is _AddArgs


def test_pydantic_tool_explicit_schema_overrides_annotation():
    class OtherArgs(BaseModel):
        x: int

    @pydantic_tool(name="y", args_schema=OtherArgs)
    def fn(args: Any) -> int:
        return 0

    assert fn.args_schema is OtherArgs


def test_pydantic_tool_no_first_param_raises():
    with pytest.raises(ValueError):

        @pydantic_tool(name="bad")
        def fn() -> int:  # type: ignore[no-untyped-def]
            return 0


def test_pydantic_tool_first_param_no_annotation_raises():
    with pytest.raises(ValueError):

        @pydantic_tool(name="bad")
        def fn(args) -> int:  # type: ignore[no-untyped-def]
            return 0


def test_pydantic_tool_first_param_not_basemodel_raises():
    with pytest.raises(ValueError):

        @pydantic_tool(name="bad")
        def fn(args: int) -> int:
            return 0


# ==================== Section G: get_parameters / to_openai_schema ====================


def test_get_parameters_required_and_optional():
    @pydantic_tool(description="d")
    def t(args: _AddArgs) -> int:
        return 0

    params = t.get_parameters()
    by_name = {p.name: p for p in params}
    assert by_name["a"].required is True
    assert by_name["a"].type == "integer"
    assert by_name["a"].description == "first number"
    assert by_name["note"].required is False
    assert by_name["note"].default == ""


def test_to_openai_schema_uses_pydantic_json_schema():
    @pydantic_tool(name="add", description="d")
    def t(args: _AddArgs) -> int:
        return 0

    s = t.to_openai_schema()
    assert s["type"] == "function"
    assert s["function"]["name"] == "add"
    params = s["function"]["parameters"]
    assert params["type"] == "object"
    assert "a" in params["properties"]
    assert "b" in params["properties"]


# ==================== Section H: run / arun ====================


def test_pydantic_tool_run_success():
    @pydantic_tool(description="d")
    def add(args: _AddArgs) -> int:
        return args.a + args.b

    resp = add.run({"a": 3, "b": 4})
    assert resp.status.value == "success"
    assert resp.text == "7"


def test_pydantic_tool_run_with_string_return():
    @pydantic_tool(description="d")
    def echo(args: _AddArgs) -> str:
        return f"sum is {args.a + args.b}"

    resp = echo.run({"a": 1, "b": 2})
    assert resp.text == "sum is 3"


def test_pydantic_tool_run_with_dict_return_includes_data():
    @pydantic_tool(description="d")
    def info(args: _AddArgs) -> dict:
        return {"sum": args.a + args.b, "note": args.note}

    resp = info.run({"a": 1, "b": 2, "note": "hi"})
    assert resp.status.value == "success"
    assert resp.data == {"result": {"sum": 3, "note": "hi"}}


def test_pydantic_tool_run_returns_tool_response_passthrough():
    """run_fn 直接返回 ToolResponse → 不再包装"""

    @pydantic_tool(description="d")
    def t(args: _AddArgs):
        return ToolResponse.success(text="custom", data={"k": "v"})

    resp = t.run({"a": 1, "b": 2})
    assert resp.text == "custom"
    assert resp.data == {"k": "v"}


def test_pydantic_tool_run_invalid_args_returns_error():
    @pydantic_tool(description="d")
    def t(args: _AddArgs) -> int:
        return 0

    resp = t.run({"a": "not_an_int", "b": 2})
    assert resp.status.value == "error"
    assert resp.error_info is not None
    assert resp.error_info.get("code") == "INVALID_ARGS"


def test_pydantic_tool_run_function_failure_returns_error():
    @pydantic_tool(description="d")
    def t(args: _AddArgs) -> int:
        raise RuntimeError("boom")

    resp = t.run({"a": 1, "b": 2})
    assert resp.status.value == "error"
    assert resp.error_info.get("code") == "TOOL_FAILED"


def test_pydantic_tool_async_run_fn():
    @pydantic_tool(name="aadd", description="async add")
    async def aadd(args: _AddArgs) -> str:
        await asyncio.sleep(0.001)
        return str(args.a + args.b)

    resp = aadd.run({"a": 5, "b": 7})
    assert resp.status.value == "success"
    assert resp.text == "12"


def test_pydantic_tool_arun_calls_async_path():
    @pydantic_tool(name="aadd", description="d")
    async def aadd(args: _AddArgs) -> int:
        return args.a + args.b

    resp = asyncio.run(aadd.arun({"a": 1, "b": 1}))
    assert resp.text == "2"


def test_pydantic_tool_validate_args_false_passes_dict():
    """validate_args=False 时 run_fn 接受原始 dict"""
    seen = {}

    def run_fn(args):
        seen["args"] = args
        return "ok"

    t = tool_from_pydantic(
        name="x",
        description="d",
        args_schema=_AddArgs,
        run_fn=run_fn,
        validate_args=False,
    )
    t.run({"a": 1, "b": 2, "extra": "field"})
    assert seen["args"] == {"a": 1, "b": 2, "extra": "field"}


def test_pydantic_tool_registers_to_registry():
    """生成的 tool 可以注册到 ToolRegistry"""
    from clear_agent.tools.registry import ToolRegistry

    @pydantic_tool(name="my_add", description="d")
    def add(args: _AddArgs) -> int:
        return args.a + args.b

    reg = ToolRegistry()
    reg.register_tool(add)
    assert "my_add" in reg.list_tools()


# ==================== Section I: 顶层导入 ====================


def test_top_level_resilience_imports():
    from clear_agent.core.resilience import (
        retry,
        with_retry,
        aretry,
        with_retry_async,
        with_fallbacks,
        with_fallbacks_async,
        round_robin,
        random_choice,
        RetryPolicy,
    )

    assert callable(retry)
    assert callable(with_retry)
    assert RetryPolicy is not None


def test_top_level_pydantic_tool_imports():
    from clear_agent.tools.from_pydantic import (
        pydantic_tool,
        tool_from_pydantic,
    )

    assert callable(pydantic_tool)
    assert callable(tool_from_pydantic)
