"""LCEL-lite Runnable + Trace 训练数据导出 测试"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from clear_agent.core.runnable import (
    Runnable,
    RunnableAdapter,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnableSequence,
    assign,
    parser_json,
    parser_str,
    passthrough,
    prompt,
)
from clear_agent.observability.trace_export import (
    export_to_dpo_pairs,
    export_to_sft_jsonl,
    export_traces_to_sft_jsonl,
    read_trace_events,
)


# ==================== Section A: Runnable 基础 ====================


def test_runnable_lambda_sync():
    r = RunnableLambda(lambda x: x * 2)
    assert r.invoke(5) == 10


def test_runnable_lambda_async_via_invoke():
    """async fn 被同步 invoke 调用 → asyncio.run 包装"""

    async def afn(x):
        return x + 100

    r = RunnableLambda(afn)
    assert r.invoke(5) == 105


def test_runnable_lambda_ainvoke():
    async def afn(x):
        return x * 3

    r = RunnableLambda(afn)
    assert asyncio.run(r.ainvoke(4)) == 12


def test_runnable_lambda_repr_contains_name():
    r = RunnableLambda(lambda x: x, name="my_step")
    assert "my_step" in repr(r)


def test_runnable_adapter_uses_target_invoke():
    target = MagicMock()
    target.invoke.return_value = "result"
    r = RunnableAdapter(target)
    assert r.invoke("input") == "result"
    target.invoke.assert_called_with("input")


def test_runnable_adapter_async_uses_target_ainvoke():
    """target 有 ainvoke → 优先用"""
    from unittest.mock import AsyncMock

    target = MagicMock()
    target.ainvoke = AsyncMock(return_value="async_result")
    r = RunnableAdapter(target)
    assert asyncio.run(r.ainvoke("x")) == "async_result"


# ==================== Section B: Pipe 操作符 ====================


def test_pipe_creates_sequence():
    r1 = RunnableLambda(lambda x: x + 1)
    r2 = RunnableLambda(lambda x: x * 2)
    chain = r1 | r2
    assert isinstance(chain, RunnableSequence)
    assert chain.invoke(5) == 12  # (5+1)*2


def test_pipe_three_steps():
    chain = (
        RunnableLambda(lambda x: x + 1)
        | RunnableLambda(lambda x: x * 2)
        | RunnableLambda(lambda x: x - 3)
    )
    assert chain.invoke(5) == 9  # ((5+1)*2)-3


def test_pipe_with_callable():
    chain = RunnableLambda(lambda x: x + 1) | (lambda x: x * 10)
    assert chain.invoke(5) == 60


def test_pipe_left_with_callable():
    chain = (lambda x: x + 1) | RunnableLambda(lambda x: x * 10)
    assert chain.invoke(5) == 60


def test_pipe_with_invoke_object():
    target = MagicMock()
    target.invoke.return_value = "wrapped"
    chain = RunnableLambda(lambda x: x) | target
    assert chain.invoke("input") == "wrapped"


def test_sequence_flattens_in_pipe():
    """seq | r 不会嵌套 RunnableSequence"""
    seq = RunnableLambda(lambda x: x) | RunnableLambda(lambda x: x)
    extended = seq | RunnableLambda(lambda x: x)
    assert isinstance(extended, RunnableSequence)
    assert len(extended.steps) == 3


def test_unsupported_pipe_target_raises():
    with pytest.raises(TypeError):
        RunnableLambda(lambda x: x) | 42  # 整数无法包成 Runnable


# ==================== Section C: RunnableSequence ====================


def test_sequence_empty_raises():
    with pytest.raises(ValueError):
        RunnableSequence([])


def test_sequence_async():
    async def af(x):
        return x + 1

    seq = RunnableSequence([RunnableLambda(af), RunnableLambda(lambda x: x * 2)])
    assert asyncio.run(seq.ainvoke(3)) == 8


# ==================== Section D: RunnableParallel ====================


def test_parallel_basic():
    p = RunnableParallel({"a": lambda x: x + 1, "b": lambda x: x * 2})
    assert p.invoke(5) == {"a": 6, "b": 10}


def test_parallel_async():
    async def af1(x):
        return x + 1

    async def af2(x):
        return x * 2

    p = RunnableParallel({"a": af1, "b": af2})
    out = asyncio.run(p.ainvoke(5))
    assert out == {"a": 6, "b": 10}


def test_parallel_with_runnable_values():
    p = RunnableParallel({"x": RunnableLambda(lambda x: x), "y": passthrough()})
    assert p.invoke("hi") == {"x": "hi", "y": "hi"}


# ==================== Section E: RunnableBranch ====================


def test_branch_first_match_wins():
    br = RunnableBranch(
        [
            (lambda x: x < 0, lambda x: "negative"),
            (lambda x: x == 0, lambda x: "zero"),
            (lambda x: x > 0, lambda x: "positive"),
        ],
        default=lambda x: "n/a",
    )
    assert br.invoke(-5) == "negative"
    assert br.invoke(0) == "zero"
    assert br.invoke(7) == "positive"


def test_branch_default_when_no_match():
    br = RunnableBranch(
        [(lambda x: False, lambda x: "never")],
        default=lambda x: "default",
    )
    assert br.invoke(42) == "default"


def test_branch_predicate_exception_skipped():
    """predicate 抛异常 → 跳过该分支，继续往下试"""

    def bad_pred(x):
        raise RuntimeError("oops")

    br = RunnableBranch(
        [
            (bad_pred, lambda x: "should_not_run"),
            (lambda x: True, lambda x: "ok"),
        ],
        default=lambda x: "default",
    )
    assert br.invoke(1) == "ok"


def test_branch_async():
    br = RunnableBranch(
        [(lambda x: x > 0, lambda x: x * 2)],
        default=lambda x: 0,
    )
    assert asyncio.run(br.ainvoke(5)) == 10
    assert asyncio.run(br.ainvoke(-1)) == 0


# ==================== Section F: 便捷 helpers ====================


def test_prompt_with_dict_input():
    p = prompt("Hello {name}, age {age}")
    assert p.invoke({"name": "Alice", "age": 30}) == "Hello Alice, age 30"


def test_prompt_with_string_input():
    """单 {input} 占位符可吃字符串"""
    p = prompt("got: {input}")
    assert p.invoke("xxx") == "got: xxx"


def test_parser_str_extracts_content():
    fake = MagicMock()
    fake.content = "the answer"
    assert parser_str().invoke(fake) == "the answer"


def test_parser_str_fallback_to_str():
    """没有 content 属性 → str(obj)"""
    assert parser_str().invoke(42) == "42"


def test_parser_json_basic():
    fake = MagicMock()
    fake.content = '{"a": 1}'
    assert parser_json().invoke(fake) == {"a": 1}


def test_parser_json_with_code_fence():
    fake = MagicMock()
    fake.content = '```json\n{"x": 9}\n```'
    assert parser_json().invoke(fake) == {"x": 9}


def test_parser_json_invalid_raises_value_error():
    fake = MagicMock()
    fake.content = "not json"
    with pytest.raises(ValueError):
        parser_json().invoke(fake)


def test_passthrough():
    assert passthrough().invoke("anything") == "anything"


def test_assign_adds_fields():
    a = assign(b=99, c=lambda d: d["a"] + 1)
    assert a.invoke({"a": 1}) == {"a": 1, "b": 99, "c": 2}


def test_assign_non_dict_raises():
    with pytest.raises(TypeError):
        assign(b=1).invoke("not a dict")


# ==================== Section G: 端到端管道 ====================


def test_chain_prompt_lambda_parser():
    chain = (
        prompt("Q: {question}")
        | RunnableLambda(lambda s: s + " | A: 42")
    )
    out = chain.invoke({"question": "the answer?"})
    assert out == "Q: the answer? | A: 42"


def test_chain_with_llm_like_object():
    """模拟一个 LLM-like 对象"""

    class FakeLLM:
        def invoke(self, msg):
            r = MagicMock()
            r.content = f"echo: {msg}"
            return r

    chain = prompt("query: {q}") | FakeLLM() | parser_str()
    assert chain.invoke({"q": "hi"}) == "echo: query: hi"


# ==================== Section H: Trace 导出 ====================


def _write_trace(p: Path, events: list) -> None:
    p.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n",
        encoding="utf-8",
    )


def test_read_trace_events_basic(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    _write_trace(p, [{"event_type": "x", "data": 1}, {"event_type": "y"}])
    events = list(read_trace_events(p))
    assert len(events) == 2
    assert events[0]["event_type"] == "x"


def test_read_trace_events_skips_blank_and_invalid(tmp_path: Path):
    p = tmp_path / "t.jsonl"
    p.write_text(
        '{"event_type": "ok"}\n\n{not json\n{"event_type": "ok2"}\n',
        encoding="utf-8",
    )
    events = list(read_trace_events(p))
    # 空行跳过 + 非法行警告但不中断
    assert len(events) == 2
    assert events[0]["event_type"] == "ok"
    assert events[1]["event_type"] == "ok2"


def test_read_trace_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        list(read_trace_events(tmp_path / "nope.jsonl"))


def test_export_sft_basic(tmp_path: Path):
    trace = tmp_path / "t.jsonl"
    _write_trace(
        trace,
        [
            {"event_type": "user_input", "data": {"content": "Q?"}},
            {"event_type": "agent_response", "data": {"content": "A!"}},
        ],
    )
    out = tmp_path / "sft.jsonl"
    n = export_to_sft_jsonl(trace, out)
    assert n == 1
    sample = json.loads(out.read_text().strip())
    assert sample["messages"] == [
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "A!"},
    ]


def test_export_sft_multiple_turns(tmp_path: Path):
    trace = tmp_path / "t.jsonl"
    _write_trace(
        trace,
        [
            {"event_type": "user_input", "data": {"content": "Q1"}},
            {"event_type": "agent_response", "data": {"content": "A1"}},
            {"event_type": "user_input", "data": {"content": "Q2"}},
            {"event_type": "agent_response", "data": {"content": "A2"}},
        ],
    )
    out = tmp_path / "sft.jsonl"
    export_to_sft_jsonl(trace, out)
    sample = json.loads(out.read_text().strip())
    assert len(sample["messages"]) == 4


def test_export_sft_skips_when_error_event(tmp_path: Path):
    trace = tmp_path / "t.jsonl"
    _write_trace(
        trace,
        [
            {"event_type": "user_input", "data": {"content": "Q"}},
            {"event_type": "error", "data": {}},
        ],
    )
    out = tmp_path / "sft.jsonl"
    n = export_to_sft_jsonl(trace, out, only_successful=True)
    assert n == 0
    assert not out.exists() or out.stat().st_size == 0


def test_export_sft_keeps_when_only_successful_false(tmp_path: Path):
    trace = tmp_path / "t.jsonl"
    _write_trace(
        trace,
        [
            {"event_type": "user_input", "data": {"content": "Q"}},
            {"event_type": "error", "data": {}},
            {"event_type": "agent_response", "data": {"content": "recovered"}},
        ],
    )
    out = tmp_path / "sft.jsonl"
    n = export_to_sft_jsonl(trace, out, only_successful=False)
    assert n == 1


def test_export_sft_min_messages_threshold(tmp_path: Path):
    """只有 user 没 assistant → 不写入"""
    trace = tmp_path / "t.jsonl"
    _write_trace(trace, [{"event_type": "user_input", "data": {"content": "Q"}}])
    out = tmp_path / "sft.jsonl"
    n = export_to_sft_jsonl(trace, out, min_messages=2)
    assert n == 0


def test_export_traces_to_sft_jsonl_batch(tmp_path: Path):
    trace1 = tmp_path / "t1.jsonl"
    trace2 = tmp_path / "t2.jsonl"
    _write_trace(
        trace1,
        [
            {"event_type": "user_input", "data": {"content": "Q1"}},
            {"event_type": "agent_response", "data": {"content": "A1"}},
        ],
    )
    _write_trace(
        trace2,
        [
            {"event_type": "user_input", "data": {"content": "Q2"}},
            {"event_type": "agent_response", "data": {"content": "A2"}},
        ],
    )
    out = tmp_path / "all.jsonl"
    n = export_traces_to_sft_jsonl([trace1, trace2], out)
    assert n == 2
    lines = [json.loads(line) for line in out.read_text().splitlines() if line]
    assert len(lines) == 2


def test_export_dpo_pairs_basic(tmp_path: Path):
    pass_t = tmp_path / "pass.jsonl"
    fail_t = tmp_path / "fail.jsonl"
    _write_trace(
        pass_t,
        [
            {"event_type": "user_input", "data": {"content": "Q?"}},
            {"event_type": "agent_response", "data": {"content": "GOOD"}},
        ],
    )
    _write_trace(
        fail_t,
        [
            {"event_type": "user_input", "data": {"content": "Q?"}},
            {"event_type": "agent_response", "data": {"content": "BAD"}},
        ],
    )
    out = tmp_path / "dpo.jsonl"
    n = export_to_dpo_pairs([pass_t], [fail_t], out)
    assert n == 1
    pair = json.loads(out.read_text().strip())
    assert pair["prompt"] == "Q?"
    assert pair["chosen"] == "GOOD"
    assert pair["rejected"] == "BAD"


def test_export_dpo_pairs_unequal_lengths(tmp_path: Path):
    """zip 短的一方决定长度"""
    pass_t = tmp_path / "p.jsonl"
    fail_t1 = tmp_path / "f1.jsonl"
    fail_t2 = tmp_path / "f2.jsonl"
    _write_trace(
        pass_t,
        [
            {"event_type": "user_input", "data": {"content": "Q"}},
            {"event_type": "agent_response", "data": {"content": "G"}},
        ],
    )
    for fp in (fail_t1, fail_t2):
        _write_trace(
            fp,
            [
                {"event_type": "user_input", "data": {"content": "Q"}},
                {"event_type": "agent_response", "data": {"content": "B"}},
            ],
        )
    out = tmp_path / "dpo.jsonl"
    n = export_to_dpo_pairs([pass_t], [fail_t1, fail_t2], out)
    # zip 会取最短，结果 1 对
    assert n == 1


def test_export_dpo_pairs_skips_incomplete(tmp_path: Path):
    """trace 没 assistant 消息 → 跳过该对"""
    pass_t = tmp_path / "p.jsonl"
    fail_t = tmp_path / "f.jsonl"
    _write_trace(pass_t, [{"event_type": "user_input", "data": {"content": "Q"}}])
    _write_trace(
        fail_t,
        [
            {"event_type": "user_input", "data": {"content": "Q"}},
            {"event_type": "agent_response", "data": {"content": "B"}},
        ],
    )
    out = tmp_path / "dpo.jsonl"
    n = export_to_dpo_pairs([pass_t], [fail_t], out)
    assert n == 0


# ==================== Section I: 顶层导入 ====================


def test_top_level_runnable_imports():
    from clear_agent.core.runnable import (
        Runnable,
        RunnableLambda,
        RunnableSequence,
        RunnableParallel,
        RunnableBranch,
    )

    for x in (Runnable, RunnableLambda, RunnableSequence, RunnableParallel, RunnableBranch):
        assert x is not None


def test_top_level_trace_export_imports():
    from clear_agent.observability.trace_export import (
        read_trace_events,
        export_to_sft_jsonl,
        export_traces_to_sft_jsonl,
        export_to_dpo_pairs,
    )

    for fn in (
        read_trace_events,
        export_to_sft_jsonl,
        export_traces_to_sft_jsonl,
        export_to_dpo_pairs,
    ):
        assert callable(fn)
