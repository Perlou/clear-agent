"""Eval-harness 测试

不调用真实 LLM —— 通过 mock target / mock evaluator 验证：
- Dataset.from_jsonl / from_list / filter / sample / take / 切片
- 四种 Evaluator: ExactMatch / Contains / LLMAsJudge / Custom
- run_eval 同步与并发流程
- markdown 报告与 results.jsonl 落盘
- example 抛异常时不打断整批，error 字段记录
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel

from clear_agent.eval import (
    Contains,
    Custom,
    Dataset,
    EvalResult,
    Example,
    ExactMatch,
    LLMAsJudge,
    run_eval,
)
from clear_agent.eval.runner import _default_extract_predicted, _format_report_md


# ==================== Section A: Dataset ====================


def test_dataset_from_list_basic():
    ds = Dataset.from_list(
        [
            {"id": "1", "input": "Q1", "expected": "A1"},
            {"id": "2", "input": "Q2", "expected": "A2", "tags": ["x"]},
        ]
    )
    assert len(ds) == 2
    assert ds[0].id == "1"
    assert ds[1].tags == ["x"]


def test_dataset_from_list_missing_required_raises():
    with pytest.raises(ValueError):
        Dataset.from_list([{"id": "1"}])  # 缺 input


def test_dataset_from_jsonl(tmp_path: Path):
    p = tmp_path / "ds.jsonl"
    p.write_text(
        '{"id":"1","input":"Q1","expected":"A1"}\n'
        '\n'
        '# this line is a comment\n'
        '{"id":"2","input":"Q2","expected":"A2","tags":["x"]}\n',
        encoding="utf-8",
    )
    ds = Dataset.from_jsonl(p)
    assert len(ds) == 2  # 空行和注释被跳过
    assert ds[1].tags == ["x"]


def test_dataset_from_jsonl_invalid_raises(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        Dataset.from_jsonl(p)


def test_dataset_from_jsonl_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        Dataset.from_jsonl("/nonexistent/path.jsonl")


def test_dataset_filter():
    ds = Dataset.from_list(
        [
            {"id": "1", "input": "x", "tags": ["a"]},
            {"id": "2", "input": "y", "tags": ["b"]},
            {"id": "3", "input": "z", "tags": ["a", "b"]},
        ]
    )
    filtered = ds.filter(lambda ex: "a" in ex.tags)
    assert len(filtered) == 2
    assert {ex.id for ex in filtered} == {"1", "3"}


def test_dataset_sample_deterministic_with_seed():
    ds = Dataset.from_list([{"id": str(i), "input": i} for i in range(20)])
    s1 = ds.sample(5, seed=42)
    s2 = ds.sample(5, seed=42)
    assert [ex.id for ex in s1] == [ex.id for ex in s2]
    assert len(s1) == 5


def test_dataset_sample_n_larger_than_size():
    ds = Dataset.from_list([{"id": "1", "input": "x"}])
    s = ds.sample(10, seed=0)
    assert len(s) == 1


def test_dataset_take_and_slice():
    ds = Dataset.from_list([{"id": str(i), "input": i} for i in range(10)])
    assert len(ds.take(3)) == 3
    sliced = ds[2:5]
    assert isinstance(sliced, Dataset)
    assert len(sliced) == 3
    assert sliced[0].id == "2"


def test_dataset_to_jsonl_roundtrip(tmp_path: Path):
    ds = Dataset.from_list(
        [
            {"id": "1", "input": "x", "expected": "y", "tags": ["t"], "metadata": {"k": "v"}},
        ]
    )
    p = tmp_path / "out.jsonl"
    ds.to_jsonl(p)
    ds2 = Dataset.from_jsonl(p)
    assert ds2[0].metadata == {"k": "v"}


# ==================== Section B: ExactMatch ====================


def test_exact_match_case_insensitive_default():
    e = ExactMatch()
    r = e.evaluate("Paris", "paris", Example(id="x", input=""))
    assert r.pass_ and r.score == 1.0


def test_exact_match_case_sensitive():
    e = ExactMatch(case_sensitive=True)
    r = e.evaluate("Paris", "paris", Example(id="x", input=""))
    assert not r.pass_


def test_exact_match_whitespace_normalize():
    e = ExactMatch()
    r = e.evaluate("  hello   world  ", "hello world", Example(id="x", input=""))
    assert r.pass_


def test_exact_match_no_normalize():
    e = ExactMatch(normalize_whitespace=False)
    r = e.evaluate(" hello ", "hello", Example(id="x", input=""))
    assert not r.pass_


def test_exact_match_mismatch_provides_feedback():
    e = ExactMatch()
    r = e.evaluate("Lyon", "Paris", Example(id="x", input=""))
    assert "Expected" in r.feedback


# ==================== Section C: Contains ====================


def test_contains_basic_substring_match():
    c = Contains()
    r = c.evaluate("The capital is Paris.", "Paris", Example(id="x", input=""))
    assert r.pass_


def test_contains_negative_case():
    c = Contains()
    r = c.evaluate("The capital is Lyon.", "Paris", Example(id="x", input=""))
    assert not r.pass_
    assert "not found" in r.feedback


def test_contains_case_insensitive_default():
    c = Contains()
    r = c.evaluate("PARIS is in France", "paris", Example(id="x", input=""))
    assert r.pass_


def test_contains_empty_expected_returns_zero():
    c = Contains()
    r = c.evaluate("anything", "", Example(id="x", input=""))
    assert not r.pass_


# ==================== Section D: Custom ====================


def test_custom_returns_eval_result():
    def fn(predicted, expected, example):
        return EvalResult(score=0.7, pass_=True, feedback="ok")

    c = Custom(fn)
    r = c.evaluate("p", "e", Example(id="x", input=""))
    assert r.score == 0.7
    assert r.pass_


def test_custom_returns_float():
    c = Custom(lambda p, e, ex: 0.85, pass_threshold=0.8)
    r = c.evaluate("p", "e", Example(id="x", input=""))
    assert r.score == 0.85
    assert r.pass_


def test_custom_returns_float_below_threshold():
    c = Custom(lambda p, e, ex: 0.3, pass_threshold=0.5)
    r = c.evaluate("p", "e", Example(id="x", input=""))
    assert not r.pass_


def test_custom_returns_bool():
    c = Custom(lambda p, e, ex: True)
    r = c.evaluate("p", "e", Example(id="x", input=""))
    assert r.score == 1.0


def test_custom_invalid_return_raises():
    c = Custom(lambda p, e, ex: "not a valid type")
    with pytest.raises(TypeError):
        c.evaluate("p", "e", Example(id="x", input=""))


def test_custom_name_property():
    def my_fn(p, e, ex):
        return 1.0

    c = Custom(my_fn)
    assert "my_fn" in c.name


# ==================== Section E: LLMAsJudge ====================


class _MockJudgeLLM:
    """模拟 ClearAgentLLM；with_structured_output 返回一个可控的 fake StructuredLLM"""

    def __init__(self, scripted_outputs: List[Any]):
        self._outputs = list(scripted_outputs)
        self.calls: List[Any] = []
        self.model = "mock-judge"
        self.base_url = "https://mock/v1"

    def with_structured_output(self, schema, **kwargs):
        outer = self
        sched = self._outputs

        class _FakeStructured:
            def invoke(self_inner, messages, **kw):
                outer.calls.append({"messages": messages, "schema": schema})
                if not sched:
                    raise AssertionError("LLMAsJudge: no more scripted outputs")
                return sched.pop(0)

        return _FakeStructured()


def test_llm_as_judge_basic():
    class JudgeOut(BaseModel):
        score: float
        reasoning: str

    out = JudgeOut(score=0.8, reasoning="mostly correct")
    judge_llm = _MockJudgeLLM([out])
    j = LLMAsJudge(
        llm=judge_llm,
        rubric="Score 1.0 if accurate",
        output_schema=JudgeOut,
        pass_threshold=0.7,
    )

    r = j.evaluate(
        "predicted", "expected", Example(id="x", input="What?")
    )
    assert r.score == 0.8
    assert r.pass_
    assert r.feedback == "mostly correct"
    # 调用 prompt 含 rubric / question / expected / predicted
    p = judge_llm.calls[0]["messages"][0]["content"]
    assert "Score 1.0 if accurate" in p
    assert "What?" in p
    assert "expected" in p
    assert "predicted" in p


def test_llm_as_judge_below_threshold_fails():
    class JudgeOut(BaseModel):
        score: float
        reasoning: str

    judge_llm = _MockJudgeLLM([JudgeOut(score=0.4, reasoning="off")])
    j = LLMAsJudge(
        llm=judge_llm,
        rubric="x",
        output_schema=JudgeOut,
        pass_threshold=0.7,
    )
    r = j.evaluate("p", "e", Example(id="x", input="?"))
    assert r.score == 0.4
    assert not r.pass_


def test_llm_as_judge_default_schema():
    """不传 output_schema 时用默认 schema"""
    judge_llm = _MockJudgeLLM([])
    j = LLMAsJudge(llm=judge_llm, rubric="x")
    # output_schema 应被自动填充
    assert j.output_schema is not None


def test_llm_as_judge_clamps_score_above_one():
    class JudgeOut(BaseModel):
        score: float
        reasoning: str

    # judge LLM 给出 1.2 → clamp 到 1.0
    judge_llm = _MockJudgeLLM([JudgeOut(score=1.2, reasoning="great")])
    j = LLMAsJudge(llm=judge_llm, rubric="x", output_schema=JudgeOut)
    r = j.evaluate("p", "e", Example(id="x", input="?"))
    assert r.score == 1.0


# ==================== Section F: _default_extract_predicted ====================


def test_extract_predicted_uses_final_answer():
    out = {"final_answer": "FA", "messages": [{"role": "assistant", "content": "AM"}]}
    assert _default_extract_predicted(out) == "FA"


def test_extract_predicted_falls_back_to_messages():
    out = {
        "messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "the answer"},
        ]
    }
    assert _default_extract_predicted(out) == "the answer"


def test_extract_predicted_returns_string_passthrough():
    assert _default_extract_predicted("hello") == "hello"


# ==================== Section G: run_eval 端到端 ====================


def _toy_target(input_text: Any) -> Dict[str, Any]:
    """玩具 target：把字符串 upper case 并模拟 token / 延迟"""
    return {
        "final_answer": str(input_text).upper(),
        "total_tokens": 10,
    }


def test_run_eval_basic_no_disk(tmp_path: Path):
    ds = Dataset.from_list(
        [
            {"id": "a", "input": "hello", "expected": "HELLO"},
            {"id": "b", "input": "world", "expected": "WORLD"},
            {"id": "c", "input": "wrong", "expected": "NOPE"},
        ]
    )
    report = run_eval(
        target=_toy_target,
        dataset=ds,
        evaluator=ExactMatch(),
        output_dir=tmp_path / "run1",
        write_report=False,
    )
    assert report.dataset_size == 3
    assert report.pass_count() == 2
    assert abs(report.pass_rate() - 2 / 3) < 0.001
    assert report.total_tokens() == 30


def test_run_eval_writes_report_md_and_results_jsonl(tmp_path: Path):
    ds = Dataset.from_list(
        [
            {"id": "a", "input": "hi", "expected": "HI", "tags": ["g1"]},
            {"id": "b", "input": "no", "expected": "WRONG", "tags": ["g2"]},
        ]
    )
    out_dir = tmp_path / "run2"
    run_eval(
        target=_toy_target,
        dataset=ds,
        evaluator=ExactMatch(),
        output_dir=out_dir,
    )
    # 文件存在
    rmd = out_dir / "report.md"
    rjsonl = out_dir / "results.jsonl"
    assert rmd.exists()
    assert rjsonl.exists()

    # report.md 必含三段
    content = rmd.read_text(encoding="utf-8")
    assert "## Summary" in content
    assert "## By tag" in content
    assert "## Top failures" in content
    # by tag 列出两个标签
    assert "g1" in content
    assert "g2" in content

    # results.jsonl 每行能解析
    with rjsonl.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert len(rows) == 2
    assert all("score" in r for r in rows)


def test_run_eval_handles_target_exception(tmp_path: Path):
    """某个 example 让 target 抛异常 → 该条记 error，整批继续"""

    def boom_target(input_text: Any) -> Dict[str, Any]:
        if input_text == "BOOM":
            raise RuntimeError("kaboom")
        return {"final_answer": str(input_text).upper(), "total_tokens": 5}

    ds = Dataset.from_list(
        [
            {"id": "ok", "input": "hi", "expected": "HI"},
            {"id": "bad", "input": "BOOM", "expected": "anything"},
            {"id": "ok2", "input": "world", "expected": "WORLD"},
        ]
    )
    report = run_eval(
        target=boom_target,
        dataset=ds,
        evaluator=ExactMatch(),
        write_report=False,
    )
    assert report.dataset_size == 3
    assert report.error_count() == 1
    # 其他 2 条仍然正常评估
    bad_results = [r for r in report.results if r.example_id == "bad"]
    assert bad_results[0].error is not None
    assert "RuntimeError" in bad_results[0].error
    ok_pass = sum(1 for r in report.results if r.eval_result and r.eval_result.pass_)
    assert ok_pass == 2


def test_run_eval_parallel_executes_concurrently(tmp_path: Path):
    """parallel=4 应明显比 parallel=1 快（每条 example sleep 0.1s）"""

    def slow_target(input_text: Any) -> Dict[str, Any]:
        time.sleep(0.05)
        return {"final_answer": str(input_text), "total_tokens": 1}

    ds = Dataset.from_list(
        [{"id": str(i), "input": str(i), "expected": str(i)} for i in range(8)]
    )

    t0 = time.time()
    report = run_eval(
        target=slow_target,
        dataset=ds,
        evaluator=ExactMatch(),
        parallel=4,
        write_report=False,
    )
    elapsed_parallel = time.time() - t0

    # 8 条 × 0.05s = 0.4s 串行；并行 4 路应在 ~0.15s 完成
    assert report.dataset_size == 8
    assert report.pass_count() == 8
    assert elapsed_parallel < 0.3, (
        f"parallel=4 应明显更快，实测 {elapsed_parallel:.2f}s"
    )


def test_run_eval_no_expected_skips_evaluation(tmp_path: Path):
    """expected=None 时只跑不评分"""
    ds = Dataset.from_list(
        [
            {"id": "1", "input": "x"},  # 无 expected
            {"id": "2", "input": "y"},
        ]
    )
    report = run_eval(
        target=_toy_target,
        dataset=ds,
        evaluator=ExactMatch(),
        write_report=False,
    )
    assert report.dataset_size == 2
    # 没有评估结果
    assert all(r.eval_result is None for r in report.results)
    assert report.pass_rate() == 0.0
    # 但 predicted 还是被记录
    assert all(r.predicted is not None for r in report.results)


def test_run_eval_top_failures_sorted_by_score():
    """top_failures 按分数升序：分数最低的 fail 排第一"""

    def fn(predicted, expected, example):
        # input 携带分数
        return EvalResult(
            score=float(example.input), pass_=float(example.input) >= 0.7
        )

    ds = Dataset.from_list(
        [
            {"id": "a", "input": "0.9", "expected": "x"},
            {"id": "b", "input": "0.2", "expected": "x"},
            {"id": "c", "input": "0.5", "expected": "x"},
        ]
    )
    report = run_eval(
        target=lambda x: {"final_answer": x, "total_tokens": 0},
        dataset=ds,
        evaluator=Custom(fn),
        write_report=False,
    )
    fails = report.top_failures(k=5)
    # b (0.2) 在前，c (0.5) 在后
    assert fails[0].example_id == "b"
    assert fails[1].example_id == "c"


def test_run_eval_by_tag_aggregation():
    ds = Dataset.from_list(
        [
            {"id": "a", "input": "hi", "expected": "HI", "tags": ["math"]},
            {"id": "b", "input": "no", "expected": "WRONG", "tags": ["math"]},
            {"id": "c", "input": "world", "expected": "WORLD", "tags": ["geo"]},
        ]
    )
    report = run_eval(
        target=_toy_target,
        dataset=ds,
        evaluator=ExactMatch(),
        write_report=False,
    )
    by_tag = report.by_tag()
    assert "math" in by_tag
    assert "geo" in by_tag
    # math 1/2 pass, geo 1/1 pass
    assert abs(by_tag["math"]["pass_rate"] - 0.5) < 0.001
    assert abs(by_tag["geo"]["pass_rate"] - 1.0) < 0.001


def test_run_eval_report_md_format_contains_metrics():
    ds = Dataset.from_list(
        [{"id": "a", "input": "hi", "expected": "HI"}]
    )
    report = run_eval(
        target=_toy_target,
        dataset=ds,
        evaluator=ExactMatch(),
        write_report=False,
    )
    md = _format_report_md(report)
    assert "Pass rate" in md
    assert "Mean score" in md
    assert "Latency p50/p95" in md
    assert "Total tokens" in md


# ==================== Section H: 顶层导入 ====================


def test_top_level_eval_imports():
    from clear_agent.eval import (
        BaseEvaluator,
        Contains,
        Custom,
        Dataset,
        EvalReport,
        EvalResult,
        Example,
        ExactMatch,
        ExampleRunResult,
        LLMAsJudge,
        run_eval,
    )

    assert callable(run_eval)
    assert Dataset is not None
