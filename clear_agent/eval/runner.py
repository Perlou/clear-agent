"""Eval-harness Runner —— ``run_eval`` 主入口

输入：``target`` (CompiledGraph 或 callable)、``Dataset``、``BaseEvaluator``
输出：
- ``output_dir/results.jsonl``：每条 example 的原始结果
- ``output_dir/report.md``：聚合报告（pass rate / 延迟 / token / by-tag / top failures）
- 返回 ``EvalReport`` 对象供进一步处理

并发执行用 ``ThreadPoolExecutor``（不强求异步——多数评估场景 IO 密集，线程足够）。

详见 project_docs/05-eval-harness.md
"""

from __future__ import annotations

import json
import statistics
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .dataset import Dataset, Example
from .evaluator import BaseEvaluator, EvalResult


# ==================== 数据结构 ====================


@dataclass
class ExampleRunResult:
    """单条 example 的完整运行结果"""

    example_id: str
    input: Any
    expected: Any
    predicted: Any
    eval_result: Optional[EvalResult]
    latency_ms: int
    tokens: int
    error: Optional[str]
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.example_id,
            "input": self.input,
            "expected": self.expected,
            "predicted": self.predicted,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "tags": list(self.tags),
        }
        if self.eval_result is not None:
            d.update(
                {
                    "score": self.eval_result.score,
                    "pass": self.eval_result.pass_,
                    "feedback": self.eval_result.feedback,
                }
            )
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class EvalReport:
    """聚合评估报告

    Attributes:
        run_id: 唯一运行 ID
        target_name: target 名称（用于报告标题）
        evaluator_name: evaluator 名称
        dataset_size: 数据集大小
        results: 全部 ExampleRunResult
        output_dir: 输出目录
    """

    run_id: str
    target_name: str
    evaluator_name: str
    dataset_size: int
    results: List[ExampleRunResult]
    output_dir: Path

    # ---------- 聚合统计 ----------

    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.eval_result and r.eval_result.pass_)

    def pass_rate(self) -> float:
        n = sum(1 for r in self.results if r.eval_result is not None)
        return self.pass_count() / n if n > 0 else 0.0

    def mean_score(self) -> float:
        scores = [r.eval_result.score for r in self.results if r.eval_result]
        return statistics.mean(scores) if scores else 0.0

    def latency_p50(self) -> int:
        lats = [r.latency_ms for r in self.results if r.latency_ms >= 0]
        return int(statistics.median(lats)) if lats else 0

    def latency_p95(self) -> int:
        lats = sorted(r.latency_ms for r in self.results if r.latency_ms >= 0)
        if not lats:
            return 0
        idx = max(0, int(0.95 * (len(lats) - 1)))
        return int(lats[idx])

    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.results)

    def error_count(self) -> int:
        return sum(1 for r in self.results if r.error)

    def by_tag(self) -> Dict[str, Dict[str, Any]]:
        agg: Dict[str, Dict[str, Any]] = {}
        for r in self.results:
            for tag in r.tags or ["<untagged>"]:
                slot = agg.setdefault(
                    tag, {"count": 0, "pass_count": 0, "score_sum": 0.0}
                )
                slot["count"] += 1
                if r.eval_result:
                    slot["score_sum"] += r.eval_result.score
                    if r.eval_result.pass_:
                        slot["pass_count"] += 1
        # 转换为 pass_rate / mean_score
        out: Dict[str, Dict[str, Any]] = {}
        for tag, slot in sorted(agg.items()):
            n = slot["count"]
            out[tag] = {
                "count": n,
                "pass_rate": slot["pass_count"] / n if n else 0.0,
                "mean_score": slot["score_sum"] / n if n else 0.0,
            }
        return out

    def top_failures(self, k: int = 5) -> List[ExampleRunResult]:
        """按 score 升序取前 k 个失败 example"""
        failed = [
            r
            for r in self.results
            if (r.eval_result and not r.eval_result.pass_) or r.error
        ]
        failed.sort(
            key=lambda r: (r.eval_result.score if r.eval_result else -1.0, r.example_id)
        )
        return failed[:k]


# ==================== 默认 extract_predicted ====================


def _default_extract_predicted(output: Any) -> Any:
    """从 graph / callable 输出中提取 predicted 字段

    支持：
    - dict 含 ``final_answer`` → 直接取
    - dict 含 ``messages`` → 取最后一条 assistant 消息的 content
    - dict 其他 → 整个 dict 作为 predicted
    - 字符串等其他类型 → 原样返回
    """
    if isinstance(output, dict):
        if "final_answer" in output and output["final_answer"] is not None:
            return output["final_answer"]
        msgs = output.get("messages")
        if msgs:
            for m in reversed(msgs):
                role = m.get("role") if isinstance(m, dict) else None
                if role == "assistant":
                    return m.get("content")
        return output
    return output


# ==================== 内部 helpers ====================


def _extract_tokens(output: Any) -> int:
    if isinstance(output, dict):
        v = output.get("total_tokens", 0)
        try:
            return int(v) if v is not None else 0
        except (TypeError, ValueError):
            return 0
    return 0


def _invoke_target(target: Any, ex: Example) -> Any:
    """支持 CompiledGraph (.invoke) / callable / 任意带 invoke 方法的对象"""
    if hasattr(target, "invoke") and callable(target.invoke):
        if isinstance(ex.input, dict):
            return target.invoke(ex.input)
        # CompiledGraph 期望 dict-like state；字符串则放进 messages
        return target.invoke(
            {"messages": [{"role": "user", "content": str(ex.input)}]}
        )
    if callable(target):
        return target(ex.input)
    raise TypeError(f"target 必须是 CompiledGraph / callable，得到 {type(target)}")


# ==================== 主入口 ====================


def run_eval(
    target: Any,
    dataset: Dataset,
    evaluator: BaseEvaluator,
    output_dir: Optional[Union[str, Path]] = None,
    parallel: int = 1,
    extract_predicted: Optional[Callable[[Any], Any]] = None,
    target_name: Optional[str] = None,
    run_id: Optional[str] = None,
    write_report: bool = True,
) -> EvalReport:
    """跑一遍数据集，输出 markdown 报告 + jsonl 原始结果

    Args:
        target: ``CompiledGraph`` 或 ``callable(input) -> output``
        dataset: ``Dataset`` 对象
        evaluator: ``BaseEvaluator`` 实例
        output_dir: 报告输出目录；缺省 ``memory/eval/<run_id>/``
        parallel: 并发线程数（默认 1，IO 密集时建议 4-8）
        extract_predicted: ``output -> predicted`` 抽取函数；缺省用
            ``_default_extract_predicted``（兼容 graph state dict / 字符串）
        target_name: 报告里展示的 target 名；缺省取 ``target.__class__.__name__``
        run_id: 自定义 run id；缺省 ``r-YYYYMMDD-HHMMSS-<rand>``
        write_report: 是否落盘 report.md / results.jsonl（测试时可关）

    Returns:
        ``EvalReport`` 实例
    """
    extract_fn = extract_predicted or _default_extract_predicted

    if run_id is None:
        run_id = "r-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    if output_dir is None:
        output_dir = Path("memory/eval") / run_id
    output_dir = Path(output_dir)

    if target_name is None:
        target_name = (
            target.__class__.__name__
            if not callable(target) or hasattr(target, "invoke")
            else getattr(target, "__name__", "callable")
        )

    examples = list(dataset)
    results: List[ExampleRunResult] = [None] * len(examples)  # type: ignore[list-item]

    def _run_one(idx: int, ex: Example) -> ExampleRunResult:
        start = time.time()
        predicted: Any = None
        eval_res: Optional[EvalResult] = None
        error: Optional[str] = None
        tokens = 0
        try:
            output = _invoke_target(target, ex)
            tokens = _extract_tokens(output)
            predicted = extract_fn(output)
            if ex.expected is not None:
                eval_res = evaluator.evaluate(predicted, ex.expected, ex)
        except Exception as e:
            error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        latency_ms = int((time.time() - start) * 1000)
        return ExampleRunResult(
            example_id=ex.id,
            input=ex.input,
            expected=ex.expected,
            predicted=predicted,
            eval_result=eval_res,
            latency_ms=latency_ms,
            tokens=tokens,
            error=error,
            tags=list(ex.tags),
        )

    if parallel <= 1:
        for i, ex in enumerate(examples):
            results[i] = _run_one(i, ex)
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(_run_one, i, ex): i for i, ex in enumerate(examples)}
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()

    report = EvalReport(
        run_id=run_id,
        target_name=str(target_name),
        evaluator_name=evaluator.name,
        dataset_size=len(examples),
        results=results,
        output_dir=output_dir,
    )

    if write_report:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_results_jsonl(report)
        _write_report_md(report)

    return report


# ==================== 落盘 ====================


def _write_results_jsonl(report: EvalReport) -> None:
    p = report.output_dir / "results.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in report.results:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False, default=str) + "\n")


def _write_report_md(report: EvalReport) -> None:
    p = report.output_dir / "report.md"
    p.write_text(_format_report_md(report), encoding="utf-8")


def _format_report_md(report: EvalReport) -> str:
    lines: List[str] = []
    lines.append(f"# Eval Report: {report.run_id}")
    lines.append("")
    lines.append(f"- **Target**: {report.target_name}")
    lines.append(f"- **Evaluator**: {report.evaluator_name}")
    lines.append(f"- **Dataset size**: {report.dataset_size}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    n_eval = sum(1 for r in report.results if r.eval_result is not None)
    lines.append(
        f"- **Pass rate**: {report.pass_count()} / {n_eval} "
        f"({report.pass_rate() * 100:.1f}%)"
    )
    lines.append(f"- **Mean score**: {report.mean_score():.3f}")
    lines.append(
        f"- **Latency p50/p95**: {report.latency_p50()} ms / {report.latency_p95()} ms"
    )
    lines.append(f"- **Total tokens**: {report.total_tokens()}")
    lines.append(f"- **Errors**: {report.error_count()}")
    lines.append("")

    # By tag
    by_tag = report.by_tag()
    if by_tag:
        lines.append("## By tag")
        lines.append("| Tag | Count | Pass rate | Mean score |")
        lines.append("|---|---|---|---|")
        for tag, slot in by_tag.items():
            lines.append(
                f"| {tag} | {slot['count']} | "
                f"{slot['pass_rate'] * 100:.1f}% | "
                f"{slot['mean_score']:.3f} |"
            )
        lines.append("")

    # Top failures
    fails = report.top_failures(k=5)
    if fails:
        lines.append("## Top failures (first 5)")
        lines.append("| ID | Input | Expected | Predicted | Feedback / Error |")
        lines.append("|---|---|---|---|---|")
        for r in fails:
            inp = _short(r.input)
            exp = _short(r.expected)
            pred = _short(r.predicted)
            fb = r.error or (r.eval_result.feedback if r.eval_result else "")
            lines.append(
                f"| {r.example_id} | {inp} | {exp} | {pred} | {_short(fb, 80)} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _short(v: Any, n: int = 60) -> str:
    s = str(v).replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


__all__ = [
    "run_eval",
    "EvalReport",
    "ExampleRunResult",
]
