# Eval-harness（2.0 用户向 quickstart）

> 设计 spec 详见 [`project_docs/05-eval-harness.md`](../project_docs/05-eval-harness.md)。

跑「100 个用例 → 对比 graph / prompt / 模型 → 输出 markdown 报告」的最小工具集。

## 1. 数据集

JSONL 格式（每行一个 example）：

```jsonl
{"id": "ex-001", "input": "What is 2+2?", "expected": "4", "tags": ["math"]}
{"id": "ex-002", "input": "Capital of France?", "expected": "Paris", "tags": ["geo"]}
```

```python
from clear_agent.eval import Dataset

ds = Dataset.from_jsonl("examples/eval/qa.jsonl")
ds = Dataset.from_list([{"id": "1", "input": "...", "expected": "..."}])
ds = ds.filter(lambda ex: "math" in ex.tags)
ds = ds.sample(20, seed=42)
ds = ds.take(10)        # 前 10
```

`expected` 缺省时只跑不评分（用于离线探索）。

## 2. 四种 Evaluator

```python
from clear_agent.eval import ExactMatch, Contains, Custom, LLMAsJudge

# 完全匹配（默认大小写不敏感 + 空白归一化）
ExactMatch()
ExactMatch(case_sensitive=True, normalize_whitespace=False)

# 子串包含
Contains()

# 自定义函数
def numeric_close(predicted, expected, example):
    return 1.0 if abs(float(predicted) - float(expected)) < 0.01 else 0.0

Custom(numeric_close, pass_threshold=0.5)

# LLM-as-judge（用另一个 LLM 当裁判）
from pydantic import BaseModel
class Score(BaseModel):
    score: float
    reasoning: str

LLMAsJudge(
    llm=judge_llm,
    rubric="Score 1.0 if accurate",
    output_schema=Score,
    pass_threshold=0.7,
)
```

## 3. Runner

```python
from clear_agent.eval import run_eval

report = run_eval(
    target=compiled_graph,            # 或 callable: input -> output
    dataset=ds,
    evaluator=ExactMatch(),
    output_dir="memory/eval/run1",    # 缺省 memory/eval/<run_id>/
    parallel=4,                       # 并发线程数
)

print(f"Pass rate: {report.pass_rate() * 100:.1f}%")
print(f"Mean score: {report.mean_score():.3f}")
```

`target` 支持：
- `CompiledGraph` —— 自动调 `target.invoke(state_dict)`，input 是字符串时包成 `{"messages":[{"role":"user","content":...}]}`
- `callable` —— 直接 `target(input)`

## 4. extract_predicted

`target` 输出可能是字典 / 字符串。默认提取规则：

1. dict 含 `final_answer` → 取 `final_answer`
2. dict 含 `messages` → 取最后一条 assistant 的 content
3. 其他 → 原样返回

需要自定义：

```python
report = run_eval(
    target=graph,
    dataset=ds,
    evaluator=ExactMatch(),
    extract_predicted=lambda state: state["my_field"]["nested"],
)
```

## 5. 报告

`run_eval` 落盘两份文件：

- `output_dir/results.jsonl` —— 每条 example 的 predicted / expected / score / latency / tokens
- `output_dir/report.md` —— Summary / By tag / Top failures 三段式 markdown

```markdown
# Eval Report: r-20260502-1430

## Summary
- Pass rate: 78 / 100 (78.0%)
- Mean score: 0.81
- Latency p50/p95: 1200 ms / 4800 ms
- Total tokens: 245310

## By tag
| Tag | Count | Pass rate | Mean score |
|---|---|---|---|
| math | 30 | 90.0% | 0.92 |
| geo | 25 | 80.0% | 0.81 |

## Top failures (first 5)
| ID | Input | Expected | Predicted | Feedback |
| ex-007 | What is 17 * 23? | 391 | 392 | Off by 1 |
```

`report.results` 也可在内存里直接处理：

```python
fails = report.top_failures(k=10)
by_tag = report.by_tag()       # {"math": {"count":30, "pass_rate":0.9, "mean_score":0.92}, ...}
errors = report.error_count()  # target 抛异常的条数
```

## 6. 失败容错

某条 example 的 target 抛异常 → 该条记 `error` 字段，整批继续跑。

```python
for r in report.results:
    if r.error:
        print(r.example_id, r.error)
```

## 7. 不写盘

```python
report = run_eval(..., write_report=False)   # 不落 results.jsonl / report.md
```
