# 05 · Eval-Harness 设计（MVP）

> **阶段**：2.0-α / W4
> **目标文件**：`clear_agent/eval/{dataset,evaluator,runner}.py`
> **关联文档**：01（agent → graph）、04（结构化输出常被 eval 用作 judge）

---

## 1. 设计目标

让用户能跑「**100 个测试用例 → 对比两个 graph / prompt / 模型 → 输出 markdown 报告**」，覆盖三类典型评估：
- **完全匹配**：exact_match（QA、分类）
- **LLM-as-judge**：用另一个 LLM 当裁判（开放式回答、风格、安全）
- **自定义函数**：用户传 `(predicted, expected) -> float`（数值容差、JSON diff 等）

**核心要求**：
- 离线批跑，不强求实时
- 不依赖外部服务（LangSmith、W&B 等）—— 但留导出口
- 数据格式简单：JSONL（每行一个 example）
- 输出 markdown 报告 + 原始 jsonl 便于二次分析

**非目标（推迟）**：
- A/B 多 graph 并排对比 → 2.0-β
- 在线 dataset 增量追加 → 2.0-RC
- LangSmith / Langfuse 导出 → 2.0-β

---

## 2. 数据格式

### 2.1 Dataset JSONL

```jsonl
{"id": "ex-001", "input": "What is 2+2?", "expected": "4", "tags": ["math", "easy"]}
{"id": "ex-002", "input": "Capital of France?", "expected": "Paris", "tags": ["geo"]}
{"id": "ex-003", "input": {"messages": [...]}, "expected": {"answer": "X"}, "tags": ["multi-turn"]}
```

**字段**：
- `id`（必填）：唯一 ID
- `input`（必填）：传给 graph 的输入；可以是字符串、字典、列表
- `expected`（可选）：基线答案；缺省时只跑不评分
- `tags`（可选）：用于报告分组聚合
- `metadata`（可选）：透传到报告

### 2.2 Dataset API

```python
from clear_agent.eval import Dataset

ds = Dataset.from_jsonl("examples/eval/qa.jsonl")
ds = Dataset.from_list([{"id": "1", "input": "...", "expected": "..."}])
ds = ds.filter(lambda ex: "math" in ex.tags)
ds = ds.sample(n=20, seed=42)
```

---

## 3. Evaluator 接口

```python
class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, predicted: Any, expected: Any, example: Example) -> EvalResult: ...
```

```python
@dataclass
class EvalResult:
    score: float            # 0.0 - 1.0
    pass_: bool             # 是否通过（score >= threshold）
    feedback: str           # 人类可读评价
    metadata: dict          # judge 的 reasoning、token 消耗等
```

### 3.1 内置 Evaluator

#### `ExactMatch`

```python
from clear_agent.eval.evaluator import ExactMatch

ev = ExactMatch(case_sensitive=False, normalize_whitespace=True)
result = ev.evaluate("paris", "Paris", example)  # score=1.0, pass_=True
```

#### `Contains`

```python
from clear_agent.eval.evaluator import Contains

ev = Contains(case_sensitive=False)
ev.evaluate("The capital of France is Paris.", "Paris", example)  # score=1.0
```

#### `LLMAsJudge`

```python
from clear_agent.eval.evaluator import LLMAsJudge
from pydantic import BaseModel

class Score(BaseModel):
    score: float           # 0.0 - 1.0
    reasoning: str

ev = LLMAsJudge(
    llm=ClearAgentLLM(),                     # 复用主 llm 或换独立 judge
    rubric="Score 1.0 if accurate, 0.5 if partially, 0.0 if wrong",
    output_schema=Score,                     # 复用 04-structured-output.md
    pass_threshold=0.7,
)
```

> 注意：`LLMAsJudge` 直接调用 `llm.with_structured_output(Score)`，把评分这件事本身做成结构化输出。

#### `Custom`

```python
from clear_agent.eval.evaluator import Custom

def numeric_close(predicted, expected, example):
    score = 1.0 if abs(float(predicted) - float(expected)) < 0.01 else 0.0
    return EvalResult(score=score, pass_=score>0.5, feedback="", metadata={})

ev = Custom(numeric_close)
```

---

## 4. Runner（核心入口）

```python
from clear_agent.eval import run_eval

report = run_eval(
    target=compiled_graph,                   # 或 callable: input -> output
    dataset=ds,
    evaluator=ExactMatch(),
    output_dir="memory/eval/2026-05-02-react-baseline",
    parallel=4,                              # 并发跑 example
    extract_predicted=lambda state: state["messages"][-1].content,  # graph 输出 → predicted
)
```

**行为**：
1. 对每条 example 调 `target(input)` → 拿到原始输出
2. `extract_predicted(output) → predicted`
3. `evaluator.evaluate(predicted, expected, example) → EvalResult`
4. 落盘原始结果到 `output_dir/results.jsonl`
5. 生成 `output_dir/report.md`（见下）

---

## 5. 报告格式

`memory/eval/<run_id>/report.md`：

```markdown
# Eval Report: react-baseline-2026-05-02

**Dataset**: examples/eval/qa.jsonl (100 examples)
**Target**: build_react_graph(deepseek-chat)
**Evaluator**: ExactMatch
**Run ID**: r-20260502-1430-a3f2

## Summary
- **Pass rate**: 78 / 100 (78.0%)
- **Mean score**: 0.81
- **Latency p50/p95**: 1.2s / 4.8s
- **Total tokens**: 245,310 (avg 2,453 / example)
- **Estimated cost**: $0.43 USD (deepseek pricing)
- **Failures**: 22

## By tag
| Tag | Count | Pass rate | Mean score |
|---|---|---|---|
| math | 30 | 90.0% | 0.92 |
| geo | 25 | 80.0% | 0.81 |
| multi-turn | 45 | 68.9% | 0.71 |

## Top failures (first 5)
| ID | Input | Expected | Predicted | Feedback |
|---|---|---|---|---|
| ex-007 | What is 17 * 23? | 391 | 392 | Off by 1 |
| ... | | | | |

## Token / latency distribution
[简单的 ASCII 直方图或 mermaid]
```

同时输出 `results.jsonl`：
```jsonl
{"id": "ex-001", "predicted": "4", "expected": "4", "score": 1.0, "pass": true, "latency_ms": 850, "tokens": 120}
{"id": "ex-002", "predicted": "Lyon", "expected": "Paris", "score": 0.0, "pass": false, "latency_ms": 1200, "tokens": 95, "feedback": "Wrong answer"}
```

---

## 6. CLI 入口

```bash
python -m clear_agent.eval \
  --dataset examples/eval/qa.jsonl \
  --graph react \
  --evaluator exact_match \
  --output memory/eval/run1 \
  --parallel 4
```

底层等价：
```python
from clear_agent.eval import Dataset, ExactMatch, run_eval
from clear_agent.agents import build_react_graph
ds = Dataset.from_jsonl("examples/eval/qa.jsonl")
g = build_react_graph(...)
run_eval(g, ds, ExactMatch(), output_dir="memory/eval/run1", parallel=4)
```

---

## 7. 配置项

`Config` 新增：

```python
eval_enabled: bool = True
eval_output_dir: str = "memory/eval"
eval_default_parallel: int = 4
eval_judge_model: str = ""        # LLMAsJudge 默认用的模型；空时复用主 llm
eval_judge_base_url: str = ""
eval_judge_api_key: str = ""
```

---

## 8. 测试清单（W4 出口）

`tests/test_eval_runner.py`：

| # | 测试 | 通过标准 |
|---|---|---|
| 1 | Dataset.from_jsonl | 加载/过滤/采样正常 |
| 2 | ExactMatch | 大小写/空白归一化正确 |
| 3 | Contains | 子串匹配 |
| 4 | LLMAsJudge | mock LLM 返回 0.8 → result.score=0.8 |
| 5 | Custom | 自定义函数被调用，结果透传 |
| 6 | run_eval 基本流程 | 10 example 全部产出 EvalResult |
| 7 | 并发执行 | parallel=4 时实际并发跑 |
| 8 | 报告生成 | report.md 包含 summary / by_tag / top_failures 三段 |
| 9 | 失败 example 不打断 | example 抛异常时记入 results 并继续 |

---

## 9. 与 TraceLogger 的协作

每条 example 跑完后：
- 复用 `TraceLogger` 写一份 trace（同 example_id 关联）
- `results.jsonl` 字段 `trace_path` 指向对应 trace 文件
- 方便从 fail 列表点击进入逐步审查

---

## 10. 与 RL 模块的关系（说明）

按计划文件决策：**永不引入 RL 模块**。但 eval-harness 同时承担了「训练数据导出」职责：

```python
# 2.0-β 新增（这里只占位说明）
from clear_agent.eval import export_to_sft_jsonl, export_to_dpo_pairs

# 把 results.jsonl 的成功 example 转成 SFT 格式
export_to_sft_jsonl("memory/eval/run1/results.jsonl", "out.sft.jsonl")

# 把 fail/pass 配对转成 DPO 偏好对
export_to_dpo_pairs("memory/eval/run1-pass/results.jsonl",
                    "memory/eval/run1-fail/results.jsonl",
                    "out.dpo.jsonl")
```

→ 用户拿这些 jsonl 喂给独立的 `trl` / `axolotl` 训练脚本。**框架不长肉，用户能训。**

---

## 11. 待决问题

1. **`extract_predicted` 默认实现？**
   - 建议：input 为字符串时假设 graph 返回 `state["messages"][-1].content`；其他情况强制用户传

2. **`LLMAsJudge` 的 judge model 是否必须与被测 llm 不同？**
   - 建议：不强制但 WARNING；`eval_judge_model` 留作配置

3. **`pass_threshold` 默认值**
   - 建议：0.5（任何 evaluator score >= 0.5 都算 pass）；用户可覆盖

4. **报告里 cost USD 怎么算？**
   - 2.0-α 用静态 pricing.yaml（`gpt-4o-input/output`、`deepseek-chat-input/output` 等）
   - 模型不在表里 → cost 字段显示 `unknown`，不打 ERROR

请确认。
