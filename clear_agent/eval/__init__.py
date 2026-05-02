"""ClearAgent eval-harness MVP

用法:

```python
from clear_agent.eval import Dataset, ExactMatch, run_eval
from clear_agent.agents import build_react_graph

ds = Dataset.from_jsonl("examples/eval/qa.jsonl")
graph = build_react_graph(llm)

report = run_eval(
    target=graph,
    dataset=ds,
    evaluator=ExactMatch(),
    output_dir="memory/eval/run1",
    parallel=4,
)
print(f"Pass rate: {report.pass_rate() * 100:.1f}%")
```

详见 project_docs/05-eval-harness.md
"""

from .dataset import Dataset, Example
from .evaluator import (
    BaseEvaluator,
    Contains,
    Custom,
    EvalResult,
    ExactMatch,
    LLMAsJudge,
)
from .runner import EvalReport, ExampleRunResult, run_eval

__all__ = [
    # data
    "Dataset",
    "Example",
    # evaluators
    "BaseEvaluator",
    "EvalResult",
    "ExactMatch",
    "Contains",
    "LLMAsJudge",
    "Custom",
    # runner
    "run_eval",
    "EvalReport",
    "ExampleRunResult",
]
