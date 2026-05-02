"""Eval-harness Dataset 抽象

从 JSONL / list / dict 加载评估样本，支持 filter / sample / 切片。

JSONL 格式（每行一个 example）：

```jsonl
{"id": "ex-001", "input": "What is 2+2?", "expected": "4", "tags": ["math"]}
```

字段：
- ``id``（必填）：唯一标识
- ``input``（必填）：传给 graph / callable 的输入
- ``expected``（可选）：基线答案
- ``tags``（可选）：报告分组聚合
- ``metadata``（可选）：透传到报告

详见 project_docs/05-eval-harness.md
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union


@dataclass
class Example:
    """单条评估样本

    Attributes:
        id: 唯一标识
        input: graph / callable 的输入（任意 JSON 可序列化）
        expected: 基线答案（可选；缺省时只跑不评分）
        tags: 标签列表，用于报告分组
        metadata: 透传到结果的额外信息
    """

    id: str
    input: Any
    expected: Any = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Example":
        if "id" not in d or "input" not in d:
            raise ValueError(f"Example 必须含 id 和 input 字段，缺失：{d}")
        return cls(
            id=str(d["id"]),
            input=d["input"],
            expected=d.get("expected"),
            tags=list(d.get("tags") or []),
            metadata=dict(d.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "input": self.input,
            "expected": self.expected,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }


class Dataset:
    """Eval 数据集 —— 内存中的 ``Example`` 序列

    创建方式：
    - ``Dataset.from_jsonl(path)``：每行一个 JSON 对象
    - ``Dataset.from_list([{"id":..,"input":..,"expected":..}, ...])``
    - ``Dataset(examples=[Example(...), ...])`` 直接构造

    操作（链式）：
    - ``ds.filter(predicate)``：返回新 Dataset
    - ``ds.sample(n, seed=42)``：随机抽 n 条
    - ``ds.take(n)`` / ``ds[idx]`` / ``len(ds)`` / ``for ex in ds:``
    """

    def __init__(self, examples: Optional[List[Example]] = None):
        self.examples: List[Example] = list(examples or [])

    # --------- 加载 ---------

    @classmethod
    def from_jsonl(cls, path: Union[str, Path]) -> "Dataset":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"数据集文件不存在：{p}")
        examples: List[Example] = []
        with p.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{p}:{line_no} JSON 解析失败：{e}") from e
                examples.append(Example.from_dict(obj))
        return cls(examples)

    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> "Dataset":
        return cls([Example.from_dict(d) for d in data])

    # --------- 操作 ---------

    def filter(self, predicate: Callable[[Example], bool]) -> "Dataset":
        return Dataset([ex for ex in self.examples if predicate(ex)])

    def sample(self, n: int, seed: Optional[int] = None) -> "Dataset":
        if n >= len(self.examples):
            return Dataset(list(self.examples))
        rng = random.Random(seed)
        picked = rng.sample(self.examples, n)
        return Dataset(picked)

    def take(self, n: int) -> "Dataset":
        return Dataset(self.examples[:n])

    # --------- 容器协议 ---------

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Example, "Dataset"]:
        if isinstance(idx, slice):
            return Dataset(self.examples[idx])
        return self.examples[idx]

    def __repr__(self) -> str:
        return f"Dataset(n={len(self.examples)})"

    # --------- 落盘（便于调试） ---------

    def to_jsonl(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for ex in self.examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")


__all__ = ["Example", "Dataset"]
