"""Eval-harness Evaluator 抽象与内置实现

四种内置 evaluator：

- ``ExactMatch``：完全匹配（带大小写 / 空白归一化开关）
- ``Contains``：子串包含
- ``LLMAsJudge``：用另一个 LLM 当裁判，复用 ``with_structured_output`` 让评分本身也是结构化输出
- ``Custom``：用户传 ``(predicted, expected, example) -> EvalResult``

"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING, Type

if TYPE_CHECKING:
    from pydantic import BaseModel
    from ..core.llm import ClearAgentLLM
    from .dataset import Example

# ==================== 核心数据 ====================

@dataclass
class EvalResult:
    """单条评估结果

    Attributes:
        score: 0.0 - 1.0 的分数
        pass_: 是否通过（``score >= threshold``）
        feedback: 人类可读评语
        metadata: judge reasoning / token 消耗等
    """

    score: float
    pass_: bool
    feedback: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "pass": self.pass_,
            "feedback": self.feedback,
            "metadata": dict(self.metadata),
        }

# ==================== Base ====================

class BaseEvaluator(ABC):
    """评估器基类

    子类需实现 ``evaluate(predicted, expected, example) -> EvalResult``
    """

    @abstractmethod
    def evaluate(
        self, predicted: Any, expected: Any, example: "Example"
    ) -> EvalResult:
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

# ==================== ExactMatch ====================

def _normalize(text: Any, *, case_sensitive: bool, normalize_whitespace: bool) -> str:
    s = "" if text is None else str(text)
    if normalize_whitespace:
        s = re.sub(r"\s+", " ", s).strip()
    if not case_sensitive:
        s = s.lower()
    return s

class ExactMatch(BaseEvaluator):
    """完全匹配；可选大小写与空白归一化

    Args:
        case_sensitive: 是否区分大小写（默认 False）
        normalize_whitespace: 是否归一化空白字符（默认 True）
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        normalize_whitespace: bool = True,
    ):
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace

    def evaluate(self, predicted, expected, example) -> EvalResult:
        p = _normalize(
            predicted,
            case_sensitive=self.case_sensitive,
            normalize_whitespace=self.normalize_whitespace,
        )
        e = _normalize(
            expected,
            case_sensitive=self.case_sensitive,
            normalize_whitespace=self.normalize_whitespace,
        )
        match = p == e
        return EvalResult(
            score=1.0 if match else 0.0,
            pass_=match,
            feedback="" if match else f"Expected '{e}', got '{p}'",
        )

# ==================== Contains ====================

class Contains(BaseEvaluator):
    """子串匹配：``expected`` 是否在 ``predicted`` 中出现

    适合 expected 是关键短语 / 关键词的开放式回答。
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        normalize_whitespace: bool = True,
    ):
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace

    def evaluate(self, predicted, expected, example) -> EvalResult:
        p = _normalize(
            predicted,
            case_sensitive=self.case_sensitive,
            normalize_whitespace=self.normalize_whitespace,
        )
        e = _normalize(
            expected,
            case_sensitive=self.case_sensitive,
            normalize_whitespace=self.normalize_whitespace,
        )
        if not e:
            # expected 为空 → 视为不评分，给 0.0 + fail
            return EvalResult(
                score=0.0,
                pass_=False,
                feedback="expected 为空，无法判断 Contains",
            )
        match = e in p
        return EvalResult(
            score=1.0 if match else 0.0,
            pass_=match,
            feedback="" if match else f"'{e}' not found in predicted",
        )

# ==================== LLMAsJudge ====================

class LLMAsJudge(BaseEvaluator):
    """用另一个 LLM 给开放式回答打分

    依赖 ``ClearAgentLLM.with_structured_output`` —— 评分本身也是结构化输出，
    让裁判 LLM 一次性返回 ``score`` (0.0-1.0) 和 ``reasoning``。

    Args:
        llm: 裁判 LLM（可与被测 LLM 不同）
        rubric: 评分准则的自然语言描述
        output_schema: 评分结果 Pydantic schema；若为 None 用默认 ``_DefaultJudgeSchema``
        pass_threshold: ``score >= threshold`` 视为通过（默认 0.7）
        score_field: schema 里 score 字段的属性名（默认 "score"）
        reasoning_field: schema 里 reasoning 字段的属性名（默认 "reasoning"）
    """

    def __init__(
        self,
        llm: "ClearAgentLLM",
        rubric: str,
        output_schema: Optional[Type["BaseModel"]] = None,
        pass_threshold: float = 0.7,
        score_field: str = "score",
        reasoning_field: str = "reasoning",
    ):
        self.llm = llm
        self.rubric = rubric
        self.pass_threshold = pass_threshold
        self.score_field = score_field
        self.reasoning_field = reasoning_field

        if output_schema is None:
            output_schema = _default_judge_schema()
        self.output_schema = output_schema

        # 复用 with_structured_output；method=auto
        self._structured = llm.with_structured_output(output_schema)

    def evaluate(self, predicted, expected, example) -> EvalResult:
        prompt = (
            f"You are an evaluator. Score the predicted answer based on the rubric.\n\n"
            f"Rubric:\n{self.rubric}\n\n"
            f"Question / Input:\n{example.input}\n\n"
            f"Expected (ground truth):\n{expected}\n\n"
            f"Predicted (model output):\n{predicted}\n\n"
            f"Return your score in [0.0, 1.0] with brief reasoning."
        )
        out = self._structured.invoke([{"role": "user", "content": prompt}])
        score = float(getattr(out, self.score_field, 0.0))
        reasoning = str(getattr(out, self.reasoning_field, ""))
        # clamp
        score = max(0.0, min(1.0, score))
        return EvalResult(
            score=score,
            pass_=score >= self.pass_threshold,
            feedback=reasoning,
            metadata={"judge_model": getattr(self.llm, "model", "")},
        )

def _default_judge_schema() -> Type["BaseModel"]:
    """默认 judge schema（懒加载 pydantic 避免顶层 import 失败）"""
    from pydantic import BaseModel, Field

    class _DefaultJudgeSchema(BaseModel):
        """LLM-as-judge default schema."""

        score: float = Field(..., description="Quality score from 0.0 to 1.0")
        reasoning: str = Field(..., description="Brief justification")

    return _DefaultJudgeSchema

# ==================== Custom ====================

class Custom(BaseEvaluator):
    """用户自定义函数 evaluator

    传入 ``fn(predicted, expected, example) -> EvalResult`` 或 ``-> float``。
    返回 float 时自动用 ``pass_threshold`` 判断 pass_。
    """

    def __init__(
        self,
        fn: Callable[[Any, Any, "Example"], Any],
        pass_threshold: float = 0.5,
        name: Optional[str] = None,
    ):
        self.fn = fn
        self.pass_threshold = pass_threshold
        self._name = name or getattr(fn, "__name__", "custom")

    @property
    def name(self) -> str:
        return f"Custom({self._name})"

    def evaluate(self, predicted, expected, example) -> EvalResult:
        out = self.fn(predicted, expected, example)
        if isinstance(out, EvalResult):
            return out
        # 把 float / int / bool 转 EvalResult
        if isinstance(out, (int, float)):
            score = float(out)
            score = max(0.0, min(1.0, score))
            return EvalResult(
                score=score,
                pass_=score >= self.pass_threshold,
            )
        if isinstance(out, bool):
            return EvalResult(score=1.0 if out else 0.0, pass_=out)
        raise TypeError(
            f"Custom evaluator fn 必须返回 EvalResult / float / bool，得到 {type(out)}"
        )

__all__ = [
    "EvalResult",
    "BaseEvaluator",
    "ExactMatch",
    "Contains",
    "LLMAsJudge",
    "Custom",
]
