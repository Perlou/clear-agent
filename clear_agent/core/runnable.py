"""LCEL-lite —— Runnable 协议 + ``|`` 管道组合

LangChain Expression Language 的精简版本，让用户用 ``|`` 管道把 prompt /
LLM / parser / tool 等串起来，无需自己写 graph：

```python
from clear_agent.core.runnable import Runnable, prompt, parser_str

chain = (
    prompt("回答问题：{question}")
    | llm                       # ClearAgentLLM 已经是 Runnable
    | parser_str()              # 取 .content
)
result = chain.invoke({"question": "Python 是什么？"})
```

特点：
- 任何带 ``invoke(input)`` 的对象都自动是 Runnable
- ``|`` 创建 ``RunnableSequence`` 串行管道
- ``RunnableParallel({"a": r1, "b": r2})`` 字典并发
- ``RunnableLambda(fn)`` 把任意 callable 包成 Runnable
- ``RunnableBranch`` 条件分支
- 全部支持 sync ``invoke`` 与 async ``ainvoke``

不引入 ``langchain-core`` 依赖；自研约 ~200 行。
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple


# ==================== Runnable 抽象 ====================


class Runnable(ABC):
    """Runnable 协议基类

    任何子类必须实现 ``invoke(input)``；``ainvoke`` 默认走线程池包装，
    子类可覆写为真异步。
    """

    @abstractmethod
    def invoke(self, input: Any, **kwargs: Any) -> Any:
        ...

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.invoke(input, **kwargs))

    # ``|`` 管道 → RunnableSequence
    def __or__(self, other: Any) -> "RunnableSequence":
        right = _to_runnable(other)
        if isinstance(self, RunnableSequence):
            return RunnableSequence([*self.steps, right])
        return RunnableSequence([self, right])

    def __ror__(self, other: Any) -> "RunnableSequence":
        left = _to_runnable(other)
        if isinstance(self, RunnableSequence):
            return RunnableSequence([left, *self.steps])
        return RunnableSequence([left, self])


# ==================== 工厂：把任意对象包成 Runnable ====================


def _to_runnable(obj: Any) -> Runnable:
    """把 callable / 已有 Runnable / 含 invoke 方法的对象 → Runnable

    优先级：
    1. 已经是 ``Runnable`` 实例 → 直接返回
    2. 含 ``invoke`` 方法（明确协议）→ ``RunnableAdapter``
    3. 一般 callable → ``RunnableLambda``
    """
    if isinstance(obj, Runnable):
        return obj
    # 优先 invoke 协议（避免 MagicMock 等"既 callable 又有 invoke"的对象走错路）
    if hasattr(obj, "invoke") and callable(getattr(obj, "invoke")):
        return RunnableAdapter(obj)
    if callable(obj):
        return RunnableLambda(obj)
    raise TypeError(
        f"无法把 {type(obj).__name__} 转为 Runnable；"
        "需要 callable 或带 invoke 方法"
    )


class RunnableLambda(Runnable):
    """把任意 sync/async callable 包成 Runnable"""

    def __init__(self, func: Callable[[Any], Any], name: Optional[str] = None):
        self.func = func
        self.name = name or getattr(func, "__name__", "lambda")

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        result = self.func(input)
        if inspect.isawaitable(result):
            # async 函数被同步入口调用 → 用 asyncio.run
            return asyncio.run(result)
        return result

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        result = self.func(input)
        if inspect.isawaitable(result):
            return await result
        return result

    def __repr__(self) -> str:
        return f"RunnableLambda({self.name})"


class RunnableAdapter(Runnable):
    """把含 ``invoke`` 方法的对象（如 ``ClearAgentLLM``）包成 Runnable"""

    def __init__(self, target: Any):
        self.target = target

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        return self.target.invoke(input, **kwargs)

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        afn = getattr(self.target, "ainvoke", None)
        if callable(afn):
            return await afn(input, **kwargs)
        return await super().ainvoke(input, **kwargs)


# ==================== Sequence ====================


class RunnableSequence(Runnable):
    """``|`` 串联多个 Runnable，前一步的输出作为下一步的输入"""

    def __init__(self, steps: List[Runnable]):
        if not steps:
            raise ValueError("RunnableSequence 至少需要 1 个步骤")
        self.steps = list(steps)

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        cur = input
        for s in self.steps:
            cur = s.invoke(cur, **kwargs)
        return cur

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        cur = input
        for s in self.steps:
            cur = await s.ainvoke(cur, **kwargs)
        return cur

    def __repr__(self) -> str:
        return " | ".join(repr(s) for s in self.steps)


# ==================== Parallel ====================


class RunnableParallel(Runnable):
    """字典并发：同一 input 喂给多个 Runnable，返回 ``{key: output}``"""

    def __init__(self, mapping: Dict[str, Any]):
        self.runnables: Dict[str, Runnable] = {
            k: _to_runnable(v) for k, v in mapping.items()
        }

    def invoke(self, input: Any, **kwargs: Any) -> Dict[str, Any]:
        return {k: r.invoke(input, **kwargs) for k, r in self.runnables.items()}

    async def ainvoke(self, input: Any, **kwargs: Any) -> Dict[str, Any]:
        keys = list(self.runnables.keys())
        results = await asyncio.gather(
            *[r.ainvoke(input, **kwargs) for r in self.runnables.values()]
        )
        return dict(zip(keys, results))


# ==================== Branch ====================


class RunnableBranch(Runnable):
    """条件分支：第一个 ``predicate(input)`` 为真的 ``runnable`` 被执行；
    所有都不匹配走 ``default``"""

    def __init__(
        self,
        branches: List[Tuple[Callable[[Any], bool], Any]],
        default: Any,
    ):
        self.branches: List[Tuple[Callable[[Any], bool], Runnable]] = [
            (pred, _to_runnable(r)) for pred, r in branches
        ]
        self.default: Runnable = _to_runnable(default)

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        for pred, r in self.branches:
            try:
                if pred(input):
                    return r.invoke(input, **kwargs)
            except Exception:
                continue
        return self.default.invoke(input, **kwargs)

    async def ainvoke(self, input: Any, **kwargs: Any) -> Any:
        for pred, r in self.branches:
            try:
                hit = pred(input)
                if inspect.isawaitable(hit):
                    hit = await hit
                if hit:
                    return await r.ainvoke(input, **kwargs)
            except Exception:
                continue
        return await self.default.ainvoke(input, **kwargs)


# ==================== 便捷 helper ====================


def prompt(template: str) -> Runnable:
    """字符串模板 Runnable：input dict → str.format(**input)"""

    def _format(input: Any) -> str:
        if isinstance(input, dict):
            return template.format(**input)
        return template.format(input=input)

    return RunnableLambda(_format, name=f"prompt({template[:30]!r})")


def parser_str() -> Runnable:
    """从 ``LLMResponse`` / 任意对象提取字符串

    优先取 ``.content``；fallback ``str(obj)``。
    """

    def _parse(obj: Any) -> str:
        content = getattr(obj, "content", None)
        if isinstance(content, str):
            return content
        return str(obj)

    return RunnableLambda(_parse, name="parser_str")


def parser_json() -> Runnable:
    """从 ``LLMResponse.content`` 解析 JSON；失败抛 ValueError"""
    import json

    def _parse(obj: Any) -> Any:
        content = getattr(obj, "content", None)
        text = content if isinstance(content, str) else str(obj)
        text = text.strip()
        # 剥 ```json 围栏
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 2:
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"parser_json 解析失败: {e}; 原文: {text[:200]}")

    return RunnableLambda(_parse, name="parser_json")


def passthrough() -> Runnable:
    """恒等 Runnable（input → input），用于 RunnableParallel 占位"""
    return RunnableLambda(lambda x: x, name="passthrough")


def assign(**kwargs: Any) -> Runnable:
    """把 dict input 与新字段合并

    例：``assign(timestamp=lambda x: now())`` 给每个流过的 dict 加一个 timestamp 字段。
    """

    def _assign(input: Any) -> Dict[str, Any]:
        if not isinstance(input, dict):
            raise TypeError(f"assign 期望 dict input，得到 {type(input).__name__}")
        out = dict(input)
        for k, v in kwargs.items():
            out[k] = v(input) if callable(v) else v
        return out

    return RunnableLambda(_assign, name="assign")


__all__ = [
    "Runnable",
    "RunnableLambda",
    "RunnableAdapter",
    "RunnableSequence",
    "RunnableParallel",
    "RunnableBranch",
    "prompt",
    "parser_str",
    "parser_json",
    "passthrough",
    "assign",
]
