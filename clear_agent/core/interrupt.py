"""Human-in-the-Loop（HITL）核心原语

提供 ``interrupt(payload)`` 函数和配套异常，让节点能在执行中暂停，等待外部
``compiled.resume(thread_id, value=...)`` 注入决策后从同一节点继续。

设计要点：
- ``interrupt()`` 用 ContextVar 维护当前 run 的上下文
- 第一次进入节点遇到 ``interrupt(payload)`` → 抛 ``GraphInterrupt``
  → CompiledGraph 捕获 → 写 ``source=interrupt`` 的 ckpt → 抛 ``GraphPaused``
- resume 时 CompiledGraph 把 ``value`` 写入 ContextVar，节点函数重入，
  ``interrupt()`` 直接返回该 value 而非再次抛
- ``GraphInterrupt`` 继承 BaseException 而非 Exception，避免被节点的
  ``try/except Exception`` 误吞


"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class _RunContext:
    """单次 graph 执行的运行时上下文（线程/任务级）

    支持节点内多个 ``interrupt()`` 调用：
    - resume_values 是历史回放队列（前 N 次 interrupt 重入时按序消费）
    - live_value 是本次 resume 注入的最新值（消费后追加到 resume_values）
    - counter 跟踪当前节点已消费的 interrupt 数；每次进入新节点时由 graph 重置

    Attributes:
        thread_id: 当前 run 的 thread id
        resume_values: 历史 resume 值队列（按消费顺序）
        live_value: 最新一次 resume 注入的值
        has_live_value: live_value 是否可用
        counter: 当前节点已成功消费的 interrupt 数
    """

    thread_id: Optional[str] = None
    resume_values: List[Any] = field(default_factory=list)
    live_value: Any = None
    has_live_value: bool = False
    counter: int = 0

    def reset_counter(self) -> None:
        """进入新节点前重置计数器"""
        self.counter = 0


# ContextVar 保证多线程 / asyncio 任务隔离
_current_run_ctx: contextvars.ContextVar[Optional[_RunContext]] = contextvars.ContextVar(
    "clear_agent_run_ctx", default=None
)


# ==================== 异常 ====================


class GraphInterrupt(BaseException):
    """节点内调 ``interrupt()`` 时抛出，由 CompiledGraph 捕获

    继承 BaseException 而非 Exception，避免被节点函数的
    ``try/except Exception`` 块误吞。
    """

    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload
        super().__init__(f"GraphInterrupt: {payload.get('type', 'unknown')}")


class GraphPaused(Exception):
    """CompiledGraph 在写完 interrupt ckpt 后抛给调用方的信号

    调用方应捕获此异常，向用户/前端展示 ``payload``，待用户决策后调
    ``compiled.resume(thread_id, value=...)`` 续跑。
    """

    def __init__(
        self,
        thread_id: str,
        checkpoint_id: str,
        payload: Dict[str, Any],
    ):
        self.thread_id = thread_id
        self.checkpoint_id = checkpoint_id
        self.payload = payload
        super().__init__(
            f"GraphPaused at thread={thread_id} ckpt={checkpoint_id} "
            f"type={payload.get('type', 'unknown')}"
        )


class InterruptExpiredError(Exception):
    """interrupt ckpt 超过 hitl_interrupt_ttl_seconds 后再 resume 时抛出"""


# ==================== 公开 API ====================


def interrupt(payload: Dict[str, Any]) -> Any:
    """在节点内暂停执行，等待外部 resume

    用法:
        def risky_node(state):
            decision = interrupt({
                "type": "approval",
                "message": "Send email?",
                "draft": state["draft"],
            })
            if not decision.get("approved"):
                return {"messages": [...]}
            ...

    第一次调用：抛 GraphInterrupt → CompiledGraph 捕获 → 写 ckpt → 抛 GraphPaused。
    Resume 后再次调用：直接返回 ``compiled.resume(thread_id, value=...)`` 注入的 value。

    支持节点内多个 interrupt：每次 resume 重入节点时，前 N 个 interrupt
    按 resume 历史顺序回放历史值；第 N+1 个返回最新注入的 value；再后面的
    继续 raise 触发新一轮中断。

    Args:
        payload: 任意 JSON 可序列化字典；至少建议含 'type' 字段
                ('approval' / 'edit' / 'tool_validation' / 'custom')

    Returns:
        Resume 时注入的 value

    Raises:
        GraphInterrupt: 没有可用的回放/live 值时抛
        RuntimeError: 不在 graph 节点内调用
    """
    ctx = _current_run_ctx.get()
    if ctx is None:
        raise RuntimeError(
            "interrupt() 只能在 graph 节点函数内调用。"
            "请确保你在 CompiledGraph.invoke()/stream() 触发的节点里使用。"
        )

    # 1. 历史回放：消费 resume_values[counter]
    if ctx.counter < len(ctx.resume_values):
        val = ctx.resume_values[ctx.counter]
        ctx.counter += 1
        return val

    # 2. live 值：消费一次性注入并追加到历史
    if ctx.has_live_value:
        val = ctx.live_value
        ctx.live_value = None
        ctx.has_live_value = False
        ctx.resume_values.append(val)
        ctx.counter += 1
        return val

    # 3. 无可用值：触发新一轮中断
    raise GraphInterrupt(payload)


# ==================== 内部工具（CompiledGraph 用） ====================


def _set_run_ctx(ctx: _RunContext) -> contextvars.Token:
    """CompiledGraph 入口处调，设置当前 run 的 context"""
    return _current_run_ctx.set(ctx)


def _reset_run_ctx(token: contextvars.Token) -> None:
    """CompiledGraph 出口处调，恢复上层 context"""
    _current_run_ctx.reset(token)


def _get_run_ctx() -> Optional[_RunContext]:
    """读当前 run context（CompiledGraph 内部用）"""
    return _current_run_ctx.get()


__all__ = [
    "interrupt",
    "GraphInterrupt",
    "GraphPaused",
    "InterruptExpiredError",
]
