"""Resilience 原语 —— Retry / Fallback / 负载均衡

为 LLM / 工具 / 任意 callable 提供生产级容错：

- ``with_retry(fn, max_attempts, backoff, retry_on)`` 失败重试 + 指数退避
- ``with_fallbacks(primary, fallbacks)`` 主调失败时按序尝试备选
- ``@retry`` / ``@fallback`` 装饰器形态

设计原则：
- **零外部依赖**（不引入 ``tenacity``）
- 同步 + 异步双轨
- 可注入 ``on_retry`` / ``on_fallback`` 回调用于日志 / 监控
- 默认重试条件：``Exception`` 子类（用户可白名单 ``retry_on=(MyError,)``）
- jitter 防雷击（多副本同时重试不会击穿后端）

典型用法::

    @retry(max_attempts=3, backoff=0.5)
    def call_api():
        return llm.invoke(...)

    safe_llm = with_fallbacks(primary_llm, [backup_llm_1, backup_llm_2])
    response = safe_llm.invoke(messages)   # 主失败自动回退到次选
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import random
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)


logger = logging.getLogger(__name__)


T = TypeVar("T")
ExcTypes = Union[Type[BaseException], Tuple[Type[BaseException], ...]]


# ==================== 配置 ====================


@dataclass
class RetryPolicy:
    """重试策略

    Attributes:
        max_attempts: 总尝试次数（含首次）；``3`` 表示首次 + 重试 2 次
        backoff: 基础退避秒数（指数：``backoff * 2^(attempt-1)``）
        max_backoff: 退避上限
        jitter: 抖动比例 [0, 1]；最终延迟 = backoff * (1 - jitter + 2*jitter*random)
        retry_on: 仅在这些异常类型时重试（其他异常立刻抛）
        on_retry: ``(attempt, exception, delay)`` 回调
    """

    max_attempts: int = 3
    backoff: float = 0.5
    max_backoff: float = 30.0
    jitter: float = 0.2
    retry_on: ExcTypes = Exception
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None

    def compute_delay(self, attempt: int) -> float:
        """attempt = 1, 2, 3, ... 计算第 attempt 次失败后的等待时间"""
        if attempt < 1:
            return 0.0
        base = min(self.backoff * (2 ** (attempt - 1)), self.max_backoff)
        if self.jitter > 0:
            j = max(0.0, min(1.0, self.jitter))
            base = base * (1.0 - j + 2.0 * j * random.random())
        return max(0.0, base)


# ==================== Retry 同步 ====================


def with_retry(
    fn: Callable[..., T],
    max_attempts: int = 3,
    backoff: float = 0.5,
    max_backoff: float = 30.0,
    jitter: float = 0.2,
    retry_on: ExcTypes = Exception,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> Callable[..., T]:
    """把任意同步 callable 包成带重试的版本（包装器形态）

    用法::

        safe_invoke = with_retry(llm.invoke, max_attempts=3, backoff=0.5)
        response = safe_invoke(messages)   # 自动重试
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        backoff=backoff,
        max_backoff=max_backoff,
        jitter=jitter,
        retry_on=retry_on,
        on_retry=on_retry,
    )
    return _wrap_sync(fn, policy)


def retry(
    max_attempts: int = 3,
    backoff: float = 0.5,
    max_backoff: float = 30.0,
    jitter: float = 0.2,
    retry_on: ExcTypes = Exception,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """``@retry`` 装饰器形态

    用法::

        @retry(max_attempts=3, retry_on=(ConnectionError,))
        def fetch():
            ...
    """

    def deco(fn: Callable[..., T]) -> Callable[..., T]:
        return with_retry(
            fn,
            max_attempts=max_attempts,
            backoff=backoff,
            max_backoff=max_backoff,
            jitter=jitter,
            retry_on=retry_on,
            on_retry=on_retry,
        )

    return deco


# ==================== Retry 异步 ====================


def with_retry_async(
    fn: Callable[..., Awaitable[T]],
    max_attempts: int = 3,
    backoff: float = 0.5,
    max_backoff: float = 30.0,
    jitter: float = 0.2,
    retry_on: ExcTypes = Exception,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> Callable[..., Awaitable[T]]:
    """async 版本的 with_retry"""
    policy = RetryPolicy(
        max_attempts=max_attempts,
        backoff=backoff,
        max_backoff=max_backoff,
        jitter=jitter,
        retry_on=retry_on,
        on_retry=on_retry,
    )
    return _wrap_async(fn, policy)


def aretry(
    max_attempts: int = 3,
    backoff: float = 0.5,
    max_backoff: float = 30.0,
    jitter: float = 0.2,
    retry_on: ExcTypes = Exception,
    on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """``@aretry`` 装饰器（async 版本）"""

    def deco(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        return with_retry_async(
            fn,
            max_attempts=max_attempts,
            backoff=backoff,
            max_backoff=max_backoff,
            jitter=jitter,
            retry_on=retry_on,
            on_retry=on_retry,
        )

    return deco


def _wrap_sync(fn: Callable[..., T], policy: RetryPolicy) -> Callable[..., T]:
    @functools.wraps(fn)
    def _inner(*args: Any, **kwargs: Any) -> T:
        last_err: Optional[BaseException] = None
        for attempt in range(1, policy.max_attempts + 1):
            try:
                return fn(*args, **kwargs)
            except policy.retry_on as e:
                last_err = e
                if attempt >= policy.max_attempts:
                    break
                delay = policy.compute_delay(attempt)
                if policy.on_retry:
                    try:
                        policy.on_retry(attempt, e, delay)
                    except Exception:
                        pass
                logger.warning(
                    f"⚠️ retry {attempt}/{policy.max_attempts - 1} after "
                    f"{type(e).__name__}: {e}; sleeping {delay:.2f}s"
                )
                if delay > 0:
                    time.sleep(delay)
        # 重试耗尽
        assert last_err is not None
        raise last_err

    return _inner


def _wrap_async(
    fn: Callable[..., Awaitable[T]], policy: RetryPolicy
) -> Callable[..., Awaitable[T]]:
    @functools.wraps(fn)
    async def _inner(*args: Any, **kwargs: Any) -> T:
        last_err: Optional[BaseException] = None
        for attempt in range(1, policy.max_attempts + 1):
            try:
                return await fn(*args, **kwargs)
            except policy.retry_on as e:
                last_err = e
                if attempt >= policy.max_attempts:
                    break
                delay = policy.compute_delay(attempt)
                if policy.on_retry:
                    try:
                        policy.on_retry(attempt, e, delay)
                    except Exception:
                        pass
                logger.warning(
                    f"⚠️ async retry {attempt}/{policy.max_attempts - 1} after "
                    f"{type(e).__name__}: {e}; sleeping {delay:.2f}s"
                )
                if delay > 0:
                    await asyncio.sleep(delay)
        assert last_err is not None
        raise last_err

    return _inner


# ==================== Fallback ====================


@dataclass
class FallbackResult:
    """fallback 链最终结果（仅在 ``return_result_obj=True`` 时返回）

    Attributes:
        value: 实际成功的返回值
        used_index: 哪个候选成功了（0=primary, 1=fallback[0], ...）
        errors: 此前每个候选的失败原因
    """

    value: Any
    used_index: int
    errors: List[BaseException] = field(default_factory=list)


def with_fallbacks(
    primary: Callable[..., T],
    fallbacks: List[Callable[..., T]],
    fallback_on: ExcTypes = Exception,
    on_fallback: Optional[Callable[[int, BaseException], None]] = None,
) -> Callable[..., T]:
    """主调失败 → 按序尝试 fallbacks

    Args:
        primary: 主调用 callable
        fallbacks: 备选 callable 列表（按顺序尝试）
        fallback_on: 仅在这些异常时回退（其他异常立刻抛）
        on_fallback: ``(fallback_index_starting_from_0, exception)`` 回调

    Returns:
        包装后的 callable，签名与 ``primary`` 一致
    """

    @functools.wraps(primary)
    def _inner(*args: Any, **kwargs: Any) -> T:
        try:
            return primary(*args, **kwargs)
        except fallback_on as e:
            errors: List[BaseException] = [e]
            logger.warning(f"⚠️ primary 失败，尝试 fallbacks: {type(e).__name__}: {e}")
            for i, fb in enumerate(fallbacks):
                if on_fallback:
                    try:
                        on_fallback(i, errors[-1])
                    except Exception:
                        pass
                try:
                    return fb(*args, **kwargs)
                except fallback_on as e2:
                    errors.append(e2)
                    logger.warning(
                        f"⚠️ fallback[{i}] 失败: {type(e2).__name__}: {e2}"
                    )
            # 全部失败 → 抛最后一个错误，并附带前面的 errors
            final = errors[-1]
            try:
                final.__notes__ = [
                    f"primary + {len(fallbacks)} fallbacks 全部失败"
                ]  # type: ignore[attr-defined]
            except Exception:
                pass
            raise final

    return _inner


def with_fallbacks_async(
    primary: Callable[..., Awaitable[T]],
    fallbacks: List[Callable[..., Awaitable[T]]],
    fallback_on: ExcTypes = Exception,
    on_fallback: Optional[Callable[[int, BaseException], None]] = None,
) -> Callable[..., Awaitable[T]]:
    """async fallback 链"""

    @functools.wraps(primary)
    async def _inner(*args: Any, **kwargs: Any) -> T:
        try:
            return await primary(*args, **kwargs)
        except fallback_on as e:
            errors: List[BaseException] = [e]
            logger.warning(
                f"⚠️ async primary 失败，尝试 fallbacks: {type(e).__name__}: {e}"
            )
            for i, fb in enumerate(fallbacks):
                if on_fallback:
                    try:
                        on_fallback(i, errors[-1])
                    except Exception:
                        pass
                try:
                    return await fb(*args, **kwargs)
                except fallback_on as e2:
                    errors.append(e2)
                    logger.warning(
                        f"⚠️ async fallback[{i}] 失败: {type(e2).__name__}: {e2}"
                    )
            raise errors[-1]

    return _inner


# ==================== 负载均衡（轮询 / 随机） ====================


def round_robin(
    candidates: List[Callable[..., T]],
) -> Callable[..., T]:
    """轮询负载均衡：每次调用按顺序选下一个候选

    所有候选共享同一签名，调用计数器线程安全（用 ``threading.Lock``）。
    """
    import threading

    if not candidates:
        raise ValueError("candidates 不能为空")
    state = {"index": 0}
    lock = threading.Lock()

    def _inner(*args: Any, **kwargs: Any) -> T:
        with lock:
            i = state["index"] % len(candidates)
            state["index"] += 1
        return candidates[i](*args, **kwargs)

    return _inner


def random_choice(
    candidates: List[Callable[..., T]],
    seed: Optional[int] = None,
) -> Callable[..., T]:
    """随机负载均衡：每次随机选一个候选"""
    if not candidates:
        raise ValueError("candidates 不能为空")
    rng = random.Random(seed)

    def _inner(*args: Any, **kwargs: Any) -> T:
        return rng.choice(candidates)(*args, **kwargs)

    return _inner


__all__ = [
    "RetryPolicy",
    "FallbackResult",
    "with_retry",
    "retry",
    "with_retry_async",
    "aretry",
    "with_fallbacks",
    "with_fallbacks_async",
    "round_robin",
    "random_choice",
]
