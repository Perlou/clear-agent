"""Resilience（Retry / Fallback / 负载均衡）演示

跑这个文件可看到：
1. ``@retry`` 装饰器：自动重试 + 指数退避 + 异常白名单
2. ``with_fallbacks`` 包装器：主调失败按序回退到次选
3. ``round_robin`` / ``random_choice`` 负载均衡

不依赖外部服务；用纯 Python 函数模拟瞬时故障。

运行：
    python examples/resilience_demo.py
"""

from __future__ import annotations

import random
import time

from clear_agent.core.resilience import (
    aretry,
    random_choice,
    retry,
    round_robin,
    with_fallbacks,
    with_retry,
)


# ==================================================================
# Part 1: @retry —— 自动重试瞬时故障
# ==================================================================


def demo_retry() -> None:
    print("=" * 60)
    print("Part 1: @retry —— 模拟 API 瞬时抖动")
    print("=" * 60)

    attempt = {"n": 0}

    @retry(max_attempts=4, backoff=0.05, retry_on=(ConnectionError,))
    def call_flaky_api() -> str:
        attempt["n"] += 1
        if attempt["n"] < 3:
            print(f"  [attempt {attempt['n']}] 模拟连接失败...")
            raise ConnectionError(f"transient {attempt['n']}")
        return f"success at attempt {attempt['n']}"

    print(f"  → {call_flaky_api()}")

    # 重置：演示其他异常立即抛
    print()
    print("  其他异常类型不会重试:")
    @retry(max_attempts=3, backoff=0.01, retry_on=(ConnectionError,))
    def value_error_call():
        raise ValueError("not a network error")

    try:
        value_error_call()
    except ValueError as e:
        print(f"  → 立即抛 ValueError: {e}")


# ==================================================================
# Part 2: with_fallbacks —— 主调失败回退到次选
# ==================================================================


def demo_fallbacks() -> None:
    print()
    print("=" * 60)
    print("Part 2: with_fallbacks —— 主 LLM 挂了用备用")
    print("=" * 60)

    def primary_llm(prompt: str) -> str:
        print(f"  [primary] 尝试...")
        raise RuntimeError("primary LLM 503")

    def backup_openai(prompt: str) -> str:
        print(f"  [backup_openai] 尝试...")
        raise RuntimeError("backup_openai 429 rate limit")

    def backup_local(prompt: str) -> str:
        print(f"  [backup_local] 尝试...")
        return f"locally generated for: {prompt}"

    safe_llm = with_fallbacks(primary_llm, [backup_openai, backup_local])
    result = safe_llm("How to refactor?")
    print(f"  → {result}")


# ==================================================================
# Part 3: round_robin —— 多副本负载均衡
# ==================================================================


def demo_load_balance() -> None:
    print()
    print("=" * 60)
    print("Part 3: round_robin / random_choice —— 多副本均衡")
    print("=" * 60)

    def replica(i: int):
        def _call(prompt: str) -> str:
            return f"replica-{i}: {prompt[:20]}"

        return _call

    print("  Round Robin（按顺序循环）:")
    rr = round_robin([replica(0), replica(1), replica(2)])
    for q in ["q1", "q2", "q3", "q4", "q5"]:
        print(f"    {rr(q)}")

    print()
    print("  Random Choice（随机选）:")
    rc = random_choice([replica(0), replica(1), replica(2)], seed=42)
    for q in ["q1", "q2", "q3", "q4", "q5"]:
        print(f"    {rc(q)}")


# ==================================================================
# Part 4: 组合：retry + fallback
# ==================================================================


def demo_compose() -> None:
    print()
    print("=" * 60)
    print("Part 4: 组合 —— 先 retry，仍失败再 fallback")
    print("=" * 60)

    fail_count = {"primary": 0, "backup": 0}

    @retry(max_attempts=2, backoff=0.01)
    def primary(prompt: str) -> str:
        fail_count["primary"] += 1
        raise ConnectionError(f"primary fail #{fail_count['primary']}")

    def backup(prompt: str) -> str:
        fail_count["backup"] += 1
        return f"backup result for: {prompt}"

    safe = with_fallbacks(primary, [backup])
    print(f"  → {safe('hello')}")
    print(f"  primary 总尝试: {fail_count['primary']} 次（含 retry）")
    print(f"  backup 总尝试: {fail_count['backup']} 次")


def main() -> None:
    demo_retry()
    demo_fallbacks()
    demo_load_balance()
    demo_compose()
    print()
    print("✅ Resilience demo 跑通")


if __name__ == "__main__":
    main()
