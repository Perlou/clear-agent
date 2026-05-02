"""并发执行原语 —— 多 tool_calls 并行 / 异步聚合

按 plan §三 RC 阶段补：节点内多 tool_calls 默认顺序 ``for tc in tool_calls`` 优化为
并发。提供两套 API：

- ``run_tools_parallel(tool_calls, registry, max_workers=4)`` —— 同步入口
  内用 ``ThreadPoolExecutor`` 并发；适合纯 IO 密集工具
- ``arun_tools_parallel(tool_calls, registry, max_concurrency=4)`` —— async 入口
  用 ``asyncio.gather`` + ``Semaphore`` 限流；适合用户已经在 async 场景下

两者都返回 ``List[Dict]`` ``[{"tool_call_id", "name", "content", "error"}]``，
保持与 tool_calls 顺序对齐。

不修改现有 ``_react_graph.py`` / ``_simple_graph.py``（保持向后兼容）；
用户可在自定义 graph 节点里显式调用本模块。
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Awaitable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..tools.registry import ToolRegistry


logger = logging.getLogger(__name__)


# ==================== 单条执行 ====================


def _execute_one(tool_call: Any, registry: "ToolRegistry") -> Dict[str, Any]:
    """执行单个 tool_call，返回标准化结果

    Args:
        tool_call: 含 ``name`` / ``arguments`` / ``id`` 属性的对象（``ToolCall`` 或 dict）
        registry: ClearAgent ``ToolRegistry``

    Returns:
        ``{"tool_call_id", "name", "content", "error"}``，``error`` 为 None 表示成功
    """
    tool_name = getattr(tool_call, "name", None) or (
        tool_call.get("name") if isinstance(tool_call, dict) else None
    )
    tool_call_id = getattr(tool_call, "id", None) or (
        tool_call.get("id") if isinstance(tool_call, dict) else None
    )
    args_raw = getattr(tool_call, "arguments", None) or (
        tool_call.get("arguments") if isinstance(tool_call, dict) else None
    )

    out: Dict[str, Any] = {
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": "",
        "error": None,
    }

    # 解析参数
    args: Dict[str, Any]
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError as e:
            out["content"] = f"参数解析失败: {e}"
            out["error"] = "JSON_DECODE_ERROR"
            return out
    elif isinstance(args_raw, dict):
        args = args_raw
    else:
        args = {}

    # 查工具
    tool = None
    if registry is not None:
        get_fn = getattr(registry, "get_tool", None)
        if callable(get_fn) and tool_name:
            tool = get_fn(tool_name)
    if tool is None:
        out["content"] = f"工具 {tool_name} 未注册"
        out["error"] = "TOOL_NOT_FOUND"
        return out

    # 执行
    try:
        resp = tool.run_with_timing(args)
        out["content"] = getattr(resp, "text", None) or str(resp)
    except Exception as e:
        out["content"] = f"工具执行失败: {e}"
        out["error"] = type(e).__name__
    return out


# ==================== 同步并发 ====================


def run_tools_parallel(
    tool_calls: List[Any],
    registry: "ToolRegistry",
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """用 ``ThreadPoolExecutor`` 并发执行多个 tool_calls

    顺序保留：返回结果与 tool_calls 入参顺序一致。

    Args:
        tool_calls: ``ToolCall`` 对象或 dict 列表
        registry: ``ToolRegistry``
        max_workers: 最大并发线程数（默认 4）

    Returns:
        ``[{"tool_call_id", "name", "content", "error"}]`` 与 tool_calls 同序
    """
    if not tool_calls:
        return []
    if len(tool_calls) == 1 or max_workers <= 1:
        return [_execute_one(tc, registry) for tc in tool_calls]

    results: List[Optional[Dict[str, Any]]] = [None] * len(tool_calls)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_execute_one, tc, registry): i
            for i, tc in enumerate(tool_calls)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = {
                    "tool_call_id": getattr(tool_calls[idx], "id", None),
                    "name": getattr(tool_calls[idx], "name", None),
                    "content": f"未捕获异常: {e}",
                    "error": type(e).__name__,
                }
    return [r for r in results if r is not None]


# ==================== 异步并发 ====================


async def arun_tools_parallel(
    tool_calls: List[Any],
    registry: "ToolRegistry",
    max_concurrency: int = 4,
) -> List[Dict[str, Any]]:
    """用 ``asyncio.gather`` + ``Semaphore`` 限流并发执行

    单个工具优先用 ``tool.arun(args)`` 真异步；缺失则降级到线程池。
    """
    if not tool_calls:
        return []

    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def _aexecute_one(tc: Any) -> Dict[str, Any]:
        async with sem:
            tool_name = getattr(tc, "name", None) or (
                tc.get("name") if isinstance(tc, dict) else None
            )
            tool_call_id = getattr(tc, "id", None) or (
                tc.get("id") if isinstance(tc, dict) else None
            )
            args_raw = getattr(tc, "arguments", None) or (
                tc.get("arguments") if isinstance(tc, dict) else None
            )
            out: Dict[str, Any] = {
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": "",
                "error": None,
            }
            # 解析 args
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError as e:
                    out["content"] = f"参数解析失败: {e}"
                    out["error"] = "JSON_DECODE_ERROR"
                    return out
            elif isinstance(args_raw, dict):
                args = args_raw
            else:
                args = {}

            tool = None
            if registry is not None:
                get_fn = getattr(registry, "get_tool", None)
                if callable(get_fn) and tool_name:
                    tool = get_fn(tool_name)
            if tool is None:
                out["content"] = f"工具 {tool_name} 未注册"
                out["error"] = "TOOL_NOT_FOUND"
                return out

            # 优先 async run
            try:
                arun = getattr(tool, "arun", None)
                if callable(arun):
                    resp = await arun(args)
                else:
                    # 降级到 run（在事件循环里直接调）
                    resp = await asyncio.to_thread(tool.run_with_timing, args)
                out["content"] = getattr(resp, "text", None) or str(resp)
            except Exception as e:
                out["content"] = f"工具执行失败: {e}"
                out["error"] = type(e).__name__
            return out

    return await asyncio.gather(*[_aexecute_one(tc) for tc in tool_calls])


# ==================== 通用并发 helper ====================


async def gather_with_concurrency(
    coros: List[Awaitable[Any]], max_concurrency: int = 4
) -> List[Any]:
    """异步并发限流：同一时刻最多 ``max_concurrency`` 个 coroutine 在跑

    适合用户在 graph 节点里需要并发调多个 LLM / RAG 检索的场景。
    """
    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def _wrap(coro: Awaitable[Any]) -> Any:
        async with sem:
            return await coro

    return await asyncio.gather(*[_wrap(c) for c in coros])


__all__ = [
    "run_tools_parallel",
    "arun_tools_parallel",
    "gather_with_concurrency",
]
