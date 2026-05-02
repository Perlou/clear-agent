"""Callbacks 协议 —— 跨模块事件订阅

提供 LangChain 风格的 ``BaseCallbackHandler`` 11+ hooks 协议，让用户
统一订阅 LLM / 工具 / Agent / Graph / Retriever 的生命周期事件。

应用场景：
- 自定义 trace 输出（如导出到 LangSmith / Langfuse）
- 实时进度展示（前端 SSE 推送）
- 性能监控（延迟 / token 累计）
- 调试 / 审计

11 个 hooks（与 LangChain 对齐）::

    # LLM
    on_llm_start(prompts, model, **kw)
    on_llm_end(response, **kw)
    on_llm_error(error, **kw)
    on_llm_new_token(token, **kw)        # 流式

    # Tool
    on_tool_start(tool_name, arguments, **kw)
    on_tool_end(tool_name, response, **kw)
    on_tool_error(tool_name, error, **kw)

    # Agent / Graph 节点
    on_agent_start(agent_name, input, **kw)
    on_agent_end(agent_name, output, **kw)
    on_node_start(node_name, state, **kw)
    on_node_end(node_name, state, **kw)

    # Retriever（RAG）
    on_retriever_start(query, **kw)
    on_retriever_end(results, **kw)
"""

from __future__ import annotations

import logging
import time
from abc import ABC
from typing import Any, Awaitable, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


# 全部 hook 名称
HOOK_NAMES = (
    "on_llm_start",
    "on_llm_end",
    "on_llm_error",
    "on_llm_new_token",
    "on_tool_start",
    "on_tool_end",
    "on_tool_error",
    "on_agent_start",
    "on_agent_end",
    "on_node_start",
    "on_node_end",
    "on_retriever_start",
    "on_retriever_end",
)


class BaseCallbackHandler(ABC):
    """Callback handler 基类

    所有 hook 默认是 no-op；子类按需重写。Hook 抛异常时由 ``CallbackManager``
    捕获记录但不传播（避免 callback 失败影响主流程）。
    """

    # ==================== LLM hooks ====================

    def on_llm_start(
        self, prompts: List[Any], model: Optional[str] = None, **kwargs: Any
    ) -> None:
        """LLM 调用前触发（同步入口前 / async 入口前 / 流式入口前）"""

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """LLM 调用成功后触发；``response`` 类型为 ``LLMResponse`` / ``LLMToolResponse``"""

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """LLM 调用失败"""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """流式调用每收到一个 token 片段"""

    # ==================== Tool hooks ====================

    def on_tool_start(
        self, tool_name: str, arguments: Dict[str, Any], **kwargs: Any
    ) -> None:
        """工具调用前"""

    def on_tool_end(self, tool_name: str, response: Any, **kwargs: Any) -> None:
        """工具调用成功"""

    def on_tool_error(
        self, tool_name: str, error: BaseException, **kwargs: Any
    ) -> None:
        """工具调用异常"""

    # ==================== Agent hooks ====================

    def on_agent_start(
        self, agent_name: str, input: Any, **kwargs: Any
    ) -> None:
        """Agent.run / graph.invoke 入口前"""

    def on_agent_end(self, agent_name: str, output: Any, **kwargs: Any) -> None:
        """Agent.run / graph.invoke 退出"""

    # ==================== Graph node hooks ====================

    def on_node_start(self, node_name: str, state: Any, **kwargs: Any) -> None:
        """图节点开始执行"""

    def on_node_end(self, node_name: str, state: Any, **kwargs: Any) -> None:
        """图节点结束执行"""

    # ==================== Retriever hooks ====================

    def on_retriever_start(self, query: str, **kwargs: Any) -> None:
        """RAG 检索前"""

    def on_retriever_end(self, results: List[Any], **kwargs: Any) -> None:
        """RAG 检索后"""


class CallbackManager:
    """注册并广播事件到多个 ``BaseCallbackHandler``

    Args:
        handlers: 初始注册列表
        swallow_errors: handler 抛异常时是否吞掉（默认 True，仅打 warning）

    用法::

        mgr = CallbackManager()
        mgr.add(LoggingCallbackHandler())
        mgr.add(MetricsCallbackHandler())

        mgr.fire("on_tool_start", tool_name="calc", arguments={"x": 1})
        # 等价于对每个 handler 调 handler.on_tool_start(...)
    """

    def __init__(
        self,
        handlers: Optional[List[BaseCallbackHandler]] = None,
        swallow_errors: bool = True,
    ):
        self.handlers: List[BaseCallbackHandler] = list(handlers or [])
        self.swallow_errors = swallow_errors

    def add(self, handler: BaseCallbackHandler) -> None:
        if handler not in self.handlers:
            self.handlers.append(handler)

    def remove(self, handler: BaseCallbackHandler) -> bool:
        try:
            self.handlers.remove(handler)
            return True
        except ValueError:
            return False

    def clear(self) -> None:
        self.handlers.clear()

    def __len__(self) -> int:
        return len(self.handlers)

    def fire(self, hook: str, *args: Any, **kwargs: Any) -> None:
        """同步广播事件到所有 handler"""
        if hook not in HOOK_NAMES:
            raise ValueError(f"未知 hook: {hook}; 已知: {HOOK_NAMES}")
        for h in list(self.handlers):
            fn = getattr(h, hook, None)
            if not callable(fn):
                continue
            try:
                fn(*args, **kwargs)
            except Exception as e:
                if self.swallow_errors:
                    logger.warning(
                        f"⚠️ callback {type(h).__name__}.{hook} 抛异常（已吞）: {e}"
                    )
                else:
                    raise

    async def afire(self, hook: str, *args: Any, **kwargs: Any) -> None:
        """异步广播事件 —— hook 可同步或 async 实现"""
        import inspect as _inspect

        if hook not in HOOK_NAMES:
            raise ValueError(f"未知 hook: {hook}; 已知: {HOOK_NAMES}")
        for h in list(self.handlers):
            fn = getattr(h, hook, None)
            if not callable(fn):
                continue
            try:
                result = fn(*args, **kwargs)
                if _inspect.isawaitable(result):
                    await result
            except Exception as e:
                if self.swallow_errors:
                    logger.warning(
                        f"⚠️ async callback {type(h).__name__}.{hook} 抛异常（已吞）: {e}"
                    )
                else:
                    raise


# ==================== 内置 handlers ====================


class LoggingCallbackHandler(BaseCallbackHandler):
    """把所有事件打印到 ``logging``"""

    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level
        self._logger = logging.getLogger("clear_agent.callbacks")

    def _log(self, msg: str) -> None:
        self._logger.log(self.log_level, msg)

    def on_llm_start(self, prompts, model=None, **kw):
        n = len(prompts) if hasattr(prompts, "__len__") else "?"
        self._log(f"🧠 LLM start: model={model} n_messages={n}")

    def on_llm_end(self, response, **kw):
        usage = getattr(response, "usage", None) or {}
        total = usage.get("total_tokens", "?") if isinstance(usage, dict) else "?"
        self._log(f"✅ LLM end: tokens={total}")

    def on_llm_error(self, error, **kw):
        self._log(f"❌ LLM error: {type(error).__name__}: {error}")

    def on_tool_start(self, tool_name, arguments, **kw):
        self._log(f"🔧 Tool start: {tool_name} args={arguments}")

    def on_tool_end(self, tool_name, response, **kw):
        status = getattr(response, "status", "?")
        if hasattr(status, "value"):
            status = status.value
        self._log(f"✅ Tool end: {tool_name} status={status}")

    def on_tool_error(self, tool_name, error, **kw):
        self._log(f"❌ Tool error: {tool_name}: {error}")

    def on_node_start(self, node_name, state, **kw):
        self._log(f"▶ Node start: {node_name}")

    def on_node_end(self, node_name, state, **kw):
        self._log(f"⏹ Node end: {node_name}")

    def on_retriever_start(self, query, **kw):
        self._log(f"🔍 Retriever start: query={query[:60]!r}")

    def on_retriever_end(self, results, **kw):
        n = len(results) if hasattr(results, "__len__") else "?"
        self._log(f"✅ Retriever end: hits={n}")


class MetricsCallbackHandler(BaseCallbackHandler):
    """累计 LLM / 工具 / 节点的调用次数与延迟"""

    def __init__(self):
        self.metrics: Dict[str, Dict[str, Any]] = {
            "llm": {"calls": 0, "errors": 0, "total_tokens": 0, "total_latency_ms": 0},
            "tool": {"calls": 0, "errors": 0, "by_name": {}},
            "node": {"calls": 0, "by_name": {}, "total_latency_ms": 0},
            "retriever": {"calls": 0, "total_hits": 0},
        }
        self._llm_start_ts: Dict[int, float] = {}
        self._tool_start_ts: Dict[tuple, float] = {}
        self._node_start_ts: Dict[str, float] = {}

    def on_llm_start(self, prompts, model=None, **kw):
        self.metrics["llm"]["calls"] += 1
        # 用 prompts id 关联 start/end（可能并发）
        self._llm_start_ts[id(prompts)] = time.time()

    def on_llm_end(self, response, **kw):
        usage = getattr(response, "usage", None) or {}
        if isinstance(usage, dict):
            self.metrics["llm"]["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        # 简化：从最近一个 start 取延迟
        if self._llm_start_ts:
            _, ts = self._llm_start_ts.popitem()
            self.metrics["llm"]["total_latency_ms"] += int((time.time() - ts) * 1000)

    def on_llm_error(self, error, **kw):
        self.metrics["llm"]["errors"] += 1

    def on_tool_start(self, tool_name, arguments, **kw):
        self.metrics["tool"]["calls"] += 1
        by_name = self.metrics["tool"]["by_name"].setdefault(
            tool_name, {"calls": 0, "errors": 0}
        )
        by_name["calls"] += 1
        self._tool_start_ts[(tool_name, id(arguments))] = time.time()

    def on_tool_end(self, tool_name, response, **kw):
        pass  # 占位 hook；可扩展

    def on_tool_error(self, tool_name, error, **kw):
        self.metrics["tool"]["errors"] += 1
        by_name = self.metrics["tool"]["by_name"].setdefault(
            tool_name, {"calls": 0, "errors": 0}
        )
        by_name["errors"] += 1

    def on_node_start(self, node_name, state, **kw):
        self.metrics["node"]["calls"] += 1
        by_name = self.metrics["node"]["by_name"].setdefault(node_name, 0)
        self.metrics["node"]["by_name"][node_name] = by_name + 1
        self._node_start_ts[node_name] = time.time()

    def on_node_end(self, node_name, state, **kw):
        ts = self._node_start_ts.pop(node_name, None)
        if ts is not None:
            self.metrics["node"]["total_latency_ms"] += int(
                (time.time() - ts) * 1000
            )

    def on_retriever_start(self, query, **kw):
        self.metrics["retriever"]["calls"] += 1

    def on_retriever_end(self, results, **kw):
        n = len(results) if hasattr(results, "__len__") else 0
        self.metrics["retriever"]["total_hits"] += n

    def reset(self) -> None:
        """重置全部计数"""
        self.__init__()  # type: ignore[misc]


__all__ = [
    "HOOK_NAMES",
    "BaseCallbackHandler",
    "CallbackManager",
    "LoggingCallbackHandler",
    "MetricsCallbackHandler",
]
