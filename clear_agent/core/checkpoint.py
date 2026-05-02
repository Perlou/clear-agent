"""Checkpointer 协议与实现

为 ClearAgent 2.0 StateGraph 提供 per-node 状态快照与恢复能力。

设计要点：
- BaseCheckpointer 抽象，支持同步/异步两套接口
- InMemoryCheckpointer：开发与单元测试默认
- JsonFileCheckpointer / SqliteCheckpointer 在 W2 实现
- thread_id 隔离不同会话；checkpoint_id 单调递增（uuid7 时间排序友好）

详见：project_docs/02-checkpoint-and-resume.md
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


def _uuid7() -> str:
    """生成时间排序友好的 ID（不依赖 uuid7 库）

    格式: <millis_hex>-<random_hex>
    长度足够避免冲突，且字典序与时间序一致
    """
    millis = int(time.time() * 1000)
    return f"{millis:014x}-{uuid.uuid4().hex[:12]}"


@dataclass
class Checkpoint:
    """单个节点边界的 state 快照

    Attributes:
        id: 唯一 ID（_uuid7 生成）
        thread_id: 会话/用户隔离 ID
        parent_id: 上一个 checkpoint 的 ID（None 表示首个）
        state: 序列化后的 State 字段（任意可 json 序列化的字典）
        next_nodes: 即将执行的下一组节点（resume 时回到此处）
        created_at: 创建时间
        metadata: 扩展信息，惯例字段：
            source: "loop" | "interrupt" | "error" | "user_save"
            node: 上一个执行的节点名
            payload: source=interrupt 时的中断 payload
    """

    id: str
    thread_id: str
    parent_id: Optional[str]
    state: Dict[str, Any]
    next_nodes: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转为可 json 序列化的字典"""
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Checkpoint":
        """从字典恢复"""
        d = dict(d)  # 浅拷贝避免修改入参
        if isinstance(d.get("created_at"), str):
            d["created_at"] = datetime.fromisoformat(d["created_at"])
        return cls(**d)


class BaseCheckpointer(ABC):
    """Checkpointer 协议基类

    实现类至少要提供同步 put/get_tuple/list 三个方法。
    异步对偶 a* 默认基于 run_in_executor 包装；真异步实现可覆写。
    """

    @abstractmethod
    def put(self, checkpoint: Checkpoint) -> None:
        """保存 checkpoint"""
        raise NotImplementedError

    @abstractmethod
    def get_tuple(
        self, thread_id: str, checkpoint_id: Optional[str] = None
    ) -> Optional[Checkpoint]:
        """获取指定 checkpoint，checkpoint_id=None 时返回该 thread 最新的

        Returns:
            Checkpoint 或 None（thread 不存在或为空）
        """
        raise NotImplementedError

    @abstractmethod
    def list(
        self,
        thread_id: str,
        before: Optional[str] = None,
        limit: int = 50,
    ) -> List[Checkpoint]:
        """列出指定 thread 的 checkpoint，按时间倒序

        Args:
            thread_id: 会话 ID
            before: 仅返回此 ID 之前创建的 checkpoint（用于分页）
            limit: 最多返回数

        Returns:
            checkpoint 列表，按 created_at 倒序
        """
        raise NotImplementedError

    # ==================== 异步接口（默认线程池包装）====================

    async def aput(self, checkpoint: Checkpoint) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.put, checkpoint)

    async def aget_tuple(
        self, thread_id: str, checkpoint_id: Optional[str] = None
    ) -> Optional[Checkpoint]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.get_tuple, thread_id, checkpoint_id
        )

    async def alist(
        self,
        thread_id: str,
        before: Optional[str] = None,
        limit: int = 50,
    ) -> List[Checkpoint]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.list(thread_id, before, limit)
        )


class InMemoryCheckpointer(BaseCheckpointer):
    """内存 checkpointer（开发与测试默认）

    特性：
    - 进程退出即丢
    - 无并发安全保证（单进程同步场景够用）
    - O(1) put / O(1) get_tuple latest / O(n) list
    """

    def __init__(self) -> None:
        # thread_id -> [checkpoints in append order]
        self._threads: Dict[str, List[Checkpoint]] = {}
        # (thread_id, checkpoint_id) -> Checkpoint （快速查找用）
        self._index: Dict[tuple, Checkpoint] = {}

    def put(self, checkpoint: Checkpoint) -> None:
        if checkpoint.thread_id not in self._threads:
            self._threads[checkpoint.thread_id] = []
        self._threads[checkpoint.thread_id].append(checkpoint)
        self._index[(checkpoint.thread_id, checkpoint.id)] = checkpoint

    def get_tuple(
        self, thread_id: str, checkpoint_id: Optional[str] = None
    ) -> Optional[Checkpoint]:
        if checkpoint_id is None:
            ckpts = self._threads.get(thread_id, [])
            return ckpts[-1] if ckpts else None
        return self._index.get((thread_id, checkpoint_id))

    def list(
        self,
        thread_id: str,
        before: Optional[str] = None,
        limit: int = 50,
    ) -> List[Checkpoint]:
        ckpts = list(self._threads.get(thread_id, []))
        # 倒序（最新在前）
        ckpts.reverse()
        if before is not None:
            # 取严格在 before 之前的
            new_list: List[Checkpoint] = []
            seen_before = False
            for c in ckpts:
                if seen_before:
                    new_list.append(c)
                if c.id == before:
                    seen_before = True
            ckpts = new_list
        return ckpts[:limit]

    def clear(self, thread_id: Optional[str] = None) -> None:
        """清空指定 thread（thread_id=None 清空全部）

        测试与开发用，非协议方法。
        """
        if thread_id is None:
            self._threads.clear()
            self._index.clear()
        else:
            for c in self._threads.pop(thread_id, []):
                self._index.pop((thread_id, c.id), None)
