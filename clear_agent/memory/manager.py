"""MemoryManager —— 协调多种记忆子系统的统一入口

**重写说明**（详见 project_docs/07-anton-agents-port.md §2.2）：
AntonAgents 的 ``manager.py`` 源文件 0 字节但被 ``__init__.py`` import，
**说明从未跑通**。本模块是 ClearAgent 自研重写，提供：

- 按 ``memory_type`` 路由到对应子系统（``WorkingMemory`` / ``SemanticMemory`` / ...）
- 跨子系统的统一 ``add / retrieve / update / remove / has_memory / clear``
- 跨子系统的聚合 ``get_stats``

设计原则：
- 不假设具体子系统的实现细节，只用 ``BaseMemory`` 抽象接口
- 路由失败优雅降级（找不到 type → ValueError；空查询 → 返回 []）
- 检索结果按 ``importance`` 与子系统打分合并

典型用法::

    mgr = MemoryManager()
    mgr.register("working", WorkingMemory(MemoryConfig()))
    mgr.register("semantic", SemanticMemory(MemoryConfig()))

    mgr.add(MemoryItem(id="m1", memory_type="working", ...))   # 自动路由到 working
    hits = mgr.retrieve("query", limit=10)                      # 跨所有子系统
    hits_only = mgr.retrieve("query", memory_types=["semantic"])
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .base import BaseMemory, MemoryItem


logger = logging.getLogger(__name__)


class MemoryManager:
    """多记忆子系统协调器

    Attributes:
        memories: ``memory_type -> BaseMemory`` 的注册表
    """

    def __init__(self):
        self.memories: Dict[str, BaseMemory] = {}

    # ==================== 注册 / 路由 ====================

    def register(self, memory_type: str, memory: BaseMemory) -> None:
        """注册一个记忆子系统

        Args:
            memory_type: ``"working"`` / ``"semantic"`` / 自定义
            memory: ``BaseMemory`` 子类实例

        Raises:
            ValueError: ``memory_type`` 已被注册
        """
        if memory_type in self.memories:
            raise ValueError(f"memory_type 已注册: {memory_type}")
        self.memories[memory_type] = memory
        logger.info(f"✅ 注册记忆子系统: {memory_type} ({type(memory).__name__})")

    def unregister(self, memory_type: str) -> bool:
        """解注册；返回是否成功"""
        if memory_type in self.memories:
            del self.memories[memory_type]
            return True
        return False

    def get(self, memory_type: str) -> Optional[BaseMemory]:
        """按 type 查找子系统；不存在返回 None"""
        return self.memories.get(memory_type)

    def types(self) -> List[str]:
        """返回所有注册的 ``memory_type`` 列表"""
        return list(self.memories.keys())

    def __contains__(self, memory_type: str) -> bool:
        return memory_type in self.memories

    def __len__(self) -> int:
        return len(self.memories)

    # ==================== 统一接口 ====================

    def add(self, memory_item: MemoryItem) -> str:
        """按 ``memory_item.memory_type`` 自动路由到对应子系统

        Raises:
            ValueError: 没有注册对应 ``memory_type``
        """
        target = self.memories.get(memory_item.memory_type)
        if target is None:
            raise ValueError(
                f"未注册 memory_type='{memory_item.memory_type}'；"
                f"已注册: {self.types()}"
            )
        return target.add(memory_item)

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        memory_types: Optional[List[str]] = None,
        per_type_limit: Optional[int] = None,
        **kwargs: Any,
    ) -> List[MemoryItem]:
        """跨多个子系统检索 + 合并

        Args:
            query: 查询文本
            limit: 总返回数上限
            memory_types: 限定查询哪些子系统；缺省为全部
            per_type_limit: 每个子系统单独的上限；缺省 ``limit``
            **kwargs: 透传给子系统的 ``retrieve``（如 ``user_id``）

        Returns:
            按 ``importance`` 倒序合并；同 id 去重（保留 importance 高的）
        """
        if not query:
            return []
        targets = memory_types or self.types()
        per = per_type_limit or limit

        merged: Dict[str, MemoryItem] = {}
        for mt in targets:
            mem = self.memories.get(mt)
            if mem is None:
                logger.debug(f"跳过未注册类型: {mt}")
                continue
            try:
                hits = mem.retrieve(query, limit=per, **kwargs)
            except Exception as e:
                logger.warning(f"⚠️ {mt} retrieve 失败: {e}")
                continue
            for h in hits:
                existing = merged.get(h.id)
                if existing is None or h.importance > existing.importance:
                    merged[h.id] = h

        sorted_items = sorted(
            merged.values(), key=lambda x: x.importance, reverse=True
        )
        return sorted_items[:limit]

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        memory_type: Optional[str] = None,
    ) -> bool:
        """更新记忆；指定 ``memory_type`` 时只在该子系统查找，否则遍历

        Returns:
            任一子系统更新成功即 True
        """
        if memory_type is not None:
            mem = self.memories.get(memory_type)
            if mem is None:
                return False
            return mem.update(memory_id, content, importance, metadata)

        for mem in self.memories.values():
            if mem.has_memory(memory_id):
                return mem.update(memory_id, content, importance, metadata)
        return False

    def remove(
        self, memory_id: str, memory_type: Optional[str] = None
    ) -> bool:
        """删除记忆（同 update 的路由策略）"""
        if memory_type is not None:
            mem = self.memories.get(memory_type)
            if mem is None:
                return False
            return mem.remove(memory_id)

        for mem in self.memories.values():
            if mem.has_memory(memory_id):
                return mem.remove(memory_id)
        return False

    def has_memory(self, memory_id: str) -> bool:
        """任一子系统含该 id 即返回 True"""
        return any(mem.has_memory(memory_id) for mem in self.memories.values())

    def clear(self, memory_types: Optional[List[str]] = None) -> None:
        """清空指定子系统；缺省清空全部"""
        targets = memory_types or self.types()
        for mt in targets:
            mem = self.memories.get(mt)
            if mem is None:
                continue
            try:
                mem.clear()
            except Exception as e:
                logger.warning(f"⚠️ {mt}.clear 失败: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """聚合所有子系统的统计

        Returns:
            形如::

                {
                    "total_count": 42,
                    "by_type": {
                        "working": {"count": 10, ...},
                        "semantic": {"count": 32, ...},
                    },
                    "registered_types": ["working", "semantic"],
                }
        """
        by_type: Dict[str, Dict[str, Any]] = {}
        total = 0
        for mt, mem in self.memories.items():
            try:
                stats = mem.get_stats()
                by_type[mt] = stats
                total += int(stats.get("count", 0) or 0)
            except Exception as e:
                logger.warning(f"⚠️ {mt}.get_stats 失败: {e}")
                by_type[mt] = {"error": str(e)}

        return {
            "total_count": total,
            "by_type": by_type,
            "registered_types": list(self.memories.keys()),
        }


__all__ = ["MemoryManager"]
