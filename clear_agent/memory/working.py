"""WorkingMemory —— 短期工作记忆

Ported from AntonAgents (CC-BY-NC-SA-4.0)
Original: anton_agents/memory/types/working.py

特点：
- 容量有限（默认 10-20 条）
- TTL 过期（默认 120 分钟）
- token 预算（默认 2000）
- 优先级管理（重要性 × 时间衰减）
- 检索 = TF-IDF 向量（sklearn 可选）+ 关键词匹配 + 时间衰减加权
- 三种 forget 策略：``importance_based / time_based / capacity_based``

不持久化 —— 进程结束即销毁。需要长期记忆请用 ``SemanticMemory`` (2.0-β/W3)。
"""

from __future__ import annotations

import heapq
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .base import BaseMemory, MemoryConfig, MemoryItem


class WorkingMemory(BaseMemory):
    """短期工作记忆"""

    def __init__(
        self, config: MemoryConfig, storage_backend: Optional[Any] = None
    ):
        super().__init__(config, storage_backend)

        # 工作记忆专属配置
        self.max_capacity = self.config.working_memory_capacity
        self.max_tokens = self.config.working_memory_tokens
        self.max_age_minutes = getattr(
            self.config, "working_memory_ttl_minutes", 120
        )
        self.current_tokens = 0
        self.session_start = datetime.now()

        # 内存存储 + 优先级堆 (priority, timestamp, memory_item)
        self.memories: List[MemoryItem] = []
        self.memory_heap: List[Any] = []

    # ==================== 7 个抽象接口实现 ====================

    def add(self, memory_item: MemoryItem) -> str:
        """添加工作记忆项"""
        self._expire_old_memories()
        priority = self._calculate_priority(memory_item)
        heapq.heappush(
            self.memory_heap, (-priority, memory_item.timestamp, memory_item)
        )
        self.memories.append(memory_item)
        self.current_tokens += len(memory_item.content.split())
        self._enforce_capacity_limits()
        return memory_item.id

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[MemoryItem]:
        """混合检索：TF-IDF 语义 + 关键词匹配 + 时间衰减 + 重要性"""
        self._expire_old_memories()
        if not self.memories:
            return []

        # 过滤已遗忘 + 按 user_id
        active = [m for m in self.memories if not m.metadata.get("forgotten", False)]
        filtered = [m for m in active if user_id is None or m.user_id == user_id]
        if not filtered:
            return []

        # 尝试 TF-IDF 向量检索（sklearn 可选）
        vector_scores: Dict[str, float] = {}
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
            from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

            documents = [query] + [m.content for m in filtered]
            vectorizer = TfidfVectorizer(stop_words=None, lowercase=True)
            tfidf_matrix = vectorizer.fit_transform(documents)
            query_vec = tfidf_matrix[0:1]
            doc_vecs = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vec, doc_vecs).flatten()
            for i, m in enumerate(filtered):
                vector_scores[m.id] = float(similarities[i])
        except Exception:
            vector_scores = {}

        # 计算最终分数
        query_lower = query.lower()
        scored: List[tuple] = []
        for m in filtered:
            content_lower = m.content.lower()
            vec_score = vector_scores.get(m.id, 0.0)

            # 关键词分数
            kw_score = 0.0
            if query_lower in content_lower:
                kw_score = len(query_lower) / max(1, len(content_lower))
            else:
                q_words = set(query_lower.split())
                c_words = set(content_lower.split())
                inter = q_words & c_words
                if inter:
                    kw_score = len(inter) / len(q_words | c_words) * 0.8

            base = (
                vec_score * 0.7 + kw_score * 0.3 if vec_score > 0 else kw_score
            )

            # 时间衰减 + 重要性加权
            time_decay = self._calculate_time_decay(m.timestamp)
            base *= time_decay
            importance_weight = 0.8 + (m.importance * 0.4)
            final = base * importance_weight

            if final > 0:
                scored.append((final, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        for m in self.memories:
            if m.id == memory_id:
                old_tokens = len(m.content.split())
                if content is not None:
                    m.content = content
                    new_tokens = len(content.split())
                    self.current_tokens = self.current_tokens - old_tokens + new_tokens
                if importance is not None:
                    m.importance = importance
                if metadata is not None:
                    m.metadata.update(metadata)
                self._update_heap_priority(m)
                return True
        return False

    def remove(self, memory_id: str) -> bool:
        for i, m in enumerate(self.memories):
            if m.id == memory_id:
                removed = self.memories.pop(i)
                self._mark_deleted_in_heap(memory_id)
                self.current_tokens = max(
                    0, self.current_tokens - len(removed.content.split())
                )
                return True
        return False

    def has_memory(self, memory_id: str) -> bool:
        return any(m.id == memory_id for m in self.memories)

    def clear(self) -> None:
        self.memories.clear()
        self.memory_heap.clear()
        self.current_tokens = 0

    def get_stats(self) -> Dict[str, Any]:
        self._expire_old_memories()
        active = self.memories
        avg_importance = (
            sum(m.importance for m in active) / len(active) if active else 0.0
        )
        return {
            "count": len(active),
            "forgotten_count": 0,  # 工作记忆中已遗忘的会被直接删除
            "total_count": len(self.memories),
            "current_tokens": self.current_tokens,
            "max_capacity": self.max_capacity,
            "max_tokens": self.max_tokens,
            "max_age_minutes": self.max_age_minutes,
            "session_duration_minutes": (
                datetime.now() - self.session_start
            ).total_seconds()
            / 60,
            "avg_importance": avg_importance,
            "capacity_usage": (
                len(active) / self.max_capacity if self.max_capacity > 0 else 0.0
            ),
            "token_usage": (
                self.current_tokens / self.max_tokens if self.max_tokens > 0 else 0.0
            ),
            "memory_type": "working",
        }

    # ==================== 额外便捷接口 ====================

    def get_recent(self, limit: int = 10) -> List[MemoryItem]:
        """按 timestamp 倒序取最近 N 条"""
        return sorted(self.memories, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_important(self, limit: int = 10) -> List[MemoryItem]:
        """按重要性倒序取 top N"""
        return sorted(self.memories, key=lambda x: x.importance, reverse=True)[:limit]

    def get_all(self) -> List[MemoryItem]:
        """全部记忆的浅拷贝"""
        return self.memories.copy()

    def get_context_summary(self, max_length: int = 500) -> str:
        """生成上下文摘要文本（按 (importance, timestamp) 倒序拼接）"""
        if not self.memories:
            return "No working memories available."
        sorted_mems = sorted(
            self.memories,
            key=lambda m: (m.importance, m.timestamp),
            reverse=True,
        )
        parts: List[str] = []
        cur_len = 0
        for m in sorted_mems:
            content = m.content
            if cur_len + len(content) <= max_length:
                parts.append(content)
                cur_len += len(content)
            else:
                remain = max_length - cur_len
                if remain > 50:
                    parts.append(content[:remain] + "...")
                break
        return "Working Memory Context:\n" + "\n".join(parts)

    def forget(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 1,
    ) -> int:
        """触发遗忘策略

        Args:
            strategy: ``"importance_based"`` / ``"time_based"`` / ``"capacity_based"``
            threshold: importance 阈值（仅 importance_based 使用）
            max_age_days: 时间阈值（仅 time_based 使用）

        Returns:
            实际删除的记忆数
        """
        forgotten = 0
        now = datetime.now()
        to_remove: List[str] = []

        # 始终先清理 TTL（分钟级）
        cutoff_ttl = now - timedelta(minutes=self.max_age_minutes)
        for m in self.memories:
            if m.timestamp < cutoff_ttl:
                to_remove.append(m.id)

        if strategy == "importance_based":
            for m in self.memories:
                if m.importance < threshold:
                    to_remove.append(m.id)
        elif strategy == "time_based":
            cutoff = now - timedelta(hours=max_age_days * 24)
            for m in self.memories:
                if m.timestamp < cutoff:
                    to_remove.append(m.id)
        elif strategy == "capacity_based":
            if len(self.memories) > self.max_capacity:
                sorted_mems = sorted(
                    self.memories, key=lambda m: self._calculate_priority(m)
                )
                excess = len(self.memories) - self.max_capacity
                for m in sorted_mems[:excess]:
                    to_remove.append(m.id)

        # 去重 + 执行
        for mid in set(to_remove):
            if self.remove(mid):
                forgotten += 1
        return forgotten

    # ==================== 内部工具 ====================

    def _calculate_priority(self, memory: MemoryItem) -> float:
        """优先级 = 重要性 × 时间衰减"""
        priority = memory.importance
        priority *= self._calculate_time_decay(memory.timestamp)
        return priority

    def _calculate_time_decay(self, timestamp: datetime) -> float:
        """指数衰减：每 6 小时一档；最低保持 10%"""
        time_diff = datetime.now() - timestamp
        hours_passed = time_diff.total_seconds() / 3600
        decay = self.config.decay_factor ** (hours_passed / 6)
        return max(0.1, decay)

    def _enforce_capacity_limits(self) -> None:
        """容量 / token 超限时删除最低优先级"""
        while len(self.memories) > self.max_capacity:
            self._remove_lowest_priority_memory()
        while self.current_tokens > self.max_tokens:
            self._remove_lowest_priority_memory()

    def _expire_old_memories(self) -> None:
        """按 TTL 过滤过期记忆，同步重建堆与 token 计数"""
        if not self.memories:
            return
        cutoff = datetime.now() - timedelta(minutes=self.max_age_minutes)
        kept: List[MemoryItem] = []
        removed_tokens = 0
        for m in self.memories:
            if m.timestamp >= cutoff:
                kept.append(m)
            else:
                removed_tokens += len(m.content.split())
        if len(kept) == len(self.memories):
            return
        self.memories = kept
        self.current_tokens = max(0, self.current_tokens - removed_tokens)
        self.memory_heap = []
        for m in self.memories:
            heapq.heappush(
                self.memory_heap,
                (-self._calculate_priority(m), m.timestamp, m),
            )

    def _remove_lowest_priority_memory(self) -> None:
        """删除优先级最低的一条"""
        if not self.memories:
            return
        lowest_priority = float("inf")
        lowest_memory: Optional[MemoryItem] = None
        for m in self.memories:
            p = self._calculate_priority(m)
            if p < lowest_priority:
                lowest_priority = p
                lowest_memory = m
        if lowest_memory is not None:
            self.remove(lowest_memory.id)

    def _update_heap_priority(self, _memory: MemoryItem) -> None:
        """简单实现：重建堆"""
        self.memory_heap = []
        for m in self.memories:
            heapq.heappush(
                self.memory_heap,
                (-self._calculate_priority(m), m.timestamp, m),
            )

    def _mark_deleted_in_heap(self, _memory_id: str) -> None:
        """heapq 不支持随机删除，靠后续重建清理"""
        pass


__all__ = ["WorkingMemory"]
