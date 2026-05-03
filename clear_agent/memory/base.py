"""记忆系统基础类与配置

Ported from AntonAgents (CC-BY-NC-SA-4.0)
Original: anton_agents/memory/base.py

提供：
- ``MemoryItem``：记忆项数据结构（Pydantic）
- ``MemoryConfig``：记忆系统配置（容量 / TTL / 衰减系数 / 多模态等）
- ``BaseMemory``：所有记忆类型的抽象基类（``add / retrieve / update / remove /
  has_memory / clear / get_stats``）

子类（``WorkingMemory`` / ``SemanticMemory``）须实现这 7 个抽象方法。
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class MemoryItem(BaseModel):
    """单条记忆项

    Attributes:
        id: 唯一 ID
        content: 记忆内容
        memory_type: 记忆类型（``working`` / ``semantic`` / ``episodic`` / ``perceptual`` 等）
        user_id: 所属用户 ID
        timestamp: 创建时间（可参与时间衰减计算）
        importance: 重要性分数 [0.0, 1.0]
        metadata: 任意元数据（``forgotten / source / tags`` 等）
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    metadata: Dict[str, Any] = {}


class MemoryConfig(BaseModel):
    """记忆系统配置

    所有字段都有默认值，可按需覆盖。
    """

    # 存储路径（用于持久化记忆，例如 SemanticMemory 的 SQLite 文件）
    storage_path: str = "memory/memory_data"

    # 通用容量与重要性
    max_capacity: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95

    # WorkingMemory 专属
    working_memory_capacity: int = 10
    working_memory_tokens: int = 2000
    working_memory_ttl_minutes: int = 120

    # PerceptualMemory 专属（后续版本会用到，留字段不破坏）
    perceptual_memory_modalities: List[str] = ["text", "image", "audio", "video"]


class BaseMemory(ABC):
    """记忆基类 —— 定义所有记忆类型的通用接口"""

    def __init__(
        self, config: MemoryConfig, storage_backend: Optional[Any] = None
    ):
        self.config = config
        self.storage = storage_backend
        # ``WorkingMemory`` → ``"working"``、``SemanticMemory`` → ``"semantic"``
        self.memory_type = self.__class__.__name__.lower().replace("memory", "")

    # -------- 7 个抽象接口 --------

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """添加记忆项；返回记忆 ID"""

    @abstractmethod
    def retrieve(
        self, query: str, limit: int = 5, **kwargs: Any
    ) -> List[MemoryItem]:
        """根据 ``query`` 检索相关记忆"""

    @abstractmethod
    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """部分字段更新；返回是否成功"""

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """删除记忆；返回是否成功"""

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        """记忆是否存在"""

    @abstractmethod
    def clear(self) -> None:
        """清空所有记忆"""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息（``count / current_tokens / avg_importance`` 等）"""

    # -------- 通用工具 --------

    def _generate_id(self) -> str:
        """生成 UUID4 记忆 ID"""
        return str(uuid.uuid4())

    def _calculate_importance(
        self, content: str, base_importance: float = 0.5
    ) -> float:
        """基于内容启发式估算重要性

        - 长度 > 100 字符 → +0.1
        - 含关键词（重要 / 关键 / 必须 / 注意 / 警告 / 错误）→ +0.2
        - 结果 clip 到 [0.0, 1.0]
        """
        importance = base_importance
        if len(content) > 100:
            importance += 0.1
        important_keywords = ["重要", "关键", "必须", "注意", "警告", "错误"]
        if any(kw in content for kw in important_keywords):
            importance += 0.2
        return max(0.0, min(1.0, importance))

    def __str__(self) -> str:
        try:
            stats = self.get_stats()
        except Exception:
            stats = {}
        return f"{self.__class__.__name__}(count={stats.get('count', 0)})"

    def __repr__(self) -> str:
        return self.__str__()


__all__ = ["MemoryItem", "MemoryConfig", "BaseMemory"]
