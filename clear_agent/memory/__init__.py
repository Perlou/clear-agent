"""ClearAgent 记忆系统

完整模块：
- ``MemoryItem`` / ``MemoryConfig`` / ``BaseMemory`` 抽象（base.py）
- ``WorkingMemory`` 短期记忆（working.py）
- ``SemanticMemory`` + ``Entity`` / ``Relation`` 长期语义记忆（semantic.py）
- ``MemoryManager`` 多子系统协调（manager.py，重写）
"""

from .base import BaseMemory, MemoryConfig, MemoryItem
from .manager import MemoryManager
from .semantic import Entity, Relation, SemanticMemory
from .working import WorkingMemory

__all__ = [
    "MemoryItem",
    "MemoryConfig",
    "BaseMemory",
    "WorkingMemory",
    "Entity",
    "Relation",
    "SemanticMemory",
    "MemoryManager",
]
