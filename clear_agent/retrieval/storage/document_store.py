"""SQLite 文档/记忆存储实现

Ported from AntonAgents (CC-BY-NC-SA-4.0)
Original: anton_agents/memory/storage/document_store.py

提供两层抽象：
- ``DocumentStore``：基类，定义记忆 CRUD + 文档 add/get 接口
- ``SQLiteDocumentStore``：基于 sqlite3 的轻量实现（同路径单例 + 线程本地连接）

设计要点：
- **同路径单例**：同一个 ``db_path`` 多次构造返回同一实例，避免锁冲突
- **线程本地连接**：每个线程独立的 sqlite3 connection，避免跨线程访问异常
- **行工厂**：``row_factory = sqlite3.Row``，结果可按列名访问
- 表结构包含 ``users / memories / concepts / memory_concepts / concept_relationships``
  及索引（``user_id / memory_type / timestamp / importance``）
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, cast


class DocumentStore(ABC):
    """文档/记忆存储基类"""

    @abstractmethod
    def add_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        timestamp: int,
        importance: float,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """添加记忆"""

    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """获取单个记忆"""

    @abstractmethod
    def search_memories(
        self,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        importance_threshold: Optional[float] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """搜索记忆（多条件 + 重要性/时间排序）"""

    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """更新记忆字段（部分更新）"""

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""

    @abstractmethod
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计"""

    @abstractmethod
    def add_document(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """添加文档（语义化包装 add_memory）"""

    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """获取文档"""


class SQLiteDocumentStore(DocumentStore):
    """SQLite 文档存储实现（同路径单例）

    Args:
        db_path: 数据库文件路径；默认 ``./memory.db``。
            ``":memory:"`` 走纯内存模式，进程结束自动销毁，便于测试。

    Note:
        测试时如需重置全局单例，调用 ``SQLiteDocumentStore.reset_instances()``。
    """

    _instances: Dict[str, "SQLiteDocumentStore"] = {}
    _initialized_dbs: set[str] = set()
    _class_lock = threading.RLock()
    _initialized: bool = False

    def __new__(cls, db_path: str = "./memory.db") -> "SQLiteDocumentStore":
        """同路径单例"""
        key = ":memory:" if db_path == ":memory:" else os.path.abspath(db_path)
        with cls._class_lock:
            if key not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[key] = instance
            return cls._instances[key]

    def __init__(self, db_path: str = "./memory.db") -> None:
        # 避免重复初始化
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.db_path = db_path
        self.local = threading.local()

        # 确保目录存在（:memory: 跳过）
        if db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)

        # 初始化数据库（每个 db_path 只一次）
        key = ":memory:" if db_path == ":memory:" else os.path.abspath(db_path)
        if key not in self._initialized_dbs:
            self._init_database()
            self._initialized_dbs.add(key)
            print(f"[OK] SQLite 文档存储初始化完成: {db_path}")

        self._initialized = True

    @classmethod
    def reset_instances(cls) -> None:
        """测试钩子：清空所有单例与 init 标记"""
        with cls._class_lock:
            for inst in cls._instances.values():
                try:
                    inst.close()
                except Exception:
                    pass
            cls._instances.clear()
            cls._initialized_dbs.clear()

    def _get_connection(self) -> sqlite3.Connection:
        """获取线程本地连接"""
        if not hasattr(self.local, "connection"):
            self.local.connection = sqlite3.connect(self.db_path)
            self.local.connection.row_factory = sqlite3.Row
        return cast(sqlite3.Connection, self.local.connection)

    def _init_database(self) -> None:
        """初始化所有表与索引"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                importance REAL NOT NULL,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_concepts (
                memory_id TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                relevance_score REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (memory_id, concept_id),
                FOREIGN KEY (memory_id) REFERENCES memories (id) ON DELETE CASCADE,
                FOREIGN KEY (concept_id) REFERENCES concepts (id) ON DELETE CASCADE
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS concept_relationships (
                from_concept_id TEXT NOT NULL,
                to_concept_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                properties TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (from_concept_id, to_concept_id, relationship_type),
                FOREIGN KEY (from_concept_id) REFERENCES concepts (id) ON DELETE CASCADE,
                FOREIGN KEY (to_concept_id) REFERENCES concepts (id) ON DELETE CASCADE
            )
            """
        )

        for index_sql in [
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories (user_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories (memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories (importance)",
            "CREATE INDEX IF NOT EXISTS idx_memory_concepts_memory ON memory_concepts (memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_concepts_concept ON memory_concepts (concept_id)",
        ]:
            cursor.execute(index_sql)

        conn.commit()

    # -------- 公共 API --------

    def add_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        memory_type: str,
        timestamp: int,
        importance: float,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)",
            (user_id, user_id),
        )
        cursor.execute(
            """
            INSERT OR REPLACE INTO memories
            (id, user_id, content, memory_type, timestamp, importance, properties, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (
                memory_id,
                user_id,
                content,
                memory_type,
                timestamp,
                importance,
                json.dumps(properties) if properties else None,
            ),
        )
        conn.commit()
        return memory_id

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, user_id, content, memory_type, timestamp, importance, properties, created_at
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return _row_to_memory_dict(row)

    def search_memories(
        self,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        importance_threshold: Optional[float] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()

        where_conditions: List[str] = []
        params: List[Any] = []

        if user_id:
            where_conditions.append("user_id = ?")
            params.append(user_id)
        if memory_type:
            where_conditions.append("memory_type = ?")
            params.append(memory_type)
        if start_time:
            where_conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time:
            where_conditions.append("timestamp <= ?")
            params.append(end_time)
        if importance_threshold is not None:
            where_conditions.append("importance >= ?")
            params.append(importance_threshold)

        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)

        cursor.execute(
            f"""
            SELECT id, user_id, content, memory_type, timestamp, importance, properties, created_at
            FROM memories
            {where_clause}
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
            """,
            params + [limit],
        )
        return [_row_to_memory_dict(row) for row in cursor.fetchall()]

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()

        update_fields: List[str] = []
        params: List[Any] = []

        if content is not None:
            update_fields.append("content = ?")
            params.append(content)
        if importance is not None:
            update_fields.append("importance = ?")
            params.append(importance)
        if properties is not None:
            update_fields.append("properties = ?")
            params.append(json.dumps(properties))

        if not update_fields:
            return False

        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        params.append(memory_id)

        cursor.execute(
            f"""
            UPDATE memories
            SET {", ".join(update_fields)}
            WHERE id = ?
            """,
            params,
        )
        conn.commit()
        return cursor.rowcount > 0

    def delete_memory(self, memory_id: str) -> bool:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        deleted_count = cursor.rowcount
        conn.commit()
        return deleted_count > 0

    def get_database_stats(self) -> Dict[str, Any]:
        conn = self._get_connection()
        cursor = conn.cursor()

        stats: Dict[str, Any] = {}

        for table in [
            "users",
            "memories",
            "concepts",
            "memory_concepts",
            "concept_relationships",
        ]:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            stats[f"{table}_count"] = cursor.fetchone()["count"]

        cursor.execute(
            """
            SELECT memory_type, COUNT(*) as count
            FROM memories
            GROUP BY memory_type
            """
        )
        memory_types: Dict[str, int] = {}
        for row in cursor.fetchall():
            memory_types[row["memory_type"]] = row["count"]
        stats["memory_types"] = memory_types

        cursor.execute(
            """
            SELECT user_id, COUNT(*) as count
            FROM memories
            GROUP BY user_id
            ORDER BY count DESC
            LIMIT 10
            """
        )
        top_users: Dict[str, int] = {}
        for row in cursor.fetchall():
            top_users[row["user_id"]] = row["count"]
        stats["top_users"] = top_users

        stats["store_type"] = "sqlite"
        stats["db_path"] = self.db_path
        return stats

    def add_document(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """添加文档（包装 add_memory）"""
        doc_id = str(uuid.uuid4())
        user_id = (metadata or {}).get("user_id", "system")
        return self.add_memory(
            memory_id=doc_id,
            user_id=user_id,
            content=content,
            memory_type="document",
            timestamp=int(time.time()),
            importance=0.5,
            properties=metadata or {},
        )

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        return self.get_memory(document_id)

    def close(self) -> None:
        """关闭当前线程的数据库连接"""
        if hasattr(self.local, "connection"):
            self.local.connection.close()
            delattr(self.local, "connection")


# -------- 内部辅助 --------


def _row_to_memory_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "memory_id": row["id"],
        "user_id": row["user_id"],
        "content": row["content"],
        "memory_type": row["memory_type"],
        "timestamp": row["timestamp"],
        "importance": row["importance"],
        "properties": json.loads(row["properties"]) if row["properties"] else {},
        "created_at": row["created_at"],
    }


__all__ = [
    "DocumentStore",
    "SQLiteDocumentStore",
]
