"""Checkpointer 协议与实现

为 ClearAgent StateGraph 提供 per-node 状态快照与恢复能力。

设计要点：
- BaseCheckpointer 抽象，支持同步/异步两套接口
- InMemoryCheckpointer：开发与单元测试默认
- JsonFileCheckpointer / SqliteCheckpointer 在 W2 实现
- thread_id 隔离不同会话；checkpoint_id 单调递增（uuid7 时间排序友好）


"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterator, List, Optional


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


# ==================== JsonFileCheckpointer ====================


class JsonFileCheckpointer(BaseCheckpointer):
    """文件后端 checkpointer

    目录布局:
        <base_dir>/<thread_id>/<checkpoint_id>.json   # 单个 ckpt 数据
        <base_dir>/<thread_id>/_index.jsonl           # 倒序追加，加速 list()

    特性:
    - 原子写入（tmp + os.replace），与现有 SessionStore.save 一致
    - 兼容 1.x SessionStore：能读取 memory/sessions/session-*.json 转为单 ckpt thread
    - 进程退出后状态完整保留
    - 单进程下并发安全（用 _lock 保护索引写入）

    Args:
        base_dir: checkpoint 根目录（默认 memory/checkpoints）
        legacy_session_dir: 旧 SessionStore 目录（用于读取兼容；写入仍用 base_dir）
    """

    def __init__(
        self,
        base_dir: str = "memory/checkpoints",
        legacy_session_dir: Optional[str] = "memory/sessions",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_session_dir = (
            Path(legacy_session_dir) if legacy_session_dir else None
        )
        self._lock = Lock()

    def _thread_dir(self, thread_id: str) -> Path:
        d = self.base_dir / thread_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def put(self, ckpt: Checkpoint) -> None:
        td = self._thread_dir(ckpt.thread_id)
        filepath = td / f"{ckpt.id}.json"
        tmp_path = filepath.with_suffix(".json.tmp")

        # 原子写入正文
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(ckpt.to_dict(), f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp_path, filepath)

        # 追加索引（受锁保护）
        idx_path = td / "_index.jsonl"
        with self._lock:
            with idx_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "id": ckpt.id,
                            "created_at": ckpt.created_at.isoformat(),
                            "parent_id": ckpt.parent_id,
                        }
                    )
                    + "\n"
                )

    def get_tuple(
        self, thread_id: str, checkpoint_id: Optional[str] = None
    ) -> Optional[Checkpoint]:
        td = self.base_dir / thread_id
        if not td.exists():
            # 尝试从 legacy session 目录读取
            return self._load_legacy(thread_id, checkpoint_id)

        if checkpoint_id is not None:
            filepath = td / f"{checkpoint_id}.json"
            if not filepath.exists():
                return None
            return self._read_ckpt(filepath)

        # 取最新：用索引
        latest_id = self._latest_id_from_index(td)
        if latest_id is None:
            return None
        filepath = td / f"{latest_id}.json"
        return self._read_ckpt(filepath) if filepath.exists() else None

    def list(
        self,
        thread_id: str,
        before: Optional[str] = None,
        limit: int = 50,
    ) -> List[Checkpoint]:
        td = self.base_dir / thread_id
        if not td.exists():
            return []

        idx_path = td / "_index.jsonl"
        if not idx_path.exists():
            return []

        # 读全部索引项（小数据量）
        entries: List[Dict[str, Any]] = []
        with idx_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # 倒序
        entries.reverse()

        # 过滤 before
        if before is not None:
            new_entries: List[Dict[str, Any]] = []
            seen = False
            for e in entries:
                if seen:
                    new_entries.append(e)
                if e["id"] == before:
                    seen = True
            entries = new_entries

        results: List[Checkpoint] = []
        for e in entries[:limit]:
            filepath = td / f"{e['id']}.json"
            if filepath.exists():
                ckpt = self._read_ckpt(filepath)
                if ckpt:
                    results.append(ckpt)
        return results

    @staticmethod
    def _read_ckpt(filepath: Path) -> Optional[Checkpoint]:
        try:
            with filepath.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return Checkpoint.from_dict(data)
        except (OSError, json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _latest_id_from_index(thread_dir: Path) -> Optional[str]:
        idx_path = thread_dir / "_index.jsonl"
        if not idx_path.exists():
            return None
        last_line = ""
        with idx_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last_line = line
        if not last_line:
            return None
        try:
            return json.loads(last_line)["id"]
        except (json.JSONDecodeError, KeyError):
            return None

    def _load_legacy(
        self, thread_id: str, checkpoint_id: Optional[str]
    ) -> Optional[Checkpoint]:
        """尝试从 1.x SessionStore 文件加载（兼容模式）

        1.x session 文件用 thread_id 作为文件名（不带 .json 后缀也接受），
        转换为单 checkpoint thread 返回。
        """
        if self.legacy_session_dir is None or not self.legacy_session_dir.exists():
            return None

        candidates = [
            self.legacy_session_dir / f"{thread_id}.json",
            self.legacy_session_dir / thread_id,
        ]
        for filepath in candidates:
            if filepath.is_file():
                try:
                    with filepath.open("r", encoding="utf-8") as f:
                        legacy = json.load(f)
                except (OSError, json.JSONDecodeError):
                    continue
                # 转为单 checkpoint
                return Checkpoint(
                    id=legacy.get("session_id") or _uuid7(),
                    thread_id=thread_id,
                    parent_id=None,
                    state={"messages": legacy.get("history", [])},
                    next_nodes=[],
                    created_at=datetime.now(),
                    metadata={
                        "source": "legacy_session",
                        "legacy_path": str(filepath),
                        **(legacy.get("metadata") or {}),
                    },
                )
        return None


# ==================== SqliteCheckpointer ====================


_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL,
    parent_id TEXT,
    state_json TEXT NOT NULL,
    next_nodes_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_thread_created
    ON checkpoints(thread_id, created_at DESC);
"""


class SqliteCheckpointer(BaseCheckpointer):
    """SQLite 后端（生产推荐）

    单文件 .db；WAL 模式 + synchronous=NORMAL 平衡性能与持久性。
    标准库 sqlite3 实现，零额外依赖。
    """

    def __init__(self, db_path: str = "memory/checkpoints.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SQLITE_DDL)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # check_same_thread=False 支持线程池调用；用 isolation_level=None 自管事务
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,
            timeout=30.0,
        )
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    def put(self, ckpt: Checkpoint) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                    (id, thread_id, parent_id, state_json, next_nodes_json, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ckpt.id,
                    ckpt.thread_id,
                    ckpt.parent_id,
                    json.dumps(ckpt.state, ensure_ascii=False, default=str),
                    json.dumps(ckpt.next_nodes, ensure_ascii=False),
                    ckpt.created_at.isoformat(),
                    json.dumps(ckpt.metadata, ensure_ascii=False, default=str),
                ),
            )

    def get_tuple(
        self, thread_id: str, checkpoint_id: Optional[str] = None
    ) -> Optional[Checkpoint]:
        with self._connect() as conn:
            if checkpoint_id is not None:
                row = conn.execute(
                    "SELECT id, thread_id, parent_id, state_json, next_nodes_json, created_at, metadata_json "
                    "FROM checkpoints WHERE thread_id=? AND id=?",
                    (thread_id, checkpoint_id),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT id, thread_id, parent_id, state_json, next_nodes_json, created_at, metadata_json "
                    "FROM checkpoints WHERE thread_id=? ORDER BY created_at DESC LIMIT 1",
                    (thread_id,),
                ).fetchone()
        return self._row_to_ckpt(row) if row else None

    def list(
        self,
        thread_id: str,
        before: Optional[str] = None,
        limit: int = 50,
    ) -> List[Checkpoint]:
        with self._connect() as conn:
            if before is not None:
                # 找到 before 的 created_at
                row = conn.execute(
                    "SELECT created_at FROM checkpoints WHERE id=? AND thread_id=?",
                    (before, thread_id),
                ).fetchone()
                if row is None:
                    return []
                before_ts = row[0]
                rows = conn.execute(
                    "SELECT id, thread_id, parent_id, state_json, next_nodes_json, created_at, metadata_json "
                    "FROM checkpoints WHERE thread_id=? AND created_at < ? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (thread_id, before_ts, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, thread_id, parent_id, state_json, next_nodes_json, created_at, metadata_json "
                    "FROM checkpoints WHERE thread_id=? "
                    "ORDER BY created_at DESC LIMIT ?",
                    (thread_id, limit),
                ).fetchall()
        return [c for c in (self._row_to_ckpt(r) for r in rows) if c is not None]

    @staticmethod
    def _row_to_ckpt(row: tuple) -> Optional[Checkpoint]:
        if row is None:
            return None
        try:
            return Checkpoint(
                id=row[0],
                thread_id=row[1],
                parent_id=row[2],
                state=json.loads(row[3]),
                next_nodes=json.loads(row[4]),
                created_at=datetime.fromisoformat(row[5]),
                metadata=json.loads(row[6]),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return None


# ==================== 工厂函数 ====================


def make_checkpointer(
    backend: str = "memory",
    base_dir: str = "memory/checkpoints",
    db_path: str = "memory/checkpoints.db",
    legacy_session_dir: Optional[str] = "memory/sessions",
) -> BaseCheckpointer:
    """根据 backend 名创建对应 checkpointer

    Args:
        backend: "memory" | "json" | "sqlite"
    """
    backend = backend.lower()
    if backend == "memory":
        return InMemoryCheckpointer()
    if backend == "json":
        return JsonFileCheckpointer(
            base_dir=base_dir, legacy_session_dir=legacy_session_dir
        )
    if backend == "sqlite":
        return SqliteCheckpointer(db_path=db_path)
    raise ValueError(f"未知 checkpoint backend: {backend}")
