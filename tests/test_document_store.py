"""SQLiteDocumentStore 测试

纯 Python（仅依赖 sqlite3），无外部 ML 依赖。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from clear_agent.retrieval import DocumentStore, SQLiteDocumentStore


@pytest.fixture(autouse=True)
def _reset_singletons():
    """每个测试前清空单例，避免不同 db_path 之间互相污染"""
    SQLiteDocumentStore.reset_instances()
    yield
    SQLiteDocumentStore.reset_instances()


# ==================== 构造与单例 ====================


def test_constructor_in_memory():
    ds = SQLiteDocumentStore(":memory:")
    assert isinstance(ds, DocumentStore)


def test_same_path_returns_same_instance(tmp_path: Path):
    p = tmp_path / "a.db"
    a = SQLiteDocumentStore(str(p))
    b = SQLiteDocumentStore(str(p))
    assert a is b


def test_different_paths_return_different_instances(tmp_path: Path):
    a = SQLiteDocumentStore(str(tmp_path / "a.db"))
    b = SQLiteDocumentStore(str(tmp_path / "b.db"))
    assert a is not b


def test_reset_instances_creates_fresh_object(tmp_path: Path):
    p = str(tmp_path / "x.db")
    a = SQLiteDocumentStore(p)
    SQLiteDocumentStore.reset_instances()
    b = SQLiteDocumentStore(p)
    assert a is not b


def test_creates_parent_dir(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "c.db"
    SQLiteDocumentStore(str(nested))
    assert nested.parent.is_dir()


# ==================== add_memory / get_memory ====================


def test_add_and_get_memory_roundtrip():
    ds = SQLiteDocumentStore(":memory:")
    mid = ds.add_memory(
        memory_id="m1",
        user_id="u1",
        content="hello",
        memory_type="fact",
        timestamp=1000,
        importance=0.7,
        properties={"source": "test"},
    )
    assert mid == "m1"

    got = ds.get_memory("m1")
    assert got is not None
    assert got["memory_id"] == "m1"
    assert got["user_id"] == "u1"
    assert got["content"] == "hello"
    assert got["memory_type"] == "fact"
    assert got["timestamp"] == 1000
    assert got["importance"] == 0.7
    assert got["properties"] == {"source": "test"}


def test_get_nonexistent_memory_returns_none():
    ds = SQLiteDocumentStore(":memory:")
    assert ds.get_memory("nope") is None


def test_add_memory_replaces_on_same_id():
    """INSERT OR REPLACE 行为：同 ID 二次写入覆盖原值"""
    ds = SQLiteDocumentStore(":memory:")
    ds.add_memory("m1", "u1", "v1", "fact", 1000, 0.5)
    ds.add_memory("m1", "u1", "v2", "fact", 2000, 0.9)

    got = ds.get_memory("m1")
    assert got["content"] == "v2"
    assert got["importance"] == 0.9
    assert got["timestamp"] == 2000


def test_add_memory_without_properties():
    ds = SQLiteDocumentStore(":memory:")
    ds.add_memory("m1", "u1", "c", "fact", 1, 0.5)
    got = ds.get_memory("m1")
    assert got["properties"] == {}


def test_add_memory_creates_user_row():
    """add_memory 时若 user 不存在 → INSERT OR IGNORE 自动建用户"""
    ds = SQLiteDocumentStore(":memory:")
    ds.add_memory("m1", "user-A", "c", "fact", 1, 0.5)
    stats = ds.get_database_stats()
    assert stats["users_count"] == 1


# ==================== search_memories ====================


def _seed_searchable(ds: SQLiteDocumentStore) -> None:
    ds.add_memory("m1", "u1", "c1", "fact", 1000, 0.9)
    ds.add_memory("m2", "u1", "c2", "event", 2000, 0.5)
    ds.add_memory("m3", "u2", "c3", "fact", 3000, 0.7)
    ds.add_memory("m4", "u1", "c4", "fact", 4000, 0.3)


def test_search_no_filter_returns_all_ordered_by_importance_desc():
    ds = SQLiteDocumentStore(":memory:")
    _seed_searchable(ds)
    rows = ds.search_memories(limit=10)
    assert len(rows) == 4
    # importance: 0.9, 0.7, 0.5, 0.3
    assert [r["importance"] for r in rows] == [0.9, 0.7, 0.5, 0.3]


def test_search_filter_by_user():
    ds = SQLiteDocumentStore(":memory:")
    _seed_searchable(ds)
    rows = ds.search_memories(user_id="u1", limit=10)
    assert len(rows) == 3
    assert all(r["user_id"] == "u1" for r in rows)


def test_search_filter_by_memory_type():
    ds = SQLiteDocumentStore(":memory:")
    _seed_searchable(ds)
    rows = ds.search_memories(memory_type="fact", limit=10)
    assert len(rows) == 3
    assert all(r["memory_type"] == "fact" for r in rows)


def test_search_filter_by_time_range():
    ds = SQLiteDocumentStore(":memory:")
    _seed_searchable(ds)
    rows = ds.search_memories(start_time=2000, end_time=3000, limit=10)
    assert {r["memory_id"] for r in rows} == {"m2", "m3"}


def test_search_filter_by_importance_threshold():
    ds = SQLiteDocumentStore(":memory:")
    _seed_searchable(ds)
    rows = ds.search_memories(importance_threshold=0.7, limit=10)
    assert {r["memory_id"] for r in rows} == {"m1", "m3"}


def test_search_limit_truncates():
    ds = SQLiteDocumentStore(":memory:")
    _seed_searchable(ds)
    rows = ds.search_memories(limit=2)
    assert len(rows) == 2


def test_search_combined_filters():
    ds = SQLiteDocumentStore(":memory:")
    _seed_searchable(ds)
    rows = ds.search_memories(
        user_id="u1", memory_type="fact", importance_threshold=0.5, limit=10
    )
    # u1 + fact + >=0.5 → 只有 m1
    assert len(rows) == 1
    assert rows[0]["memory_id"] == "m1"


# ==================== update_memory ====================


def test_update_partial_content_only():
    ds = SQLiteDocumentStore(":memory:")
    ds.add_memory("m1", "u1", "old", "fact", 1, 0.5)
    assert ds.update_memory("m1", content="new")
    got = ds.get_memory("m1")
    assert got["content"] == "new"
    assert got["importance"] == 0.5  # 不变


def test_update_partial_importance_only():
    ds = SQLiteDocumentStore(":memory:")
    ds.add_memory("m1", "u1", "c", "fact", 1, 0.5)
    assert ds.update_memory("m1", importance=0.9)
    assert ds.get_memory("m1")["importance"] == 0.9


def test_update_properties():
    ds = SQLiteDocumentStore(":memory:")
    ds.add_memory("m1", "u1", "c", "fact", 1, 0.5, properties={"a": 1})
    assert ds.update_memory("m1", properties={"a": 2, "b": "x"})
    got = ds.get_memory("m1")
    assert got["properties"] == {"a": 2, "b": "x"}


def test_update_no_fields_returns_false():
    ds = SQLiteDocumentStore(":memory:")
    ds.add_memory("m1", "u1", "c", "fact", 1, 0.5)
    assert ds.update_memory("m1") is False


def test_update_nonexistent_returns_false():
    ds = SQLiteDocumentStore(":memory:")
    assert ds.update_memory("nope", content="x") is False


# ==================== delete_memory ====================


def test_delete_memory_returns_true_when_existed():
    ds = SQLiteDocumentStore(":memory:")
    ds.add_memory("m1", "u1", "c", "fact", 1, 0.5)
    assert ds.delete_memory("m1") is True
    assert ds.get_memory("m1") is None


def test_delete_nonexistent_returns_false():
    ds = SQLiteDocumentStore(":memory:")
    assert ds.delete_memory("nope") is False


# ==================== document API ====================


def test_add_and_get_document():
    ds = SQLiteDocumentStore(":memory:")
    doc_id = ds.add_document("hello world", metadata={"source": "x"})
    assert isinstance(doc_id, str) and len(doc_id) == 36  # uuid4

    got = ds.get_document(doc_id)
    assert got is not None
    assert got["content"] == "hello world"
    assert got["memory_type"] == "document"
    assert got["properties"]["source"] == "x"


def test_add_document_default_user_is_system():
    ds = SQLiteDocumentStore(":memory:")
    doc_id = ds.add_document("x")
    assert ds.get_document(doc_id)["user_id"] == "system"


def test_add_document_custom_user_id():
    ds = SQLiteDocumentStore(":memory:")
    doc_id = ds.add_document("x", metadata={"user_id": "alice"})
    assert ds.get_document(doc_id)["user_id"] == "alice"


# ==================== get_database_stats ====================


def test_database_stats_basic():
    ds = SQLiteDocumentStore(":memory:")
    _seed_searchable(ds)
    stats = ds.get_database_stats()
    assert stats["memories_count"] == 4
    assert stats["users_count"] == 2
    assert stats["concepts_count"] == 0
    assert stats["store_type"] == "sqlite"
    # memory_types 聚合
    assert stats["memory_types"] == {"fact": 3, "event": 1}
    # top_users
    assert stats["top_users"]["u1"] == 3


def test_database_stats_empty():
    ds = SQLiteDocumentStore(":memory:")
    stats = ds.get_database_stats()
    assert stats["memories_count"] == 0
    assert stats["users_count"] == 0
    assert stats["memory_types"] == {}


# ==================== 持久化（文件路径） ====================


def test_persistence_across_singleton_reset(tmp_path: Path):
    """关闭单例 + 重开同路径 → 数据仍在"""
    p = str(tmp_path / "p.db")
    a = SQLiteDocumentStore(p)
    a.add_memory("m1", "u1", "saved", "fact", 1, 0.5)

    SQLiteDocumentStore.reset_instances()
    b = SQLiteDocumentStore(p)
    assert b is not a
    got = b.get_memory("m1")
    assert got is not None
    assert got["content"] == "saved"


# ==================== 顶层导入 ====================


def test_top_level_retrieval_imports():
    from clear_agent.retrieval import (
        DocumentStore,
        SQLiteDocumentStore,
        EmbeddingModel,
        create_embedding_model,
    )

    assert DocumentStore is not None
    assert SQLiteDocumentStore is not None
    assert EmbeddingModel is not None
    assert callable(create_embedding_model)
