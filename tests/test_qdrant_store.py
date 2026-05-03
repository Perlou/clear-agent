"""QdrantVectorStore 测试

qdrant-client 是 optional dep —— 测试通过 monkeypatch 模块属性的方式
mock 整个 qdrant 客户端，不需要真实 Qdrant 服务。
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List
from unittest.mock import MagicMock, call, patch

import pytest

from clear_agent.core.exceptions import RetrievalException
from clear_agent.retrieval import (
    DEFAULT_COLLECTION,
    QDRANT_AVAILABLE,
    QdrantConnectionManager,
    QdrantVectorStore,
)
from clear_agent.retrieval.storage import qdrant_store as qstore


# ==================== 公共 fixture：注入伪 qdrant 类 ====================


@pytest.fixture
def fake_qdrant(monkeypatch):
    """把 qdrant_store 模块里依赖的 qdrant 类全部替换成 mock，让构造能跑通。

    yield 一个 dict 含：``client_cls`` / ``models`` / ``Distance`` 等，
    供测试函数检查调用。
    """
    fake_client_cls = MagicMock(name="QdrantClientCls")
    fake_models = MagicMock(name="qdrant_models")

    # Distance enum
    class _FakeDistance:
        COSINE = MagicMock(name="COSINE", value="Cosine")
        DOT = MagicMock(name="DOT", value="Dot")
        EUCLID = MagicMock(name="EUCLID", value="Euclid")

    # 为字符串值赋予真实属性
    _FakeDistance.COSINE.value = "Cosine"
    _FakeDistance.DOT.value = "Dot"
    _FakeDistance.EUCLID.value = "Euclid"

    fake_vp = MagicMock(name="VectorParamsCls")
    fake_filter = MagicMock(name="FilterCls")
    fake_field_cond = MagicMock(name="FieldConditionCls")
    fake_match_value = MagicMock(name="MatchValueCls")
    fake_point_struct = MagicMock(name="PointStructCls")

    monkeypatch.setattr(qstore, "QDRANT_AVAILABLE", True)
    monkeypatch.setattr(qstore, "QdrantClient", fake_client_cls)
    monkeypatch.setattr(qstore, "models", fake_models)
    monkeypatch.setattr(qstore, "Distance", _FakeDistance)
    monkeypatch.setattr(qstore, "VectorParams", fake_vp)
    monkeypatch.setattr(qstore, "Filter", fake_filter)
    monkeypatch.setattr(qstore, "FieldCondition", fake_field_cond)
    monkeypatch.setattr(qstore, "MatchValue", fake_match_value)
    monkeypatch.setattr(qstore, "PointStruct", fake_point_struct)

    # 让 client 实例的链式方法都返回 mock
    fake_client_inst = MagicMock(name="QdrantClientInst")
    # get_collections().collections 返回空（确保走"创建集合"分支）
    empty_collections = MagicMock()
    empty_collections.collections = []
    fake_client_inst.get_collections.return_value = empty_collections
    fake_client_cls.return_value = fake_client_inst

    # query_points 默认成功；search 兜底
    response = MagicMock()
    response.points = []
    fake_client_inst.query_points.return_value = response

    yield {
        "client_cls": fake_client_cls,
        "client": fake_client_inst,
        "models": fake_models,
        "Distance": _FakeDistance,
        "VectorParams": fake_vp,
        "Filter": fake_filter,
        "FieldCondition": fake_field_cond,
        "MatchValue": fake_match_value,
        "PointStruct": fake_point_struct,
    }
    # 清理 connection manager 单例
    QdrantConnectionManager.reset()


# ==================== Section A: 包未装时的行为 ====================


def test_qdrant_unavailable_when_package_missing(monkeypatch):
    """模拟 qdrant-client 未装 → ``QDRANT_AVAILABLE = False``

    用 monkeypatch 临时改模块级常量，而非依赖真实卸载。
    """
    from clear_agent.retrieval.storage import qdrant_store as qs

    monkeypatch.setattr(qs, "QDRANT_AVAILABLE", False)
    assert qs.QDRANT_AVAILABLE is False


def test_construct_without_package_raises_import_error(monkeypatch):
    from clear_agent.retrieval.storage import qdrant_store as qs

    monkeypatch.setattr(qs, "QDRANT_AVAILABLE", False)
    with pytest.raises(ImportError) as exc_info:
        qs.QdrantVectorStore()
    assert "qdrant-client" in str(exc_info.value)
    assert "retrieval-qdrant" in str(exc_info.value)


# ==================== Section B: _env_int + 距离映射 ====================


def test_env_int_default():
    assert QdrantVectorStore._env_int("__NONEXISTENT__", 42) == 42


def test_env_int_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("CA_TEST_ENV_INT", "not_an_int")
    assert QdrantVectorStore._env_int("CA_TEST_ENV_INT", 7) == 7


def test_env_int_valid(monkeypatch):
    monkeypatch.setenv("CA_TEST_ENV_INT", "99")
    assert QdrantVectorStore._env_int("CA_TEST_ENV_INT", 7) == 99


def test_distance_map_cosine(fake_qdrant):
    s = QdrantVectorStore(distance="cosine")
    assert s.distance is fake_qdrant["Distance"].COSINE


def test_distance_map_dot(fake_qdrant):
    s = QdrantVectorStore(distance="DOT")  # 大小写不敏感
    assert s.distance is fake_qdrant["Distance"].DOT


def test_distance_map_euclidean(fake_qdrant):
    s = QdrantVectorStore(distance="euclidean")
    assert s.distance is fake_qdrant["Distance"].EUCLID


def test_distance_map_unknown_falls_back_to_cosine(fake_qdrant):
    s = QdrantVectorStore(distance="bogus")
    assert s.distance is fake_qdrant["Distance"].COSINE


# ==================== Section C: 连接初始化 ====================


def test_init_local_when_no_url(fake_qdrant):
    QdrantVectorStore()
    fake_qdrant["client_cls"].assert_called_with(
        host="localhost", port=6333, timeout=30
    )


def test_init_url_only(fake_qdrant):
    QdrantVectorStore(url="http://example.com:6333")
    fake_qdrant["client_cls"].assert_called_with(
        url="http://example.com:6333", timeout=30
    )


def test_init_url_with_api_key(fake_qdrant):
    QdrantVectorStore(url="https://cloud.qdrant.io", api_key="sk-x")
    fake_qdrant["client_cls"].assert_called_with(
        url="https://cloud.qdrant.io", api_key="sk-x", timeout=30
    )


def test_connection_failure_raises_retrieval_exception(fake_qdrant):
    fake_qdrant["client"].get_collections.side_effect = RuntimeError("conn refused")
    with pytest.raises(RetrievalException) as exc_info:
        QdrantVectorStore()
    assert "Qdrant" in str(exc_info.value)


# ==================== Section D: 集合创建 ====================


def test_creates_collection_when_missing(fake_qdrant):
    QdrantVectorStore(collection_name="my_col", vector_size=128)
    fake_qdrant["client"].create_collection.assert_called_once()
    kwargs = fake_qdrant["client"].create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "my_col"


def test_uses_existing_collection(fake_qdrant):
    existing = MagicMock()
    existing.name = "my_col"
    coll_resp = MagicMock()
    coll_resp.collections = [existing]
    fake_qdrant["client"].get_collections.return_value = coll_resp

    QdrantVectorStore(collection_name="my_col")
    fake_qdrant["client"].create_collection.assert_not_called()


# ==================== Section E: add_vectors ====================


def test_add_vectors_empty_returns_false(fake_qdrant):
    s = QdrantVectorStore(vector_size=4)
    assert s.add_vectors([], []) is False


def test_add_vectors_dimension_mismatch_filters_out(fake_qdrant):
    s = QdrantVectorStore(vector_size=4)
    # 一条对一条错
    ok = s.add_vectors(
        vectors=[[0.1, 0.2, 0.3, 0.4], [0.1, 0.2]],  # 第二条维度错
        metadata=[{"a": 1}, {"a": 2}],
    )
    assert ok is True
    # upsert 被调用一次，points 只含 1 个有效点
    fake_qdrant["client"].upsert.assert_called_once()
    upsert_kwargs = fake_qdrant["client"].upsert.call_args.kwargs
    assert len(upsert_kwargs["points"]) == 1


def test_add_vectors_all_invalid_returns_false(fake_qdrant):
    s = QdrantVectorStore(vector_size=4)
    ok = s.add_vectors(
        vectors=[[1.0]],  # 维度错
        metadata=[{}],
    )
    assert ok is False


def test_add_vectors_with_explicit_uuid_keeps_id(fake_qdrant):
    s = QdrantVectorStore(vector_size=2)
    valid_uuid = "12345678-1234-1234-1234-123456789012"
    s.add_vectors(
        vectors=[[0.1, 0.2]],
        metadata=[{"x": 1}],
        ids=[valid_uuid],
    )
    point_struct_calls = fake_qdrant["PointStruct"].call_args_list
    assert len(point_struct_calls) == 1
    assert point_struct_calls[0].kwargs["id"] == valid_uuid


def test_add_vectors_non_uuid_string_id_replaced(fake_qdrant):
    s = QdrantVectorStore(vector_size=2)
    s.add_vectors(
        vectors=[[0.1, 0.2]],
        metadata=[{"x": 1}],
        ids=["not-a-uuid"],  # 会被替换为 uuid4
    )
    point_struct_calls = fake_qdrant["PointStruct"].call_args_list
    used_id = point_struct_calls[0].kwargs["id"]
    # 不再是原字符串
    assert used_id != "not-a-uuid"
    # 是有效 UUID
    import uuid as _uuid

    _uuid.UUID(used_id)


def test_add_vectors_int_id_kept_as_is(fake_qdrant):
    s = QdrantVectorStore(vector_size=2)
    s.add_vectors(
        vectors=[[0.1, 0.2]],
        metadata=[{}],
        ids=[42],
    )
    used_id = fake_qdrant["PointStruct"].call_args_list[0].kwargs["id"]
    assert used_id == 42


def test_add_vectors_external_field_normalized_to_bool(fake_qdrant):
    s = QdrantVectorStore(vector_size=2)
    s.add_vectors(
        vectors=[[0.1, 0.2]],
        metadata=[{"external": "yes"}],
    )
    payload = fake_qdrant["PointStruct"].call_args_list[0].kwargs["payload"]
    assert payload["external"] is True


def test_add_vectors_adds_timestamps_to_payload(fake_qdrant):
    s = QdrantVectorStore(vector_size=2)
    s.add_vectors(vectors=[[0.1, 0.2]], metadata=[{"x": 1}])
    payload = fake_qdrant["PointStruct"].call_args_list[0].kwargs["payload"]
    assert "timestamp" in payload
    assert "added_at" in payload
    # 原元数据保留
    assert payload["x"] == 1


def test_add_vectors_qdrant_failure_returns_false(fake_qdrant):
    fake_qdrant["client"].upsert.side_effect = RuntimeError("network down")
    s = QdrantVectorStore(vector_size=2)
    ok = s.add_vectors(vectors=[[0.1, 0.2]], metadata=[{}])
    assert ok is False


# ==================== Section F: search_similar ====================


def test_search_similar_dimension_mismatch_returns_empty(fake_qdrant):
    s = QdrantVectorStore(vector_size=4)
    assert s.search_similar([0.1, 0.2]) == []


def test_search_similar_uses_query_points(fake_qdrant):
    """新版 qdrant-client (>=1.16) 应优先调 query_points"""
    hit = MagicMock()
    hit.id = "p1"
    hit.score = 0.9
    hit.payload = {"k": "v"}
    response = MagicMock()
    response.points = [hit]
    fake_qdrant["client"].query_points.return_value = response

    s = QdrantVectorStore(vector_size=2)
    results = s.search_similar([0.1, 0.2], limit=5)

    fake_qdrant["client"].query_points.assert_called_once()
    assert len(results) == 1
    assert results[0] == {"id": "p1", "score": 0.9, "metadata": {"k": "v"}}


def test_search_similar_falls_back_to_search(fake_qdrant):
    """旧版 qdrant-client 没有 query_points → 走 search()"""
    fake_qdrant["client"].query_points.side_effect = AttributeError("no query_points")
    hit = MagicMock()
    hit.id = "x"
    hit.score = 0.5
    hit.payload = {}
    fake_qdrant["client"].search.return_value = [hit]

    s = QdrantVectorStore(vector_size=2)
    results = s.search_similar([0.1, 0.2])
    fake_qdrant["client"].search.assert_called_once()
    assert results[0]["id"] == "x"


def test_search_similar_with_where_builds_filter(fake_qdrant):
    s = QdrantVectorStore(vector_size=2)
    s.search_similar([0.1, 0.2], where={"memory_type": "fact", "user_id": "u1"})
    # FieldCondition 被调用 2 次
    assert fake_qdrant["FieldCondition"].call_count >= 2
    # Filter 被构造（must=...)
    fake_qdrant["Filter"].assert_called()


def test_search_similar_qdrant_failure_returns_empty(fake_qdrant):
    fake_qdrant["client"].query_points.side_effect = RuntimeError("boom")
    fake_qdrant["client"].search.side_effect = RuntimeError("boom2")
    s = QdrantVectorStore(vector_size=2)
    assert s.search_similar([0.1, 0.2]) == []


# ==================== Section G: delete ====================


def test_delete_vectors_empty_returns_true(fake_qdrant):
    s = QdrantVectorStore(vector_size=2)
    assert s.delete_vectors([]) is True
    fake_qdrant["client"].delete.assert_not_called()


def test_delete_vectors_calls_client(fake_qdrant):
    s = QdrantVectorStore(vector_size=2)
    assert s.delete_vectors(["a", "b"]) is True
    fake_qdrant["client"].delete.assert_called_once()


def test_delete_vectors_failure_returns_false(fake_qdrant):
    fake_qdrant["client"].delete.side_effect = RuntimeError("nope")
    s = QdrantVectorStore(vector_size=2)
    assert s.delete_vectors(["x"]) is False


def test_delete_memories_uses_payload_filter(fake_qdrant):
    s = QdrantVectorStore(vector_size=2)
    s.delete_memories(["m1", "m2"])
    # 走 FilterSelector 路径
    fake_qdrant["client"].delete.assert_called_once()
    # FieldCondition 用每个 memory_id 各调一次
    assert fake_qdrant["FieldCondition"].call_count >= 2


def test_delete_memories_empty_no_op(fake_qdrant):
    s = QdrantVectorStore(vector_size=2)
    s.delete_memories([])
    fake_qdrant["client"].delete.assert_not_called()


def test_delete_memories_failure_raises(fake_qdrant):
    fake_qdrant["client"].delete.side_effect = RuntimeError("boom")
    s = QdrantVectorStore(vector_size=2)
    with pytest.raises(RetrievalException):
        s.delete_memories(["m1"])


# ==================== Section H: clear / info / health ====================


def test_clear_collection(fake_qdrant):
    s = QdrantVectorStore(vector_size=2)
    assert s.clear_collection() is True
    fake_qdrant["client"].delete_collection.assert_called_once()


def test_clear_collection_failure(fake_qdrant):
    fake_qdrant["client"].delete_collection.side_effect = RuntimeError("x")
    s = QdrantVectorStore(vector_size=2)
    assert s.clear_collection() is False


def test_get_collection_info_success(fake_qdrant):
    info_obj = MagicMock()
    info_obj.vectors_count = 100
    info_obj.indexed_vectors_count = 95
    info_obj.points_count = 100
    info_obj.segments_count = 2
    fake_qdrant["client"].get_collection.return_value = info_obj

    s = QdrantVectorStore(vector_size=128, collection_name="c1")
    info = s.get_collection_info()
    assert info["vectors_count"] == 100
    assert info["points_count"] == 100
    assert info["config"]["vector_size"] == 128


def test_get_collection_info_failure_returns_empty(fake_qdrant):
    fake_qdrant["client"].get_collection.side_effect = RuntimeError("boom")
    s = QdrantVectorStore(vector_size=2)
    assert s.get_collection_info() == {}


def test_get_collection_stats_includes_store_type(fake_qdrant):
    info_obj = MagicMock()
    info_obj.vectors_count = 1
    info_obj.indexed_vectors_count = 1
    info_obj.points_count = 1
    info_obj.segments_count = 1
    fake_qdrant["client"].get_collection.return_value = info_obj
    s = QdrantVectorStore(vector_size=2)
    stats = s.get_collection_stats()
    assert stats["store_type"] == "qdrant"


def test_get_collection_stats_failure_still_has_store_type(fake_qdrant):
    fake_qdrant["client"].get_collection.side_effect = RuntimeError("boom")
    s = QdrantVectorStore(vector_size=2, collection_name="c1")
    stats = s.get_collection_stats()
    assert stats == {"store_type": "qdrant", "name": "c1"}


def test_health_check_true(fake_qdrant):
    s = QdrantVectorStore(vector_size=2)
    assert s.health_check() is True


def test_health_check_false_on_exception(fake_qdrant):
    s = QdrantVectorStore(vector_size=2)
    fake_qdrant["client"].get_collections.side_effect = RuntimeError("boom")
    assert s.health_check() is False


# ==================== Section I: ConnectionManager 单例 ====================


def test_connection_manager_returns_same_instance(fake_qdrant):
    QdrantConnectionManager.reset()
    a = QdrantConnectionManager.get_instance(collection_name="c1")
    b = QdrantConnectionManager.get_instance(collection_name="c1")
    assert a is b


def test_connection_manager_different_collection_different_instance(fake_qdrant):
    QdrantConnectionManager.reset()
    a = QdrantConnectionManager.get_instance(collection_name="c1")
    b = QdrantConnectionManager.get_instance(collection_name="c2")
    assert a is not b


def test_connection_manager_different_url_different_instance(fake_qdrant):
    QdrantConnectionManager.reset()
    a = QdrantConnectionManager.get_instance(url="http://x", collection_name="c")
    b = QdrantConnectionManager.get_instance(url="http://y", collection_name="c")
    assert a is not b


def test_connection_manager_reset(fake_qdrant):
    QdrantConnectionManager.reset()
    a = QdrantConnectionManager.get_instance(collection_name="c1")
    QdrantConnectionManager.reset()
    b = QdrantConnectionManager.get_instance(collection_name="c1")
    assert a is not b


# ==================== Section J: 顶层导出 ====================


def test_top_level_qdrant_exports():
    from clear_agent.retrieval import (
        DEFAULT_COLLECTION,
        QDRANT_AVAILABLE,
        QdrantConnectionManager,
        QdrantVectorStore,
    )

    assert DEFAULT_COLLECTION == "clear_agent_vectors"
    assert isinstance(QDRANT_AVAILABLE, bool)
    assert QdrantVectorStore is not None
    assert QdrantConnectionManager is not None
