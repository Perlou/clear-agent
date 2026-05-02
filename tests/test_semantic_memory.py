"""SemanticMemory + MemoryManager 测试

mock 嵌入模型 + Qdrant store；不依赖 spaCy（走 fallback）。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from clear_agent.memory import (
    BaseMemory,
    Entity,
    MemoryConfig,
    MemoryItem,
    MemoryManager,
    Relation,
    SemanticMemory,
    WorkingMemory,
)


# ==================== Section A: Entity / Relation ====================


def test_entity_basic():
    e = Entity(entity_id="e1", name="Alice", entity_type="PERSON")
    assert e.entity_id == "e1"
    assert e.name == "Alice"
    assert e.entity_type == "PERSON"
    assert e.frequency == 1


def test_entity_to_dict():
    e = Entity(entity_id="e1", name="X", description="d", properties={"k": "v"})
    d = e.to_dict()
    assert d["entity_id"] == "e1"
    assert d["properties"] == {"k": "v"}
    assert d["frequency"] == 1


def test_relation_basic():
    r = Relation(
        from_entity="e1", to_entity="e2", relation_type="KNOWS",
        strength=0.8, evidence="text",
    )
    assert r.from_entity == "e1"
    assert r.relation_type == "KNOWS"
    assert r.strength == 0.8
    assert r.frequency == 1


def test_relation_to_dict():
    r = Relation(from_entity="a", to_entity="b", relation_type="T")
    d = r.to_dict()
    assert d["from_entity"] == "a"
    assert d["relation_type"] == "T"


# ==================== Section B: SemanticMemory fixtures ====================


def _fake_embedder(dim: int = 384):
    e = MagicMock()
    e.encode.side_effect = lambda text: [float(len(text))] * dim
    return e


def _fake_store():
    s = MagicMock()
    s.add_vectors.return_value = True
    s.search_similar.return_value = []
    return s


def _make_sm(nlp=None, store=None, embedder=None) -> SemanticMemory:
    return SemanticMemory(
        config=MemoryConfig(),
        embedding_model=embedder or _fake_embedder(),
        vector_store=store or _fake_store(),
        nlp=nlp,
    )


def _make_item(
    mid: str = "m1",
    content: str = "Alice meets Bob",
    importance: float = 0.5,
    user_id: str = "u1",
    timestamp: datetime = None,
) -> MemoryItem:
    return MemoryItem(
        id=mid, content=content, memory_type="semantic",
        user_id=user_id, timestamp=timestamp or datetime.now(),
        importance=importance,
    )


# ==================== Section C: SemanticMemory.add ====================


def test_semantic_add_returns_id():
    sm = _make_sm()
    assert sm.add(_make_item("m1")) == "m1"


def test_semantic_add_calls_embedding_and_store():
    embedder = _fake_embedder()
    store = _fake_store()
    sm = _make_sm(embedder=embedder, store=store)
    sm.add(_make_item("m1", content="hello world"))
    embedder.encode.assert_called()
    store.add_vectors.assert_called_once()


def test_semantic_add_extracts_fallback_entities():
    """fallback：把单词当作实体（最多 5 个）"""
    sm = _make_sm()
    sm.add(_make_item("m1", content="alpha beta gamma delta epsilon zeta"))
    # 5 个实体上限
    assert len(sm.entities) == 5


def test_semantic_add_creates_cooccurrence_relations():
    """N 个实体 → N*(N-1)/2 条 CO_OCCURS 边"""
    sm = _make_sm()
    sm.add(_make_item("m1", content="alpha beta gamma"))
    # 3 个实体 → 3 条关系
    assert len(sm.relations) == 3
    assert all(r.relation_type == "CO_OCCURS" for r in sm.relations)


def test_semantic_add_writes_entities_to_metadata():
    sm = _make_sm()
    item = _make_item("m1", content="alpha beta")
    sm.add(item)
    assert "entities" in item.metadata
    assert "relations" in item.metadata
    assert len(item.metadata["entities"]) == 2


def test_semantic_add_appends_to_semantic_memories():
    sm = _make_sm()
    sm.add(_make_item("m1"))
    assert len(sm.semantic_memories) == 1


def test_semantic_add_uses_spacy_when_available():
    """nlp 不为 None 时使用 spaCy NER"""
    fake_nlp = MagicMock()
    fake_doc = MagicMock()
    fake_ent = MagicMock()
    fake_ent.text = "Alice"
    fake_ent.label_ = "PERSON"
    fake_doc.ents = [fake_ent]
    fake_nlp.return_value = fake_doc

    sm = _make_sm(nlp=fake_nlp)
    sm.add(_make_item("m1", content="Alice and Bob"))
    fake_nlp.assert_called()
    assert any(e.entity_type == "PERSON" for e in sm.entities.values())


def test_semantic_add_falls_back_when_spacy_fails():
    """spaCy 抛异常 → 走 fallback"""
    bad_nlp = MagicMock()
    bad_nlp.side_effect = RuntimeError("model down")
    sm = _make_sm(nlp=bad_nlp)
    sm.add(_make_item("m1", content="alpha beta"))
    # fallback 仍然提取了实体
    assert len(sm.entities) >= 2


def test_semantic_add_increments_entity_frequency_on_repeat():
    sm = _make_sm()
    sm.add(_make_item("m1", content="alpha beta"))
    sm.add(_make_item("m2", content="alpha beta"))
    # 同 entity_id 第二次出现 → frequency 累加
    e_ids = list(sm.entities.keys())
    assert any(sm.entities[eid].frequency >= 2 for eid in e_ids)


# ==================== Section D: SemanticMemory.retrieve ====================


def test_retrieve_empty_returns_empty():
    sm = _make_sm()
    assert sm.retrieve("anything") == []


def test_retrieve_via_graph_search_when_vector_empty():
    """vector_store 返回空 → 仅走 graph search（基于实体重叠）"""
    sm = _make_sm()
    sm.add(_make_item("m1", content="alpha beta gamma"))
    out = sm.retrieve("alpha")
    # graph 命中（含 'alpha' 实体）
    assert len(out) >= 1
    assert out[0].id == "m1"


def test_retrieve_combines_vector_and_graph():
    """vector_store 命中 → combined_score 含 vector_score"""
    store = _fake_store()
    store.search_similar.return_value = [
        {
            "id": "m1",
            "score": 0.9,
            "metadata": {
                "memory_id": "m1",
                "user_id": "u1",
                "content": "alpha beta",
                "memory_type": "semantic",
                "importance": 0.6,
                "timestamp": int(datetime.now().timestamp()),
                "entities": [],
            },
        }
    ]
    sm = _make_sm(store=store)
    sm.add(_make_item("m1", content="alpha beta"))
    out = sm.retrieve("alpha")
    assert len(out) >= 1
    assert "combined_score" in out[0].metadata
    assert "probability" in out[0].metadata


def test_retrieve_skips_forgotten():
    sm = _make_sm()
    item = _make_item("m1", content="alpha beta")
    sm.add(item)
    item.metadata["forgotten"] = True
    out = sm.retrieve("alpha")
    assert all(m.id != "m1" for m in out)


def test_retrieve_filters_by_user_id_via_graph():
    sm = _make_sm()
    sm.add(_make_item("m1", content="alpha", user_id="alice"))
    sm.add(_make_item("m2", content="alpha", user_id="bob"))
    out = sm.retrieve("alpha", user_id="alice")
    assert all(m.user_id == "alice" for m in out)


def test_retrieve_handles_string_timestamp_metadata():
    """metadata.timestamp 为 ISO 字符串时正确解析"""
    store = _fake_store()
    store.search_similar.return_value = [
        {
            "id": "m1",
            "score": 0.9,
            "metadata": {
                "memory_id": "m1",
                "content": "x",
                "user_id": "u1",
                "timestamp": "2026-05-02T10:00:00",
                "importance": 0.5,
                "entities": [],
            },
        }
    ]
    sm = _make_sm(store=store)
    out = sm.retrieve("x")
    assert len(out) == 1
    assert isinstance(out[0].timestamp, datetime)


# ==================== Section E: update / remove / has_memory ====================


def test_update_content_regenerates_entities():
    sm = _make_sm()
    sm.add(_make_item("m1", content="alpha beta"))
    initial_entities = len(sm.entities)
    sm.update("m1", content="gamma delta epsilon")
    # 新增了 entities（可能 + 旧 entities 还在）
    assert len(sm.entities) >= initial_entities


def test_update_importance_only():
    sm = _make_sm()
    sm.add(_make_item("m1", importance=0.5))
    assert sm.update("m1", importance=0.9)
    assert sm.semantic_memories[0].importance == 0.9


def test_update_metadata_merges():
    sm = _make_sm()
    sm.add(_make_item("m1"))
    assert sm.update("m1", metadata={"key": "value"})
    assert sm.semantic_memories[0].metadata["key"] == "value"


def test_update_nonexistent_returns_false():
    sm = _make_sm()
    assert sm.update("nope", content="x") is False


def test_remove_existing():
    sm = _make_sm()
    sm.add(_make_item("m1"))
    assert sm.remove("m1") is True
    assert not sm.has_memory("m1")


def test_remove_calls_qdrant_delete():
    store = _fake_store()
    sm = _make_sm(store=store)
    sm.add(_make_item("m1"))
    sm.remove("m1")
    store.delete_memories.assert_called_with(["m1"])


def test_remove_nonexistent_returns_false():
    sm = _make_sm()
    assert sm.remove("nope") is False


def test_has_memory_true_false():
    sm = _make_sm()
    sm.add(_make_item("m1"))
    assert sm.has_memory("m1")
    assert not sm.has_memory("m2")


# ==================== Section F: clear / get_stats / get_all ====================


def test_clear_resets_everything():
    store = _fake_store()
    sm = _make_sm(store=store)
    sm.add(_make_item("m1", content="a b c"))
    sm.clear()
    assert sm.semantic_memories == []
    assert sm.entities == {}
    assert sm.relations == []
    assert sm.memory_embeddings == {}
    store.clear_collection.assert_called_once()


def test_clear_swallows_qdrant_failure():
    store = _fake_store()
    store.clear_collection.side_effect = RuntimeError("boom")
    sm = _make_sm(store=store)
    sm.add(_make_item("m1"))
    sm.clear()  # 不应抛
    assert sm.semantic_memories == []


def test_get_stats_basic_fields():
    sm = _make_sm()
    sm.add(_make_item("m1", importance=0.7, content="alpha beta"))
    stats = sm.get_stats()
    assert stats["count"] == 1
    assert stats["memory_type"] == "semantic"
    assert stats["entities_count"] >= 2
    assert stats["graph_nodes"] >= 2
    assert stats["avg_importance"] == pytest.approx(0.7)


def test_get_stats_empty():
    sm = _make_sm()
    stats = sm.get_stats()
    assert stats["count"] == 0
    assert stats["avg_importance"] == 0.0


def test_get_all_returns_copy():
    sm = _make_sm()
    sm.add(_make_item("m1"))
    out = sm.get_all()
    out.clear()
    assert sm.has_memory("m1")


# ==================== Section G: forget ====================


def test_forget_importance_based():
    sm = _make_sm()
    sm.add(_make_item("low", importance=0.05))
    sm.add(_make_item("high", importance=0.9))
    n = sm.forget(strategy="importance_based", threshold=0.1)
    assert n == 1
    assert sm.has_memory("high")
    assert not sm.has_memory("low")


def test_forget_time_based():
    sm = _make_sm()
    old = datetime.now() - timedelta(days=60)
    sm.add(_make_item("old", timestamp=old))
    sm.add(_make_item("new"))
    n = sm.forget(strategy="time_based", max_age_days=30)
    assert n == 1
    assert sm.has_memory("new")
    assert not sm.has_memory("old")


def test_forget_capacity_based():
    cfg = MemoryConfig(max_capacity=2)
    sm = SemanticMemory(
        config=cfg, embedding_model=_fake_embedder(), vector_store=_fake_store()
    )
    for i in range(5):
        sm.add(_make_item(f"m{i}", importance=i / 10))
    n = sm.forget(strategy="capacity_based")
    assert n >= 1
    assert len(sm.semantic_memories) <= 2


# ==================== Section H: 实体查询 ====================


def test_get_entity():
    sm = _make_sm()
    sm.add(_make_item("m1", content="Alice"))
    eids = list(sm.entities.keys())
    e = sm.get_entity(eids[0])
    assert e is not None
    assert e.name == "Alice"


def test_get_entity_nonexistent_returns_none():
    sm = _make_sm()
    assert sm.get_entity("nope") is None


def test_search_entities_by_name():
    sm = _make_sm()
    sm.add(_make_item("m1", content="Alice Bob Charlie"))
    out = sm.search_entities("Alice", limit=10)
    assert any(e.name == "Alice" for e in out)


def test_search_entities_no_match():
    sm = _make_sm()
    sm.add(_make_item("m1", content="hello world"))
    assert sm.search_entities("zzzzzz") == []


def test_get_related_entities_via_graph():
    sm = _make_sm()
    sm.add(_make_item("m1", content="alpha beta gamma"))
    eids = list(sm.entities.keys())
    related = sm.get_related_entities(eids[0], max_hops=1)
    # alpha 与 beta、gamma 都有 CO_OCCURS 边
    assert len(related) >= 2


def test_get_related_entities_unknown_id_returns_empty():
    sm = _make_sm()
    assert sm.get_related_entities("nope") == []


def test_get_related_entities_filter_by_relation_type():
    sm = _make_sm()
    sm.add(_make_item("m1", content="alpha beta"))
    eids = list(sm.entities.keys())
    out = sm.get_related_entities(eids[0], relation_types=["CO_OCCURS"])
    assert all(r["relation_type"] == "CO_OCCURS" for r in out)
    out2 = sm.get_related_entities(eids[0], relation_types=["NONEXISTENT"])
    assert out2 == []


# ==================== Section I: export_knowledge_graph ====================


def test_export_knowledge_graph():
    sm = _make_sm()
    sm.add(_make_item("m1", content="alpha beta"))
    g = sm.export_knowledge_graph()
    assert "entities" in g
    assert "relations" in g
    assert "graph_stats" in g
    assert g["graph_stats"]["total_nodes"] == len(sm.entities)
    assert g["graph_stats"]["memory_nodes"] == 1


def test_export_knowledge_graph_empty():
    sm = _make_sm()
    g = sm.export_knowledge_graph()
    assert g["entities"] == {}
    assert g["relations"] == []


# ==================== Section J: _detect_language ====================


def test_detect_language_chinese():
    sm = _make_sm()
    assert sm._detect_language("你好世界这是中文") == "zh"


def test_detect_language_english():
    sm = _make_sm()
    assert sm._detect_language("hello world this is english") == "en"


def test_detect_language_empty():
    sm = _make_sm()
    assert sm._detect_language("") == "en"


def test_detect_language_mixed_low_chinese_ratio():
    sm = _make_sm()
    # 中文比例 < 30% → en
    out = sm._detect_language("hello world 你")
    assert out == "en"


# ==================== Section K: MemoryManager ====================


def test_manager_register_and_types():
    mgr = MemoryManager()
    wm = WorkingMemory(MemoryConfig())
    mgr.register("working", wm)
    assert "working" in mgr
    assert mgr.types() == ["working"]
    assert len(mgr) == 1


def test_manager_register_duplicate_raises():
    mgr = MemoryManager()
    mgr.register("w", WorkingMemory(MemoryConfig()))
    with pytest.raises(ValueError):
        mgr.register("w", WorkingMemory(MemoryConfig()))


def test_manager_unregister():
    mgr = MemoryManager()
    mgr.register("w", WorkingMemory(MemoryConfig()))
    assert mgr.unregister("w") is True
    assert mgr.unregister("nope") is False


def test_manager_get_returns_instance():
    mgr = MemoryManager()
    wm = WorkingMemory(MemoryConfig())
    mgr.register("w", wm)
    assert mgr.get("w") is wm
    assert mgr.get("nope") is None


def test_manager_add_routes_by_type():
    mgr = MemoryManager()
    wm = WorkingMemory(MemoryConfig())
    mgr.register("working", wm)
    item = _make_item("w1")
    item.memory_type = "working"
    mgr.add(item)
    assert wm.has_memory("w1")


def test_manager_add_unknown_type_raises():
    mgr = MemoryManager()
    item = _make_item("w1")
    item.memory_type = "ghost"
    with pytest.raises(ValueError):
        mgr.add(item)


def test_manager_retrieve_aggregates_across_types():
    mgr = MemoryManager()
    wm = WorkingMemory(MemoryConfig())
    sm = _make_sm()
    mgr.register("working", wm)
    mgr.register("semantic", sm)

    w_item = _make_item("w1", content="alpha beta")
    w_item.memory_type = "working"
    mgr.add(w_item)
    s_item = _make_item("s1", content="alpha gamma")
    mgr.add(s_item)

    out = mgr.retrieve("alpha", limit=10)
    ids = {m.id for m in out}
    assert "w1" in ids or "s1" in ids


def test_manager_retrieve_filter_by_types():
    mgr = MemoryManager()
    wm = WorkingMemory(MemoryConfig())
    sm = _make_sm()
    mgr.register("working", wm)
    mgr.register("semantic", sm)

    w_item = _make_item("w1", content="alpha beta")
    w_item.memory_type = "working"
    mgr.add(w_item)

    out = mgr.retrieve("alpha", memory_types=["semantic"])
    assert all(m.id != "w1" for m in out)


def test_manager_retrieve_empty_query_returns_empty():
    mgr = MemoryManager()
    mgr.register("w", WorkingMemory(MemoryConfig()))
    assert mgr.retrieve("") == []


def test_manager_update_searches_all_systems():
    mgr = MemoryManager()
    wm = WorkingMemory(MemoryConfig())
    mgr.register("working", wm)
    item = _make_item("w1", importance=0.3)
    item.memory_type = "working"
    mgr.add(item)

    assert mgr.update("w1", importance=0.9)
    assert wm.memories[0].importance == 0.9


def test_manager_update_with_explicit_type():
    mgr = MemoryManager()
    wm = WorkingMemory(MemoryConfig())
    mgr.register("working", wm)
    item = _make_item("w1", importance=0.3)
    item.memory_type = "working"
    mgr.add(item)
    assert mgr.update("w1", memory_type="working", importance=0.7)


def test_manager_update_unknown_type_returns_false():
    mgr = MemoryManager()
    assert mgr.update("x", memory_type="ghost") is False


def test_manager_remove_searches_all():
    mgr = MemoryManager()
    wm = WorkingMemory(MemoryConfig())
    mgr.register("working", wm)
    item = _make_item("w1")
    item.memory_type = "working"
    mgr.add(item)
    assert mgr.remove("w1")
    assert not mgr.has_memory("w1")


def test_manager_has_memory_across_systems():
    mgr = MemoryManager()
    wm = WorkingMemory(MemoryConfig())
    mgr.register("w", wm)
    item = _make_item("a")
    item.memory_type = "w"
    mgr.add(item)
    assert mgr.has_memory("a")
    assert not mgr.has_memory("nope")


def test_manager_clear_specific_type():
    mgr = MemoryManager()
    wm = WorkingMemory(MemoryConfig())
    sm = _make_sm()
    mgr.register("working", wm)
    mgr.register("semantic", sm)
    w_item = _make_item("w1")
    w_item.memory_type = "working"
    mgr.add(w_item)
    s_item = _make_item("s1")
    mgr.add(s_item)

    mgr.clear(memory_types=["working"])
    assert not wm.has_memory("w1")
    assert sm.has_memory("s1")


def test_manager_clear_all():
    mgr = MemoryManager()
    wm = WorkingMemory(MemoryConfig())
    mgr.register("working", wm)
    item = _make_item("w1")
    item.memory_type = "working"
    mgr.add(item)
    mgr.clear()
    assert not wm.has_memory("w1")


def test_manager_get_stats_aggregates():
    mgr = MemoryManager()
    wm = WorkingMemory(MemoryConfig())
    sm = _make_sm()
    mgr.register("working", wm)
    mgr.register("semantic", sm)

    w_item = _make_item("w1")
    w_item.memory_type = "working"
    mgr.add(w_item)
    mgr.add(_make_item("s1"))
    mgr.add(_make_item("s2"))

    stats = mgr.get_stats()
    assert stats["total_count"] == 3
    assert "working" in stats["by_type"]
    assert "semantic" in stats["by_type"]
    assert set(stats["registered_types"]) == {"working", "semantic"}


def test_manager_get_stats_swallows_per_type_errors():
    mgr = MemoryManager()
    bad = MagicMock(spec=BaseMemory)
    bad.get_stats.side_effect = RuntimeError("boom")
    mgr.memories["bad"] = bad
    stats = mgr.get_stats()
    assert "error" in stats["by_type"]["bad"]


# ==================== Section L: 顶层导出 ====================


def test_top_level_memory_exports():
    from clear_agent.memory import (
        BaseMemory,
        Entity,
        MemoryConfig,
        MemoryItem,
        MemoryManager,
        Relation,
        SemanticMemory,
        WorkingMemory,
    )

    for x in (
        BaseMemory, Entity, MemoryConfig, MemoryItem,
        MemoryManager, Relation, SemanticMemory, WorkingMemory,
    ):
        assert x is not None
