"""WorkingMemory + base 测试"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List
from unittest.mock import MagicMock

import pytest

from clear_agent.memory import BaseMemory, MemoryConfig, MemoryItem, WorkingMemory


# ==================== Section A: MemoryItem ====================


def test_memory_item_basic():
    item = MemoryItem(
        id="m1",
        content="hello",
        memory_type="working",
        user_id="u1",
        timestamp=datetime.now(),
        importance=0.7,
    )
    assert item.id == "m1"
    assert item.importance == 0.7
    assert item.metadata == {}


def test_memory_item_with_metadata():
    item = MemoryItem(
        id="m1",
        content="x",
        memory_type="working",
        user_id="u1",
        timestamp=datetime.now(),
        metadata={"source": "chat", "tag": "important"},
    )
    assert item.metadata["source"] == "chat"


def test_memory_item_default_importance():
    item = MemoryItem(
        id="m1", content="x", memory_type="working",
        user_id="u1", timestamp=datetime.now(),
    )
    assert item.importance == 0.5


def test_memory_item_required_fields_missing():
    with pytest.raises(Exception):
        MemoryItem(id="m1")  # 大量字段缺失 → ValidationError


# ==================== Section B: MemoryConfig ====================


def test_memory_config_defaults():
    c = MemoryConfig()
    assert c.max_capacity == 100
    assert c.working_memory_capacity == 10
    assert c.working_memory_tokens == 2000
    assert c.working_memory_ttl_minutes == 120
    assert c.decay_factor == 0.95
    assert c.importance_threshold == 0.1
    assert "text" in c.perceptual_memory_modalities


def test_memory_config_override():
    c = MemoryConfig(
        working_memory_capacity=50,
        working_memory_tokens=5000,
        working_memory_ttl_minutes=60,
    )
    assert c.working_memory_capacity == 50
    assert c.working_memory_tokens == 5000
    assert c.working_memory_ttl_minutes == 60


# ==================== Section C: BaseMemory ====================


def test_base_memory_is_abstract():
    """BaseMemory 直接实例化抛 TypeError"""
    with pytest.raises(TypeError):
        BaseMemory(MemoryConfig())  # type: ignore[abstract]


def test_base_memory_generate_id_unique():
    """子类工具方法可独立测试"""
    wm = WorkingMemory(MemoryConfig())
    a = wm._generate_id()
    b = wm._generate_id()
    assert a != b
    assert len(a) == 36  # uuid4


def test_base_memory_calculate_importance_long_content():
    wm = WorkingMemory(MemoryConfig())
    base = 0.5
    long_content = "x" * 200
    out = wm._calculate_importance(long_content, base)
    assert out == pytest.approx(0.6)  # +0.1 due to len > 100


def test_base_memory_calculate_importance_with_keyword():
    wm = WorkingMemory(MemoryConfig())
    out = wm._calculate_importance("这是一个重要消息", base_importance=0.5)
    assert out == pytest.approx(0.7)  # +0.2 keyword "重要"


def test_base_memory_calculate_importance_clipped():
    wm = WorkingMemory(MemoryConfig())
    long_kw = "重要 " + "x" * 200  # +0.1 长度 + 0.2 关键词
    out = wm._calculate_importance(long_kw, base_importance=0.9)
    assert out == 1.0  # clip


def test_base_memory_str_format():
    wm = WorkingMemory(MemoryConfig())
    s = str(wm)
    assert "WorkingMemory" in s
    assert "count=" in s


# ==================== Section D: WorkingMemory.add ====================


def _make_item(
    mid: str = "m1",
    content: str = "hello world",
    importance: float = 0.5,
    user_id: str = "u1",
    timestamp: datetime = None,
) -> MemoryItem:
    return MemoryItem(
        id=mid,
        content=content,
        memory_type="working",
        user_id=user_id,
        timestamp=timestamp or datetime.now(),
        importance=importance,
    )


def test_add_returns_memory_id():
    wm = WorkingMemory(MemoryConfig())
    rid = wm.add(_make_item("m1"))
    assert rid == "m1"


def test_add_increments_token_count():
    wm = WorkingMemory(MemoryConfig())
    wm.add(_make_item("m1", content="three word here"))
    assert wm.current_tokens == 3


def test_add_pushes_to_heap():
    wm = WorkingMemory(MemoryConfig())
    wm.add(_make_item("m1"))
    assert len(wm.memory_heap) == 1


def test_add_enforces_capacity():
    """超过 max_capacity 时挤掉低优先级"""
    cfg = MemoryConfig(working_memory_capacity=3)
    wm = WorkingMemory(cfg)
    for i in range(5):
        wm.add(_make_item(f"m{i}", importance=i / 10))
    # 容量 3 但加了 5 → 留下优先级最高的 3 条
    assert len(wm.memories) <= 3


def test_add_enforces_token_limit():
    """超过 max_tokens 时也挤掉"""
    cfg = MemoryConfig(working_memory_tokens=10, working_memory_capacity=100)
    wm = WorkingMemory(cfg)
    # 每条 5 token，加 5 条 → 总 25 token，应被挤
    for i in range(5):
        wm.add(_make_item(f"m{i}", content="a b c d e"))
    assert wm.current_tokens <= 10


# ==================== Section E: retrieve ====================


def test_retrieve_empty_memory():
    wm = WorkingMemory(MemoryConfig())
    assert wm.retrieve("anything") == []


def test_retrieve_keyword_match():
    wm = WorkingMemory(MemoryConfig())
    wm.add(_make_item("m1", content="apple banana cherry"))
    wm.add(_make_item("m2", content="dog cat mouse"))
    found = wm.retrieve("apple")
    assert len(found) >= 1
    assert found[0].id == "m1"


def test_retrieve_no_match_returns_empty():
    wm = WorkingMemory(MemoryConfig())
    wm.add(_make_item("m1", content="apple banana"))
    assert wm.retrieve("zzzzzz") == []


def test_retrieve_filter_by_user_id():
    wm = WorkingMemory(MemoryConfig())
    wm.add(_make_item("m1", content="hello", user_id="alice"))
    wm.add(_make_item("m2", content="hello", user_id="bob"))
    out = wm.retrieve("hello", user_id="alice")
    assert all(m.user_id == "alice" for m in out)


def test_retrieve_skips_forgotten_memories():
    wm = WorkingMemory(MemoryConfig())
    item = _make_item("m1", content="hello")
    wm.add(item)
    # 标记为遗忘
    item.metadata["forgotten"] = True
    assert wm.retrieve("hello") == []


def test_retrieve_respects_limit():
    wm = WorkingMemory(MemoryConfig(working_memory_capacity=20))
    for i in range(10):
        wm.add(_make_item(f"m{i}", content="hello world"))
    out = wm.retrieve("hello", limit=3)
    assert len(out) == 3


def test_retrieve_higher_importance_ranks_higher():
    wm = WorkingMemory(MemoryConfig(working_memory_capacity=20))
    wm.add(_make_item("low", content="apple banana", importance=0.1))
    wm.add(_make_item("high", content="apple banana", importance=0.9))
    out = wm.retrieve("apple")
    # 重要性高的应排在前面
    assert out[0].id == "high"


# ==================== Section F: update ====================


def test_update_content_adjusts_token_count():
    wm = WorkingMemory(MemoryConfig())
    wm.add(_make_item("m1", content="a b c"))  # 3 tokens
    initial = wm.current_tokens
    assert wm.update("m1", content="a b c d e")  # 5 tokens
    assert wm.current_tokens == initial + 2


def test_update_importance_only():
    wm = WorkingMemory(MemoryConfig())
    wm.add(_make_item("m1", importance=0.3))
    assert wm.update("m1", importance=0.8)
    assert wm.memories[0].importance == 0.8


def test_update_metadata_merges():
    wm = WorkingMemory(MemoryConfig())
    item = _make_item("m1")
    item.metadata = {"a": 1}
    wm.add(item)
    assert wm.update("m1", metadata={"b": 2})
    assert wm.memories[0].metadata == {"a": 1, "b": 2}


def test_update_nonexistent_returns_false():
    wm = WorkingMemory(MemoryConfig())
    assert wm.update("nope", content="x") is False


# ==================== Section G: remove ====================


def test_remove_existing_returns_true():
    wm = WorkingMemory(MemoryConfig())
    wm.add(_make_item("m1", content="x y z"))
    assert wm.remove("m1") is True
    assert not wm.has_memory("m1")


def test_remove_decrements_token_count():
    wm = WorkingMemory(MemoryConfig())
    wm.add(_make_item("m1", content="a b c"))
    wm.remove("m1")
    assert wm.current_tokens == 0


def test_remove_nonexistent_returns_false():
    wm = WorkingMemory(MemoryConfig())
    assert wm.remove("nope") is False


# ==================== Section H: has_memory / clear ====================


def test_has_memory_true_false():
    wm = WorkingMemory(MemoryConfig())
    wm.add(_make_item("m1"))
    assert wm.has_memory("m1")
    assert not wm.has_memory("m2")


def test_clear_resets_state():
    wm = WorkingMemory(MemoryConfig())
    for i in range(5):
        wm.add(_make_item(f"m{i}"))
    wm.clear()
    assert wm.memories == []
    assert wm.memory_heap == []
    assert wm.current_tokens == 0


# ==================== Section I: get_stats ====================


def test_get_stats_basic_fields():
    wm = WorkingMemory(MemoryConfig())
    wm.add(_make_item("m1", importance=0.6))
    wm.add(_make_item("m2", importance=0.8))
    stats = wm.get_stats()
    assert stats["count"] == 2
    assert stats["memory_type"] == "working"
    assert stats["avg_importance"] == pytest.approx(0.7)
    assert 0 <= stats["capacity_usage"] <= 1
    assert "session_duration_minutes" in stats


def test_get_stats_empty():
    wm = WorkingMemory(MemoryConfig())
    stats = wm.get_stats()
    assert stats["count"] == 0
    assert stats["avg_importance"] == 0.0


# ==================== Section J: get_recent / get_important / get_all / get_context_summary ====================


def test_get_recent_orders_by_timestamp_desc():
    wm = WorkingMemory(MemoryConfig())
    base = datetime.now()
    for i in range(3):
        wm.add(_make_item(f"m{i}", timestamp=base + timedelta(seconds=i)))
    out = wm.get_recent(limit=2)
    assert out[0].id == "m2"
    assert out[1].id == "m1"


def test_get_important_orders_by_importance_desc():
    wm = WorkingMemory(MemoryConfig())
    for i, imp in enumerate([0.3, 0.9, 0.5]):
        wm.add(_make_item(f"m{i}", importance=imp))
    out = wm.get_important(limit=2)
    assert out[0].importance == 0.9
    assert out[1].importance == 0.5


def test_get_all_returns_copy():
    wm = WorkingMemory(MemoryConfig())
    wm.add(_make_item("m1"))
    out = wm.get_all()
    out.clear()  # 不应影响内部
    assert wm.has_memory("m1")


def test_get_context_summary_empty():
    wm = WorkingMemory(MemoryConfig())
    s = wm.get_context_summary()
    assert "No working memories" in s


def test_get_context_summary_has_label_and_content():
    wm = WorkingMemory(MemoryConfig())
    wm.add(_make_item("m1", content="alpha"))
    s = wm.get_context_summary()
    assert "Working Memory Context" in s
    assert "alpha" in s


def test_get_context_summary_truncates():
    wm = WorkingMemory(MemoryConfig())
    wm.add(_make_item("m1", content="x" * 1000, importance=0.9))
    s = wm.get_context_summary(max_length=200)
    # 包含截断标记
    assert "..." in s


# ==================== Section K: forget 策略 ====================


def test_forget_importance_based():
    wm = WorkingMemory(MemoryConfig(working_memory_capacity=20))
    wm.add(_make_item("low", importance=0.05))
    wm.add(_make_item("high", importance=0.9))
    n = wm.forget(strategy="importance_based", threshold=0.1)
    assert n >= 1
    assert wm.has_memory("high")
    assert not wm.has_memory("low")


def test_forget_time_based():
    # ttl 设得很大让 add() 时不被 TTL 自动清掉；old_time 2 天前 → max_age_days=1 会清
    wm = WorkingMemory(
        MemoryConfig(working_memory_capacity=20, working_memory_ttl_minutes=60 * 24 * 30)
    )
    old_time = datetime.now() - timedelta(days=2)
    wm.add(_make_item("old", timestamp=old_time))
    wm.add(_make_item("new"))
    n = wm.forget(strategy="time_based", max_age_days=1)
    assert n >= 1
    assert wm.has_memory("new")
    assert not wm.has_memory("old")


def test_forget_capacity_based():
    cfg = MemoryConfig(working_memory_capacity=2, working_memory_tokens=10000)
    wm = WorkingMemory(cfg)
    # 直接在 memories 列表里塞入超过 capacity 的项目，然后调 forget
    # （由于 add 自身会 enforce capacity，先扩大配置再缩回）
    wm.max_capacity = 5
    for i in range(4):
        wm.add(_make_item(f"m{i}", importance=i / 10))
    wm.max_capacity = 2  # 缩回
    n = wm.forget(strategy="capacity_based")
    assert n >= 1
    assert len(wm.memories) <= 2


def test_forget_unknown_strategy_only_runs_ttl():
    """未知 strategy 仅触发 TTL 过期"""
    cfg = MemoryConfig(working_memory_ttl_minutes=10000)
    wm = WorkingMemory(cfg)
    wm.add(_make_item("m1"))
    n = wm.forget(strategy="unknown_strategy")
    # 没东西过期 → 0
    assert n == 0
    assert wm.has_memory("m1")


# ==================== Section L: TTL 过期机制 ====================


def test_ttl_expires_old_memories_via_get_stats():
    """get_stats 内部会调 _expire_old_memories"""
    cfg = MemoryConfig(working_memory_ttl_minutes=1)
    wm = WorkingMemory(cfg)
    old_time = datetime.now() - timedelta(minutes=5)
    wm.add(_make_item("old", timestamp=old_time))
    wm.add(_make_item("new"))
    stats = wm.get_stats()
    # old 已过期
    assert stats["count"] == 1


def test_ttl_expires_on_retrieve():
    cfg = MemoryConfig(working_memory_ttl_minutes=1)
    wm = WorkingMemory(cfg)
    old_time = datetime.now() - timedelta(minutes=5)
    wm.add(_make_item("old", content="hello", timestamp=old_time))
    wm.add(_make_item("new", content="hello"))
    out = wm.retrieve("hello")
    ids = {m.id for m in out}
    assert "old" not in ids
    assert "new" in ids


def test_ttl_resets_token_count_after_expiration():
    cfg = MemoryConfig(working_memory_ttl_minutes=1)
    wm = WorkingMemory(cfg)
    old_time = datetime.now() - timedelta(minutes=5)
    wm.add(_make_item("old", content="a b c", timestamp=old_time))
    wm.get_stats()  # 触发过期
    assert wm.current_tokens == 0


# ==================== Section M: 优先级 / 时间衰减 ====================


def test_calculate_priority_uses_importance_and_decay():
    wm = WorkingMemory(MemoryConfig())
    fresh = _make_item("f", importance=0.5)
    p = wm._calculate_priority(fresh)
    assert 0 < p <= 0.5  # 时间衰减 ≤ 1


def test_time_decay_minimum_floor():
    wm = WorkingMemory(MemoryConfig())
    very_old = datetime.now() - timedelta(days=365)
    decay = wm._calculate_time_decay(very_old)
    assert decay >= 0.1  # 最小保持 10%


def test_time_decay_recent_close_to_one():
    wm = WorkingMemory(MemoryConfig())
    decay = wm._calculate_time_decay(datetime.now())
    assert decay > 0.99


# ==================== Section N: 顶层导出 ====================


def test_top_level_memory_exports():
    from clear_agent.memory import (
        BaseMemory,
        MemoryConfig,
        MemoryItem,
        WorkingMemory,
    )

    assert BaseMemory is not None
    assert MemoryConfig is not None
    assert MemoryItem is not None
    assert WorkingMemory is not None
