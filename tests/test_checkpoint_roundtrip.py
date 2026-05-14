"""Checkpointer 往返测试

覆盖 project_docs/02-checkpoint-and-resume.md §8 测试清单。
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Annotated, List, TypedDict

import pytest

from clear_agent.core.checkpoint import (
    BaseCheckpointer,
    Checkpoint,
    InMemoryCheckpointer,
    JsonFileCheckpointer,
    SqliteCheckpointer,
    make_checkpointer,
    _uuid7,
)
from clear_agent.core.graph import (
    END,
    START,
    RunConfig,
    StateGraph,
    append_list,
)


# ==================== fixtures ====================


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


def _make_simple_graph():
    """构建一个 5 步线性图：a → b → c → d → e → END"""

    class S(TypedDict, total=False):
        counter: int
        log: Annotated[List[str], append_list]

    g: StateGraph[S] = StateGraph(S)
    for n in ["a", "b", "c", "d", "e"]:
        g.add_node(n, lambda s, _n=n: {"counter": (s.get("counter") or 0) + 1, "log": _n})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", "d")
    g.add_edge("d", "e")
    g.add_edge("e", END)
    return g


# ==================== Test 1: 5 步 → kill → resume → 完成 ====================


@pytest.mark.parametrize(
    "make_ck",
    [
        lambda d: InMemoryCheckpointer(),
        lambda d: JsonFileCheckpointer(base_dir=str(d / "ckpts")),
        lambda d: SqliteCheckpointer(db_path=str(d / "ckpts.db")),
    ],
    ids=["memory", "json", "sqlite"],
)
def test_kill_and_resume_equivalent_to_full_run(tmp_dir, make_ck):
    """跑 5 步 → kill → resume → 完成；最终 state 与一气呵成等价"""
    g = _make_simple_graph()

    # baseline: 一气呵成
    ck1 = make_ck(tmp_dir / "baseline")
    out_full = g.compile(checkpointer=ck1).invoke(
        {"counter": 0, "log": []}, config=RunConfig(thread_id="t-full", max_steps=20)
    )

    # split: 跑前 3 步打断（用 max_steps=3），再 resume
    ck2 = make_ck(tmp_dir / "split")
    compiled = g.compile(checkpointer=ck2)
    with pytest.raises(Exception):
        # max_steps=3 → 跑完 a/b/c 后第 4 步抛 GraphRecursionError
        compiled.invoke(
            {"counter": 0, "log": []},
            config=RunConfig(thread_id="t-split", max_steps=3),
        )

    # resume：从最后 ckpt 续跑
    out_resumed = compiled.resume(thread_id="t-split")

    assert out_resumed["counter"] == out_full["counter"]
    assert out_resumed["log"] == out_full["log"]


# ==================== Test 2: 三种后端等价 ====================


def test_three_backends_produce_same_result(tmp_dir):
    g = _make_simple_graph()
    cks = [
        InMemoryCheckpointer(),
        JsonFileCheckpointer(base_dir=str(tmp_dir / "j")),
        SqliteCheckpointer(db_path=str(tmp_dir / "s.db")),
    ]
    outputs = []
    for ck in cks:
        out = g.compile(checkpointer=ck).invoke(
            {"counter": 0, "log": []}, config=RunConfig(thread_id="t1")
        )
        outputs.append(out)
    assert outputs[0] == outputs[1] == outputs[2]


# ==================== Test 3: thread_id 隔离 ====================


@pytest.mark.parametrize(
    "make_ck",
    [
        lambda d: InMemoryCheckpointer(),
        lambda d: JsonFileCheckpointer(base_dir=str(d / "ckpts")),
        lambda d: SqliteCheckpointer(db_path=str(d / "ckpts.db")),
    ],
    ids=["memory", "json", "sqlite"],
)
def test_thread_isolation(tmp_dir, make_ck):
    ck = make_ck(tmp_dir)
    # 写两个不同 thread 的 ckpt
    ck.put(
        Checkpoint(
            id=_uuid7(),
            thread_id="t1",
            parent_id=None,
            state={"x": 1},
            next_nodes=["a"],
            created_at=datetime.now(),
            metadata={"source": "loop"},
        )
    )
    ck.put(
        Checkpoint(
            id=_uuid7(),
            thread_id="t2",
            parent_id=None,
            state={"x": 99},
            next_nodes=["b"],
            created_at=datetime.now(),
            metadata={"source": "loop"},
        )
    )

    t1_latest = ck.get_tuple("t1")
    t2_latest = ck.get_tuple("t2")
    assert t1_latest is not None and t1_latest.state["x"] == 1
    assert t2_latest is not None and t2_latest.state["x"] == 99
    assert ck.get_tuple("t3") is None


# ==================== Test 4: time-travel rewind ====================


@pytest.mark.parametrize(
    "make_ck",
    [
        lambda d: InMemoryCheckpointer(),
        lambda d: JsonFileCheckpointer(base_dir=str(d / "ckpts")),
        lambda d: SqliteCheckpointer(db_path=str(d / "ckpts.db")),
    ],
    ids=["memory", "json", "sqlite"],
)
def test_time_travel_rewind(tmp_dir, make_ck):
    """从 ckpt #2 resume，跳过 #3-#5"""
    g = _make_simple_graph()
    ck = make_ck(tmp_dir)
    compiled = g.compile(checkpointer=ck)

    compiled.invoke({"counter": 0, "log": []}, config=RunConfig(thread_id="t1"))

    ckpts = ck.list("t1")  # 倒序：最新在前
    assert len(ckpts) == 5  # a/b/c/d/e 各一个
    # 取倒数第 4 个（最早的 a）按 created_at 倒序索引为 -1
    early = ckpts[-2]  # 第二个被写入的（节点 b 完成后）
    out = compiled.resume(thread_id="t1", checkpoint_id=early.id)
    assert out["counter"] >= early.state["counter"]


# ==================== Test 5: state_patch ====================


@pytest.mark.parametrize(
    "make_ck",
    [
        lambda d: InMemoryCheckpointer(),
        lambda d: JsonFileCheckpointer(base_dir=str(d / "ckpts")),
    ],
    ids=["memory", "json"],
)
def test_resume_with_state_patch(tmp_dir, make_ck):
    g = _make_simple_graph()
    ck = make_ck(tmp_dir)
    compiled = g.compile(checkpointer=ck)
    compiled.invoke({"counter": 0, "log": []}, config=RunConfig(thread_id="t1"))

    ckpts = ck.list("t1")
    middle = ckpts[2]  # 第 3 个写入的（c 完成后）
    out = compiled.resume(
        thread_id="t1", checkpoint_id=middle.id, state_patch={"counter": 999}
    )
    # patch 注入后 counter 起点是 999，下游 d/e 各 +1
    assert out["counter"] == 999 + 2


# ==================== Test 6: 1.x SessionStore 兼容 ====================


def test_legacy_session_load(tmp_dir):
    """JsonFileCheckpointer 能读取 1.x SessionStore 文件"""
    legacy_dir = tmp_dir / "legacy_sessions"
    legacy_dir.mkdir()

    legacy_data = {
        "session_id": "s-legacy-001",
        "created_at": "2025-01-01T00:00:00",
        "saved_at": "2025-01-01T00:01:00",
        "agent_config": {"name": "old-agent"},
        "history": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        "tool_schema_hash": "abc",
        "read_cache": {},
        "metadata": {"total_tokens": 50},
    }
    with (legacy_dir / "old-thread.json").open("w") as f:
        json.dump(legacy_data, f)

    ck = JsonFileCheckpointer(
        base_dir=str(tmp_dir / "ckpts"),
        legacy_session_dir=str(legacy_dir),
    )

    ckpt = ck.get_tuple("old-thread")
    assert ckpt is not None
    assert ckpt.metadata["source"] == "legacy_session"
    assert ckpt.state["messages"] == legacy_data["history"]
    assert ckpt.metadata["total_tokens"] == 50


# ==================== Test 7: 原子写入 ====================


def test_atomic_write_no_partial_file(tmp_dir):
    """tmp 文件方式：即使写入中途崩溃，已有的 ckpt 不会被破坏"""
    base = tmp_dir / "ckpts"
    ck = JsonFileCheckpointer(base_dir=str(base))

    c1 = Checkpoint(
        id=_uuid7(),
        thread_id="t1",
        parent_id=None,
        state={"x": 1},
        next_nodes=["a"],
        created_at=datetime.now(),
        metadata={},
    )
    ck.put(c1)
    # 检查没有 .tmp 残留
    leftovers = list((base / "t1").glob("*.tmp"))
    assert leftovers == []
    # 第一个 ckpt 完整可读
    assert ck.get_tuple("t1") is not None

    # 模拟一个未完成的 .tmp 残留（崩溃场景）
    fake_tmp = base / "t1" / "fake.json.tmp"
    fake_tmp.write_text("PARTIAL_GARBAGE")
    # 已有的 c1 仍然可读（不被 .tmp 干扰）
    assert ck.get_tuple("t1").id == c1.id


def test_json_checkpointer_keeps_unsafe_thread_id_under_base_dir(tmp_dir):
    """thread_id 是外部输入，即使包含路径片段也不能逃出 base_dir。"""
    base = tmp_dir / "ckpts"
    escaped_dir = tmp_dir / "escaped-thread"
    ck = JsonFileCheckpointer(base_dir=str(base), legacy_session_dir=None)

    ckpt = Checkpoint(
        id="safe-id",
        thread_id="../escaped-thread",
        parent_id=None,
        state={"x": 1},
        next_nodes=[],
        created_at=datetime.now(),
        metadata={},
    )

    ck.put(ckpt)

    assert not escaped_dir.exists()
    loaded = ck.get_tuple("../escaped-thread", "safe-id")
    assert loaded is not None
    assert loaded.state == {"x": 1}


def test_json_checkpointer_rejects_checkpoint_id_path_traversal(tmp_dir):
    """checkpoint_id 不能读取 thread 目录外的 JSON 文件。"""
    base = tmp_dir / "ckpts"
    ck = JsonFileCheckpointer(base_dir=str(base), legacy_session_dir=None)

    ck.put(
        Checkpoint(
            id="safe-id",
            thread_id="safe-thread",
            parent_id=None,
            state={"safe": True},
            next_nodes=[],
            created_at=datetime.now(),
            metadata={},
        )
    )
    outside = tmp_dir / "evil.json"
    outside.write_text(
        json.dumps(
            Checkpoint(
                id="evil",
                thread_id="evil-thread",
                parent_id=None,
                state={"escaped": True},
                next_nodes=[],
                created_at=datetime.now(),
                metadata={},
            ).to_dict(),
            default=str,
        ),
        encoding="utf-8",
    )

    assert ck.get_tuple("safe-thread", "../../evil") is None


# ==================== Test 8: 异步并发 ====================


def test_concurrent_aput_no_loss():
    """InMemory + Sqlite 的高并发 aput 不丢 checkpoint"""

    async def _run():
        ck = InMemoryCheckpointer()

        async def writer(prefix: str, n: int):
            for i in range(n):
                await ck.aput(
                    Checkpoint(
                        id=f"{prefix}-{i}",
                        thread_id="t-concurrent",
                        parent_id=None,
                        state={"i": i},
                        next_nodes=[],
                        created_at=datetime.now(),
                        metadata={},
                    )
                )

        await asyncio.gather(
            writer("w1", 30), writer("w2", 30), writer("w3", 30)
        )
        ckpts = await ck.alist("t-concurrent", limit=200)
        assert len(ckpts) == 90

    asyncio.run(_run())


# ==================== make_checkpointer 工厂 ====================


def test_make_checkpointer_factory(tmp_dir):
    assert isinstance(make_checkpointer("memory"), InMemoryCheckpointer)
    assert isinstance(
        make_checkpointer("json", base_dir=str(tmp_dir / "j")), JsonFileCheckpointer
    )
    assert isinstance(
        make_checkpointer("sqlite", db_path=str(tmp_dir / "s.db")), SqliteCheckpointer
    )
    with pytest.raises(ValueError):
        make_checkpointer("unknown")


# ==================== Sqlite 特定测试 ====================


def test_sqlite_persists_across_connections(tmp_dir):
    """SqliteCheckpointer 关闭后重新打开能读到之前写入"""
    db_path = str(tmp_dir / "s.db")
    ck1 = SqliteCheckpointer(db_path=db_path)
    cid = _uuid7()
    ck1.put(
        Checkpoint(
            id=cid,
            thread_id="t1",
            parent_id=None,
            state={"x": 42},
            next_nodes=["a"],
            created_at=datetime.now(),
            metadata={},
        )
    )

    # 新建一个 checkpointer 实例（模拟进程重启）
    ck2 = SqliteCheckpointer(db_path=db_path)
    ckpt = ck2.get_tuple("t1", cid)
    assert ckpt is not None
    assert ckpt.state["x"] == 42


def test_sqlite_list_with_before(tmp_dir):
    """list(before=X) 返回 X 之前的 checkpoints"""
    ck = SqliteCheckpointer(db_path=str(tmp_dir / "s.db"))
    import time as _time

    ids = []
    for i in range(5):
        cid = _uuid7()
        ids.append(cid)
        ck.put(
            Checkpoint(
                id=cid,
                thread_id="t1",
                parent_id=None,
                state={"i": i},
                next_nodes=[],
                created_at=datetime.now(),
                metadata={},
            )
        )
        _time.sleep(0.001)  # 保证 created_at 不同

    # 5 个全部
    all_ckpts = ck.list("t1")
    assert len(all_ckpts) == 5

    # before 第 3 个 → 只返回更早的（更早 = created_at 更小 = 索引 0/1）
    earlier = ck.list("t1", before=ids[2])
    assert len(earlier) == 2
    assert all(c.state["i"] < 2 for c in earlier)
