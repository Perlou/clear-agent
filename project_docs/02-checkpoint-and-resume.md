# 02 · Checkpointer 与 Resume 设计

> **阶段**：2.0-α / W2
> **目标文件**：`clear_agent/core/checkpoint.py`
> **关联文档**：01（图执行模型，每节点结束写 checkpoint）、03（HITL 中断点写 checkpoint）

---

## 1. 设计目标

让 ClearAgent 2.0 拥有**真正的可恢复执行**：每个节点完成后落盘，崩溃后能从最后成功节点继续；失败工具能在修复后从该步重跑；HITL 暂停时把 state 压入持久层，等用户回来后无缝续跑。

**核心要求**：
- per-node 自动 checkpoint（CompiledGraph 内部触发，用户透明）
- thread_id 隔离不同会话/用户
- 三种存储后端：内存（开发）、JSON 文件（兼容现有 `memory/sessions/`）、SQLite（生产）
- 与现有 `SessionStore` 平滑过渡（旧文件可读，新文件双向兼容）

**非目标**：
- 分布式锁 / 多进程并发写入 → 不做（单进程足够）
- Postgres 后端 → 推迟到 2.0-β（用户呼声看上线后）

---

## 2. 核心数据模型

### 2.1 Checkpoint

```python
@dataclass
class Checkpoint:
    """单个 super-step 边界的 state 快照"""
    id: str                      # 唯一 ID（uuid7 时间排序友好）
    thread_id: str
    parent_id: str | None        # 用于时间旅行树
    state: dict[str, Any]        # 序列化后的 State 字段
    next_nodes: list[str]        # 即将执行的节点（resume 用）
    created_at: datetime
    metadata: dict[str, Any]     # source: "loop" | "interrupt" | "error" | "user_save"
```

### 2.2 BaseCheckpointer 协议

```python
class BaseCheckpointer(ABC):
    @abstractmethod
    def put(self, checkpoint: Checkpoint) -> None: ...

    @abstractmethod
    def get_tuple(
        self, thread_id: str, checkpoint_id: str | None = None
    ) -> Checkpoint | None: ...
    """checkpoint_id=None 时返回该 thread 最新的"""

    @abstractmethod
    def list(
        self,
        thread_id: str,
        before: str | None = None,
        limit: int = 50,
    ) -> list[Checkpoint]: ...

    # 异步对偶
    async def aput(self, checkpoint: Checkpoint) -> None: ...
    async def aget_tuple(self, thread_id: str, checkpoint_id: str | None = None) -> Checkpoint | None: ...
    async def alist(self, thread_id: str, before: str | None = None, limit: int = 50) -> list[Checkpoint]: ...
```

**默认异步实现**：`async def a*` 默认 `loop.run_in_executor` 包装同步版本，子类可覆写为真异步。

---

## 3. 三种实现

### 3.1 InMemoryCheckpointer（开发默认）

- 存 `dict[(thread_id, checkpoint_id) → Checkpoint]`
- `list()` 按 `created_at` 倒序
- 进程退出即丢

### 3.2 JsonFileCheckpointer（兼容现有 SessionStore）

- 目录布局：
  ```
  memory/checkpoints/<thread_id>/<checkpoint_id>.json
  memory/checkpoints/<thread_id>/_index.jsonl    # 倒序追加，加速 list()
  ```
- 写入用 `os.replace`（**直接复用 `SessionStore.save` 的原子写入逻辑**）
- 兼容性：能读取 1.x 的 `memory/sessions/session-*.json`，自动转换为 single-checkpoint thread
- 反向：2.x 写入的 checkpoint 也能被旧 `SessionStore.list_sessions()` 看到（同 schema 子集）

**关键代码骨架**：
```python
class JsonFileCheckpointer(BaseCheckpointer):
    def __init__(self, base_dir: str = "memory/checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def put(self, ckpt: Checkpoint) -> None:
        thread_dir = self.base_dir / ckpt.thread_id
        thread_dir.mkdir(exist_ok=True)
        filepath = thread_dir / f"{ckpt.id}.json"
        tmp = filepath.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(ckpt), default=str, ensure_ascii=False))
        os.replace(tmp, filepath)
        # 追加索引
        idx = thread_dir / "_index.jsonl"
        with idx.open("a") as f:
            f.write(json.dumps({"id": ckpt.id, "created_at": str(ckpt.created_at)}) + "\n")
```

### 3.3 SqliteCheckpointer（生产）

- 单文件 `.db`，schema：
  ```sql
  CREATE TABLE checkpoints (
      id TEXT PRIMARY KEY,
      thread_id TEXT NOT NULL,
      parent_id TEXT,
      state BLOB NOT NULL,        -- json bytes
      next_nodes TEXT NOT NULL,   -- json
      created_at TEXT NOT NULL,
      metadata TEXT NOT NULL      -- json
  );
  CREATE INDEX idx_thread_created ON checkpoints(thread_id, created_at DESC);
  ```
- 用 `sqlite3` 标准库，不引入新依赖
- WAL 模式 + `PRAGMA synchronous=NORMAL` 平衡性能与持久性
- `aput` 实现真异步：`aiosqlite` optional 依赖；不装则降级到线程池

---

## 4. 执行流集成

CompiledGraph 在每个节点完成后自动调用：

```python
def _execute_loop(self, state, config: RunConfig):
    while next_node != END:
        new_state = self._run_node(next_node, state)
        state = self._merge(state, new_state)

        # ↓ 自动 checkpoint
        if self.checkpointer:
            ckpt = Checkpoint(
                id=str(uuid7()),
                thread_id=config.thread_id,
                parent_id=self._last_ckpt_id,
                state=self._serialize(state),
                next_nodes=[self._next_node(state)],
                created_at=datetime.now(),
                metadata={"source": "loop", "node": next_node},
            )
            self.checkpointer.put(ckpt)
            self._last_ckpt_id = ckpt.id

        next_node = self._route(state)
    return state
```

---

## 5. Resume 语义

```python
# 场景 1: 崩溃后续跑
result = compiled.invoke({"messages": [...]}, config=RunConfig(thread_id="t1"))
# ... process killed ...
result = compiled.resume(thread_id="t1")  # 从最后成功 checkpoint 继续

# 场景 2: 时间旅行（rewind）
ckpts = compiled.list_checkpoints("t1")
target = ckpts[3]
result = compiled.resume(thread_id="t1", checkpoint_id=target.id)

# 场景 3: 修改 state 后续跑（time travel + edit）
result = compiled.resume(
    thread_id="t1",
    checkpoint_id=target.id,
    state_patch={"messages": [...modified...]},
)
```

**Resume 算法**：
```
1. checkpointer.get_tuple(thread_id, checkpoint_id) → ckpt
2. state = ckpt.state（反序列化）
3. 如果有 state_patch，按 reducer 合并到 state
4. next_node = ckpt.next_nodes[0]
5. 进入主循环（同 invoke）
```

---

## 6. 配置项

`Config` 新增字段（沿用现有 `session_*` 命名风格）：

```python
# 旧字段（保留向后兼容，但不再推荐）
session_enabled: bool = True
session_dir: str = "memory/sessions"
auto_save_enabled: bool = False
auto_save_interval: int = 10

# 新字段
checkpoint_enabled: bool = True
checkpoint_backend: Literal["memory", "json", "sqlite"] = "json"
checkpoint_dir: str = "memory/checkpoints"     # json backend
checkpoint_db_path: str = "memory/checkpoints.db"  # sqlite backend
checkpoint_keep_last_n: int = 100              # 同 thread 自动清理
```

---

## 7. 与 SessionStore 的兼容性

| 维度 | 兼容性 |
|---|---|
| 读 1.x session 文件 | ✅ JsonFileCheckpointer 自动识别旧 schema，转为单 checkpoint thread |
| 1.x 用户调 `agent.save_session()` | ✅ 仍可用，内部映射到 `checkpointer.put(metadata={"source": "user_save"})` |
| 1.x `SessionStore.list_sessions()` | ✅ 保留，返回的 `filename` 字段对所有 thread 列出最新 checkpoint |
| 1.x `agent.load_session(filepath)` | ✅ 保留，内部包装为 `compiled.resume(thread_id, checkpoint_id)` |

---

## 8. 测试清单（W2 出口）

`tests/test_checkpoint_roundtrip.py`：

| # | 测试 | 通过标准 |
|---|---|---|
| 1 | 5 步 → kill → resume → 完成 | 最终 state 与一气呵成等价（去除时间戳） |
| 2 | 三种后端等价 | 同输入下 InMemory / Json / Sqlite 结果一致 |
| 3 | thread_id 隔离 | t1 和 t2 互不影响 |
| 4 | 时间旅行 rewind | 从 ckpt #3 resume 后，不读取 ckpt #4 #5 |
| 5 | state_patch | resume 时注入修改，下游节点看到修改后的值 |
| 6 | 1.x 会话兼容 | 加载现有 `memory/sessions/session-*.json` 不抛错 |
| 7 | 原子写入 | mock `os.replace` 在写入中崩溃，重启后能读到上一个完整 ckpt |
| 8 | 异步并发 | aput 高并发不丢 ckpt（仅 InMemory + Sqlite） |

---

## 9. 待决问题

1. **State 序列化用 json 还是 pickle？**
   - 推荐 json：可读、跨语言、与现有 SessionStore 一致
   - pydantic 模型 → `model_dump()`；TypedDict → `default=str` 兜底
   - **建议**：json 优先；不可序列化字段（如 LLM client 句柄）禁止入 State

2. **`checkpoint_keep_last_n` 默认多大？**
   - 太小会丢时间旅行能力；太大会撑爆磁盘
   - **建议**：100（够用 + 单 thread 最多 ~100MB）

3. **Sqlite WAL 是否默认开？**
   - WAL 在多进程下要小心（不同进程对同一 db 的可见性）
   - **建议**：默认开（单进程场景）；多进程时文档提示

4. **要不要支持「跨 thread 共享 state」（即 LangGraph Store）？**
   - 长期记忆跨会话 → 是 P1 需求，但属于 memory 范畴
   - **建议**：不在 checkpoint 内做；2.0-β 通过移植 AntonAgents `SemanticMemory` 解决

请逐项确认或调整。
