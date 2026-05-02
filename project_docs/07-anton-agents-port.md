# 07 · AntonAgents 移植 SOP

> **阶段**：2.0-α (W4，部分) + 2.0-β（完整）
> **来源仓库**：`/Users/perlou/Desktop/personal/AntonAgents`（License: CC-BY-NC-SA-4.0，与 ClearAgent 一致）
> **关联文档**：00（整体决策）、05（eval 用 SFT/DPO 导出）

---

## 1. 背景

ClearAgent 作者另一仓库 `AntonAgents` 已实现 ~10K LOC 的 memory + protocols + rl 模块。代码审查后决定**选择性移植**，避免重复造轮子的同时不破坏 ClearAgent「轻量」定位。

详细决策表见 `/Users/perlou/.claude/plans/lucky-rolling-nebula.md` §「AntonAgents 资产复用决策」。

---

## 2. 移植清单

### 2.1 ✅ 移植（按阶段）

| 阶段 | 源文件 | 目标位置 | LOC | 备注 |
|---|---|---|---|---|
| **2.0-α (W4)** | `anton_agents/memory/embedding.py` | `clear_agent/retrieval/embeddings.py` | 349 | Embedding 抽象 + OpenAI/Dashscope 实现 |
| **2.0-α (W4)** | `anton_agents/memory/storage/document_store.py` | `clear_agent/retrieval/storage/document_store.py` | 481 | SQLiteDocumentStore |
| **2.0-α (W4)** | `anton_agents/memory/base.py` | `clear_agent/memory/base.py` | 182 | MemoryItem / MemoryConfig / BaseMemory |
| **2.0-β** | `anton_agents/memory/storage/qdrant_store.py` | `clear_agent/retrieval/storage/qdrant_store.py` | 571 | QdrantVectorStore |
| **2.0-β** | `anton_agents/memory/rag/pipeline.py` | `clear_agent/retrieval/rag/pipeline.py` | 1380 | 完整 RAG pipeline |
| **2.0-β** | `anton_agents/memory/rag/document.py` | `clear_agent/retrieval/rag/document.py` | 277 | Document / Chunker |
| **2.0-β** | `anton_agents/memory/types/working.py` | `clear_agent/memory/working.py` | 426 | WorkingMemory |
| **2.0-β** | `anton_agents/memory/types/semantic.py` | `clear_agent/memory/semantic.py` | 1238 | SemanticMemory |

**移植总量**：~5000 LOC（不含测试）

### 2.2 ⚠️ 重写（不直接移植）

| 模块 | 原因 | 行动 |
|---|---|---|
| `anton_agents/memory/manager.py` | 源文件 0 字节但被 `__init__.py` import，明显从未跑通 | 2.0-β 重写为 `clear_agent/memory/manager.py`，集成 Working + Semantic 两类记忆 |
| Multi-agent 协议层 | AntonAgents 的 A2A 是自定义协议，非业界标准 | 2.0-RC 基于 ClearAgent StateGraph 重新设计 supervisor/swarm/handoff |

### 2.3 ❌ 不移植

| 模块 | 原因 |
|---|---|
| `anton_agents/memory/types/episodic.py` (632 LOC) | 4 类记忆是 over-engineering，等用户需求 |
| `anton_agents/memory/types/perceptual.py` (778 LOC) | 多模态记忆需求未达临界点 |
| `anton_agents/memory/storage/neo4j_store.py` (467 LOC) | Qdrant 已覆盖 90% 检索场景，图记忆学习曲线陡 |
| `anton_agents/protocols/a2a/` (494 LOC) | 自定义协议成孤岛；改用 graph 原生 multi-agent |
| `anton_agents/protocols/anp/` (413 LOC) | 服务发现场景太窄 |
| `anton_agents/protocols/mcp/` (800 LOC) | **改用官方 `mcp` SDK** 而非 AntonAgents 的 fastmcp 包装 |
| `anton_agents/rl/` (1300 LOC) | 5GB+ 重依赖；改为 `TraceLogger.export_to_sft_jsonl()` |

---

## 3. 移植 SOP（每个文件必须遵守）

### 步骤 1：复制 + 改 namespace

```bash
cp anton_agents/memory/embedding.py clear_agent/retrieval/embeddings.py
```

`sed` 替换：
```
anton_agents.       → clear_agent.
from .. import      → from clear_agent.* import
AntonAgentsException → ClearAgentException
```

### 步骤 2：并入 Config

AntonAgents 各模块自带局部 config（如 `MemoryConfig`）。移植后**所有配置项必须并入** `clear_agent/core/config.py`：

```python
# 旧（AntonAgents）
class MemoryConfig(BaseModel):
    storage_path: str = "./memory_data"
    working_memory_capacity: int = 10
    ...

# 新（ClearAgent）→ 字段并入 Config，加 memory_ 前缀
class Config(BaseModel):
    # ...
    memory_storage_path: str = "memory/data"          # 沿用 memory/ 一级目录
    memory_working_capacity: int = 10
    memory_working_tokens: int = 2000
    memory_working_ttl_minutes: int = 120
    memory_importance_threshold: float = 0.1
    memory_decay_factor: float = 0.95
    # ...
```

**例外**：纯运行时小配置（如 `Embedding.batch_size`）可保留为构造参数，不一定进 Config。

### 步骤 3：替换异常类

```diff
- raise AntonAgentsException(...)
+ raise ClearAgentException(...)
```

新增子类（如 `MemoryError`、`RetrievalError`）必须继承 `ClearAgentException`。

### 步骤 4：替换日志

AntonAgents 用 `logger = logging.getLogger(__name__)`，ClearAgent 也用同样模式 → 直接保留即可，但确保模块名前缀变成 `clear_agent.*`。

### 步骤 5：替换打印（如有）

AntonAgents 部分代码用 `print(...)` 输出 emoji 状态，ClearAgent 风格保留这种 emoji 打印，但**统一前缀**：
- `🔍`：检索操作
- `💾`：持久化操作
- `🧠`：记忆操作
- `✅`：成功
- `❌`：失败

### 步骤 6：补测试

每个移植文件必须新增对应 `tests/test_*.py`：

| 移植文件 | 测试文件 |
|---|---|
| `retrieval/embeddings.py` | `tests/test_embeddings.py` |
| `retrieval/storage/document_store.py` | `tests/test_document_store.py` |
| `retrieval/storage/qdrant_store.py` | `tests/test_qdrant_store.py`（mock Qdrant 或装真实 Qdrant 测试） |
| `retrieval/rag/pipeline.py` | `tests/test_rag_pipeline.py` |
| `memory/working.py` | `tests/test_working_memory.py` |
| `memory/semantic.py` | `tests/test_semantic_memory.py` |

**最少覆盖**：
- 构造 + 默认配置
- 主接口的 happy path（add / retrieve / search / update / delete）
- 至少 1 个边缘 case（空查询、不存在 ID、并发写入）

### 步骤 7：optional dependency 标记

`pyproject.toml`：

```toml
[project.optional-dependencies]
memory = [
    "qdrant-client>=1.9.0",
    "scikit-learn>=1.0.0",
    "spacy>=3.4.0",
    # 不要把 sentence-transformers / torch 进 memory，它们更适合放 rag
]

rag = [
    "sentence-transformers>=2.2.0",
    "transformers>=4.20.0",
    "torch>=1.12.0",
    "markitdown>=0.0.1",
    "pypdf>=3.9.0",
]
```

模块 import 时按 1.x 模式 try/except 包装：

```python
try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

class QdrantVectorStore:
    def __init__(self, ...):
        if not QDRANT_AVAILABLE:
            raise ImportError("Install: pip install clear-agent[memory]")
```

### 步骤 8：License 与作者标注

每个移植文件**头部 docstring 加注**：

```python
"""SemanticMemory implementation.

Ported from AntonAgents (CC-BY-NC-SA-4.0)
Original: anton_agents/memory/types/semantic.py
"""
```

不需要重写实现；保留作者归属，避免抄袭嫌疑。

---

## 4. 可行性 spike（W4 启动前必做）

**目标**：在花一周大规模移植前，先用半天-一天验证「移植路径走得通」。

### 4.1 Spike 内容

1. 移植 `embedding.py` 到 `clear_agent/retrieval/embeddings.py`，按 §3 全套 SOP 走一遍
2. 移植 `document_store.py`（SQLite）到 `clear_agent/retrieval/storage/document_store.py`
3. 写 `tests/test_embeddings.py` 和 `tests/test_document_store.py`
4. 写 demo 脚本 `examples/rag_hello_world.py`：
   ```python
   emb = OpenAIEmbeddings()
   store = SQLiteDocumentStore("test.db")
   store.add(["doc1 content", "doc2 content"], embeddings=emb)
   results = store.search("query", k=2)
   print(results)
   ```
5. 跑通

### 4.2 Spike 出口

| 通过 | 行动 |
|---|---|
| ✅ Spike 1 天内跑通 | 进入 W4 完整移植 + 2.0-β 大规模移植 |
| ⚠️ Spike 用了 2-3 天 | 调整 2.0-β 排期，可能从 3 周拉到 4 周 |
| ❌ Spike 1 周还没跑通 | **重大警报**：放弃移植路径，2.0-β 改回自研 RAG（参考 LangChain 接口） |

---

## 5. MCP 协议引入（独立路径，不走 AntonAgents 移植）

按决策表，MCP 用**官方 `mcp` SDK**：

```toml
[project.optional-dependencies]
mcp = [
    "mcp>=1.0.0",            # Anthropic 官方 MCP SDK
]
```

新增模块（**自研**，不抄 AntonAgents）：

```
clear_agent/mcp/
├── __init__.py
├── client.py        # MCPClient: 让 ClearAgent 接入外部 MCP server
├── server.py        # MCPServer: 把 ClearAgent agent 暴露为 MCP server
└── adapter.py       # MCP Tool ↔ ClearAgent Tool 双向转换
```

**目标 API**（2.0-β）：

```python
# ClearAgent 作为客户端：吃外部 MCP 工具
from clear_agent.mcp import MCPClient

mcp = MCPClient.connect("stdio://path/to/mcp-server")
tools = mcp.list_tools()       # 返回 ClearAgent Tool 列表
registry.register_tools(tools)  # 直接注册到现有 ToolRegistry

# ClearAgent 作为服务端：把自己暴露给 Cursor/Claude Desktop
from clear_agent.mcp import MCPServer

server = MCPServer(agent=my_agent)
server.run(transport="stdio")
```

---

## 6. 移植后的目录结构

```
clear_agent/
├── ... (1.x 原有)
├── memory/                         # 新增（2.0-β）
│   ├── __init__.py
│   ├── base.py                     # ← 移植
│   ├── working.py                  # ← 移植
│   ├── semantic.py                 # ← 移植
│   └── manager.py                  # ⚠️ 重写
├── retrieval/                      # 新增（2.0-α 起步）
│   ├── __init__.py
│   ├── embeddings.py               # ← 移植 (W4)
│   ├── storage/
│   │   ├── document_store.py       # ← 移植 (W4)
│   │   └── qdrant_store.py         # ← 移植 (β)
│   └── rag/
│       ├── pipeline.py             # ← 移植 (β)
│       └── document.py             # ← 移植 (β)
└── mcp/                            # 新增（2.0-β，自研）
    ├── client.py
    ├── server.py
    └── adapter.py
```

---

## 7. License 与版权声明

ClearAgent 与 AntonAgents 都是 `CC-BY-NC-SA-4.0`，作者均为 Perlou，**移植无版权问题**。但仍按 §3 步骤 8 的格式在每个移植文件头部标注来源，便于未来审计与回溯。

---

## 8. 风险

| 风险 | 概率 | 兜底 |
|---|---|---|
| AntonAgents 代码与 ClearAgent 风格冲突大 | 高 | §3 SOP 强制走一遍；不允许直接 copy-paste |
| RAG pipeline (1380 LOC) 在新环境跑不通 | 高 | §4 spike 一天验证；不通则改回自研 |
| Qdrant client API 在新版本中漂移 | 中 | pin 版本到 `qdrant-client>=1.9.0,<2.0.0`；写适配层 |
| MCP 官方 SDK API 漂移 | 中 | pin 到具体小版本；提供 `clear-agent[mcp]` optional 隔离 |
| 移植测试覆盖不够 | 中 | 每个移植文件新增至少 5 个测试用例 |
| `MemoryManager` 重写复杂度低估 | 中 | 2.0-β 启动前先画 30 分钟设计草图，对齐 Working+Semantic 接口 |

---

## 9. 待决问题

1. **是否在 ClearAgent README 中显式声明「memory 模块移植自 AntonAgents」？**
   - 推荐：是（透明 + 引流到另一仓库）

2. **AntonAgents 是否最终归档？**
   - 不归档（保持独立仓库，作为 ClearAgent 的实验沙盒）
   - ClearAgent 是「主线产品」，AntonAgents 是「教学/实验」定位

3. **移植的代码是否反向同步 AntonAgents？**
   - **不反向同步**（ClearAgent 在 SOP 后已与 AntonAgents 漂移；维护两个分叉只会累死自己）

请逐项确认。
