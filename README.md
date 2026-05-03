# ClearAgent

> 🤖 生产级多智能体框架 —— 基于 OpenAI 原生 API，**v2.0 引入 StateGraph + Checkpoint + Human-in-the-Loop + 完整 RAG + 多层 Memory**，集成上下文工程、子代理、Skills、结构化输出、Eval-harness 等 25+ 核心能力。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-2.0.0-brightgreen.svg)]()
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## ✨ 核心特性

### 🆕 2.0-β 新增（RAG + Memory）

- **完整 RAG Pipeline** —— 7 大职责（加载/分块/索引/检索/重排/合并/压缩）+ MQE / HyDE 查询扩展 + 引用合并；支持 50+ 文档格式（PDF/DOCX/XLSX/图像 OCR/音频转写）；移植自 AntonAgents（CC-BY-NC-SA-4.0）
- **多层 Memory 体系** —— `WorkingMemory`（短期，TF-IDF + TTL + 优先级堆）+ `SemanticMemory`（长期，向量 + 内存知识图谱）+ `MemoryManager`（多子系统协调）
- **向量库** —— `QdrantVectorStore`（云/本地双连接 + HNSW 调优 + payload 索引）+ `SQLiteDocumentStore`
- **嵌入抽象** —— `LocalTransformerEmbedding` / `DashScopeEmbedding`（OpenAI 兼容 REST）/ `TFIDFEmbedding` 兜底 + 工厂回退

### 2.0-α 新增（图编排时代）

- **StateGraph 抽象**：声明式图构建（节点 / 边 / 条件边 / 字段级 reducer），4 种内置 Agent 全部跑在 graph 上
- **Checkpointer**：每节点自动快照，支持 `InMemory` / `JsonFile` / `Sqlite` 三种后端，**kill 进程也能 resume**
- **Human-in-the-Loop**：节点内调 `interrupt(payload)` 暂停 → 外部 `compiled.resume(thread_id, value=...)` 注入决策续跑
- **结构化输出**：`llm.with_structured_output(MyPydanticModel)` 一行打通三种 method
- **Eval-harness**：Dataset / 4 种 Evaluator（含 `LLMAsJudge`）/ 并发 Runner / Markdown 报告

### 1.x 已有（继续保留，100% 向后兼容）

- **4 种 Agent 范式**：`SimpleAgent` · `ReActAgent` · `ReflectionAgent` · `PlanSolveAgent`
- **统一 LLM 接口**：自动适配 OpenAI 兼容（DeepSeek/Qwen/Kimi/Ollama 等）、Anthropic、Gemini，支持同步 / 异步 / 流式 / Function Calling
- **上下文工程**：GSSC 流水线、历史压缩、工具输出截断、Token 增量计数
- **工具响应协议**：`ToolResponse` 统一返回，配套熔断器、权限过滤、文件编辑乐观锁
- **子代理机制**：`TaskTool` 派发隔离子任务，工具权限可精确裁剪
- **Skills 知识外化**：按需加载 `SKILL.md`，预期节省 ~85% Token
- **工程化套件**：会话持久化、TodoWrite、DevLog、TraceLogger（JSONL+HTML）、异步生命周期钩子、SSE 流式

## 🚀 快速开始

```bash
# 最小安装
pip install -e .

# RAG 完整套件（β）
pip install -e ".[retrieval,retrieval-qdrant,rag]"

# Memory 完整套件（β）
pip install -e ".[memory,retrieval-qdrant]"
python -m spacy download zh_core_web_sm    # 可选：中文 NER

# 多 provider
pip install -e ".[anthropic,gemini,dashscope]"

# 配置环境变量
cp .env.example .env       # 填入 LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL
```

### 1.x 风格（向后兼容）

```python
from clear_agent import ClearAgentLLM, ReActAgent, ToolRegistry, CalculatorTool

llm = ClearAgentLLM()
registry = ToolRegistry(); registry.register_tool(CalculatorTool())
agent = ReActAgent(name="demo", llm=llm, tool_registry=registry)
print(agent.run("计算 (123 + 456) * 2"))
```

### 2.0-α：StateGraph + Checkpoint

```python
from clear_agent import ClearAgentLLM, ReActAgent, SqliteCheckpointer
from clear_agent.core.graph import RunConfig

agent = ReActAgent(name="demo", llm=ClearAgentLLM())
graph = agent.as_graph(checkpointer=SqliteCheckpointer("memory/runs.db"))

result = graph.invoke(
    {"messages": [{"role": "user", "content": "..."}], "max_steps": 5},
    config=RunConfig(thread_id="thread-1"),
)
# kill 进程后任意时间 resume：
# graph.resume("thread-1") 继续从最后 checkpoint 跑下去
```

### 2.0-β：完整 RAG Pipeline

```python
from clear_agent.retrieval.rag import create_rag_pipeline

rag = create_rag_pipeline(qdrant_url="http://localhost:6333", rag_namespace="my_kb")

rag["add_documents"](["docs/a.pdf", "docs/b.md"])             # 自动分块 + 索引
hits = rag["search"]("如何配置 LLM？", top_k=5)                # 基础检索
hits = rag["search_advanced"]("...", enable_mqe=True, enable_hyde=True)  # 查询扩展
```

### 2.0-β：多层 Memory

```python
from clear_agent.memory import (
    MemoryManager, WorkingMemory, SemanticMemory, MemoryConfig, MemoryItem,
)

mgr = MemoryManager()
mgr.register("working", WorkingMemory(MemoryConfig()))
mgr.register("semantic", SemanticMemory(config=MemoryConfig()))

mgr.add(MemoryItem(id="m1", content="...", memory_type="working", ...))
hits = mgr.retrieve("query", limit=10)   # 跨子系统聚合
```

更多示例：[`examples/memory_demo.py`](examples/memory_demo.py) · [`examples/rag_hello_world.py`](examples/rag_hello_world.py) · [`examples/async_agent_demo.py`](examples/async_agent_demo.py)

## 📊 vs LangGraph + LangChain（对照表）

| 能力 | LangGraph/Chain | ClearAgent 2.0-β | 备注 |
|---|---|---|---|
| 声明式图（节点 / 边 / 条件边） | ✅ | ✅ | 字段级 reducer 内置 |
| Checkpointer | ✅ Memory/Sqlite/Postgres | ✅ Memory/JsonFile/Sqlite | Postgres 待 RC |
| Human-in-the-Loop | ✅ `interrupt()` | ✅ 行为对齐 + 同节点多次中断回放 | |
| 时间旅行（rewind） | ✅ | ✅ | |
| 结构化输出 | ✅ | ✅ 同名 API + 3 种 method | |
| **完整 RAG Pipeline** | ✅ 50+ vectorstore | ✅ Qdrant + 完整 7 段 + MQE/HyDE | β 移植自 AntonAgents |
| **多层 Memory** | ✅ BaseStore | ✅ Working + Semantic + Manager | β 新增 |
| **文档加载** | ✅ 100+ loaders | ✅ MarkItDown 50+ 格式 | β 新增 |
| **嵌入抽象** | ✅ Embeddings | ✅ Local/DashScope/TFIDF + 工厂 | β 新增 |
| **重排** | ✅ | ✅ Cross-encoder + 图信号融合 | β 新增 |
| Eval / Tracing | LangSmith | TraceLogger + Eval-harness | 不依赖外部服务 |
| Multi-agent supervisor | ✅ | ⚠️ TaskTool（RC 加 graph 原生） | |
| 工具并行 | ✅ | ⚠️ 顺序（待优化） | |
| Multimodal / Prompt caching | ✅ | ❌ 后续 | |
| 运行时核心依赖 | langchain-core 全家桶 | pydantic + tiktoken + pyyaml + networkx + numpy | 轻量 |

> 设计取舍：ClearAgent 2.0-β 在保留**轻量、单包、零重型依赖**定位的同时，把 LangGraph + LangChain 编排能力的 **~85%** 压进 ~7K LOC 自研代码（含 ~4470 LOC β 阶段移植自 AntonAgents）。

## 📦 项目结构

```
clear_agent/
├── core/                # Agent 基类 / LLM / Config
│   ├── graph.py             # StateGraph + reducers
│   ├── checkpoint.py        # Memory/JsonFile/Sqlite
│   ├── interrupt.py         # interrupt() + GraphPaused
│   └── structured.py        # with_structured_output
├── agents/              # 4 种范式 + graph builders
├── hitl/                # Approval / Edit / ToolValidation
├── eval/                # Dataset / Evaluator / Runner
├── retrieval/           # 🆕 β
│   ├── embeddings.py        # Local / DashScope / TFIDF + 全局单例
│   ├── rag/
│   │   ├── document.py          # Document / Chunker
│   │   └── pipeline.py          # 7 大职责 + create_rag_pipeline
│   └── storage/
│       ├── document_store.py    # SQLite
│       └── qdrant_store.py      # Qdrant + ConnectionManager
├── memory/              # 🆕 β
│   ├── base.py              # MemoryItem/Config/BaseMemory
│   ├── working.py           # 短期 + TF-IDF
│   ├── semantic.py          # 长期 + 向量 + 内存图谱（移除 Neo4j）
│   └── manager.py           # 多子系统协调（重写）
├── context/             # GSSC 流水线
├── tools/               # 工具系统
├── observability/       # TraceLogger
└── skills/              # SkillLoader

project_docs/            # 9 篇设计 spec（00-08）
docs/                    # 18+ 篇专项指南
tests/                   # 500+ pytest 测试
```

## 📚 文档

- **新用户**：[`README.md`](README.md) → [`CLAUDE.md`](CLAUDE.md) → [`docs/graph-architecture.md`](docs/graph-architecture.md)
- **2.0-β 用户向**：[`docs/rag-guide.md`](docs/rag-guide.md) · [`docs/memory-guide.md`](docs/memory-guide.md)
- **2.0-α 用户向**：[`docs/graph-architecture.md`](docs/graph-architecture.md) · [`docs/structured-output.md`](docs/structured-output.md) · [`docs/eval-harness.md`](docs/eval-harness.md) · [`docs/migration-1.x-to-2.x.md`](docs/migration-1.x-to-2.x.md)
- **2.0 设计 spec**：[`project_docs/00-overview.md`](project_docs/00-overview.md) … `08-rag-memory-integration.md`
- **1.x 专项**：上下文工程、子代理、Skills、TodoWrite、DevLog、SSE、异步等位于 [`docs/`](docs/)

## 🛠️ 开发

```bash
pytest                                                                # 500+ 测试
pytest tests/test_rag_pipeline.py tests/test_semantic_memory.py -q    # β 专项
black clear_agent tests
isort clear_agent tests
mypy clear_agent
```

## 🗺️ 路线图

- **2.0-α**（已发版 ✅）：StateGraph + Checkpoint + HITL + 结构化输出 + Eval-harness + Retrieval spike
- **2.0-β**（已发版 ✅）：完整 RAG pipeline、Memory 体系（Working/Semantic）、MemoryManager 重写、QdrantVectorStore、嵌入抽象
- **2.0-RC**（已发版 ✅）：Multi-agent 范式包（supervisor/swarm/handoff）、MCP 协议（client/server）、Callbacks 协议（13 hooks）、工具并行、真异步 OpenAI 客户端、LCEL-lite Runnable + `|` 管道、TraceLogger SFT/DPO 训练数据导出
- **2.0 GA**（**当前** ✅）：Resilience（Retry/Fallback/负载均衡）、Pydantic 自动 Tool schema 推导、Anthropic+Gemini 真异步路径、Multimodal（vision/audio）+ Prompt caching helpers
- **2.1+**：Skill marketplace（远程加载 + 版本管理）、更多 provider 集成、性能优化迭代

详见 [`project_docs/00-overview.md`](project_docs/00-overview.md) 与 [`project_docs/08-rag-memory-integration.md`](project_docs/08-rag-memory-integration.md)。

## 📄 License

[CC BY-NC-SA 4.0](LICENSE) —— 允许学习/研究/分享，**禁止商业使用**。商用请联系作者 `perloukevin@gmail.com`。

memory + RAG 模块**移植自姊妹项目 [AntonAgents](https://github.com/Perlou/AntonAgents)**（同作者，同 License）。详见 [`project_docs/07-anton-agents-port.md`](project_docs/07-anton-agents-port.md)。

## ✨ 核心特性

### 🆕 2.0 新增（图编排时代）

- **StateGraph 抽象**：声明式图构建（节点 / 边 / 条件边 / 字段级 reducer），4 种内置 Agent 全部跑在 graph 上
- **Checkpointer**：每节点自动快照，支持 `InMemory` / `JsonFile` / `Sqlite` 三种后端，**kill 进程也能 resume**
- **Human-in-the-Loop**：节点内调 `interrupt(payload)` 暂停 → 外部 `compiled.resume(thread_id, value=...)` 注入决策续跑
- **结构化输出**：`llm.with_structured_output(MyPydanticModel)` 一行打通三种 method（`function_calling` / `json_mode` / `json_schema`），失败自动重试
- **Eval-harness**：Dataset / 4 种 Evaluator（含 `LLMAsJudge`）/ 并发 Runner / Markdown 报告
- **Retrieval 起步**：`SQLiteDocumentStore` + `EmbeddingModel` 抽象（local / DashScope / TFIDF），为 2.0-β 完整 RAG 套件打底

### 1.x 已有（继续保留，100% 向后兼容）

- **4 种 Agent 范式**：`SimpleAgent` · `ReActAgent` · `ReflectionAgent` · `PlanSolveAgent`
- **统一 LLM 接口**：自动适配 OpenAI 兼容（DeepSeek/Qwen/Kimi/Ollama 等）、Anthropic、Gemini，支持同步 / 异步 / 流式 / Function Calling
- **上下文工程**：GSSC 流水线、历史压缩、工具输出截断、Token 增量计数
- **工具响应协议**：`ToolResponse` 统一返回，配套熔断器、权限过滤、文件编辑乐观锁
- **子代理机制**：`TaskTool` 派发隔离子任务，工具权限可精确裁剪
- **Skills 知识外化**：按需加载 `SKILL.md`，预期节省 ~85% Token
- **工程化套件**：会话持久化、TodoWrite、DevLog、TraceLogger（JSONL+HTML）、异步生命周期钩子、SSE 流式

## 🚀 快速开始

```bash
# 安装
pip install -e .                      # 或 uv sync
# 可选扩展
pip install -e ".[retrieval]"         # sklearn TFIDF
pip install -e ".[rag]"               # sentence-transformers + torch
pip install -e ".[anthropic,gemini]"  # 多 provider

# 配置环境变量
cp .env.example .env                  # 填入 LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL
```

### 1.x 风格（向后兼容）

```python
from clear_agent import ClearAgentLLM, ReActAgent, ToolRegistry, CalculatorTool

llm = ClearAgentLLM()
registry = ToolRegistry(); registry.register_tool(CalculatorTool())
agent = ReActAgent(name="demo", llm=llm, tool_registry=registry)
print(agent.run("计算 (123 + 456) * 2"))
```

### 2.0 风格（StateGraph + Checkpoint）

```python
from clear_agent import ClearAgentLLM, ReActAgent, SqliteCheckpointer
from clear_agent.core.graph import RunConfig

agent = ReActAgent(name="demo", llm=ClearAgentLLM())
graph = agent.as_graph(checkpointer=SqliteCheckpointer("memory/runs.db"))

# 跑 + 自动每节点 checkpoint
result = graph.invoke(
    {"messages": [{"role": "user", "content": "..."}], "max_steps": 5},
    config=RunConfig(thread_id="thread-1"),
)
# kill 进程后任意时间 resume：
# graph.resume("thread-1") 继续从最后 checkpoint 跑下去
```

### Human-in-the-Loop

```python
from clear_agent.core.interrupt import interrupt, GraphPaused

def risky_node(state):
    decision = interrupt({"type": "approval", "message": "Send email?", "draft": state["draft"]})
    if not decision.get("approved"):
        return {"messages": [...]}
    # 已获批，继续
    ...

try:
    graph.invoke(state, config=RunConfig(thread_id="t1"))
except GraphPaused as p:
    # 把 p.payload 展示给用户/前端，等用户决策
    graph.resume("t1", value={"approved": True})
```

### 结构化输出

```python
from pydantic import BaseModel
class Person(BaseModel):
    name: str
    age: int

structured = llm.with_structured_output(Person)
p = structured.invoke([{"role": "user", "content": "Alice 是 30 岁的老师"}])
print(p.name, p.age)  # Alice 30
```

### Eval-harness

```python
from clear_agent.eval import Dataset, ExactMatch, run_eval

ds = Dataset.from_jsonl("examples/eval/qa.jsonl")
report = run_eval(target=graph, dataset=ds, evaluator=ExactMatch(), parallel=4)
print(f"Pass rate: {report.pass_rate() * 100:.1f}%")
# 同时落盘 memory/eval/<run_id>/{report.md, results.jsonl}
```

异步 Agent 示例见 [`examples/async_agent_demo.py`](examples/async_agent_demo.py)；
RAG hello-world 见 [`examples/rag_hello_world.py`](examples/rag_hello_world.py)。

## 📊 vs LangGraph（对照表）

| 能力 | LangGraph | ClearAgent 2.0 | 备注 |
|---|---|---|---|
| 声明式图（节点 / 边 / 条件边） | ✅ | ✅ | 字段级 reducer 内置 `add_messages / append_list / merge_dict` |
| Checkpointer | ✅ Memory/Sqlite/Postgres | ✅ Memory/JsonFile/Sqlite | 2.0-α 不含 Postgres，2.0-β 起 |
| Human-in-the-Loop | ✅ `interrupt()` | ✅ `interrupt()` + 同节点多次中断顺序回放 | 行为对齐 |
| 时间旅行（rewind） | ✅ | ✅ | `checkpointer.list(thread_id)` + 回放 |
| 结构化输出 | ✅ `with_structured_output` | ✅ 同名 API | OpenAI 兼容 3 种 method |
| Eval / Tracing | LangSmith | TraceLogger + Eval-harness | 不依赖外部服务 |
| 流式 | ✅ | ✅ | `astream_invoke` 真异步 |
| Multi-agent 模式 | ✅ supervisor/swarm | ⚠️ TaskTool（2.0-RC 加 graph 原生 multi-agent） | |
| 工具并行 | ✅ | ⚠️ 顺序（2.0-β） | |
| RAG / 文档加载 | ✅ 50+ vectorstore | ⚠️ SQLite + Embedding 抽象（2.0-β 加 Qdrant + 完整 RAG） | |
| 运行时依赖（核心） | `langchain-core` 全家桶 | `pydantic` + `tiktoken` + `pyyaml` + `networkx` | 轻量 |
| Multimodal / Prompt caching | ✅ | ❌（2.0-β） | |

> 设计取舍：ClearAgent 2.0 保留**轻量、单包、零重型依赖**定位，把 LangGraph 编排能力的 80% 压进 ~2K LOC 自研代码。完整路线图见 [`project_docs/00-overview.md`](project_docs/00-overview.md)。

## 📦 项目结构

```
clear_agent/
├── core/             # Agent 基类 / LLM / Config / 生命周期
│   ├── graph.py            # 🆕 StateGraph + CompiledGraph + reducers
│   ├── checkpoint.py       # 🆕 BaseCheckpointer + Memory/JsonFile/Sqlite
│   ├── interrupt.py        # 🆕 interrupt() + GraphPaused
│   └── structured.py       # 🆕 StructuredLLM + with_structured_output
├── agents/           # 4 种范式 + 子代理工厂
│   ├── _react_graph.py        # 🆕 build_react_graph
│   ├── _simple_graph.py       # 🆕
│   ├── _reflection_graph.py   # 🆕
│   └── _plan_solve_graph.py   # 🆕
├── hitl/             # 🆕 HITL patterns: Approval / Edit / ToolValidation
├── eval/             # 🆕 Dataset / Evaluator / Runner
├── retrieval/        # 🆕 Embeddings + SQLiteDocumentStore (移植自 AntonAgents)
├── context/          # GSSC 流水线
├── tools/            # 工具系统 + 内置工具
├── observability/    # TraceLogger
└── skills/           # SkillLoader
project_docs/         # 🆕 8 篇 2.0 设计 spec（00-overview … 07-anton-port）
docs/                 # 16+ 篇专项指南（1.x + 2.0 quickstart）
tests/                # 460+ pytest 测试
```

## 📚 文档

- **新用户**：[`README.md`](README.md) → [`CLAUDE.md`](CLAUDE.md) → [`docs/graph-architecture.md`](docs/graph-architecture.md)
- **2.0 用户向**：[`docs/graph-architecture.md`](docs/graph-architecture.md) · [`docs/structured-output.md`](docs/structured-output.md) · [`docs/eval-harness.md`](docs/eval-harness.md) · [`docs/migration-1.x-to-2.x.md`](docs/migration-1.x-to-2.x.md)
- **2.0 设计 spec**：[`project_docs/00-overview.md`](project_docs/00-overview.md) … `07-anton-agents-port.md`
- **1.x 专项**：上下文工程、子代理、Skills、TodoWrite、DevLog、SSE、异步、可观测性等位于 [`docs/`](docs/)

## 🛠️ 开发

```bash
pytest                           # 测试（460+ 用例）
pytest tests/test_graph_basics.py tests/test_react_graph.py -q   # 仅 2.0 graph
black clear_agent tests          # 格式化
isort clear_agent tests
mypy clear_agent                 # 类型检查
```

## 🗺️ 路线图

- **2.0-α (当前)**：StateGraph + Checkpoint + HITL + 结构化输出 + Eval-harness + Retrieval spike ✅
- **2.0-β**：完整 RAG pipeline、Memory 体系（Working/Semantic）、MCP 协议、工具并行、真异步 OpenAI 客户端、Callbacks 协议、Anthropic/Gemini 适配器
- **2.0-RC**：基于 graph 原生的 multi-agent（supervisor/swarm/handoff）、LCEL-lite

详见 [`project_docs/00-overview.md`](project_docs/00-overview.md)。

## 📄 License

[CC BY-NC-SA 4.0](LICENSE) —— 允许学习/研究/分享，**禁止商业使用**。商用请联系作者 `perloukevin@gmail.com`。
