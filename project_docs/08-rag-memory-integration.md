# 08 · β 阶段 RAG + Memory 集成 spec

> **阶段**：2.0-β（W1-W4）
> **状态**：已完成 ✅
> **关联文档**：00（决策）、07（AntonAgents 移植 SOP）、`docs/rag-guide.md`、`docs/memory-guide.md`

---

## 1. β 阶段交付摘要

| 周 | 子任务 | 移植/新增 LOC | 测试数 |
|---|---|---|---|
| W1.1 | RAG Document + Chunker | ~190 (移植 277) | 28 |
| W1.2 | QdrantVectorStore | ~430 (移植 571) | 48 |
| W1.3 | RAG Pipeline (7 大职责) | ~1100 (移植 1380) | 78 |
| W2 | Memory 基础（base + Working） | ~440 (移植 608) | 52 |
| W3 | SemanticMemory + Manager（重写） | ~810 (移植 1238 + 重写 190) | 67 |
| W4 | 集成 + 文档 + 发版 | ~1500（demo + docs + pyproject） | – |
| **β 累计** | | **~4470 LOC** | **273** |
| 全量回归 | W1-βW3 graph + RAG + Memory | – | **500 passed** |

---

## 2. 模块结构

```
clear_agent/
├── retrieval/
│   ├── embeddings.py            # Local / DashScope / TFIDF + 工厂 + 全局单例
│   ├── rag/
│   │   ├── document.py          # Document / DocumentChunk / DocumentProcessor
│   │   └── pipeline.py          # 7 大职责 + create_rag_pipeline 工厂
│   └── storage/
│       ├── document_store.py    # SQLiteDocumentStore（α 已交付）
│       └── qdrant_store.py      # QdrantVectorStore + ConnectionManager
├── memory/
│   ├── base.py                  # MemoryItem / MemoryConfig / BaseMemory
│   ├── working.py               # WorkingMemory（短期，纯内存）
│   ├── semantic.py              # SemanticMemory + Entity + Relation（移除 Neo4j）
│   └── manager.py               # MemoryManager（重写）
└── ...
```

## 3. 关键设计决策

### 3.1 不引入 Neo4j —— SemanticMemory 改造

AntonAgents 的 `SemanticMemory` 1238 LOC 中约 30% 是 Neo4j 集成代码（`_add_entity_to_graph`、`_add_relation_to_graph`、`_calculate_graph_relevance_neo4j`、`_store_linguistic_analysis`、`get_related_entities` 走 Neo4j）。

按 plan §07 §2.3 决策**不引入 Neo4j**，本期改造为：
- 内存图谱：`self.entities: Dict[str, Entity]` + `self.relations: List[Relation]`
- `_graph_search`：基于内存中的实体重叠 + 实体密度 + 关系密度计算 graph_score（公式：`entity_score * 0.6 + entity_density * 0.2 + relation_density * 0.2`）
- `get_related_entities`：基于内存图的 BFS 遍历（max_hops 跳）

**取舍**：
- ✅ 减少一个外部依赖（Neo4j 服务 + neo4j-driver 包）
- ✅ 单进程内 BFS 速度比 Cypher 查询快
- ❌ 图谱不持久化（重启即失）—— 持久化的是 Qdrant 向量与原始 MemoryItem，重启可重新提取
- ❌ 复杂图查询能力有限（不支持 Cypher / 多跳 + 边类型过滤组合）

未来若需要持久化图谱，可在 `clear_agent.memory.graph_store` 下接入 Neo4j / FalkorDB / NetworkX 持久化层，作为 SemanticMemory 的可选 backend。

### 3.2 MemoryManager 重写（AntonAgents 0 字节）

AntonAgents 的 `manager.py` 源文件 0 字节但被 `__init__.py` import —— 说明从未跑通。本期 ClearAgent 自研重写：
- 注册式：`mgr.register("working", WorkingMemory(...))` —— 不假设具体子系统类型
- 路由：`add(item)` 按 `memory_type` 自动路由；`update/remove` 可显式指定 `memory_type` 或自动遍历查找
- 聚合检索：跨子系统并集 + 按 `importance` 倒序去重合并
- 统计聚合：`{total_count, by_type, registered_types}`
- 错误隔离：单个子系统 retrieve / get_stats 抛错不影响其他子系统

### 3.3 Optional dependency 矩阵

| 包 | optional group | 用途 |
|---|---|---|
| `qdrant-client` | `[retrieval-qdrant]` | QdrantVectorStore + SemanticMemory 的向量后端 |
| `scikit-learn` | `[retrieval]` / `[memory]` | TFIDFEmbedding + WorkingMemory.retrieve TF-IDF |
| `markitdown` | `[rag]` | RAG 通用文档加载（PDF/DOCX/...） |
| `langdetect` | `[rag]` | RAG 语言检测 |
| `sentence-transformers` | `[rag]` | LocalTransformerEmbedding + cross-encoder 重排 |
| `transformers` + `torch` | `[rag]` | sentence-transformers 上游 |
| `spacy` | `[memory]` | SemanticMemory NER（fallback 走简单分词） |
| `dashscope` | `[dashscope]` | DashScope SDK 模式（REST 模式不需要） |
| `neo4j` | — | **不引入** |

**安装路径建议**：
```bash
# 最小集（α）
pip install clear-agent

# 加 RAG 起步（Qdrant + sklearn）
pip install clear-agent[retrieval,retrieval-qdrant]

# 完整 RAG（含 markitdown / 重排 / langdetect）
pip install clear-agent[retrieval,retrieval-qdrant,rag]

# 完整 Memory（含 spacy NER）
pip install clear-agent[memory,retrieval-qdrant]
python -m spacy download zh_core_web_sm
python -m spacy download en_core_web_sm

# 全套
pip install clear-agent[retrieval,retrieval-qdrant,rag,memory,dashscope]
```

## 4. 集成模式

### 4.1 RAG 节点

把 RAG 检索当作 graph 中的一个节点，注入 context 到 state：

```python
def retrieve_node(state):
    hits = rag["search"](state["question"], top_k=5)
    return {"context": "\n\n".join(h["metadata"]["content"] for h in hits)}
```

### 4.2 Memory-aware Agent

让 agent 在每轮对话前先 retrieve 相关历史，回答后写入 memory：

```python
def memory_aware_node(state):
    history = mgr.retrieve(state["question"], limit=5)
    response = llm.invoke(prompt_with_history(state["question"], history))
    mgr.add(MemoryItem(content=f"Q: {state['question']}\nA: {response.content}", ...))
    return {"answer": response.content}
```

详见 `examples/memory_demo.py`。

### 4.3 多租户隔离

- RAG：`rag_namespace` payload 字段
- Memory：`MemoryItem.user_id` 字段 + `retrieve(..., user_id=...)` 过滤

两层都用 Qdrant payload 索引（已在 `_ensure_payload_indexes` 中创建）。

## 5. 测试覆盖（β 阶段共 273 测试）

| 测试文件 | 用例数 | 主要覆盖 |
|---|---|---|
| `test_rag_document.py` | 28 | Document/Chunk MD5 ID + 分块边界 + 后处理 + dedup |
| `test_qdrant_store.py` | 48 | 连接（云/本地/url-only）+ add/search/delete + ConnectionManager 单例 + 异常容错 |
| `test_rag_pipeline.py` | 78 | 7 大职责 + MQE/HyDE + 重排 fallback + 图信号 + merge/expand/compress |
| `test_working_memory.py` | 52 | 7 接口 + forget 三策略 + TTL + 优先级堆 + TF-IDF retrieve |
| `test_semantic_memory.py` | 67 | Entity/Relation + add/retrieve/update/remove + spaCy fallback + 内存 BFS + Manager 协调 |

ML 重依赖（markitdown / qdrant-client / spacy / sentence-transformers）全部用 mock + monkeypatch + try/except ImportError 兜底。

## 6. 性能与限制

- **WorkingMemory**：纯内存，单进程；`retrieve` 是 O(n) TF-IDF + 关键词匹配，对 < 10K 条记忆足够
- **SemanticMemory**：检索召回靠 Qdrant（HNSW），单 query < 50ms（10万级向量）；图遍历 BFS 是 O(E)，对万级实体足够
- **RAG Pipeline**：`load_and_chunk_texts` 单文件 < 1s（PDF 后处理）；`index_chunks` 批量 64 + 失败重试小批 8
- **图谱不持久化**：SemanticMemory 进程结束图谱即失；重启需要重新 `add` 触发实体提取

## 7. 已知限制 / 后续工作

- ❌ Neo4j / 持久化图存储 → 2.0-RC 评估
- ❌ 多模态 RAG（图像/音频检索）→ 2.0-RC 评估
- ❌ 流式 RAG（边检索边生成）→ 2.0-RC
- ❌ MCP 协议接入 → 单独阶段
- ❌ 真异步 OpenAI 客户端 → 单独阶段
- ❌ Multi-agent supervisor/swarm → 2.0-RC

## 8. β 出口标志（已达成）

- [x] AntonAgents 5 大模块（embedding + document_store + qdrant_store + rag_pipeline + working/semantic memory）按 SOP 全部移植 + 测试 + license 标注
- [x] MemoryManager 重写（独立于 AntonAgents 0 字节存根）
- [x] 全量 pytest 500 passed（含 W1-α + W1-β）
- [x] `memory_demo.py` 端到端跑通
- [x] `rag_hello_world.py` 仍可跑通（α 起步代码 0 破坏）
- [x] `docs/rag-guide.md` + `docs/memory-guide.md` 用户向 quickstart
- [x] `pyproject.toml` 升 `2.0.0b1` + 新增 `[retrieval-qdrant] / [rag] / [memory]` optional deps
- [x] 100% 向后兼容（α 与 1.x 全部接口保留）
