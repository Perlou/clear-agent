# RAG 指南（2.0-β 用户向 quickstart）

> 设计 spec 详见 [`project_docs/07-anton-agents-port.md`](../project_docs/07-anton-agents-port.md)。
> 本文给最短上手路径。

## 1. 心智模型

```
   ┌──────────────┐
   │ 文档（PDF/MD/...）
   └──────┬───────┘
          │ load_and_chunk_texts
          ▼
   ┌──────────────┐
   │ chunks (List[Dict])  metadata 含 doc_id/start/end/heading_path
   └──────┬───────┘
          │ index_chunks (embed + Qdrant upsert)
          ▼
   ┌──────────────┐
   │ Qdrant 向量库（带 RAG 标签 is_rag_data=True）
   └──────┬───────┘
          │ search_vectors / search_vectors_expanded
          ▼
   ┌──────────────┐
   │ ranked_items → rerank → merge_snippets_grouped (含引用)
   └──────────────┘
```

## 2. 安装

```bash
pip install clear-agent[retrieval-qdrant]    # 必须：Qdrant 向量库
pip install clear-agent[rag]                 # 推荐：markitdown 文档加载 + 重排
```

启动本地 Qdrant：
```bash
docker run -p 6333:6333 qdrant/qdrant
```

## 3. 一行管道

```python
from clear_agent.retrieval.rag import create_rag_pipeline

rag = create_rag_pipeline(
    qdrant_url="http://localhost:6333",
    rag_namespace="my_kb",
)

# 添加文档
rag["add_documents"](["docs/a.pdf", "docs/b.md"])

# 检索
hits = rag["search"]("如何配置 LLM？", top_k=5)
for h in hits:
    print(h["score"], h["metadata"]["content"][:100])
```

## 4. 散件接口（细粒度控制）

```python
from clear_agent.retrieval.rag import (
    load_and_chunk_texts,
    index_chunks,
    search_vectors,
    search_vectors_expanded,
    rerank_with_cross_encoder,
    rank,
    compute_graph_signals_from_pool,
    merge_snippets_grouped,
    compress_ranked_items,
)
from clear_agent.retrieval import QdrantVectorStore

store = QdrantVectorStore(collection_name="my_kb", vector_size=384)

# 1. 加载 + 分块
chunks = load_and_chunk_texts(["a.pdf"], chunk_size=800, chunk_overlap=100)

# 2. 索引
index_chunks(store=store, chunks=chunks, rag_namespace="my_kb")

# 3. 检索
hits = search_vectors(store=store, query="问题", top_k=20, rag_namespace="my_kb")

# 4. 图信号 + 排序融合
graph_sig = compute_graph_signals_from_pool(hits)
ranked = rank(hits, graph_signals=graph_sig, w_vector=0.7, w_graph=0.3)

# 5. 重排（如装了 sentence-transformers）
top = rerank_with_cross_encoder("问题", ranked, top_k=5)

# 6. 压缩 + 合并
compressed = compress_ranked_items(top, max_per_doc=2)
context = merge_snippets_grouped(compressed, max_chars=1500, include_citations=True)

print(context)  # 含 References: 尾注
```

## 5. 高级：MQE / HyDE 查询扩展

```python
from clear_agent import ClearAgentLLM

llm = ClearAgentLLM()
rag = create_rag_pipeline(qdrant_url="http://localhost:6333", llm=llm)

# enable_mqe: 让 LLM 生成 N 个等价/互补查询
# enable_hyde: 让 LLM 写一段假设性回答用于检索
hits = rag["search_advanced"](
    "如何调试图执行卡死？",
    top_k=5,
    enable_mqe=True,
    enable_hyde=True,
)
```

## 6. 文档加载支持的格式

`load_and_chunk_texts` 通过 markitdown 自动识别：
- **文档**：PDF / DOCX / XLSX / PPTX
- **文本**：TXT / MD / CSV / JSON / XML / HTML
- **代码**：PY / JS / TS / JAVA / CPP / ...
- **图像**（OCR）：JPG / PNG / TIFF / WEBP
- **音频**（转写）：MP3 / WAV / M4A / FLAC
- **配置**：YAML / TOML / INI / CONF

PDF 走增强后处理（短行合并 + 段落重组）。

## 7. 与 graph 节点集成

```python
from clear_agent.core.graph import StateGraph, START, END
from clear_agent.retrieval.rag import create_rag_pipeline

rag = create_rag_pipeline(qdrant_url="http://localhost:6333")

def retrieve_node(state):
    """RAG 节点：把检索结果注入到 state.context"""
    query = state["question"]
    hits = rag["search"](query, top_k=5)
    context = "\n\n".join(h["metadata"]["content"] for h in hits)
    return {"context": context}

def answer_node(state):
    """生成节点：用 context + question 调 LLM"""
    ...

g = StateGraph(MyState)
g.add_node("retrieve", retrieve_node)
g.add_node("answer", answer_node)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "answer")
g.add_edge("answer", END)
```

## 8. 多租户（按 namespace 隔离）

```python
# 用户 A 的知识库
rag_a = create_rag_pipeline(qdrant_url="...", rag_namespace="user_a")
rag_a["add_documents"]([...])

# 用户 B 的知识库（同一个 Qdrant 集合，按 namespace 隔离）
rag_b = create_rag_pipeline(qdrant_url="...", rag_namespace="user_b")
rag_b["add_documents"]([...])

# 检索时自动按 namespace 过滤
hits_a = rag_a["search"]("query")  # 只命中 user_a 的文档
```

## 9. 常见坑

- **维度不匹配** → 嵌入模型 dim 必须等于 Qdrant 集合的 vector_size。embeddings.get_dimension() 探测一次后做单例。
- **markitdown 没装** → PDF 走 fallback 文本读，质量较差。强烈建议 `pip install markitdown`。
- **collection 已存在** → `clear_agent_rag_vectors` 是默认集合名，跨多个 pipeline 共享。改 `collection_name` 隔离。
- **MQE / HyDE 无效** → 必须传 `llm=ClearAgentLLM()`，否则函数返回原 query。
