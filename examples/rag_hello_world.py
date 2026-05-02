"""RAG hello-world —— W4-3 spike demo

演示 ``clear_agent.retrieval`` 模块的最小可用流程：
1. 用 ``SQLiteDocumentStore`` 存储文档（纯 stdlib，零外部依赖）
2. （可选）用 ``DashScopeEmbedding`` REST 模式获取向量
3. （可选）用 ``TFIDFEmbedding`` 在无 ML 模型时跑通 RAG 雏形

运行：

    python examples/rag_hello_world.py

详见 project_docs/07-anton-agents-port.md §4 (Spike 出口标准)
"""

from __future__ import annotations

import os
import time

from clear_agent.retrieval import SQLiteDocumentStore


# ==================================================================
# Part 1: 纯 SQLite 文档存储（必跑通）
# ==================================================================


def demo_document_store() -> None:
    print("=" * 60)
    print("Part 1: SQLiteDocumentStore CRUD")
    print("=" * 60)

    SQLiteDocumentStore.reset_instances()
    store = SQLiteDocumentStore(":memory:")

    # 1. 批量添加文档
    docs = [
        ("Python is a programming language.", {"topic": "programming"}),
        ("LangGraph is a graph framework for LLMs.", {"topic": "ai"}),
        ("ClearAgent is a lightweight agent framework.", {"topic": "ai"}),
        ("SQLite is an embedded database.", {"topic": "database"}),
    ]
    doc_ids = []
    for content, meta in docs:
        doc_ids.append(store.add_document(content, metadata=meta))
    print(f"添加 {len(doc_ids)} 条文档")

    # 2. 关键词搜索（基于结构化字段）
    print("\n查询 memory_type=document 的全部记录:")
    rows = store.search_memories(memory_type="document", limit=10)
    for r in rows:
        print(f"  - [{r['properties'].get('topic'):>11}] {r['content']}")

    # 3. 单条修改
    target = doc_ids[0]
    store.update_memory(target, importance=0.9)
    print(f"\n更新 {target[:8]}… 的 importance 为 0.9")
    print(f"按 importance>=0.7 检索:")
    for r in store.search_memories(importance_threshold=0.7, limit=10):
        print(f"  - {r['content']}  (importance={r['importance']})")

    # 4. 统计
    print("\n数据库统计:")
    stats = store.get_database_stats()
    print(f"  memories_count: {stats['memories_count']}")
    print(f"  memory_types:   {stats['memory_types']}")
    print(f"  store_type:     {stats['store_type']}")


# ==================================================================
# Part 2: 嵌入向量（需配置环境变量；缺失则跳过）
# ==================================================================


def demo_embedding() -> None:
    print()
    print("=" * 60)
    print("Part 2: Embeddings（按需，缺依赖即跳过）")
    print("=" * 60)

    embed_type = os.getenv("EMBED_MODEL_TYPE", "").strip()
    embed_key = os.getenv("EMBED_API_KEY", "").strip()
    embed_url = os.getenv("EMBED_BASE_URL", "").strip()

    if not embed_type:
        print("[skip] 未设置 EMBED_MODEL_TYPE，跳过嵌入演示。")
        print("       配置 .env 中 EMBED_MODEL_TYPE / EMBED_API_KEY / EMBED_BASE_URL 后重跑。")
        print("       例如：EMBED_MODEL_TYPE=dashscope EMBED_BASE_URL=https://...")
        return

    try:
        from clear_agent.retrieval import get_text_embedder, get_dimension

        embedder = get_text_embedder()
        dim = get_dimension()
        print(f"已构造 embedder: {type(embedder).__name__}, dim={dim}")
        sample = embedder.encode("ClearAgent is great")
        # 兼容 numpy.ndarray / list
        head = list(sample)[:5] if hasattr(sample, "__iter__") else [sample]
        print(f"示例向量前 5 维: {head}")
    except ImportError as e:
        print(f"[skip] 缺少依赖：{e}")
    except Exception as e:
        print(f"[error] 嵌入加载失败：{type(e).__name__}: {e}")


# ==================================================================
# Part 3: 简单 RAG 雏形（如果 sklearn 可用）
# ==================================================================


def demo_tiny_rag() -> None:
    print()
    print("=" * 60)
    print("Part 3: 极简 RAG 雏形（基于 TF-IDF；缺 sklearn 即跳过）")
    print("=" * 60)
    try:
        import numpy as np  # type: ignore

        from clear_agent.retrieval import TFIDFEmbedding
    except ImportError as e:
        print(f"[skip] {e}")
        return

    try:
        embedder = TFIDFEmbedding(max_features=100)
    except ImportError as e:
        print(f"[skip] {e}")
        return

    corpus = [
        "Python is a high-level programming language popular in AI.",
        "LangGraph builds stateful multi-actor applications with LLMs.",
        "ClearAgent is a lightweight agent framework based on OpenAI API.",
        "SQLite is a small embedded relational database.",
    ]
    embedder.fit(corpus)
    doc_vecs = [embedder.encode(d) for d in corpus]

    query = "Tell me about agent frameworks"
    q_vec = embedder.encode(query)

    def cos_sim(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        return float(np.dot(a, b) / denom)

    scored = sorted(
        zip(corpus, doc_vecs), key=lambda pair: cos_sim(q_vec, pair[1]), reverse=True
    )

    print(f"Query: {query}")
    print("Top-2 命中:")
    for doc, _ in scored[:2]:
        print(f"  - {doc}")


def main() -> None:
    demo_document_store()
    demo_embedding()
    demo_tiny_rag()
    print()
    print("✅ rag_hello_world 跑通 —— W4-3 spike 出口达成（详见 project_docs/07）。")


if __name__ == "__main__":
    main()
