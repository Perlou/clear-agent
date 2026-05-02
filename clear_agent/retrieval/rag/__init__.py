"""RAG pipeline 模块 —— 完整流水线（β-W1.3 移植）"""

from .document import (
    Document,
    DocumentChunk,
    DocumentProcessor,
    create_document,
    load_text_file,
)
from .pipeline import (
    DEFAULT_RAG_COLLECTION,
    build_graph_from_chunks,
    compress_ranked_items,
    compute_graph_signals_from_pool,
    create_rag_pipeline,
    embed_query,
    expand_neighbors_from_pool,
    index_chunks,
    load_and_chunk_texts,
    merge_snippets,
    merge_snippets_grouped,
    rank,
    rerank_with_cross_encoder,
    search_vectors,
    search_vectors_expanded,
    tldr_summarize,
)

__all__ = [
    # document
    "Document",
    "DocumentChunk",
    "DocumentProcessor",
    "create_document",
    "load_text_file",
    # pipeline
    "DEFAULT_RAG_COLLECTION",
    "load_and_chunk_texts",
    "build_graph_from_chunks",
    "index_chunks",
    "embed_query",
    "search_vectors",
    "search_vectors_expanded",
    "rerank_with_cross_encoder",
    "compute_graph_signals_from_pool",
    "rank",
    "merge_snippets",
    "expand_neighbors_from_pool",
    "merge_snippets_grouped",
    "compress_ranked_items",
    "tldr_summarize",
    "create_rag_pipeline",
]
