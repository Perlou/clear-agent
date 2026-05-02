"""ClearAgent retrieval —— 嵌入与文档存储

2.0-α (W4) spike：从 AntonAgents 移植 Embedding + DocumentStore 打底，
为 2.0-β 完整 RAG / Memory 套件做准备。

公共入口：

```python
from clear_agent.retrieval import (
    EmbeddingModel,
    create_embedding_model,
    create_embedding_model_with_fallback,
    get_text_embedder,
    SQLiteDocumentStore,
)
```

详见 project_docs/07-anton-agents-port.md
"""

from .embeddings import (
    DashScopeEmbedding,
    EmbeddingModel,
    LocalTransformerEmbedding,
    TFIDFEmbedding,
    create_embedding_model,
    create_embedding_model_with_fallback,
    get_dimension,
    get_text_embedder,
    refresh_embedder,
)
from .rag import (
    Document,
    DocumentChunk,
    DocumentProcessor,
    create_document,
    load_text_file,
)
from .storage import (
    DEFAULT_COLLECTION,
    QDRANT_AVAILABLE,
    DocumentStore,
    QdrantConnectionManager,
    QdrantVectorStore,
    SQLiteDocumentStore,
)

__all__ = [
    # embeddings
    "EmbeddingModel",
    "LocalTransformerEmbedding",
    "TFIDFEmbedding",
    "DashScopeEmbedding",
    "create_embedding_model",
    "create_embedding_model_with_fallback",
    "get_text_embedder",
    "get_dimension",
    "refresh_embedder",
    # rag
    "Document",
    "DocumentChunk",
    "DocumentProcessor",
    "create_document",
    "load_text_file",
    # storage
    "DocumentStore",
    "SQLiteDocumentStore",
    "QdrantVectorStore",
    "QdrantConnectionManager",
    "QDRANT_AVAILABLE",
    "DEFAULT_COLLECTION",
]
