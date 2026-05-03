"""ClearAgent retrieval —— 嵌入与文档存储

嵌入 + 文档存储模块（移植自 AntonAgents）Embedding + DocumentStore 打底，
。

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
