"""RAG pipeline 模块

2.0-β 起步移植：先有 ``Document`` / ``DocumentProcessor`` 打底，
完整 ``pipeline.py`` 的移植在 β-W1.3 接下来推进。
"""

from .document import (
    Document,
    DocumentChunk,
    DocumentProcessor,
    create_document,
    load_text_file,
)

__all__ = [
    "Document",
    "DocumentChunk",
    "DocumentProcessor",
    "create_document",
    "load_text_file",
]
