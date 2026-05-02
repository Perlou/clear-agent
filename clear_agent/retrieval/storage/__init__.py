"""存储后端"""

from .document_store import DocumentStore, SQLiteDocumentStore

__all__ = ["DocumentStore", "SQLiteDocumentStore"]
