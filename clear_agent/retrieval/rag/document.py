"""文档与分块处理

Ported from AntonAgents (CC-BY-NC-SA-4.0)
Original: anton_agents/memory/rag/document.py

提供 RAG pipeline 上游所需的文档建模与分块工具：

- ``Document``：文档对象（content + metadata + 自动 doc_id）
- ``DocumentChunk``：分块后的文档片段（继承 doc_id 并附 chunk_index）
- ``DocumentProcessor``：递归分块器（按分隔符列表 + 重叠窗口）
- ``load_text_file()`` / ``create_document()``：便捷构造器

关键设计：
- doc_id 默认用内容 MD5（同内容幂等去重）
- 分块按 ``separators`` 列表回溯寻找最佳分割点（``\n\n`` → ``\n`` → ``。`` → ``.`` → 空格 → 强制截断）
- ``chunk_overlap`` 控制相邻块之间的重叠字符数（保留语义连续性）
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """文档对象

    Attributes:
        content: 文档原文
        metadata: 任意元数据（source、type、tags 等）
        doc_id: 唯一 ID；缺省时基于 content MD5 自动生成（同内容幂等）
    """

    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.doc_id is None:
            self.doc_id = hashlib.md5(self.content.encode()).hexdigest()


@dataclass
class DocumentChunk:
    """文档分块对象

    Attributes:
        content: 块原文
        metadata: 元数据（继承 Document 并附加 chunk_index / total_chunks / processed_at）
        chunk_id: 唯一块 ID；基于 doc_id + chunk_index + content 前缀 MD5 生成
        doc_id: 来源文档 ID
        chunk_index: 块在文档中的序号（0-based）
    """

    content: str
    metadata: Dict[str, Any]
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    chunk_index: int = 0

    def __post_init__(self) -> None:
        if self.chunk_id is None:
            chunk_content = f"{self.doc_id}_{self.chunk_index}_{self.content[:50]}"
            self.chunk_id = hashlib.md5(chunk_content.encode()).hexdigest()


class DocumentProcessor:
    """递归字符分块处理器

    工作流程：
    1. 文本 ≤ ``chunk_size`` → 不分块，直接返回
    2. 否则按 ``separators`` 列表（优先级从高到低）回溯找分割点
    3. 找不到分割点 → 在 ``chunk_size`` 处强制截断
    4. 下一块起点 = 上一块终点 - ``chunk_overlap``（保留语义重叠）

    Args:
        chunk_size: 单块最大字符数（默认 1000）
        chunk_overlap: 相邻块的重叠字符数（默认 200）
        separators: 分割符优先级列表；缺省 ``["\\n\\n", "\\n", "。", ".", " "]``
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ".", " "]

    # -------- 主接口 --------

    def process_document(self, document: Document) -> List[DocumentChunk]:
        """将单个 Document 切分为 DocumentChunk 列表"""
        chunks = self._split_text(document.content)

        document_chunks: List[DocumentChunk] = []
        for i, chunk_content in enumerate(chunks):
            chunk_metadata = document.metadata.copy()
            chunk_metadata.update(
                {
                    "doc_id": document.doc_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "processed_at": datetime.now().isoformat(),
                }
            )
            chunk = DocumentChunk(
                content=chunk_content,
                metadata=chunk_metadata,
                doc_id=document.doc_id,
                chunk_index=i,
            )
            document_chunks.append(chunk)
        return document_chunks

    def process_documents(self, documents: List[Document]) -> List[DocumentChunk]:
        """批量分块"""
        all_chunks: List[DocumentChunk] = []
        for document in documents:
            all_chunks.extend(self.process_document(document))
        return all_chunks

    # -------- 后处理 --------

    def merge_chunks(
        self, chunks: List[DocumentChunk], max_length: int = 2000
    ) -> List[DocumentChunk]:
        """合并相邻的小块（同 doc_id 且合并后长度 ≤ max_length）"""
        if not chunks:
            return []

        merged_chunks: List[DocumentChunk] = []
        current_chunk = chunks[0]

        for next_chunk in chunks[1:]:
            combined_length = len(current_chunk.content) + len(next_chunk.content)
            if (
                combined_length <= max_length
                and current_chunk.doc_id == next_chunk.doc_id
            ):
                current_chunk.content += "\n" + next_chunk.content
                current_chunk.metadata["total_chunks"] = (
                    current_chunk.metadata.get("total_chunks", 1) + 1
                )
            else:
                merged_chunks.append(current_chunk)
                current_chunk = next_chunk

        merged_chunks.append(current_chunk)
        return merged_chunks

    def filter_chunks(
        self, chunks: List[DocumentChunk], min_length: int = 50
    ) -> List[DocumentChunk]:
        """过滤过短的块"""
        return [c for c in chunks if len(c.content.strip()) >= min_length]

    def add_chunk_metadata(
        self, chunks: List[DocumentChunk], metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """为所有块统一追加元数据（in-place + 返回同一列表）"""
        for chunk in chunks:
            chunk.metadata.update(metadata)
        return chunks

    # -------- 内部 --------

    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break

            split_point = self._find_split_point(text, start, end)
            if split_point == -1:
                split_point = end

            chunks.append(text[start:split_point])
            start = max(start + 1, split_point - self.chunk_overlap)

        return chunks

    def _find_split_point(self, text: str, start: int, end: int) -> int:
        """在 [start, end] 范围内回溯寻找最佳分割点；找不到返回 -1"""
        for separator in self.separators:
            search_start = max(start, end - 100)
            for i in range(end - len(separator), search_start - 1, -1):
                if text[i : i + len(separator)] == separator:
                    return i + len(separator)
        return -1


# -------- 便捷函数 --------


def load_text_file(file_path: str, encoding: str = "utf-8") -> Document:
    """从文件加载为 ``Document``，自动填充 source / type / loaded_at 元数据"""
    with open(file_path, "r", encoding=encoding) as f:
        content = f.read()
    metadata = {
        "source": file_path,
        "type": "text_file",
        "loaded_at": datetime.now().isoformat(),
    }
    return Document(content=content, metadata=metadata)


def create_document(content: str, **metadata: Any) -> Document:
    """便捷构造 ``Document(content=..., metadata={...})``"""
    return Document(content=content, metadata=metadata)


__all__ = [
    "Document",
    "DocumentChunk",
    "DocumentProcessor",
    "load_text_file",
    "create_document",
]
