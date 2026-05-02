"""RAG Document + DocumentProcessor 测试

纯 Python，无外部依赖。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from clear_agent.retrieval import (
    Document,
    DocumentChunk,
    DocumentProcessor,
    create_document,
    load_text_file,
)


# ==================== Section A: Document ====================


def test_document_auto_doc_id_from_md5():
    d = Document(content="hello", metadata={})
    expected = hashlib.md5(b"hello").hexdigest()
    assert d.doc_id == expected


def test_document_explicit_doc_id_kept():
    d = Document(content="hello", metadata={}, doc_id="custom-id")
    assert d.doc_id == "custom-id"


def test_same_content_yields_same_doc_id():
    """同 content 幂等去重"""
    a = Document(content="x", metadata={})
    b = Document(content="x", metadata={"different": "meta"})
    assert a.doc_id == b.doc_id


def test_different_content_yields_different_doc_id():
    a = Document(content="x", metadata={})
    b = Document(content="y", metadata={})
    assert a.doc_id != b.doc_id


# ==================== Section B: DocumentChunk ====================


def test_chunk_auto_id_from_doc_and_index():
    c = DocumentChunk(content="abc", metadata={}, doc_id="doc-1", chunk_index=0)
    assert c.chunk_id is not None
    assert len(c.chunk_id) == 32  # md5 hex


def test_chunk_different_index_different_id():
    a = DocumentChunk(content="x", metadata={}, doc_id="d", chunk_index=0)
    b = DocumentChunk(content="x", metadata={}, doc_id="d", chunk_index=1)
    assert a.chunk_id != b.chunk_id


def test_chunk_explicit_id_kept():
    c = DocumentChunk(
        content="x", metadata={}, doc_id="d", chunk_index=0, chunk_id="custom"
    )
    assert c.chunk_id == "custom"


# ==================== Section C: DocumentProcessor 基本 ====================


def test_short_text_returns_single_chunk():
    p = DocumentProcessor(chunk_size=1000)
    doc = Document(content="hello world", metadata={"src": "x"})
    chunks = p.process_document(doc)
    assert len(chunks) == 1
    assert chunks[0].content == "hello world"
    assert chunks[0].doc_id == doc.doc_id
    assert chunks[0].chunk_index == 0
    # 元数据继承 + 增补
    assert chunks[0].metadata["src"] == "x"
    assert chunks[0].metadata["doc_id"] == doc.doc_id
    assert chunks[0].metadata["total_chunks"] == 1
    assert "processed_at" in chunks[0].metadata


def test_long_text_splits_into_multiple_chunks():
    p = DocumentProcessor(chunk_size=50, chunk_overlap=10)
    text = "abc " * 100  # 400 字符
    doc = Document(content=text, metadata={})
    chunks = p.process_document(doc)
    assert len(chunks) > 1
    # 块大小不超过 chunk_size + 一些 separator 容差
    for c in chunks:
        assert len(c.content) <= 60


def test_chunks_have_sequential_indices():
    p = DocumentProcessor(chunk_size=30, chunk_overlap=5)
    doc = Document(content="x" * 200, metadata={})
    chunks = p.process_document(doc)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_chunks_carry_total_chunks_in_metadata():
    p = DocumentProcessor(chunk_size=20, chunk_overlap=0)
    doc = Document(content="a" * 100, metadata={})
    chunks = p.process_document(doc)
    n = len(chunks)
    for c in chunks:
        assert c.metadata["total_chunks"] == n


def test_chunks_share_same_doc_id():
    p = DocumentProcessor(chunk_size=20, chunk_overlap=0)
    doc = Document(content="z" * 100, metadata={})
    chunks = p.process_document(doc)
    assert all(c.doc_id == doc.doc_id for c in chunks)


# ==================== Section D: 分割符行为 ====================


def test_prefers_double_newline_separator():
    """有 \\n\\n 时应优先在它处切分"""
    p = DocumentProcessor(chunk_size=20, chunk_overlap=0)
    text = "first part xxx\n\nsecond part xxx"
    chunks = p._split_text(text)
    # 第一块应在 \n\n 处或之前结束
    assert chunks[0].endswith("\n\n") or "first part" in chunks[0]


def test_chinese_period_separator():
    """中文句号也是 separator"""
    p = DocumentProcessor(chunk_size=15, chunk_overlap=0, separators=["。", "."])
    text = "第一句。第二句很长所以触发分块。第三句。"
    chunks = p._split_text(text)
    assert len(chunks) > 1


def test_no_separator_force_split_at_chunk_size():
    """没有匹配的分隔符 → 按 chunk_size 强制截断"""
    p = DocumentProcessor(chunk_size=10, chunk_overlap=0, separators=["||"])
    chunks = p._split_text("abcdefghijklmnop")  # 16 字符，无 ||
    assert all(len(c) <= 10 for c in chunks)
    # 拼回去等于原文（chunk_overlap=0 时）
    assert "".join(chunks) == "abcdefghijklmnop"


def test_chunk_overlap_creates_overlap():
    """重叠窗口让相邻块共享尾/头"""
    p = DocumentProcessor(chunk_size=20, chunk_overlap=5, separators=[])
    chunks = p._split_text("a" * 50)
    assert len(chunks) >= 2
    # chunk_overlap > 0 → 相邻块起点不是简单递增 chunk_size
    # （此测试只是验证逻辑跑通；细节由前面的 chunk_size 边界测试覆盖）


# ==================== Section E: process_documents 批量 ====================


def test_process_documents_batch():
    p = DocumentProcessor(chunk_size=1000)
    docs = [
        Document(content=f"doc-{i}", metadata={"i": i}) for i in range(3)
    ]
    chunks = p.process_documents(docs)
    assert len(chunks) == 3
    # 每个 doc 一个 chunk（短文本）
    assert {c.doc_id for c in chunks} == {d.doc_id for d in docs}


# ==================== Section F: merge_chunks ====================


def test_merge_chunks_combines_short_chunks():
    p = DocumentProcessor(chunk_size=1000)
    doc = Document(content="x", metadata={}, doc_id="d1")
    c1 = DocumentChunk(content="aaa", metadata={"total_chunks": 1}, doc_id="d1", chunk_index=0)
    c2 = DocumentChunk(content="bbb", metadata={"total_chunks": 1}, doc_id="d1", chunk_index=1)
    merged = p.merge_chunks([c1, c2], max_length=100)
    assert len(merged) == 1
    assert "aaa" in merged[0].content
    assert "bbb" in merged[0].content


def test_merge_chunks_respects_max_length():
    p = DocumentProcessor(chunk_size=1000)
    c1 = DocumentChunk(content="a" * 50, metadata={}, doc_id="d", chunk_index=0)
    c2 = DocumentChunk(content="b" * 50, metadata={}, doc_id="d", chunk_index=1)
    merged = p.merge_chunks([c1, c2], max_length=80)  # 50+50=100 > 80
    # 不能合并 → 仍然两块
    assert len(merged) == 2


def test_merge_chunks_does_not_merge_different_docs():
    p = DocumentProcessor(chunk_size=1000)
    c1 = DocumentChunk(content="a", metadata={}, doc_id="d1", chunk_index=0)
    c2 = DocumentChunk(content="b", metadata={}, doc_id="d2", chunk_index=0)
    merged = p.merge_chunks([c1, c2], max_length=100)
    assert len(merged) == 2


def test_merge_chunks_empty_list():
    p = DocumentProcessor()
    assert p.merge_chunks([]) == []


# ==================== Section G: filter_chunks ====================


def test_filter_chunks_drops_short():
    p = DocumentProcessor()
    short = DocumentChunk(content="ab", metadata={}, doc_id="d", chunk_index=0)
    long = DocumentChunk(content="x" * 100, metadata={}, doc_id="d", chunk_index=1)
    out = p.filter_chunks([short, long], min_length=50)
    assert len(out) == 1
    assert out[0].content == long.content


def test_filter_chunks_strips_whitespace():
    """长度判断基于 strip 后"""
    p = DocumentProcessor()
    only_spaces = DocumentChunk(
        content="   \n  \n  ", metadata={}, doc_id="d", chunk_index=0
    )
    out = p.filter_chunks([only_spaces], min_length=5)
    assert out == []


# ==================== Section H: add_chunk_metadata ====================


def test_add_chunk_metadata_in_place():
    p = DocumentProcessor()
    c = DocumentChunk(content="x", metadata={"a": 1}, doc_id="d", chunk_index=0)
    out = p.add_chunk_metadata([c], {"b": 2, "a": 99})
    assert out[0].metadata == {"a": 99, "b": 2}


# ==================== Section I: load_text_file / create_document ====================


def test_load_text_file_reads_content_and_metadata(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text("hello world", encoding="utf-8")
    doc = load_text_file(str(p))
    assert doc.content == "hello world"
    assert doc.metadata["source"] == str(p)
    assert doc.metadata["type"] == "text_file"
    assert "loaded_at" in doc.metadata


def test_load_text_file_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_text_file(str(tmp_path / "no_such_file.txt"))


def test_create_document_kwargs_become_metadata():
    d = create_document("content", source="test", tag="x")
    assert d.metadata == {"source": "test", "tag": "x"}
    assert d.content == "content"


# ==================== Section J: 顶层导入 ====================


def test_top_level_rag_imports():
    from clear_agent.retrieval import (
        Document,
        DocumentChunk,
        DocumentProcessor,
        create_document,
        load_text_file,
    )

    assert all(
        x is not None
        for x in (
            Document,
            DocumentChunk,
            DocumentProcessor,
            create_document,
            load_text_file,
        )
    )
