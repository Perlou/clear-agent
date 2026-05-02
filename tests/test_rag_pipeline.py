"""RAG Pipeline 测试

覆盖 1380 LOC 移植的 7 大职责：
- 文档加载 / 后处理 / fallback
- 语言检测 / token 近似
- Markdown-aware 分块
- 索引（mock embedder + store）
- 检索 / MQE / HyDE（mock LLM）
- 重排 / 图信号融合 / rank
- 合并 / 邻居扩展 / 分组带引用 / 压缩 / TL;DR

ML 重依赖（markitdown / sentence-transformers / langdetect）全部走 ImportError 兜底，
测试默认跳过 / 用 mock。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from clear_agent.retrieval.rag.pipeline import (
    DEFAULT_RAG_COLLECTION,
    _approx_token_len,
    _chunk_paragraphs,
    _detect_lang,
    _fallback_text_reader,
    _is_cjk,
    _is_markitdown_supported_format,
    _normalize_vec,
    _post_process_pdf_text,
    _preprocess_markdown_for_embedding,
    _prompt_hyde,
    _prompt_mqe,
    _split_paragraphs_with_headings,
    build_graph_from_chunks,
    compress_ranked_items,
    compute_graph_signals_from_pool,
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


# ==================== Section A: CJK / token / lang ====================


def test_is_cjk_basic_chars():
    assert _is_cjk("中")
    assert _is_cjk("你")
    assert _is_cjk("好")


def test_is_cjk_non_cjk():
    assert not _is_cjk("a")
    assert not _is_cjk("1")
    assert not _is_cjk(" ")
    assert not _is_cjk("！")  # 全角标点不在 CJK 统一汉字范围


def test_approx_token_len_pure_english():
    assert _approx_token_len("hello world") == 2
    assert _approx_token_len("the quick brown fox") == 4
    assert _approx_token_len("") == 0


def test_approx_token_len_pure_cjk():
    """4 个 CJK 字符 + 整段算 1 word（无空格）= 5"""
    assert _approx_token_len("你好世界") == 5


def test_approx_token_len_mixed():
    """hello + 你 + 好 + 'hello' word + '你好' word = 4"""
    assert _approx_token_len("hello 你好") == 4


def test_detect_lang_returns_unknown_when_langdetect_missing():
    """venv 里没装 langdetect → 返回 unknown"""
    assert _detect_lang("hello world") == "unknown"


def test_detect_lang_empty_string():
    assert _detect_lang("") == "unknown"


# ==================== Section B: 后缀名支持 ====================


def test_is_markitdown_supported_format_pdf():
    assert _is_markitdown_supported_format("a.pdf")
    assert _is_markitdown_supported_format("PATH/B.PDF")  # 大写


def test_is_markitdown_supported_format_office():
    for ext in [".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt"]:
        assert _is_markitdown_supported_format(f"x{ext}")


def test_is_markitdown_supported_format_image():
    for ext in [".jpg", ".png", ".tiff", ".webp"]:
        assert _is_markitdown_supported_format(f"x{ext}")


def test_is_markitdown_supported_format_unknown():
    assert not _is_markitdown_supported_format("x.exe")
    assert not _is_markitdown_supported_format("x")  # 无后缀


# ==================== Section C: PDF 后处理 ====================


def test_post_process_pdf_strips_pure_digit_lines():
    """纯数字行（页码）被丢弃"""
    text = "# Title\n\n123\n\nReal content here."
    out = _post_process_pdf_text(text)
    assert "123" not in out
    assert "Real content" in out


def test_post_process_pdf_strips_short_noise_lines():
    """长度 ≤2 且非数字的行被丢弃"""
    text = "real content\nx\nab\nmore content"
    out = _post_process_pdf_text(text)
    # 'x' 'ab' 都是噪音
    assert " x " not in out
    assert "real content" in out


def test_post_process_pdf_merges_short_lines():
    """两行都不长且不是标题 → 合并"""
    text = "short line one\nshort line two\nshort line three"
    out = _post_process_pdf_text(text)
    # 至少两行被合并成一行
    assert len(out.splitlines()) <= 3


def test_post_process_pdf_preserves_headings():
    """# 标题行不和正文合并"""
    text = "# Heading\nbody text\nmore body"
    out = _post_process_pdf_text(text)
    assert "# Heading" in out


# ==================== Section D: Markdown 预处理 ====================


def test_preprocess_strips_headers():
    out = _preprocess_markdown_for_embedding("# Title\n## Sub")
    assert "#" not in out


def test_preprocess_strips_links_keeps_text():
    out = _preprocess_markdown_for_embedding("see [docs](http://x)")
    assert "docs" in out
    assert "http" not in out


def test_preprocess_strips_emphasis():
    out = _preprocess_markdown_for_embedding("**bold** and *italic*")
    assert "*" not in out
    assert "bold" in out
    assert "italic" in out


def test_preprocess_strips_inline_code():
    out = _preprocess_markdown_for_embedding("use `foo()` here")
    assert "`" not in out
    assert "foo()" in out


def test_preprocess_strips_code_blocks_keeps_content():
    out = _preprocess_markdown_for_embedding("```python\nx = 1\n```")
    assert "```" not in out
    assert "x = 1" in out


def test_preprocess_collapses_whitespace():
    out = _preprocess_markdown_for_embedding("a    b\n\n\n\nc")
    assert "    " not in out
    assert "\n\n\n" not in out


# ==================== Section E: 段落切分 ====================


def test_split_paragraphs_no_heading():
    out = _split_paragraphs_with_headings("para 1\n\npara 2")
    assert len(out) == 2
    assert out[0]["heading_path"] is None


def test_split_paragraphs_single_heading():
    out = _split_paragraphs_with_headings("# Title\n\nbody")
    assert len(out) >= 1
    # 第一段（body）的 heading_path 是 Title
    assert any(p["heading_path"] == "Title" for p in out)


def test_split_paragraphs_nested_heading_path():
    """heading 嵌套 → heading_path 用 ' > ' 串联"""
    out = _split_paragraphs_with_headings(
        "# A\n\nbody1\n\n## B\n\nbody2\n\n### C\n\nbody3"
    )
    paths = [p["heading_path"] for p in out]
    assert "A" in paths
    assert any("A > B" == p for p in paths)
    assert any("A > B > C" == p for p in paths)


def test_split_paragraphs_returns_full_text_when_empty():
    """没有任何段落 → 兜底返回整段"""
    out = _split_paragraphs_with_headings("")
    assert len(out) == 1


def test_split_paragraphs_records_offsets():
    out = _split_paragraphs_with_headings("# T\n\nhello")
    for p in out:
        assert "start" in p and "end" in p
        assert p["start"] >= 0
        assert p["end"] >= p["start"]


# ==================== Section F: chunk_paragraphs ====================


def test_chunk_paragraphs_single_under_budget():
    """单段且 token 数小 → 一块"""
    paras = [{"content": "hello world", "start": 0, "end": 11, "heading_path": None}]
    chunks = _chunk_paragraphs(paras, chunk_tokens=100, overlap_tokens=0)
    assert len(chunks) == 1


def test_chunk_paragraphs_token_budget_splits():
    """多段且超出 chunk_tokens → 多块"""
    paras = [
        {"content": " ".join(["w"] * 50), "start": i * 100, "end": (i + 1) * 100, "heading_path": None}
        for i in range(5)
    ]
    chunks = _chunk_paragraphs(paras, chunk_tokens=60, overlap_tokens=0)
    assert len(chunks) > 1


def test_chunk_paragraphs_inherits_heading_path():
    """块的 heading_path = 内部最后一个有 heading 的段落的 heading_path"""
    paras = [
        {"content": "x", "start": 0, "end": 1, "heading_path": "A"},
        {"content": "y", "start": 1, "end": 2, "heading_path": "A > B"},
    ]
    chunks = _chunk_paragraphs(paras, chunk_tokens=100, overlap_tokens=0)
    assert chunks[0]["heading_path"] == "A > B"


def test_chunk_paragraphs_overlap_keeps_tail():
    """overlap > 0 时尾部段落被保留进下一块"""
    paras = [
        {"content": " ".join(["w"] * 40), "start": 0, "end": 80, "heading_path": None},
        {"content": " ".join(["w"] * 40), "start": 80, "end": 160, "heading_path": None},
        {"content": " ".join(["w"] * 40), "start": 160, "end": 240, "heading_path": None},
    ]
    chunks = _chunk_paragraphs(paras, chunk_tokens=50, overlap_tokens=20)
    assert len(chunks) >= 2


# ==================== Section G: load_and_chunk_texts（端到端 + fallback） ====================


def test_load_and_chunk_skip_missing_files(tmp_path: Path):
    """不存在的路径直接跳过，不抛错"""
    chunks = load_and_chunk_texts([str(tmp_path / "nope.txt")])
    assert chunks == []


def test_load_and_chunk_basic(tmp_path: Path):
    """无 markitdown → 走 fallback，分块带正确的 metadata"""
    p = tmp_path / "a.md"
    p.write_text("# Title\n\nhello world", encoding="utf-8")

    chunks = load_and_chunk_texts([str(p)], namespace="ns1")
    assert len(chunks) >= 1
    meta = chunks[0]["metadata"]
    assert meta["source_path"] == str(p)
    assert meta["file_ext"] == ".md"
    assert meta["namespace"] == "ns1"
    assert meta["external"] is True
    assert "doc_id" in meta
    assert "content_hash" in meta
    assert meta["format"] == "markdown"


def test_load_and_chunk_dedupe(tmp_path: Path):
    """同内容写两份 → 第二份的同 hash chunks 被去重"""
    p1 = tmp_path / "a.md"
    p2 = tmp_path / "b.md"
    same = "shared content for dedupe testing only"
    p1.write_text(same, encoding="utf-8")
    p2.write_text(same, encoding="utf-8")
    chunks = load_and_chunk_texts([str(p1), str(p2)])
    # 去重后只剩一条（content_hash 相同）
    hashes = {c["metadata"]["content_hash"] for c in chunks}
    assert len(hashes) == len(chunks)  # 没有重复 hash


def test_fallback_text_reader_utf8(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("hello 你好", encoding="utf-8")
    assert _fallback_text_reader(str(p)) == "hello 你好"


def test_fallback_text_reader_missing_returns_empty(tmp_path: Path):
    assert _fallback_text_reader(str(tmp_path / "no.txt")) == ""


# ==================== Section H: build_graph_from_chunks（mock neo4j） ====================


def test_build_graph_creates_doc_and_chunk_nodes():
    neo = MagicMock()
    chunks = [
        {
            "id": "c1",
            "metadata": {"doc_id": "d1", "source_path": "x.md", "start": 0, "end": 10},
        },
        {
            "id": "c2",
            "metadata": {"doc_id": "d1", "source_path": "x.md", "start": 10, "end": 20},
        },
    ]
    build_graph_from_chunks(neo, chunks)
    # add_entity 被调用：1 个 doc + 2 个 memory
    assert neo.add_entity.call_count >= 3
    # add_relationship: 2 条 HAS_CHUNK
    assert neo.add_relationship.call_count == 2


def test_build_graph_swallows_neo4j_exceptions():
    """neo4j 抛错也不传播，保证整体不中断"""
    neo = MagicMock()
    neo.add_entity.side_effect = RuntimeError("db down")
    neo.add_relationship.side_effect = RuntimeError("db down")
    # 不应抛异常
    build_graph_from_chunks(
        neo,
        [{"id": "c1", "metadata": {"doc_id": "d1", "source_path": "x", "start": 0, "end": 1}}],
    )


# ==================== Section I: 索引（mock embedder + store） ====================


class _FakeEmbedder:
    """简易 embedder：把文本长度作为单维向量"""

    def encode(self, texts):
        if isinstance(texts, str):
            return [float(len(texts))] * 384
        return [[float(len(t))] * 384 for t in texts]


def test_index_chunks_no_chunks_returns_early():
    store = MagicMock()
    index_chunks(store=store, chunks=[])
    store.add_vectors.assert_not_called()


def test_index_chunks_calls_store_add_vectors():
    store = MagicMock()
    store.add_vectors.return_value = True
    chunks = [
        {
            "id": "c1",
            "content": "hello world",
            "metadata": {"source_path": "x.md", "doc_id": "d1"},
        }
    ]
    index_chunks(
        store=store, chunks=chunks, batch_size=8,
        rag_namespace="testns", embedder=_FakeEmbedder(),
    )
    store.add_vectors.assert_called_once()
    call = store.add_vectors.call_args
    metas = call.kwargs["metadata"]
    assert metas[0]["is_rag_data"] is True
    assert metas[0]["data_source"] == "rag_pipeline"
    assert metas[0]["rag_namespace"] == "testns"
    assert metas[0]["memory_type"] == "rag_chunk"
    # content 透传
    assert metas[0]["content"] == "hello world"
    # 原始 metadata 被合并
    assert metas[0]["doc_id"] == "d1"


def test_index_chunks_failure_raises():
    store = MagicMock()
    store.add_vectors.return_value = False
    chunks = [{"id": "c1", "content": "x", "metadata": {}}]
    with pytest.raises(RuntimeError):
        index_chunks(store=store, chunks=chunks, embedder=_FakeEmbedder())


def test_index_chunks_embedder_failure_uses_zero_vec_fallback():
    """embedder.encode 抛错 → 重试 + 零向量兜底"""
    store = MagicMock()
    store.add_vectors.return_value = True

    fail_embedder = MagicMock()
    fail_embedder.encode.side_effect = RuntimeError("model down")

    chunks = [{"id": "c1", "content": "x", "metadata": {}}]
    # 不应抛异常
    with patch("time.sleep"):  # 避免真等 2 秒
        index_chunks(
            store=store, chunks=chunks, batch_size=8, embedder=fail_embedder
        )
    store.add_vectors.assert_called_once()


# ==================== Section J: embed_query / search ====================


def test_embed_query_with_fake_embedder():
    v = embed_query("hello", embedder=_FakeEmbedder())
    assert len(v) == 384
    assert all(x == 5.0 for x in v)  # len('hello')=5


def test_embed_query_failure_returns_zero():
    bad = MagicMock()
    bad.encode.side_effect = RuntimeError("x")
    v = embed_query("hello", embedder=bad)
    assert v == [0.0] * 384


def test_search_vectors_empty_query():
    assert search_vectors(store=MagicMock(), query="") == []


def test_search_vectors_calls_store_with_filter():
    store = MagicMock()
    store.search_similar.return_value = [{"id": "x", "score": 0.9, "metadata": {}}]
    out = search_vectors(
        store=store,
        query="q",
        top_k=5,
        rag_namespace="ns1",
        embedder=_FakeEmbedder(),
    )
    assert len(out) == 1
    where = store.search_similar.call_args.kwargs["where"]
    assert where["memory_type"] == "rag_chunk"
    assert where["is_rag_data"] is True
    assert where["data_source"] == "rag_pipeline"
    assert where["rag_namespace"] == "ns1"


def test_search_vectors_only_rag_data_false_omits_flags():
    store = MagicMock()
    store.search_similar.return_value = []
    search_vectors(
        store=store,
        query="q",
        only_rag_data=False,
        embedder=_FakeEmbedder(),
    )
    where = store.search_similar.call_args.kwargs["where"]
    assert "is_rag_data" not in where
    assert "data_source" not in where


def test_search_vectors_failure_returns_empty():
    store = MagicMock()
    store.search_similar.side_effect = RuntimeError("boom")
    assert search_vectors(store=store, query="q", embedder=_FakeEmbedder()) == []


# ==================== Section K: MQE / HyDE / TL;DR（无 LLM 退化） ====================


def test_prompt_mqe_without_llm_returns_original():
    assert _prompt_mqe("query", 3) == ["query"]


def test_prompt_hyde_without_llm_returns_none():
    assert _prompt_hyde("query") is None


def test_tldr_summarize_without_llm_returns_none():
    assert tldr_summarize("text") is None


def test_tldr_summarize_empty_returns_none():
    assert tldr_summarize("") is None


def test_prompt_mqe_with_mock_llm():
    """mock LLM 返回 3 行 → 期待 3 个扩展查询"""
    llm = MagicMock()
    response = MagicMock()
    response.content = "expand 1\nexpand 2\nexpand 3"
    llm.invoke.return_value = response
    out = _prompt_mqe("orig", 3, llm=llm)
    assert out == ["expand 1", "expand 2", "expand 3"]


def test_prompt_mqe_llm_failure_falls_back():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("api down")
    assert _prompt_mqe("q", 3, llm=llm) == ["q"]


def test_prompt_hyde_with_mock_llm():
    llm = MagicMock()
    response = MagicMock()
    response.content = "假设答案段落"
    llm.invoke.return_value = response
    assert _prompt_hyde("q", llm=llm) == "假设答案段落"


# ==================== Section L: search_vectors_expanded ====================


def test_search_vectors_expanded_no_llm_uses_only_original_query():
    store = MagicMock()
    store.search_similar.return_value = [
        {"id": "h1", "score": 0.9, "metadata": {"memory_id": "m1"}}
    ]
    out = search_vectors_expanded(
        store=store,
        query="q",
        top_k=3,
        enable_mqe=True,  # 启用但没 llm → 退化
        enable_hyde=True,
        embedder=_FakeEmbedder(),
    )
    # 只有原 query 一次 search
    assert store.search_similar.call_count == 1
    assert len(out) == 1


def test_search_vectors_expanded_dedup_across_expansions():
    """多扩展间相同 memory_id 取最高分"""
    store = MagicMock()
    store.search_similar.side_effect = [
        [{"id": "h1", "score": 0.5, "metadata": {"memory_id": "m1"}}],
        [{"id": "h1", "score": 0.9, "metadata": {"memory_id": "m1"}}],
    ]
    llm = MagicMock()
    resp = MagicMock()
    resp.content = "alt query"
    llm.invoke.return_value = resp
    out = search_vectors_expanded(
        store=store,
        query="q",
        top_k=3,
        enable_mqe=True,
        mqe_expansions=1,
        llm=llm,
        embedder=_FakeEmbedder(),
    )
    # 去重后 1 条且取 0.9
    assert len(out) == 1
    assert out[0]["score"] == 0.9


# ==================== Section M: 重排 ====================


def test_rerank_no_cross_encoder_returns_topk():
    """sentence-transformers 没装 → 直接返回前 k 个"""
    items = [{"score": 0.9, "content": "a"}, {"score": 0.5, "content": "b"}]
    out = rerank_with_cross_encoder("q", items, top_k=1)
    assert len(out) == 1


def test_rerank_empty_returns_empty():
    assert rerank_with_cross_encoder("q", [], top_k=5) == []


# ==================== Section N: 图信号 + rank ====================


def test_compute_graph_signals_same_doc_density():
    hits = [
        {"id": "h1", "metadata": {"memory_id": "a", "doc_id": "d1", "start": 0}},
        {"id": "h2", "metadata": {"memory_id": "b", "doc_id": "d1", "start": 100}},
        {"id": "h3", "metadata": {"memory_id": "c", "doc_id": "d2", "start": 0}},
    ]
    sig = compute_graph_signals_from_pool(hits, proximity_window_chars=200)
    assert sig
    # d1 的两个 chunk 互为邻居（距离 100 < 200），分数应该不为零
    assert sig["a"] > 0
    assert sig["b"] > 0
    # 都归一化到 [0, 1]
    assert max(sig.values()) == pytest.approx(1.0)


def test_compute_graph_signals_empty_returns_empty():
    assert compute_graph_signals_from_pool([]) == {}


def test_rank_combines_vector_and_graph():
    hits = [
        {"id": "a", "metadata": {"memory_id": "a", "content": "A"}, "score": 0.6},
        {"id": "b", "metadata": {"memory_id": "b", "content": "B"}, "score": 0.6},
    ]
    out = rank(hits, graph_signals={"a": 1.0, "b": 0.0}, w_vector=0.5, w_graph=0.5)
    # a 的图分高，应排前
    assert out[0]["memory_id"] == "a"
    assert out[0]["score"] == pytest.approx(0.5 * 0.6 + 0.5 * 1.0)


def test_rank_no_graph_signals():
    hits = [{"id": "a", "metadata": {"memory_id": "a"}, "score": 0.9}]
    out = rank(hits)
    assert out[0]["score"] == pytest.approx(0.9 * 0.7)


# ==================== Section O: 合并 / 邻居 / 分组 ====================


def test_merge_snippets_concatenates():
    items = [{"content": "AAA"}, {"content": "BBB"}]
    assert merge_snippets(items, max_chars=100) == "AAA\n\nBBB"


def test_merge_snippets_truncates_at_limit():
    items = [{"content": "x" * 100}, {"content": "y" * 100}]
    out = merge_snippets(items, max_chars=50)
    assert len(out) <= 50


def test_merge_snippets_empty_items():
    assert merge_snippets([]) == ""


def test_expand_neighbors_adds_adjacent_chunks():
    selected = [{"memory_id": "m1", "metadata": {"doc_id": "d1"}}]
    pool = [
        {"memory_id": "m0", "metadata": {"doc_id": "d1", "start": 0}, "score": 0.5},
        {"memory_id": "m1", "metadata": {"doc_id": "d1", "start": 100}, "score": 0.9},
        {"memory_id": "m2", "metadata": {"doc_id": "d1", "start": 200}, "score": 0.5},
    ]
    out = expand_neighbors_from_pool(selected, pool, neighbors=1, max_additions=5)
    ids = {it.get("memory_id") for it in out}
    assert "m0" in ids and "m2" in ids


def test_expand_neighbors_no_op_when_neighbors_zero():
    selected = [{"memory_id": "m1", "metadata": {"doc_id": "d1"}}]
    pool = [{"memory_id": "m0", "metadata": {"doc_id": "d1"}}]
    out = expand_neighbors_from_pool(selected, pool, neighbors=0)
    assert out == selected


def test_merge_snippets_grouped_with_citations():
    items = [
        {
            "score": 0.9,
            "content": "alpha",
            "metadata": {"doc_id": "d1", "source_path": "a.md", "start": 0, "end": 5},
        },
        {
            "score": 0.5,
            "content": "beta",
            "metadata": {"doc_id": "d2", "source_path": "b.md", "start": 0, "end": 4},
        },
    ]
    out = merge_snippets_grouped(items, max_chars=200, include_citations=True)
    assert "alpha" in out
    assert "beta" in out
    assert "[1]" in out
    assert "References:" in out


def test_merge_snippets_grouped_no_citations():
    items = [
        {
            "score": 0.9,
            "content": "alpha",
            "metadata": {"doc_id": "d1", "source_path": "a.md", "start": 0, "end": 5},
        }
    ]
    out = merge_snippets_grouped(items, include_citations=False)
    assert "alpha" in out
    assert "[1]" not in out
    assert "References:" not in out


# ==================== Section P: compress_ranked_items ====================


def test_compress_disabled_passthrough():
    items = [{"metadata": {"doc_id": "d1", "start": 0, "end": 10}, "content": "a", "score": 1.0}]
    assert compress_ranked_items(items, enable_compression=False) == items


def test_compress_merges_close_chunks_in_same_doc():
    items = [
        {
            "metadata": {"doc_id": "d1", "start": 0, "end": 100},
            "content": "X",
            "score": 0.9,
        },
        {
            "metadata": {"doc_id": "d1", "start": 150, "end": 250},
            "content": "Y",
            "score": 0.5,
        },
    ]
    out = compress_ranked_items(items, max_per_doc=2, join_gap=200)
    # 合并后剩 1 条
    assert len(out) == 1
    assert "X" in out[0]["content"]
    assert "Y" in out[0]["content"]
    # score 取较大值
    assert out[0]["score"] == 0.9


def test_compress_respects_max_per_doc():
    items = [
        {
            "metadata": {"doc_id": "d1", "start": i * 10000, "end": i * 10000 + 100},
            "content": f"chunk{i}",
            "score": 1.0,
        }
        for i in range(5)
    ]
    out = compress_ranked_items(items, max_per_doc=2, join_gap=100)
    assert len(out) == 2


def test_compress_keeps_different_docs():
    items = [
        {
            "metadata": {"doc_id": f"d{i}", "start": 0, "end": 100},
            "content": f"X{i}",
            "score": 1.0,
        }
        for i in range(3)
    ]
    out = compress_ranked_items(items, max_per_doc=1, join_gap=100)
    assert len(out) == 3


# ==================== Section Q: _normalize_vec ====================


def test_normalize_vec_pad_with_zeros():
    assert _normalize_vec([1.0, 2.0], 4) == [1.0, 2.0, 0.0, 0.0]


def test_normalize_vec_truncate():
    assert _normalize_vec([1, 2, 3, 4, 5], 3) == [1.0, 2.0, 3.0]


def test_normalize_vec_exact_dimension():
    assert _normalize_vec([1, 2, 3], 3) == [1.0, 2.0, 3.0]


def test_normalize_vec_invalid_input_returns_zeros():
    assert _normalize_vec([1, "bad", 3], 3) == [0.0, 0.0, 0.0]


# ==================== Section R: 顶层导入 ====================


def test_top_level_rag_pipeline_imports():
    from clear_agent.retrieval.rag import (
        DEFAULT_RAG_COLLECTION,
        create_rag_pipeline,
        index_chunks,
        load_and_chunk_texts,
        rank,
        search_vectors,
    )

    assert DEFAULT_RAG_COLLECTION == "clear_agent_rag_vectors"
    assert callable(create_rag_pipeline)
    assert callable(index_chunks)
    assert callable(load_and_chunk_texts)
    assert callable(rank)
    assert callable(search_vectors)
