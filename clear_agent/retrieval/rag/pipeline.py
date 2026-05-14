"""RAG Pipeline —— 端到端检索增强生成流水线

Ported from AntonAgents (CC-BY-NC-SA-4.0)
Original: anton_agents/memory/rag/pipeline.py

完整覆盖 RAG 7 大职责：

1. **文档加载** —— MarkItDown 通用读取（PDF / DOCX / XLSX / PPTX / 图像 OCR /
   音频转写 / HTML / 代码 / 配置等），PDF 走增强后处理（短行合并 + 段落重组）
2. **分块** —— Markdown-aware 段落切分（按 ``#`` 标题保留结构）+ token 预算 +
   overlap 重叠窗口
3. **图谱集成（可选）** —— ``build_graph_from_chunks(neo4j, chunks)``
4. **索引** —— 批量 embedding（带小批重试 / 维度自动对齐 / 零向量兜底）+
   Qdrant upsert，所有 chunk 打 ``is_rag_data / rag_namespace / data_source`` 标签
5. **检索** —— 向量相似度搜索 + 可选 MQE（多查询扩展）+ HyDE（假设性回答检索）
6. **重排** —— Cross-encoder（``cross-encoder/ms-marco-MiniLM-L-6-v2``）+ 图信号
   融合（同文档密度 + 邻近度）+ 加权排序
7. **结果组装** —— Snippet 合并、邻居扩展、按文档分组带引用合并、
   压缩相邻片段、TL;DR 摘要

依赖（全部 optional）：

```bash
pip install clear-agent[retrieval-qdrant]   # 必需：QdrantVectorStore
pip install markitdown                      # 文档加载（推荐）
pip install langdetect                      # 语言检测（可选）
pip install sentence-transformers           # cross-encoder 重排（可选）
```

使用入口：

```python
from clear_agent.retrieval.rag import create_rag_pipeline

rag = create_rag_pipeline(qdrant_url="http://localhost:6333")
rag["add_documents"](["a.pdf", "b.md"])
hits = rag["search"]("query", top_k=5)
```

或散件直接用：``load_and_chunk_texts`` → ``index_chunks`` → ``search_vectors`` →
``rerank_with_cross_encoder`` → ``rank`` → ``merge_snippets_grouped``。

LLM 增强检索（MQE / HyDE / TL;DR）：原 AntonAgents 硬编码 ``AntonAgentsLLM()``，
本移植版改为**可选传入 ``llm`` 参数**（``ClearAgentLLM`` 实例），未传则降级（返回原始 query 或 None）。
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING, cast

from .document import Document, DocumentChunk  # noqa: F401  (re-export friendly)
from ..embeddings import get_dimension, get_text_embedder
from ..storage.qdrant_store import QdrantVectorStore

if TYPE_CHECKING:
    from ...core.llm import ClearAgentLLM


DEFAULT_RAG_COLLECTION = "clear_agent_rag_vectors"


# ==================================================================
# Section 1: 文档加载（MarkItDown / PDF 增强 / fallback）
# ==================================================================


def _get_markitdown_instance() -> Any:
    """获取 MarkItDown 实例；未装则返回 None"""
    try:
        from markitdown import MarkItDown  # type: ignore

        return MarkItDown()
    except ImportError:
        print("[WARNING] MarkItDown 未安装。安装：pip install markitdown")
        return None


_MARKITDOWN_SUPPORTED_EXTS = {
    # Documents
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    # Text
    ".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm",
    # Images (OCR + metadata)
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp",
    # Audio (transcription + metadata)
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
    # Archives
    ".zip", ".tar", ".gz", ".rar",
    # Code
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".css", ".scss",
    # Config / log
    ".log", ".conf", ".ini", ".cfg", ".yaml", ".yml", ".toml",
}


def _is_markitdown_supported_format(path: str) -> bool:
    """文件后缀是否在 MarkItDown 支持范围内"""
    ext = (os.path.splitext(path)[1] or "").lower()
    return ext in _MARKITDOWN_SUPPORTED_EXTS


def _convert_to_markdown(path: str) -> str:
    """通用文档读取：转为 markdown 文本

    PDF 走 ``_enhanced_pdf_processing``；其他格式走 MarkItDown；
    MarkItDown 未装时降级 ``_fallback_text_reader``。
    """
    if not os.path.exists(path):
        return ""

    ext = (os.path.splitext(path)[1] or "").lower()
    if ext == ".pdf":
        return _enhanced_pdf_processing(path)

    md = _get_markitdown_instance()
    if md is None:
        return _fallback_text_reader(path)

    try:
        result = md.convert(path)
        text = getattr(result, "text_content", None)
        if isinstance(text, str) and text.strip():
            return text
        return ""
    except Exception as e:
        print(f"[WARNING] MarkItDown failed for {path}: {e}")
        return _fallback_text_reader(path)


def _enhanced_pdf_processing(path: str) -> str:
    """PDF 专用：MarkItDown 提取 + 后处理（行合并 + 段落重组）"""
    print(f"[RAG] Using enhanced PDF processing for: {path}")
    md = _get_markitdown_instance()
    if md is None:
        return _fallback_text_reader(path)

    try:
        result = md.convert(path)
        raw_text = getattr(result, "text_content", None)
        if not raw_text or not raw_text.strip():
            return ""
        cleaned = _post_process_pdf_text(raw_text)
        print(f"[RAG] PDF post-processing: {len(raw_text)} -> {len(cleaned)} chars")
        return cleaned
    except Exception as e:
        print(f"[WARNING] Enhanced PDF processing failed for {path}: {e}")
        return _fallback_text_reader(path)


def _post_process_pdf_text(text: str) -> str:
    """PDF 文本后处理：去噪音 + 智能合并短行 + 段落重组"""
    import re

    # 1. 按行清理
    lines = text.splitlines()
    cleaned_lines: List[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) <= 2 and not line.isdigit():
            continue  # 单字符噪音
        if re.match(r"^\d+$", line):
            continue  # 纯数字（页码）
        if line.lower() in ["github", "project", "forks", "stars", "language"]:
            continue
        cleaned_lines.append(line)

    # 2. 智能合并短行
    merged_lines: List[str] = []
    i = 0
    while i < len(cleaned_lines):
        cur = cleaned_lines[i]
        if len(cur) < 60 and i + 1 < len(cleaned_lines):
            nxt = cleaned_lines[i + 1]
            if (
                not cur.endswith("：")
                and not cur.endswith(":")
                and not cur.startswith("#")
                and not nxt.startswith("#")
                and len(nxt) < 120
            ):
                merged_lines.append(cur + " " + nxt)
                i += 2
                continue
        merged_lines.append(cur)
        i += 1

    # 3. 段落重组
    paragraphs: List[str] = []
    current: List[str] = []
    for line in merged_lines:
        if (
            line.startswith("#")
            or line.endswith("：")
            or line.endswith(":")
            or len(line) > 150
            or not current
        ):
            if current:
                paragraphs.append(" ".join(current))
                current = []
            paragraphs.append(line)
        else:
            current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def _fallback_text_reader(path: str) -> str:
    """简易文本读取（utf-8 → latin-1 兜底）"""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        try:
            with open(path, "r", encoding="latin-1", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""


# ==================================================================
# Section 2: 语言检测 + 近似 token 计数
# ==================================================================


def _detect_lang(sample: str) -> str:
    """语言检测；未装 langdetect 或失败时返回 ``"unknown"``"""
    try:
        from langdetect import detect  # type: ignore

        return detect(sample[:1000]) if sample else "unknown"
    except Exception:
        return "unknown"


def _is_cjk(ch: str) -> bool:
    """是否为 CJK 统一汉字（含扩展平面）"""
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


def _approx_token_len(text: str) -> int:
    """近似 token 数：CJK 字符按 1，其他按空白分词"""
    cjk = sum(1 for ch in text if _is_cjk(ch))
    non_cjk = len([t for t in text.split() if t])
    return cjk + non_cjk


# ==================================================================
# Section 3: Markdown-aware 分块
# ==================================================================


def _split_paragraphs_with_headings(text: str) -> List[Dict]:
    """按 markdown 标题切段，记录段落 ``heading_path / start / end``"""
    lines = text.splitlines()
    heading_stack: List[str] = []
    paragraphs: List[Dict] = []
    buf: List[str] = []
    char_pos = 0

    def flush_buf(end_pos: int) -> None:
        if not buf:
            return
        content = "\n".join(buf).strip()
        if not content:
            return
        paragraphs.append(
            {
                "content": content,
                "heading_path": " > ".join(heading_stack) if heading_stack else None,
                "start": max(0, end_pos - len(content)),
                "end": end_pos,
            }
        )

    for ln in lines:
        if ln.strip().startswith("#"):
            flush_buf(char_pos)
            level = len(ln) - len(ln.lstrip("#"))
            title = ln.lstrip("#").strip()
            if level <= 0:
                level = 1
            if level <= len(heading_stack):
                heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            char_pos += len(ln) + 1
            continue
        if ln.strip() == "":
            flush_buf(char_pos)
            buf = []
        else:
            buf.append(ln)
        char_pos += len(ln) + 1
    flush_buf(char_pos)

    if not paragraphs:
        paragraphs = [
            {"content": text, "heading_path": None, "start": 0, "end": len(text)}
        ]
    return paragraphs


def _chunk_paragraphs(
    paragraphs: List[Dict], chunk_tokens: int, overlap_tokens: int
) -> List[Dict]:
    """段落 → 块（按 token 预算 + overlap）"""
    chunks: List[Dict] = []
    cur: List[Dict] = []
    cur_tokens = 0
    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        p_tokens = _approx_token_len(p["content"]) or 1
        if cur_tokens + p_tokens <= chunk_tokens or not cur:
            cur.append(p)
            cur_tokens += p_tokens
            i += 1
        else:
            content = "\n\n".join(x["content"] for x in cur)
            start = cur[0]["start"]
            end = cur[-1]["end"]
            heading_path = next(
                (x["heading_path"] for x in reversed(cur) if x.get("heading_path")),
                None,
            )
            chunks.append(
                {
                    "content": content,
                    "start": start,
                    "end": end,
                    "heading_path": heading_path,
                }
            )
            # build overlap by keeping tail tokens
            if overlap_tokens > 0 and cur:
                kept: List[Dict] = []
                kept_tokens = 0
                for x in reversed(cur):
                    t = _approx_token_len(x["content"]) or 1
                    if kept_tokens + t > overlap_tokens:
                        break
                    kept.append(x)
                    kept_tokens += t
                cur = list(reversed(kept))
                cur_tokens = kept_tokens
            else:
                cur = []
                cur_tokens = 0
    if cur:
        content = "\n\n".join(x["content"] for x in cur)
        start = cur[0]["start"]
        end = cur[-1]["end"]
        heading_path = next(
            (x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None
        )
        chunks.append(
            {
                "content": content,
                "start": start,
                "end": end,
                "heading_path": heading_path,
            }
        )
    return chunks


def load_and_chunk_texts(
    paths: List[str],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    namespace: Optional[str] = None,
    source_label: str = "rag",
) -> List[Dict]:
    """通用文档加载 + 分块（markdown-aware，去重）

    Args:
        paths: 文件路径列表
        chunk_size: 单块 token 预算（近似）
        chunk_overlap: 相邻块重叠 token 数
        namespace: RAG 命名空间（写入 metadata）
        source_label: 元数据 ``source`` 字段

    Returns:
        chunks 列表，每条形如 ``{"id":..., "content":..., "metadata":{...}}``。
        metadata 含 ``source_path / file_ext / doc_id / lang / start / end /
        content_hash / namespace / source / external / heading_path / format``。
    """
    print(
        f"[RAG] Universal loader start: files={len(paths)} "
        f"chunk_size={chunk_size} overlap={chunk_overlap} ns={namespace or 'default'}"
    )
    chunks: List[Dict] = []
    seen_hashes: set = set()

    for path in paths:
        if not os.path.exists(path):
            print(f"[WARNING] File not found: {path}")
            continue

        print(f"[RAG] Processing: {path}")
        ext = (os.path.splitext(path)[1] or "").lower()

        markdown_text = _convert_to_markdown(path)
        if not markdown_text.strip():
            print(f"[WARNING] No content extracted from: {path}")
            continue

        lang = _detect_lang(markdown_text)
        doc_id = hashlib.md5(
            f"{path}|{len(markdown_text)}".encode("utf-8")
        ).hexdigest()

        para = _split_paragraphs_with_headings(markdown_text)
        token_chunks = _chunk_paragraphs(
            para, chunk_tokens=max(1, chunk_size), overlap_tokens=max(0, chunk_overlap)
        )

        for ch in token_chunks:
            content = ch["content"]
            start = ch.get("start", 0)
            end = ch.get("end", start + len(content))
            norm = content.strip()
            if not norm:
                continue
            content_hash = hashlib.md5(norm.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            chunk_id = hashlib.md5(
                f"{doc_id}|{start}|{end}|{content_hash}".encode("utf-8")
            ).hexdigest()
            chunks.append(
                {
                    "id": chunk_id,
                    "content": content,
                    "metadata": {
                        "source_path": path,
                        "file_ext": ext,
                        "doc_id": doc_id,
                        "lang": lang,
                        "start": start,
                        "end": end,
                        "content_hash": content_hash,
                        "namespace": namespace or "default",
                        "source": source_label,
                        "external": True,
                        "heading_path": ch.get("heading_path"),
                        "format": "markdown",
                    },
                }
            )

    print(f"[RAG] Universal loader done: total_chunks={len(chunks)}")
    return chunks


# ==================================================================
# Section 4: 图谱集成（Neo4j 可选）
# ==================================================================


def build_graph_from_chunks(neo4j: Any, chunks: List[Dict]) -> None:
    """把 chunks 写入图谱：每个 doc 一个 Document 节点，每个 chunk 一个 Memory 节点

    ``neo4j`` 期望提供 ``add_entity(entity_id, name, entity_type, properties)``
    和 ``add_relationship(from_id, to_id, rel_type, properties)`` 接口。
    """
    created_docs: set = set()
    for ch in chunks:
        mem_id = ch["id"]
        meta = ch.get("metadata", {})
        source_path = meta.get("source_path")
        doc_id = meta.get("doc_id")
        if doc_id and doc_id not in created_docs:
            created_docs.add(doc_id)
            try:
                neo4j.add_entity(
                    entity_id=doc_id,
                    name=os.path.basename(source_path or doc_id),
                    entity_type="Document",
                    properties={
                        "source_path": source_path,
                        "lang": meta.get("lang"),
                    },
                )
            except Exception:
                pass
        try:
            neo4j.add_entity(
                entity_id=mem_id,
                name=mem_id,
                entity_type="Memory",
                properties={
                    "source_path": source_path,
                    "doc_id": doc_id,
                    "start": meta.get("start"),
                    "end": meta.get("end"),
                },
            )
        except Exception:
            pass
        if doc_id:
            try:
                neo4j.add_relationship(
                    from_id=doc_id,
                    to_id=mem_id,
                    rel_type="HAS_CHUNK",
                    properties={},
                )
            except Exception:
                pass


# ==================================================================
# Section 5: 索引（embedding 预处理 + 批 upsert）
# ==================================================================


def _preprocess_markdown_for_embedding(text: str) -> str:
    """嵌入前去除 markdown markup，保留语义内容"""
    import re

    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # links
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # bold
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # italic
    text = re.sub(r"`([^`]+)`", r"\1", text)  # inline code
    text = re.sub(r"```[^\n]*\n([\s\S]*?)```", r"\1", text)  # code blocks
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _create_default_vector_store(dimension: Optional[int] = None) -> QdrantVectorStore:
    """创建默认 Qdrant 向量库（走 ConnectionManager 单例）"""
    if dimension is None:
        dimension = get_dimension(384)
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    from ..storage.qdrant_store import QdrantConnectionManager

    return QdrantConnectionManager.get_instance(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=DEFAULT_RAG_COLLECTION,
        vector_size=dimension,
        distance="cosine",
    )


def _embedder_dimension(embedder: Any, default: int = 384) -> int:
    """Prefer an explicitly supplied embedder's dimension over the global singleton."""
    dim = getattr(embedder, "dimension", None)
    if isinstance(dim, int) and dim > 0:
        return dim
    return int(get_dimension(default))


def _normalize_vec(v: Any, dimension: int) -> List[float]:
    """numpy / list / tuple → ``List[float]``，并对齐维度（不足填 0，超长截断）"""
    if hasattr(v, "tolist"):
        v = v.tolist()
    try:
        out = [float(x) for x in v]
    except Exception:
        return [0.0] * dimension
    if len(out) < dimension:
        out.extend([0.0] * (dimension - len(out)))
    elif len(out) > dimension:
        out = out[:dimension]
    return out


def index_chunks(
    store: Optional[QdrantVectorStore] = None,
    chunks: Optional[List[Dict]] = None,
    cache_db: Optional[str] = None,  # noqa: ARG001  (保留参数兼容性)
    batch_size: int = 64,
    rag_namespace: str = "default",
    embedder: Optional[Any] = None,
) -> None:
    """对 chunks 批量 embedding 并写入 Qdrant

    Args:
        store: ``QdrantVectorStore``；缺省时走默认连接
        chunks: ``load_and_chunk_texts`` 输出格式
        batch_size: embedding 批大小（失败自动小批重试）
        rag_namespace: 写入 metadata.rag_namespace
        embedder: 可选自定义 embedder；缺省用全局 ``get_text_embedder()``
    """
    if not chunks:
        print("[RAG] No chunks to index")
        return

    embedder = embedder or get_text_embedder()
    dimension = _embedder_dimension(embedder, 384)

    if store is None:
        store = _create_default_vector_store(dimension)
        print(f"[RAG] Created default Qdrant store with dimension {dimension}")

    processed_texts = [_preprocess_markdown_for_embedding(c["content"]) for c in chunks]
    print(
        f"[RAG] Embedding start: total={len(processed_texts)} batch_size={batch_size}"
    )

    vecs: List[List[float]] = []
    for i in range(0, len(processed_texts), batch_size):
        part = processed_texts[i : i + batch_size]
        try:
            part_vecs = embedder.encode(part)
            # 标准化为 List[List[float]]
            if not isinstance(part_vecs, list):
                if hasattr(part_vecs, "tolist"):
                    part_vecs = [part_vecs.tolist()]
                else:
                    part_vecs = [list(part_vecs)]
            else:
                # 处理 numpy array of vectors
                if (
                    part_vecs
                    and not isinstance(part_vecs[0], (list, tuple))
                    and hasattr(part_vecs[0], "__len__")
                ):
                    part_vecs = [
                        v.tolist() if hasattr(v, "tolist") else list(v)
                        for v in part_vecs
                    ]
                elif part_vecs and not isinstance(part_vecs[0], (list, tuple)):
                    if hasattr(part_vecs, "tolist"):
                        part_vecs = [part_vecs.tolist()]
                    else:
                        part_vecs = [list(part_vecs)]

            for v in part_vecs:
                vecs.append(_normalize_vec(v, dimension))
        except Exception as e:
            print(f"[WARNING] Batch {i} encoding failed: {e}; retrying small batches")
            # 重试：拆小批
            success = False
            for j in range(0, len(part), 8):
                small_part = part[j : j + 8]
                try:
                    import time

                    time.sleep(2)
                    small_vecs = embedder.encode(small_part)
                    if (
                        isinstance(small_vecs, list)
                        and small_vecs
                        and not isinstance(small_vecs[0], list)
                    ):
                        small_vecs = [small_vecs]
                    for v in small_vecs:
                        vecs.append(_normalize_vec(v, dimension))
                        success = True
                except Exception as e2:
                    print(f"[WARNING] 小批次 {j // 8} 仍失败: {e2}")
                    for _ in range(len(small_part)):
                        vecs.append([0.0] * dimension)
            if not success:
                print(f"[ERROR] 批次 {i} 完全失败，使用零向量")

        print(
            f"[RAG] Embedding progress: "
            f"{min(i + batch_size, len(processed_texts))}/{len(processed_texts)}"
        )

    # 准备 metadata + ids
    metas: List[Dict] = []
    ids: List[str] = []
    for ch in chunks:
        meta = {
            "memory_id": ch["id"],
            "user_id": "rag_user",
            "memory_type": "rag_chunk",
            "content": ch["content"],
            "data_source": "rag_pipeline",
            "rag_namespace": rag_namespace,
            "is_rag_data": True,
        }
        meta.update(ch.get("metadata", {}))
        metas.append(meta)
        ids.append(ch["id"])

    print(f"[RAG] Qdrant upsert start: n={len(vecs)}")
    success = store.add_vectors(vectors=vecs, metadata=metas, ids=ids)
    if success:
        print(f"[RAG] Qdrant upsert done: {len(vecs)} vectors indexed")
    else:
        raise RuntimeError("Failed to index vectors to Qdrant")


# ==================================================================
# Section 6: 检索（基础 + MQE + HyDE）
# ==================================================================


def embed_query(query: str, embedder: Optional[Any] = None) -> List[float]:
    """对查询文本做 embedding；失败返回零向量"""
    embedder = embedder or get_text_embedder()
    dimension = _embedder_dimension(embedder, 384)
    try:
        vec = embedder.encode(query)
        if hasattr(vec, "tolist"):
            vec = vec.tolist()
        if isinstance(vec, list) and vec and isinstance(vec[0], (list, tuple)):
            vec = vec[0]
        return _normalize_vec(vec, dimension)
    except Exception as e:
        print(f"[WARNING] Query embedding failed: {e}")
        return [0.0] * dimension


def search_vectors(
    store: Optional[QdrantVectorStore] = None,
    query: str = "",
    top_k: int = 8,
    rag_namespace: Optional[str] = None,
    only_rag_data: bool = True,
    score_threshold: Optional[float] = None,
    embedder: Optional[Any] = None,
) -> List[Dict]:
    """RAG 向量检索 + payload 过滤"""
    if not query:
        return []
    if store is None:
        store = _create_default_vector_store()
    qv = embed_query(query, embedder=embedder)

    where: Dict[str, Any] = {"memory_type": "rag_chunk"}
    if only_rag_data:
        where["is_rag_data"] = True
        where["data_source"] = "rag_pipeline"
    if rag_namespace:
        where["rag_namespace"] = rag_namespace

    try:
        return cast(
            List[Dict[str, Any]],
            store.search_similar(
                query_vector=qv,
                limit=top_k,
                score_threshold=score_threshold,
                where=where,
            ),
        )
    except Exception as e:
        print(f"[WARNING] RAG search failed: {e}")
        return []


def _prompt_mqe(
    query: str, n: int, llm: Optional["ClearAgentLLM"] = None
) -> List[str]:
    """Multi-query expansion: 生成 ``n`` 个等价 / 互补查询

    传 ``llm`` 才会真扩展；缺省返回 ``[query]``。
    """
    if llm is None:
        return [query]
    try:
        prompt = [
            {
                "role": "system",
                "content": (
                    "你是检索查询扩展助手。生成语义等价或互补的多样化查询。"
                    "使用中文，简短，避免标点。"
                ),
            },
            {
                "role": "user",
                "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。",
            },
        ]
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        lines = [ln.strip("- \t") for ln in (text or "").splitlines()]
        outs = [ln for ln in lines if ln]
        return outs[:n] or [query]
    except Exception:
        return [query]


def _prompt_hyde(
    query: str, llm: Optional["ClearAgentLLM"] = None
) -> Optional[str]:
    """HyDE: 让 LLM 写一段假设性答案用于向量检索

    传 ``llm`` 才生效；缺省返回 ``None``。
    """
    if llm is None:
        return None
    try:
        prompt = [
            {
                "role": "system",
                "content": (
                    "根据用户问题，先写一段可能的答案性段落，用于向量检索的"
                    "查询文档（不要分析过程）。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。",
            },
        ]
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception:
        return None


def search_vectors_expanded(
    store: Optional[QdrantVectorStore] = None,
    query: str = "",
    top_k: int = 8,
    rag_namespace: Optional[str] = None,
    only_rag_data: bool = True,
    score_threshold: Optional[float] = None,
    enable_mqe: bool = False,
    mqe_expansions: int = 2,
    enable_hyde: bool = False,
    candidate_pool_multiplier: int = 4,
    llm: Optional["ClearAgentLLM"] = None,
    embedder: Optional[Any] = None,
) -> List[Dict]:
    """带查询扩展的检索（MQE + HyDE）

    Args:
        enable_mqe / enable_hyde: 是否启用对应扩展（需要传 ``llm`` 才生效）
        mqe_expansions: MQE 生成几个变体
        candidate_pool_multiplier: 每个扩展查询的召回 pool 大小倍数
    """
    if not query:
        return []
    if store is None:
        store = _create_default_vector_store()

    expansions: List[str] = [query]
    if enable_mqe and mqe_expansions > 0:
        expansions.extend(_prompt_mqe(query, mqe_expansions, llm=llm))
    if enable_hyde:
        hyde_text = _prompt_hyde(query, llm=llm)
        if hyde_text:
            expansions.append(hyde_text)

    # 去重保序
    uniq: List[str] = []
    for e in expansions:
        if e and e not in uniq:
            uniq.append(e)
    expansions = uniq[: max(1, len(uniq))]

    pool = max(top_k * candidate_pool_multiplier, 20)
    per = max(1, pool // max(1, len(expansions)))

    where: Dict[str, Any] = {"memory_type": "rag_chunk"}
    if only_rag_data:
        where["is_rag_data"] = True
        where["data_source"] = "rag_pipeline"
    if rag_namespace:
        where["rag_namespace"] = rag_namespace

    agg: Dict[str, Dict] = {}
    for q in expansions:
        qv = embed_query(q, embedder=embedder)
        hits = store.search_similar(
            query_vector=qv,
            limit=per,
            score_threshold=score_threshold,
            where=where,
        )
        for h in hits:
            mid = h.get("metadata", {}).get("memory_id", h.get("id"))
            s = float(h.get("score", 0.0))
            if mid not in agg or s > float(agg[mid].get("score", 0.0)):
                agg[mid] = h

    merged = list(agg.values())
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged[:top_k]


# ==================================================================
# Section 7: 重排（cross-encoder + 图信号融合）
# ==================================================================


def _try_load_cross_encoder(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> Any:
    """加载 cross-encoder；未装 sentence-transformers 返回 None"""
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        return CrossEncoder(model_name)
    except Exception:
        return None


def rerank_with_cross_encoder(
    query: str,
    items: List[Dict],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 10,
) -> List[Dict]:
    """对召回结果做 cross-encoder 重排；模型不可用时直接返回前 ``top_k`` 个"""
    ce = _try_load_cross_encoder(model_name)
    if ce is None or not items:
        return items[:top_k]
    pairs = [[query, it.get("content", "")] for it in items]
    try:
        scores = ce.predict(pairs)
        for it, s in zip(items, scores):
            it["rerank_score"] = float(s)
        items.sort(
            key=lambda x: x.get("rerank_score", x.get("score", 0.0)),
            reverse=True,
        )
        return items[:top_k]
    except Exception:
        return items[:top_k]


def compute_graph_signals_from_pool(
    vector_hits: List[Dict],
    same_doc_weight: float = 1.0,
    proximity_weight: float = 1.0,
    proximity_window_chars: int = 1600,
) -> Dict[str, float]:
    """从召回 pool 计算图信号：同文档密度 + 邻近度，归一化到 [0, 1]"""
    by_doc: Dict[str, List[Dict]] = {}
    for h in vector_hits:
        meta = h.get("metadata", {})
        did = meta.get("doc_id") or meta.get("memory_id") or h.get("id")
        if did is None:
            continue
        did = str(did)
        by_doc.setdefault(did, []).append(h)

    doc_counts = {d: len(arr) for d, arr in by_doc.items()}
    max_count = max(doc_counts.values()) if doc_counts else 1

    graph_signal: Dict[str, float] = {}
    for did, arr in by_doc.items():
        arr.sort(key=lambda x: x.get("metadata", {}).get("start", 0))
        density = doc_counts.get(did, 1) / max_count
        for i, h in enumerate(arr):
            mid = h.get("metadata", {}).get("memory_id", h.get("id"))
            pos_i = h.get("metadata", {}).get("start", 0)
            prox_acc = 0.0
            # 左邻居
            j = i - 1
            while j >= 0:
                pos_j = arr[j].get("metadata", {}).get("start", 0)
                dist = abs(pos_i - pos_j)
                if dist > proximity_window_chars:
                    break
                prox_acc += max(
                    0.0, 1.0 - (dist / max(1.0, float(proximity_window_chars)))
                )
                j -= 1
            # 右邻居
            j = i + 1
            while j < len(arr):
                pos_j = arr[j].get("metadata", {}).get("start", 0)
                dist = abs(pos_i - pos_j)
                if dist > proximity_window_chars:
                    break
                prox_acc += max(
                    0.0, 1.0 - (dist / max(1.0, float(proximity_window_chars)))
                )
                j += 1
            score = same_doc_weight * density + proximity_weight * prox_acc
            graph_signal[mid] = graph_signal.get(mid, 0.0) + score

    if graph_signal:
        max_v = max(graph_signal.values())
        if max_v > 0:
            for k in list(graph_signal.keys()):
                graph_signal[k] = graph_signal[k] / max_v
    return graph_signal


def rank(
    vector_hits: List[Dict],
    graph_signals: Optional[Dict[str, float]] = None,
    w_vector: float = 0.7,
    w_graph: float = 0.3,
) -> List[Dict]:
    """向量分数与图分数加权融合排序"""
    items: List[Dict] = []
    graph_signals = graph_signals or {}
    for h in vector_hits:
        mid = h.get("metadata", {}).get("memory_id", h.get("id"))
        g = float(graph_signals.get(mid, 0.0))
        v = float(h.get("score", 0.0))
        score = w_vector * v + w_graph * g
        items.append(
            {
                "memory_id": mid,
                "score": score,
                "vector_score": v,
                "graph_score": g,
                "content": h.get("metadata", {}).get("content", ""),
                "metadata": h.get("metadata", {}),
            }
        )
    items.sort(key=lambda x: x["score"], reverse=True)
    return items


# ==================================================================
# Section 8: 结果组装（合并 / 邻居扩展 / 分组带引用 / 压缩）
# ==================================================================


def merge_snippets(ranked_items: List[Dict], max_chars: int = 1200) -> str:
    """简单按顺序拼接，不超过 ``max_chars``"""
    out: List[str] = []
    total = 0
    for it in ranked_items:
        text = it.get("content", "").strip()
        if not text:
            continue
        if total + len(text) > max_chars:
            remain = max_chars - total
            if remain <= 0:
                break
            out.append(text[:remain])
            total += remain
            break
        out.append(text)
        total += len(text)
    return "\n\n".join(out)


def expand_neighbors_from_pool(
    selected: List[Dict],
    pool: List[Dict],
    neighbors: int = 1,
    max_additions: int = 5,
) -> List[Dict]:
    """从召回 pool 给每个 selected 块补邻居块（同 doc，按 start 排序）"""
    if not selected or not pool or neighbors <= 0:
        return selected

    by_doc: Dict[str, List[Dict]] = {}
    for it in pool:
        meta = it.get("metadata", {})
        did = meta.get("doc_id")
        if not did:
            continue
        by_doc.setdefault(did, []).append(it)
    for did, arr in by_doc.items():
        arr.sort(key=lambda x: x.get("metadata", {}).get("start", 0))

    selected_ids = set(it.get("memory_id") for it in selected)
    additions: List[Dict] = []
    for it in selected:
        meta = it.get("metadata", {})
        did = meta.get("doc_id")
        if not did or did not in by_doc:
            continue
        arr = by_doc[did]
        try:
            idx = next(
                i
                for i, x in enumerate(arr)
                if x.get("memory_id") == it.get("memory_id")
            )
        except StopIteration:
            continue
        for offset in range(1, neighbors + 1):
            for j in (idx - offset, idx + offset):
                if 0 <= j < len(arr):
                    cand = arr[j]
                    mid = cand.get("memory_id")
                    if mid not in selected_ids:
                        additions.append(cand)
                        selected_ids.add(mid)
                        if len(additions) >= max_additions:
                            break
            if len(additions) >= max_additions:
                break
        if len(additions) >= max_additions:
            break

    extended = list(selected) + additions
    extended.sort(
        key=lambda x: x.get("rerank_score", x.get("score", 0.0)),
        reverse=True,
    )
    return extended


def merge_snippets_grouped(
    ranked_items: List[Dict],
    max_chars: int = 1200,
    include_citations: bool = True,
) -> str:
    """按 doc 分组合并，可选附引用尾注"""
    by_doc: Dict[str, List[Dict]] = {}
    doc_score: Dict[str, float] = {}
    for it in ranked_items:
        meta = it.get("metadata", {})
        did = meta.get("doc_id") or meta.get("source_path") or "unknown"
        by_doc.setdefault(did, []).append(it)
        doc_score[did] = doc_score.get(did, 0.0) + float(it.get("score", 0.0))

    ordered_docs = sorted(
        by_doc.keys(), key=lambda d: doc_score.get(d, 0.0), reverse=True
    )
    for d in ordered_docs:
        by_doc[d].sort(key=lambda x: x.get("metadata", {}).get("start", 0))

    out: List[str] = []
    citations: List[Dict] = []
    total = 0
    cite_index = 1
    for did in ordered_docs:
        for it in by_doc[did]:
            text = (it.get("content", "") or "").strip()
            if not text:
                continue
            suffix = f" [{cite_index}]" if include_citations else ""
            need = len(text) + len(suffix)
            if total + need > max_chars:
                remain = max_chars - total
                if remain <= 0:
                    break
                clipped = text[: max(0, remain - len(suffix))]
                if clipped:
                    out.append(clipped + suffix)
                    total += len(clipped) + len(suffix)
                    if include_citations:
                        m = it.get("metadata", {})
                        citations.append(
                            {
                                "index": cite_index,
                                "source_path": m.get("source_path"),
                                "doc_id": m.get("doc_id"),
                                "start": m.get("start"),
                                "end": m.get("end"),
                                "heading_path": m.get("heading_path"),
                            }
                        )
                        cite_index += 1
                break
            out.append(text + suffix)
            total += need
            if include_citations:
                m = it.get("metadata", {})
                citations.append(
                    {
                        "index": cite_index,
                        "source_path": m.get("source_path"),
                        "doc_id": m.get("doc_id"),
                        "start": m.get("start"),
                        "end": m.get("end"),
                        "heading_path": m.get("heading_path"),
                    }
                )
                cite_index += 1
        if total >= max_chars:
            break

    merged = "\n\n".join(out)
    if include_citations and citations:
        lines: List[str] = [merged, "", "References:"]
        for c in citations:
            loc = ""
            if c.get("start") is not None and c.get("end") is not None:
                loc = f" ({c['start']}-{c['end']})"
            hp = f" – {c['heading_path']}" if c.get("heading_path") else ""
            sp = c.get("source_path") or c.get("doc_id") or "source"
            lines.append(f"[{c['index']}] {sp}{loc}{hp}")
        return "\n".join(lines)
    return merged


def compress_ranked_items(
    ranked_items: List[Dict],
    enable_compression: bool = True,
    max_per_doc: int = 2,
    join_gap: int = 200,
) -> List[Dict]:
    """合并相邻片段（同 doc 且间距 ≤ ``join_gap``）+ 限制每个 doc 最多 ``max_per_doc`` 条"""
    if not enable_compression:
        return ranked_items
    by_doc_count: Dict[str, int] = {}
    last_by_doc: Dict[str, Dict] = {}
    new_items: List[Dict] = []

    for it in ranked_items:
        meta = it.get("metadata", {})
        did = meta.get("doc_id") or meta.get("source_path") or "unknown"
        start = int(meta.get("start") or 0)
        end = int(meta.get("end") or (start + len(it.get("content", "") or "")))
        if did not in last_by_doc:
            last_by_doc[did] = it
            by_doc_count[did] = 1
            new_items.append(it)
            continue
        last = last_by_doc[did]
        lmeta = last.get("metadata", {})
        lstart = int(lmeta.get("start") or 0)
        lend = int(lmeta.get("end") or (lstart + len(last.get("content", "") or "")))
        if start - lend <= join_gap and start >= lstart:
            # 合并到 last
            merged_text = (last.get("content", "") or "").strip()
            add_text = (it.get("content", "") or "").strip()
            if add_text:
                merged_text = (merged_text + "\n\n" + add_text) if merged_text else add_text
                last["content"] = merged_text
                lmeta["end"] = max(lend, end)
                try:
                    last["score"] = max(
                        float(last.get("score", 0.0)),
                        float(it.get("score", 0.0)),
                    )
                except Exception:
                    pass
            last_by_doc[did] = last
        else:
            cnt = by_doc_count.get(did, 0)
            if cnt >= max_per_doc:
                continue
            new_items.append(it)
            last_by_doc[did] = it
            by_doc_count[did] = cnt + 1
    return new_items


def tldr_summarize(
    text: str,
    bullets: int = 3,
    llm: Optional["ClearAgentLLM"] = None,
) -> Optional[str]:
    """LLM TL;DR 摘要；未传 ``llm`` 或失败返回 None"""
    if not text or len(text.strip()) == 0:
        return None
    if llm is None:
        return None
    try:
        prompt = [
            {
                "role": "system",
                "content": (
                    "请将以下内容概括为简洁的要点列表（最多3-5条），"
                    "用中文，避免重复，突出关键信息。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"请用 {max(1, min(5, int(bullets)))} 条要点总结：\n\n{text}"
                ),
            },
        ]
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception:
        return None


# ==================================================================
# Section 9: 高层 Pipeline 工厂
# ==================================================================


def create_rag_pipeline(
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    collection_name: str = DEFAULT_RAG_COLLECTION,
    rag_namespace: str = "default",
    llm: Optional["ClearAgentLLM"] = None,
    embedder: Optional[Any] = None,
) -> Dict[str, Any]:
    """创建端到端 RAG pipeline，返回 namespace dict（含 store + 4 个 helper）

    Args:
        qdrant_url / qdrant_api_key: Qdrant 连接（缺省走 localhost）
        collection_name: 向量库集合名
        rag_namespace: 该 pipeline 的 RAG 命名空间（写入 metadata，便于多租户隔离）
        llm: 可选 ``ClearAgentLLM``，传入即可启用 MQE / HyDE / TL;DR
        embedder: 可选自定义 embedder；缺省走全局 ``get_text_embedder()``

    Returns:
        ``{"store", "namespace", "add_documents", "search", "search_advanced",
        "get_stats", "rerank", "summarize"}``
    """
    active_embedder = embedder or get_text_embedder()
    dimension = _embedder_dimension(active_embedder, 384)
    store = QdrantVectorStore(
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
        vector_size=dimension,
        distance="cosine",
    )

    def add_documents(
        file_paths: List[str], chunk_size: int = 800, chunk_overlap: int = 100
    ) -> int:
        chunks = load_and_chunk_texts(
            paths=file_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            namespace=rag_namespace,
            source_label="rag",
        )
        index_chunks(
            store=store,
            chunks=chunks,
            rag_namespace=rag_namespace,
            embedder=active_embedder,
        )
        return len(chunks)

    def search(
        query: str, top_k: int = 8, score_threshold: Optional[float] = None
    ) -> List[Dict]:
        return search_vectors(
            store=store,
            query=query,
            top_k=top_k,
            rag_namespace=rag_namespace,
            score_threshold=score_threshold,
            embedder=active_embedder,
        )

    def search_advanced(
        query: str,
        top_k: int = 8,
        enable_mqe: bool = False,
        enable_hyde: bool = False,
        score_threshold: Optional[float] = None,
    ) -> List[Dict]:
        return search_vectors_expanded(
            store=store,
            query=query,
            top_k=top_k,
            rag_namespace=rag_namespace,
            enable_mqe=enable_mqe,
            enable_hyde=enable_hyde,
            score_threshold=score_threshold,
            llm=llm,
            embedder=active_embedder,
        )

    def get_stats() -> Dict[str, Any]:
        return cast(Dict[str, Any], store.get_collection_stats())

    def rerank(
        query: str, items: List[Dict], top_k: int = 10
    ) -> List[Dict]:
        return rerank_with_cross_encoder(query, items, top_k=top_k)

    def summarize(text: str, bullets: int = 3) -> Optional[str]:
        return tldr_summarize(text, bullets=bullets, llm=llm)

    return {
        "store": store,
        "namespace": rag_namespace,
        "add_documents": add_documents,
        "search": search,
        "search_advanced": search_advanced,
        "get_stats": get_stats,
        "rerank": rerank,
        "summarize": summarize,
    }


__all__ = [
    "DEFAULT_RAG_COLLECTION",
    # 加载
    "load_and_chunk_texts",
    # 图谱
    "build_graph_from_chunks",
    # 索引
    "index_chunks",
    # 检索
    "embed_query",
    "search_vectors",
    "search_vectors_expanded",
    # 重排
    "rerank_with_cross_encoder",
    "compute_graph_signals_from_pool",
    "rank",
    # 组装
    "merge_snippets",
    "expand_neighbors_from_pool",
    "merge_snippets_grouped",
    "compress_ranked_items",
    "tldr_summarize",
    # 工厂
    "create_rag_pipeline",
]
