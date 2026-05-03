"""SemanticMemory —— 长期语义记忆（向量 + 内存知识图谱）

Ported from AntonAgents (CC-BY-NC-SA-4.0)
Original: anton_agents/memory/types/semantic.py

**改造决策**：
ClearAgent **不引入 Neo4jStore**（Qdrant 已覆盖 90% 检索场景），
故本移植版把原始的 Neo4j 图数据库集成**全部替换为内存中的 entities/relations 缓存**：
- ``self.entities: Dict[entity_id, Entity]``
- ``self.relations: List[Relation]``

保留接口：``add / retrieve / update / remove / has_memory / clear / get_stats /
forget / get_all / get_entity / search_entities / get_related_entities /
export_knowledge_graph``。

特点：
- 嵌入模型 + Qdrant 向量库 → 快速相似度召回
- spaCy NER（可选，缺省走简单 fallback）→ 实体提取
- 共现关系 → 实体关系图
- 混合检索：``_vector_search`` + ``_graph_search`` → ``_combine_and_rank_results``

依赖（按需）：
- ``clear-agent[retrieval-qdrant]`` — Qdrant 向量库
- ``spacy`` — 实体提取（``zh_core_web_sm`` / ``en_core_web_sm``）
- ``numpy`` — 向量运算（已在核心依赖）
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import BaseMemory, MemoryConfig, MemoryItem

if TYPE_CHECKING:
    from ..retrieval.embeddings import EmbeddingModel
    from ..retrieval.storage.qdrant_store import QdrantVectorStore

logger = logging.getLogger(__name__)

# ==================== Entity / Relation ====================

class Entity:
    """语义实体节点（内存中的图节点）"""

    def __init__(
        self,
        entity_id: str,
        name: str,
        entity_type: str = "MISC",
        description: str = "",
        properties: Optional[Dict[str, Any]] = None,
    ):
        self.entity_id = entity_id
        self.name = name
        self.entity_type = entity_type  # PERSON / ORG / PRODUCT / CONCEPT 等
        self.description = description
        self.properties = properties or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.frequency = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "properties": dict(self.properties),
            "frequency": self.frequency,
        }

class Relation:
    """语义关系边（内存中的图边）"""

    def __init__(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str,
        strength: float = 1.0,
        evidence: str = "",
        properties: Optional[Dict[str, Any]] = None,
    ):
        self.from_entity = from_entity
        self.to_entity = to_entity
        self.relation_type = relation_type
        self.strength = strength
        self.evidence = evidence  # 支持该关系的原文
        self.properties = properties or {}
        self.created_at = datetime.now()
        self.frequency = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_entity": self.from_entity,
            "to_entity": self.to_entity,
            "relation_type": self.relation_type,
            "strength": self.strength,
            "evidence": self.evidence,
            "properties": dict(self.properties),
            "frequency": self.frequency,
        }

# ==================== SemanticMemory ====================

class SemanticMemory(BaseMemory):
    """长期语义记忆（向量 + 内存知识图谱）

    构造参数全部支持注入（便于测试 / 自定义 backend）：

    Args:
        config: ``MemoryConfig``
        storage_backend: 兼容父类（一般不用）
        embedding_model: 已构造的 ``EmbeddingModel``；缺省走 ``get_text_embedder()``
        vector_store: 已构造的 ``QdrantVectorStore``；缺省走 ``QdrantConnectionManager``
        nlp: 已加载的 spaCy ``Language``；缺省尝试 ``zh_core_web_sm`` → ``en_core_web_sm``
            → None（fallback 走 ``_fallback_extract_entities``）
    """

    def __init__(
        self,
        config: MemoryConfig,
        storage_backend: Optional[Any] = None,
        embedding_model: Optional["EmbeddingModel"] = None,
        vector_store: Optional["QdrantVectorStore"] = None,
        nlp: Optional[Any] = None,
    ):
        super().__init__(config, storage_backend)

        # 嵌入模型
        self.embedding_model = embedding_model
        if self.embedding_model is None:
            self._init_embedding_model()

        # 向量库
        self.vector_store = vector_store
        if self.vector_store is None:
            self._init_vector_store()

        # 内存知识图谱
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []

        # NLP（spaCy 可选）
        self.nlp = nlp
        if self.nlp is None:
            self._init_nlp()

        # 记忆缓存
        self.semantic_memories: List[MemoryItem] = []
        self.memory_embeddings: Dict[str, Any] = {}  # id -> ndarray | list

        logger.info("✅ SemanticMemory 初始化完成（Qdrant + 内存图谱）")

    # ==================== 初始化 ====================

    def _init_embedding_model(self) -> None:
        try:
            from ..retrieval.embeddings import get_text_embedder

            self.embedding_model = get_text_embedder()
            try:
                self.embedding_model.encode("health_check")
                logger.info("✅ 嵌入模型就绪")
            except Exception:
                logger.info("✅ 嵌入模型就绪（健康检查跳过）")
        except Exception as e:
            logger.error(f"❌ 嵌入模型初始化失败: {e}")
            raise

    def _init_vector_store(self) -> None:
        try:
            import os

            from ..retrieval.embeddings import get_dimension
            from ..retrieval.storage.qdrant_store import QdrantConnectionManager

            self.vector_store = QdrantConnectionManager.get_instance(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                collection_name="clear_agent_semantic",
                vector_size=get_dimension(384),
                distance="cosine",
            )
            logger.info("✅ Qdrant 向量库初始化完成（集合 clear_agent_semantic）")
        except Exception as e:
            logger.error(f"❌ Qdrant 初始化失败: {e}")
            raise

    def _init_nlp(self) -> None:
        """尝试加载 spaCy；缺失时降级到 fallback"""
        try:
            import spacy  # type: ignore

            for model_name in ("zh_core_web_sm", "en_core_web_sm"):
                try:
                    self.nlp = spacy.load(model_name)
                    logger.info(f"✅ 加载 spaCy 模型: {model_name}")
                    return
                except OSError:
                    continue
            self.nlp = None
            logger.warning("⚠️ 未找到可用的 spaCy 模型；使用 fallback 实体提取")
        except ImportError:
            self.nlp = None
            logger.warning("⚠️ 未安装 spaCy；使用 fallback 实体提取")

    # ==================== 7 个抽象接口 ====================

    def add(self, memory_item: MemoryItem) -> str:
        """添加语义记忆：嵌入 + 实体/关系提取 + Qdrant 写入 + 内存图谱更新"""
        try:
            # 1. 嵌入
            embedding = self.embedding_model.encode(memory_item.content)
            self.memory_embeddings[memory_item.id] = embedding

            # 2. 实体 / 关系
            entities = self._extract_entities(memory_item.content)
            relations = self._extract_relations(memory_item.content, entities)

            # 3. 更新内存图谱
            for e in entities:
                self._add_or_update_entity(e)
            for r in relations:
                self._add_or_update_relation(r)

            # 4. 写入 Qdrant
            metadata = {
                "memory_id": memory_item.id,
                "user_id": memory_item.user_id,
                "content": memory_item.content,
                "memory_type": memory_item.memory_type,
                "timestamp": int(memory_item.timestamp.timestamp()),
                "importance": memory_item.importance,
                "entities": [e.entity_id for e in entities],
                "entity_count": len(entities),
                "relation_count": len(relations),
            }
            try:
                vec = (
                    embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
                )
            except Exception:
                vec = list(embedding) if embedding is not None else []
            success = self.vector_store.add_vectors(
                vectors=[vec], metadata=[metadata], ids=[memory_item.id]
            )
            if not success:
                logger.warning("⚠️ 向量写入失败，但已更新内存图谱")

            # 5. 更新记忆缓存
            memory_item.metadata["entities"] = [e.entity_id for e in entities]
            memory_item.metadata["relations"] = [
                f"{r.from_entity}-{r.relation_type}-{r.to_entity}" for r in relations
            ]
            self.semantic_memories.append(memory_item)

            logger.info(
                f"✅ 语义记忆已添加: {len(entities)} 实体 / {len(relations)} 关系"
            )
            return memory_item.id
        except Exception as e:
            logger.error(f"❌ 添加语义记忆失败: {e}")
            raise

    def retrieve(
        self, query: str, limit: int = 5, **kwargs: Any
    ) -> List[MemoryItem]:
        """混合检索：向量 + 内存图，softmax 概率附在 metadata"""
        try:
            user_id = kwargs.get("user_id")

            vector_results = self._vector_search(query, limit * 2, user_id)
            graph_results = self._graph_search(query, limit * 2, user_id)
            combined = self._combine_and_rank_results(
                vector_results, graph_results, query, limit
            )

            # softmax 概率
            scores = [r.get("combined_score", r.get("vector_score", 0.0)) for r in combined]
            if scores:
                max_s = max(scores)
                exps = [math.exp(s - max_s) for s in scores]
                denom = sum(exps) or 1.0
                probs = [e / denom for e in exps]
            else:
                probs = []

            results: List[MemoryItem] = []
            for idx, r in enumerate(combined):
                memory_id = r.get("memory_id") or r.get("id")
                # 过滤已遗忘
                local = next(
                    (m for m in self.semantic_memories if m.id == memory_id), None
                )
                if local and local.metadata.get("forgotten", False):
                    continue

                # 时间戳兼容
                ts = r.get("timestamp")
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts)
                    except ValueError:
                        ts = datetime.now()
                elif isinstance(ts, (int, float)):
                    ts = datetime.fromtimestamp(ts)
                else:
                    ts = datetime.now()

                results.append(
                    MemoryItem(
                        id=memory_id,
                        content=r.get("content", ""),
                        memory_type="semantic",
                        user_id=r.get("user_id", "default"),
                        timestamp=ts,
                        importance=r.get("importance", 0.5),
                        metadata={
                            **r.get("metadata", {}),
                            "combined_score": r.get("combined_score", 0.0),
                            "vector_score": r.get("vector_score", 0.0),
                            "graph_score": r.get("graph_score", 0.0),
                            "probability": probs[idx] if idx < len(probs) else 0.0,
                        },
                    )
                )

            logger.info(f"✅ 检索到 {len(results)} 条相关记忆")
            return results[:limit]
        except Exception as e:
            logger.error(f"❌ 检索语义记忆失败: {e}")
            return []

    def update(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        memory = self._find_memory_by_id(memory_id)
        if not memory:
            return False
        try:
            if content is not None:
                # 重新生成嵌入 + 实体
                embedding = self.embedding_model.encode(content)
                self.memory_embeddings[memory_id] = embedding
                memory.content = content
                entities = self._extract_entities(content)
                relations = self._extract_relations(content, entities)
                for e in entities:
                    self._add_or_update_entity(e)
                for r in relations:
                    self._add_or_update_relation(r)
                memory.metadata["entities"] = [e.entity_id for e in entities]
                memory.metadata["relations"] = [
                    f"{r.from_entity}-{r.relation_type}-{r.to_entity}"
                    for r in relations
                ]
            if importance is not None:
                memory.importance = importance
            if metadata is not None:
                memory.metadata.update(metadata)
            return True
        except Exception as e:
            logger.error(f"❌ 更新记忆失败: {e}")
            return False

    def remove(self, memory_id: str) -> bool:
        memory = self._find_memory_by_id(memory_id)
        if not memory:
            return False
        try:
            try:
                self.vector_store.delete_memories([memory_id])
            except Exception as e:
                logger.warning(f"⚠️ Qdrant 删除失败（继续删除本地）: {e}")
            self.semantic_memories.remove(memory)
            self.memory_embeddings.pop(memory_id, None)
            return True
        except Exception as e:
            logger.error(f"❌ 删除记忆失败: {e}")
            return False

    def has_memory(self, memory_id: str) -> bool:
        return self._find_memory_by_id(memory_id) is not None

    def clear(self) -> None:
        """清空：Qdrant 集合 + 内存缓存 + 图谱"""
        try:
            if self.vector_store is not None:
                try:
                    self.vector_store.clear_collection()
                    logger.info("✅ Qdrant 集合已清空")
                except Exception as e:
                    logger.warning(f"⚠️ Qdrant 清空失败: {e}")
        finally:
            self.semantic_memories.clear()
            self.memory_embeddings.clear()
            self.entities.clear()
            self.relations.clear()
            logger.info("🧹 SemanticMemory 已完全清空")

    def get_stats(self) -> Dict[str, Any]:
        active = self.semantic_memories
        avg_importance = (
            sum(m.importance for m in active) / len(active) if active else 0.0
        )
        return {
            "count": len(active),
            "forgotten_count": 0,
            "total_count": len(self.semantic_memories),
            "entities_count": len(self.entities),
            "relations_count": len(self.relations),
            "graph_nodes": len(self.entities),
            "graph_edges": len(self.relations),
            "avg_importance": avg_importance,
            "memory_type": "semantic",
        }

    # ==================== forget ====================

    def forget(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30,
    ) -> int:
        """硬删除遗忘"""
        forgotten = 0
        now = datetime.now()
        to_remove: List[str] = []

        for m in self.semantic_memories:
            should = False
            if strategy == "importance_based":
                if m.importance < threshold:
                    should = True
            elif strategy == "time_based":
                if m.timestamp < now - timedelta(days=max_age_days):
                    should = True
            elif strategy == "capacity_based":
                if len(self.semantic_memories) > self.config.max_capacity:
                    sorted_mems = sorted(self.semantic_memories, key=lambda x: x.importance)
                    excess = len(self.semantic_memories) - self.config.max_capacity
                    if m in sorted_mems[:excess]:
                        should = True
            if should:
                to_remove.append(m.id)

        for mid in to_remove:
            if self.remove(mid):
                forgotten += 1
        return forgotten

    # ==================== 检索内部 ====================

    def _vector_search(
        self, query: str, limit: int, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            qv = self.embedding_model.encode(query)
            qv_list = qv.tolist() if hasattr(qv, "tolist") else list(qv)

            where = {"memory_type": "semantic"}
            if user_id:
                where["user_id"] = user_id

            hits = self.vector_store.search_similar(
                query_vector=qv_list, limit=limit, where=where
            )
            results = []
            for h in hits:
                results.append(
                    {
                        "id": h["id"],
                        "score": h.get("score", 0.0),
                        **h.get("metadata", {}),
                    }
                )
            return results
        except Exception as e:
            logger.error(f"❌ 向量搜索失败: {e}")
            return []

    def _graph_search(
        self, query: str, limit: int, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """内存图搜索：从查询提取实体 → 找包含实体的记忆"""
        try:
            query_entities = self._extract_entities(query)
            if not query_entities:
                # 名称模糊匹配兜底
                ql = query.lower()
                query_entities = [
                    e for e in self.entities.values() if ql in e.name.lower()
                ][:3]
            if not query_entities:
                return []
            query_entity_ids = {e.entity_id for e in query_entities}

            results: List[Dict[str, Any]] = []
            for m in self.semantic_memories:
                if user_id and m.user_id != user_id:
                    continue
                m_entities = set(m.metadata.get("entities", []))
                if not m_entities:
                    continue
                overlap = m_entities & query_entity_ids
                if not overlap:
                    continue
                # 简单图相关性：实体重叠度 + 实体密度
                entity_score = len(overlap) / max(1, len(query_entity_ids))
                entity_density = min(len(m_entities) / 10, 1.0)
                relation_density = min(
                    len(m.metadata.get("relations", [])) / 5, 1.0
                )
                graph_score = (
                    entity_score * 0.6
                    + entity_density * 0.2
                    + relation_density * 0.2
                )
                results.append(
                    {
                        "id": m.id,
                        "memory_id": m.id,
                        "content": m.content,
                        "similarity": min(graph_score, 1.0),
                        "user_id": m.user_id,
                        "memory_type": m.memory_type,
                        "importance": m.importance,
                        "timestamp": int(m.timestamp.timestamp()),
                        "entities": list(m_entities),
                    }
                )

            results.sort(key=lambda x: x["similarity"], reverse=True)
            return results[:limit]
        except Exception as e:
            logger.error(f"❌ 图搜索失败: {e}")
            return []

    def _combine_and_rank_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        query: str,  # noqa: ARG002
        limit: int,
    ) -> List[Dict[str, Any]]:
        """混合排序（向量 + 图 + 重要性加权）"""
        combined: Dict[str, Dict[str, Any]] = {}
        seen_hashes: set = set()

        for r in vector_results:
            mid = r.get("memory_id") or r.get("id")
            content = r.get("content", "")
            ch = hash(content.strip())
            if ch in seen_hashes:
                continue
            seen_hashes.add(ch)
            combined[mid] = {
                **r,
                "vector_score": r.get("score", 0.0),
                "graph_score": 0.0,
                "memory_id": mid,
            }

        for r in graph_results:
            mid = r.get("memory_id") or r.get("id")
            ch = hash(r.get("content", "").strip())
            if mid in combined:
                combined[mid]["graph_score"] = r.get("similarity", 0.0)
            elif ch not in seen_hashes:
                seen_hashes.add(ch)
                combined[mid] = {
                    **r,
                    "vector_score": 0.0,
                    "graph_score": r.get("similarity", 0.0),
                    "memory_id": mid,
                }

        # 综合分数
        for r in combined.values():
            v = r.get("vector_score", 0.0)
            g = r.get("graph_score", 0.0)
            imp = r.get("importance", 0.5)
            base = v * 0.7 + g * 0.3
            weight = 0.8 + (imp * 0.4)
            r["combined_score"] = base * weight

        # 阈值过滤 + 排序
        filtered = [r for r in combined.values() if r["combined_score"] >= 0.1]
        filtered.sort(key=lambda x: x["combined_score"], reverse=True)
        return filtered[:limit]

    # ==================== 实体 / 关系提取 ====================

    def _detect_language(self, text: str) -> str:
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        total = len(text.replace(" ", ""))
        if total == 0:
            return "en"
        return "zh" if (cjk / total) > 0.3 else "en"

    def _extract_entities(self, text: str) -> List[Entity]:
        """spaCy NER 优先；失败 fallback 简单关键词提取"""
        if not text:
            return []
        if self.nlp is not None:
            try:
                doc = self.nlp(text)
                out: List[Entity] = []
                for ent in doc.ents:
                    out.append(
                        Entity(
                            entity_id=f"entity_{hash(ent.text)}",
                            name=ent.text,
                            entity_type=getattr(ent, "label_", "MISC"),
                            description=f"从文本提取的 {getattr(ent, 'label_', 'MISC')} 实体",
                        )
                    )
                if out:
                    return out
            except Exception as e:
                logger.warning(f"⚠️ spaCy 实体提取失败: {e}")
        return self._fallback_extract_entities(text)

    def _fallback_extract_entities(self, text: str) -> List[Entity]:
        """无 spaCy 时：把长度 ≥ 2 的非停用词当作潜在实体（最多 5 个）"""
        seen: set = set()
        out: List[Entity] = []
        for token in text.split():
            t = token.strip(".,;:!?\"'()[]{}").strip()
            if len(t) < 2 or t.lower() in seen:
                continue
            # CJK 内每个字单独不入；这里只接 word
            if t.isspace():
                continue
            seen.add(t.lower())
            out.append(
                Entity(
                    entity_id=f"entity_{hash(t)}",
                    name=t,
                    entity_type="MISC",
                    description="fallback 提取",
                )
            )
            if len(out) >= 5:
                break
        return out

    def _extract_relations(
        self, text: str, entities: List[Entity]
    ) -> List[Relation]:
        """共现关系（简化：每对实体一条 CO_OCCURS 边）"""
        rels: List[Relation] = []
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1:]:
                rels.append(
                    Relation(
                        from_entity=e1.entity_id,
                        to_entity=e2.entity_id,
                        relation_type="CO_OCCURS",
                        strength=0.5,
                        evidence=text[:100],
                    )
                )
        return rels

    # ==================== 内存图谱更新 ====================

    def _add_or_update_entity(self, entity: Entity) -> None:
        if entity.entity_id in self.entities:
            existing = self.entities[entity.entity_id]
            existing.frequency += 1
            existing.updated_at = datetime.now()
        else:
            self.entities[entity.entity_id] = entity

    def _add_or_update_relation(self, relation: Relation) -> None:
        for r in self.relations:
            if (
                r.from_entity == relation.from_entity
                and r.to_entity == relation.to_entity
                and r.relation_type == relation.relation_type
            ):
                r.frequency += 1
                r.strength = min(1.0, r.strength + 0.1)
                return
        self.relations.append(relation)

    # ==================== 查询接口 ====================

    def _find_memory_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        for m in self.semantic_memories:
            if m.id == memory_id:
                return m
        return None

    def get_all(self) -> List[MemoryItem]:
        return self.semantic_memories.copy()

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """按名称 / 类型 / 描述模糊匹配，附 frequency 加权"""
        ql = query.lower()
        scored: List[tuple] = []
        for e in self.entities.values():
            score = 0.0
            if ql in e.name.lower():
                score += 2.0
            if ql in e.entity_type.lower():
                score += 1.0
            if ql in e.description.lower():
                score += 0.5
            score *= math.log(1 + e.frequency)
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:limit]]

    def get_related_entities(
        self,
        entity_id: str,
        relation_types: Optional[List[str]] = None,
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        """从内存图遍历相关实体（BFS，最多 max_hops 跳）"""
        if entity_id not in self.entities:
            return []

        # 邻接表
        adj: Dict[str, List[Relation]] = {}
        for r in self.relations:
            if relation_types and r.relation_type not in relation_types:
                continue
            adj.setdefault(r.from_entity, []).append(r)
            adj.setdefault(r.to_entity, []).append(r)

        visited: Dict[str, int] = {entity_id: 0}
        queue: List[str] = [entity_id]
        results: List[Dict[str, Any]] = []

        while queue:
            cur = queue.pop(0)
            cur_dist = visited[cur]
            if cur_dist >= max_hops:
                continue
            for r in adj.get(cur, []):
                neighbor = r.to_entity if r.from_entity == cur else r.from_entity
                if neighbor in visited:
                    continue
                visited[neighbor] = cur_dist + 1
                ent = self.entities.get(neighbor) or Entity(
                    entity_id=neighbor, name=neighbor
                )
                results.append(
                    {
                        "entity": ent,
                        "relation_type": r.relation_type,
                        "strength": r.strength / max(1, cur_dist + 1),
                        "distance": cur_dist + 1,
                    }
                )
                queue.append(neighbor)

        results.sort(key=lambda x: (x["distance"], -x["strength"]))
        return results

    def export_knowledge_graph(self) -> Dict[str, Any]:
        """导出当前内存图谱（entities + relations + 统计）"""
        return {
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "relations": [r.to_dict() for r in self.relations],
            "graph_stats": {
                "total_nodes": len(self.entities),
                "entity_nodes": len(self.entities),
                "memory_nodes": len(self.semantic_memories),
                "total_relationships": len(self.relations),
                "cached_entities": len(self.entities),
                "cached_relations": len(self.relations),
            },
        }

__all__ = ["Entity", "Relation", "SemanticMemory"]
