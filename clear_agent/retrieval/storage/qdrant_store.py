"""Qdrant 向量库存储

Ported from AntonAgents (CC-BY-NC-SA-4.0)
Original: anton_agents/memory/storage/qdrant_store.py

封装 ``qdrant-client`` 提供向量增删改查与过滤搜索能力，与 ``EmbeddingModel`` 解耦：
本类只接受/返回 ``List[float]`` 向量，不在内部做 embedding。

依赖（optional）：

```bash
pip install clear-agent[retrieval-qdrant]   # 仅 qdrant-client
```

环境变量（用于调优）：

- ``QDRANT_HNSW_M`` (默认 32)
- ``QDRANT_HNSW_EF_CONSTRUCT`` (默认 256)
- ``QDRANT_SEARCH_EF`` (默认 128)
- ``QDRANT_SEARCH_EXACT`` (``"1"`` 启用精确搜索；默认 0)

构造连接的便捷方式见 ``QdrantConnectionManager.get_instance()``（同连接和向量配置
共享单例，避免重复建立连接）。
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, cast

from ...core.exceptions import RetrievalException

try:  # qdrant-client 是 optional dep
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http import models  # type: ignore
    from qdrant_client.http.models import (  # type: ignore
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    # 占位绑定，让 monkeypatch / 反射访问不报 AttributeError
    QdrantClient = None  # type: ignore[assignment,misc]
    models = None  # type: ignore[assignment]
    Distance = None  # type: ignore[assignment,misc]
    VectorParams = None  # type: ignore[assignment,misc]
    PointStruct = None  # type: ignore[assignment,misc]
    Filter = None  # type: ignore[assignment,misc]
    FieldCondition = None  # type: ignore[assignment,misc]
    MatchValue = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)


DEFAULT_COLLECTION = "clear_agent_vectors"


class QdrantConnectionManager:
    """Qdrant 连接管理器（同连接和向量配置共享单例）

    用于避免在同一进程内对相同集合反复建立连接。
    """

    _instances: Dict[Any, "QdrantVectorStore"] = {}
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION,
        vector_size: int = 384,
        distance: str = "cosine",
        timeout: int = 30,
        **kwargs: Any,
    ) -> "QdrantVectorStore":
        """获取或创建 ``QdrantVectorStore`` 实例（同 key 复用）"""
        key = (
            url or "local",
            api_key or "",
            collection_name,
            int(vector_size),
            distance.lower(),
        )
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    logger.debug(f"🔄 创建新的 Qdrant 连接: {collection_name}")
                    cls._instances[key] = QdrantVectorStore(
                        url=url,
                        api_key=api_key,
                        collection_name=collection_name,
                        vector_size=vector_size,
                        distance=distance,
                        timeout=timeout,
                        **kwargs,
                    )
                else:
                    logger.debug(f"♻️ 复用现有 Qdrant 连接: {collection_name}")
        else:
            logger.debug(f"♻️ 复用现有 Qdrant 连接: {collection_name}")
        return cls._instances[key]

    @classmethod
    def reset(cls) -> None:
        """测试钩子：清空所有连接（不调 close）"""
        with cls._lock:
            cls._instances.clear()


class QdrantVectorStore:
    """Qdrant 向量数据库存储

    Args:
        url: Qdrant 云服务 URL（例如 ``https://xxx.cloud.qdrant.io``）；缺省走本地 ``localhost:6333``
        api_key: 云服务 API key
        collection_name: 集合名称
        vector_size: 向量维度（须与 embedder.dimension 一致）
        distance: ``"cosine"`` | ``"dot"`` | ``"euclidean"``
        timeout: 连接超时秒数
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION,
        vector_size: int = 384,
        distance: str = "cosine",
        timeout: int = 30,
        **kwargs: Any,
    ):
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client 未安装。请运行: "
                "pip install clear-agent[retrieval-qdrant]"
            )

        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.timeout = timeout

        # HNSW / 查询参数（环境变量调优）
        self.hnsw_m = self._env_int("QDRANT_HNSW_M", 32)
        self.hnsw_ef_construct = self._env_int("QDRANT_HNSW_EF_CONSTRUCT", 256)
        self.search_ef = self._env_int("QDRANT_SEARCH_EF", 128)
        self.search_exact = os.getenv("QDRANT_SEARCH_EXACT", "0") == "1"

        # 距离度量映射
        distance_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLID,
        }
        self.distance = distance_map.get(distance.lower(), Distance.COSINE)

        self.client: Any = None
        self._initialize_client()

    @staticmethod
    def _env_int(key: str, default: int) -> int:
        try:
            return int(os.getenv(key, str(default)))
        except (TypeError, ValueError):
            return default

    # ==================== 连接 / 集合 ====================

    def _initialize_client(self) -> None:
        try:
            if self.url and self.api_key:
                self.client = QdrantClient(
                    url=self.url, api_key=self.api_key, timeout=self.timeout
                )
                logger.info(f"✅ 成功连接到 Qdrant 云服务: {self.url}")
            elif self.url:
                self.client = QdrantClient(url=self.url, timeout=self.timeout)
                logger.info(f"✅ 成功连接到 Qdrant 服务: {self.url}")
            else:
                self.client = QdrantClient(
                    host="localhost", port=6333, timeout=self.timeout
                )
                logger.info("✅ 成功连接到本地 Qdrant: localhost:6333")

            # 探活
            self.client.get_collections()
            self._ensure_collection()
        except Exception as e:
            logger.error(f"❌ Qdrant 连接失败: {e}")
            raise RetrievalException(f"Qdrant 连接失败: {e}") from e

    def _ensure_collection(self) -> None:
        """确保集合存在；不存在则创建（带 HNSW 配置）"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name not in collection_names:
                hnsw_cfg = None
                try:
                    hnsw_cfg = models.HnswConfigDiff(
                        m=self.hnsw_m, ef_construct=self.hnsw_ef_construct
                    )
                except Exception:
                    hnsw_cfg = None
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, distance=self.distance
                    ),
                    hnsw_config=hnsw_cfg,
                )
                logger.info(f"✅ 创建 Qdrant 集合: {self.collection_name}")
            else:
                logger.info(f"✅ 使用现有 Qdrant 集合: {self.collection_name}")
                self._validate_existing_collection()
                try:
                    self.client.update_collection(
                        collection_name=self.collection_name,
                        hnsw_config=models.HnswConfigDiff(
                            m=self.hnsw_m, ef_construct=self.hnsw_ef_construct
                        ),
                    )
                except Exception as ie:
                    logger.debug(f"跳过更新 HNSW 配置: {ie}")

            self._ensure_payload_indexes()
        except Exception as e:
            logger.error(f"❌ 集合初始化失败: {e}")
            raise RetrievalException(f"Qdrant 集合初始化失败: {e}") from e

    def _validate_existing_collection(self) -> None:
        """Validate existing collection vector config when the server exposes it."""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
        except Exception as e:
            logger.debug(f"跳过校验现有集合配置: {e}")
            return

        vectors = getattr(getattr(getattr(info, "config", None), "params", None), "vectors", None)
        if isinstance(vectors, dict):
            vectors = vectors.get("") or vectors.get("default") or next(
                iter(vectors.values()), None
            )

        existing_size = getattr(vectors, "size", None)
        if isinstance(existing_size, int) and existing_size != self.vector_size:
            raise RetrievalException(
                "现有 Qdrant 集合向量维度不匹配: "
                f"collection={self.collection_name}, "
                f"existing={existing_size}, requested={self.vector_size}"
            )

    def _ensure_payload_indexes(self) -> None:
        """为常用过滤字段创建 payload 索引（已存在则忽略）"""
        try:
            index_fields = [
                ("memory_type", models.PayloadSchemaType.KEYWORD),
                ("user_id", models.PayloadSchemaType.KEYWORD),
                ("memory_id", models.PayloadSchemaType.KEYWORD),
                ("timestamp", models.PayloadSchemaType.INTEGER),
                ("modality", models.PayloadSchemaType.KEYWORD),
                ("source", models.PayloadSchemaType.KEYWORD),
                ("external", models.PayloadSchemaType.BOOL),
                ("namespace", models.PayloadSchemaType.KEYWORD),
                # RAG 相关
                ("is_rag_data", models.PayloadSchemaType.BOOL),
                ("rag_namespace", models.PayloadSchemaType.KEYWORD),
                ("data_source", models.PayloadSchemaType.KEYWORD),
            ]
            for field_name, schema_type in index_fields:
                try:
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=schema_type,
                    )
                except Exception as ie:
                    logger.debug(f"索引 {field_name} 已存在或创建失败: {ie}")
        except Exception as e:
            logger.debug(f"创建 payload 索引时出错: {e}")

    # ==================== add / search / delete ====================

    def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> bool:
        """批量添加向量

        Args:
            vectors: 向量列表（每个长度须等于 ``vector_size``）
            metadata: 与 vectors 对齐的 payload 列表
            ids: 可选 ID 列表；缺省自动生成

        Returns:
            ``True`` 成功；维度不匹配/qdrant 异常时返回 ``False``
        """
        try:
            if not vectors:
                logger.warning("⚠️ 向量列表为空")
                return False

            if ids is None:
                ts = int(datetime.now().timestamp() * 1000000)
                ids = [f"vec_{i}_{ts}" for i in range(len(vectors))]

            logger.info(
                f"[Qdrant] add_vectors: n={len(vectors)} collection={self.collection_name}"
            )
            points: List[Any] = []
            for i, (vector, meta, point_id) in enumerate(zip(vectors, metadata, ids)):
                try:
                    vlen = len(vector)
                except Exception:
                    logger.error(f"[Qdrant] 非法向量类型: index={i} type={type(vector)}")
                    continue
                if vlen != self.vector_size:
                    logger.warning(
                        f"⚠️ 向量维度不匹配: 期望 {self.vector_size}, 实际 {vlen}"
                    )
                    continue

                meta_with_ts = dict(meta)
                now_ts = int(datetime.now().timestamp())
                meta_with_ts.setdefault("timestamp", now_ts)
                meta_with_ts["added_at"] = now_ts

                if "external" in meta_with_ts and not isinstance(
                    meta_with_ts["external"], bool
                ):
                    val = meta_with_ts["external"]
                    meta_with_ts["external"] = str(val).lower() in ("1", "true", "yes")

                # Qdrant 只接受 unsigned int 或 UUID 字符串
                safe_id: Any
                if isinstance(point_id, int):
                    safe_id = point_id
                elif isinstance(point_id, str):
                    try:
                        uuid.UUID(point_id)
                        safe_id = point_id
                    except Exception:
                        safe_id = str(uuid.uuid4())
                else:
                    safe_id = str(uuid.uuid4())

                points.append(
                    PointStruct(id=safe_id, vector=vector, payload=meta_with_ts)
                )

            if not points:
                logger.warning("⚠️ 没有有效的向量点")
                return False

            self.client.upsert(
                collection_name=self.collection_name, points=points, wait=True
            )
            logger.info(f"✅ 成功添加 {len(points)} 个向量到 Qdrant")
            return True
        except Exception as e:
            logger.error(f"❌ 添加向量失败: {e}")
            return False

    def search_similar(
        self,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """向量相似度搜索 + 可选 payload 过滤

        Args:
            query_vector: 查询向量
            limit: 最多返回数
            score_threshold: 最低相似度阈值（视 distance 而定）
            where: 等值过滤条件 ``{"memory_type": "fact", ...}``

        Returns:
            结果列表，每条形如 ``{"id":..., "score":..., "metadata":{...}}``。
            维度不匹配返回 ``[]``，qdrant 异常返回 ``[]``。
        """
        try:
            if len(query_vector) != self.vector_size:
                logger.error(
                    f"❌ 查询向量维度错误: 期望 {self.vector_size}, 实际 {len(query_vector)}"
                )
                return []

            query_filter = None
            if where:
                conditions: List[Any] = [
                    FieldCondition(key=k, match=MatchValue(value=cast(Any, v)))
                    for k, v in where.items()
                    if isinstance(v, (str, int, float, bool))
                ]
                if conditions:
                    query_filter = Filter(must=conditions)

            try:
                search_params = models.SearchParams(
                    hnsw_ef=self.search_ef, exact=self.search_exact
                )
            except Exception:
                search_params = None

            # 兼容 qdrant-client >=1.16 (query_points) 与 <1.16 (search)
            try:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                    search_params=search_params,
                )
                search_result = response.points
            except AttributeError:
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False,
                    search_params=search_params,
                )

            results = [
                {"id": hit.id, "score": hit.score, "metadata": hit.payload or {}}
                for hit in search_result
            ]
            logger.debug(f"🔍 Qdrant 搜索返回 {len(results)} 个结果")
            return results
        except Exception as e:
            logger.error(f"❌ 向量搜索失败: {e}")
            return []

    def delete_vectors(self, ids: List[Any]) -> bool:
        """按点 ID 删除"""
        try:
            if not ids:
                return True
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=ids),
                wait=True,
            )
            logger.info(f"✅ 成功删除 {len(ids)} 个向量")
            return True
        except Exception as e:
            logger.error(f"❌ 删除向量失败: {e}")
            return False

    def delete_memories(self, memory_ids: List[str]) -> None:
        """按 payload 中的 ``memory_id`` 字段删除（不依赖点 ID）

        说明：写入时可能将非 UUID 的字符串点 ID 转成 UUID，因此用 payload
        过滤删除更可靠。
        """
        try:
            if not memory_ids:
                return
            conditions: List[Any] = [
                FieldCondition(key="memory_id", match=MatchValue(value=cast(Any, mid)))
                for mid in memory_ids
            ]
            query_filter = Filter(should=conditions)
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=query_filter),
                wait=True,
            )
            logger.info(f"✅ 成功按 memory_id 删除 {len(memory_ids)} 个 Qdrant 向量")
        except Exception as e:
            logger.error(f"❌ 删除记忆失败: {e}")
            raise RetrievalException(f"按 memory_id 删除失败: {e}") from e

    def clear_collection(self) -> bool:
        """删除并重建集合（保留 schema）"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._ensure_collection()
            logger.info(f"✅ 成功清空 Qdrant 集合: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"❌ 清空集合失败: {e}")
            return False

    # ==================== 信息查询 ====================

    def get_collection_info(self) -> Dict[str, Any]:
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": getattr(info, "vectors_count", None),
                "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
                "points_count": getattr(info, "points_count", None),
                "segments_count": getattr(info, "segments_count", None),
                "config": {
                    "vector_size": self.vector_size,
                    "distance": self.distance.value,
                },
            }
        except Exception as e:
            logger.error(f"❌ 获取集合信息失败: {e}")
            return {}

    def get_collection_stats(self) -> Dict[str, Any]:
        """与 ``DocumentStore.get_database_stats`` 风格对齐的统计接口"""
        info = self.get_collection_info()
        if not info:
            return {"store_type": "qdrant", "name": self.collection_name}
        info["store_type"] = "qdrant"
        return info

    def health_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"❌ Qdrant 健康检查失败: {e}")
            return False

    def __del__(self) -> None:
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
            except Exception:
                pass


__all__ = [
    "QdrantVectorStore",
    "QdrantConnectionManager",
    "QDRANT_AVAILABLE",
    "DEFAULT_COLLECTION",
]
