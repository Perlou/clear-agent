"""Embedding 抽象与多后端实现

Ported from AntonAgents (CC-BY-NC-SA-4.0)
Original: anton_agents/memory/embedding.py

提供统一的文本嵌入接口与多实现：
- ``LocalTransformerEmbedding``：sentence-transformers 优先，回退 transformers + torch
- ``DashScopeEmbedding``：阿里云 DashScope（OpenAI 兼容 REST 优先，否则用官方 SDK）
- ``TFIDFEmbedding``：基于 sklearn 的 TF-IDF 兜底（无深度模型时仍能用）

公共入口：
- ``create_embedding_model(type, **kwargs)`` 显式工厂
- ``create_embedding_model_with_fallback(preferred, **kwargs)`` 带回退工厂
- ``get_text_embedder()``：线程安全单例（按 ``EMBED_*`` 环境变量构造）
- ``get_dimension()``：当前 embedder 的向量维度
- ``refresh_embedder()``：强制重建（环境变量切换后调）

环境变量（与 .env.example 一致）：
- ``EMBED_MODEL_TYPE``：``"dashscope"`` | ``"local"`` | ``"tfidf"``（默认 dashscope）
- ``EMBED_MODEL_NAME``：模型名称
- ``EMBED_API_KEY``：嵌入 API Key
- ``EMBED_BASE_URL``：嵌入 Base URL（OpenAI 兼容时使用）

依赖（按需安装，全部 optional）：

```bash
pip install clear-agent[retrieval]   # sklearn + numpy
pip install clear-agent[rag]         # sentence-transformers + torch
```
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional, Union

try:
    import numpy as np  # type: ignore
except ImportError as _e:  # pragma: no cover
    np = None  # type: ignore[assignment]

from ..core.exceptions import RetrievalException


# ==============
# 抽象与实现
# ==============


class EmbeddingModel:
    """嵌入模型基类（最小接口）

    子类需实现 ``encode(texts)`` 和 ``dimension`` property。
    """

    def encode(self, texts: Union[str, List[str]]) -> Any:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError


class LocalTransformerEmbedding(EmbeddingModel):
    """本地 Transformer 嵌入（sentence-transformers 优先 → transformers+torch 回退）"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._backend: Optional[str] = None  # "st" 或 "hf"
        self._st_model: Any = None
        self._hf_tokenizer: Any = None
        self._hf_model: Any = None
        self._dimension: Optional[int] = None
        self._load_backend()

    def _load_backend(self) -> None:
        # 优先 sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(self.model_name)
            test_vec = self._st_model.encode("test_text")
            self._dimension = len(test_vec)
            self._backend = "st"
            return
        except Exception:
            self._st_model = None

        # 回退 transformers + torch
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
            import torch  # type: ignore

            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._hf_model = AutoModel.from_pretrained(self.model_name)
            with torch.no_grad():
                inputs = self._hf_tokenizer(
                    "test_text", return_tensors="pt", padding=True, truncation=True
                )
                outputs = self._hf_model(**inputs)
                test_embedding = outputs.last_hidden_state.mean(dim=1)
                self._dimension = int(test_embedding.shape[1])
            self._backend = "hf"
            return
        except Exception:
            self._hf_tokenizer = None
            self._hf_model = None

        raise ImportError(
            "未找到可用的本地嵌入后端，请安装：\n"
            "  pip install sentence-transformers\n"
            "或：\n"
            "  pip install transformers torch"
        )

    def encode(self, texts: Union[str, List[str]]) -> Any:
        if isinstance(texts, str):
            inputs = [texts]
            single = True
        else:
            inputs = list(texts)
            single = False

        if self._backend == "st":
            if self._st_model is None:
                raise RetrievalException("sentence-transformers 后端未初始化")
            vecs = self._st_model.encode(inputs)
            if hasattr(vecs, "tolist"):
                vecs = [v for v in vecs]
        else:
            import torch  # type: ignore

            if self._hf_tokenizer is None or self._hf_model is None:
                raise RetrievalException("transformers 后端未初始化")

            tokenized = self._hf_tokenizer(
                inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                outputs = self._hf_model(**tokenized)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            vecs = [v for v in embeddings]

        if single:
            return vecs[0]
        return vecs

    @property
    def dimension(self) -> int:
        return int(self._dimension or 0)


class TFIDFEmbedding(EmbeddingModel):
    """TF-IDF 简易兜底（在无深度模型时保证可用）

    需要 ``scikit-learn``：``pip install clear-agent[retrieval]``。
    与 LocalTransformer / DashScope 不同的是，TF-IDF **必须先 fit**：

    >>> e = TFIDFEmbedding()
    >>> e.fit(["doc 1", "doc 2", "another doc"])
    >>> e.encode("query").shape  # (max_features,)
    """

    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self._vectorizer: Any = None
        self._is_fitted = False
        self._dimension = max_features
        self._init_vectorizer()

    def _init_vectorizer(self) -> None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

            self._vectorizer = TfidfVectorizer(
                max_features=self.max_features, stop_words="english"
            )
        except ImportError:
            raise ImportError(
                "请安装 scikit-learn: pip install scikit-learn 或 pip install clear-agent[retrieval]"
            )

    def fit(self, texts: List[str]) -> None:
        if self._vectorizer is None:
            self._init_vectorizer()
        self._vectorizer.fit(texts)
        self._is_fitted = True
        self._dimension = len(self._vectorizer.get_feature_names_out())

    def encode(self, texts: Union[str, List[str]]) -> Any:
        if not self._is_fitted:
            raise RetrievalException("TF-IDF 未训练，请先调用 fit()")
        if isinstance(texts, str):
            inputs = [texts]
            single = True
        else:
            inputs = list(texts)
            single = False
        tfidf_matrix = self._vectorizer.transform(inputs)
        embeddings = tfidf_matrix.toarray()
        if single:
            return embeddings[0]
        return [e for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension


class DashScopeEmbedding(EmbeddingModel):
    """阿里云 DashScope（通义千问）/ OpenAI 兼容 REST 嵌入

    行为：
    - 提供 ``base_url`` → 走 OpenAI 兼容 REST：``POST {base_url}/embeddings``
    - 否则走官方 ``dashscope`` SDK 的 ``TextEmbedding.call``
    """

    def __init__(
        self,
        model_name: str = "text-embedding-v3",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self._dimension: Optional[int] = None
        if not self.base_url:
            self._init_client()
        # 探测维度
        test = self.encode("health_check")
        self._dimension = len(test)

    def _init_client(self) -> None:
        try:
            if self.api_key:
                os.environ["DASHSCOPE_API_KEY"] = self.api_key
            import dashscope  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            raise ImportError("请安装 dashscope: pip install dashscope")

    def encode(self, texts: Union[str, List[str]]) -> Any:
        if isinstance(texts, str):
            inputs = [texts]
            single = True
        else:
            inputs = list(texts)
            single = False

        # REST 模式（OpenAI 兼容）
        if self.base_url:
            import requests  # type: ignore

            url = self.base_url.rstrip("/") + "/embeddings"
            headers = {
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                "Content-Type": "application/json",
            }
            payload = {"model": self.model_name, "input": inputs}
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code >= 400:
                raise RetrievalException(
                    f"Embedding REST 调用失败: {resp.status_code} {resp.text}"
                )
            data = resp.json()
            items = data.get("data") or []
            if np is None:
                raise ImportError("需要 numpy: pip install numpy")
            vecs = [np.array(item.get("embedding")) for item in items]
            if single:
                return vecs[0]
            return vecs

        # SDK 模式
        from dashscope import TextEmbedding  # type: ignore[import-untyped]

        rsp = TextEmbedding.call(model=self.model_name, input=inputs)
        embeddings_obj = None
        if isinstance(rsp, dict):
            embeddings_obj = (rsp.get("output") or {}).get("embeddings")
        else:
            embeddings_obj = getattr(getattr(rsp, "output", None), "embeddings", None)
        if not embeddings_obj:
            raise RetrievalException("DashScope 返回为空或格式不匹配")
        if np is None:
            raise ImportError("需要 numpy: pip install numpy")
        vecs = [
            np.array(item.get("embedding") or item.get("vector"))
            for item in embeddings_obj
        ]
        if single:
            return vecs[0]
        return vecs

    @property
    def dimension(self) -> int:
        return int(self._dimension or 0)


# ==============
# 工厂与回退
# ==============


def create_embedding_model(model_type: str = "local", **kwargs: Any) -> EmbeddingModel:
    """显式创建嵌入模型实例

    Args:
        model_type: ``"dashscope"`` | ``"local"`` | ``"tfidf"``
        **kwargs: 透传到具体实现的构造参数（model_name / api_key / base_url）

    Raises:
        ValueError: model_type 不识别
    """
    if model_type in ("local", "sentence_transformer", "huggingface"):
        return LocalTransformerEmbedding(**kwargs)
    elif model_type == "dashscope":
        return DashScopeEmbedding(**kwargs)
    elif model_type == "tfidf":
        return TFIDFEmbedding(**kwargs)
    else:
        raise ValueError(f"不支持的嵌入模型类型: {model_type}")


def _filter_embedding_kwargs(model_type: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return kwargs accepted by the selected embedding backend."""
    if model_type in ("local", "sentence_transformer", "huggingface"):
        allowed = {"model_name"}
    elif model_type == "dashscope":
        allowed = {"model_name", "api_key", "base_url"}
    elif model_type == "tfidf":
        allowed = {"max_features"}
    else:
        allowed = set(kwargs)
    return {k: v for k, v in kwargs.items() if k in allowed and v is not None}


def create_embedding_model_with_fallback(
    preferred_type: str = "dashscope", **kwargs: Any
) -> EmbeddingModel:
    """带回退的创建：``preferred -> dashscope -> local -> tfidf``"""
    if preferred_type in ("sentence_transformer", "huggingface"):
        preferred_type = "local"
    fallback = ["dashscope", "local", "tfidf"]
    if preferred_type in fallback:
        fallback.remove(preferred_type)
        fallback.insert(0, preferred_type)
    last_err: Optional[Exception] = None
    for t in fallback:
        try:
            return create_embedding_model(t, **_filter_embedding_kwargs(t, kwargs))
        except Exception as e:
            last_err = e
            continue
    raise RetrievalException(
        f"所有嵌入模型都不可用，请安装依赖或检查配置。最后错误: {last_err}"
    )


# ==================
# Provider（线程安全单例）
# ==================

_lock = threading.RLock()
_embedder: Optional[EmbeddingModel] = None


def _build_embedder() -> EmbeddingModel:
    preferred = os.getenv("EMBED_MODEL_TYPE", "dashscope").strip()
    default_model = (
        "text-embedding-v3"
        if preferred == "dashscope"
        else "sentence-transformers/all-MiniLM-L6-v2"
    )
    model_name = os.getenv("EMBED_MODEL_NAME", default_model).strip()
    kwargs: dict = {}
    if model_name:
        kwargs["model_name"] = model_name
    api_key = os.getenv("EMBED_API_KEY")
    if api_key:
        kwargs["api_key"] = api_key
    base_url = os.getenv("EMBED_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return create_embedding_model_with_fallback(preferred_type=preferred, **kwargs)


def get_text_embedder() -> EmbeddingModel:
    """获取全局共享的文本嵌入实例（线程安全单例）"""
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is None:
            _embedder = _build_embedder()
        return _embedder


def get_dimension(default: int = 384) -> int:
    """获取统一向量维度（失败回退默认值）"""
    try:
        return int(getattr(get_text_embedder(), "dimension", default))
    except Exception:
        return int(default)


def refresh_embedder() -> EmbeddingModel:
    """强制重建嵌入实例（可用于动态切换环境变量后）"""
    global _embedder
    with _lock:
        _embedder = _build_embedder()
        return _embedder


def _reset_embedder_for_test() -> None:
    """测试钩子：清空全局单例，让下次调用重新构造"""
    global _embedder
    with _lock:
        _embedder = None


__all__ = [
    "EmbeddingModel",
    "LocalTransformerEmbedding",
    "TFIDFEmbedding",
    "DashScopeEmbedding",
    "create_embedding_model",
    "create_embedding_model_with_fallback",
    "get_text_embedder",
    "get_dimension",
    "refresh_embedder",
]
