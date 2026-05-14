"""Embeddings 测试

由于 venv 里 ML 依赖（sklearn / sentence_transformers / torch / dashscope）多半不可用，
本测试集分两层：
- 不依赖 ML 库：基类、工厂错误路径、DashScope REST 模式（mock requests）、单例机制
- 依赖 sklearn：用 ``pytest.importorskip`` 跳过

覆盖范围对应 project_docs/07-anton-agents-port.md §3 的 SOP 验收。
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from clear_agent.core.exceptions import RetrievalException
from clear_agent.retrieval.embeddings import (
    DashScopeEmbedding,
    EmbeddingModel,
    LocalTransformerEmbedding,
    TFIDFEmbedding,
    _reset_embedder_for_test,
    create_embedding_model,
    create_embedding_model_with_fallback,
    get_dimension,
    get_text_embedder,
    refresh_embedder,
)


# ==================== Section A: 基类与工厂 ====================


def test_base_class_methods_raise_not_implemented():
    e = EmbeddingModel()
    with pytest.raises(NotImplementedError):
        e.encode("x")
    with pytest.raises(NotImplementedError):
        _ = e.dimension


def test_create_embedding_model_unknown_type_raises():
    with pytest.raises(ValueError):
        create_embedding_model("bogus")


def test_create_embedding_model_aliases_local():
    """sentence_transformer / huggingface 都映射到 local"""
    # 不真的构造（会试图加载模型）；只检查类型分支：mock _load_backend
    with patch.object(LocalTransformerEmbedding, "_load_backend", lambda self: None):
        e1 = create_embedding_model("local")
        e2 = create_embedding_model("sentence_transformer")
        e3 = create_embedding_model("huggingface")
    assert isinstance(e1, LocalTransformerEmbedding)
    assert isinstance(e2, LocalTransformerEmbedding)
    assert isinstance(e3, LocalTransformerEmbedding)


def test_fallback_returns_first_available():
    """preferred 不可用 → 自动尝试下一个"""
    # 让 local 假装可用，dashscope 不可用
    with patch.object(LocalTransformerEmbedding, "_load_backend", lambda self: None):
        with patch.object(
            DashScopeEmbedding,
            "_init_client",
            side_effect=ImportError("no dashscope"),
        ):
            e = create_embedding_model_with_fallback("dashscope")
            assert isinstance(e, LocalTransformerEmbedding)


def test_fallback_all_fail_raises_retrieval_exception():
    """三个都不可用 → 抛 RetrievalException"""
    with patch.object(
        LocalTransformerEmbedding, "_load_backend", side_effect=ImportError("x")
    ):
        with patch.object(
            DashScopeEmbedding, "_init_client", side_effect=ImportError("y")
        ):
            with patch.object(
                TFIDFEmbedding, "_init_vectorizer", side_effect=ImportError("z")
            ):
                with pytest.raises(RetrievalException):
                    create_embedding_model_with_fallback("dashscope")


def test_fallback_filters_kwargs_for_local_backend():
    """Fallback 到 local 时不应把 api_key/base_url 传给 local 构造器。"""
    with patch.object(DashScopeEmbedding, "__init__", side_effect=ImportError("dash")):
        with patch.object(LocalTransformerEmbedding, "_load_backend", lambda self: None):
            e = create_embedding_model_with_fallback(
                "dashscope",
                model_name="local-model",
                api_key="sk-x",
                base_url="https://embed.example/v1",
            )
    assert isinstance(e, LocalTransformerEmbedding)
    assert e.model_name == "local-model"


def test_fallback_filters_kwargs_for_tfidf_backend():
    """Fallback 到 tfidf 时只应传 max_features 这类 TF-IDF 参数。"""
    with patch.object(DashScopeEmbedding, "__init__", side_effect=ImportError("dash")):
        with patch.object(
            LocalTransformerEmbedding, "__init__", side_effect=ImportError("local")
        ):
            e = create_embedding_model_with_fallback(
                "dashscope",
                model_name="ignored",
                api_key="sk-x",
                base_url="https://embed.example/v1",
                max_features=12,
            )
    assert isinstance(e, TFIDFEmbedding)
    assert e.max_features == 12


# ==================== Section B: 单例 / Provider ====================


@pytest.fixture(autouse=True)
def _reset_global_embedder():
    _reset_embedder_for_test()
    yield
    _reset_embedder_for_test()


def test_get_text_embedder_returns_singleton():
    """两次调用 get_text_embedder 返回同一对象"""
    fake = MagicMock(spec=EmbeddingModel)
    with patch(
        "clear_agent.retrieval.embeddings._build_embedder", return_value=fake
    ):
        a = get_text_embedder()
        b = get_text_embedder()
    assert a is b is fake


def test_refresh_embedder_rebuilds():
    """refresh_embedder 强制重建实例"""
    instances = [MagicMock(spec=EmbeddingModel), MagicMock(spec=EmbeddingModel)]
    with patch(
        "clear_agent.retrieval.embeddings._build_embedder",
        side_effect=instances,
    ):
        a = get_text_embedder()
        b = refresh_embedder()
    assert a is not b
    assert a is instances[0] and b is instances[1]


def test_get_dimension_default_on_failure():
    """get_dimension 在 _build_embedder 失败时回退到 default"""
    with patch(
        "clear_agent.retrieval.embeddings._build_embedder",
        side_effect=RuntimeError("boom"),
    ):
        d = get_dimension(default=42)
    assert d == 42


def test_get_dimension_uses_embedder_property():
    fake = MagicMock(spec=EmbeddingModel)
    fake.dimension = 768
    with patch(
        "clear_agent.retrieval.embeddings._build_embedder", return_value=fake
    ):
        assert get_dimension() == 768


# ==================== Section C: DashScope REST 模式（mock requests） ====================


def _fake_post_response(payload: Dict[str, Any], status: int = 200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.text = "" if status < 400 else "error body"
    return resp


def test_dashscope_rest_mode_single_text():
    payload = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    with patch(
        "requests.post", return_value=_fake_post_response(payload)
    ) as mock_post:
        emb = DashScopeEmbedding(
            model_name="text-embedding-v3",
            api_key="sk-x",
            base_url="https://api.example.com/v1",
        )
    # 探测调用 + 用户调用 = 至少 1 次 post（探测）
    assert mock_post.call_count >= 1
    assert emb.dimension == 3
    # base_url 路径正确
    call = mock_post.call_args
    assert call.args[0] == "https://api.example.com/v1/embeddings"
    assert call.kwargs["headers"]["Authorization"] == "Bearer sk-x"


def test_dashscope_rest_mode_batch():
    payload = {
        "data": [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ]
    }
    # 第 1 次构造时探测维度（单文本，返回多 embeddings 也只看 first），
    # 第 2 次真正批量 encode
    with patch("requests.post", return_value=_fake_post_response(payload)):
        emb = DashScopeEmbedding(
            model_name="m", api_key="k", base_url="https://api/v1"
        )
        vecs = emb.encode(["a", "b"])
    assert isinstance(vecs, list)
    assert len(vecs) == 2


def test_dashscope_rest_mode_http_error_raises():
    """REST 返回 4xx → RetrievalException"""
    with patch(
        "requests.post", return_value=_fake_post_response({}, status=401)
    ):
        with pytest.raises(RetrievalException):
            DashScopeEmbedding(model_name="m", api_key="bad", base_url="https://api/v1")


# ==================== Section D: TFIDF（如果 sklearn 可用） ====================


def test_tfidf_requires_fit_before_encode():
    sklearn = pytest.importorskip("sklearn")  # noqa: F841
    e = TFIDFEmbedding(max_features=100)
    with pytest.raises(RetrievalException):
        e.encode("hello")


def test_tfidf_fit_then_encode():
    sklearn = pytest.importorskip("sklearn")  # noqa: F841
    e = TFIDFEmbedding(max_features=50)
    e.fit(["the quick brown fox", "lazy dog jumps"])
    vec = e.encode("the dog")
    assert vec is not None
    # 训练后维度更新到实际特征数（<= max_features）
    assert e.dimension >= 1
    assert e.dimension <= 50


# ==================== Section E: 顶层导入 ====================


def test_top_level_imports():
    from clear_agent.retrieval import (
        EmbeddingModel,
        LocalTransformerEmbedding,
        TFIDFEmbedding,
        DashScopeEmbedding,
        create_embedding_model,
        create_embedding_model_with_fallback,
        get_text_embedder,
        get_dimension,
        refresh_embedder,
    )

    assert all(
        callable(x) or x is not None
        for x in (
            EmbeddingModel,
            LocalTransformerEmbedding,
            TFIDFEmbedding,
            DashScopeEmbedding,
            create_embedding_model,
            create_embedding_model_with_fallback,
            get_text_embedder,
            get_dimension,
            refresh_embedder,
        )
    )
