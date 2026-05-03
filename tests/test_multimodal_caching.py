"""GA-W3 测试 —— Multimodal + Prompt caching"""

from __future__ import annotations

import base64
from typing import Any

import pytest

from clear_agent.core.multimodal import (
    assistant_message,
    audio_part,
    cache_breakpoint,
    file_part,
    image_base64_part,
    image_url_part,
    system_message,
    text_part,
    user_message,
    with_cache_control,
)


# ==================== Section A: 文本块 ====================


def test_text_part_basic():
    p = text_part("hello")
    assert p == {"type": "text", "text": "hello"}


def test_text_part_empty_string():
    p = text_part("")
    assert p["type"] == "text"
    assert p["text"] == ""


# ==================== Section B: 图像块 ====================


def test_image_url_part_basic():
    p = image_url_part("https://x.com/a.jpg")
    assert p["type"] == "image_url"
    assert p["image_url"]["url"] == "https://x.com/a.jpg"
    assert "detail" not in p["image_url"]


def test_image_url_part_with_detail():
    p = image_url_part("https://x.com/a.jpg", detail="high")
    assert p["image_url"]["detail"] == "high"


def test_image_base64_openai_default():
    raw = b"hello"
    p = image_base64_part(raw, media_type="image/png")
    assert p["type"] == "image_url"
    expected_b64 = base64.b64encode(raw).decode("ascii")
    assert p["image_url"]["url"] == f"data:image/png;base64,{expected_b64}"


def test_image_base64_openai_string_input():
    """已 base64 字符串直接用"""
    b64 = base64.b64encode(b"x").decode("ascii")
    p = image_base64_part(b64, media_type="image/jpeg")
    assert p["image_url"]["url"] == f"data:image/jpeg;base64,{b64}"


def test_image_base64_anthropic_format():
    raw = b"hello"
    p = image_base64_part(raw, media_type="image/png", provider="anthropic")
    assert p["type"] == "image"
    assert p["source"]["type"] == "base64"
    assert p["source"]["media_type"] == "image/png"
    assert p["source"]["data"] == base64.b64encode(raw).decode("ascii")


# ==================== Section C: 音频块 ====================


def test_audio_part_bytes():
    raw = b"audio"
    p = audio_part(raw, format="wav")
    assert p["type"] == "input_audio"
    assert p["input_audio"]["format"] == "wav"
    assert p["input_audio"]["data"] == base64.b64encode(raw).decode("ascii")


def test_audio_part_string():
    b64 = base64.b64encode(b"x").decode("ascii")
    p = audio_part(b64, format="mp3")
    assert p["input_audio"]["data"] == b64
    assert p["input_audio"]["format"] == "mp3"


def test_file_part():
    p = file_part("file_abc123")
    assert p == {"type": "file", "file": {"file_id": "file_abc123"}}


# ==================== Section D: 消息构造器 ====================


def test_user_message_from_string():
    m = user_message("hi")
    assert m["role"] == "user"
    assert m["content"] == [text_part("hi")]


def test_user_message_from_parts_list():
    parts = [text_part("see"), image_url_part("u.jpg")]
    m = user_message(parts)
    assert m["role"] == "user"
    assert m["content"] == parts


def test_user_message_from_single_dict():
    m = user_message(text_part("hi"))
    assert m["content"] == [text_part("hi")]


def test_user_message_invalid_content_raises():
    with pytest.raises(TypeError):
        user_message(42)


def test_system_message():
    m = system_message("be brief")
    assert m["role"] == "system"


def test_assistant_message():
    m = assistant_message("hello")
    assert m["role"] == "assistant"


def test_user_message_multimodal_full_example():
    """完整多模态消息示例：text + image"""
    msg = user_message([
        text_part("What's in this picture?"),
        image_url_part("https://example.com/cat.jpg", detail="high"),
    ])
    assert msg["role"] == "user"
    assert len(msg["content"]) == 2
    assert msg["content"][0]["type"] == "text"
    assert msg["content"][1]["type"] == "image_url"


# ==================== Section E: Prompt caching ====================


def test_with_cache_control_string_content():
    """字符串 content 自动转 list 后加 cache_control"""
    msg = {"role": "system", "content": "long prefix"}
    out = with_cache_control(msg)
    assert isinstance(out["content"], list)
    assert out["content"][-1]["cache_control"] == {"type": "ephemeral"}
    # 文本被保留
    assert out["content"][-1]["text"] == "long prefix"


def test_with_cache_control_does_not_mutate_input():
    msg = {"role": "system", "content": "x"}
    out = with_cache_control(msg)
    assert msg.get("content") == "x"
    assert out is not msg


def test_with_cache_control_list_content_marks_last():
    parts = [text_part("first"), text_part("second")]
    msg = {"role": "system", "content": parts}
    out = with_cache_control(msg)
    # 第一个不带 cache_control
    assert "cache_control" not in out["content"][0]
    assert out["content"][1]["cache_control"] == {"type": "ephemeral"}
    # 不修改原 list
    assert "cache_control" not in parts[1]


def test_with_cache_control_persistent_type():
    msg = {"role": "system", "content": "x"}
    out = with_cache_control(msg, cache_type="persistent")
    assert out["content"][-1]["cache_control"]["type"] == "persistent"


def test_with_cache_control_empty_content_returns_unchanged():
    msg = {"role": "user", "content": []}
    out = with_cache_control(msg)
    # 空列表 → content 不变（列表也保持空）
    assert out["content"] in ([], "")


def test_cache_breakpoint_shape():
    bp = cache_breakpoint()
    assert bp["type"] == "text"
    assert bp["text"] == ""
    assert bp["cache_control"] == {"type": "ephemeral"}


# ==================== Section F: 与 LLM 调用集成（mock） ====================


def test_multimodal_message_passes_through_invoke_kwargs():
    """ClearAgentLLM.invoke 把 messages 透传给 adapter；多模态消息应原样传过去"""
    from unittest.mock import MagicMock
    from clear_agent import ClearAgentLLM
    from clear_agent.core.llm_response import LLMResponse

    llm = ClearAgentLLM(
        model="gpt-4o", api_key="x", base_url="https://api.openai.com/v1"
    )
    fake_resp = LLMResponse(content="ok", model="gpt-4o", usage={})
    llm._adapter.invoke = MagicMock(return_value=fake_resp)

    msg = user_message([
        text_part("What's this?"),
        image_url_part("https://x/cat.jpg"),
    ])
    llm.invoke([msg])
    # adapter.invoke 被调用且 messages 内容正确
    call_args = llm._adapter.invoke.call_args
    passed_messages = call_args.args[0]
    assert passed_messages[0]["role"] == "user"
    assert passed_messages[0]["content"][0]["type"] == "text"
    assert passed_messages[0]["content"][1]["type"] == "image_url"


def test_cache_control_message_compatible_with_llm_invoke():
    """加了 cache_control 的消息应该能被 invoke 接受（adapter 透传）"""
    from unittest.mock import MagicMock
    from clear_agent import ClearAgentLLM
    from clear_agent.core.llm_response import LLMResponse

    llm = ClearAgentLLM(
        model="claude-3-5-sonnet",
        api_key="x",
        base_url="https://api.anthropic.com",
    )
    llm._adapter.invoke = MagicMock(
        return_value=LLMResponse(content="ok", model="c", usage={})
    )

    msg = with_cache_control(system_message("very long prompt"))
    llm.invoke([msg, user_message("hi")])

    call_args = llm._adapter.invoke.call_args
    passed = call_args.args[0]
    assert passed[0]["content"][-1].get("cache_control") == {"type": "ephemeral"}


# ==================== Section G: 顶层导入 ====================


def test_top_level_multimodal_imports():
    from clear_agent.core.multimodal import (
        text_part,
        image_url_part,
        image_base64_part,
        audio_part,
        file_part,
        user_message,
        system_message,
        assistant_message,
        with_cache_control,
        cache_breakpoint,
    )

    for fn in (
        text_part,
        image_url_part,
        image_base64_part,
        audio_part,
        file_part,
        user_message,
        system_message,
        assistant_message,
        with_cache_control,
        cache_breakpoint,
    ):
        assert callable(fn)
