"""Multimodal + Prompt caching helpers

为 ``messages`` 列表提供构造多模态 content parts 与 prompt cache 标记的工具，
**不**修改各 adapter 的 invoke 接口（adapter 已经 ``**kwargs`` 透传），
直接用消息内容上的 schema 即可让各 provider 自动识别。

## Multimodal

OpenAI / Anthropic / Gemini 都支持 ``content`` 是数组形式的多模态消息。
本模块提供构造器，规避手写嵌套字典：

```python
from clear_agent.core.multimodal import (
    text_part, image_url_part, image_base64_part, audio_part, user_message,
)

msg = user_message([
    text_part("What's in this image?"),
    image_url_part("https://example.com/cat.jpg"),
])
response = llm.invoke([msg])
```

## Prompt caching

Anthropic 通过 ``cache_control`` 注解；OpenAI 自动隐式缓存（无需用户配置）。

```python
from clear_agent.core.multimodal import with_cache_control

system_msg = with_cache_control({
    "role": "system",
    "content": "very long system prompt...",
})
# system_msg["content"] 会被改成 list[block] 格式且最后一段带 cache_control
```

详见 plan §三 RC-W4 Multimodal / Prompt caching 与 GA 阶段。
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Literal, Optional, Union


# ==================== Multimodal content parts ====================


def text_part(text: str) -> Dict[str, Any]:
    """OpenAI / Anthropic / Gemini 共用的文本块（兼容三家）"""
    return {"type": "text", "text": text}


def image_url_part(
    url: str, detail: Optional[Literal["auto", "low", "high"]] = None
) -> Dict[str, Any]:
    """OpenAI 风格的 image_url 块（多数 OpenAI 兼容 provider 接受）

    ``detail`` 仅 OpenAI gpt-4o 等支持。
    """
    img: Dict[str, Any] = {"url": url}
    if detail is not None:
        img["detail"] = detail
    return {"type": "image_url", "image_url": img}


def image_base64_part(
    data: Union[str, bytes],
    media_type: str = "image/jpeg",
    provider: Literal["openai", "anthropic"] = "openai",
) -> Dict[str, Any]:
    """从字节或 base64 字符串构造图像块

    Args:
        data: 原始字节 / 已 base64 字符串
        media_type: ``image/jpeg`` / ``image/png`` / ``image/webp`` / ``image/gif``
        provider: 选 ``"openai"`` 用 ``image_url`` 协议（data: URL 形式）；
                  选 ``"anthropic"`` 用 ``source.type=base64`` 原生协议
    """
    if isinstance(data, bytes):
        b64 = base64.b64encode(data).decode("ascii")
    else:
        b64 = data

    if provider == "anthropic":
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64,
            },
        }
    # OpenAI: data URL
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{b64}"},
    }


def audio_part(
    data: Union[str, bytes],
    format: Literal["wav", "mp3"] = "wav",
) -> Dict[str, Any]:
    """OpenAI gpt-4o-audio 等支持的 audio 输入块

    Args:
        data: 原始字节 / 已 base64 字符串
        format: ``wav`` / ``mp3``
    """
    if isinstance(data, bytes):
        b64 = base64.b64encode(data).decode("ascii")
    else:
        b64 = data
    return {
        "type": "input_audio",
        "input_audio": {"data": b64, "format": format},
    }


def file_part(file_id: str) -> Dict[str, Any]:
    """OpenAI 文件 ID 引用块（先用 Files API 上传得到 file_id）"""
    return {"type": "file", "file": {"file_id": file_id}}


# ==================== 消息构造器 ====================


def _coerce_content_parts(
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """规整化为 ``List[content_part]``"""
    if isinstance(content, str):
        return [text_part(content)]
    if isinstance(content, dict):
        return [content]
    if isinstance(content, list):
        return list(content)
    raise TypeError(f"无法把 {type(content).__name__} 转为 content parts")


def user_message(
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """构造 user 消息，支持纯文本或 multipart"""
    return {"role": "user", "content": _coerce_content_parts(content)}


def system_message(
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """构造 system 消息（多数 provider 仍接受字符串，但统一用 list 形式更灵活）"""
    return {"role": "system", "content": _coerce_content_parts(content)}


def assistant_message(
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """构造 assistant 消息"""
    return {"role": "assistant", "content": _coerce_content_parts(content)}


# ==================== Prompt caching ====================


def with_cache_control(
    message: Dict[str, Any],
    cache_type: str = "ephemeral",
) -> Dict[str, Any]:
    """给 ``message["content"]`` 的最后一段 part 加 ``cache_control`` 注解

    Anthropic Claude 3.5+ 支持：在 system / 多轮 messages 上标记 cache point，
    后续 5 分钟内同 prefix 的请求享受 ~90% 折扣。

    若 ``content`` 是字符串，先转为 ``[text_part(s)]`` list 形式再加注解。

    Args:
        message: 标准 ClearAgent ``{"role", "content"}`` dict
        cache_type: ``"ephemeral"``（5 分钟）/ ``"persistent"``（1 小时；高级订阅）

    Returns:
        新 dict（不修改原 message）
    """
    new = dict(message)
    parts = _coerce_content_parts(new.get("content", ""))
    if not parts:
        return new
    new_parts = [dict(p) for p in parts]
    new_parts[-1]["cache_control"] = {"type": cache_type}
    new["content"] = new_parts
    return new


def cache_breakpoint() -> Dict[str, Any]:
    """返回一个空的 cache breakpoint 块，可插入 messages 列表中标记缓存点

    多数情况下用 ``with_cache_control(msg)`` 更简洁；本函数适合需要在
    单条 message 内部精确控制缓存边界的场景。
    """
    return {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}}


__all__ = [
    # parts
    "text_part",
    "image_url_part",
    "image_base64_part",
    "audio_part",
    "file_part",
    # messages
    "user_message",
    "system_message",
    "assistant_message",
    # caching
    "with_cache_control",
    "cache_breakpoint",
]
