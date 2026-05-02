"""结构化输出（Structured Output）支持

让 LLM 严格输出符合 Pydantic schema 的对象，覆盖三种 method：
- ``function_calling``：通过 OpenAI Function Calling + ``tool_choice`` 强制工具调用，
  解析 tool_calls[0].arguments → schema.model_validate
- ``json_mode``：传 ``response_format={"type":"json_object"}``，
  在 system prompt 里附带 schema 提示
- ``json_schema``：传 ``response_format={"type":"json_schema","strict":True,...}``
  （仅 OpenAI gpt-4o-2024-08-06+）

所有 method 在解析失败时按 ``max_retries`` 自动重试（把错误信息追加到对话让 LLM 修正）。

详见 project_docs/04-structured-output.md
"""

from __future__ import annotations

import json
import re
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

try:
    from pydantic import BaseModel, ValidationError
except ImportError as e:  # pragma: no cover
    raise ImportError("结构化输出需要 pydantic（已是 ClearAgent 默认依赖）") from e

from .exceptions import ClearAgentException
from .llm_response import LLMResponse, LLMToolResponse

if TYPE_CHECKING:
    from .llm import ClearAgentLLM


T = TypeVar("T", bound=BaseModel)

Method = Literal["auto", "function_calling", "json_mode", "json_schema"]


# ==================== 异常 ====================


class StructuredOutputError(ClearAgentException):
    """结构化输出在 ``max_retries`` 次重试后仍未成功"""

    def __init__(self, message: str, last_error: Optional[Exception] = None):
        super().__init__(message)
        self.last_error = last_error


# ==================== 自动 method 选择 ====================


def _auto_method(model: str, base_url: Optional[str]) -> str:
    """根据 model + base_url 选最合适的 method

    规则：
    - OpenAI 官方 gpt-4o-2024-08-06+ → ``json_schema``（最严格）
    - 其余 OpenAI 兼容 / Anthropic / Gemini → ``function_calling``
    """
    base = (base_url or "").lower()
    m = (model or "").lower()
    # OpenAI 官方且模型支持 strict json schema
    if "openai.com" in base or "api.openai.com" in base:
        # gpt-4o-2024-08-06 及更新 / gpt-4.1 / o1 / gpt-5 等都支持
        # 简化规则：包含 gpt-4o 或更新代号即可
        if any(tag in m for tag in ("gpt-4o", "gpt-4.1", "gpt-5", "o3-")):
            return "json_schema"
    # 默认走 function_calling（Anthropic / DeepSeek / Qwen / Kimi / Ollama 均支持）
    return "function_calling"


# ==================== schema 工具 ====================


def _schema_to_function(schema: Type[BaseModel]) -> Dict[str, Any]:
    """把 Pydantic 模型转成 OpenAI function-calling schema"""
    return {
        "type": "function",
        "function": {
            "name": schema.__name__,
            "description": (schema.__doc__ or f"Extract {schema.__name__} fields.").strip(),
            "parameters": schema.model_json_schema(),
        },
    }


def _strip_json_fence(text: str) -> str:
    """剥掉 ```json ... ``` 之类的代码围栏"""
    text = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text


def _extract_first_json_object(text: str) -> Optional[str]:
    """宽容兜底：从一段杂文里抽取第一段 ``{...}``，不保证完美"""
    text = _strip_json_fence(text)
    # 先尝试整段
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    # 抽 {...}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return None


# ==================== 核心 StructuredLLM ====================


class StructuredLLM:
    """让 ``ClearAgentLLM`` 输出符合给定 ``BaseModel`` 的实例

    一般通过 ``llm.with_structured_output(schema)`` 创建，不直接构造。
    """

    def __init__(
        self,
        llm: "ClearAgentLLM",
        schema: Type[T],
        method: Method = "auto",
        include_raw: bool = False,
        max_retries: int = 2,
    ):
        if method == "auto":
            method = _auto_method(llm.model or "", llm.base_url)
        if method not in ("function_calling", "json_mode", "json_schema"):
            raise ClearAgentException(f"不支持的 structured output method: {method}")
        self.llm = llm
        self.schema = schema
        self.method: str = method
        self.include_raw = include_raw
        self.max_retries = max_retries

    # ---------- 底层单次调用：返回 (raw_text, raw_response) ----------

    def _call_function_calling(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> tuple[str, Union[LLMToolResponse, LLMResponse]]:
        fn = _schema_to_function(self.schema)
        tool_choice = {
            "type": "function",
            "function": {"name": self.schema.__name__},
        }
        resp = self.llm.invoke_with_tools(
            messages=messages, tools=[fn], tool_choice=tool_choice, **kwargs
        )
        if not resp.tool_calls:
            # 模型没听话调工具，返回 content 让上层试一次解析（多半会失败 → 触发重试）
            return resp.content or "", resp
        return resp.tool_calls[0].arguments or "", resp

    def _call_json_mode(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> tuple[str, LLMResponse]:
        # 在 system 消息里附 schema 描述
        schema_hint = (
            "你必须仅以 JSON 对象返回，且严格符合以下 JSON Schema：\n"
            + json.dumps(self.schema.model_json_schema(), ensure_ascii=False)
        )
        injected = self._inject_system_prompt(messages, schema_hint)
        resp = self.llm.invoke(
            injected, response_format={"type": "json_object"}, **kwargs
        )
        return resp.content or "", resp

    def _call_json_schema(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> tuple[str, LLMResponse]:
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": self.schema.__name__,
                "strict": True,
                "schema": self.schema.model_json_schema(),
            },
        }
        resp = self.llm.invoke(messages, response_format=rf, **kwargs)
        return resp.content or "", resp

    @staticmethod
    def _inject_system_prompt(
        messages: List[Dict[str, Any]], extra: str
    ) -> List[Dict[str, Any]]:
        """在 messages 顶部追加（或合并）一段 system 提示"""
        new = list(messages)
        if new and new[0].get("role") == "system":
            new[0] = {
                "role": "system",
                "content": (new[0].get("content") or "") + "\n\n" + extra,
            }
        else:
            new.insert(0, {"role": "system", "content": extra})
        return new

    # ---------- 解析单次结果 ----------

    def _parse(self, raw_text: str) -> T:
        text = raw_text or ""
        # function_calling 直接返回 JSON 字符串
        # json_mode / json_schema 也是 JSON 字符串；偶尔模型加了 fence
        text = _strip_json_fence(text)
        try:
            return self.schema.model_validate_json(text)
        except (ValidationError, ValueError, json.JSONDecodeError):
            pass
        # 兜底：先抽对象再 parse
        candidate = _extract_first_json_object(text)
        if candidate is not None:
            return self.schema.model_validate_json(candidate)
        # 抛原始错误以便重试 prompt
        return self.schema.model_validate_json(text)  # 触发原始异常上抛

    # ---------- 公共入口（同步） ----------

    def invoke(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Union[T, Dict[str, Any]]:
        return self._loop(messages, async_mode=False, **kwargs)  # type: ignore[return-value]

    # ---------- 公共入口（异步） ----------

    async def ainvoke(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Union[T, Dict[str, Any]]:
        # 借同步 _loop 的逻辑——LLM 调用走 ainvoke / ainvoke_with_tools
        return await self._aloop(messages, **kwargs)  # type: ignore[return-value]

    # ---------- 同步主循环 ----------

    def _loop(
        self, messages: List[Dict[str, Any]], async_mode: bool, **kwargs: Any
    ) -> Any:
        cur_messages = list(messages)
        last_err: Optional[Exception] = None
        last_raw: Optional[Union[LLMResponse, LLMToolResponse]] = None
        last_text = ""

        for attempt in range(self.max_retries + 1):
            try:
                if self.method == "function_calling":
                    text, raw = self._call_function_calling(cur_messages, **kwargs)
                elif self.method == "json_mode":
                    text, raw = self._call_json_mode(cur_messages, **kwargs)
                else:  # json_schema
                    text, raw = self._call_json_schema(cur_messages, **kwargs)

                last_text = text
                last_raw = raw
                parsed = self._parse(text)
                if self.include_raw:
                    return {"parsed": parsed, "raw": raw, "parsing_error": None}
                return parsed
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                last_err = e
                if attempt < self.max_retries:
                    cur_messages = list(cur_messages) + [
                        {"role": "assistant", "content": last_text},
                        {
                            "role": "user",
                            "content": (
                                f"Your previous output failed validation: {e}. "
                                "Please fix and return only valid output matching the schema."
                            ),
                        },
                    ]
                    continue
                break

        if self.include_raw:
            return {"parsed": None, "raw": last_raw, "parsing_error": last_err}
        raise StructuredOutputError(
            f"Failed to produce valid {self.schema.__name__} after "
            f"{self.max_retries + 1} attempts: {last_err}",
            last_error=last_err,
        )

    # ---------- 异步主循环 ----------

    async def _aloop(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Any:
        cur_messages = list(messages)
        last_err: Optional[Exception] = None
        last_raw: Optional[Union[LLMResponse, LLMToolResponse]] = None
        last_text = ""

        for attempt in range(self.max_retries + 1):
            try:
                if self.method == "function_calling":
                    fn = _schema_to_function(self.schema)
                    tool_choice = {
                        "type": "function",
                        "function": {"name": self.schema.__name__},
                    }
                    raw = await self.llm.ainvoke_with_tools(
                        cur_messages, [fn], tool_choice=tool_choice, **kwargs
                    )
                    text = (
                        raw.tool_calls[0].arguments
                        if raw.tool_calls
                        else (raw.content or "")
                    )
                elif self.method == "json_mode":
                    schema_hint = (
                        "你必须仅以 JSON 对象返回，且严格符合以下 JSON Schema：\n"
                        + json.dumps(self.schema.model_json_schema(), ensure_ascii=False)
                    )
                    injected = self._inject_system_prompt(cur_messages, schema_hint)
                    raw = await self.llm.ainvoke(
                        injected, response_format={"type": "json_object"}, **kwargs
                    )
                    text = raw.content or ""
                else:  # json_schema
                    rf = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": self.schema.__name__,
                            "strict": True,
                            "schema": self.schema.model_json_schema(),
                        },
                    }
                    raw = await self.llm.ainvoke(
                        cur_messages, response_format=rf, **kwargs
                    )
                    text = raw.content or ""

                last_text = text
                last_raw = raw
                parsed = self._parse(text)
                if self.include_raw:
                    return {"parsed": parsed, "raw": raw, "parsing_error": None}
                return parsed
            except (ValidationError, json.JSONDecodeError, ValueError) as e:
                last_err = e
                if attempt < self.max_retries:
                    cur_messages = list(cur_messages) + [
                        {"role": "assistant", "content": last_text},
                        {
                            "role": "user",
                            "content": (
                                f"Your previous output failed validation: {e}. "
                                "Please fix and return only valid output matching the schema."
                            ),
                        },
                    ]
                    continue
                break

        if self.include_raw:
            return {"parsed": None, "raw": last_raw, "parsing_error": last_err}
        raise StructuredOutputError(
            f"Failed to produce valid {self.schema.__name__} after "
            f"{self.max_retries + 1} attempts: {last_err}",
            last_error=last_err,
        )


__all__ = [
    "StructuredLLM",
    "StructuredOutputError",
    "Method",
    "_auto_method",
    "_schema_to_function",
]
