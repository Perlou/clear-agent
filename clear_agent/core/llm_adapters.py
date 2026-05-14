"""LLM适配器 - 支持OpenAI、Anthropic、Gemini等不同接口格式"""

import time
import asyncio
import json
from abc import ABC, abstractmethod
from typing import Optional, Iterator, List, Dict, Any, Union, AsyncIterator, cast

from .llm_response import LLMResponse, StreamStats, LLMToolResponse, ToolCall
from .exceptions import ClearAgentException


class BaseLLMAdapter(ABC):
    """LLM适配器基类"""

    def __init__(
        self, api_key: str, base_url: Optional[str], timeout: int, model: str
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.model = model
        self._client: Any = None
        self._async_client: Any = None
        self.last_stats: Optional[StreamStats] = None

    @abstractmethod
    def create_client(self) -> Any:
        """创建客户端实例"""
        pass

    def create_async_client(self) -> Any:
        """创建异步客户端实例（子类可选实现）"""
        return None

    @abstractmethod
    def invoke(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> LLMResponse:
        """非流式调用"""
        pass

    @abstractmethod
    def stream_invoke(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Iterator[str]:
        """流式调用，返回生成器"""
        pass

    async def astream_invoke(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """异步流式调用（子类可选实现真正的异步）

        默认实现：使用队列 + 线程池包装同步流式方法
        """
        queue: asyncio.Queue[Union[str, Exception, None]] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _stream_to_queue() -> None:
            try:
                for chunk in self.stream_invoke(messages, **kwargs):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put(e), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        # 在线程池中运行同步流式方法
        loop.run_in_executor(None, _stream_to_queue)

        # 从队列中逐个取出 chunk
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    @abstractmethod
    def invoke_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> LLMToolResponse:
        """工具调用（Function Calling）"""
        pass

    def _is_thinking_model(self, model_name: str) -> bool:
        """判断是否为 thinking model（仅作启发式提示用，不再用于门控 reasoning_content 捕获）"""
        thinking_keywords = [
            "reasoner",
            "o1",
            "o3",
            "thinking",
            "v4-flash",
            "v4-pro",
            "qwq",
        ]
        model_lower = model_name.lower()
        return any(keyword in model_lower for keyword in thinking_keywords)

    # ==================== Reasoning artifact 策略 ====================
    #
    # 不同 provider 对 reasoning_content / thinking_blocks / thinking_signature
    # 的多轮回传协议并不统一：
    #   - DeepSeek V4 (thinking 模式)：必须把上一轮的 reasoning_content 回写到
    #     assistant message，否则 400 'must be passed back'
    #   - DeepSeek-R1 (deepseek-reasoner)：禁止回写，回写会 400
    #     'not allowed in conversation history'
    #   - OpenAI o1/o3：服务端自管 state，客户端无 reasoning_content
    #   - Anthropic extended thinking、Gemini 2.5 thinking：另有协议
    #
    # 因此 capture 总是做（始终尝试取 reasoning_content）；echo 由策略钩子决定。

    def _should_echo_reasoning(self) -> bool:
        """是否在多轮对话中把 reasoning_content 回写给 API。

        默认 ``True``（回写）。绝大多数 provider 要么要求回写、要么忽略它。
        子类可覆盖此方法处理特殊情况。
        """
        return True

    def serialize_assistant_message(
        self, response: "LLMToolResponse"
    ) -> Dict[str, Any]:
        """把 ``LLMToolResponse`` 序列化成下一轮请求要回传的 assistant message。

        默认实现按 OpenAI 协议生成 ``{role, content, tool_calls?, reasoning_content?}``；
        是否携带 ``reasoning_content`` 取决于 ``_should_echo_reasoning()``。

        Anthropic / Gemini 等使用不同协议的 adapter 可覆盖此方法。
        """
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": response.content,
        }
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in response.tool_calls
            ]
        if response.reasoning_content and self._should_echo_reasoning():
            msg["reasoning_content"] = response.reasoning_content
        return msg

    @staticmethod
    def _capture_reasoning_content(choice_or_message: Any) -> Optional[str]:
        """通用的 reasoning_content 捕获：不再依赖模型名单，只看字段是否存在且非空。

        覆盖以下结构：
        - OpenAI o1：``choice.message.reasoning_content``
        - DeepSeek：``choice.message.reasoning_content`` 或 ``choice.reasoning_content``
        - 部分 provider 把字段放在 message 上，部分放在 choice 上
        """
        if choice_or_message is None:
            return None
        message = getattr(choice_or_message, "message", choice_or_message)
        # 优先从 message 取
        rc = getattr(message, "reasoning_content", None)
        if rc:
            return str(rc)
        # 兜底从 choice 取
        rc = getattr(choice_or_message, "reasoning_content", None)
        return cast(Optional[str], rc or None)


class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI兼容接口适配器（默认）

    支持：
    - OpenAI官方API
    - 所有OpenAI兼容接口（DeepSeek、Qwen、Kimi、智谱等）
    - Thinking Models（o1、deepseek-reasoner、deepseek-v4-{flash,pro}、Qwen QwQ 等）
    """

    def _should_echo_reasoning(self) -> bool:
        """OpenAI 兼容协议下是否在多轮中回写 reasoning_content。

        - DeepSeek-R1 (``deepseek-reasoner``)：禁止回写，否则 400
          'reasoning_content is not allowed in conversation history'
        - 其余模型默认回写：DeepSeek-V4 thinking 必须回写；其余 provider
          看到这个字段会忽略，不影响调用。
        """
        m = (self.model or "").lower()
        if "reasoner" in m or "deepseek-r1" in m:
            return False
        return True

    def create_client(self) -> Any:
        """创建OpenAI客户端"""
        from openai import OpenAI

        return OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    def create_async_client(self) -> Any:
        """创建OpenAI异步客户端"""
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    def invoke(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> LLMResponse:
        """非流式调用"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()

        try:
            response = self._client.chat.completions.create(
                model=self.model, messages=messages, **kwargs
            )

            latency_ms = int((time.time() - start_time) * 1000)

            choice = response.choices[0]
            content = choice.message.content or ""
            # 通用捕获：不限于 thinking model，字段存在即提取
            reasoning_content = self._capture_reasoning_content(choice)

            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content=reasoning_content,
            )

        except Exception as e:
            raise ClearAgentException(f"OpenAI API调用失败: {str(e)}")

    def stream_invoke(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Iterator[str]:
        """流式调用"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()

        try:
            response = self._client.chat.completions.create(
                model=self.model, messages=messages, stream=True, **kwargs
            )

            collected_content = []
            reasoning_content = None
            usage = {}

            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    if delta.content:
                        collected_content.append(delta.content)
                        yield delta.content

                    # 通用捕获 reasoning_content（流式增量）
                    delta_rc = getattr(delta, "reasoning_content", None)
                    if delta_rc:
                        if reasoning_content is None:
                            reasoning_content = ""
                        reasoning_content += delta_rc

                if hasattr(chunk, "usage") and chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

            latency_ms = int((time.time() - start_time) * 1000)

            # 返回统计信息（存储到适配器，供外部获取）
            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content=reasoning_content,
            )

        except Exception as e:
            raise ClearAgentException(f"OpenAI API流式调用失败: {str(e)}")

    async def astream_invoke(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[str]:
        """真正的异步流式调用（使用 OpenAI 原生异步客户端）"""
        if not self._async_client:
            self._async_client = self.create_async_client()

        start_time = time.time()

        try:
            response = await self._async_client.chat.completions.create(
                model=self.model, messages=messages, stream=True, **kwargs
            )

            collected_content = []
            reasoning_content = None
            usage = {}

            async for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    if delta.content:
                        collected_content.append(delta.content)
                        yield delta.content

                    # 通用捕获 reasoning_content（流式增量）
                    delta_rc = getattr(delta, "reasoning_content", None)
                    if delta_rc:
                        if reasoning_content is None:
                            reasoning_content = ""
                        reasoning_content += delta_rc

                if hasattr(chunk, "usage") and chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }

            latency_ms = int((time.time() - start_time) * 1000)

            # 返回统计信息（存储到适配器，供外部获取）
            self.last_stats = StreamStats(
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content=reasoning_content,
            )

        except Exception as e:
            raise ClearAgentException(f"OpenAI API异步流式调用失败: {str(e)}")

    def invoke_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Union[str, Dict[str, Any]] = "auto",
        **kwargs: Any,
    ) -> LLMToolResponse:
        """工具调用（Function Calling）"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )

            latency_ms = int((time.time() - start_time) * 1000)
            choice = response.choices[0]
            message = choice.message

            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        )
                    )

            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMToolResponse(
                content=message.content,
                tool_calls=tool_calls,
                model=response.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content=self._capture_reasoning_content(choice),
            )

        except Exception as e:
            raise ClearAgentException(f"OpenAI Function Calling调用失败: {str(e)}")

    # ==================== 真异步 ====================

    async def ainvoke_async(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> LLMResponse:
        """真异步非流式（用 ``AsyncOpenAI``，不走线程池）"""
        if not self._async_client:
            self._async_client = self.create_async_client()
        start_time = time.time()
        try:
            response = await self._async_client.chat.completions.create(
                model=self.model, messages=messages, **kwargs
            )
            latency_ms = int((time.time() - start_time) * 1000)
            choice = response.choices[0]
            content = choice.message.content or ""
            # 通用捕获：不限于 thinking model，字段存在即提取
            reasoning_content = self._capture_reasoning_content(choice)
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content=reasoning_content,
            )
        except Exception as e:
            raise ClearAgentException(f"OpenAI 异步调用失败: {str(e)}")

    async def ainvoke_with_tools_async(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Union[str, Dict[str, Any]] = "auto",
        **kwargs: Any,
    ) -> LLMToolResponse:
        """真异步 Function Calling"""
        if not self._async_client:
            self._async_client = self.create_async_client()
        start_time = time.time()
        try:
            response = await self._async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )
            latency_ms = int((time.time() - start_time) * 1000)
            choice = response.choices[0]
            message = choice.message
            tool_calls = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        )
                    )
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return LLMToolResponse(
                content=message.content or "",
                tool_calls=tool_calls,
                model=response.model,
                usage=usage,
                latency_ms=latency_ms,
                reasoning_content=self._capture_reasoning_content(choice),
            )
        except Exception as e:
            raise ClearAgentException(f"OpenAI 异步 Function Calling 调用失败: {str(e)}")


class AnthropicAdapter(BaseLLMAdapter):
    """Anthropic Claude适配器

    处理Claude特有的消息格式：
    - system参数独立（不在messages中）
    - 消息格式转换
    """

    def create_client(self) -> Any:
        """创建Anthropic客户端"""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ClearAgentException("使用Anthropic需要安装: pip install anthropic")

        return Anthropic(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    @staticmethod
    def _convert_tool_schema(tool: Dict[str, Any]) -> Dict[str, Any]:
        """Convert OpenAI function schemas to Anthropic tool schemas."""
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            return {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters")
                or {"type": "object", "properties": {}},
            }

        converted = dict(tool)
        if "input_schema" not in converted:
            converted["input_schema"] = converted.pop(
                "parameters", {"type": "object", "properties": {}}
            )
        converted.pop("type", None)
        return converted

    @classmethod
    def _convert_tools(
        cls, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return [cls._convert_tool_schema(t) for t in tools]

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str) and arguments:
            try:
                parsed = json.loads(arguments)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    @classmethod
    def _convert_assistant_tool_message(cls, msg: Dict[str, Any]) -> Dict[str, Any]:
        content_blocks: List[Dict[str, Any]] = []
        content = msg.get("content")
        if isinstance(content, list):
            content_blocks.extend(content)
        elif content:
            content_blocks.append({"type": "text", "text": str(content)})

        for call in msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            name = fn.get("name") or call.get("name")
            arguments = fn.get("arguments", call.get("arguments"))
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": call.get("id"),
                    "name": name,
                    "input": cls._parse_tool_arguments(arguments),
                }
            )

        return {"role": "assistant", "content": content_blocks}

    @staticmethod
    def _convert_tool_result_message(msg: Dict[str, Any]) -> Dict[str, Any]:
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id") or msg.get("id"),
                    "content": content,
                }
            ],
        }

    def _convert_messages(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """转换消息格式，提取system消息"""
        system_content: Optional[str] = None
        converted_messages: List[Dict[str, Any]] = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = str(msg["content"])
            elif msg["role"] == "tool":
                converted_messages.append(self._convert_tool_result_message(msg))
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                converted_messages.append(self._convert_assistant_tool_message(msg))
            else:
                converted_messages.append(msg)

        return system_content, converted_messages

    def invoke(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> LLMResponse:
        """非流式调用"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()
        system_content, converted_messages = self._convert_messages(messages)

        try:
            request_params = {
                "model": self.model,
                "messages": converted_messages,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                **kwargs,
            }
            if system_content:
                request_params["system"] = system_content

            response = self._client.messages.create(**request_params)

            latency_ms = int((time.time() - start_time) * 1000)

            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text

            # 提取usage
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens
                    + response.usage.output_tokens,
                }

            return LLMResponse(
                content=content, model=self.model, usage=usage, latency_ms=latency_ms
            )

        except Exception as e:
            raise ClearAgentException(f"Anthropic API调用失败: {str(e)}")

    def stream_invoke(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Iterator[str]:
        """流式调用"""
        if not self._client:
            self._client = self.create_client()

        start_time = time.time()
        system_content, converted_messages = self._convert_messages(messages)

        try:
            request_params = {
                "model": self.model,
                "messages": converted_messages,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                "stream": True,
                **kwargs,
            }
            if system_content:
                request_params["system"] = system_content

            usage = {}

            with self._client.messages.stream(**request_params) as stream:
                for text in stream.text_stream:
                    yield text

                # 获取最终消息以提取usage
                final_message = stream.get_final_message()
                if hasattr(final_message, "usage") and final_message.usage:
                    usage = {
                        "prompt_tokens": final_message.usage.input_tokens,
                        "completion_tokens": final_message.usage.output_tokens,
                        "total_tokens": final_message.usage.input_tokens
                        + final_message.usage.output_tokens,
                    }

            latency_ms = int((time.time() - start_time) * 1000)

            self.last_stats = StreamStats(
                model=self.model, usage=usage, latency_ms=latency_ms
            )

        except Exception as e:
            raise ClearAgentException(f"Anthropic API流式调用失败: {str(e)}")

    def invoke_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> LLMToolResponse:
        """工具调用（Anthropic格式）"""
        if not self._client:
            self._client = self.create_client()

        kwargs.pop("tool_choice", None)
        system_content, converted_messages = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools)

        start_time = time.time()
        try:
            request_params = {
                "model": self.model,
                "messages": converted_messages,
                "tools": converted_tools,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                **kwargs,
            }
            if system_content:
                request_params["system"] = system_content

            response = self._client.messages.create(**request_params)
            latency_ms = int((time.time() - start_time) * 1000)

            content = ""
            tool_calls = []
            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=json.dumps(block.input),
                        )
                    )

            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens
                + response.usage.output_tokens,
            }

            return LLMToolResponse(
                content=content if content else None,
                tool_calls=tool_calls,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
            )

        except Exception as e:
            raise ClearAgentException(f"Anthropic工具调用失败: {str(e)}")

    # ==================== 真异步 ====================

    def create_async_client(self) -> Any:
        """创建 AsyncAnthropic 客户端"""
        try:
            from anthropic import AsyncAnthropic  # type: ignore
        except ImportError:
            raise ClearAgentException(
                "Anthropic 异步调用需要安装：pip install clear-agent[anthropic]"
            )
        return AsyncAnthropic(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    async def ainvoke_async(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> LLMResponse:
        """真异步非流式调用 Anthropic"""
        if not self._async_client:
            self._async_client = self.create_async_client()
        start_time = time.time()
        system_content, converted_messages = self._convert_messages(messages)
        try:
            request_params = {
                "model": self.model,
                "messages": converted_messages,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                **kwargs,
            }
            if system_content:
                request_params["system"] = system_content
            response = await self._async_client.messages.create(**request_params)
            latency_ms = int((time.time() - start_time) * 1000)
            content = ""
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    content += block.text
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
            return LLMResponse(
                content=content,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
            )
        except Exception as e:
            raise ClearAgentException(f"Anthropic 异步调用失败: {str(e)}")

    async def ainvoke_with_tools_async(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> LLMToolResponse:
        """真异步 Anthropic 工具调用"""
        if not self._async_client:
            self._async_client = self.create_async_client()
        # 移除 OpenAI 风格的 tool_choice（Anthropic 用 native tool_choice 但参数名不同）
        kwargs.pop("tool_choice", None)
        system_content, converted_messages = self._convert_messages(messages)
        converted_tools = self._convert_tools(tools)
        start_time = time.time()
        try:
            request_params = {
                "model": self.model,
                "messages": converted_messages,
                "tools": converted_tools,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                **kwargs,
            }
            if system_content:
                request_params["system"] = system_content
            response = await self._async_client.messages.create(**request_params)
            latency_ms = int((time.time() - start_time) * 1000)
            content = ""
            tool_calls = []
            for block in response.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    content += block.text
                elif btype == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=json.dumps(block.input),
                        )
                    )
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
            return LLMToolResponse(
                content=content if content else None,
                tool_calls=tool_calls,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
            )
        except Exception as e:
            raise ClearAgentException(f"Anthropic 异步工具调用失败: {str(e)}")


class GeminiAdapter(BaseLLMAdapter):
    """Google Gemini适配器

    处理Gemini特有的API格式
    使用新版 google.genai 包（替代已废弃的 google.generativeai）
    """

    def create_client(self) -> Any:
        """创建Gemini客户端"""
        try:
            from google import genai
        except ImportError:
            raise ClearAgentException("使用Gemini需要安装: pip install google-genai")

        client = genai.Client(api_key=self.api_key)
        return client

    def _convert_messages(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """转换消息格式"""
        system_instruction: Optional[str] = None
        converted_messages: List[Dict[str, Any]] = []

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = str(msg["content"])
            else:
                # Gemini使用 "user" 和 "model" 作为角色
                role = "model" if msg["role"] == "assistant" else "user"
                converted_messages.append(
                    {"role": role, "parts": [{"text": msg["content"]}]}
                )

        return system_instruction, converted_messages

    def invoke(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> LLMResponse:
        """非流式调用"""
        if not self._client:
            self._client = self.create_client()

        from google.genai import types as genai_types

        start_time = time.time()
        system_instruction, converted_messages = self._convert_messages(messages)

        try:
            # 创建生成配置
            config_params: Dict[str, Any] = {}
            if "temperature" in kwargs:
                config_params["temperature"] = kwargs.pop("temperature")
            if "max_tokens" in kwargs:
                config_params["max_output_tokens"] = kwargs.pop("max_tokens")
            if system_instruction:
                config_params["system_instruction"] = system_instruction

            response = self._client.models.generate_content(
                model=self.model,
                contents=converted_messages,
                config=genai_types.GenerateContentConfig(**cast(Any, config_params))
                if config_params
                else None,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            content = response.text if hasattr(response, "text") else ""

            # 提取usage
            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count
                    or 0,
                    "total_tokens": response.usage_metadata.total_token_count or 0,
                }

            return LLMResponse(
                content=content, model=self.model, usage=usage, latency_ms=latency_ms
            )

        except Exception as e:
            raise ClearAgentException(f"Gemini API调用失败: {str(e)}")

    def stream_invoke(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> Iterator[str]:
        """流式调用"""
        if not self._client:
            self._client = self.create_client()

        from google.genai import types as genai_types

        start_time = time.time()
        system_instruction, converted_messages = self._convert_messages(messages)

        try:
            # 创建生成配置
            config_params: Dict[str, Any] = {}
            if "temperature" in kwargs:
                config_params["temperature"] = kwargs.pop("temperature")
            if "max_tokens" in kwargs:
                config_params["max_output_tokens"] = kwargs.pop("max_tokens")
            if system_instruction:
                config_params["system_instruction"] = system_instruction

            usage = {}

            response = self._client.models.generate_content_stream(
                model=self.model,
                contents=converted_messages,
                config=genai_types.GenerateContentConfig(**cast(Any, config_params))
                if config_params
                else None,
            )

            for chunk in response:
                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text

                # 尝试提取usage（可能在最后一个chunk）
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    usage = {
                        "prompt_tokens": chunk.usage_metadata.prompt_token_count or 0,
                        "completion_tokens": chunk.usage_metadata.candidates_token_count
                        or 0,
                        "total_tokens": chunk.usage_metadata.total_token_count or 0,
                    }

            latency_ms = int((time.time() - start_time) * 1000)

            self.last_stats = StreamStats(
                model=self.model, usage=usage, latency_ms=latency_ms
            )

        except Exception as e:
            raise ClearAgentException(f"Gemini API流式调用失败: {str(e)}")

    def invoke_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> LLMToolResponse:
        """工具调用（Gemini格式）"""
        if not self._client:
            self._client = self.create_client()

        from google.genai import types as genai_types

        system_instruction, converted_messages = self._convert_messages(messages)

        start_time = time.time()
        try:
            # 转换工具格式为Gemini格式
            gemini_tools: List[Any] = []
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool["function"]
                    gemini_tools.append(
                        genai_types.FunctionDeclaration(
                            name=func["name"],
                            description=func.get("description", ""),
                            parameters=func.get("parameters", {}),
                        )
                    )

            config_params: Dict[str, Any] = {}
            if gemini_tools:
                config_params["tools"] = [
                    genai_types.Tool(function_declarations=gemini_tools)
                ]
            if system_instruction:
                config_params["system_instruction"] = system_instruction

            response = self._client.models.generate_content(
                model=self.model,
                contents=converted_messages,
                config=genai_types.GenerateContentConfig(**cast(Any, config_params))
                if config_params
                else None,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            content = response.text if hasattr(response, "text") else ""
            tool_calls = []

            # 解析 Gemini 工具调用
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{int(time.time() * 1000)}",  # Gemini 没有显式的 call_id，生成一个
                                name=part.function_call.name,
                                arguments=json.dumps(dict(part.function_call.args)),
                            )
                        )

            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                    "completion_tokens": response.usage_metadata.candidates_token_count
                    or 0,
                    "total_tokens": response.usage_metadata.total_token_count or 0,
                }

            return LLMToolResponse(
                content=content if content else None,
                tool_calls=tool_calls,
                model=self.model,
                usage=usage,
                latency_ms=latency_ms,
            )

        except Exception as e:
            raise ClearAgentException(f"Gemini工具调用失败: {str(e)}")

    # ==================== 真异步 ====================

    async def ainvoke_async(
        self, messages: List[Dict[str, Any]], **kwargs: Any
    ) -> LLMResponse:
        """异步调用 Gemini

        Gemini SDK 的 async API（``client.aio.models.generate_content``）
        在 google-genai 不同版本间形态不同，统一通过 ``asyncio.to_thread`` 包装
        同步方法 —— 至少不阻塞事件循环。
        """
        return await asyncio.to_thread(self.invoke, messages, **kwargs)

    async def ainvoke_with_tools_async(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> LLMToolResponse:
        """异步 Gemini 工具调用（同上，to_thread 包装）"""
        kwargs.pop("tool_choice", None)  # Gemini 不用 OpenAI 风格的 tool_choice
        return await asyncio.to_thread(
            self.invoke_with_tools, messages, tools, **kwargs
        )


def create_adapter(
    api_key: str, base_url: Optional[str], timeout: int, model: str
) -> BaseLLMAdapter:
    """
    根据base_url自动选择适配器

    检测逻辑：
    - anthropic.com -> AnthropicAdapter
    - googleapis.com 或 generativelanguage -> GeminiAdapter
    - 其他 -> OpenAIAdapter（默认）
    """
    if base_url:
        base_url_lower = base_url.lower()

        if "anthropic.com" in base_url_lower:
            return AnthropicAdapter(api_key, base_url, timeout, model)

        if "googleapis.com" in base_url_lower or "generativelanguage" in base_url_lower:
            return GeminiAdapter(api_key, base_url, timeout, model)

    # 默认使用OpenAI适配器（兼容所有OpenAI格式接口）
    return OpenAIAdapter(api_key, base_url, timeout, model)
