"""从 Pydantic 模型自动推导 ClearAgent ``Tool``

让用户不必手写 ``to_openai_schema`` / ``get_parameters`` —— 直接定义 Pydantic
模型描述参数，再绑一个执行函数即可。

两种用法：

1. **装饰器形态**（推荐）::

    from pydantic import BaseModel, Field
    from clear_agent.tools.from_pydantic import pydantic_tool

    class Args(BaseModel):
        '''计算两数之和'''
        a: int = Field(description="第一个数")
        b: int = Field(description="第二个数")

    @pydantic_tool(name="add", description="加法")
    def add(args: Args) -> int:
        return args.a + args.b

    registry.register_tool(add)   # add 已是 ClearAgent Tool

2. **包装器形态**::

    add_tool = tool_from_pydantic(
        name="add",
        description="加法",
        args_schema=Args,
        run_fn=lambda args: args.a + args.b,
    )

特性：
- 自动从 ``BaseModel.model_json_schema()`` 转 OpenAI function schema
- 自动从 ``Field(description=...)`` 抽取参数描述
- ``run_fn`` 可返回 ``ToolResponse`` 或任意值（自动包成 success）
- 同步 + async ``run_fn`` 通吃
- 类型检查：调用前用 ``args_schema.model_validate(parameters)`` 自动校验入参
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Union

try:
    from pydantic import BaseModel, ValidationError
except ImportError as e:  # pragma: no cover
    raise ImportError("from_pydantic 需要 pydantic（已是 ClearAgent 默认依赖）") from e

from .base import Tool, ToolParameter
from .response import ToolResponse


def _params_from_pydantic(schema: Type[BaseModel]) -> List[ToolParameter]:
    """从 Pydantic schema 抽取 ``ToolParameter`` 列表

    依赖 ``model_json_schema()``：
    - ``properties[name].type`` → ``ToolParameter.type``
    - ``properties[name].description`` → ``description``
    - ``properties[name].default`` → ``default``（无 default 即 required=True）
    - ``required: [...]`` 顶层字段
    """
    js = schema.model_json_schema()
    props = js.get("properties", {}) or {}
    required = set(js.get("required", []) or [])
    out: List[ToolParameter] = []
    for name, pdef in props.items():
        if not isinstance(pdef, dict):
            continue
        ptype = pdef.get("type", "string")
        # Pydantic 的 anyOf 或 ref 时 fallback 到 string
        if not isinstance(ptype, str):
            ptype = "string"
        pdesc = pdef.get("description", "") or ""
        pdefault = pdef.get("default", None)
        out.append(
            ToolParameter(
                name=name,
                type=ptype,
                description=pdesc,
                required=name in required,
                default=pdefault,
            )
        )
    return out


def _function_schema_from_pydantic(
    name: str, description: str, schema: Type[BaseModel]
) -> Dict[str, Any]:
    """构造 OpenAI function calling schema"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or schema.__doc__ or f"{name} tool",
            "parameters": schema.model_json_schema(),
        },
    }


class _PydanticTool(Tool):
    """内部实现：把 Pydantic schema + run_fn 包成 ``Tool``"""

    def __init__(
        self,
        name: str,
        description: str,
        args_schema: Type[BaseModel],
        run_fn: Callable[[Any], Any],
        validate_args: bool = True,
    ):
        super().__init__(name=name, description=description)
        self.args_schema = args_schema
        self.run_fn = run_fn
        self.validate_args = validate_args
        self._is_async_run = inspect.iscoroutinefunction(run_fn)

    def get_parameters(self) -> List[ToolParameter]:
        return _params_from_pydantic(self.args_schema)

    def to_openai_schema(self) -> Dict[str, Any]:
        return _function_schema_from_pydantic(
            self.name, self.description, self.args_schema
        )

    def _coerce_args(self, parameters: Dict[str, Any]) -> Any:
        """同步入口：把 dict 参数验证成 Pydantic 实例"""
        if not self.validate_args:
            return parameters
        try:
            return self.args_schema.model_validate(parameters)
        except ValidationError as e:
            raise ValueError(f"参数校验失败: {e}") from e

    def _coerce_result(self, result: Any) -> ToolResponse:
        """统一返回 ``ToolResponse``：函数返回值 != ToolResponse 时自动包装"""
        if isinstance(result, ToolResponse):
            return result
        # 字符串 / 数字 / 容器 → text + data
        text = result if isinstance(result, str) else None
        data = None
        if isinstance(result, (dict, list, tuple)):
            data = {"result": result}
            text = text or str(result)
        return ToolResponse.success(
            text=text if text is not None else str(result),
            data=data,
        )

    def run(self, parameters: Dict[str, Any]) -> ToolResponse:
        try:
            args = self._coerce_args(parameters)
        except ValueError as e:
            return ToolResponse.error(code="INVALID_ARGS", message=str(e))

        try:
            if self._is_async_run:
                # async run_fn 在同步入口里走 asyncio.run
                import asyncio

                result = asyncio.run(self.run_fn(args))
            else:
                result = self.run_fn(args)
        except Exception as e:
            return ToolResponse.error(
                code="TOOL_FAILED",
                message=f"{self.name} 执行失败: {e}",
            )

        return self._coerce_result(result)

    async def arun(self, parameters: Dict[str, Any]) -> ToolResponse:
        """async 入口"""
        try:
            args = self._coerce_args(parameters)
        except ValueError as e:
            return ToolResponse.error(code="INVALID_ARGS", message=str(e))

        try:
            if self._is_async_run:
                result = await self.run_fn(args)
            else:
                # 同步 run_fn 在 async 入口里直接调（用户自己负责不阻塞）
                result = self.run_fn(args)
                if inspect.isawaitable(result):
                    result = await result
        except Exception as e:
            return ToolResponse.error(
                code="TOOL_FAILED",
                message=f"{self.name} 执行失败: {e}",
            )

        return self._coerce_result(result)


# ==================== 公开 API ====================


def tool_from_pydantic(
    name: str,
    description: str,
    args_schema: Type[BaseModel],
    run_fn: Callable[[Any], Any],
    validate_args: bool = True,
) -> Tool:
    """从 Pydantic schema + 函数构造 ClearAgent ``Tool``

    Args:
        name: 工具名（OpenAI function name 限制：字母/数字/_/-）
        description: 工具描述
        args_schema: 描述参数的 Pydantic ``BaseModel`` 子类
        run_fn: 接受 ``args_schema`` 实例，返回任意值或 ``ToolResponse``
        validate_args: 是否在调用前用 args_schema 校验入参（默认 True）

    Returns:
        ``Tool`` 实例（可直接 ``registry.register_tool(...)``）
    """
    return _PydanticTool(
        name=name,
        description=description,
        args_schema=args_schema,
        run_fn=run_fn,
        validate_args=validate_args,
    )


def pydantic_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    args_schema: Optional[Type[BaseModel]] = None,
    validate_args: bool = True,
) -> Callable[[Callable[..., Any]], Tool]:
    """装饰器形态

    Args:
        name: 工具名（默认取被装饰函数的 ``__name__``）
        description: 工具描述（默认取函数 docstring）
        args_schema: Pydantic schema；缺省自动从函数第一个参数的类型注解推断
        validate_args: 是否校验入参

    Example::

        class AddArgs(BaseModel):
            a: int
            b: int

        @pydantic_tool(name="add", description="加法")
        def add(args: AddArgs) -> int:
            return args.a + args.b
    """

    def deco(fn: Callable[..., Any]) -> Tool:
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__ or f"{tool_name} tool").strip()

        # 自动推断 args_schema
        schema = args_schema
        if schema is None:
            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            if not params:
                raise ValueError(
                    f"@pydantic_tool 函数 '{tool_name}' 需要至少一个 BaseModel 类型的参数"
                )
            first_param = params[0]
            ann = first_param.annotation

            # PEP 563 兼容：注解可能是字符串（``from __future__ import annotations``），
            # 用 typing.get_type_hints 解析为真实类型
            if isinstance(ann, str) or ann is inspect.Parameter.empty:
                try:
                    import typing as _typing

                    hints = _typing.get_type_hints(fn)
                    ann = hints.get(first_param.name, ann)
                except Exception:
                    pass

            if (
                ann is inspect.Parameter.empty
                or not (isinstance(ann, type) and issubclass(ann, BaseModel))
            ):
                raise ValueError(
                    f"@pydantic_tool 函数 '{tool_name}' 的第一个参数必须有 "
                    "BaseModel 类型注解，或显式传 args_schema=..."
                )
            schema = ann

        return tool_from_pydantic(
            name=tool_name,
            description=tool_desc,
            args_schema=schema,
            run_fn=fn,
            validate_args=validate_args,
        )

    return deco


__all__ = [
    "tool_from_pydantic",
    "pydantic_tool",
]
