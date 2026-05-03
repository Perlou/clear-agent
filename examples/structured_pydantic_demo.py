"""结构化输出 + Pydantic 自动 Tool schema 演示

跑这个文件可看到：
1. ``llm.with_structured_output(MyModel)`` 让 LLM 严格输出 Pydantic 对象
2. ``@pydantic_tool`` 装饰器自动从 BaseModel 生成 OpenAI tool schema
3. 嵌套 / Optional / Enum 字段全支持

不调用真实 LLM —— 用 mock 演示调用路径。

运行：
    python examples/structured_pydantic_demo.py
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from unittest.mock import MagicMock

from pydantic import BaseModel, Field

from clear_agent.core.llm_response import LLMToolResponse, ToolCall
from clear_agent.core.structured import StructuredLLM
from clear_agent.tools.from_pydantic import pydantic_tool, tool_from_pydantic
from clear_agent.tools.registry import ToolRegistry


# ==================================================================
# Part 1: 结构化输出
# ==================================================================


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Task(BaseModel):
    """任务"""

    title: str = Field(description="任务标题")
    priority: Priority = Field(description="优先级")
    estimated_hours: float = Field(description="预估工时（小时）")
    tags: List[str] = Field(default_factory=list, description="标签列表")


def demo_structured_output() -> None:
    print("=" * 60)
    print("Part 1: 结构化输出 —— LLM 严格返回 Pydantic 对象")
    print("=" * 60)

    # mock 一个 LLM：function_calling 路径返回固定 tool_calls
    fake_llm = MagicMock()
    fake_llm.model = "gpt-4o-mock"
    fake_llm.base_url = "https://api.openai.com/v1"
    fake_llm.invoke_with_tools = MagicMock(
        return_value=LLMToolResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="t1",
                    name="Task",
                    arguments='{"title": "重构核心模块", "priority": "high", '
                    '"estimated_hours": 16.0, "tags": ["refactor", "core"]}',
                )
            ],
            model="gpt-4o-mock",
            usage={"total_tokens": 50},
        )
    )

    # 一行打通
    structured = StructuredLLM(
        llm=fake_llm,
        schema=Task,
        method="function_calling",
        max_retries=0,
    )

    result = structured.invoke(
        [
            {
                "role": "user",
                "content": "把「重构核心模块」拆成一个高优先级任务，预估 16 小时",
            }
        ]
    )
    print(f"  类型: {type(result).__name__}")
    print(f"  title: {result.title}")
    print(f"  priority: {result.priority.value}")
    print(f"  hours: {result.estimated_hours}")
    print(f"  tags: {result.tags}")


# ==================================================================
# Part 2: 自动 Tool schema 推导
# ==================================================================


class WeatherArgs(BaseModel):
    """查询天气"""

    city: str = Field(description="城市名（中文或英文）")
    unit: str = Field(default="celsius", description="温度单位 celsius / fahrenheit")


class CalcArgs(BaseModel):
    """计算"""

    expression: str = Field(description="数学表达式")


def demo_pydantic_tool() -> None:
    print()
    print("=" * 60)
    print("Part 2: @pydantic_tool —— 自动从 BaseModel 生成 schema")
    print("=" * 60)

    # 装饰器形态（推荐）
    @pydantic_tool(description="查询指定城市的当前天气")
    def get_weather(args: WeatherArgs) -> str:
        return f"{args.city}: 22°{'C' if args.unit == 'celsius' else 'F'}, sunny"

    # 包装器形态
    def safe_eval(args: CalcArgs) -> str:
        # 仅演示用；实际生产请用更安全的解析器
        try:
            result = eval(args.expression, {"__builtins__": {}})
            return f"= {result}"
        except Exception as e:
            return f"error: {e}"

    calc = tool_from_pydantic(
        name="calc",
        description="数学表达式求值",
        args_schema=CalcArgs,
        run_fn=safe_eval,
    )

    # 注册到 ToolRegistry
    reg = ToolRegistry()
    reg.register_tool(get_weather)
    reg.register_tool(calc)

    print(f"  已注册工具: {reg.list_tools()}")

    # 看 schema
    print()
    print("  get_weather 的 OpenAI schema:")
    schema = get_weather.to_openai_schema()
    fn = schema["function"]
    print(f"    name: {fn['name']}")
    print(f"    description: {fn['description']}")
    props = fn["parameters"].get("properties", {})
    for pname, pdef in props.items():
        req = pname in fn["parameters"].get("required", [])
        print(f"    param '{pname}': type={pdef.get('type')} required={req} desc={pdef.get('description')!r}")

    # 调用
    print()
    print("  调用 get_weather:")
    resp = get_weather.run({"city": "Paris", "unit": "celsius"})
    print(f"    status: {resp.status.value} / text: {resp.text}")

    resp = calc.run({"expression": "(123 + 456) * 2"})
    print(f"  调用 calc: {resp.text}")

    # 参数校验
    print()
    print("  非法参数自动校验:")
    bad = get_weather.run({"city": "Paris", "unit": 42})  # unit 应为字符串
    print(f"    status: {bad.status.value} / error: {bad.error_info}")


def main() -> None:
    demo_structured_output()
    demo_pydantic_tool()
    print()
    print("✅ Structured output + Pydantic tool demo 跑通")


if __name__ == "__main__":
    main()
