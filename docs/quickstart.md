# 快速开始

5 分钟跑通第一个 Agent。

## 安装

```bash
pip install clear-agent
```

按需扩展：
```bash
pip install "clear-agent[retrieval-qdrant,rag]"   # 完整 RAG
pip install "clear-agent[memory]"                  # 多层记忆
pip install "clear-agent[anthropic,gemini]"        # 多 provider
pip install "clear-agent[mcp]"                     # MCP 协议
```

## 配置

复制 `.env.example` 为 `.env`，填入 LLM 三件套：

```bash
LLM_MODEL_ID=gpt-4o
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
```

也可在代码里显式传：

```python
from clear_agent import ClearAgentLLM
llm = ClearAgentLLM(model="gpt-4o", api_key="sk-...", base_url="https://api.openai.com/v1")
```

## 第一个 Agent

```python
from clear_agent import ClearAgentLLM, ReActAgent, ToolRegistry, CalculatorTool

llm = ClearAgentLLM()                                    # 从 .env 自动加载
registry = ToolRegistry()
registry.register_tool(CalculatorTool())

agent = ReActAgent(name="demo", llm=llm, tool_registry=registry)
result = agent.run("计算 (123 + 456) * 2")
print(result)
```

## 4 种 Agent 范式

| 类 | 适用 |
|---|---|
| `SimpleAgent` | 纯对话或单轮工具调用 |
| `ReActAgent` | 推理-行动循环（推荐主力） |
| `ReflectionAgent` | 自我反思优化输出 |
| `PlanSolveAgent` | 先规划再分步执行 |

```python
from clear_agent import SimpleAgent, ReActAgent, ReflectionAgent, PlanSolveAgent
```

## 加 Checkpoint（崩了能续跑）

```python
from clear_agent import SqliteCheckpointer
from clear_agent.core.graph import RunConfig

graph = agent.as_graph(checkpointer=SqliteCheckpointer("memory/runs.db"))
graph.invoke(
    {"messages": [{"role": "user", "content": "..."}], "max_steps": 5},
    config=RunConfig(thread_id="user-42"),
)
# 进程崩了？重启后任意时间：
graph.resume("user-42")
```

## 自定义工具（最推荐：Pydantic 自动推导）

```python
from pydantic import BaseModel, Field
from clear_agent.tools.from_pydantic import pydantic_tool

class WeatherArgs(BaseModel):
    """查询天气"""
    city: str = Field(description="城市名")
    unit: str = Field(default="celsius", description="温度单位")

@pydantic_tool(description="获取指定城市天气")
def get_weather(args: WeatherArgs) -> str:
    return f"{args.city} 22°{args.unit[0].upper()}, sunny"

registry.register_tool(get_weather)
```

## 结构化输出

让 LLM 严格输出 Pydantic 对象：

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

structured = llm.with_structured_output(Person)
p = structured.invoke([{"role": "user", "content": "Alice 是 30 岁的老师"}])
print(p.name, p.age)
```

## 下一步

- [`graph-architecture.md`](graph-architecture.md) —— StateGraph + Reducer + Checkpoint 详解
- [`hitl.md`](hitl.md) —— 节点内暂停 + 等用户决策 + 续跑
- [`tool-system.md`](tool-system.md) —— 工具协议 / 自定义工具 / 熔断器
- [`rag-guide.md`](rag-guide.md) —— 完整 RAG 流水线（PDF/DOCX 加载到检索）
- [`memory-guide.md`](memory-guide.md) —— 多层记忆（短期 + 长期）
- [`multi-agent.md`](multi-agent.md) —— supervisor / swarm / handoff
- [`mcp.md`](mcp.md) —— 接入外部 MCP 工具 / 暴露给 Cursor
- [`structured-output.md`](structured-output.md) —— Pydantic 严格输出 + 重试
- [`eval-harness.md`](eval-harness.md) —— Dataset / Evaluator / Runner
- [`observability.md`](observability.md) —— TraceLogger + Callbacks + Metrics
