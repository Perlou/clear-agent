# 结构化输出（2.0 用户向 quickstart）

> 设计 spec 详见 [`project_docs/04-structured-output.md`](../project_docs/04-structured-output.md)。

让 LLM 严格输出符合 Pydantic schema 的对象，一行打通。

## 1. 一行用法

```python
from pydantic import BaseModel
from clear_agent import ClearAgentLLM

class Person(BaseModel):
    name: str
    age: int
    occupation: str | None = None

llm = ClearAgentLLM()
structured = llm.with_structured_output(Person)

p: Person = structured.invoke([
    {"role": "user", "content": "Extract: Alice is 30 and works as teacher."}
])
print(p.name, p.age, p.occupation)  # Alice 30 teacher
```

## 2. 三种 method

| method | 适用 | 说明 |
|---|---|---|
| `function_calling`（默认非 OpenAI） | 所有支持 Function Calling 的 provider | 通过 `tool_choice` 强制工具调用，**最稳** |
| `json_mode` | OpenAI 兼容 | `response_format={"type":"json_object"}`，schema 通过 system prompt 提示 |
| `json_schema` | OpenAI gpt-4o-2024-08-06+ / gpt-4.1+ | `response_format={"type":"json_schema","strict":True,...}`，**最严格** |

`method="auto"`（默认）按 `model + base_url` 自动选：OpenAI 新模型走 `json_schema`，其他走 `function_calling`。

```python
llm.with_structured_output(Person, method="json_schema")  # 显式
llm.with_structured_output(Person, method="json_mode")
llm.with_structured_output(Person, method="function_calling")
```

## 3. 失败重试

schema 校验失败时，自动把错误信息追加到对话让 LLM 修正：

```python
structured = llm.with_structured_output(Person, max_retries=3)
# 第 1 次返回非法 → 第 2 次重试 → 仍不行抛 StructuredOutputError
```

## 4. include_raw 调试模式

需要同时拿到原始响应（usage / latency 等）：

```python
structured = llm.with_structured_output(Person, include_raw=True)
out = structured.invoke(messages)
out["parsed"]          # Person 实例（解析失败时为 None）
out["raw"]             # LLMResponse / LLMToolResponse
out["parsing_error"]   # 解析失败时的异常对象（成功时 None）
```

`include_raw=True` 模式下解析失败**不抛异常**，由调用方决定是否容忍。

## 5. 嵌套 / Optional / Enum 全支持

```python
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    HIGH = "high"

class Item(BaseModel):
    sku: str
    qty: int

class Order(BaseModel):
    order_id: str
    items: list[Item]            # 嵌套
    priority: Priority           # Enum
    note: str | None = None      # Optional + 默认值

s = llm.with_structured_output(Order)
order = s.invoke(...)
```

## 6. 异步对偶

```python
order = await s.ainvoke(messages)
```

## 7. 在 Eval-harness 里当裁判

`LLMAsJudge` 直接复用 `with_structured_output` —— 评分本身也是结构化输出：

```python
from pydantic import BaseModel
from clear_agent.eval import LLMAsJudge

class Score(BaseModel):
    score: float
    reasoning: str

ev = LLMAsJudge(
    llm=llm,
    rubric="Score 1.0 if accurate, 0.5 if partial, 0.0 if wrong",
    output_schema=Score,
    pass_threshold=0.7,
)
```

## 8. 兼容范围（2.0-α）

| Provider | function_calling | json_mode | json_schema |
|---|---|---|---|
| OpenAI gpt-4o-2024-08-06+ | ✅ | ✅ | ✅ |
| OpenAI gpt-4 / 3.5 | ✅ | ✅ | ❌ |
| DeepSeek / Qwen / Kimi / 智谱 | ✅ | ⚠️ 部分 | ❌ |
| Anthropic / Gemini | ⚠️ adapter 仓库内未实现，2.0-β 补 | | |

> 2.0-β 计划补 Anthropic / Gemini 的完整结构化输出。
