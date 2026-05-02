# 04 · 结构化输出（Structured Output）设计

> **阶段**：2.0-α / W4
> **目标文件**：`clear_agent/core/llm.py`（新增方法）、`clear_agent/core/structured.py`（schema 转换工具）
> **关联文档**：无（独立功能）

---

## 1. 设计目标

让用户用一行代码就能让 LLM 返回**严格符合 Pydantic schema 的结构化数据**，覆盖：
- JSON 提取（从自然语言中抽取实体、关系、字段）
- 工作流路由（让 LLM 在多个枚举选项中选择）
- 多字段决策（一次性返回多个相关字段）

**核心要求**：
- API：`llm.with_structured_output(MyPydanticModel)` 返回新 LLM 实例，`invoke()` 直接拿到 pydantic 对象
- 自动选择最适合当前 provider 的 method（function_calling / json_mode / json_schema）
- 失败可重试（schema 校验失败 → 自动喂错误信息让 LLM 修正）
- 不破坏现有 `invoke / stream / invoke_with_tools` 接口

**非目标**：
- 流式输出（结构化输出本期同步返回完整对象；流式 partial 解析放 2.0-β）
- 多 schema 同时（一次 with 一个 schema）

---

## 2. API 设计

### 2.1 基础用法

```python
from pydantic import BaseModel
from clear_agent import ClearAgentLLM

class Person(BaseModel):
    name: str
    age: int
    occupation: str | None = None

llm = ClearAgentLLM()
structured = llm.with_structured_output(Person)

result: Person = structured.invoke([
    {"role": "user", "content": "Extract: Alice is 30 years old and works as a teacher."}
])

print(result.name)        # "Alice"
print(result.age)         # 30
print(result.occupation)  # "teacher"
```

### 2.2 method 显式指定

```python
# 自动选择（默认，推荐）
llm.with_structured_output(Person)

# 显式 function calling
llm.with_structured_output(Person, method="function_calling")

# 显式 JSON 模式（OpenAI gpt-4o+ / 兼容接口）
llm.with_structured_output(Person, method="json_mode")

# 显式 JSON Schema 强制（OpenAI gpt-4o-2024-08-06+）
llm.with_structured_output(Person, method="json_schema")
```

### 2.3 includ_raw 模式

```python
# 同时返回原始响应（调试用）
structured = llm.with_structured_output(Person, include_raw=True)

result = structured.invoke(messages)
result["parsed"]      # Person 对象
result["raw"]         # LLMResponse（含 usage、reasoning_content 等）
result["parsing_error"]  # 解析失败时的错误（成功时 None）
```

### 2.4 异步对偶

```python
result = await structured.ainvoke(messages)
```

---

## 3. 实现策略

### 3.1 自动选择 method

```python
def _auto_method(model: str, base_url: str) -> str:
    # OpenAI gpt-4o-2024-08-06+ → json_schema（最严格）
    if model.startswith("gpt-4o") and "openai" in (base_url or ""):
        return "json_schema"
    # Anthropic → function_calling（tool_choice 强制）
    if "anthropic" in (base_url or "").lower():
        return "function_calling"
    # 其他 OpenAI 兼容（DeepSeek/Qwen/Kimi/Ollama）→ function_calling
    return "function_calling"
```

### 3.2 method = "function_calling"

```python
def _build_function_calling(schema: type[BaseModel]):
    return {
        "type": "function",
        "function": {
            "name": schema.__name__,
            "description": schema.__doc__ or f"Extract {schema.__name__}",
            "parameters": schema.model_json_schema(),
        },
    }

# 调用时
response = llm.invoke_with_tools(
    messages=messages,
    tools=[fn_schema],
    tool_choice={"type": "function", "function": {"name": schema.__name__}},
)
# 提取参数并验证
args = json.loads(response.tool_calls[0].arguments)
return schema.model_validate(args)
```

### 3.3 method = "json_schema"（OpenAI 严格模式）

```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "strict": True,
            "schema": schema.model_json_schema(),
        },
    },
)
return schema.model_validate_json(response.choices[0].message.content)
```

### 3.4 method = "json_mode"

仅强制返回有效 JSON，不约束 schema —— 需要在 system prompt 里写明 schema：

```python
system_prompt = f"You must respond with JSON matching this schema:\n{schema.model_json_schema()}"
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "system", "content": system_prompt}, *messages],
    response_format={"type": "json_object"},
)
return schema.model_validate_json(response.choices[0].message.content)
```

---

## 4. Provider 兼容矩阵

| Provider / Model | function_calling | json_mode | json_schema | 推荐 |
|---|---|---|---|---|
| OpenAI gpt-4o-2024-08-06+ | ✅ | ✅ | ✅ | json_schema |
| OpenAI gpt-4o（旧） | ✅ | ✅ | ❌ | function_calling |
| OpenAI gpt-4 / 3.5 | ✅ | ✅ | ❌ | function_calling |
| DeepSeek | ✅ | ✅ | ❌ | function_calling |
| Qwen / Kimi | ✅ | ⚠️ 部分 | ❌ | function_calling |
| 智谱 GLM-4 | ✅ | ✅ | ❌ | function_calling |
| Ollama（本地） | ⚠️ 看模型 | ⚠️ 看模型 | ❌ | function_calling |
| Anthropic Claude | ✅ tool_choice 强制 | ❌ | ❌ | function_calling |
| Gemini | ✅ | ✅ | ⚠️ 受限 | function_calling |

> **2.0-α 实施范围**：`function_calling` + `json_mode` + `json_schema` 三种方法在 OpenAI 兼容接口上完整实现。Anthropic / Gemini 适配器仓库内尚未实现 → **本期文档明示「仅支持 OpenAI 兼容」**，2.0-β 补。

---

## 5. 失败重试

```python
class StructuredLLM:
    def __init__(self, llm, schema, method, max_retries=2):
        self.max_retries = max_retries
        ...

    def invoke(self, messages):
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._raw_invoke(messages)
                return self.schema.model_validate_json(response)
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e
                if attempt < self.max_retries:
                    # 把错误信息追加进消息让 LLM 修正
                    messages = [
                        *messages,
                        {"role": "assistant", "content": str(response)},
                        {"role": "user", "content": f"Your output failed validation: {e}. Please fix and try again."},
                    ]
        raise StructuredOutputError(f"Failed after {self.max_retries+1} attempts: {last_err}")
```

---

## 6. 测试清单（W4 出口）

`tests/test_structured_output.py`：

| # | 测试 | 通过标准 |
|---|---|---|
| 1 | 简单 Pydantic schema | OpenAI 返回符合 schema 的对象 |
| 2 | 嵌套字段 | `class Order: items: list[Item]` 正确解析 |
| 3 | Optional 字段 | 缺失字段不报错，置为 default |
| 4 | Enum 字段 | 限定值 enum 字段返回有效值 |
| 5 | DeepSeek function_calling | DeepSeek 走 function_calling 通过 |
| 6 | OpenAI json_schema 严格 | gpt-4o-2024-08-06+ 走 json_schema 严格通过 |
| 7 | 失败重试 | mock 第一次返回非法 JSON，第二次合法 → 成功返回 |
| 8 | include_raw | 返回 dict 包含 parsed + raw + parsing_error 三键 |
| 9 | 异步等价 | `ainvoke` 与 `invoke` 同输入返回相同对象 |

---

## 7. 与现有代码的集成

无需修改：
- `llm_adapters.py` 已有 `invoke_with_tools` → function_calling 直接复用
- `LLMResponse` 已有 `usage / latency_ms` → StructuredLLM 透传

新增：
- `clear_agent/core/llm.py` 新增 `ClearAgentLLM.with_structured_output(schema, method="auto", include_raw=False, max_retries=2)`
- `clear_agent/core/structured.py` 新增 `StructuredLLM` class
- `Config` 新增 `structured_output_max_retries: int = 2`

不破坏：
- 旧 `invoke / stream / invoke_with_tools` 完全保留
- `with_structured_output` 返回**新实例**，不污染原 llm

---

## 8. 待决问题

1. **method="auto" 默认是否启用 strict（json_schema for gpt-4o-2024-08-06+）？**
   - 严格模式更可靠但对 schema 有限制（不支持 `Optional` 默认值、union 类型受限）
   - **建议**：默认启用，schema 不兼容时降级到 function_calling 并打 WARNING

2. **重试时是否计入 token / 成本？**
   - 计入（透传到 trace）

3. **schema 内 `Field(description=...)` 是否参与 prompt？**
   - 是，function_calling 路径里 description 自动注入到 OpenAI tool schema 的 properties 描述

4. **支持 `dict` schema 而不仅 BaseModel？**
   - **建议**：第一版只支持 BaseModel；2.0-β 加 dict schema 支持

请逐项确认。
