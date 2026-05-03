# 10 · GA 阶段交付 spec

> **阶段**：2.0 GA（W1-W4）
> **状态**：已完成 ✅ —— **`clear-agent==2.0.0` 正式发布**
> **关联文档**：00（决策）、08（β spec）、09（RC spec）

---

## 1. GA 阶段交付摘要

| 周 | 子任务 | 新增 LOC | 测试数 |
|---|---|---|---|
| W1 | Resilience（Retry/Fallback/负载均衡）+ Pydantic Tool 自动推导 | ~580 | 47 |
| W2 | Anthropic + Gemini 真异步路径补全 | ~140 | 14 |
| W3 | Multimodal（vision/audio）+ Prompt caching helpers | ~190 | 26 |
| W4 | 升 2.0.0 + 完整文档收尾 | ~150 (docs) | – |
| **GA 累计** | | **~1060 LOC** | **87** |
| 全量回归 | W1-α + W1-β + W1-RC + W1-GA | – | **744 passed** |

---

## 2. GA-W1: Resilience + Pydantic Tool 推导

### 2.1 Resilience（`core/resilience.py`，~280 行）

按 plan §三 GA 决策：**零外部依赖**（不引入 ``tenacity``）。

- `RetryPolicy` dataclass：max_attempts / backoff / max_backoff / jitter / retry_on / on_retry
- `with_retry(fn, ...)` / `@retry(...)` 同步 + `with_retry_async(fn, ...)` / `@aretry(...)` 异步
- 指数退避 + 抖动（防雷击）+ 异常白名单（仅指定异常类型才重试）
- `with_fallbacks(primary, [fb1, fb2, ...])` / `with_fallbacks_async(...)` —— 主调失败按序回退
- `round_robin([candidates])` / `random_choice([candidates], seed)` —— 简易负载均衡
- ``on_retry`` / ``on_fallback`` 回调用于日志 / 监控；回调异常自动吞掉

### 2.2 Pydantic 自动 Tool schema 推导（`tools/from_pydantic.py`，~280 行）

替代当前手写 `to_openai_schema` / `get_parameters`：

- `@pydantic_tool(name, description)` 装饰器形态（推荐）
- `tool_from_pydantic(name, description, args_schema, run_fn)` 包装器形态
- 自动从 `BaseModel.model_json_schema()` 转 OpenAI function spec
- 自动从 `Field(description=...)` 抽取参数描述
- `validate_args=True` 调用前用 schema 自动校验入参；失败返回 `ToolResponse.error(code="INVALID_ARGS")`
- 同步 + async `run_fn` 通吃；返回值非 `ToolResponse` 时自动包成 success
- **PEP 563 兼容**：用 `typing.get_type_hints` 解析字符串注解（兼容 `from __future__ import annotations`）
- 是 `Tool` 子类，可直接 `registry.register_tool(...)`

**用户体验对比**：

```python
# 1.x 手写（~30 行 boilerplate）
class CalcTool(Tool):
    def __init__(self): super().__init__(name="calc", description="...")
    def get_parameters(self): return [ToolParameter(name="a", type="integer", ...), ...]
    def to_openai_schema(self): return {...嵌套字典...}
    def run(self, parameters): return ToolResponse.success(text=str(...))

# 2.0 GA 一行
@pydantic_tool(description="加法")
def add(args: AddArgs) -> int:
    return args.a + args.b
```

## 3. GA-W2: Anthropic + Gemini 真异步路径

补全 RC-W3 真异步 OpenAI 之外的两个主流 provider：

### 3.1 Anthropic 真异步

- `AnthropicAdapter.create_async_client()` 用 `AsyncAnthropic`
- `ainvoke_async` —— 真异步 `messages.create`，自动提取 system 消息到顶层 `system` 参数
- `ainvoke_with_tools_async` —— 真异步 tool use，把 `content blocks` 解析成 `LLMToolResponse.tool_calls`
- 自动剥离 OpenAI 风格的 `tool_choice`（Anthropic SDK 不接受该参数）
- 缺包时 `ClearAgentException` 友好提示：`pip install clear-agent[anthropic]`

### 3.2 Gemini 真异步

`google-genai` SDK 的 async API 在不同版本间形态不稳定，本期采用 `asyncio.to_thread` 包装同步方法 —— **不阻塞事件循环**，行为对 ClearAgentLLM 透明：

- `ainvoke_async` → `asyncio.to_thread(self.invoke, ...)`
- `ainvoke_with_tools_async` 同上
- 未来 SDK API 稳定后可一行替换为真异步 `client.aio.models.generate_content_async`

### 3.3 ClearAgentLLM 自动选路径

`ClearAgentLLM.ainvoke / ainvoke_with_tools` 优先调 `adapter.ainvoke_async / ainvoke_with_tools_async`（任意 provider 命中），缺失时 graceful fallback 到线程池包装。

## 4. GA-W3: Multimodal + Prompt caching

`core/multimodal.py`（~190 行）—— **不修改 adapter 接口**（adapter 已 `**kwargs` 透传），用 messages content 上的 schema 让各 provider 自动识别。

### 4.1 Content parts 构造器

| 函数 | 用途 | provider |
|---|---|---|
| `text_part(text)` | 文本块 | OpenAI / Anthropic / Gemini |
| `image_url_part(url, detail)` | 远程图片 | OpenAI 兼容 |
| `image_base64_part(data, media_type, provider)` | 内嵌图片 | OpenAI（data URL）/ Anthropic（native source） |
| `audio_part(data, format)` | 音频输入 | OpenAI gpt-4o-audio |
| `file_part(file_id)` | 文件引用 | OpenAI Files API |

### 4.2 消息构造器

`user_message(content)` / `system_message(content)` / `assistant_message(content)` —— 把字符串 / 单 part / list parts 统一规整为标准 `{"role", "content": [parts]}`。

### 4.3 Prompt caching

- `with_cache_control(message, cache_type="ephemeral")` —— 给消息 content 最后一段加 `cache_control` 注解（Anthropic Claude 3.5+ ~90% 折扣）
- `cache_breakpoint()` —— 返回空 cache point 块，插入 messages 列表精确控制缓存边界
- OpenAI 隐式自动缓存，无需用户配置

## 5. GA-W4: 收尾发版

- `pyproject.toml` + `version.py` 升 **`2.0.0`** 正式版
- README badge 升级 + 路线图标注 GA 完成 + 新增 2.1+ 占位
- 本文档 `project_docs/10-ga-resilience-multimodal.md`

## 6. 已知限制 / 推迟到 2.1+

| 项 | 推迟到 |
|---|---|
| Skill marketplace（version / deps / 远程加载） | 2.1 单独周期 |
| Anthropic / Gemini 完整 `with_structured_output` 严格 schema 模式 | 2.1（GA 已通过 function_calling 路径覆盖大多数用例） |
| Streaming RAG（边检索边生成） | 2.1 |
| Multimodal output（图像/音频生成） | 2.1+ |
| LangServe-style 部署辅助 | **永久不做** |
| Episodic / Perceptual Memory | **永久不做** |
| 内置 RL 训练 | **永久不做**（已通过 `trace_export` 替代） |

## 7. GA 出口标志（已达成）

- [x] Resilience（Retry/Fallback/负载均衡）零外部依赖
- [x] Pydantic 自动 Tool schema 推导（PEP 563 兼容）
- [x] Anthropic 真异步（含 tool use 完整解析）
- [x] Gemini 真异步（to_thread 不阻塞事件循环）
- [x] Multimodal content parts（text/image/audio/file）
- [x] Prompt caching helper（Anthropic ephemeral）
- [x] 全量 pytest **744 passed**（含 W1-α + W1-β + W1-RC + W1-GA）
- [x] `pyproject.toml` 升 `2.0.0` GA + 完整 description
- [x] 100% 向后兼容 1.x + α + β + RC（旧测试 0 破坏）
- [x] 文档收尾：README + `project_docs/10`

## 8. 累计交付总览（1.x → 2.0 GA）

| 维度 | 1.x | α 末 | β 末 | RC 末 | **GA** | 全程 Δ |
|---|---|---|---|---|---|---|
| 全量测试 | 17 | 227 | 500 | 657 | **744** | +727 |
| 模块数 | 8 | 13 | 17 | 20 | **22** | +resilience / from_pydantic / multimodal |
| 自研 LOC | ~3000 | ~3830 | ~8300 | ~10150 | **~11210** | +8210 |
| 设计 spec | 0 | 8 | 9 | 10 | **11** | +08 + 09 + 10 |
| 用户向 docs | 16 | 20 | 22 | 22 | 22 | +6 |
| optional deps | 2 | 4 | 6 | 7 | **7** | +5 |
| 版本 | 1.0.0 | 2.0.0a1 | 2.0.0b1 | 2.0.0rc1 | **2.0.0** | – |

## 9. 关键能力对照（vs LangChain + LangGraph）

`clear-agent==2.0.0` GA 在 **~11K 自研 LOC** 内提供以下与 LangChain + LangGraph 持平或更优的能力：

| 能力域 | LangGraph/Chain | ClearAgent 2.0 GA | 备注 |
|---|---|---|---|
| 声明式图 | ✅ | ✅ | + 字段级 reducer + 同节点多 interrupt 顺序回放 |
| Checkpointer | ✅ Memory/Sqlite/Postgres | ✅ Memory/JsonFile/Sqlite | Postgres 留作 2.1 |
| HITL | ✅ | ✅ | |
| 结构化输出 | ✅ | ✅ 三种 method auto | |
| RAG（完整 7 段） | ✅ 50+ vectorstore | ✅ Qdrant + 完整 pipeline | + MQE/HyDE |
| 多层 Memory | ✅ BaseStore | ✅ Working+Semantic+Manager | + 内存知识图谱 |
| Multi-agent | ✅ supervisor/swarm | ✅ 同名 + handoff 原语 | 基于 graph 原生 |
| MCP 协议 | ❌ | ✅ Client + Server + Adapter | ClearAgent 优势 |
| LCEL 管道 | ✅ | ✅ Runnable + `\|` | 自研 ~280 LOC |
| Callbacks（11+ hooks） | ✅ | ✅ 13 hooks | |
| Eval / Tracing | LangSmith | ✅ TraceLogger + Eval-harness + SFT/DPO 导出 | 不依赖外部服务 |
| Resilience（Retry/Fallback） | ✅ | ✅ | 零外部依赖 |
| Pydantic Tool 自动推导 | ✅ | ✅ | PEP 563 兼容 |
| Multimodal（vision/audio） | ✅ | ✅ | 三家 provider 统一 |
| Prompt caching | ✅ | ✅ helper | Anthropic ephemeral |
| 真异步 client（OpenAI/Anthropic/Gemini） | ✅ | ✅ 三家全部 | |
| 工具并行 | ✅ | ✅ sync + async | |
| 运行时核心依赖 | langchain-core 全家桶 | pydantic + tiktoken + pyyaml + networkx + numpy | **轻量** |

> 设计取舍：ClearAgent 在保留**轻量、单包、零重型依赖**定位的同时，把 LangGraph + LangChain 编排能力的 **~95%** 压进自研代码，且额外提供 MCP 集成、SFT/DPO 训练数据导出等独有能力。
