# 09 · RC 阶段交付 spec

> **阶段**：2.0-RC（W1-W4）
> **状态**：已完成 ✅
> **关联文档**：00（决策）、08（β 集成 spec）

---

## 1. RC 阶段交付摘要

| 周 | 子任务 | 新增 LOC | 测试数 |
|---|---|---|---|
| W1 | Multi-agent 范式包（supervisor / swarm / handoff） | ~360 | 36 |
| W2 | MCP 协议集成（client + server + adapter） | ~470 | 36 |
| W3 | 性能基建（callbacks + parallel + 真异步 OpenAI） | ~560 | 37 |
| W4 | LCEL-lite + TraceLogger 训练数据导出 + 升 2.0.0rc1 | ~440 | 48 |
| **RC 累计** | | **~1830 LOC** | **157** |
| 全量回归 | W1-α + W1-β + W1-RC | – | **657 passed** |

---

## 2. RC-W1: Multi-agent 范式包

按 plan §三决策：**基于 StateGraph 原生**，不抄 AntonAgents A2A，不引入 langgraph。

`clear_agent/multiagent/`：
- `handoff.py` —— `Handoff` 数据类 / `HANDOFF_END` 终止信号 / `make_handoff_tool(s)` 工具 schema 构造 / `parse_handoff_from_tool_calls` 从 LLMToolResponse 提取
- `supervisor.py` —— `build_supervisor_graph(supervisor, workers)` 中心化协调；supervisor 决策 active_agent → 路由到 worker → worker 完成自动清空 active_agent 让 supervisor 重新决策
- `swarm.py` —— `build_swarm_graph(agents, default_active)` 去中心化 handoff；agent 返回自己 = 继续工作（不计 handoff），返回他人 = 移交（计 +1）
- 共用 `max_handoffs` 防死循环 + checkpointer 集成

顶层导出：`from clear_agent import Handoff, build_supervisor_graph, build_swarm_graph, ...`

## 3. RC-W2: MCP 协议集成

按 plan §07 §5：**用官方 mcp SDK，不抄 AntonAgents fastmcp 包装**。

`clear_agent/mcp/`：
- `adapter.py` —— `MCPToolAdapter`（Tool 子类）/ `mcp_tool_to_clear_agent` / `clear_agent_tool_to_mcp_schema` 双向转换
- `client.py` —— `MCPClient.connect_stdio(command, args)` / `connect_sse(url)` / `list_tools` / `call_tool` / `register_to(registry)` 一行注册全部 MCP tools 到 ClearAgent ToolRegistry；同步 API 内部 `asyncio.run` 包装
- `server.py` —— `MCPServer(registry)` 把 ToolRegistry 暴露给 Cursor / Claude Desktop；`run(transport="stdio")` 阻塞监听

mcp SDK 走 `[mcp]` optional dep，未装时 lazy ImportError 友好提示（`list_tools` 等同步入口才检查）。

## 4. RC-W3: 性能基建

### 4.1 Callbacks 协议（`core/callbacks.py`）

13 个 hooks LangChain 风格：`on_llm_{start/end/error/new_token}` · `on_tool_{start/end/error}` · `on_agent_{start/end}` · `on_node_{start/end}` · `on_retriever_{start/end}`。

- `BaseCallbackHandler` 抽象基类（默认 no-op）
- `CallbackManager` 注册多 handler，`fire / afire` 同步异步广播；`swallow_errors=True` 默认吞掉 handler 异常
- 内置：`LoggingCallbackHandler`（emoji 日志）+ `MetricsCallbackHandler`（LLM/tool/node/retriever 累计 + 延迟 + by-name + reset）

### 4.2 工具并行（`core/parallel.py`）

不破坏现有 graph 代码，提供新 helper：
- `run_tools_parallel(tool_calls, registry, max_workers=4)` ThreadPoolExecutor 并发
- `arun_tools_parallel(tool_calls, registry, max_concurrency=4)` async + Semaphore 限流（优先 `tool.arun`，缺失降级线程池）
- `gather_with_concurrency(coros, max_concurrency=4)` 通用 async 限流 helper
- 顺序保留 + 错误隔离 + 标准化输出 `{tool_call_id, name, content, error}`

### 4.3 真异步 OpenAI 客户端

- `OpenAIAdapter` 新增 `ainvoke_async` / `ainvoke_with_tools_async`（`AsyncOpenAI`，不走线程池）
- `ClearAgentLLM.ainvoke / ainvoke_with_tools` 优先调真异步路径；缺失 graceful fallback 到线程池

## 5. RC-W4: LCEL-lite + 训练数据导出

### 5.1 LCEL-lite（`core/runnable.py`）

LangChain Expression Language 精简自研版（**不引入 langchain-core**）：
- `Runnable` 抽象基类 + `|` 操作符
- `RunnableLambda(fn)` 把任意 sync/async callable 包成 Runnable
- `RunnableAdapter(obj)` 兼容含 `invoke` 方法的对象（如 `ClearAgentLLM`）
- `RunnableSequence([r1, r2, r3])` 串行管道（自动扁平化）
- `RunnableParallel({"a": r1, "b": r2})` 字典并发（async 走 `asyncio.gather`）
- `RunnableBranch([(pred, r), ...], default=r0)` 条件分支（predicate 异常自动跳过）
- 便捷 helper：`prompt(template)` · `parser_str()` · `parser_json()` · `passthrough()` · `assign(k=v)`

`_to_runnable` 优先级：`Runnable` → 有 `invoke` 方法 → callable，避免 MagicMock 类对象走错路径。

### 5.2 TraceLogger 训练数据导出（`observability/trace_export.py`）

按 plan §三决策：**不引入 RL 模块**，让 TraceLogger 输出能转 SFT / DPO 格式。
- `read_trace_events(trace_path)` 流式读 JSONL，单行解析失败警告跳过
- `export_to_sft_jsonl(trace_path, out)` 单 trace → SFT 样本；`only_successful=True` 自动跳过含 error 事件的会话；`min_messages=2` 阈值
- `export_traces_to_sft_jsonl(trace_paths, out)` 批量
- `export_to_dpo_pairs(pass_traces, fail_traces, out)` 1:1 配对成 DPO 偏好对（`{prompt, chosen, rejected}`）

零 RL 依赖（仅 stdlib）；用户拿到 jsonl 后自己用 `trl` / `axolotl` 训练。

---

## 6. 已知限制 / 推迟到 GA

按 plan §三 RC 阶段范围决策，以下推迟：

| 项 | 原因 / 推迟到 |
|---|---|
| Anthropic / Gemini 完整 invoke_with_tools + with_structured_output | GA：当前用 OpenAI 兼容路径已覆盖 80% 用户 |
| Multimodal（vision / audio） | GA |
| Prompt caching（Anthropic ephemeral / OpenAI 缓存） | GA |
| Retry / Fallback / 负载均衡 | GA |
| Skill marketplace（version / deps / 远程加载） | GA |
| Tool schema 自动从 Pydantic 推导 | GA |
| LangServe-style 部署辅助 | **永久不做**（用户用 Docker / Modal） |
| Episodic / Perceptual Memory | **永久不做**（over-engineering） |
| 内置 RL 训练 | **永久不做**（已通过 trace_export 替代） |

---

## 7. RC 出口标志（已达成）

- [x] Multi-agent 三种范式可用（supervisor / swarm / handoff）+ 测试 + checkpointer 集成
- [x] MCP 双向桥可用（吃外部 MCP / 暴露为 MCP server）+ ImportError 友好降级
- [x] Callbacks 协议（13 hooks）+ Logging / Metrics 内置 handler
- [x] 工具并行（同步 ThreadPool + 异步 Semaphore）
- [x] 真异步 OpenAI 客户端（不再纯线程池假异步）
- [x] LCEL-lite 管道（5 种 Runnable + 5 种 helper）
- [x] TraceLogger SFT / DPO 导出
- [x] 全量 pytest 657 passed（含 W1-α + W1-β + W1-RC）
- [x] `pyproject.toml` 升 2.0.0rc1 + 新增 `[mcp]` optional dep
- [x] 100% 向后兼容 1.x + α + β（旧测试 0 破坏）

## 8. 累计交付总览（α → β → RC）

| 维度 | α 末 | β 末 | RC 末 | Δ(全程) |
|---|---|---|---|---|
| 全量测试 | 227 | 500 | **657** | +430 |
| 模块数 | 13 | 17 | **20** | +multiagent / mcp / runnable |
| 自研 LOC | ~830 | ~5300 | **~7150** | +6320 |
| 文档（设计 spec） | 8 | 9 | **10** | +08 + 09 |
| 文档（用户向） | 4 | 6 | 6 | – |
| optional deps | 4 | 6 | **7** | +mcp |
| 版本 | 2.0.0a1 | 2.0.0b1 | **2.0.0rc1** | – |

下一站：**2.0.0 GA**。
