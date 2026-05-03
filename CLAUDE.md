# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在本仓库工作时提供项目指引。

---

## 项目概览

**ClearAgent** —— 轻量、可组合的多智能体框架（Python 3.10+），基于 OpenAI 原生 API，单包提供图编排、检查点、HITL、RAG、Memory、Multi-agent、MCP、结构化输出、Eval-harness 等能力。

- **PyPI 包名**：`clear-agent`（`pip install clear-agent`）
- **Python import**：`import clear_agent`
- **作者**：Perlou
- **License**：CC BY-NC-SA 4.0（非商业）
- **入口模块**：`clear_agent/__init__.py`

## 顶层目录结构

```
clear-agent/
├── clear_agent/          # 主包
│   ├── core/             # Agent 基类 + LLM + 编排基础
│   │   ├── graph.py            # StateGraph + reducers + CompiledGraph
│   │   ├── checkpoint.py       # BaseCheckpointer + Memory/JsonFile/Sqlite
│   │   ├── interrupt.py        # interrupt() + GraphPaused
│   │   ├── structured.py       # StructuredLLM + with_structured_output
│   │   ├── runnable.py         # LCEL-lite Runnable + | 管道
│   │   ├── callbacks.py        # 13 hooks BaseCallbackHandler
│   │   ├── parallel.py         # run_tools_parallel / arun_tools_parallel
│   │   ├── resilience.py       # Retry / Fallback / 负载均衡
│   │   ├── multimodal.py       # text/image/audio/file parts + cache_control
│   │   ├── llm.py / llm_adapters.py
│   │   ├── config.py / lifecycle.py / agent.py / session_store.py
│   ├── agents/           # 4 范式 + 子代理 + graph builders
│   ├── multiagent/       # supervisor / swarm / handoff
│   ├── mcp/              # MCP client / server / adapter
│   ├── hitl/             # HITL patterns（Approval / Edit / ToolValidation）
│   ├── eval/             # Dataset / Evaluator / Runner
│   ├── retrieval/        # 嵌入 + Qdrant + SQLite + RAG pipeline
│   │   ├── embeddings.py
│   │   ├── rag/                # document + pipeline 7 段
│   │   └── storage/            # SQLite + Qdrant
│   ├── memory/           # WorkingMemory + SemanticMemory + Manager
│   ├── context/          # ContextBuilder GSSC + History + Truncator + TokenCounter
│   ├── tools/            # Tool 协议 + Pydantic 自动推导 + 内置工具
│   ├── observability/    # TraceLogger（JSONL + HTML + SFT/DPO export）
│   └── skills/           # SkillLoader
├── skills/               # 18 个 Skill 包（pdf/docx/xlsx/ASR/TTS/VLM/web-search…）
├── tests/                # 740+ pytest 测试
├── examples/             # 演示
├── docs/                 # 用户指南
├── pyproject.toml        # 构建/工具配置（black/isort/pytest/mypy）
├── requirements.txt      # 运行依赖
└── .env.example          # 环境变量模板
```

## 核心架构

### 1. Agent 范式（`clear_agent/agents/`）

| 类型 | 类 | 适用场景 |
|------|-----|---------|
| `simple` | `SimpleAgent` | 纯对话 / 单轮 Function Calling |
| `react` | `ReActAgent` | 推理-行动循环（主力） |
| `reflection` | `ReflectionAgent` | 自我反思优化 |
| `plan` | `PlanSolveAgent`（别名 `PlanAndSolveAgent`） | 规划-执行 |

每种 Agent 都暴露 `as_graph(checkpointer=...)` 转 StateGraph，原 `agent.run()` 保留向后兼容。

### 2. StateGraph（`core/graph.py`）

- `StateGraph[State]` + `add_node` / `add_edge` / `add_conditional_edges`
- 字段级 reducer：`add_messages` / `append_list` / `merge_dict` / 自定义 fn
- `CompiledGraph` 提供 `invoke` / `ainvoke` / `stream` / `resume` / `get_state` / `draw_mermaid`
- 内置 `START` / `END` 常量

### 3. Checkpointer（`core/checkpoint.py`）

- 每节点写一份快照（`thread_id` + `checkpoint_id` + `parent` + `state` + `next_nodes` + `metadata`）
- 三个后端：`InMemoryCheckpointer` / `JsonFileCheckpointer`（兼容老 `memory/sessions/`）/ `SqliteCheckpointer`
- 支持时间旅行：`ck.list(thread_id)` 倒序拿历史 + 任意点 resume

### 4. Human-in-the-Loop（`core/interrupt.py` + `hitl/`）

- 节点内调 `interrupt(payload)` → 抛 `GraphInterrupt` → CompiledGraph 捕获 → 写 ckpt → 抛 `GraphPaused`
- 调用方捕获 `GraphPaused` → 展示 → `compiled.resume(thread_id, value=...)` 续跑
- 同节点多 `interrupt()` 按 resume 顺序回放历史值
- 三种内置模式：`Approval` / `Edit` / `ToolValidation`

### 5. 结构化输出（`core/structured.py`）

`llm.with_structured_output(schema, method="auto"|"function_calling"|"json_mode"|"json_schema")` 返回 `StructuredLLM`，调用直接拿 Pydantic 实例。
- `auto` 按 model + base_url 选 method（OpenAI gpt-4o+ → `json_schema`，其他 → `function_calling`）
- 失败重试：错误信息追加到对话让 LLM 修正
- `include_raw=True` 返回 `{parsed, raw, parsing_error}`

### 6. LLM 抽象（`core/llm.py` + `llm_adapters.py`）

- `ClearAgentLLM` 通过 `base_url` 自动选 provider：OpenAI 兼容 / Anthropic / Gemini
- 同步 + 真异步（`AsyncOpenAI` / `AsyncAnthropic`，Gemini 走 `asyncio.to_thread`）+ 流式 + Function Calling
- 自动识别 thinking model（o1、deepseek-reasoner），单独返回 `reasoning_content`

### 7. 工具系统（`tools/`）

- `Tool` 基类 + `ToolParameter` + `@tool_action`
- `ToolResponse(status, text, data, error_info)` 统一响应协议
- `ToolRegistry` + `global_registry` + `ToolFilter`（`ReadOnlyFilter` / `FullAccessFilter` / `CustomFilter`）
- `CircuitBreaker` 失败熔断
- **Pydantic 自动推导**：`@pydantic_tool` 装饰器 / `tool_from_pydantic` 包装器，自动从 `BaseModel` 转 OpenAI function spec
- 内置工具：`CalculatorTool` · `Read/Write/Edit/MultiEditTool`（乐观锁）· `TodoWriteTool` · `DevLogTool` · `TaskTool`（子代理派发）· `SkillTool`

### 8. RAG Pipeline（`retrieval/rag/pipeline.py`）

7 大职责：
1. **加载** —— MarkItDown 50+ 格式 + PDF 增强后处理 + utf8/latin-1 fallback
2. **分块** —— Markdown-aware 段落 + token 预算 + overlap
3. **图谱集成** —— `build_graph_from_chunks(neo4j, chunks)`（可选）
4. **索引** —— 批量 embedding + 失败重试 + 维度对齐 + 标签
5. **检索** —— 向量 + MQE（多查询扩展）+ HyDE（假设性回答）
6. **重排** —— Cross-encoder + 图信号融合 + 加权 rank
7. **结果组装** —— merge / expand_neighbors / grouped_with_citations / compress / TL;DR

高层入口：`create_rag_pipeline(qdrant_url, rag_namespace, llm)` 返回 `{store, add_documents, search, search_advanced, rerank, summarize}`

### 9. 多层 Memory（`memory/`）

- **`MemoryItem` / `MemoryConfig` / `BaseMemory`**：7 个抽象接口（add/retrieve/update/remove/has_memory/clear/get_stats）
- **`WorkingMemory`** 短期：内存存储 + heapq 优先级 + TF-IDF 检索 + 时间衰减 + 三种 forget 策略 + TTL 过期
- **`SemanticMemory`** 长期：向量 + **内存知识图谱**（`Entity` / `Relation`）；spaCy NER 可选 + fallback；混合检索（向量 + 图）+ softmax 概率
- **`MemoryManager`** 协调多子系统：注册 / 路由 / 跨子系统聚合检索

设计决策：**不引入 Neo4j**（Qdrant 已覆盖 90% 场景），`SemanticMemory` 图谱完全在内存里。

### 10. Multi-agent（`multiagent/`）

基于 StateGraph 原生：
- `Handoff` 数据类 + `make_handoff_tool(s)` + `parse_handoff_from_tool_calls`
- `build_supervisor_graph(supervisor, workers)` 中心化路由
- `build_swarm_graph(agents, default_active)` 去中心化 handoff
- `max_handoffs` 防死循环

### 11. MCP 协议（`mcp/`）

- `MCPClient.connect_stdio(command, args)` / `connect_sse(url)` 吃外部 MCP 工具
- `client.register_to(registry)` 一行注册到 ClearAgent ToolRegistry
- `MCPServer(registry)` 暴露给 Cursor / Claude Desktop
- 用官方 `mcp` SDK，optional dep `[mcp]`

### 12. LCEL-lite Runnable（`core/runnable.py`）

自研 `Runnable` + `|` 管道（**不引入 langchain-core**）：
- `RunnableLambda` / `RunnableAdapter` / `RunnableSequence` / `RunnableParallel` / `RunnableBranch`
- helper：`prompt(template)` / `parser_str()` / `parser_json()` / `passthrough()` / `assign(k=v)`

### 13. Resilience（`core/resilience.py`）

- `with_retry(fn)` / `@retry` 同步 + `with_retry_async(fn)` / `@aretry` 异步
- 指数退避 + jitter + 异常白名单
- `with_fallbacks(primary, [fb1, fb2])` 主调失败按序回退
- `round_robin([fns])` / `random_choice([fns], seed)` 负载均衡

### 14. Multimodal + Prompt caching（`core/multimodal.py`）

content parts 构造器：`text_part` / `image_url_part` / `image_base64_part`（OpenAI/Anthropic 双格式）/ `audio_part` / `file_part`

消息构造器：`user_message` / `system_message` / `assistant_message`

`with_cache_control(message)` 加 Anthropic ephemeral 缓存注解。

### 15. Eval-harness（`eval/`）

- `Dataset.from_jsonl/from_list` + `filter/sample/take`
- 4 种 Evaluator：`ExactMatch` / `Contains` / `LLMAsJudge`（依赖 `with_structured_output`）/ `Custom(fn)`
- `run_eval(target, ds, evaluator, parallel)` 输出 `report.md` + `results.jsonl`
- 单条异常不打断整批 + 按 tag 聚合 + top failures

### 16. Callbacks（`core/callbacks.py`）

13 个 hooks：`on_llm_{start/end/error/new_token}` / `on_tool_{start/end/error}` / `on_agent_{start/end}` / `on_node_{start/end}` / `on_retriever_{start/end}`

`CallbackManager` 注册多 handler，`fire / afire` 同步异步广播；内置 `LoggingCallbackHandler` / `MetricsCallbackHandler`

### 17. 上下文工程（`context/`）

- **GSSC 流水线**：`ContextBuilder`（Gather-Select-Structure-Compress）
- `HistoryManager`：按轮次压缩，超过阈值触发
- `ObservationTruncator`：工具输出截断（行/字节双限），完整结果落盘到 `tool_output_dir`
- `TokenCounter`：基于 `tiktoken`，缓存 + 增量计算

### 18. 可观测性（`observability/`）

- `TraceLogger` 同时输出 JSONL + 自包含 HTML，可选脱敏
- 默认目录：`memory/traces/`
- `trace_export.py`：`export_to_sft_jsonl` / `export_to_dpo_pairs` 训练数据导出（**不引入 RL 模块**，用户自行接 `trl`/`axolotl`）

### 19. Skills 知识外化（`skills/`）

- 启动时仅加载 `SKILL.md` 元数据；调用 `SkillTool` 时按需注入完整内容
- 预期 token 节省 ~85%（20 个 skill 场景）

## 持久化目录约定

| 类型 | 默认路径 | 配置项 |
|------|---------|--------|
| Trace | `memory/traces/` | `trace_dir` |
| Session | `memory/sessions/` | `session_dir` |
| Todo | `memory/todos/` | `todowrite_persistence_dir` |
| DevLog | `memory/devlogs/` | `devlog_persistence_dir` |
| Tool 完整输出 | `tool-output/` | `tool_output_dir` |
| Eval 报告 | `memory/eval/<run_id>/` | runner 入参 |
| Sqlite Checkpoint | 用户指定（约定 `memory/runs.db`） | `SqliteCheckpointer(path=...)` |
| Qdrant RAG 集合 | `clear_agent_rag_vectors` | `create_rag_pipeline(collection_name=...)` |
| Qdrant Semantic 集合 | `clear_agent_semantic` | 自动创建 |

`memory/` 与 `tool-output/` 已加入 `.gitignore`。

## 配置与环境变量

通过 `clear_agent.core.config.Config`（pydantic）集中管理；运行时从 `.env` 读取（参考 `.env.example`）。

**LLM 必填**：`LLM_MODEL_ID` · `LLM_API_KEY` · `LLM_BASE_URL` · `LLM_TIMEOUT`（可选，默认 60s）

可选：`TAVILY_API_KEY` / `SERPAPI_API_KEY`（搜索）、`QDRANT_*`（向量库）、`NEO4J_*`（图数据库，已不强制）、`EMBED_*`（嵌入）、`GITHUB_PERSONAL_ACCESS_TOKEN`、`HF_TOKEN`

## 常用命令

```bash
# 安装
pip install -e .                      # 本地开发
pip install -e ".[retrieval-qdrant,rag,memory,mcp]"   # 全套扩展

# 测试
pytest                                  # 全量
pytest tests/test_graph_basics.py -v    # 单文件

# 格式化
black clear_agent tests
isort clear_agent tests

# 类型检查
mypy clear_agent

# 运行示例
python examples/async_agent_demo.py
python examples/rag_hello_world.py
python examples/memory_demo.py
```

## 开发约定

1. **新增工具** → 优先用 `@pydantic_tool` 装饰器（自动 schema 推导）；或继承 `Tool` 实现 `run() / get_parameters()`，必须返回 `ToolResponse`
2. **新增 Agent** → 继承 `clear_agent.core.agent.Agent`，实现抽象 `run()`；可选 `as_graph()` 返回 StateGraph
3. **新增 Skill** → 在 `skills/<name>/SKILL.md` 写元数据 + 正文，`SkillLoader` 自动发现
4. **持久化文件** → 不要硬编码路径，从 `Config` 字段读取
5. **不要修改 `system_prompt` 注入 Skill 内容** —— 按协议作为 `tool_result` 注入以保持缓存命中
6. **向后兼容** → `_history` property、`PlanAndSolveAgent` 别名等保留，禁止破坏
7. **代码风格**：`black` line-length=88、`isort` profile=black、`mypy` 严格模式（`disallow_untyped_defs=true`）
8. **可选依赖** → 重型库（qdrant-client / sentence-transformers / spacy / mcp / anthropic / google-genai）走 `try/except ImportError`，缺失时友好提示

## 文档索引（`docs/`）

| 主题 | 文件 |
|------|------|
| 快速开始 | `quickstart.md` |
| StateGraph 架构 | `graph-architecture.md` |
| HITL | `hitl.md` |
| 结构化输出 | `structured-output.md` |
| RAG | `rag-guide.md` |
| Memory | `memory-guide.md` |
| Multi-agent | `multi-agent.md` |
| MCP 集成 | `mcp.md` |
| Eval-harness | `eval-harness.md` |
| 工具系统 | `tool-system.md` |
| 可观测性 | `observability.md` |
| 异步与流式 | `async-streaming.md` |
| 上下文工程 | `context-engineering.md` |
| Skills 系统 | `skills.md` |

## 注意事项

- **License 是 CC BY-NC-SA 4.0**：商业使用前需联系作者
- 默认 `trace_enabled=True`、`session_enabled=True`、`skills_enabled=True`、`subagent_enabled=True`，关闭请在 `Config` 中显式设为 `False`
- 异步路径默认：`max_concurrent_tools=3`、`hook_timeout_seconds=5`、`llm_async_timeout=120`、`tool_async_timeout=30`
- Memory 模块**不引入 Neo4j**（已废弃决策），`SemanticMemory` 知识图谱完全在内存里
