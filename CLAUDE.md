# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在本仓库工作时提供项目指引。

---

## 项目概览

**ClearAgent v2.0.0a1** —— 基于 OpenAI 原生 API 构建的生产级多智能体框架（Python 3.10+）。

- **1.x 主线**：上下文工程 + 工具响应协议 + 子代理机制
- **2.0 主线**：StateGraph + Checkpointer + Human-in-the-Loop + 结构化输出 + Eval-harness + Retrieval

100% 向后兼容 1.x；`agent.run()` 老入口、`PlanAndSolveAgent` 别名、所有顶层导出全部保留。

- **包名**：`clear_agent`
- **作者**：Perlou
- **License**：CC BY-NC-SA 4.0（非商业）
- **入口模块**：`clear_agent/__init__.py`

## 顶层目录结构

```
clear-agent/
├── clear_agent/          # 主包
│   ├── core/             # Agent 基类、LLM 适配、Config、SessionStore、生命周期
│   │   ├── graph.py            # 🆕 2.0 StateGraph + CompiledGraph + reducers
│   │   ├── checkpoint.py       # 🆕 2.0 BaseCheckpointer + Memory/JsonFile/Sqlite
│   │   ├── interrupt.py        # 🆕 2.0 interrupt() + GraphPaused
│   │   └── structured.py       # 🆕 2.0 StructuredLLM
│   ├── agents/           # 4 种 Agent 范式 + 子代理工厂
│   │   └── _*_graph.py         # 🆕 2.0 build_react/simple/reflection/plan_solve_graph
│   ├── hitl/             # 🆕 2.0 HITL patterns (Approval/Edit/ToolValidation)
│   ├── eval/             # 🆕 2.0 Dataset / Evaluator / run_eval
│   ├── retrieval/        # 🆕 2.0 Embeddings + SQLiteDocumentStore (移植自 AntonAgents)
│   ├── context/          # 上下文工程：ContextBuilder（GSSC）/ History / Truncator / TokenCounter
│   ├── tools/            # 工具系统 + 内置工具（calculator/file/todowrite/devlog/task/skill）
│   ├── observability/    # TraceLogger（JSONL + HTML）
│   ├── skills/           # Skills 知识外化加载器
│   └── version.py
├── skills/               # 18 个 Skill 包（pdf、docx、xlsx、ASR、TTS、VLM、web-search…）
├── tests/                # 460+ pytest 测试
├── examples/             # async_agent_demo.py · rag_hello_world.py
├── project_docs/         # 🆕 2.0 设计 spec（00-overview … 07-anton-port）
├── docs/                 # 16+ 篇专项指南（含 2.0 quickstart）
├── pyproject.toml        # 构建/工具配置（black、isort、pytest、mypy）
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

通过 `clear_agent.agents.create_agent(agent_type, ...)` 工厂统一创建；`default_subagent_factory` 用于子代理。

### 2. LLM 抽象（`clear_agent/core/llm.py` + `llm_adapters.py`）

- `ClearAgentLLM` 通过 `base_url` 自动检测 provider，统一适配 **OpenAI 兼容（DeepSeek/Qwen/Kimi/智谱/Ollama）**、**Anthropic**、**Google Gemini**。
- 同步 / 异步 / 流式 / Function-Calling 四套接口：`invoke` · `ainvoke` · `stream_invoke` · `astream_invoke` · `invoke_with_tools` · `ainvoke_with_tools`。
- 自动识别 thinking model（o1、deepseek-reasoner），单独返回 `reasoning_content`。

### 3. 上下文工程（`clear_agent/context/`）

- **GSSC 流水线**：`ContextBuilder`（Gather-Select-Structure-Compress）。
- `HistoryManager`：按轮次压缩，保留 `min_retain_rounds`，超过 `compression_threshold * context_window` 触发压缩。
- `ObservationTruncator`：工具输出截断（行/字节双限），完整结果落盘到 `tool_output_dir`。
- `TokenCounter`：基于 `tiktoken`，缓存 + 增量计算。

### 4. 工具系统（`clear_agent/tools/`）

- `Tool` / `ToolParameter` / `@tool_action` 基础原语。
- `ToolResponse(status, text, data, error)` —— 统一响应协议，所有内置/自定义工具必须返回此对象。
- `ToolRegistry` + `global_registry`，支持 `ToolFilter`（`ReadOnlyFilter` / `FullAccessFilter` / `CustomFilter`）做子代理权限隔离。
- `CircuitBreaker`：连续失败阈值熔断，冷却期自动恢复。
- 内置工具：`CalculatorTool` · `Read/Write/Edit/MultiEditTool`（文件编辑带乐观锁）· `TodoWriteTool` · `DevLogTool` · `TaskTool`（子代理派发）· `SkillTool`（知识按需加载）。

### 5. Skills 知识外化（`clear_agent/skills/` + 仓库根 `skills/`）

- 启动时仅加载 `SKILL.md` 元数据；调用 `SkillTool` 时按需注入完整内容（作为 `tool_result`，缓存友好）。
- 预期 token 节省 ~85%（20 个 skill 场景）。

### 6. 可观测性（`clear_agent/observability/trace_logger.py`）

- `TraceLogger` 同时输出 JSONL + 自包含 HTML，可选脱敏。
- 默认目录：`memory/traces/`。

### 7. 持久化目录约定

| 类型 | 默认路径 | 配置项 |
|------|---------|--------|
| Trace | `memory/traces/` | `trace_dir` |
| Session | `memory/sessions/` | `session_dir` |
| Todo | `memory/todos/` | `todowrite_persistence_dir` |
| DevLog | `memory/devlogs/` | `devlog_persistence_dir` |
| Tool 完整输出 | `tool-output/` | `tool_output_dir` |
| 🆕 Eval 报告 | `memory/eval/<run_id>/` | `eval_output_dir`（runner 入参） |
| 🆕 Sqlite Checkpoint | `memory/runs.db`（约定） | `SqliteCheckpointer(path=...)` |

> `memory/` 和 `tool-output/` 已加入 `.gitignore`（运行时产物，不入库）。

### 8. 🆕 2.0 StateGraph + Checkpoint

- **核心抽象**：`StateGraph[State]`（`add_node` / `add_edge` / `add_conditional_edges`）→ `CompiledGraph`（`invoke` / `ainvoke` / `stream` / `resume` / `get_state` / `draw_mermaid`）
- **Reducer**：字段级合并 `add_messages` / `append_list` / `merge_dict` / 自定义 fn
- **Checkpointer**：每节点写一份快照（thread_id + checkpoint_id + parent + state + next_nodes + metadata）
- **三个后端**：`InMemoryCheckpointer` / `JsonFileCheckpointer`（兼容老 `memory/sessions/`）/ `SqliteCheckpointer`
- **接入方式**：所有内置 Agent 暴露 `as_graph(checkpointer=...)`，旧 `agent.run()` 入口不变

### 9. 🆕 2.0 Human-in-the-Loop

- 节点内 `interrupt(payload)` → 抛 `GraphInterrupt` → CompiledGraph 捕获 → 写 `source=interrupt` 的 ckpt → 抛 `GraphPaused`
- 调用方捕获 `GraphPaused` → 展示给用户 → `compiled.resume(thread_id, value=...)` 注入决策 → 节点函数重入，`interrupt()` 直接返回 value
- 同节点多个 `interrupt()` 按 resume 顺序回放历史值
- 内置三种模式：`Approval` / `Edit` / `ToolValidation`（`clear_agent/hitl/patterns.py`）

### 10. 🆕 2.0 结构化输出

- `llm.with_structured_output(schema, method="auto"|"function_calling"|"json_mode"|"json_schema", include_raw=False, max_retries=2)`
- `auto` 自动按 model + base_url 选 method（OpenAI gpt-4o+ → `json_schema`，其他 → `function_calling`）
- 失败重试：把校验错误追加到对话让 LLM 修正
- `include_raw=True` 模式下解析失败不抛异常，返回 `{parsed, raw, parsing_error}`
- 在 eval-harness 里被 `LLMAsJudge` 复用（让评分本身也是结构化输出）

### 11. 🆕 2.0 Eval-harness

- `Dataset.from_jsonl/from_list` + `filter/sample/take/[:]`
- 4 种 Evaluator：`ExactMatch` / `Contains` / `LLMAsJudge`（依赖 `with_structured_output`）/ `Custom(fn)`
- `run_eval(target, ds, evaluator, parallel=4, output_dir=...)` → 输出 `report.md` + `results.jsonl`
- `target` 支持 `CompiledGraph` 与 `callable`；`extract_predicted` 默认从 `final_answer` / `messages` 抽取
- 单条 example 抛异常不打断整批；按 tag 聚合 + top failures 排序

### 12. 🆕 2.0 Retrieval（spike）

- `EmbeddingModel` 抽象 + 三种实现：`LocalTransformerEmbedding`（sentence-transformers / hf）/ `DashScopeEmbedding`（OpenAI 兼容 REST 优先）/ `TFIDFEmbedding`（sklearn 兜底）
- 工厂带回退：`create_embedding_model_with_fallback(preferred, ...)`
- 全局单例：`get_text_embedder()` / `get_dimension()` / `refresh_embedder()`
- `SQLiteDocumentStore`：同路径单例 + 线程本地连接 + `:memory:` 支持
- 全部移植自 AntonAgents（License 一致 CC-BY-NC-SA-4.0），按 `project_docs/07` SOP 改 namespace + 异常 + 加 license 标注

## 配置与环境变量

通过 `clear_agent.core.config.Config`（pydantic）集中管理；运行时从 `.env` 读取（参考 `.env.example`）。

**LLM 必填四项**：`LLM_MODEL_ID` · `LLM_API_KEY` · `LLM_BASE_URL` · `LLM_TIMEOUT`（可选，默认 60s）。

可选：`TAVILY_API_KEY` / `SERPAPI_API_KEY`（搜索）、`QDRANT_*`（向量库）、`NEO4J_*`（图数据库）、`EMBED_*`（嵌入）、`GITHUB_PERSONAL_ACCESS_TOKEN`、`HF_TOKEN`。

## 常用命令

```bash
# 安装（本地开发）
pip install -e .
# 或使用 uv
uv sync

# 运行全部测试
pytest

# 运行单个测试文件
pytest tests/test_react_agent.py -v

# 代码格式化
black clear_agent tests
isort clear_agent tests

# 类型检查
mypy clear_agent

# 运行示例
python examples/async_agent_demo.py
```

## 开发约定

1. **新增工具** → 继承 `Tool`，`run()` 必须返回 `ToolResponse`，使用 `ToolResponse.success(...)` / `ToolResponse.error(code=..., message=...)` 构造。
2. **新增 Agent** → 继承 `clear_agent.core.agent.Agent`，实现抽象方法 `run()`；`arun()` 默认走线程池，需要真异步则覆写。
3. **新增 Skill** → 在 `skills/<name>/SKILL.md` 写元数据 + 正文，可附带 `scripts/`、`reference.md`；`SkillLoader` 自动发现。
4. **持久化文件** → 不要硬编码路径，从 `Config` 字段读取。
5. **不要修改 `system_prompt` 注入 Skill 内容** —— 按协议作为 `tool_result` 注入以保持缓存命中。
6. **向后兼容** → `_history` property、`PlanAndSolveAgent` 别名等保留，禁止破坏。
7. **代码风格**：`black` line-length=88、`isort` profile=black、`mypy` 严格模式（`disallow_untyped_defs=true`）。

## 文档索引

### 2.0 设计 spec（`project_docs/`）

| 文件 | 主题 |
|------|------|
| `00-overview.md` | 整体决策与差距分析 |
| `01-graph-architecture.md` | StateGraph + Reducer 设计 |
| `02-checkpoint-and-resume.md` | Checkpointer 三后端设计 |
| `03-hitl-guide.md` | interrupt/resume 与 HITL patterns |
| `04-structured-output.md` | with_structured_output 三种 method |
| `05-eval-harness.md` | Dataset / Evaluator / Runner |
| `06-migration-1.x-to-2.x.md` | 迁移路径 |
| `07-anton-agents-port.md` | AntonAgents 移植 SOP |

### 2.0 用户向 quickstart（`docs/`）

| 主题 | 文件 |
|------|------|
| StateGraph 上手 | `graph-architecture.md` |
| 结构化输出 | `structured-output.md` |
| Eval-harness | `eval-harness.md` |
| 1.x→2.x 迁移 | `migration-1.x-to-2.x.md` |

### 1.x 专项（`docs/`）

| 主题 | 文件 |
|------|------|
| 工具响应协议 | `tool-response-protocol.md` |
| Function Calling 架构 | `function-calling-architecture.md` |
| 上下文工程 | `context-engineering-guide.md` |
| 会话持久化 | `session-persistence-guide.md` |
| 子代理机制 | `subagent-guide.md` |
| 熔断器 | `circuit-breaker-guide.md` |
| Skills 系统 | `skills-quickstart.md` · `skills-usage-guide.md` |
| TodoWrite | `todowrite-usage-guide.md` |
| DevLog | `devlog-guide.md` |
| 流式 SSE | `streaming-sse-guide.md` |
| 异步 Agent | `async-agent-guide.md` |
| 可观测性 | `observability-guide.md` |
| 日志系统 | `logging-system-guide.md` |
| 文件工具 | `file_tools.md` |
| 自定义工具 | `custom_tools_guide.md` |

## 注意事项

- **License 是 CC BY-NC-SA 4.0**：商业使用前需联系作者。
- 默认 `trace_enabled=True`、`session_enabled=True`、`skills_enabled=True`、`subagent_enabled=True`，关闭请在 `Config` 中显式设为 `False`。
- 异步路径：`max_concurrent_tools=3`、`hook_timeout_seconds=5`、`llm_async_timeout=120`、`tool_async_timeout=30`。
