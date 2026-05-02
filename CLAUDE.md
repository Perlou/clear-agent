# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在本仓库工作时提供项目指引。

---

## 项目概览

**ClearAgent v1.0.0** —— 基于 OpenAI 原生 API 构建的生产级多智能体框架（Python 3.10+），围绕「上下文工程 + 工具响应协议 + 子代理机制」三大主线，集成 16 项核心工程化能力。

- **包名**：`clear_agent`
- **作者**：Perlou
- **License**：CC BY-NC-SA 4.0（非商业）
- **入口模块**：`clear_agent/__init__.py`

## 顶层目录结构

```
clear-agent/
├── clear_agent/          # 主包
│   ├── core/             # Agent 基类、LLM 适配、Config、SessionStore、生命周期
│   ├── agents/           # 4 种 Agent 范式 + 子代理工厂
│   ├── context/          # 上下文工程：ContextBuilder（GSSC）/ History / Truncator / TokenCounter
│   ├── tools/            # 工具系统 + 内置工具（calculator/file/todowrite/devlog/task/skill）
│   ├── observability/    # TraceLogger（JSONL + HTML）
│   ├── skills/           # Skills 知识外化加载器
│   └── version.py
├── skills/               # 18 个 Skill 包（pdf、docx、xlsx、ASR、TTS、VLM、web-search…）
├── tests/                # 17 个 pytest 测试文件
├── examples/             # async_agent_demo.py
├── docs/                 # 16 篇专项指南
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

## 文档索引（`docs/`）

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
